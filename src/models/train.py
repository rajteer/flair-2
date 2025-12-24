import inspect
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torchinfo import summary

from src.data.pre_processing.data_augmentation import FlairAugmentation
from src.utils.mlflow_utils import log_metrics_to_mlflow, log_model_to_mlflow
from src.utils.model_stats import compute_model_complexity

logger = logging.getLogger(__name__)

TEMPORAL_MODEL_NDIM = 5

BATCH_INDEX_INPUTS = 0
BATCH_INDEX_TARGETS = 1
BATCH_INDEX_PAD_MASK = 2
BATCH_INDEX_SAMPLE_IDS = 3
BATCH_INDEX_POSITIONS = 4


def _log_model_description(
    model: torch.nn.Module,
    device: torch.device,
    sample_input_shape: tuple[int, ...],
    sample_inputs: torch.Tensor | None = None,
    batch_positions: torch.Tensor | None = None,
) -> None:
    """Create and log model stats."""
    model_info = ""

    summary_kwargs: dict[str, Any] = {
        "model": model,
        "device": str(device),
        "verbose": 0,
        "col_names": [
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        "row_settings": ["var_names"],
    }

    if sample_inputs is not None:
        summary_kwargs["input_data"] = sample_inputs
        if sample_inputs.ndim == TEMPORAL_MODEL_NDIM and batch_positions is not None:
            summary_kwargs["batch_positions"] = batch_positions
    else:
        summary_kwargs["input_size"] = sample_input_shape

    with io.StringIO() as buf:
        model_summary = summary(**summary_kwargs)
        print(model_summary, file=buf)
        model_info = buf.getvalue()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(model_info)
        tmp_path = tmp.name
    mlflow.log_artifact(tmp_path, "model_architecture")
    Path(tmp_path).unlink()

    complexity = compute_model_complexity(
        model=model,
        input_size=sample_input_shape,
        batch_positions=batch_positions,
    )
    for k, v in complexity.items():
        mlflow.log_metric(k, float(v))


def _get_sample_batch(
    loader: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the first batch's inputs and optional temporal positions."""
    batch = next(iter(loader))
    inputs = batch[BATCH_INDEX_INPUTS]
    batch_positions = batch[BATCH_INDEX_POSITIONS] if len(batch) > BATCH_INDEX_POSITIONS else None
    return inputs, batch_positions


def _train_epoch_temporal(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    use_amp: bool = False,
) -> float:
    """Train a temporal model for a single epoch and return average loss.

    Temporal models receive batch_positions and pad_mask arguments.
    Note: Augmentations are not supported for temporal (5D) data.
    """
    model.train()
    optimizer.zero_grad()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
    total_loss = 0.0

    forward_sig = inspect.signature(model.forward)
    supports_pad_mask = "pad_mask" in forward_sig.parameters

    for batch_idx, batch_data in enumerate(loader):
        x = batch_data[BATCH_INDEX_INPUTS].to(device)
        y = batch_data[BATCH_INDEX_TARGETS].to(device)
        pad_mask = batch_data[BATCH_INDEX_PAD_MASK].to(device)
        batch_positions = batch_data[BATCH_INDEX_POSITIONS].to(device)

        if supports_pad_mask:
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(x, batch_positions=batch_positions, pad_mask=pad_mask)
                loss = criterion(outputs, y)
                loss_value = loss.item()
                loss = loss / accumulation_steps
        else:
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(x, batch_positions=batch_positions)
                loss = criterion(outputs, y)
                loss_value = loss.item()
                loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += float(loss_value)

    return total_loss / len(loader)


def _train_epoch_standard(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augmenter: FlairAugmentation | None = None,
    accumulation_steps: int = 1,
    use_amp: bool = False,
) -> float:
    """Train a standard (non-temporal) model for a single epoch and return average loss."""
    model.train()
    optimizer.zero_grad()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
    total_loss = 0.0

    for batch_idx, batch_data in enumerate(loader):
        x = batch_data[BATCH_INDEX_INPUTS].to(device)
        y = batch_data[BATCH_INDEX_TARGETS].to(device)

        if augmenter is not None:
            x, y = augmenter(x, y)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_value = loss.item()
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += float(loss_value)

    return total_loss / len(loader)


def _validate_epoch_temporal(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Validate a temporal model for a single epoch and return average loss.

    Temporal models receive batch_positions and pad_mask arguments.
    """
    model.eval()
    val_loss = 0.0

    forward_sig = inspect.signature(model.forward)
    supports_pad_mask = "pad_mask" in forward_sig.parameters

    with torch.no_grad():
        for batch_data in loader:
            x = batch_data[BATCH_INDEX_INPUTS].to(device)
            y = batch_data[BATCH_INDEX_TARGETS].to(device)
            pad_mask = batch_data[BATCH_INDEX_PAD_MASK].to(device)
            batch_positions = batch_data[BATCH_INDEX_POSITIONS].to(device)

            if supports_pad_mask:
                outputs = model(x, batch_positions=batch_positions, pad_mask=pad_mask)
            else:
                outputs = model(x, batch_positions=batch_positions)
            loss = criterion(outputs, y)
            val_loss += float(loss.item())
    return val_loss / len(loader)


def _validate_epoch_standard(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Validate a standard (non-temporal) model for a single epoch and return average loss."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_data in loader:
            x = batch_data[BATCH_INDEX_INPUTS].to(device)
            y = batch_data[BATCH_INDEX_TARGETS].to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += float(loss.item())
    return val_loss / len(loader)


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: LRScheduler | None = None,
    epochs: int = 100,
    patience: int = 20,
    num_classes: int = 13,
    accumulation_steps: int = 1,
    *,
    use_amp: bool = False,
    apply_augmentations: bool = True,
    augmentation_config: dict[str, Any] | None = None,
    log_evaluation_metrics: bool = True,
    log_model: bool = True,
) -> dict[str, list[float] | float]:
    """Train a segmentation model, monitoring validation loss and saving the best model.

    Detailed metrics should be calculated separately after training using an
    evaluation function.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device (CPU/GPU).
        scheduler: Optional learning rate scheduler.
        apply_augmentations: Whether to apply augmentations to the training data.
            Defaults to True.
        augmentation_config: Configuration for augmentations. Defaults to None.
        epochs: Maximum number of epochs to train. Defaults to 100.
        patience: Early stopping patience. Defaults to 20.
        num_classes: Number of classes in the segmentation task. Defaults to 13.
        accumulation_steps: Number of steps to accumulate gradients before updating.
            Defaults to 1.
        use_amp: Whether to use Automatic Mixed Precision (AMP). Defaults to False.
        log_evaluation_metrics: Whether to log metrics and models to MLflow.
            Defaults to True.
        log_model: Whether to log the best model to MLflow. Defaults to True.

    Returns:
        dict: History of training and validation losses, and best validation loss.
            The best model is logged as an MLflow artifact 'best_model' if
            log_to_mlflow is True.

    """
    model.to(device)

    sample_inputs, sample_batch_positions = _get_sample_batch(train_loader)
    sample_input_shape = tuple(int(x) for x in sample_inputs.shape)

    if log_evaluation_metrics:
        _log_model_description(
            model,
            device,
            sample_input_shape,
            sample_inputs=sample_inputs,
            batch_positions=sample_batch_positions,
        )

    best_val_loss = float("inf")
    no_improve = 0
    losses_train: list[float] = []
    losses_val: list[float] = []

    augmenter = (
        FlairAugmentation(
            augmentation_config or {},
            clamp=True,
            clamp_min=0.0,
            clamp_max=255.0,
        )
        if apply_augmentations
        else None
    )
    best_model_state = None

    is_temporal_model = sample_inputs.ndim == TEMPORAL_MODEL_NDIM

    if is_temporal_model:
        logger.info("Detected temporal model (5D input). Using temporal training loop.")
        train_epoch_fn = _train_epoch_temporal
        validate_epoch_fn = _validate_epoch_temporal
    else:
        logger.info("Detected standard model. Using standard training loop.")
        train_epoch_fn = _train_epoch_standard
        validate_epoch_fn = _validate_epoch_standard

    for epoch in range(epochs):
        if is_temporal_model:
            loss_epoch = train_epoch_fn(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                accumulation_steps,
                use_amp,
            )
        else:
            loss_epoch = train_epoch_fn(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                augmenter,
                accumulation_steps,
                use_amp,
            )
        losses_train.append(loss_epoch)
        logger.info("Epoch %d/%d: Training Loss: %.4f", epoch + 1, epochs, loss_epoch)

        val_loss = validate_epoch_fn(model, val_loader, criterion, device)
        losses_val.append(val_loss)
        logger.info("Epoch %d/%d: Validation Loss: %.4f", epoch + 1, epochs, val_loss)

        if log_evaluation_metrics:
            log_metrics_to_mlflow(
                metrics={"train_loss": loss_epoch, "val_loss": val_loss},
                step=epoch,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            logger.info("Validation loss decreased")
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            logger.info("No improvement for %d epochs.", no_improve)
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if log_evaluation_metrics:
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            logger.info("Current learning rate: %.6f", current_lr)

    if log_model and best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        log_model_to_mlflow(
            model=model,
            train_loader=train_loader,
            sample_input_shape=sample_input_shape,
            num_classes=num_classes,
        )

    return {
        "train_loss": losses_train,
        "val_loss": losses_val,
        "best_val_loss": best_val_loss,
    }
