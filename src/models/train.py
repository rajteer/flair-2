import inspect
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torchinfo import summary
from torchmetrics.classification import MulticlassJaccardIndex

from src.data.pre_processing.chessmix import ChessMix
from src.data.pre_processing.data_augmentation import FlairAugmentation
from src.utils.mlflow_utils import log_metrics_to_mlflow, log_model_to_mlflow
from src.utils.model_stats import compute_model_complexity

logger = logging.getLogger(__name__)

try:
    GradScaler = torch.amp.GradScaler
    autocast = torch.amp.autocast
    _USE_NEW_AMP_API = True
except (AttributeError, ImportError):
    from torch.cuda.amp import GradScaler, autocast

    _USE_NEW_AMP_API = False

TEMPORAL_MODEL_NDIM = 5

BATCH_INDEX_INPUTS = 0
BATCH_INDEX_TARGETS = 1
BATCH_INDEX_PAD_MASK = 2
BATCH_INDEX_SAMPLE_IDS = 3
BATCH_INDEX_POSITIONS = 4


def prepare_output_for_comparison(
    outputs: torch.Tensor,
    target_size: tuple[int, int],
    output_size: int | None = None,
) -> torch.Tensor:
    """Prepare model outputs for comparison with target mask.

    When using context window (model output larger than output_size), center-crops
    to output_size first, then upsamples to target_size.

    Args:
        outputs: Model predictions with shape (B, C, H, W)
        target_size: Target spatial size (height, width) to match mask
        output_size: Expected output spatial size for center-cropping.
            If provided and output is larger, center-crops to this size.
            Use sentinel_patch_size when using context window.

    Returns:
        Tensor with shape (B, C, target_size[0], target_size[1])

    """
    if outputs.shape[-2:] == target_size:
        return outputs

    out_h, out_w = outputs.shape[-2:]

    # Center-crop if using context window
    if output_size is not None and out_h > output_size:
        crop_margin_h = (out_h - output_size) // 2
        crop_margin_w = (out_w - output_size) // 2
        outputs = outputs[
            :,
            :,
            crop_margin_h : crop_margin_h + output_size,
            crop_margin_w : crop_margin_w + output_size,
        ]

    # Upsample to target size
    if outputs.shape[-2:] != target_size:
        outputs = F.interpolate(
            outputs,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return outputs


def _log_model_description(
    model: torch.nn.Module,
    device: torch.device,
    sample_input_shape: tuple[int, ...],
    sample_inputs: torch.Tensor | None = None,
    batch_positions: torch.Tensor | None = None,
) -> None:
    """Create and log model stats. Failures are logged but don't raise."""
    try:
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
    except Exception:
        logger.warning("Failed to log model description (non-fatal)", exc_info=True)


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
    scheduler: LRScheduler | None = None,
    output_size: int | None = None,
    gradient_clip_val: float | None = None,
    sentinel_augmenter: Any | None = None,
) -> float:
    """Train a temporal model for a single epoch and return average loss.

    Temporal models receive batch_positions and pad_mask arguments.

    Args:
        scheduler: Optional step-level scheduler (e.g., OneCycleLR). If provided,
            scheduler.step() is called after each optimizer step.
        gradient_clip_val: If provided, clip gradients to this max norm.
        sentinel_augmenter: Optional SentinelAugmentation instance for geometric augmentations.

    """
    model.train()
    optimizer.zero_grad()

    scaler_enabled = use_amp and device.type == "cuda"
    if _USE_NEW_AMP_API:
        scaler = GradScaler(device=device.type, enabled=scaler_enabled)
    else:
        scaler = GradScaler(enabled=scaler_enabled)
    total_loss = 0.0

    forward_sig = inspect.signature(model.forward)
    supports_pad_mask = "pad_mask" in forward_sig.parameters

    for batch_idx, batch_data in enumerate(loader):
        x = batch_data[BATCH_INDEX_INPUTS].to(device)
        y = batch_data[BATCH_INDEX_TARGETS].to(device)
        pad_mask = batch_data[BATCH_INDEX_PAD_MASK].to(device)
        batch_positions = batch_data[BATCH_INDEX_POSITIONS].to(device)

        # Apply sentinel augmentation if configured
        if sentinel_augmenter is not None:
            batch_size = x.size(0)
            aug_x_list = []
            aug_y_list = []
            for i in range(batch_size):
                aug_x, aug_y = sentinel_augmenter(x[i], y[i])
                aug_x_list.append(aug_x)
                aug_y_list.append(aug_y)
            x = torch.stack(aug_x_list)
            y = torch.stack(aug_y_list)

        optimizer.zero_grad()
        if _USE_NEW_AMP_API:
            ctx = autocast(device_type=device.type, enabled=use_amp)
        else:
            ctx = autocast(enabled=use_amp)

        if supports_pad_mask:
            with ctx:
                outputs = model(x, batch_positions=batch_positions, pad_mask=pad_mask)
                # Prepare outputs for loss (center-crop + upsample if using context window)
                outputs = prepare_output_for_comparison(outputs, y.shape[-2:], output_size)
                loss = criterion(outputs, y)
                loss_value = loss.item()
                loss = loss / accumulation_steps
        else:
            with ctx:
                outputs = model(x, batch_positions=batch_positions)
                outputs = prepare_output_for_comparison(outputs, y.shape[-2:], output_size)
                loss = criterion(outputs, y)
                loss_value = loss.item()
                loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Step-level scheduler update (for OneCycleLR, etc.)
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss_value)

    return total_loss / len(loader)


def _train_epoch_standard(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augmenter: FlairAugmentation | None = None,
    chessmix: ChessMix | None = None,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    scheduler: LRScheduler | None = None,
    gradient_clip_val: float | None = None,
) -> float:
    """Train a standard (non-temporal) model for a single epoch and return average loss.

    Args:
        scheduler: Optional step-level scheduler (e.g., OneCycleLR). If provided,
            scheduler.step() is called after each optimizer step.

    """
    model.train()
    optimizer.zero_grad()

    scaler_enabled = use_amp and device.type == "cuda"
    if _USE_NEW_AMP_API:
        scaler = GradScaler(device=device.type, enabled=scaler_enabled)
    else:
        scaler = GradScaler(enabled=scaler_enabled)
    total_loss = 0.0

    for batch_idx, batch_data in enumerate(loader):
        x = batch_data[BATCH_INDEX_INPUTS].to(device)
        y = batch_data[BATCH_INDEX_TARGETS].to(device)

        if augmenter is not None:
            x, y = augmenter(x, y)

        if chessmix is not None:
            x, y = chessmix(x, y)

        if _USE_NEW_AMP_API:
            ctx = autocast(device_type=device.type, enabled=use_amp)
        else:
            ctx = autocast(enabled=use_amp)

        with ctx:
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_value = loss.item()
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Step-level scheduler update (for OneCycleLR, etc.)
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss_value)

    return total_loss / len(loader)


def _validate_epoch_temporal(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int | None = None,
    output_size: int | None = None,
) -> tuple[float, float]:
    """Validate a temporal model for a single epoch and return average loss and mIoU.

    Temporal models receive batch_positions and pad_mask arguments.

    Args:
        output_size: Expected output spatial size for center-cropping when using context window.

    """
    model.eval()
    val_loss = 0.0

    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
    ).to(device)

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
            outputs = prepare_output_for_comparison(outputs, y.shape[-2:], output_size)

            loss = criterion(outputs, y)
            val_loss += float(loss.item())

            preds = outputs.argmax(dim=1)
            miou_metric.update(preds, y)

    return val_loss / len(loader), miou_metric.compute().item()


def _validate_epoch_standard(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int | None = None,
) -> tuple[float, float]:
    """Validate a standard (non-temporal) model for a single epoch and return average loss and mIoU."""
    model.eval()
    val_loss = 0.0

    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
    ).to(device)

    with torch.no_grad():
        for batch_data in loader:
            x = batch_data[BATCH_INDEX_INPUTS].to(device)
            y = batch_data[BATCH_INDEX_TARGETS].to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += float(loss.item())

            preds = outputs.argmax(dim=1)
            miou_metric.update(preds, y)

    return val_loss / len(loader), miou_metric.compute().item()


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
    other_class_index: int | None = None,
    accumulation_steps: int = 1,
    early_stopping_criterion: str = "loss",
    *,
    use_amp: bool = False,
    apply_augmentations: bool = True,
    data_config: dict[str, Any] | None = None,
    log_evaluation_metrics: bool = True,
    log_model: bool = True,
    pruning_callback: Any | None = None,
    output_size: int | None = None,
    gradient_clip_val: float | None = None,
    sentinel_augmenter: Any | None = None,
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
        data_config: Full data configuration dict (contains augmentation config,
            normalization settings, and channel selections).
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
        dict: History of training and validation losses/mIoUs, and best values.
            The best model is logged as an MLflow artifact 'best_model' if
            log_to_mlflow is True.

    """
    if early_stopping_criterion not in ("loss", "miou"):
        msg = f"early_stopping_criterion must be 'loss' or 'miou', got {early_stopping_criterion!r}"
        raise ValueError(msg)
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
    best_val_miou = 0.0
    no_improve = 0
    losses_train: list[float] = []
    losses_val: list[float] = []
    mious_val: list[float] = []

    augmenter = FlairAugmentation(data_config) if apply_augmentations and data_config else None
    chessmix = None
    if apply_augmentations and data_config:
        aug_config = data_config.get("data_augmentation", {}).get("augmentations", {})
        if "chessmix" in aug_config:
            cm_cfg = aug_config["chessmix"]
            chessmix = ChessMix(
                prob=cm_cfg.get("prob", 0.5),
                grid_sizes=cm_cfg.get("grid_sizes", [4]),
                ignore_index=cm_cfg.get("ignore_index", 12),
                class_counts=cm_cfg.get("class_counts", None),
                num_classes=data_config.get("num_classes", 13),
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

    is_step_scheduler = scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau)
    step_scheduler = scheduler if is_step_scheduler else None

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
                step_scheduler,
                output_size,
                gradient_clip_val,
                sentinel_augmenter,
            )
        else:
            loss_epoch = train_epoch_fn(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                augmenter,
                chessmix,
                accumulation_steps,
                use_amp,
                step_scheduler,
            )
        losses_train.append(loss_epoch)
        logger.info("Epoch %d/%d: Training Loss: %.4f", epoch + 1, epochs, loss_epoch)

        val_loss, val_miou = validate_epoch_fn(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            other_class_index,
            output_size,
        )
        losses_val.append(val_loss)
        mious_val.append(val_miou)
        logger.info(
            "Epoch %d/%d: Validation Loss: %.4f, Validation mIoU: %.4f",
            epoch + 1,
            epochs,
            val_loss,
            val_miou,
        )

        if log_evaluation_metrics:
            log_metrics_to_mlflow(
                metrics={"train_loss": loss_epoch, "val_loss": val_loss, "val_miou": val_miou},
                step=epoch,
            )

        if pruning_callback is not None:
            report_value = val_miou if early_stopping_criterion == "miou" else val_loss
            pruning_callback(report_value, epoch)

        if early_stopping_criterion == "miou":
            improved = val_miou > best_val_miou
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_miou = val_miou
            best_val_loss = val_loss
            no_improve = 0
            if early_stopping_criterion == "miou":
                logger.info("Validation mIoU improved to %.4f", best_val_miou)
            else:
                logger.info("Validation loss improved to %.4f", best_val_loss)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            logger.info("No improvement for %d epochs.", no_improve)
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

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
        "val_miou": mious_val,
        "best_val_loss": best_val_loss,
        "best_val_miou": best_val_miou,
    }
