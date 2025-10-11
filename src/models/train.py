import io
import logging
import tempfile
from pathlib import Path

import mlflow
import torch
from torchinfo import summary

from src.data.pre_processing.data_augmentation import FlairAugmentation
from src.models.utils import process_segmentation_tensor
from src.utils.mlflow_utils import log_metrics_to_mlflow, log_model_to_mlflow
from src.utils.model_stats import compute_model_complexity

logger = logging.getLogger(__name__)


def compute_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
    num_classes: int = 13,
) -> torch.Tensor:
    """Compute the loss for a given model, inputs, and targets.

    Args:
        model: PyTorch model.
        inputs: Input tensor.
        targets: Target tensor.
        criterion: Loss function.
        num_classes: Number of classes in the segmentation task.

    Returns:
        torch.Tensor: Computed loss.

    """
    outputs = model(inputs)  # (N, C, H, W)
    targets = process_segmentation_tensor(targets, num_classes)  # (N, H, W)
    return criterion(outputs, targets)


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 20,
    num_classes: int = 13,
    *,
    apply_augmentations: bool = True,
    augmentation_config: dict | None = None,
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
        apply_augmentations: Whether to apply augmentations to the training data.
            Defaults to True.
        augmentation_config: Configuration for augmentations. Defaults to None.
        epochs: Maximum number of epochs to train. Defaults to 100.
        patience: Early stopping patience. Defaults to 20.
        num_classes: Number of classes in the segmentation task. Defaults to 13.
        log_evaluation_metrics: Whether to log metrics and models to MLflow.
            Defaults to True.
        log_model: Whether to log the best model to MLflow. Defaults to True.

    Returns:
        dict: History of training and validation losses, and best validation loss.
            The best model is logged as an MLflow artifact 'best_model' if
            log_to_mlflow is True.

    Raises:
        ValueError: If train_loader or val_loader are empty.

    """
    if len(train_loader) == 0:
        msg = "Training data loader is empty"
        raise ValueError(msg)
    if len(val_loader) == 0:
        msg = "Validation data loader is empty"
        raise ValueError(msg)

    model.to(device)

    sample_input_shape = next(iter(train_loader))[0].shape
    model_info = ""

    if log_evaluation_metrics:
        with io.StringIO() as buf:
            model_summary = summary(
                model,
                input_size=sample_input_shape,
                device=str(device),
                verbose=0,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
                row_settings=["var_names"],
            )
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
            input_size=tuple(int(x) for x in sample_input_shape),
        )
        for k, v in complexity.items():
            mlflow.log_metric(k, float(v))

    best_val_loss = float("inf")
    no_improve = 0
    losses_train: list[float] = []
    losses_val: list[float] = []

    augmenter = FlairAugmentation(augmentation_config) if apply_augmentations else None
    best_model_state = None

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets, *_ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if apply_augmentations:
                inputs, targets = augmenter(inputs, targets)

            optimizer.zero_grad()
            loss = compute_loss(model, inputs, targets, criterion, num_classes)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        loss_epoch = total_loss / len(train_loader)
        losses_train.append(loss_epoch)
        logger.info("Epoch %d/%d: Training Loss: %.4f", epoch + 1, epochs, loss_epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, *_ in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                loss = compute_loss(model, inputs, targets, criterion, num_classes)
                val_loss += float(loss.item())

        val_loss /= len(val_loader)
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

        lr_scheduler.step()

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
