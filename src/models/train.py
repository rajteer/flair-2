import io
import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from torchinfo import summary

from src.data.pre_processing.data_augmentation import FlairAugmentation
from src.models.utils import process_segmentation_tensor
from src.utils.mlflow_utils import log_metrics_to_mlflow

logger = logging.getLogger(__name__)


def compute_loss(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
                 criterion: torch.nn.Module, num_classes: int = 13) -> torch.Tensor:
    """
    Computes the loss for a given model, inputs, and targets.

    Args:
        model: PyTorch model.
        inputs: Input tensor.
        targets: Target tensor.
        criterion: Loss function.
        num_classes: Number of classes in the segmentation task.
    Returns:
        torch.Tensor: Computed loss.
    """
    outputs = model(inputs)
    targets = process_segmentation_tensor(targets, num_classes)
    return criterion(outputs, targets)


def train(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        path_to_requirements: str,
        apply_augmentations: bool = True,
        augmentation_prob: float = 0.5,
        epochs: int = 100,
        patience: int = 20,
        num_classes: int = 13,
        log_to_mlflow: bool = True,
) -> dict[str, list[float] | float]:
    """
    Trains a segmentation model, monitoring validation loss and saving the best model.
    Detailed metrics should be calculated separately after training using an evaluation function.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device (CPU/GPU).
        path_to_requirements: Path to the requirements file for MLflow logging.
        apply_augmentations: Whether to apply augmentations to the training data.
        augmentation_prob: Probability of applying augmentations.
        epochs: Maximum number of epochs to train.
        patience: Early stopping patience.
        num_classes: Number of classes in the segmentation task.
        log_to_mlflow: Whether to log metrics and models to MLflow (disable during hyperparameter search)

    Returns:
        dict: History of training and validation losses, and best validation loss.
              The best model is logged as an MLflow artifact 'best_model' if log_to_mlflow is True.
    """

    sample_input_shape = next(iter(train_loader))[0].shape
    model_info = ""

    if log_to_mlflow:
        with io.StringIO() as buf:
            model_summary = summary(
                model,
                input_size=sample_input_shape,
                device=device,
                verbose=0,
                col_names=["input_size", "output_size", "num_params", "kernel_size",
                           "mult_adds"],
                row_settings=["var_names"]
            )
            print(model_summary, file=buf)
            model_info = buf.getvalue()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                         encoding='utf-8') as tmp:
            tmp.write(model_info)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, "model_architecture")
        Path(tmp_path).unlink()

    best_val_loss = float("inf")
    no_improve = 0
    losses_train = []
    losses_val = []
    augmenter = FlairAugmentation(augmentation_prob)
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)

            if apply_augmentations:
                inputs, targets = augmenter(inputs, targets)

            optimizer.zero_grad()
            loss = compute_loss(model, inputs, targets, criterion, num_classes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_epoch = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        losses_train.append(loss_epoch)
        logger.info(f"Epoch {epoch + 1}/{epochs}: Training Loss: {loss_epoch:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, targets, _ = batch
                inputs, targets = inputs.to(device), targets.to(device)
                loss = compute_loss(model, inputs, targets, criterion, num_classes)
                val_loss += loss.item()

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        losses_val.append(val_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}: Validation Loss: {val_loss:.4f}")

        if log_to_mlflow:
            log_metrics_to_mlflow(
                metrics={"train_loss": loss_epoch, "val_loss": val_loss},
                step=epoch,
                prefix="",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            logger.info("Validation loss decreased")
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            logger.info(f"No improvement for {no_improve} epochs.")
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

        if log_to_mlflow:
            logger.info("Logging final best model to MLflow")

            input_shape = sample_input_shape

            input_schema = Schema([
                TensorSpec(np.dtype(np.float32),
                           shape=(-1, len(train_loader.dataset.selected_channels),
                                  input_shape[2], input_shape[3]),
                           name="input_image")
            ])

            output_schema = Schema([
                TensorSpec(np.dtype(np.float32),
                           shape=(-1, num_classes, input_shape[2], input_shape[3]),
                           name="segmentation_map")
            ])

            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="best_model",
                pip_requirements=path_to_requirements,
                signature=signature,
                input_example=next(iter(train_loader))[0].cpu().numpy()[:1]
            )

    return {
        'train_loss': losses_train,
        'val_loss': losses_val,
        'best_val_loss': best_val_loss,
    }
