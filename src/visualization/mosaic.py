"""Zone-based mosaic visualization for FLAIR-2 dataset.

This module provides functions to create prediction mosaic visualizations.
"""

import inspect
import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.utils import process_segmentation_tensor
from src.visualization.utils import get_custom_colormap

logger = logging.getLogger(__name__)

# Constants for batch indexing
BATCH_INDEX_INPUTS = 0
BATCH_INDEX_PAD_MASK = 2
BATCH_INDEX_POSITIONS = 4
TEMPORAL_MODEL_NDIM = 5


def create_prediction_mosaic(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    grid_size: int = 10,
    patch_size: int = 512,
) -> np.ndarray:
    """Create a prediction mosaic by running model on all patches.

    Args:
        model: Trained model for inference.
        data_loader: DataLoader providing images in sorted order.
        device: Torch device for inference.
        num_classes: Number of output classes.
        grid_size: Size of the grid (grid_size x grid_size).
        patch_size: Size of each patch in pixels.

    Returns:
        Mosaic array of predictions with shape (H, W).

    """
    model.eval()
    model.to(device)

    forward_sig = inspect.signature(model.forward)
    supports_pad_mask = "pad_mask" in forward_sig.parameters

    all_predictions: list[np.ndarray] = []

    def _is_multimodal_batch(batch: object) -> bool:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            return False
        if not torch.is_tensor(batch[0]) or not torch.is_tensor(batch[1]):
            return False
        return batch[0].ndim == 4 and batch[1].ndim == TEMPORAL_MODEL_NDIM

    with torch.no_grad():
        for batch in data_loader:
            if _is_multimodal_batch(batch):
                aerial = batch[0].to(device)
                sentinel = batch[1].to(device)
                batch_positions = batch[4].to(device) if len(batch) > 4 else None
                pad_mask = batch[5].to(device) if len(batch) > 5 else None

                if batch_positions is not None and pad_mask is not None and supports_pad_mask:
                    outputs = model(
                        aerial,
                        sentinel,
                        batch_positions=batch_positions,
                        pad_mask=pad_mask,
                    )
                elif batch_positions is not None:
                    outputs = model(aerial, sentinel, batch_positions=batch_positions)
                else:
                    outputs = model(aerial, sentinel)
            else:
                inputs = batch[BATCH_INDEX_INPUTS].to(device)

                if inputs.ndim == TEMPORAL_MODEL_NDIM:
                    pad_mask = (
                        batch[BATCH_INDEX_PAD_MASK].to(device)
                        if len(batch) > BATCH_INDEX_PAD_MASK
                        else None
                    )
                    batch_positions = (
                        batch[BATCH_INDEX_POSITIONS].to(device)
                        if len(batch) > BATCH_INDEX_POSITIONS
                        else None
                    )

                    if batch_positions is not None and pad_mask is not None and supports_pad_mask:
                        outputs = model(inputs, batch_positions=batch_positions, pad_mask=pad_mask)
                    elif batch_positions is not None:
                        outputs = model(inputs, batch_positions=batch_positions)
                    else:
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

            preds = process_segmentation_tensor(outputs, num_classes=num_classes)
            all_predictions.extend(preds.cpu().numpy())

    num_patches = len(all_predictions)
    actual_grid = min(grid_size, int(np.ceil(np.sqrt(num_patches))))
    max_patches = actual_grid * actual_grid

    logger.info(
        "Creating prediction mosaic: %d predictions, %dx%d grid",
        min(num_patches, max_patches),
        actual_grid,
        actual_grid,
    )

    mosaic = np.zeros((actual_grid * patch_size, actual_grid * patch_size), dtype=np.uint8)

    for idx in range(min(num_patches, max_patches)):
        row = idx // actual_grid
        col = idx % actual_grid

        y_start = row * patch_size
        x_start = col * patch_size

        mosaic[y_start : y_start + patch_size, x_start : x_start + patch_size] = all_predictions[
            idx
        ]

    return mosaic


def plot_prediction_mosaic(
    prediction_mosaic: np.ndarray,
    title: str = "Prediction Mosaic",
    figsize: tuple[int, int] = (12, 12),
) -> plt.Figure:
    """Create a plot of the prediction mosaic.

    Args:
        prediction_mosaic: Class prediction mosaic (H, W).
        title: Title for the plot.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure.

    """
    custom_cmap, custom_norm = get_custom_colormap()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(prediction_mosaic, cmap=custom_cmap, norm=custom_norm)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    plt.tight_layout()

    return fig


def log_prediction_mosaic_to_mlflow(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    zone_name: str,
    grid_size: int = 10,
    patch_size: int = 512,
    artifact_path: str = "plots/zone_mosaic",
) -> None:
    """Create and log a prediction mosaic visualization to MLflow.

    Args:
        model: Trained model for inference.
        data_loader: DataLoader for the zone.
        device: Torch device for inference.
        num_classes: Number of output classes.
        zone_name: Name of the zone for labeling.
        grid_size: Size of the grid.
        patch_size: Size of each patch.
        artifact_path: MLflow artifact path.

    """
    logger.info("Creating prediction mosaic for zone: %s", zone_name)

    prediction_mosaic = create_prediction_mosaic(
        model,
        data_loader,
        device,
        num_classes,
        grid_size=grid_size,
        patch_size=patch_size,
    )

    fig = plot_prediction_mosaic(
        prediction_mosaic,
        title=f"Predictions: {zone_name}",
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fig_path = td_path / f"{zone_name}_predictions.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        logger.info("Logging prediction mosaic to MLflow: %s", zone_name)
        mlflow.log_artifact(str(fig_path), artifact_path=artifact_path)
