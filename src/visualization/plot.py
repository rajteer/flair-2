import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.visualization.utils import get_custom_colormap

logger = logging.getLogger(__name__)


def generate_comparison_figure(
    sample_id: str,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    labels: dict | None = None,
) -> plt.Figure:
    """Create a matplotlib Figure comparing ground truth and predicted mask for a single sample.

    Args:
        sample_id: Identifier for the sample.
        ground_truth: Ground truth mask (H, W) or one-hot encoded (C, H, W).
        prediction: Predicted mask (H, W) or probability map (C, H, W).
        labels: Dictionary containing label texts for ground_truth and prediction.

    Returns:
        matplotlib.figure.Figure: The comparison figure with GT and prediction side by side.

    """
    if labels is None:
        labels = {"ground_truth": "Ground Truth", "prediction": "Prediction"}

    custom_cmap, custom_norm = get_custom_colormap()

    if ground_truth.ndim == 3:  # noqa: PLR2004
        ground_truth = torch.argmax(ground_truth, dim=0)
    if prediction.ndim == 3:  # noqa: PLR2004
        prediction = torch.argmax(prediction, dim=0)

    gt_np = ground_truth.cpu().numpy()
    pred_np = prediction.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(gt_np, cmap=custom_cmap, norm=custom_norm)
    axs[0].set_title(f"{labels.get('ground_truth', 'Ground Truth')} - ID: {sample_id}")
    axs[0].axis("off")

    axs[1].imshow(pred_np, cmap=custom_cmap, norm=custom_norm)
    axs[1].set_title(f"{labels.get('prediction', 'Prediction')} - ID: {sample_id}")
    axs[1].axis("off")

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    conf_matrix: torch.Tensor,
    class_names: list[str] | None = None,
    other_class_index: int | None = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    labels: dict | None = None,
    *,
    normalize: bool = False,
) -> plt.Figure:
    """Plot a confusion matrix using seaborn heatmap.

    Args:
        conf_matrix: Confusion matrix (shape: [num_classes, num_classes]).
        class_names: List of class names.
        other_class_index: Index of class to exclude from mIoU.
        normalize: If True, normalize rows to sum to 1.
        title: Title of the plot.
        figsize (tuple): Size of the figure.
        labels: Dictionary containing label texts for predicted_class and actual_class.

    """
    if labels is None:
        labels = {"predicted_class": "Predicted Class", "actual_class": "Actual Class"}

    conf_matrix = conf_matrix.cpu().numpy()
    conf_matrix = np.delete(conf_matrix, other_class_index, axis=0)
    conf_matrix = np.delete(conf_matrix, other_class_index, axis=1)
    if class_names:
        class_names = [name for i, name in enumerate(class_names) if i != other_class_index]

    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.divide(conf_matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        square=True,
        ax=ax,
        cbar=False,
    )
    plt.xlabel(labels.get("predicted_class", "Predicted Class"))
    plt.ylabel(labels.get("actual_class", "Actual Class"))
    plt.title(title)

    plt.tight_layout()
    return fig
