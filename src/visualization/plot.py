import json
import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.visualization.utils import get_custom_colormap

logger = logging.getLogger(__name__)


def visualize_training_history(history: dict) -> plt.Figure:
    """
    Creates a matplotlib Figure showing training history.

    Args:
        history: A dictionary containing training and validation losses and metrics.

    Returns:
        A matplotlib Figure object.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(epochs, history['train_loss'], label='Training Loss')
    axs[0].plot(epochs, history['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    for metric, values in history.items():
        if metric not in ['train_loss', 'val_loss']:
            axs[1].plot(epochs, values, label=metric.capitalize())
    axs[1].set_title('Validation Metrics')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Score')
    axs[1].legend()

    fig.tight_layout()
    return fig


def save_training_history(history: dict, file_path: str = './training_history.json'):
    """
    Saves the training history to a JSON file.

    Args:
        history: A dictionary containing training and validation losses and metrics.
        file_path: The path to save the JSON file. Defaults to './training_history.json'.
    """
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)


def generate_comparison_figure(
        sample_id: str,
        ground_truth: torch.Tensor,
        prediction: torch.Tensor
) -> plt.Figure:
    """
    Creates a matplotlib Figure comparing ground truth and predicted mask for a single sample.

    Args:
        sample_id: Identifier for the sample.
        ground_truth: Ground truth mask (H, W) or one-hot encoded (C, H, W).
        prediction: Predicted mask (H, W) or probability map (C, H, W).

    Returns:
        matplotlib.figure.Figure: The comparison figure with GT and prediction side by side.
    """
    custom_cmap, custom_norm = get_custom_colormap()

    if ground_truth.ndim == 3:
        ground_truth = torch.argmax(ground_truth, dim=0)
    if prediction.ndim == 3:
        prediction = torch.argmax(prediction, dim=0)

    gt_np = ground_truth.cpu().numpy()
    pred_np = prediction.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(gt_np, cmap=custom_cmap, norm=custom_norm)
    axs[0].set_title(f"Ground Truth - ID: {sample_id}")
    axs[0].axis("off")

    axs[1].imshow(pred_np, cmap=custom_cmap, norm=custom_norm)
    axs[1].set_title(f"Prediction - ID: {sample_id}")
    axs[1].axis("off")

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
        conf_matrix: torch.Tensor,
        class_names: Optional[list[str]] = None,
        other_class_index: Optional[int] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plots a confusion matrix using seaborn heatmap.

    Args:
        conf_matrix: Confusion matrix (shape: [num_classes, num_classes]).
        class_names: List of class names.
        other_class_index: Index of class to exclude from mIoU.
        normalize: If True, normalize rows to sum to 1.
        title: Title of the plot.
        figsize (tuple): Size of the figure.
    """
    conf_matrix = conf_matrix.cpu().numpy()
    conf_matrix = np.delete(conf_matrix, other_class_index, axis=0)
    conf_matrix = np.delete(conf_matrix, other_class_index, axis=1)
    if class_names:
        class_names = [name for i, name in enumerate(class_names) if i != other_class_index]

    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.divide(conf_matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=class_names, yticklabels=class_names, cmap="Blues", square=True, ax=ax,
                cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    plt.tight_layout()
    return fig
