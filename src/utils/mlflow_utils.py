import logging
import tempfile
from typing import Optional

import mlflow
import torch
from matplotlib import pyplot as plt

from src.visualization.plot import generate_comparison_figure, plot_confusion_matrix

logger = logging.getLogger(__name__)


def init_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """
    Initializes MLflow with the given tracking URI and experiment name.

    Args:
        tracking_uri: URI to the MLflow tracking server.
        experiment_name: The name of the experiment to track in MLflow.

    Returns:
        None
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(
        "MLflow initialized with tracking URI: %s and experiment: %s",
        tracking_uri,
        experiment_name,
    )


def log_metrics_to_mlflow(
    metrics: dict, step: Optional[int] = None, prefix: str = ""
) -> None:
    """
    Logs metrics to MLflow.

    Args:
        metrics: A dictionary of metrics to log.
        step: Optional step number for logging.
        prefix: Optional prefix for metric names.
    """
    logger.info("Logging metrics to MLflow.")
    for name, value in metrics.items():
        mlflow.log_metric(key=f"{prefix}{name}", value=value, step=step)


def log_comparison_to_mlflow(
    sample_id: str, ground_truth: torch.Tensor, prediction: torch.Tensor
) -> None:
    """
    Logs comparison figures of ground truth and predicted masks to MLflow.

    Args:
        sample_id: Identifier for the sample.
        ground_truth: Ground truth mask.
        prediction: Predicted mask.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig = generate_comparison_figure(sample_id, ground_truth, prediction)
        fig_path = f"{tmp.name}_{sample_id}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="plots/comparison")


def log_confusion_to_mlflow(
    conf_matrix: torch.Tensor,
    class_names: list[str],
    other_class_index: int,
    normalize: bool = False,
) -> None:
    """
    Logs confusion matrix plot to MLflow using a temporary file.

    Args:
        conf_matrix (torch.Tensor): Final confusion matrix.
        class_names (list[str]): List of class names for axis labels.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig = plot_confusion_matrix(
            conf_matrix=conf_matrix,
            class_names=class_names,
            other_class_index=other_class_index,
            normalize=normalize,
            title="Normalized Confusion Matrix",
        )
        fig.savefig(tmp.name)
        plt.close(fig)
        mlflow.log_artifact(tmp.name, artifact_path="plots")
