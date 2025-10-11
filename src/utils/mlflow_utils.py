import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from src.data.pre_processing.FlairDataset import FlairDataset
from src.models.utils import process_segmentation_tensor
from src.visualization.plot import generate_comparison_figure, plot_confusion_matrix
from src.visualization.utils import get_custom_colormap

logger = logging.getLogger(__name__)


def init_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Initialize MLflow with the given tracking URI and experiment name.

    Args:
        tracking_uri: URI to the MLflow tracking server.
        experiment_name: The name of the experiment to track in MLflow.

    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(
        "MLflow initialized with tracking URI: %s and experiment: %s",
        tracking_uri,
        experiment_name,
    )


def log_metrics_to_mlflow(
    metrics: dict,
    step: int | None = None,
    prefix: str = "",
) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: A dictionary of metrics to log.
        step: Optional step number for logging.
        prefix: Optional prefix for metric names.

    """
    logger.info("Logging metrics to MLflow.")
    for name, value in metrics.items():
        mlflow.log_metric(key=f"{prefix}{name}", value=value, step=step)


def log_comparison_to_mlflow(
    sample_id: str,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    labels: dict | None = None,
) -> None:
    """Create and upload comparison figures to MLflow.

    Args:
        sample_id: Identifier for the sample.
        ground_truth: Ground truth mask.
        prediction: Predicted mask.
        labels: Dictionary containing label texts for visualization.

    """
    fig = generate_comparison_figure(sample_id, ground_truth, prediction, labels=labels)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fig_path = td_path / f"{sample_id}_comparison.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(fig_path), artifact_path="plots/comparison")


def log_confusion_to_mlflow(
    conf_matrix: torch.Tensor,
    class_names: list[str],
    other_class_index: int,
    labels: dict | None = None,
    title: str = "Confusion Matrix",
    *,
    normalize: bool = False,
) -> None:
    """Create and upload a confusion matrix figure to MLflow.

    Args:
        conf_matrix: Confusion matrix to visualize.
        class_names: Labels for the matrix axes.
        other_class_index: Index of the "other" class.
        normalize: If True, normalize numeric values before plotting. Defaults to False.
        labels: Dictionary containing label texts for visualization.
        title: Title for the confusion matrix plot.

    """
    fig = plot_confusion_matrix(
        conf_matrix=conf_matrix,
        class_names=class_names,
        other_class_index=other_class_index,
        normalize=normalize,
        title=title,
        labels=labels,
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        fig_path = td_path / "confusion_matrix.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix figure to MLflow: %s", fig_path)
        mlflow.log_artifact(str(fig_path), artifact_path="plots")

        cm_np = conf_matrix.detach().cpu().numpy()
        csv_path = td_path / "confusion_matrix.csv"
        header = ",".join(["", *list(class_names)])
        np.savetxt(csv_path, cm_np, delimiter=",", header=header, comments="", fmt="%g")
        logger.info("Logging confusion matrix CSV to MLflow: %s", csv_path)
        mlflow.log_artifact(str(csv_path), artifact_path="plots")


def log_model_to_mlflow(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    sample_input_shape: tuple,
    num_classes: int,
    model_name: str = "best_model",
) -> None:
    """Log a PyTorch model to MLflow with appropriate input/output signature.

    Args:
        model: The PyTorch model to log
        train_loader: DataLoader containing training dataset (for channel info)
        sample_input_shape: Shape of sample input tensor
        num_classes: Number of classes for segmentation
        model_name: Name to give the model in MLflow

    """
    logger.info("Logging %s to MLflow", model_name)

    input_shape = sample_input_shape
    dataset = train_loader.dataset

    if isinstance(dataset, FlairDataset) and dataset.selected_channels is not None:
        selected_channels = dataset.selected_channels
    else:
        selected_channels = list(range(input_shape[1]))

    input_schema = Schema(
        [
            TensorSpec(
                np.dtype(np.float32),
                shape=(-1, len(selected_channels), input_shape[2], input_shape[3]),
                name="input_image",
            ),
        ],
    )

    output_schema = Schema(
        [
            TensorSpec(
                np.dtype(np.float32),
                shape=(-1, num_classes, input_shape[2], input_shape[3]),
                name="segmentation_map",
            ),
        ],
    )

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name=model_name,
        signature=signature,
    )


def log_prediction_plots(
    outputs: torch.Tensor,
    ids: list[str],
    num_classes: int,
    intersect: set[str],
) -> None:
    """Log prediction plots to MLflow or local directory.

    Args:
        outputs: Model outputs (predictions).
        ids: List of sample IDs.
        num_classes: Number of classes for segmentation.
        intersect: Set of IDs to include in logging.

    """
    outputs = process_segmentation_tensor(outputs, num_classes=num_classes).cpu()
    custom_cmap, _ = get_custom_colormap()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for sample_id, output in zip(ids, outputs):
            if sample_id not in intersect:
                continue
            fig, ax = plt.subplots()
            ax.imshow(output.numpy(), cmap=custom_cmap)
            ax.axis("off")
            fig_path = td_path / f"{sample_id}_prediction.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(fig_path), artifact_path="plots/prediction")
