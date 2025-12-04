import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import dagshub
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from mlflow.exceptions import MlflowException, RestException
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from src.data.pre_processing.flair_dataset import FlairDataset
from src.models.utils import process_segmentation_tensor
from src.visualization.plot import generate_comparison_figure, plot_confusion_matrix
from src.visualization.utils import get_custom_colormap

logger = logging.getLogger(__name__)

DAGSHUB_PARAMS = {"repo_owner", "repo_name", "mlflow", "branch", "log_mlflow"}


def _sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename stem.

    Replaces characters that could introduce directories or be invalid on
    various filesystems. Collapses multiple underscores and strips leading/
    trailing underscores. Falls back to "sample" when the sanitized name
    would be empty.

    Args:
        name: The original filename or identifier to sanitize.

    Returns:
        A filesystem-safe string containing only alphanumeric characters,
        hyphens, and underscores.

    """
    safe = re.sub(r"[^\w\-]", "_", name.strip())
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_")


def init_mlflow(
    tracking_uri: str | None,
    experiment_name: str,
    dagshub_config: dict[str, Any] | None = None,
) -> None:
    """Initialize MLflow and optionally bootstrap Dagshub integration.

    Args:
        tracking_uri: URI to the MLflow tracking server. If ``None`` and Dagshub
            integration is disabled, MLflow will use its default local store.
        experiment_name: The name of the experiment to track in MLflow.
        dagshub_config: Optional Dagshub configuration dictionary. Expected keys
            include ``enabled``, ``repo_owner``, ``repo_name``, ``mlflow`` and
            ``branch``. When ``enabled`` is truthy, the function will call
            :func:`dagshub.init` before configuring MLflow.

    """
    dagshub_active = False
    if dagshub_config and dagshub_config.get("enabled"):
        kwargs = {k: v for k, v in dagshub_config.items() if k in DAGSHUB_PARAMS and v is not None}

        dagshub.init(**kwargs)
        dagshub_active = True
        logger.info(
            "DagsHub initialized for %s/%s (MLflow tracking configured automatically)",
            dagshub_config.get("repo_owner"),
            dagshub_config.get("repo_name"),
        )

    if not dagshub_active:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking URI: %s", tracking_uri)
        else:
            logger.info("No tracking URI configured. Using local MLflow store (./mlruns).")

    mlflow.set_experiment(experiment_name)
    logger.info("MLflow experiment set to '%s' ", experiment_name)


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


def log_confusion_matrix_to_mlflow(
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
        with Path.open(csv_path, "w", encoding="utf-8") as f:
            np.savetxt(f, cm_np, delimiter=",", header=header, comments="", fmt="%g")
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
    try:
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=model_name,
            signature=signature,
        )
    except RestException as exc:
        message = str(exc).lower()
        if "unsupported endpoint" in message or "405" in message:
            logger.warning(
                "Skipping MLflow model logging because the tracking server "
                "reported an unsupported endpoint: %s",
                exc,
            )
            return
        raise
    except MlflowException as exc:
        logger.warning("Skipping MLflow model logging due to exception: %s", exc)
        return


def log_prediction_plots(
    outputs: torch.Tensor,
    ids: list[str],
    num_classes: int,
    intersect: set[str] | None = None,
) -> None:
    """Log prediction plots to MLflow.

    Args:
        outputs: Model outputs (predictions) with shape (batch, ...).
        ids: List of sample IDs corresponding to each output.
        num_classes: Number of classes for segmentation.
        intersect: Set of IDs to include in logging. Only samples with IDs
            in this set will be logged.

    """
    n = min(len(outputs), len(ids))
    if intersect is None:
        filtered_indices = list(range(n))
    else:
        filtered_indices = [i for i, sample_id in enumerate(ids[:n]) if sample_id in intersect]

    if not filtered_indices:
        return

    outputs_np = process_segmentation_tensor(outputs, num_classes=num_classes).cpu().numpy()
    custom_cmap, _ = get_custom_colormap()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        for idx in filtered_indices:
            sample_id = ids[idx]
            output = outputs_np[idx]
            safe_id = _sanitize_filename(sample_id)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(output, cmap=custom_cmap)
            ax.axis("off")

            fig_path = td_path / f"{safe_id}_prediction.png"
            fig.savefig(fig_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            mlflow.log_artifact(str(fig_path), artifact_path="plots/prediction")


def log_mosaic_plot(
    mosaic: np.ndarray,
    name: str = "mosaic",
    artifact_path: str = "plots/mosaic",
) -> None:
    """Log a stitched mosaic image to MLflow.

    Args:
        mosaic: The stitched mosaic array (H, W) for class predictions.
        name: Name for the saved artifact file.
        artifact_path: MLflow artifact path for organization.

    """
    if mosaic is None:
        logger.warning("Mosaic is None, skipping logging")
        return

    custom_cmap, custom_norm = get_custom_colormap()

    # Calculate figure size based on mosaic dimensions
    # Using a reasonable DPI that keeps files manageable
    height, width = mosaic.shape[-2:]
    fig_width = width / 100  # 100 pixels per inch for reasonable file size
    fig_height = height / 100

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(mosaic, cmap=custom_cmap, norm=custom_norm)
    ax.axis("off")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fig_path = td_path / f"{_sanitize_filename(name)}.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150, pad_inches=0)
        plt.close(fig)

        logger.info("Logging mosaic image to MLflow: %s", fig_path)
        mlflow.log_artifact(str(fig_path), artifact_path=artifact_path)
