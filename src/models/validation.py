import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)
from tqdm import tqdm

from src.models.utils import process_segmentation_tensor
from src.utils.mlflow_utils import (
    log_confusion_to_mlflow,
    log_metrics_to_mlflow,
    log_prediction_plots,
)

logger = logging.getLogger(__name__)


def calculate_iou_scores(
    conf_matrix: torch.Tensor,
    num_classes: int,
    other_class_index: int = 13,
) -> tuple[float, dict[int, float]]:
    """Compute mean IoU (mIoU) and per-class IoU."""
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    union = tp + fp + fn

    iou = torch.where(
        union != 0,
        tp.float() / union.float(),
        torch.zeros_like(union, dtype=torch.float),
    )

    valid_classes_mask = torch.ones(
        num_classes,
        dtype=torch.bool,
        device=conf_matrix.device,
    )

    if 0 <= other_class_index < num_classes:
        valid_classes_mask[other_class_index] = False
    else:
        logger.warning(
            "other_class_index %s is out of range [0, %s). It will be ignored.",
            other_class_index,
            num_classes,
        )

    valid_indices = valid_classes_mask.nonzero(as_tuple=True)[0]
    per_class_iou = {int(i): iou[i].item() for i in valid_indices}

    mean_iou = iou[valid_classes_mask].mean().item() if valid_classes_mask.any() else 0.0

    return mean_iou, per_class_iou


def get_evaluation_metrics_dict(
    num_classes: int,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    """Initialize TorchMetrics for multiclass classification."""
    return {
        "conf_matrix": MulticlassConfusionMatrix(num_classes=num_classes).to(device),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro").to(
            device,
        ),
    }


def compute_timing_metrics(
    inference_times: list[float],
    batch_sizes: list[int],
) -> dict[str, float]:
    """Compute timing metrics from inference times and batch sizes."""
    total_inference_time = sum(inference_times)
    total_images = sum(batch_sizes)
    avg_time_per_image = total_inference_time / total_images if total_images > 0 else 0.0
    avg_time_per_batch = sum(inference_times) / len(inference_times) if inference_times else 0.0

    return {
        "total_inference_time": total_inference_time,
        "total_images": total_images,
        "avg_time_per_image": avg_time_per_image,
        "avg_time_per_batch": avg_time_per_batch,
    }


def _finalize_evaluation(
    evaluation_metrics_dict: dict[str, torch.nn.Module],
    inference_times: list[float],
    batch_sizes: list[int],
    class_name_mapping: dict[int, str],
    confusion_other_index: int,
    *,
    normalize_confusion_matrix: bool,
    visualization_labels: dict[str, str] | None,
    log_eval_metrics: bool,
    log_confusion_matrix: bool,
) -> dict[str, float]:
    """Compute final metrics, log them and return the metrics dictionary."""
    confusion_matrix = evaluation_metrics_dict["conf_matrix"].compute()
    miou, per_class_iou = calculate_iou_scores(
        confusion_matrix,
        len(class_name_mapping),
        confusion_other_index,
    )
    timing_metrics = compute_timing_metrics(inference_times, batch_sizes)

    metrics: dict[str, float] = {
        "miou": miou,
        "f1": evaluation_metrics_dict["f1"].compute().item(),
        "accuracy": evaluation_metrics_dict["accuracy"].compute().item(),
        "total_batches_processed": len(inference_times),
    }
    metrics.update(timing_metrics)

    per_class_metrics = {
        f"iou_class_{class_name_mapping.get(i, f'class_{i}')}": v for i, v in per_class_iou.items()
    }
    metrics.update(per_class_metrics)

    if log_eval_metrics:
        log_metrics_to_mlflow(metrics)

    if log_confusion_matrix:
        title = "Confusion Matrix"
        labels = {
            "predicted_class": "Predicted Class",
            "actual_class": "Actual Class",
        }
        if visualization_labels:
            title = visualization_labels.get("confusion_matrix_title", title)
            labels["predicted_class"] = visualization_labels.get(
                "predicted_class",
                labels["predicted_class"],
            )
            labels["actual_class"] = visualization_labels.get(
                "actual_class",
                labels["actual_class"],
            )

        log_confusion_to_mlflow(
            confusion_matrix,
            list(class_name_mapping.values()),
            confusion_other_index,
            normalize=normalize_confusion_matrix,
            labels=labels,
            title=title,
        )

    return metrics


def evaluate(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    num_classes: int,
    other_class_index: int = 13,
    *,
    log_eval_metrics: bool = True,
    log_confusion_matrix: bool = True,
    normalize_confusion_matrix: bool = True,
    sample_ids_to_plot: list[str] | None = None,
    warmup_runs: int = 10,
    visualization_labels: dict[str, str] | None = None,
    class_name_mapping: dict[int, str] | None = None,
) -> dict[str, float]:
    """Evaluate model and log metrics and plots.

    Args:
        model: Model to evaluate.
        device: Torch device.
        data_loader: Evaluation DataLoader.
        num_classes: Number of classes.
        other_class_index: Index of 'other' class to exclude from mIoU.
        log_eval_metrics: Whether to log scalar metrics.
        log_confusion_matrix: Whether to log confusion matrix plot and CSV.
        normalize_confusion_matrix: Normalize confusion matrix rows.
        sample_ids_to_plot: Optional list of sample ids for prediction plots.
        warmup_runs: Warmup forward passes (ignored in timing).
        visualization_labels: Optional dict overriding plot text labels.
        class_name_mapping: Mapping from class index to readable name.

    """
    if class_name_mapping is None:
        class_name_mapping = {i: f"class_{i}" for i in range(num_classes)}

    model.eval()
    model.to(device)
    evaluation_metrics_dict = get_evaluation_metrics_dict(num_classes, device)
    logger.info("Starting evaluation on %d batches", len(data_loader))

    sample_ids_to_log = set(sample_ids_to_plot) if sample_ids_to_plot else set()

    if warmup_runs > 0:
        logger.info("Performing %d warmup runs...", warmup_runs)
        warmup_count = 0
        with torch.no_grad():
            for batch in data_loader:
                if warmup_count >= warmup_runs:
                    break
                inputs, targets, sample_ids = batch
                inputs = inputs.to(device)
                _ = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warmup_count += 1
                del inputs
        logger.info("Warmup runs completed")

    inference_times = []
    batch_sizes = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, targets, sample_ids = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            if torch.cuda.is_available() and device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                outputs = model(inputs)
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                start = time.perf_counter()
                outputs = model(inputs)
                batch_time = time.perf_counter() - start

            inference_times.append(batch_time)
            batch_sizes.append(inputs.shape[0])

            targets = process_segmentation_tensor(targets, num_classes)
            for metric in evaluation_metrics_dict.values():
                metric.update(outputs, targets)

            if sample_ids_to_log:
                intersect = set(sample_ids) & sample_ids_to_log
                if intersect:
                    log_prediction_plots(outputs, sample_ids, num_classes, intersect)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Finished evaluation")

    return _finalize_evaluation(
        evaluation_metrics_dict,
        inference_times,
        batch_sizes,
        class_name_mapping,
        other_class_index,
        normalize_confusion_matrix=normalize_confusion_matrix,
        visualization_labels=visualization_labels,
        log_eval_metrics=log_eval_metrics,
        log_confusion_matrix=log_confusion_matrix,
    )
