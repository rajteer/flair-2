import inspect
import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)
from tqdm import tqdm  # type: ignore[import]

from src.models.utils import process_segmentation_tensor
from src.utils.mlflow_utils import (
    log_confusion_matrix_to_mlflow,
    log_metrics_to_mlflow,
    log_prediction_plots,
)
from src.visualization.mosaic import log_prediction_mosaic_to_mlflow
from src.data.pre_processing.flair_multimodal_dataset import (
    MM_BATCH_AERIAL,
    MM_BATCH_SENTINEL,
    MM_BATCH_MASK,
    MM_BATCH_SAMPLE_IDS,
    MM_BATCH_POSITIONS,
    MM_BATCH_PAD_MASK,
    MM_BATCH_LENGTH,
)

logger = logging.getLogger(__name__)

TEMPORAL_MODEL_NDIM = 5

BATCH_INDEX_INPUTS = 0
BATCH_INDEX_TARGETS = 1
BATCH_INDEX_PAD_MASK = 2
BATCH_INDEX_SAMPLE_IDS = 3
BATCH_INDEX_POSITIONS = 4


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
    other_class_index: int | None = None,
) -> dict[str, Metric]:
    """Initialize TorchMetrics for multiclass classification."""
    return {
        "conf_matrix": MulticlassConfusionMatrix(
            num_classes=num_classes,
            ignore_index=other_class_index,
        ).to(device),
        "macro_f1": MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
            ignore_index=other_class_index,
        ).to(device),
        "f1_per_class": MulticlassF1Score(
            num_classes=num_classes,
            average=None,
            ignore_index=other_class_index,
        ).to(device),
        "overall_f1": MulticlassF1Score(
            num_classes=num_classes,
            average="micro",
            ignore_index=other_class_index,
        ).to(device),
        "macro_accuracy": MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
            ignore_index=other_class_index,
        ).to(
            device,
        ),
        "overall_accuracy": MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
            ignore_index=other_class_index,
        ).to(
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


def _perform_warmup_temporal(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    warmup_runs: int,
) -> None:
    """Run warmup forward passes for temporal models."""
    if warmup_runs <= 0:
        return

    forward_sig = inspect.signature(model.forward)
    supports_pad_mask = "pad_mask" in forward_sig.parameters

    logger.info("Performing %d warmup runs (temporal model)...", warmup_runs)
    with torch.no_grad():
        for warmup_count, batch in enumerate(data_loader):
            if warmup_count >= warmup_runs:
                break
            inputs = batch[BATCH_INDEX_INPUTS].to(device)
            pad_mask = batch[BATCH_INDEX_PAD_MASK].to(device)
            batch_positions = batch[BATCH_INDEX_POSITIONS].to(device)

            if supports_pad_mask:
                _ = model(inputs, batch_positions=batch_positions, pad_mask=pad_mask)
            else:
                _ = model(inputs, batch_positions=batch_positions)

            if device.type == "cuda":
                torch.cuda.synchronize()
    logger.info("Warmup runs completed")


def _perform_warmup_standard(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    warmup_runs: int,
) -> None:
    """Run warmup forward passes for standard models."""
    if warmup_runs <= 0:
        return

    logger.info("Performing %d warmup runs (standard model)...", warmup_runs)
    with torch.no_grad():
        for warmup_count, batch in enumerate(data_loader):
            if warmup_count >= warmup_runs:
                break
            inputs = batch[BATCH_INDEX_INPUTS].to(device)
            _ = model(inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()
    logger.info("Warmup runs completed")


def _forward_with_timing_temporal(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    batch_positions: torch.Tensor,
    pad_mask: torch.Tensor,
    *,
    supports_pad_mask: bool = True,
) -> tuple[torch.Tensor, float]:
    """Run a forward pass for temporal models and measure its duration."""
    if torch.cuda.is_available() and device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if supports_pad_mask:
            outputs = model(inputs, batch_positions=batch_positions, pad_mask=pad_mask)
        else:
            outputs = model(inputs, batch_positions=batch_positions)
        end_event.record()
        torch.cuda.synchronize()
        batch_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        start = time.perf_counter()
        if supports_pad_mask:
            outputs = model(inputs, batch_positions=batch_positions, pad_mask=pad_mask)
        else:
            outputs = model(inputs, batch_positions=batch_positions)
        batch_time = time.perf_counter() - start

    return outputs, batch_time


def _forward_with_timing_standard(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Run a forward pass for standard models and measure its duration."""
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

    return outputs, batch_time


def _evaluate_batches_temporal(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    num_classes: int,
    evaluation_metrics_dict: dict[str, Metric],
    sample_ids_to_log: set[str],
) -> tuple[list[float], list[int]]:
    """Iterate over dataloader for temporal models, update metrics, and collect timing stats."""
    inference_times: list[float] = []
    batch_sizes: list[int] = []

    forward_sig = inspect.signature(model.forward)
    supports_pad_mask = "pad_mask" in forward_sig.parameters

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs = batch[BATCH_INDEX_INPUTS].to(device)
            targets = batch[BATCH_INDEX_TARGETS].to(device)
            pad_mask = batch[BATCH_INDEX_PAD_MASK].to(device)
            batch_positions = batch[BATCH_INDEX_POSITIONS].to(device)

            sample_ids = batch[BATCH_INDEX_SAMPLE_IDS]

            outputs, batch_time = _forward_with_timing_temporal(
                model,
                inputs,
                device,
                batch_positions,
                pad_mask,
                supports_pad_mask=supports_pad_mask,
            )
            inference_times.append(batch_time)
            batch_sizes.append(inputs.shape[0])

            processed_outputs = process_segmentation_tensor(outputs, num_classes)

            for metric in evaluation_metrics_dict.values():
                metric.update(processed_outputs, targets)

            if sample_ids_to_log:
                selected_ids = {
                    sample_id for sample_id in sample_ids if sample_id in sample_ids_to_log
                }
                if selected_ids:
                    log_prediction_plots(outputs, sample_ids, num_classes, selected_ids)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return inference_times, batch_sizes


def _evaluate_batches_standard(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    num_classes: int,
    evaluation_metrics_dict: dict[str, Metric],
    sample_ids_to_log: set[str],
) -> tuple[list[float], list[int]]:
    """Iterate over dataloader for standard models, update metrics, and collect timing stats."""
    inference_times: list[float] = []
    batch_sizes: list[int] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs = batch[BATCH_INDEX_INPUTS].to(device)
            targets = batch[BATCH_INDEX_TARGETS].to(device)

            sample_ids = batch[BATCH_INDEX_SAMPLE_IDS]

            outputs, batch_time = _forward_with_timing_standard(model, inputs, device)
            inference_times.append(batch_time)
            batch_sizes.append(inputs.shape[0])

            processed_outputs = process_segmentation_tensor(outputs, num_classes)

            for metric in evaluation_metrics_dict.values():
                metric.update(processed_outputs, targets)

            if sample_ids_to_log:
                selected_ids = {
                    sample_id for sample_id in sample_ids if sample_id in sample_ids_to_log
                }
                if selected_ids:
                    log_prediction_plots(outputs, sample_ids, num_classes, selected_ids)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return inference_times, batch_sizes


def _perform_warmup_multimodal(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    warmup_runs: int,
) -> None:
    """Run warmup forward passes for multimodal models."""
    if warmup_runs <= 0:
        return

    logger.info("Performing %d warmup runs (multimodal model)...", warmup_runs)
    with torch.no_grad():
        for warmup_count, batch in enumerate(data_loader):
            if warmup_count >= warmup_runs:
                break
            aerial_input = batch[MM_BATCH_AERIAL].to(device)
            sentinel_input = batch[MM_BATCH_SENTINEL].to(device)
            batch_positions = batch[MM_BATCH_POSITIONS].to(device)
            pad_mask = batch[MM_BATCH_PAD_MASK].to(device)

            _ = model(
                aerial_input,
                sentinel_input,
                batch_positions=batch_positions,
                pad_mask=pad_mask,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
    logger.info("Warmup runs completed")


def _forward_with_timing_multimodal(
    model: nn.Module,
    aerial_input: torch.Tensor,
    sentinel_input: torch.Tensor,
    device: torch.device,
    batch_positions: torch.Tensor,
    pad_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Run a forward pass for multimodal models and measure its duration."""
    if torch.cuda.is_available() and device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        outputs = model(
            aerial_input,
            sentinel_input,
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )
        end_event.record()
        torch.cuda.synchronize()
        batch_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        start = time.perf_counter()
        outputs = model(
            aerial_input,
            sentinel_input,
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )
        batch_time = time.perf_counter() - start

    return outputs, batch_time


def _evaluate_batches_multimodal(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    num_classes: int,
    evaluation_metrics_dict: dict[str, Metric],
    sample_ids_to_log: set[str],
) -> tuple[list[float], list[int]]:
    """Iterate over dataloader for multimodal models, update metrics, collect timing."""
    inference_times: list[float] = []
    batch_sizes: list[int] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            aerial_input = batch[MM_BATCH_AERIAL].to(device)
            sentinel_input = batch[MM_BATCH_SENTINEL].to(device)
            targets = batch[MM_BATCH_MASK].to(device)
            batch_positions = batch[MM_BATCH_POSITIONS].to(device)
            pad_mask = batch[MM_BATCH_PAD_MASK].to(device)

            sample_ids = batch[MM_BATCH_SAMPLE_IDS]

            outputs, batch_time = _forward_with_timing_multimodal(
                model,
                aerial_input,
                sentinel_input,
                device,
                batch_positions,
                pad_mask,
            )
            inference_times.append(batch_time)
            batch_sizes.append(aerial_input.shape[0])

            processed_outputs = process_segmentation_tensor(outputs, num_classes)

            for metric in evaluation_metrics_dict.values():
                metric.update(processed_outputs, targets)

            if sample_ids_to_log:
                selected_ids = {
                    sample_id for sample_id in sample_ids if sample_id in sample_ids_to_log
                }
                if selected_ids:
                    log_prediction_plots(outputs, sample_ids, num_classes, selected_ids)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return inference_times, batch_sizes


def _finalize_evaluation(
    evaluation_metrics_dict: dict[str, Metric],
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

    # Compute per-class F1 scores
    f1_per_class_tensor = evaluation_metrics_dict["f1_per_class"].compute()
    per_class_f1 = {
        i: f1_per_class_tensor[i].item()
        for i in range(len(class_name_mapping))
        if i != confusion_other_index
    }

    metrics: dict[str, float] = {
        "miou": miou,
        "macro_f1": evaluation_metrics_dict["macro_f1"].compute().item(),
        "overall_f1": evaluation_metrics_dict["overall_f1"].compute().item(),
        "macro_accuracy": evaluation_metrics_dict["macro_accuracy"].compute().item(),
        "overall_accuracy": evaluation_metrics_dict["overall_accuracy"].compute().item(),
        "total_batches_processed": len(inference_times),
    }
    metrics.update(timing_metrics)

    per_class_metrics = {
        f"iou_class_{class_name_mapping.get(i, f'class_{i}')}": v for i, v in per_class_iou.items()
    }
    metrics.update(per_class_metrics)

    # Add per-class F1 scores to metrics
    per_class_f1_metrics = {
        f"f1_class_{class_name_mapping.get(i, f'class_{i}')}": v for i, v in per_class_f1.items()
    }
    metrics.update(per_class_f1_metrics)

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

        log_confusion_matrix_to_mlflow(
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
    zone_mosaic_config: dict | None = None,
    zone_data_loader: DataLoader | None = None,
    is_multimodal: bool = False,
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
        sample_ids_to_plot: Optional list of sample ids for individual prediction plots.
        warmup_runs: Warmup forward passes (ignored in timing).
        visualization_labels: Optional dict overriding plot text labels.
        class_name_mapping: Mapping from class index to readable name.
        zone_mosaic_config: Optional config for zone prediction mosaic visualization.
            Expected keys: 'enabled', 'zone_name', 'grid_size', 'patch_size'.
        zone_data_loader: Optional DataLoader for a specific zone (for mosaic visualization).
        is_multimodal: Whether the model is multimodal (aerial + sentinel inputs).

    """
    if class_name_mapping is None:
        class_name_mapping = {i: f"class_{i}" for i in range(num_classes)}

    model.eval()
    model.to(device)
    evaluation_metrics_dict = get_evaluation_metrics_dict(
        num_classes,
        device,
        other_class_index=other_class_index,
    )
    logger.info("Starting evaluation on %d batches", len(data_loader))

    sample_ids_to_log = set(sample_ids_to_plot) if sample_ids_to_plot else set()

    first_batch = next(iter(data_loader))

    # Detect model type - multimodal can be explicitly set or detected from batch structure
    # Multimodal batches from multimodal_collate_fn have MM_BATCH_LENGTH elements
    is_multimodal_model = is_multimodal or (
        isinstance(first_batch, tuple) and len(first_batch) == MM_BATCH_LENGTH
    )

    if is_multimodal_model:
        is_temporal_model = False
    else:
        first_inputs = first_batch[BATCH_INDEX_INPUTS]
        is_temporal_model = first_inputs.ndim == TEMPORAL_MODEL_NDIM

    if is_multimodal_model:
        logger.info("Detected multimodal model. Using multimodal evaluation.")
        _perform_warmup_multimodal(model, device, data_loader, warmup_runs)
        inference_times, batch_sizes = _evaluate_batches_multimodal(
            model,
            device,
            data_loader,
            num_classes,
            evaluation_metrics_dict,
            sample_ids_to_log,
        )
    elif is_temporal_model:
        logger.info("Detected temporal model (5D input). Using temporal evaluation.")
        _perform_warmup_temporal(model, device, data_loader, warmup_runs)
        inference_times, batch_sizes = _evaluate_batches_temporal(
            model,
            device,
            data_loader,
            num_classes,
            evaluation_metrics_dict,
            sample_ids_to_log,
        )
    else:
        logger.info("Detected standard model. Using standard evaluation.")
        _perform_warmup_standard(model, device, data_loader, warmup_runs)
        inference_times, batch_sizes = _evaluate_batches_standard(
            model,
            device,
            data_loader,
            num_classes,
            evaluation_metrics_dict,
            sample_ids_to_log,
        )

    # Create and log zone prediction mosaic if configured
    if zone_mosaic_config and zone_mosaic_config.get("enabled", False):
        if zone_data_loader is None:
            logger.warning(
                "Zone mosaic enabled but no zone_data_loader provided. Skipping mosaic.",
            )
        else:
            zone_name = zone_mosaic_config.get("zone_name", "zone")
            zone_grid_size = zone_mosaic_config.get("grid_size", 10)
            zone_patch_size = zone_mosaic_config.get("patch_size", 512)

            log_prediction_mosaic_to_mlflow(
                model=model,
                data_loader=zone_data_loader,
                device=device,
                num_classes=num_classes,
                zone_name=zone_name,
                grid_size=zone_grid_size,
                patch_size=zone_patch_size,
            )

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
