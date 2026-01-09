import inspect
import logging
import time

import torch
import torch.nn.functional as F
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

logger = logging.getLogger(__name__)

TEMPORAL_MODEL_NDIM = 5

BATCH_INDEX_INPUTS = 0
BATCH_INDEX_TARGETS = 1
BATCH_INDEX_PAD_MASK = 2
BATCH_INDEX_SAMPLE_IDS = 3
BATCH_INDEX_POSITIONS = 4


def upsample_predictions(
    outputs: torch.Tensor,
    target_size: tuple[int, int],
    output_size: int | None = None,
) -> torch.Tensor:
    """Upsample model predictions to match the target mask size.

    Uses bilinear interpolation on logits (before argmax) for smoother boundaries.
    When using context window, center-crops to output_size before upsampling.

    Args:
        outputs: Model outputs with shape (B, C, H, W) where C is num_classes.
        target_size: Tuple of (height, width) to upsample to (mask size).
        output_size: Expected output spatial size after center-crop. If provided and
            model output is larger, center-crops to this size before upsampling.
            Use sentinel_patch_size when using context window.

    Returns:
        Upsampled logits with shape (B, C, target_size[0], target_size[1]).

    """
    if outputs.shape[-2:] == target_size:
        return outputs

    out_h, out_w = outputs.shape[-2:]

    # Center-crop if using context window (output_size specified and output is larger)
    if output_size is not None and out_h > output_size:
        crop_margin_h = (out_h - output_size) // 2
        crop_margin_w = (out_w - output_size) // 2
        outputs = outputs[
            :,
            :,
            crop_margin_h : out_h - crop_margin_h,
            crop_margin_w : out_w - crop_margin_w,
        ]

    upsampled = F.interpolate(
        outputs,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    return upsampled


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
    output_size: int | None = None,
) -> tuple[list[float], list[int]]:
    """Iterate over dataloader for temporal models, update metrics, and collect timing stats.

    Args:
        output_size: Expected output spatial size for center-cropping when using context window.
            Pass sentinel_patch_size when context_size > sentinel_patch_size.

    """
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

            target_size = (targets.shape[-2], targets.shape[-1])
            upsampled_outputs = upsample_predictions(
                outputs,
                target_size,
                output_size=output_size,
            )

            processed_preds = process_segmentation_tensor(upsampled_outputs, num_classes)

            for metric in evaluation_metrics_dict.values():
                metric.update(processed_preds, targets)

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

            # Upsample raw logits first (before argmax) for smoother boundaries
            target_size = (targets.shape[-2], targets.shape[-1])
            upsampled_outputs = upsample_predictions(outputs, target_size)

            processed_preds = process_segmentation_tensor(upsampled_outputs, num_classes)

            for metric in evaluation_metrics_dict.values():
                metric.update(processed_preds, targets)

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
    output_size: int | None = None,
    log_eval_metrics: bool = True,
    log_confusion_matrix: bool = True,
    normalize_confusion_matrix: bool = True,
    sample_ids_to_plot: list[str] | None = None,
    warmup_runs: int = 10,
    visualization_labels: dict[str, str] | None = None,
    class_name_mapping: dict[int, str] | None = None,
    zone_mosaic_config: dict | None = None,
    zone_data_loader: DataLoader | None = None,
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
    is_temporal_model = first_batch[BATCH_INDEX_INPUTS].ndim == TEMPORAL_MODEL_NDIM

    if is_temporal_model:
        logger.info("Detected temporal model (5D input). Using temporal evaluation.")
        _perform_warmup_temporal(model, device, data_loader, warmup_runs)
        inference_times, batch_sizes = _evaluate_batches_temporal(
            model,
            device,
            data_loader,
            num_classes,
            evaluation_metrics_dict,
            sample_ids_to_log,
            output_size=output_size,
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
