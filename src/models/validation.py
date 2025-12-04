import inspect
import logging
import time

import numpy as np
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
    log_mosaic_plot,
    log_prediction_plots,
)

logger = logging.getLogger(__name__)

TEMPORAL_MODEL_NDIM = 5

BATCH_INDEX_INPUTS = 0
BATCH_INDEX_TARGETS = 1
BATCH_INDEX_PAD_MASK = 2
BATCH_INDEX_SAMPLE_IDS = 3
BATCH_INDEX_POSITIONS = 4


def infer_grid_from_id_list(
    sample_ids: list[str],
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """Infer grid dimensions and positions from the order of sample IDs in the list.

    This function assumes that the sample_ids list is ordered in a consistent pattern
    (e.g., row-major or column-major order) and tries to infer the grid shape.

    The function looks for a rectangular grid where:
    - IDs are grouped by common prefixes (suggesting they belong to the same row/column)
    - The total count matches rows x cols

    If no clear pattern is found, it attempts to create a roughly square grid.

    Args:
        sample_ids: List of sample ID strings in their intended spatial order.

    Returns:
        Tuple of (num_rows, num_cols, position_map) where position_map
        maps sample_id -> (row, col) based on list order.

    """
    n = len(sample_ids)
    if n == 0:
        return 0, 0, {}

    # Try to detect grid dimensions by looking at ID patterns
    # Group IDs by their prefix (all but last 1-2 digits)
    prefix_groups: dict[str, list[str]] = {}
    for sid in sample_ids:
        # Try different prefix lengths to find grouping
        prefix = sid[:-1] if len(sid) > 1 else sid
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(sid)

    # Check if we have consistent group sizes (suggests column count)
    group_sizes = [len(g) for g in prefix_groups.values()]
    if group_sizes and all(s == group_sizes[0] for s in group_sizes):
        # Consistent grouping found
        num_cols = group_sizes[0]
        num_rows = len(prefix_groups)
    else:
        # Fall back to square-ish grid
        num_cols = int(np.ceil(np.sqrt(n)))
        num_rows = int(np.ceil(n / num_cols))

    # Create position map based on list order (row-major)
    positions = {}
    for idx, sid in enumerate(sample_ids):
        row = idx // num_cols
        col = idx % num_cols
        positions[sid] = (row, col)

    logger.info(
        "Inferred grid dimensions: %d rows x %d cols from %d sample IDs",
        num_rows,
        num_cols,
        n,
    )

    return num_rows, num_cols, positions


def stitch_patches_to_mosaic(
    patches: dict[str, np.ndarray],
    sample_ids: list[str],
    patch_size: int = 512,
    grid_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Stitch multiple patches into a single mosaic image.

    Args:
        patches: Dictionary mapping sample_id to patch array (H, W) or (C, H, W).
        sample_ids: List of sample IDs that form the mosaic, ordered spatially
            (row-major order: left-to-right, top-to-bottom).
        patch_size: Size of each patch (assumed square).
        grid_shape: Optional explicit (num_rows, num_cols) shape. If not provided,
            the grid shape is inferred from the sample_ids list.

    Returns:
        Stitched mosaic array, or None if stitching fails.

    """
    if grid_shape is not None:
        num_rows, num_cols = grid_shape
        # Create position map based on list order (row-major)
        positions = {
            sid: (idx // num_cols, idx % num_cols) for idx, sid in enumerate(sample_ids)
        }
    else:
        num_rows, num_cols, positions = infer_grid_from_id_list(sample_ids)

    if num_rows == 0 or num_cols == 0:
        logger.warning("Could not determine grid dimensions for mosaic")
        return None

    # Check if we have all patches
    available_ids = set(patches.keys()) & set(sample_ids)
    if len(available_ids) < len(sample_ids):
        logger.warning(
            "Missing %d patches for mosaic. Available: %d, Expected: %d",
            len(sample_ids) - len(available_ids),
            len(available_ids),
            len(sample_ids),
        )

    # Determine if patches are 2D (H, W) or have channels
    sample_patch = next(iter(patches.values()))
    if sample_patch.ndim == 2:  # noqa: PLR2004
        mosaic = np.zeros((num_rows * patch_size, num_cols * patch_size), dtype=sample_patch.dtype)
    else:
        # Assume (C, H, W) format
        num_channels = sample_patch.shape[0]
        mosaic = np.zeros(
            (num_channels, num_rows * patch_size, num_cols * patch_size),
            dtype=sample_patch.dtype,
        )

    for sample_id in sample_ids:
        if sample_id not in patches:
            continue
        if sample_id not in positions:
            continue

        row, col = positions[sample_id]
        patch = patches[sample_id]

        y_start = row * patch_size
        y_end = y_start + patch_size
        x_start = col * patch_size
        x_end = x_start + patch_size

        if patch.ndim == 2:  # noqa: PLR2004
            mosaic[y_start:y_end, x_start:x_end] = patch
        else:
            mosaic[:, y_start:y_end, x_start:x_end] = patch

    logger.info(
        "Created mosaic of size %s from %d patches (%d rows x %d cols)",
        mosaic.shape,
        len(available_ids),
        num_rows,
        num_cols,
    )

    return mosaic


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
) -> dict[str, Metric]:
    """Initialize TorchMetrics for multiclass classification."""
    return {
        "conf_matrix": MulticlassConfusionMatrix(num_classes=num_classes).to(device),
        "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "f1_per_class": MulticlassF1Score(num_classes=num_classes, average=None).to(device),
        "overall_f1": MulticlassF1Score(num_classes=num_classes, average="micro").to(device),
        "macro_accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro").to(
            device,
        ),
        "overall_accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro").to(
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
) -> tuple[list[float], list[int], dict[str, np.ndarray]]:
    """Iterate over dataloader for temporal models, update metrics, and collect timing stats."""
    inference_times: list[float] = []
    batch_sizes: list[int] = []
    collected_patches: dict[str, np.ndarray] = {}

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

                    # Collect patches for mosaic
                    processed_np = processed_outputs.cpu().numpy()
                    for idx, sample_id in enumerate(sample_ids):
                        if sample_id in sample_ids_to_log:
                            collected_patches[sample_id] = processed_np[idx]

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return inference_times, batch_sizes, collected_patches


def _evaluate_batches_standard(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    num_classes: int,
    evaluation_metrics_dict: dict[str, Metric],
    sample_ids_to_log: set[str],
) -> tuple[list[float], list[int], dict[str, np.ndarray]]:
    """Iterate over dataloader for standard models, update metrics, and collect timing stats."""
    inference_times: list[float] = []
    batch_sizes: list[int] = []
    collected_patches: dict[str, np.ndarray] = {}

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

                    # Collect patches for mosaic
                    processed_np = processed_outputs.cpu().numpy()
                    for idx, sample_id in enumerate(sample_ids):
                        if sample_id in sample_ids_to_log:
                            collected_patches[sample_id] = processed_np[idx]

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return inference_times, batch_sizes, collected_patches


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
    log_mosaic: bool = True,
    patch_size: int = 512,
    mosaic_grid_shape: tuple[int, int] | None = None,
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
            The order should be row-major (left-to-right, top-to-bottom) for mosaic.
        warmup_runs: Warmup forward passes (ignored in timing).
        visualization_labels: Optional dict overriding plot text labels.
        class_name_mapping: Mapping from class index to readable name.
        log_mosaic: Whether to stitch patches and log mosaic image.
        patch_size: Size of each patch (assumed square, default 512).
        mosaic_grid_shape: Optional explicit (rows, cols) for the mosaic grid.
            If not provided, the shape is inferred from sample_ids_to_plot.

    """
    if class_name_mapping is None:
        class_name_mapping = {i: f"class_{i}" for i in range(num_classes)}

    model.eval()
    model.to(device)
    evaluation_metrics_dict = get_evaluation_metrics_dict(num_classes, device)
    logger.info("Starting evaluation on %d batches", len(data_loader))

    sample_ids_to_log = set(sample_ids_to_plot) if sample_ids_to_plot else set()

    first_batch = next(iter(data_loader))
    is_temporal_model = first_batch[BATCH_INDEX_INPUTS].ndim == TEMPORAL_MODEL_NDIM

    collected_patches: dict[str, np.ndarray] = {}

    if is_temporal_model:
        logger.info("Detected temporal model (5D input). Using temporal evaluation.")
        _perform_warmup_temporal(model, device, data_loader, warmup_runs)
        inference_times, batch_sizes, collected_patches = _evaluate_batches_temporal(
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
        inference_times, batch_sizes, collected_patches = _evaluate_batches_standard(
            model,
            device,
            data_loader,
            num_classes,
            evaluation_metrics_dict,
            sample_ids_to_log,
        )

    # Create and log mosaic if patches were collected
    if log_mosaic and collected_patches and sample_ids_to_plot:
        logger.info("Creating mosaic from %d collected patches", len(collected_patches))
        mosaic = stitch_patches_to_mosaic(
            collected_patches,
            sample_ids_to_plot,
            patch_size=patch_size,
            grid_shape=mosaic_grid_shape,
        )
        if mosaic is not None:
            log_mosaic_plot(mosaic, name="prediction_mosaic")

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
