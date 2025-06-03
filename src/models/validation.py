import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)

from src.models.utils import process_segmentation_tensor
from src.utils.mlflow_utils import (
    log_metrics_to_mlflow,
    log_comparison_to_mlflow,
    log_confusion_to_mlflow,
)
from src.visualization.utils import msk_to_name

logger = logging.getLogger(__name__)


def calculate_iou_scores(
        conf_matrix: torch.Tensor, num_classes: int, other_class_index: int = 13
) -> tuple[float, dict[int, float]]:
    """
    Computes mean Intersection over Union (mIoU) and per-class IoU for a confusion matrix.

    Args:
        conf_matrix (torch.Tensor): Confusion matrix of shape (C, C).
        num_classes (int): Total number of classes.
        other_class_index (int): Index of class to exclude from mIoU.

    Returns:
        tuple: (mean_iou, per_class_iou_dict)
    """
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
        num_classes, dtype=torch.bool, device=conf_matrix.device
    )
    if 0 <= other_class_index < num_classes:
        valid_classes_mask[other_class_index] = False

    per_class_iou = {
        i: iou[i].item() for i in range(num_classes) if i != other_class_index
    }

    mean_iou = (
        iou[valid_classes_mask].mean().item() if valid_classes_mask.any() else 0.0
    )

    return mean_iou, per_class_iou


def get_metrics(num_classes: int, device: torch.device) -> dict[str, torch.nn.Module]:
    """
    Initializes dictionary of TorchMetrics metrics for multiclass classification.

    Args:
        num_classes (int): Number of classes.
        device (torch.device): Torch device.

    Returns:
        dict[str, nn.Module]: Dictionary of metric modules.
    """
    return {
        "conf_matrix": MulticlassConfusionMatrix(num_classes=num_classes).to(device),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro").to(
            device
        ),
    }


def log_evaluation_results(
        metrics: dict[str, float],
        confusion_matrix: torch.Tensor,
        log_confusion_matrix: bool = True,
):
    """
    Logs evaluation results to MLflow.

    Args:
        metrics: Aggregated metrics to log.
        confusion_matrix: Final confusion matrix.
        log_confusion_matrix: Whether to log confusion matrix.
    """
    log_metrics_to_mlflow(metrics, prefix="agg_")
    if log_confusion_matrix:
        log_confusion_to_mlflow(confusion_matrix, list(msk_to_name.values()))


def evaluate(
        model: nn.Module,
        device: torch.device,
        data_loader: DataLoader,
        num_classes: int,
        other_class_index: int = 13,
        log_confusion_matrix: bool = True,
        log_comparison: bool = True,
        ids_to_plot: Optional[list[str]] = None,
        eval_per_sample: bool = False,
        log_to_mlflow: bool = True,
) -> dict[str, float]:
    """
    Evaluates a semantic segmentation model and logs aggregated performance metrics to MLflow.
    """
    model.eval()
    model.to(device)
    metrics_dict = get_metrics(num_classes, device)
    logger.info(f"Starting evaluation on {len(data_loader)} batches")

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets, ids = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            preds = process_segmentation_tensor(outputs, num_classes)
            targets = process_segmentation_tensor(targets, num_classes)

            for metric in metrics_dict.values():
                metric.update(outputs, targets)

            if eval_per_sample:
                sample_cm_metric = MulticlassConfusionMatrix(
                    num_classes=num_classes
                ).to(device)
                for i, (pred, target, id_) in enumerate(zip(preds, targets, ids)):
                    sample_cm_metric.reset()
                    sample_cm_metric.update(pred.unsqueeze(0), target.unsqueeze(0))
                    sample_conf_matrix = sample_cm_metric.compute()
                    sample_miou, _ = calculate_iou_scores(
                        sample_conf_matrix, num_classes, other_class_index
                    )

                    logger.info(
                        "[Sample %s] mIoU: %.4f", id_, sample_miou,
                    )

                    should_log_sample = (
                            log_to_mlflow
                            and log_comparison
                            and (ids_to_plot is None or id_ in ids_to_plot)
                    )
                    if should_log_sample:
                        log_comparison_to_mlflow(id_, target, pred)

            del inputs, outputs, targets, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Finished evaluation")

    final_conf_matrix = metrics_dict["conf_matrix"].compute()
    miou, per_class_iou = calculate_iou_scores(
        final_conf_matrix, num_classes, other_class_index
    )

    metrics = {
        "miou": miou,
        "f1": metrics_dict["f1"].compute().item(),
        "accuracy": metrics_dict["accuracy"].compute().item(),
    }

    for class_idx, iou_value in per_class_iou.items():
        class_name = msk_to_name.get(class_idx, f"class_{class_idx}")
        metrics[f"iou_class_{class_name}"] = iou_value

    if log_to_mlflow:
        log_metrics_to_mlflow(metrics, prefix="agg_")

    logger.info("AGGREGATED METRICS")
    logger.info(
        f"mIoU: {metrics['miou']:.4f} | F1: {metrics['f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}"
    )

    logger.info("PER-CLASS IoU")
    per_class_log_msg = " | ".join(
        [
            f"{msk_to_name.get(idx, f'class_{idx}')}: {iou_val:.4f}"
            for idx, iou_val in per_class_iou.items()
        ]
    )
    logger.info(per_class_log_msg)

    if log_to_mlflow and log_confusion_matrix:
        log_confusion_to_mlflow(conf_matrix=final_conf_matrix,
                                class_names=list(msk_to_name.values()),
                                other_class_index=other_class_index,
                                normalize=True)

    for metric in metrics_dict.values():
        metric.reset()

    return metrics
