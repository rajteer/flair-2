"""Training and evaluation pipeline for multimodal fusion experiments.

This pipeline handles training with both aerial and Sentinel-2 data for
the MultimodalLateFusion model architecture.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from src.data.pre_processing.flair_multimodal_dataset import (
    FlairMultimodalDataset,
    multimodal_collate_fn,
)
from src.data.pre_processing.flair_dataset import MultiChannelNormalize
from src.models.model_builder import (
    build_loss_function,
    build_lr_scheduler,
    build_model,
    build_optimizer,
)
from src.models.validation import evaluate
from src.utils.logging_utils import LOG_FORMATTER, setup_logging
from src.utils.mlflow_utils import init_mlflow, log_metrics_to_mlflow, log_model_to_mlflow
from src.utils.read_yaml import read_yaml
from src.utils.reproducibility import create_generator, seed_everything, seed_worker

logger = logging.getLogger(__name__)

try:
    GradScaler = torch.amp.GradScaler
    autocast = torch.amp.autocast
    _USE_NEW_AMP_API = True
except (AttributeError, ImportError):
    from torch.cuda.amp import GradScaler, autocast

    _USE_NEW_AMP_API = False


# Batch indices for multimodal collate output
BATCH_AERIAL = 0
BATCH_SENTINEL = 1
BATCH_MASK = 2
BATCH_SAMPLE_IDS = 3
BATCH_POSITIONS = 4
BATCH_PAD_MASK = 5


def _train_epoch_multimodal(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    scheduler: LRScheduler | None = None,
    max_grad_norm: float | None = None,
) -> float:
    """Train multimodal model for one epoch.

    Args:
        model: MultimodalLateFusion model.
        loader: DataLoader yielding multimodal batches.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Training device.
        accumulation_steps: Gradient accumulation steps.
        use_amp: Whether to use automatic mixed precision.
        scheduler: Optional step-level scheduler.
        max_grad_norm: Optional gradient clipping.

    Returns:
        Average training loss for the epoch.

    """
    model.train()
    optimizer.zero_grad()

    scaler_enabled = use_amp and device.type == "cuda"
    if _USE_NEW_AMP_API:
        scaler = GradScaler(device=device.type, enabled=scaler_enabled)
    else:
        scaler = GradScaler(enabled=scaler_enabled)

    total_loss = 0.0

    for batch_idx, batch_data in enumerate(loader):
        aerial = batch_data[BATCH_AERIAL].to(device)
        sentinel = batch_data[BATCH_SENTINEL].to(device)
        masks = batch_data[BATCH_MASK].to(device)
        positions = batch_data[BATCH_POSITIONS].to(device)
        pad_mask = batch_data[BATCH_PAD_MASK].to(device)

        optimizer.zero_grad()

        if _USE_NEW_AMP_API:
            ctx = autocast(device_type=device.type, enabled=use_amp)
        else:
            ctx = autocast(enabled=use_amp)

        with ctx:
            outputs = model(
                aerial,
                sentinel,
                batch_positions=positions,
                pad_mask=pad_mask,
            )
            loss = criterion(outputs, masks)
            loss_value = loss.item()
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss_value)

    return total_loss / len(loader)


def _validate_epoch_multimodal(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int | None = None,
) -> tuple[float, float]:
    """Validate multimodal model for one epoch.

    Args:
        model: MultimodalLateFusion model.
        loader: DataLoader yielding multimodal batches.
        criterion: Loss function.
        device: Validation device.
        num_classes: Number of segmentation classes.
        ignore_index: Class index to ignore in mIoU calculation.

    Returns:
        Tuple of (average loss, mIoU).

    """
    model.eval()
    val_loss = 0.0

    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
    ).to(device)

    with torch.no_grad():
        for batch_data in loader:
            aerial = batch_data[BATCH_AERIAL].to(device)
            sentinel = batch_data[BATCH_SENTINEL].to(device)
            masks = batch_data[BATCH_MASK].to(device)
            positions = batch_data[BATCH_POSITIONS].to(device)
            pad_mask = batch_data[BATCH_PAD_MASK].to(device)

            outputs = model(
                aerial,
                sentinel,
                batch_positions=positions,
                pad_mask=pad_mask,
            )
            loss = criterion(outputs, masks)
            val_loss += float(loss.item())

            preds = outputs.argmax(dim=1)
            miou_metric.update(preds, masks)

    return val_loss / len(loader), miou_metric.compute().item()


def train_multimodal(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: LRScheduler | None = None,
    epochs: int = 100,
    patience: int = 20,
    num_classes: int = 13,
    other_class_index: int | None = None,
    accumulation_steps: int = 1,
    early_stopping_criterion: str = "miou",
    *,
    use_amp: bool = False,
    log_evaluation_metrics: bool = True,
    log_model: bool = True,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Train a multimodal fusion model.

    Args:
        model: MultimodalLateFusion model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Training device.
        scheduler: Optional learning rate scheduler.
        epochs: Maximum epochs.
        patience: Early stopping patience.
        num_classes: Number of classes.
        other_class_index: Index of "other" class to ignore.
        accumulation_steps: Gradient accumulation steps.
        early_stopping_criterion: 'loss' or 'miou'.
        use_amp: Whether to use AMP.
        log_evaluation_metrics: Whether to log to MLflow.
        log_model: Whether to log model to MLflow.
        max_grad_norm: Optional gradient clipping.

    Returns:
        Dictionary with training history and best metrics.

    """
    if early_stopping_criterion not in ("loss", "miou"):
        msg = f"early_stopping_criterion must be 'loss' or 'miou', got {early_stopping_criterion!r}"
        raise ValueError(msg)

    model.to(device)

    best_val_loss = float("inf")
    best_val_miou = 0.0
    no_improve = 0
    losses_train: list[float] = []
    losses_val: list[float] = []
    mious_val: list[float] = []
    best_model_state = None

    is_step_scheduler = scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau)
    step_scheduler = scheduler if is_step_scheduler else None

    for epoch in range(epochs):
        loss_epoch = _train_epoch_multimodal(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            scheduler=step_scheduler,
            max_grad_norm=max_grad_norm,
        )
        losses_train.append(loss_epoch)
        logger.info("Epoch %d/%d: Training Loss: %.4f", epoch + 1, epochs, loss_epoch)

        val_loss, val_miou = _validate_epoch_multimodal(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=other_class_index,
        )
        losses_val.append(val_loss)
        mious_val.append(val_miou)
        logger.info(
            "Epoch %d/%d: Validation Loss: %.4f, Validation mIoU: %.4f",
            epoch + 1,
            epochs,
            val_loss,
            val_miou,
        )

        if log_evaluation_metrics:
            log_metrics_to_mlflow(
                metrics={"train_loss": loss_epoch, "val_loss": val_loss, "val_miou": val_miou},
                step=epoch,
            )

        if early_stopping_criterion == "miou":
            improved = val_miou > best_val_miou
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_miou = val_miou
            best_val_loss = val_loss
            no_improve = 0
            if early_stopping_criterion == "miou":
                logger.info("Validation mIoU improved to %.4f", best_val_miou)
            else:
                logger.info("Validation loss improved to %.4f", best_val_loss)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            logger.info("No improvement for %d epochs.", no_improve)
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        if log_evaluation_metrics:
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

        # Log fusion weights if available
        if hasattr(model, "get_fusion_weights"):
            weights = model.get_fusion_weights()
            if weights:
                logger.info(
                    "Learned fusion weights (aerial per class): %s", weights.get("aerial_weights")
                )
                logger.info(
                    "Learned fusion weights (sentinel per class): %s",
                    weights.get("sentinel_weights"),
                )

    return {
        "train_loss": losses_train,
        "val_loss": losses_val,
        "val_miou": mious_val,
        "best_val_loss": best_val_loss,
        "best_val_miou": best_val_miou,
    }


class MultimodalTrainEvalPipeline:
    """Pipeline for multimodal fusion training and evaluation."""

    def __init__(self, run_name: str | None = None, logs_dir: str | None = None) -> None:
        """Initialize the multimodal pipeline."""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("_") if run_name else ""
        run_suffix = f"_{safe_name}" if safe_name else ""
        logs_path = Path(logs_dir).expanduser().resolve() if logs_dir else Path.cwd()
        self.log_file = logs_path / f"multimodal_pipeline_{timestamp}{run_suffix}.log"

    def run(self, config: dict[str, Any], *, no_stdout_logs: bool = False) -> None:
        """Execute the multimodal training and evaluation pipeline."""
        mlflow_cfg = config["mlflow"]

        init_mlflow(
            tracking_uri=mlflow_cfg.get("tracking_uri"),
            experiment_name=mlflow_cfg["name"],
            dagshub_config=mlflow_cfg.get("dagshub"),
        )

        setup_logging(
            log_file=self.log_file,
            log_formatter=LOG_FORMATTER,
            no_stdout_logs=no_stdout_logs,
        )

        exp_cfg = config.get("experiment", {})
        seed = int(exp_cfg.get("seed", 42))
        deterministic = bool(exp_cfg.get("deterministic", True))
        seed_everything(seed=seed, deterministic=deterministic)

        logger.info("Starting multimodal fusion train-evaluation pipeline.")

        with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
            mlflow.log_dict(config, artifact_file="config_resolved.json")

            mlflow.log_params(
                {
                    "seed": seed,
                    "deterministic": deterministic,
                    "model_type": config["model"]["model_type"],
                    "fusion_mode": config["model"].get("fusion_mode", "weighted"),
                    "freeze_encoders": config["model"].get("freeze_encoders", True),
                    "use_cloud_uncertainty": config["model"].get("use_cloud_uncertainty", False),
                }
            )

            mlflow.log_params(
                {
                    "optimizer": config["training"]["optimizer"]["type"],
                    "learning_rate": config["training"]["optimizer"]["learning_rate"],
                    "loss_function": config["training"]["loss_function"]["type"],
                    "epochs": config["training"]["epochs"],
                    "batch_size": config["data"]["batch_size"],
                }
            )

            if note := config["mlflow"].get("note"):
                mlflow.set_tag("note", note)
            mlflow.set_tag("data_type", "multimodal_fusion")

            # Create datasets
            data_cfg = config["data"]

            norm_cfg = data_cfg.get("normalization")
            image_transform = None
            if norm_cfg is not None and norm_cfg.get("enabled", True):
                image_transform = MultiChannelNormalize(
                    mean=norm_cfg["mean"],
                    std=norm_cfg["std"],
                    scale_to_unit=norm_cfg.get("scale_to_unit"),
                    elevation_range=tuple(norm_cfg["elevation_range"])
                    if norm_cfg.get("elevation_range") is not None
                    else None,
                    elevation_channel_index=norm_cfg.get("elevation_channel_index"),
                )

            selected_channels = data_cfg.get("selected_channels")

            train_dataset = FlairMultimodalDataset(
                image_dir=data_cfg["train"]["images"],
                mask_dir=data_cfg["train"]["masks"],
                sentinel_dir=data_cfg["train"]["sentinel"],
                centroids_path=data_cfg["centroids_path"],
                num_classes=data_cfg["num_classes"],
                image_transform=image_transform,
                selected_channels=selected_channels,
                sentinel_patch_size=data_cfg.get("sentinel_patch_size", 10),
                context_size=data_cfg.get("context_size"),
                use_monthly_average=data_cfg.get("use_monthly_average", True),
                cloud_snow_cover_threshold=data_cfg.get("cloud_snow_cover_threshold", 0.6),
                cloud_snow_prob_threshold=data_cfg.get("cloud_snow_prob_threshold", 50),
                sentinel_scale_factor=data_cfg.get("sentinel_scale_factor", 10000.0),
                sentinel_mean=data_cfg.get("sentinel_mean"),
                sentinel_std=data_cfg.get("sentinel_std"),
            )

            val_dataset = FlairMultimodalDataset(
                image_dir=data_cfg["val"]["images"],
                mask_dir=data_cfg["val"]["masks"],
                sentinel_dir=data_cfg["val"]["sentinel"],
                centroids_path=data_cfg["centroids_path"],
                num_classes=data_cfg["num_classes"],
                image_transform=image_transform,
                selected_channels=selected_channels,
                sentinel_patch_size=data_cfg.get("sentinel_patch_size", 10),
                context_size=data_cfg.get("context_size"),
                use_monthly_average=data_cfg.get("use_monthly_average", True),
                cloud_snow_cover_threshold=data_cfg.get("cloud_snow_cover_threshold", 0.6),
                cloud_snow_prob_threshold=data_cfg.get("cloud_snow_prob_threshold", 50),
                sentinel_scale_factor=data_cfg.get("sentinel_scale_factor", 10000.0),
                sentinel_mean=data_cfg.get("sentinel_mean"),
                sentinel_std=data_cfg.get("sentinel_std"),
            )

            test_dataset = FlairMultimodalDataset(
                image_dir=data_cfg["test"]["images"],
                mask_dir=data_cfg["test"]["masks"],
                sentinel_dir=data_cfg["test"]["sentinel"],
                centroids_path=data_cfg["centroids_path"],
                num_classes=data_cfg["num_classes"],
                image_transform=image_transform,
                selected_channels=selected_channels,
                sentinel_patch_size=data_cfg.get("sentinel_patch_size", 10),
                context_size=data_cfg.get("context_size"),
                use_monthly_average=data_cfg.get("use_monthly_average", True),
                cloud_snow_cover_threshold=data_cfg.get("cloud_snow_cover_threshold", 0.6),
                cloud_snow_prob_threshold=data_cfg.get("cloud_snow_prob_threshold", 50),
                sentinel_scale_factor=data_cfg.get("sentinel_scale_factor", 10000.0),
                sentinel_mean=data_cfg.get("sentinel_mean"),
                sentinel_std=data_cfg.get("sentinel_std"),
            )

            generator = create_generator(seed)
            num_workers = data_cfg.get("num_workers", 4)

            train_loader = DataLoader(
                train_dataset,
                batch_size=data_cfg["batch_size"],
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=generator,
                persistent_workers=bool(num_workers > 0),
                collate_fn=multimodal_collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=data_cfg["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                persistent_workers=bool(num_workers > 0),
                collate_fn=multimodal_collate_fn,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=data_cfg["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                persistent_workers=bool(num_workers > 0),
                collate_fn=multimodal_collate_fn,
            )

            # Build model
            device = torch.device(
                "cuda:0"
                if torch.cuda.is_available() and config["training"]["device"] == "cuda"
                else "cpu",
            )
            logger.info(
                "Using device: %s",
                torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            )

            model = build_model(
                model_type=config["model"]["model_type"],
                encoder_name=config["model"].get("encoder_name", ""),
                encoder_weights=config["model"].get("encoder_weights"),
                in_channels=config["model"].get("aerial_in_channels", 5),
                n_classes=data_cfg["num_classes"],
                model_config=config["model"],
            )

            criterion = build_loss_function(
                loss_type=config["training"]["loss_function"]["type"],
                kwargs=config["training"]["loss_function"].get("args", {}),
            )

            model.to(device)
            criterion.to(device)

            optimizer = build_optimizer(
                model=model,
                optimizer_type=config["training"]["optimizer"]["type"],
                learning_rate=config["training"]["optimizer"]["learning_rate"],
                weight_decay=config["training"]["optimizer"].get("weight_decay", 0.0001),
                betas=config["training"]["optimizer"].get("betas", [0.9, 0.999]),
            )

            accumulation_steps = config["training"].get("accumulation_steps", 1)
            optimizer_steps_per_epoch = (
                len(train_loader) + accumulation_steps - 1
            ) // accumulation_steps

            lr_scheduler = build_lr_scheduler(
                optimizer=optimizer,
                scheduler_config=config["training"].get("lr_scheduler"),
                steps_per_epoch=optimizer_steps_per_epoch,
                epochs=config["training"]["epochs"],
            )

            logger.info("Starting multimodal fusion training")
            logger.info("Fusion mode: %s", config["model"].get("fusion_mode", "weighted"))
            logger.info("Freeze encoders: %s", config["model"].get("freeze_encoders", True))

            train_multimodal(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                device=device,
                epochs=config["training"]["epochs"],
                patience=config["training"]["early_stopping_patience"],
                num_classes=data_cfg["num_classes"],
                other_class_index=data_cfg.get("other_class_index"),
                accumulation_steps=accumulation_steps,
                early_stopping_criterion=config["training"].get(
                    "early_stopping_criterion",
                    "miou",
                ),
                use_amp=config["training"].get("use_amp", False),
            )

            logger.info("Training finished. Evaluating on test set...")

            # Note: evaluate() expects single-input model, for now just compute mIoU
            test_loss, test_miou = _validate_epoch_multimodal(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                num_classes=data_cfg["num_classes"],
                ignore_index=data_cfg.get("other_class_index"),
            )

            logger.info("Test Loss: %.4f, Test mIoU: %.4f", test_loss, test_miou)
            mlflow.log_metrics({"test_loss": test_loss, "test_miou": test_miou})

            logger.info("Multimodal pipeline completed successfully.")


def add_train_eval_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for the multimodal pipeline."""
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to multimodal pipeline configuration file.",
    )
    parser.add_argument(
        "-l",
        "--logs-dir",
        type=str,
        default=None,
        help="Directory to write pipeline logs.",
    )
    parser.add_argument(
        "-q",
        "--no-stdout-logs",
        required=False,
        action="store_true",
        help="Suppress logging output in the terminal.",
    )
    return parser


def run_train_eval(args: argparse.Namespace) -> None:
    """Run the multimodal pipeline with the provided configuration."""
    config_file = Path(args.config)
    if not config_file.is_file():
        msg = f"config path {config_file} does not exist"
        raise ValueError(msg)
    config = read_yaml(config_file)

    pipeline = MultimodalTrainEvalPipeline(
        run_name=config["mlflow"]["run_name"],
        logs_dir=args.logs_dir,
    )
    pipeline.run(config, no_stdout_logs=args.no_stdout_logs)


def main() -> None:
    """Run the multimodal training/evaluation CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Multimodal fusion training pipeline")
    parser = add_train_eval_arguments(parser)
    args = parser.parse_args()
    run_train_eval(args)


if __name__ == "__main__":
    main()
