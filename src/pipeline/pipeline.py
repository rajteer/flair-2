"""Training and evaluation pipeline."""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch.utils.data import DataLoader

from src.data.dataset_utils import collate_standard, pad_collate_flair
from src.data.pre_processing.flair_dataset import FlairDataset, MultiChannelNormalize
from src.models.model_builder import (
    build_loss_function,
    build_lr_scheduler,
    build_model,
    build_optimizer,
)
from src.models.train import train
from src.models.validation import evaluate
from src.utils.logging_utils import LOG_FORMATTER, setup_logging
from src.utils.mlflow_utils import init_mlflow
from src.utils.read_yaml import read_yaml
from src.utils.reproducibility import create_generator, seed_everything, seed_worker
from src.visualization.utils import class_name_mapping

logger = logging.getLogger(__name__)


class TrainEvalPipeline:
    """Pipeline for training and evaluation."""

    def __init__(self, run_name: str | None = None, logs_dir: str | None = None) -> None:
        """Initialize the TrainEvalPipeline class and set up logging configurations."""
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("_") if run_name else ""
        run_suffix = f"_{safe_name}" if safe_name else ""
        logs_path: Path = Path(logs_dir).expanduser().resolve() if logs_dir else Path.cwd()
        if logs_dir:
            logs_path.mkdir(parents=True, exist_ok=True)
        self.log_file = logs_path / f"pipeline_{timestamp}{run_suffix}.log"

    def run(
        self,
        config: dict[str, Any],
        *,
        no_stdout_logs: bool = False,
        pruning_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Execute the training and evaluation pipeline.

        Args:
            config: Configuration dictionary for the pipeline.
            no_stdout_logs: Flag to control logging to stdout. Defaults to False.

        """
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

        logger.info("Starting train-evaluation pipeline.")

        with mlflow.start_run(
            run_name=config["mlflow"]["run_name"],
            nested=mlflow.active_run() is not None,
        ):
            mlflow.log_dict(config, artifact_file="config_resolved.json")

            mlflow.log_params(
                {
                    "seed": seed,
                    "deterministic": deterministic,
                },
            )

            mlflow.log_params(
                {
                    "model_type": config["model"]["model_type"],
                    "encoder_name": config["model"]["encoder_name"],
                    "encoder_weights": config["model"]["encoder_weights"],
                    "in_channels": len(config["data"]["selected_channels"]),
                    "n_classes": config["data"]["num_classes"],
                    "activation": config["model"].get("activation"),
                    "stochastic_depth": config["model"].get("stochastic_depth"),
                    "decoder_norm": config["model"].get("decoder_norm"),
                },
            )

            mlflow.log_params(
                {
                    "optimizer": config["training"]["optimizer"]["type"],
                    "learning_rate": config["training"]["optimizer"]["learning_rate"],
                    "loss_function": config["training"]["loss_function"]["type"],
                    "weight_decay": config["training"]["optimizer"]["weight_decay"],
                    "epochs": config["training"]["epochs"],
                    "patience": config["training"]["early_stopping_patience"],
                    "lr_scheduler": config["training"].get("lr_scheduler", {}).get("type", "None"),
                    "accumulation_steps": config["training"].get("accumulation_steps", 1),
                    "use_amp": config["training"].get("use_amp", False),
                },
            )

            if scheduler_args := config["training"].get("lr_scheduler", {}).get("args"):
                mlflow.log_param("lr_scheduler_args", json.dumps(scheduler_args))

            mlflow.log_params(
                {
                    "batch_size": config["data"]["batch_size"],
                    "selected_channels": config["data"]["selected_channels"],
                    "with_augmentation": config["data"]["data_augmentation"]["apply_augmentations"],
                },
            )

            if note := config["mlflow"].get("note"):
                mlflow.set_tag("note", note)

            mlflow.set_tag("dataset_version", config["data"]["dataset_version"])

            norm_cfg = config["data"].get("normalization")
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

            use_sentinel = config["data"].get("use_sentinel", False)
            sentinel_config = {
                "centroids_path": config["data"].get("centroids_path") if use_sentinel else None,
                "use_sentinel": use_sentinel,
                "use_monthly_average": config["data"].get("use_monthly_average", False),
                "remove_cloudy_snowy_timesteps": config["data"].get(
                    "remove_cloudy_snowy_timesteps",
                    False,
                ),
                "cloud_snow_cover_threshold": config["data"].get("cloud_snow_cover_threshold", 0.6),
                "cloud_snow_prob_threshold": config["data"].get("cloud_snow_prob_threshold", 50),
            }

            if use_sentinel:
                mlflow.log_params(
                    {
                        "use_sentinel": use_sentinel,
                        "use_monthly_average": sentinel_config["use_monthly_average"],
                        "remove_cloudy_snowy_timesteps": sentinel_config[
                            "remove_cloudy_snowy_timesteps"
                        ],
                        "cloud_snow_cover_threshold": sentinel_config["cloud_snow_cover_threshold"],
                        "cloud_snow_prob_threshold": sentinel_config["cloud_snow_prob_threshold"],
                    },
                )

            test_dataset = FlairDataset(
                image_dir=config["data"]["test"]["images"],
                mask_dir=config["data"]["test"]["masks"],
                sentinel_dir=config["data"]["test"].get("sentinel") if use_sentinel else None,
                num_classes=config["data"]["num_classes"],
                image_transform=image_transform,
                selected_channels=config["data"]["selected_channels"],
                **sentinel_config,
            )

            train_dataset = FlairDataset(
                image_dir=config["data"]["train"]["images"],
                mask_dir=config["data"]["train"]["masks"],
                sentinel_dir=config["data"]["train"].get("sentinel") if use_sentinel else None,
                num_classes=config["data"]["num_classes"],
                image_transform=image_transform,
                selected_channels=config["data"]["selected_channels"],
                **sentinel_config,
            )

            val_dataset = FlairDataset(
                image_dir=config["data"]["val"]["images"],
                mask_dir=config["data"]["val"]["masks"],
                sentinel_dir=config["data"]["val"].get("sentinel") if use_sentinel else None,
                num_classes=config["data"]["num_classes"],
                image_transform=image_transform,
                selected_channels=config["data"]["selected_channels"],
                **sentinel_config,
            )

            generator = create_generator(seed)
            num_workers = config["data"]["num_workers"]
            pin_memory = config["data"].get("pin_memory", True)

            train_loader = DataLoader(
                train_dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=generator,
                persistent_workers=bool(num_workers > 0),
                pin_memory=pin_memory,
                collate_fn=pad_collate_flair if use_sentinel else collate_standard,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                persistent_workers=bool(num_workers > 0),
                pin_memory=pin_memory,
                collate_fn=pad_collate_flair if use_sentinel else collate_standard,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                persistent_workers=bool(num_workers > 0),
                pin_memory=pin_memory,
                collate_fn=pad_collate_flair if use_sentinel else collate_standard,
            )

            criterion = build_loss_function(
                loss_type=config["training"]["loss_function"]["type"],
                kwargs=config["training"]["loss_function"].get("args", {}),
            )

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
                encoder_name=config["model"]["encoder_name"],
                encoder_weights=config["model"]["encoder_weights"],
                in_channels=len(config["data"]["selected_channels"]),
                n_classes=config["data"]["num_classes"],
                stochastic_depth=config["model"].get("stochastic_depth"),
                decoder_norm=config["model"].get("decoder_norm"),
                model_config=config["model"].get("model_config"),
            )

            model.to(device)
            criterion.to(device)

            optimizer = build_optimizer(
                model=model,
                optimizer_type=config["training"]["optimizer"]["type"],
                learning_rate=config["training"]["optimizer"]["learning_rate"],
                weight_decay=config["training"]["optimizer"]["weight_decay"],
                betas=config["training"]["optimizer"]["betas"],
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

            logger.info("Starting training model %s", config["model"]["model_type"])

            metrics = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                device=device,
                apply_augmentations=config["data"]["data_augmentation"]["apply_augmentations"],
                data_config=config["data"],
                epochs=config["training"]["epochs"],
                patience=config["training"]["early_stopping_patience"],
                num_classes=config["data"]["num_classes"],
                other_class_index=config["data"]["other_class_index"],
                accumulation_steps=config["training"].get("accumulation_steps", 1),
                early_stopping_criterion=config["training"].get("early_stopping_criterion", "loss"),
                use_amp=config["training"].get("use_amp", False),
                pruning_callback=pruning_callback,
            )

            logger.info("Training finished. Evaluating the model...")

            visualization_labels = config.get("visualization", {}).get("labels", None)

            labels_language = config.get("visualization", {}).get("language")
            class_labels = class_name_mapping.get(labels_language)

            if labels_language is not None and class_labels is None:
                logger.warning(
                    "Unsupported visualization.language '%s'. No class name mapping will be used.",
                    labels_language,
                )

            zone_data_loader = None
            zone_mosaic_cfg = config.get("evaluation", {}).get("zone_mosaic")
            if zone_mosaic_cfg and zone_mosaic_cfg.get("enabled", False):
                zone_image_dir = zone_mosaic_cfg.get("image_dir")
                zone_mask_dir = zone_mosaic_cfg.get("mask_dir")

                if zone_image_dir and zone_mask_dir:
                    logger.info("Creating zone dataloader for mosaic: %s", zone_image_dir)
                    zone_dataset = FlairDataset(
                        image_dir=zone_image_dir,
                        mask_dir=zone_mask_dir,
                        sentinel_dir=zone_mosaic_cfg.get("sentinel_dir") if use_sentinel else None,
                        num_classes=config["data"]["num_classes"],
                        image_transform=image_transform,
                        selected_channels=config["data"]["selected_channels"],
                        **sentinel_config,
                    )
                    zone_data_loader = DataLoader(
                        zone_dataset,
                        batch_size=config["data"]["batch_size"],
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=pad_collate_flair if use_sentinel else collate_standard,
                    )
                else:
                    logger.warning(
                        "Zone mosaic enabled but image_dir or mask_dir not provided.",
                    )

            evaluate(
                model=model,
                device=device,
                data_loader=test_loader,
                num_classes=config["data"]["num_classes"],
                other_class_index=config["data"]["other_class_index"],
                class_name_mapping=class_labels,
                log_confusion_matrix=config["evaluation"]["log_confusion_matrix"],
                sample_ids_to_plot=config["evaluation"]["log_sample_ids"],
                visualization_labels=visualization_labels,
                zone_mosaic_config=zone_mosaic_cfg,
                zone_data_loader=zone_data_loader,
            )

            mlflow.log_artifact(str(self.log_file), artifact_path="logs")

            return metrics


def add_train_eval_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments for the training and evaluation pipeline to an argument parser.

    Args:
        parser: The argument parser to which arguments will be added.

    Returns:
        argparse.ArgumentParser: The argument parser with the added arguments.

    """
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to train/eval pipeline configuration file.",
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
    """Run the training and evaluation pipeline with the provided configuration.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Raises:
        ValueError: If the configuration file path does not exist.

    """
    config_file = Path(args.config)
    if not config_file.is_file():
        msg = f"config path {config_file} does not exist"
        raise ValueError(msg)
    config = read_yaml(config_file)

    pipeline = TrainEvalPipeline(run_name=config["mlflow"]["run_name"], logs_dir=args.logs_dir)
    pipeline.run(config, no_stdout_logs=args.no_stdout_logs)


def main() -> None:
    """Run the training/evaluation CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser = add_train_eval_arguments(parser)

    args = parser.parse_args()
    run_train_eval(args)


if __name__ == "__main__":
    main()
