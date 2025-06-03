"""
This module contains the train and evaluation pipeline.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.pre_processing.FlairDataset import FlairDataset
from src.models.model_builder import build_model, build_optimizer, build_loss_function
from src.models.train import train
from src.models.validation import evaluate
from src.utils.logging_utils import setup_logging, LOG_FORMATTER
from src.utils.mlflow_utils import init_mlflow
from src.utils.read_yaml import read_yaml

logger = logging.getLogger(__name__)


class TrainEvalPipeline:
    """
    A pipeline class to manage the training and evaluation of the segmentation models.
    """

    def __init__(self) -> None:
        """Initializes the TrainEvalPipeline class and sets up logging configurations."""
        self.log_file = Path.cwd() / "pipeline.log"

    def run(self, config: dict[str, Any], no_stdout_logs: bool = False) -> None:
        """
        Executes the training and evaluation pipeline.

        Args:
            config: A dictionary containing the configuration parameters for the pipeline.
            no_stdout_logs: A flag to control logging to stdout. Defaults to True.
        """
        init_mlflow(
            tracking_uri=config["mlflow"]["tracking_uri"],
            experiment_name=config["mlflow"]["name"],
        )

        setup_logging(
            log_file=self.log_file,
            log_formatter=LOG_FORMATTER,
            no_stdout_logs=no_stdout_logs,
        )

        logger.info("Starting train-evaluation pipeline.")

        with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
            mlflow.log_params({
                "model_type": config["model"]["model_type"],
                "encoder_name": config["model"]["encoder_name"],
                "encoder_weights": config["model"]["encoder_weights"],
                "in_channels": len(config["data"]["selected_channels"]),
                "n_classes": config["data"]["num_classes"],
            })

            mlflow.log_params({
                "optimizer": config["training"]["optimizer"],
                "learning_rate": config["training"]["learning_rate"],
                "loss_function": config["training"]["loss_function"]["type"],
                "weight_decay": config["training"]["weight_decay"],
                "epochs": config["training"]["epochs"],
                "patience": config["training"]["early_stopping_patience"],
            })

            mlflow.log_params({
                "batch_size": config["data"]["batch_size"],
                "selected_channels": config["data"]["selected_channels"],
                "with_augmentation": config["data"]["apply_augmentation"],
                "augmentation_prob": config["data"]["augmentation_prob"],
            })

            if 'notes' in config['mlflow'] and config['mlflow']['notes']:
                mlflow.set_tag("note", config['mlflow']['notes'])

            mlflow.set_tag("model_type", config["model"]["model_type"])
            mlflow.set_tag("dataset_version", config["data"]["dataset_version"])
            mlflow.set_tag("loss_function", config["training"]["loss_function"]["type"])

            test_dataset = FlairDataset(
                image_dir=config["data"]["test_images_dir"],
                mask_dir=config["data"]["test_masks_dir"],
                num_classes=config["data"]["num_classes"],
                selected_channels=config["data"]["selected_channels"],
            )

            train_dataset = FlairDataset(
                image_dir=config["data"]["train_images_dir"],
                mask_dir=config["data"]["train_masks_dir"],
                num_classes=config["data"]["num_classes"],
                selected_channels=config["data"]["selected_channels"],
                augmentation_prob=config["data"]["augmentation_prob"],
            )

            val_dataset = FlairDataset(
                image_dir=config["data"]["val_images_dir"],
                mask_dir=config["data"]["val_masks_dir"],
                num_classes=config["data"]["num_classes"],
                selected_channels=config["data"]["selected_channels"],
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=True,
                num_workers=config["data"]["num_workers"],
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
                num_workers=config["data"]["num_workers"],
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
                num_workers=config["data"]["num_workers"],
            )

            loss_config = config["training"]["loss_function"]
            criterion = build_loss_function(
                loss_type=loss_config["type"], **loss_config["args"]
            )

            device = torch.device(
                "cuda:0"
                if torch.cuda.is_available() and config["training"]["device"] == "cuda"
                else "cpu"
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
            )
            model.to(device)

            optimizer = build_optimizer(
                model=model,
                optimizer_type=config["training"]["optimizer"],
                learning_rate=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )

            logger.info("Starting training model %s", config["model"]["model_type"])

            training_history = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                path_to_requirements=config["mlflow"]["requirements_path"],
                apply_augmentations=config["data"]["apply_augmentation"],
                augmentation_prob=config["data"]["augmentation_prob"],
                epochs=config["training"]["epochs"],
                patience=config["training"]["early_stopping_patience"],
                num_classes=config["data"]["num_classes"],
            )

            logger.info("Training finished. Evaluating the model...")

            ids_to_plot = config["training"].get("plot_image_ids_path", None)
            if ids_to_plot and Path(ids_to_plot).exists():
                ids_to_plot = np.load(ids_to_plot)

            evaluate(model=model,
                     device=device,
                     data_loader=test_loader,
                     num_classes=config["data"]["num_classes"],
                     other_class_index=config["data"]["other_class_index"],
                     log_confusion_matrix=config["evaluation"]["log_confusion_matrix"],
                     log_comparison=config["evaluation"]["log_comparison"],
                     ids_to_plot=ids_to_plot,
                     eval_per_sample=config["evaluation"]["eval_per_sample"], )


def add_train_eval_arguments(
        parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Adds arguments for the training and evaluation pipeline to an argument parser.

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
        "--no-stdout-logs",
        required=False,
        action="store_true",
        help="Disable logging in the terminal.",
    )

    return parser


def run_train_eval(args: argparse.Namespace) -> None:
    """
    Runs the training and evaluation pipeline with.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Raises:
        ValueError: If the configuration file path does not exist.
    """
    config_file = Path(args.config)
    if not config_file.is_file():
        raise ValueError(f"config path {config_file} does not exist")
    config = read_yaml(config_file)

    pipeline = TrainEvalPipeline()
    pipeline.run(config, args.no_stdout_logs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser = add_train_eval_arguments(parser)

    args = parser.parse_args()
    run_train_eval(args)


if __name__ == "__main__":
    main()
