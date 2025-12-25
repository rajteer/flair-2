"""Hyperparameter optimization with Optuna and MLflow integration."""

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import optuna
from optuna.integration.mlflow import MLflowCallback

from src.pipeline.pipeline import TrainEvalPipeline
from src.utils.logging_utils import LOG_FORMATTER, setup_logging
from src.utils.read_yaml import read_yaml

logger = logging.getLogger(__name__)


def update_nested_dict(d: dict[str, Any], key_path: str, value: Any) -> None:
    """Update a nested dictionary using a dot-notation key path."""
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


class OptimizationPipeline:
    """Pipeline for hyperparameter optimization."""

    def __init__(
        self,
        base_config_path: Path,
        opt_config_path: Path,
    ) -> None:
        """Initialize the optimization pipeline."""
        self.base_config = read_yaml(base_config_path)
        self.opt_config = read_yaml(opt_config_path)

        self.study_name = self.opt_config["optimization"]["study_name"]
        self.n_trials = self.opt_config["optimization"]["n_trials"]
        self.direction = self.opt_config["optimization"].get("direction", "minimize")

        # Override epochs if specified in optimization config
        if "n_epochs" in self.opt_config["optimization"]:
            self.base_config["training"]["epochs"] = self.opt_config["optimization"]["n_epochs"]

    def _suggest_params(self, trial: optuna.Trial, config: dict[str, Any]) -> None:
        """Suggest parameters based on configuration and update the config dict."""
        search_space = self.opt_config["optimization"]["search_space"]

        for item in search_space:
            name = item["name"]
            param_type = item["type"]

            if param_type == "float":
                val = trial.suggest_float(
                    name,
                    item["low"],
                    item["high"],
                    log=item.get("log", False),
                )
            elif param_type == "int":
                val = trial.suggest_int(name, item["low"], item["high"], log=item.get("log", False))
            elif param_type == "categorical":
                val = trial.suggest_categorical(name, item["choices"])
            else:
                msg = f"Unknown parameter type: {param_type}"
                raise ValueError(msg)

            update_nested_dict(config, name, val)

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        config = copy.deepcopy(self.base_config)

        self._suggest_params(trial, config)

        run_name = f"optuna_trial_{trial.number}"
        config["mlflow"]["run_name"] = run_name

        def pruning_callback(value: float, step: int) -> None:
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            pipeline = TrainEvalPipeline(run_name=run_name, logs_dir="logs_optuna")
            metrics = pipeline.run(config, no_stdout_logs=True, pruning_callback=pruning_callback)
            return metrics["best_val_loss"]
        except optuna.TrialPruned:
            raise
        except Exception:
            logger.exception("Trial %d failed", trial.number)
            return float("inf")

    def run(self) -> None:
        """Run the optimization."""
        mlflow_kwargs = {
            "tracking_uri": self.base_config["mlflow"]["tracking_uri"],
            "metric_name": "best_val_loss",
        }

        mlc = MLflowCallback(
            tracking_uri=mlflow_kwargs["tracking_uri"],
            metric_name=mlflow_kwargs["metric_name"],
            mlflow_kwargs={"nested": True},
        )

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )

        study.optimize(self.objective, n_trials=self.n_trials, callbacks=[mlc])

        logger.info("Best trial:")
        trial = study.best_trial
        logger.info("  Value: %s", trial.value)
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    %s: %s", key, value)


def main() -> None:
    """Run the optimization CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to base training configuration file.",
    )
    parser.add_argument(
        "-o",
        "--opt-config",
        type=str,
        required=True,
        help="Path to optimization configuration file.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="optimization.log",
        help="Path to log file.",
    )

    args = parser.parse_args()

    setup_logging(
        log_file=Path(args.log_file),
        log_formatter=LOG_FORMATTER,
        no_stdout_logs=False,
    )

    optimization = OptimizationPipeline(
        base_config_path=Path(args.config),
        opt_config_path=Path(args.opt_config),
    )
    optimization.run()


if __name__ == "__main__":
    main()
