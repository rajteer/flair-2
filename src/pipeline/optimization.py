"""Hyperparameter optimization with Optuna and MLflow integration."""

import argparse
import copy
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.visualization import plot_param_importances

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
        timeout_seconds: int | None = None,
        storage_path: str | None = None,
    ) -> None:
        """Initialize the optimization pipeline.

        Args:
            base_config_path: Path to the base training configuration.
            opt_config_path: Path to the optimization configuration.
            timeout_seconds: Timeout in seconds. Overrides config value if provided.
            storage_path: Path to SQLite database for study persistence.
                If provided, the study will be saved to disk and can be resumed.

        """
        self.base_config = read_yaml(base_config_path)
        self.opt_config = read_yaml(opt_config_path)

        self.study_name = self.opt_config["optimization"]["study_name"]
        self.n_trials = self.opt_config["optimization"]["n_trials"]
        self.direction = self.opt_config["optimization"].get("direction", "maximize")
        self.timeout = timeout_seconds or self.opt_config["optimization"].get("timeout_seconds")
        self.storage = f"sqlite:///{storage_path}" if storage_path else None

        # Override epochs if specified in optimization config
        if "n_epochs" in self.opt_config["optimization"]:
            self.base_config["training"]["epochs"] = self.opt_config["optimization"]["n_epochs"]

    def _suggest_params(self, trial: optuna.Trial, config: dict[str, Any]) -> None:
        """Suggest parameters based on configuration and update the config dict."""
        search_space = self.opt_config["optimization"]["search_space"]
        model_specific = self.opt_config["optimization"].get("model_specific_search_space", {})
        model_specific = {k.lower(): v for k, v in model_specific.items()}

        for item in search_space:
            self._suggest_single_param(trial, config, item)

        model_type = config.get("model", {}).get("model_type", "").lower()
        if model_type in model_specific:
            for item in model_specific[model_type]:
                self._suggest_single_param(trial, config, item)

    def _suggest_single_param(
        self,
        trial: optuna.Trial,
        config: dict[str, Any],
        item: dict[str, Any],
    ) -> None:
        """Suggest a single parameter and update the config."""
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
        config["training"]["early_stopping_criterion"] = "miou"

        def pruning_callback(value: float, step: int) -> None:
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned

        try:
            pipeline = TrainEvalPipeline(run_name=run_name, logs_dir="logs_optuna")
            metrics = pipeline.run(config, no_stdout_logs=True, pruning_callback=pruning_callback)
            return metrics["best_val_miou"]
        except optuna.TrialPruned:
            raise
        except Exception:
            logger.exception("Trial %d failed", trial.number)
            return 0.0

    def _plot_param_importances(
        self,
        study: optuna.Study,
    ) -> None:
        """Generate and log hyperparameter importance plot to MLflow.

        Args:
            study: Completed Optuna study.

        """
        min_trials_for_importance = 2
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if len(completed_trials) < min_trials_for_importance:
            logger.warning(
                "Not enough completed trials (%d) to compute hyperparameter importance. "
                "Need at least %d.",
                len(completed_trials),
                min_trials_for_importance,
            )
            return

        try:
            fig = plot_param_importances(study)

            with tempfile.TemporaryDirectory() as tmpdir:
                html_path = Path(tmpdir) / f"{study.study_name}_param_importances.html"
                fig.write_html(str(html_path))
                mlflow.log_artifact(str(html_path), artifact_path="optuna_plots")
                logger.info("Logged hyperparameter importance plot (HTML) to MLflow")

                try:
                    png_path = Path(tmpdir) / f"{study.study_name}_param_importances.png"
                    fig.write_image(str(png_path))
                    mlflow.log_artifact(str(png_path), artifact_path="optuna_plots")
                    logger.info("Logged hyperparameter importance plot (PNG) to MLflow")
                except (ImportError, ValueError, OSError):
                    logger.warning(
                        "Could not save PNG (install kaleido: pip install kaleido). "
                        "HTML version was logged successfully.",
                    )

        except Exception:
            logger.exception("Failed to generate hyperparameter importance plot")

    def run(self) -> None:
        """Run the optimization."""
        mlflow_kwargs = {
            "tracking_uri": self.base_config["mlflow"]["tracking_uri"],
            "metric_name": "best_val_miou",
        }

        mlc = MLflowCallback(
            tracking_uri=mlflow_kwargs["tracking_uri"],
            metric_name=mlflow_kwargs["metric_name"],
            mlflow_kwargs={"nested": True},
        )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )

        if self.storage:
            logger.info("Loaded/created study with %d existing trials", len(study.trials))

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[mlc],
        )

        logger.info("Best trial:")
        trial = study.best_trial
        logger.info("  Value: %s", trial.value)
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    %s: %s", key, value)

        mlflow.set_tracking_uri(mlflow_kwargs["tracking_uri"])
        with mlflow.start_run(run_name=f"{self.study_name}_summary"):
            mlflow.log_params({"study_name": self.study_name, "n_trials": self.n_trials})
            mlflow.log_metrics({"best_val_miou": trial.value})
            for key, value in trial.params.items():
                mlflow.log_param(f"best_{key}", value)
            mlflow.log_dict(self.opt_config, artifact_file="optimization_config.json")
            self._plot_param_importances(study)


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
    parser.add_argument(
        "--timeout",
        type=int,
        default=172740,  # 47h 59min
        help="Timeout in seconds for the entire optimization (default: 172740 = 47h 59min).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database for study persistence (enables resume). "
        "Example: optuna_study.db",
    )

    args = parser.parse_args()

    setup_logging(
        log_file=Path(args.log_file),
        log_formatter=LOG_FORMATTER,
        no_stdout_logs=False,
    )

    logger.info("=" * 60)
    logger.info("Optuna optimization pipeline starting")
    logger.info("Base config: %s", args.config)
    logger.info("Optimization config: %s", args.opt_config)
    logger.info("Log file: %s", args.log_file)
    logger.info("=" * 60)

    optimization = OptimizationPipeline(
        base_config_path=Path(args.config),
        opt_config_path=Path(args.opt_config),
        timeout_seconds=args.timeout,
        storage_path=args.db,
    )

    logger.info("OptimizationPipeline initialized successfully")
    logger.info(
        "Study name: %s, Trials: %d, Timeout: %s, Storage: %s",
        optimization.study_name,
        optimization.n_trials,
        f"{optimization.timeout}s" if optimization.timeout else "None",
        optimization.storage or "in-memory",
    )

    optimization.run()


if __name__ == "__main__":
    main()
