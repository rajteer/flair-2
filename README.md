# flair-2

Training and evaluation pipeline for aerial image segmentation.

## Quick Start

Install dependencies (uv is recommended):

```bash
# With uv
uv sync

# Or with pip
pip install -e .
```

## Run Training + Evaluation

Minimal launcher script:

```bash
./scripts/run_train_eval.sh                # uses configs/config.yaml
./scripts/run_train_eval.sh path/to/other_config.yaml
./scripts/run_train_eval.sh --no-stdout-logs  # pass flags directly after (uses default config)
```

All extra arguments after an optional config path are forwarded to the Python pipeline.
If the first argument starts with a dash it is treated as an option and the default config path `configs/config.yaml` is used.

## Direct Invocation

```bash
python -m src.pipeline.pipeline -c configs/config.yaml
```

## Configuration

Edit `configs/config.yaml` to change data paths, model, optimizer, loss, MLflow, and evaluation options.

## Logs & Artifacts

- Log file: `pipeline_<timestamp>[_run_name].log` in the current working directory.
- MLflow tracking: controlled by keys under `mlflow` in the config.
- Config snapshot logged to MLflow as `config_resolved.json`.

## Reproducibility

Controlled via `experiment.seed` and `experiment.deterministic` in the config.

## License

MIT
