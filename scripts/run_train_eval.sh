#!/usr/bin/env bash
# Basic launcher for the training + evaluation pipeline.
# Usage:
#   ./scripts/run_train_eval.sh [CONFIG_PATH] [--no-stdout-logs]
# If CONFIG_PATH is omitted it defaults to configs/config.yaml
# Any additional args are passed through to the Python pipeline.

set -euo pipefail

CONFIG_PATH=${1:-configs/config.yaml}

# If the first arg looks like an option (starts with -), treat it as missing config and use default
if [[ "$CONFIG_PATH" == -* ]]; then
  CONFIG_PATH="configs/config.yaml"
else
  shift || true
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

python3 -m src.pipeline.pipeline -c "$CONFIG_PATH" "$@"
