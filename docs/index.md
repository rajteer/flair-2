# FLAIR-2

Modular training and evaluation framework for semantic segmentation of aerial and Sentinel-2 satellite imagery.

## Overview

This project provides three training pipelines for the [FLAIR-2 dataset](https://ignf.github.io/FLAIR/):

- **Aerial pipeline** — high-resolution aerial imagery segmentation
- **Sentinel pipeline** — Sentinel-2 temporal sequence segmentation
- **Multimodal pipeline** — late fusion of aerial and Sentinel-2 models

## Quick Start

```bash
# Install dependencies
uv sync

# Run aerial training
uv run python -m src.pipeline.pipeline --config configs/config.yaml

# Run Sentinel-2 training
uv run python -m src.pipeline.sentinel_pipeline --config configs/config_sentinel_tsvit.yaml
```

## API Documentation

Browse the full API reference in the sidebar to explore all model architectures, training utilities, and loss functions.
