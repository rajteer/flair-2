# FLAIR-2 Segmentation Pipeline

> A modular training and evaluation framework for semantic segmentation of aerial and Sentinel-2 satellite imagery, built on the [FLAIR-2 dataset](https://ignf.github.io/FLAIR/).

Developed as part of a master's thesis on multimodal fusion of high-resolution aerial imagery and Sentinel-2 time series for land cover mapping.

---

## Key Features

- **7+ model architectures** — SMP-based (i.e. Unet, DeepLabV3+, Segformer, FPN), UNetFormer, RS3Mamba, Samba, TSViT, UTAE++
- **3 training pipelines** — aerial-only, Sentinel-2 temporal, and multimodal late fusion
- **Hyperparameter optimization** — Optuna-powered search with MLflow integration
- **Experiment tracking** — MLflow logging with confusion matrices, prediction visualizations, and config snapshots
- **Data augmentation** — Flips, rotations, contrast/brightness, elevation perturbations, and ChessMix [[arXiv:2108.11535]](https://arxiv.org/abs/2108.11535) 
- **Fully configurable** — YAML-driven with per-experiment configuration

---

## Repository Structure

```
flair-2/
├── configs/                    # Configuration files
│   ├── examples/               # Documented example configs for every pipeline
│   ├── config.yaml             # Default aerial pipeline config
│   ├── config_multimodal.yaml  # Multimodal late fusion config
│   ├── config_sentinel_*.yaml  # Sentinel-only pipeline configs
│   └── optimization*.yaml     # Optuna hyperparameter search configs
├── scripts/                    # Utility scripts
│   ├── run_train_eval.sh       # Launcher script
│   ├── benchmark_inference.py  # Inference speed benchmarking
│   ├── compute_channel_stats.py # NIR/elevation normalization stats
│   ├── compute_class_weights.py # Class frequency weights
│   └── compute_sentinel_stats.py # Sentinel-2 band statistics
├── src/
│   ├── data/
│   │   ├── pre_processing/     # Dataset classes and augmentation
│   │   │   ├── flair_dataset.py           # Aerial (+optional Sentinel) dataset
│   │   │   ├── flair_sentinel_dataset.py  # Sentinel-2 only dataset
│   │   │   ├── flair_multimodal_dataset.py # Multimodal dataset
│   │   │   ├── data_augmentation.py       # Augmentation transforms
│   │   │   ├── chessmix.py                # ChessMix augmentation
│   │   │   └── sentinel_utils.py          # Sentinel-2 data loading utilities
│   │   └── dataset_utils.py    # Collate functions and path utilities
│   ├── models/
│   │   ├── architectures/      # Model architectures
│   │   │   ├── unetformer.py       # UNetFormer architecture
│   │   │   ├── tsvit.py            # Temporal-Spatial Vision Transformer
│   │   │   ├── tsvit_lookup.py     # TSViT variant with lookup embeddings
│   │   │   ├── rs3mamba.py         # RS3Mamba (SSM-based segmentation)
│   │   │   ├── utae_pp.py          # U-TAE++ (modernized U-TAE)
│   │   │   └── multimodal_fusion.py # Late fusion model
│   │   ├── encoders/           # Encoder backbones
│   │   │   ├── samba_encoder.py    # Samba encoder backbone
│   │   │   └── vssm_encoder.py     # Visual State Space Model encoder
│   │   ├── common_blocks.py    # Shared building blocks
│   │   ├── model_builder.py    # Unified model/optimizer/loss factory
│   │   └── utils.py            # Model utility functions
│   ├── training/               # Training and evaluation
│   │   ├── train.py            # Training and validation loops
│   │   ├── validation.py       # Evaluation metrics computation
│   │   └── losses.py           # Custom loss functions
│   ├── pipeline/
│   │   ├── pipeline.py              # Aerial training/eval pipeline
│   │   ├── sentinel_pipeline.py     # Sentinel-2 training/eval pipeline
│   │   ├── multimodal_pipeline.py   # Multimodal fusion pipeline
│   │   ├── optimization.py          # Optuna HPO for aerial models
│   │   └── sentinel_optimization.py # Optuna HPO for Sentinel models
│   ├── utils/                   # MLflow, logging, reproducibility
│   └── visualization/          # Prediction mosaics and plots
└── tests/                      # Unit tests
```

---

## Supported Models

| Model | Type | Pipeline | Description |
|-------|------|----------|-------------|
| **Unet** | SMP | Aerial | Classic U-Net with configurable encoder |
| **DeepLabV3+** | SMP | Aerial | Atrous spatial pyramid pooling decoder |
| **Segformer** | SMP | Aerial | Transformer-based segmentation |
| **FPN** | SMP | Aerial | Feature Pyramid Network |
| **UNetFormer** | Custom | Aerial | UNet-like Transformer with Global-Local Attention |
| **RS3Mamba** | Custom | Aerial | State Space Model for remote sensing |
| **Samba** | Custom | Aerial | Samba encoder with UNetFormer decoder |
| **TSViT** | Custom | Sentinel | Temporal-Spatial Vision Transformer |
| **UTAE++** | Custom | Sentinel | Modernized U-TAE with ConvNeXt blocks |
| **MultimodalLateFusion** | Custom | Multimodal | Late fusion of aerial + Sentinel models |

---

## Dataset

This pipeline is designed for the [FLAIR-2 dataset](https://ignf.github.io/FLAIR/) — a large-scale benchmark for land cover segmentation in France.

### Data Modalities

| Modality | Resolution | Channels | Description |
|----------|-----------|----------|-------------|
| **Aerial** | 0.2 m/px | 5 (R, G, B, NIR, Elevation) | High-resolution ortho-imagery |
| **Sentinel-2** | 10 m/px | 10 spectral bands | Multispectral time series (variable length) |

### Expected Directory Layout

```
data/
├── flair_2_aerial_train/       # Training aerial images (IMG_*.tif)
├── flair_2_aerial_test/        # Test aerial images
├── flair_2_labels_train/       # Training masks (MSK_*.tif)
├── flair_2_labels_test/        # Test masks
├── flair_2_sen_train/          # Training Sentinel-2 super-patches
├── flair_2_sen_test/           # Test Sentinel-2 super-patches
└── flair-2_centroids_sp_to_patch.json  # Aerial → Sentinel coordinate mapping
```
---

## Installation

**Prerequisites**: Python ≥ 3.11, CUDA-capable GPU recommended.

```bash
# Clone the repository
git clone https://github.com/rajteer/flair-2.git
cd flair-2

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

---

## Quick Start

### 1. Aerial Image Segmentation

Train a segmentation model on aerial imagery:

```bash
# Using the launcher script (default: configs/config.yaml)
./scripts/run_train_eval.sh

# With a specific config
./scripts/run_train_eval.sh configs/examples/aerial_unetformer.yaml

# Direct Python invocation
python -m src.pipeline.pipeline -c configs/examples/aerial_segformer.yaml
```

### 2. Sentinel-2 Temporal Segmentation

Train a temporal model on Sentinel-2 time series:

```bash
python -m src.pipeline.sentinel_pipeline -c configs/examples/sentinel_tsvit.yaml
```

### 3. Multimodal Late Fusion

Combine pre-trained aerial and Sentinel models:

```bash
python -m src.pipeline.multimodal_pipeline -c configs/examples/multimodal_late_fusion.yaml
```

> **Note**: Multimodal fusion requires pre-trained checkpoints for both the aerial and Sentinel models. Set `model.aerial_checkpoint` and `model.sentinel_checkpoint` in the config.

### 4. Hyperparameter Optimization

Run Optuna-powered hyperparameter search:

```bash
# Aerial models
python -m src.pipeline.optimization \
    -c configs/examples/aerial_segformer.yaml \
    -o configs/examples/optimization_aerial.yaml

# Sentinel models
python -m src.pipeline.sentinel_optimization \
    -c configs/examples/sentinel_tsvit.yaml \
    -o configs/examples/optimization_sentinel.yaml
```

---

## Configuration Reference

All behavior is controlled through YAML configuration files. See `configs/examples/` for fully documented examples.

| Section | Description |
|---------|-------------|
| `data` | Dataset paths, batch size, channels, normalization, augmentation, Sentinel options |
| `model` | Architecture type, encoder, weights, model-specific hyperparameters |
| `training` | Optimizer, LR scheduler, epochs, early stopping, loss function |
| `experiment` | Random seed, deterministic mode |
| `mlflow` | Experiment name, run name, tracking URI, DagsHub integration |
| `evaluation` | Confusion matrix logging, sample comparison visualizations |
| `visualization` | Language (`en`/`pl`), display labels |

### Loss Functions

| Loss | Config Key | Description |
|------|-----------|-------------|
| CrossEntropyLoss | `CrossEntropyLoss` | Standard CE with optional `ignore_index` |
| LovaszLoss | `LovaszLoss` | Lovász-Softmax for IoU optimization |
| CombinedDiceFocalLoss | `CombinedDiceFocalLoss` | Weighted Dice + Focal with class weights |
| WeightedCrossEntropyDiceLoss | `WeightedCrossEntropyDiceLoss` | Weighted CE + Dice |

### LR Schedulers

| Scheduler | Config Key |
|-----------|-----------|
| ReduceLROnPlateau | `ReduceLROnPlateau` |
| OneCycleLR | `OneCycleLR` |
| CosineAnnealingWarmRestarts | `CosineAnnealingWarmRestarts` |
| StepLR | `StepLR` |

---

## Experiment Tracking

Experiments are tracked with **MLflow**. Optionally, [DagsHub](https://dagshub.com/) can be used for remote tracking.

### What gets logged

- Training/validation loss and mIoU per epoch
- Model parameters count (total and trainable)
- Model FLOPs (per sample)
- Resolved configuration snapshot (`config_resolved.json`)
- Best model checkpoint (`best_model.pt`)
- Confusion matrix (as artifact)
- Prediction comparison mosaics for selected samples
- Learning rate and optimizer state

### Log files

Each pipeline run creates a timestamped log file in the working directory:
```
pipeline_20260221_120000_ExperimentName.log
```

---

## Utility Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `compute_channel_stats.py` | Compute NIR/elevation normalization statistics | `python scripts/compute_channel_stats.py --images-dir data/flair_2_aerial_train` |
| `compute_class_weights.py` | Calculate class frequency weights for loss balancing | `python scripts/compute_class_weights.py --config configs/config.yaml` |
| `compute_sentinel_stats.py` | Compute Sentinel-2 band statistics | `python scripts/compute_sentinel_stats.py --data-dir data/flair_2_sen_train` |
| `benchmark_inference.py` | Benchmark model inference speed | `python scripts/benchmark_inference.py --config configs/config.yaml --checkpoint path/to/model.pt` |

---

## Reproducibility

Reproducibility is controlled via the `experiment` section:

```yaml
experiment:
  seed: 42            # Random seed for all RNGs (Python, NumPy, PyTorch, CUDA)
  deterministic: true  # Enable PyTorch deterministic algorithms
```

All experiments in this project were conducted with `seed: 42`. All random generators (data loaders, augmentation, model initialization) are seeded from `experiment.seed`.

---

## Testing

Run the test suite:

```bash
# With uv
uv run pytest tests/ -v

# With pip
pytest tests/ -v
```