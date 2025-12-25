"""Shared pytest fixtures for the FLAIR-2 test suite."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return the appropriate torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size() -> int:
    """Default batch size for testing."""
    return 4


@pytest.fixture
def num_classes() -> int:
    """Default number of classes for segmentation."""
    return 13


@pytest.fixture
def image_size() -> int:
    """Default image size (H and W)."""
    return 64


@pytest.fixture
def in_channels() -> int:
    """Default number of input channels (RGBIR + elevation)."""
    return 5


@pytest.fixture
def sample_images(batch_size: int, in_channels: int, image_size: int) -> torch.Tensor:
    """Generate sample image batch for testing."""
    return torch.randn(batch_size, in_channels, image_size, image_size)


@pytest.fixture
def sample_masks(batch_size: int, num_classes: int, image_size: int) -> torch.Tensor:
    """Generate sample segmentation mask batch (class indices)."""
    return torch.randint(0, num_classes, (batch_size, image_size, image_size))


@pytest.fixture
def sample_predictions(batch_size: int, num_classes: int, image_size: int) -> torch.Tensor:
    """Generate sample model predictions (logits)."""
    return torch.randn(batch_size, num_classes, image_size, image_size)


@pytest.fixture
def sample_data_config(in_channels: int) -> dict:
    """Generate a sample data configuration dict for FlairAugmentation."""
    return {
        "data_augmentation": {
            "clamp": True,
            "augmentations": {
                "hflip": {"prob": 0.5},
                "vflip": {"prob": 0.5},
                "rotation": {"prob": 0.5, "angles": [0, 90, 180, 270]},
                "contrast": {"prob": 0.3, "range": [0.8, 1.2]},
                "brightness": {"prob": 0.3, "range": [0.8, 1.2]},
            },
        },
        "normalization": {
            "mean": [0.4, 0.4, 0.4, 0.4, 0.0],
            "std": [0.2, 0.2, 0.2, 0.2, 1.0],
            "elevation_range": [0.0, 1000.0],
            "elevation_channel_index": 4,
        },
        "selected_channels": [0, 1, 2, 3, 4],
    }
