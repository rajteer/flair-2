"""Tests for FlairAugmentation in src/data/pre_processing/data_augmentation.py."""

import pytest
import torch

from src.data.pre_processing.data_augmentation import FlairAugmentation


class TestFlairAugmentationInit:
    """Tests for FlairAugmentation initialization."""

    def test_parses_data_config_correctly(self, sample_data_config: dict) -> None:
        """Should correctly parse the data configuration."""
        aug = FlairAugmentation(sample_data_config)

        assert aug.clamp is True
        assert "hflip" in aug.augmentation_config
        assert "vflip" in aug.augmentation_config

    def test_handles_missing_augmentation_section(self) -> None:
        """Should handle config without data_augmentation section."""
        config = {"normalization": {"mean": [0.5], "std": [0.2]}}
        aug = FlairAugmentation(config)

        assert aug.augmentation_config == {}
        assert aug.clamp is False

    def test_handles_empty_config(self) -> None:
        """Should handle completely empty config."""
        aug = FlairAugmentation({})

        assert aug.augmentation_config == {}
        assert aug.clamp is False

    def test_initializes_clamp_bounds(self, sample_data_config: dict) -> None:
        """Should initialize clamp bounds from normalization config."""
        aug = FlairAugmentation(sample_data_config)

        assert aug.channel_clamp_min is not None
        assert aug.channel_clamp_max is not None


class TestFlairAugmentationGeometric:
    """Tests for geometric augmentations (hflip, vflip, rotation)."""

    @pytest.fixture
    def augmenter_geometric_only(self) -> FlairAugmentation:
        """Create augmenter with only geometric augmentations at 100% probability."""
        config = {
            "data_augmentation": {
                "clamp": False,
                "augmentations": {
                    "hflip": {"prob": 1.0},
                },
            },
        }
        return FlairAugmentation(config)

    def test_hflip_preserves_shape(self, augmenter_geometric_only: FlairAugmentation) -> None:
        """Horizontal flip should preserve tensor shapes."""
        image = torch.randn(5, 64, 64)
        mask = torch.randint(0, 13, (64, 64))

        aug_image, aug_mask = augmenter_geometric_only.apply_flair_augmentations(image, mask)

        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape

    def test_hflip_transforms_mask_consistently(self) -> None:
        """Horizontal flip should apply same transform to image and mask."""
        config = {
            "data_augmentation": {
                "augmentations": {"hflip": {"prob": 1.0}},
            },
        }
        aug = FlairAugmentation(config)

        # Create image and mask with known asymmetric pattern
        image = torch.zeros(5, 4, 4)
        image[:, :, 0] = 1.0  # Left column is 1
        mask = torch.zeros(4, 4, dtype=torch.long)
        mask[:, 0] = 1  # Left column is class 1

        aug_image, aug_mask = aug.apply_flair_augmentations(image, mask)

        # After hflip, the right column should be 1
        assert torch.allclose(aug_image[:, :, -1], torch.ones(5, 4))
        assert torch.all(aug_mask[:, -1] == 1)

    def test_rotation_preserves_shape(self) -> None:
        """Rotation should preserve tensor shapes for square images."""
        config = {
            "data_augmentation": {
                "augmentations": {"rotation": {"prob": 1.0, "angles": [90]}},
            },
        }
        aug = FlairAugmentation(config)

        image = torch.randn(5, 64, 64)
        mask = torch.randint(0, 13, (64, 64))

        aug_image, aug_mask = aug.apply_flair_augmentations(image, mask)

        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape


class TestFlairAugmentationPhotometric:
    """Tests for photometric augmentations (brightness, contrast)."""

    def test_brightness_only_affects_optical_channels(self) -> None:
        """Brightness should only modify channels 0-3, not elevation (channel 4)."""
        config = {
            "data_augmentation": {
                "augmentations": {"brightness": {"prob": 1.0, "range": [1.5, 1.5]}},
            },
        }
        aug = FlairAugmentation(config)

        image = torch.ones(5, 64, 64)  # All channels = 1
        original_elevation = image[4].clone()
        mask = torch.zeros(64, 64, dtype=torch.long)

        aug_image, _ = aug.apply_flair_augmentations(image, mask)

        # Elevation channel should be unchanged
        assert torch.allclose(aug_image[4], original_elevation)
        # Optical channels should be modified
        assert not torch.allclose(aug_image[:4], torch.ones(4, 64, 64))

    def test_contrast_only_affects_optical_channels(self) -> None:
        """Contrast should only modify channels 0-3, not elevation."""
        config = {
            "data_augmentation": {
                "augmentations": {"contrast": {"prob": 1.0, "range": [1.5, 1.5]}},
            },
        }
        aug = FlairAugmentation(config)

        # Create image with known values
        image = torch.ones(5, 64, 64) * 0.5
        image[4] = 100.0  # Distinct elevation value
        mask = torch.zeros(64, 64, dtype=torch.long)

        aug_image, _ = aug.apply_flair_augmentations(image, mask)

        # Elevation channel should be unchanged
        assert torch.allclose(aug_image[4], torch.ones(64, 64) * 100.0)


class TestFlairAugmentationElevation:
    """Tests for elevation-specific augmentations."""

    def test_elevation_shift_only_affects_channel_4(self) -> None:
        """Elevation shift should only modify channel 4."""
        config = {
            "data_augmentation": {
                "augmentations": {"elevation_shift": {"prob": 1.0, "range": [10.0, 10.0]}},
            },
        }
        aug = FlairAugmentation(config)

        image = torch.zeros(5, 64, 64)
        original_optical = image[:4].clone()
        mask = torch.zeros(64, 64, dtype=torch.long)

        aug_image, _ = aug.apply_flair_augmentations(image, mask)

        # Optical channels should be unchanged
        assert torch.allclose(aug_image[:4], original_optical)

    def test_elevation_augmentations_skip_4channel_images(self) -> None:
        """Elevation augmentations should be no-op for images without elevation channel."""
        config = {
            "data_augmentation": {
                "augmentations": {
                    "elevation_shift": {"prob": 1.0, "range": [10.0, 10.0]},
                    "elevation_scale": {"prob": 1.0, "range": [2.0, 2.0]},
                },
            },
        }
        aug = FlairAugmentation(config)

        image = torch.ones(4, 64, 64)  # Only 4 channels, no elevation
        mask = torch.zeros(64, 64, dtype=torch.long)

        aug_image, _ = aug.apply_flair_augmentations(image, mask)

        # Image should be unchanged
        assert torch.allclose(aug_image, image)


class TestFlairAugmentationBatch:
    """Tests for batch processing via __call__."""

    def test_batch_processing(self, sample_data_config: dict) -> None:
        """Should process batched inputs (B, C, H, W)."""
        aug = FlairAugmentation(sample_data_config)

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        aug_images, aug_masks = aug(images, masks)

        assert aug_images.shape == images.shape
        assert aug_masks.shape == masks.shape

    def test_single_image_processing(self, sample_data_config: dict) -> None:
        """Should process single images (C, H, W)."""
        aug = FlairAugmentation(sample_data_config)

        image = torch.randn(5, 64, 64)
        mask = torch.randint(0, 13, (64, 64))

        aug_image, aug_mask = aug(image, mask)

        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape


class TestFlairAugmentationClamp:
    """Tests for clamping functionality."""

    def test_clamp_limits_values(self, sample_data_config: dict) -> None:
        """Clamp should limit values to valid normalized range."""
        aug = FlairAugmentation(sample_data_config)

        # Create image with extreme values
        image = torch.ones(5, 64, 64) * 10.0  # Very high values
        mask = torch.zeros(64, 64, dtype=torch.long)

        # Temporarily disable augmentations to only test clamping
        aug.augmentation_config = {}

        aug_image, _ = aug.apply_flair_augmentations(image, mask)

        # Values should be clamped
        assert aug_image.max() <= aug.channel_clamp_max.max()
