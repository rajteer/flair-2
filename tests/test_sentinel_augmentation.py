"""Tests for SentinelAugmentation and MultimodalAugmentation classes."""

import pytest
import torch

from src.data.pre_processing.multimodal_augmentation import MultimodalAugmentation
from src.data.pre_processing.sentinel_augmentation import SentinelAugmentation


class TestSentinelAugmentationInit:
    """Tests for SentinelAugmentation initialization."""

    def test_initialization_with_full_config(self) -> None:
        """Should correctly parse the configuration."""
        config = {
            "enabled": True,
            "normalization": {
                "mean": [1000.0, 2000.0, 3000.0],
                "std": [100.0, 200.0, 300.0],
            },
            "temporal": {
                "random_drop": {"enabled": True, "max_drop_ratio": 0.3},
                "temporal_shift": {"enabled": True, "max_shift": 2},
            },
        }
        aug = SentinelAugmentation(config)
        assert aug.enabled is True
        assert aug.apply_normalization is True
        assert aug.random_drop_enabled is True
        assert aug.max_drop_ratio == 0.3
        assert aug.temporal_shift_enabled is True
        assert aug.max_shift == 2

    def test_initialization_with_empty_config(self) -> None:
        """Should handle empty config with defaults."""
        aug = SentinelAugmentation({})
        assert aug.enabled is False
        assert aug.apply_normalization is False
        assert aug.random_drop_enabled is False
        assert aug.temporal_shift_enabled is False


class TestSentinelNormalization:
    """Tests for Sentinel normalization."""

    def test_normalize_applies_mean_std(self) -> None:
        """Should apply channel-wise normalization."""
        config = {
            "enabled": True,
            "normalization": {
                "mean": [100.0, 200.0],
                "std": [10.0, 20.0],
            },
        }
        aug = SentinelAugmentation(config)

        # (B, T, C, H, W) with 2 channels
        data = torch.tensor([[[100.0, 200.0]]]).view(1, 1, 2, 1, 1)
        normalized = aug.normalize(data)

        # (100 - 100) / 10 = 0, (200 - 200) / 20 = 0
        assert torch.allclose(normalized, torch.zeros_like(normalized))

    def test_normalize_preserves_shape(self) -> None:
        """Should preserve tensor shape."""
        config = {
            "enabled": True,
            "normalization": {
                "mean": [1.0] * 10,
                "std": [1.0] * 10,
            },
        }
        aug = SentinelAugmentation(config)

        data = torch.randn(4, 12, 10, 10, 10)  # (B, T, C, H, W)
        normalized = aug.normalize(data)
        assert normalized.shape == data.shape


class TestSentinelTemporalDrop:
    """Tests for temporal dropout augmentation."""

    def test_temporal_drop_preserves_at_least_one(self) -> None:
        """Should always keep at least one valid timestep."""
        config = {
            "enabled": True,
            "temporal": {
                "random_drop": {"enabled": True, "max_drop_ratio": 0.9},
            },
        }
        aug = SentinelAugmentation(config)

        # 5 timesteps, all valid
        data = torch.randn(2, 5, 10, 10, 10)
        positions = torch.arange(5).unsqueeze(0).expand(2, -1)
        pad_mask = torch.ones(2, 5, dtype=torch.bool)

        # Run multiple times to ensure at least one is kept
        for _ in range(10):
            _, _, new_pad_mask = aug.random_temporal_drop(data, positions, pad_mask)
            for b in range(2):
                assert new_pad_mask[b].sum() >= 1

    def test_temporal_drop_disabled_returns_unchanged(self) -> None:
        """Should return unchanged when disabled."""
        config = {"enabled": True, "temporal": {"random_drop": {"enabled": False}}}
        aug = SentinelAugmentation(config)

        data = torch.randn(2, 5, 10, 10, 10)
        positions = torch.arange(5).unsqueeze(0).expand(2, -1)
        pad_mask = torch.ones(2, 5, dtype=torch.bool)

        _, _, new_pad_mask = aug.random_temporal_drop(data, positions, pad_mask)
        assert torch.equal(new_pad_mask, pad_mask)


class TestSentinelTemporalShift:
    """Tests for temporal shift augmentation."""

    def test_temporal_shift_within_bounds(self) -> None:
        """Shifted positions should wrap around 0-11."""
        config = {
            "enabled": True,
            "temporal": {
                "temporal_shift": {"enabled": True, "max_shift": 2},
            },
        }
        aug = SentinelAugmentation(config)

        positions = torch.tensor([[0, 1, 11]])  # Edge cases
        pad_mask = torch.ones(1, 3, dtype=torch.bool)

        # Run multiple times
        for _ in range(10):
            shifted = aug.temporal_shift(positions.clone(), pad_mask)
            assert (shifted >= 0).all() and (shifted <= 11).all()


class TestMultimodalAugmentation:
    """Tests for MultimodalAugmentation."""

    @pytest.fixture
    def sample_config(self) -> dict:
        """Sample configuration for testing."""
        return {
            "data_augmentation": {
                "apply_augmentations": True,
                "clamp": True,
                "augmentations": {
                    "hflip": {"prob": 1.0},  # Always flip for testing
                    "vflip": {"prob": 0.0},
                    "rotation": {"prob": 0.0},
                },
            },
            "sentinel_augmentation": {
                "enabled": True,
                "normalization": {"mean": [0.0] * 10, "std": [1.0] * 10},
            },
            "normalization": {
                "mean": [0.5, 0.5, 0.5, 0.5, 0.0],
                "std": [0.25, 0.25, 0.25, 0.25, 1.0],
            },
            "selected_channels": [0, 1, 2, 3, 4],
        }

    def test_synchronized_hflip(self, sample_config: dict) -> None:
        """Horizontal flip should be synchronized between modalities."""
        aug = MultimodalAugmentation(sample_config)

        aerial = torch.arange(16).view(1, 1, 4, 4).float()
        sentinel = torch.arange(16).view(1, 1, 1, 4, 4).float()
        mask = torch.arange(16).view(1, 4, 4)
        positions = torch.tensor([[0]])
        pad_mask = torch.ones(1, 1, dtype=torch.bool)

        # With hflip prob=1.0, flip should always happen
        flipped_aerial, flipped_sentinel, flipped_mask = aug._apply_hflip(
            aerial,
            sentinel,
            mask,
        )

        # Check last column is now first
        assert torch.equal(flipped_aerial[0, 0, :, 0], aerial[0, 0, :, -1])
        assert torch.equal(flipped_sentinel[0, 0, 0, :, 0], sentinel[0, 0, 0, :, -1])
        assert torch.equal(flipped_mask[0, :, 0], mask[0, :, -1])

    def test_shape_preservation(self, sample_config: dict) -> None:
        """All shapes should be preserved after augmentation."""
        aug = MultimodalAugmentation(sample_config)

        aerial = torch.randn(2, 5, 512, 512)
        sentinel = torch.randn(2, 12, 10, 10, 10)
        mask = torch.randint(0, 13, (2, 512, 512))
        positions = torch.arange(12).unsqueeze(0).expand(2, -1)
        pad_mask = torch.ones(2, 12, dtype=torch.bool)

        out_aerial, out_sentinel, out_mask, out_pos, out_pad = aug(
            aerial,
            sentinel,
            mask,
            positions,
            pad_mask,
            training=True,
        )

        assert out_aerial.shape == aerial.shape
        assert out_sentinel.shape == sentinel.shape
        assert out_mask.shape == mask.shape
        assert out_pos.shape == positions.shape
        assert out_pad.shape == pad_mask.shape

    def test_training_false_skips_augmentations(self, sample_config: dict) -> None:
        """Should skip geometric and aerial augmentations when not training."""
        # Set all probs to 1.0 so we can detect changes
        sample_config["data_augmentation"]["augmentations"]["hflip"]["prob"] = 1.0
        aug = MultimodalAugmentation(sample_config)

        aerial = torch.randn(1, 5, 32, 32)
        sentinel = torch.randn(1, 4, 10, 4, 4)
        mask = torch.randint(0, 13, (1, 32, 32))
        positions = torch.arange(4).unsqueeze(0)
        pad_mask = torch.ones(1, 4, dtype=torch.bool)

        # With training=False, aerial should only have normalization applied
        # (sentinel normalization is always applied)
        out_aerial, _, _, _, _ = aug(
            aerial.clone(),
            sentinel.clone(),
            mask.clone(),
            positions.clone(),
            pad_mask.clone(),
            training=False,
        )

        # Aerial should be unchanged (no flip, no photometric)
        assert torch.equal(out_aerial, aerial)
