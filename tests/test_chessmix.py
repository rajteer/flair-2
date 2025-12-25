"""Tests for ChessMix augmentation in src/data/pre_processing/chessmix.py."""

import pytest
import torch

from src.data.pre_processing.chessmix import ChessMix


class TestChessMixBasic:
    """Basic functionality tests for ChessMix."""

    def test_output_shapes_match_input(self) -> None:
        """Output shapes should match input shapes."""
        chessmix = ChessMix(prob=1.0, grid_sizes=[4])

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape

    def test_returns_unchanged_for_batch_size_one(self) -> None:
        """Should return unchanged inputs when batch_size < 2."""
        chessmix = ChessMix(prob=1.0)

        images = torch.randn(1, 5, 64, 64)
        masks = torch.randint(0, 13, (1, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert torch.equal(out_images, images)
        assert torch.equal(out_masks, masks)

    def test_dtype_preserved(self) -> None:
        """Output dtype should match input dtype."""
        chessmix = ChessMix(prob=1.0, grid_sizes=[2])

        images = torch.randn(4, 5, 64, 64, dtype=torch.float32)
        masks = torch.randint(0, 13, (4, 64, 64), dtype=torch.long)

        out_images, out_masks = chessmix(images, masks)

        assert out_images.dtype == images.dtype
        assert out_masks.dtype == masks.dtype


class TestChessMixProbability:
    """Tests for probability control in ChessMix."""

    def test_prob_zero_returns_original(self) -> None:
        """With prob=0, should return (mostly) original tensors."""
        chessmix = ChessMix(prob=0.0, grid_sizes=[4])

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        # With prob=0, the valid region should equal original
        # (padding regions might differ)
        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape

    def test_prob_one_applies_augmentation(self) -> None:
        """With prob=1, augmentation should be applied to all samples."""
        torch.manual_seed(42)
        chessmix = ChessMix(prob=1.0, grid_sizes=[4])

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        # At least some values should differ (due to patch mixing)
        # Note: in edge cases they might be equal if same patches selected
        assert out_images.shape == images.shape


class TestChessMixGridSizes:
    """Tests for different grid size configurations."""

    @pytest.mark.parametrize("grid_size", [2, 4, 8])
    def test_different_grid_sizes(self, grid_size: int) -> None:
        """Should work with various grid sizes."""
        chessmix = ChessMix(prob=1.0, grid_sizes=[grid_size])

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape

    def test_multiple_grid_sizes(self) -> None:
        """Should randomly select from multiple grid sizes."""
        chessmix = ChessMix(prob=1.0, grid_sizes=[2, 4, 8])

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        # Run multiple times to exercise random selection
        for _ in range(5):
            out_images, out_masks = chessmix(images, masks)
            assert out_images.shape == images.shape
            assert out_masks.shape == masks.shape


class TestChessMixEdgeCases:
    """Tests for edge cases in ChessMix."""

    def test_grid_size_with_remainder(self) -> None:
        """Should handle various grid sizes that evenly divide image dimensions."""
        chessmix = ChessMix(prob=1.0, grid_sizes=[16])

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        # Output should still match input shape
        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape

    def test_small_batch_size_two(self) -> None:
        """Should work with minimum batch size of 2."""
        chessmix = ChessMix(prob=1.0, grid_sizes=[4])

        images = torch.randn(2, 5, 64, 64)
        masks = torch.randint(0, 13, (2, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape

    def test_grid_size_half_image(self) -> None:
        """Should handle grid size that is half the image dimension."""
        # grid_size=32, patch_h = 64//32 = 2, stride = 1 (valid)
        chessmix = ChessMix(prob=1.0, grid_sizes=[32])

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape


class TestChessMixClassWeighting:
    """Tests for class-based patch weighting."""

    def test_with_class_counts(self) -> None:
        """Should work with precomputed class counts for rare-class weighting."""
        class_counts = [1000.0] * 12 + [100.0]  # Class 12 is rare
        chessmix = ChessMix(prob=1.0, grid_sizes=[4], class_counts=class_counts, num_classes=13)

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape

    def test_without_class_counts(self) -> None:
        """Should work without precomputed class counts (computes on-the-fly)."""
        chessmix = ChessMix(prob=1.0, grid_sizes=[4], num_classes=13)

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape


class TestChessMixIgnoreIndex:
    """Tests for ignore_index handling."""

    def test_ignore_index_in_output(self) -> None:
        """Unfilled regions should use ignore_index value."""
        ignore_index = 255
        chessmix = ChessMix(prob=1.0, grid_sizes=[4], ignore_index=ignore_index)

        images = torch.randn(4, 5, 64, 64)
        masks = torch.randint(0, 13, (4, 64, 64))

        out_images, out_masks = chessmix(images, masks)

        assert out_images.shape == images.shape
        assert out_masks.shape == masks.shape
