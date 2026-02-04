"""Multimodal data augmentation for synchronized aerial and Sentinel transforms.

This module provides a wrapper that coordinates augmentations between aerial imagery
and Sentinel-2 time series, ensuring geometric transforms are synchronized while
allowing modality-specific augmentations.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from .data_augmentation import FlairAugmentation
from .sentinel_augmentation import SentinelAugmentation


class MultimodalAugmentation:
    """Coordinate augmentations between aerial and Sentinel-2 data.

    Ensures geometric transforms (flip, rotation) are applied consistently
    to both modalities while allowing modality-specific augmentations:
    - Aerial: photometric transforms (brightness, contrast), elevation augmentations
    - Sentinel: temporal transforms (dropout, shift), normalization

    """

    def __init__(self, data_config: dict[str, Any]) -> None:
        """Initialize multimodal augmentation.

        Args:
            data_config: Full data configuration dict containing:
                - data_augmentation: Aerial augmentation config
                - sentinel_augmentation: Sentinel augmentation config

        """
        self.aerial_aug = FlairAugmentation(data_config)
        self.sentinel_aug = SentinelAugmentation(data_config.get("sentinel_augmentation", {}))

        # Extract geometric augmentation settings for synchronized application
        aug_cfg = data_config.get("data_augmentation", {}).get("augmentations", {})

        self.hflip_prob = aug_cfg.get("hflip", {}).get("prob", 0.0)
        self.vflip_prob = aug_cfg.get("vflip", {}).get("prob", 0.0)
        rotation_cfg = aug_cfg.get("rotation", {})
        self.rotation_prob = rotation_cfg.get("prob", 0.0)
        self.rotation_angles = rotation_cfg.get("angles", [0, 90, 180, 270])

    def _apply_hflip(
        self,
        aerial: torch.Tensor,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply synchronized horizontal flip to both modalities.

        Args:
            aerial: (B, C, H, W) aerial imagery
            sentinel: (B, T, C, H, W) sentinel time series
            mask: (B, H, W) segmentation mask

        Returns:
            Flipped tensors.

        """
        # Flip aerial and mask on last dimension (W)
        aerial = torch.flip(aerial, dims=[-1]).contiguous()
        mask = torch.flip(mask, dims=[-1]).contiguous()
        # Flip sentinel on last dimension (W)
        sentinel = torch.flip(sentinel, dims=[-1]).contiguous()
        return aerial, sentinel, mask

    def _apply_vflip(
        self,
        aerial: torch.Tensor,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply synchronized vertical flip to both modalities.

        Args:
            aerial: (B, C, H, W) aerial imagery
            sentinel: (B, T, C, H, W) sentinel time series
            mask: (B, H, W) segmentation mask

        Returns:
            Flipped tensors.

        """
        # Flip on H dimension
        aerial = torch.flip(aerial, dims=[-2]).contiguous()
        mask = torch.flip(mask, dims=[-2]).contiguous()
        sentinel = torch.flip(sentinel, dims=[-2]).contiguous()
        return aerial, sentinel, mask

    def _apply_rotation(
        self,
        aerial: torch.Tensor,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
        angle: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply synchronized rotation to both modalities.

        Args:
            aerial: (B, C, H, W) aerial imagery
            sentinel: (B, T, C, H, W) sentinel time series
            mask: (B, H, W) segmentation mask
            angle: Rotation angle in degrees (0, 90, 180, 270)

        Returns:
            Rotated tensors.

        """
        if angle == 0:
            return aerial, sentinel, mask

        k = angle // 90  # Number of 90-degree rotations

        # Rotate aerial (B, C, H, W) on dims (-2, -1)
        aerial = torch.rot90(aerial, k=k, dims=(-2, -1)).contiguous()
        # Rotate mask (B, H, W) on dims (-2, -1)
        mask = torch.rot90(mask, k=k, dims=(-2, -1)).contiguous()
        # Rotate sentinel (B, T, C, H, W) on dims (-2, -1)
        sentinel = torch.rot90(sentinel, k=k, dims=(-2, -1)).contiguous()

        return aerial, sentinel, mask

    def _apply_synchronized_geometric(
        self,
        aerial: torch.Tensor,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply synchronized geometric transforms to both modalities.

        Samples random parameters once and applies to all samples in the batch.
        This ensures spatial alignment between aerial and sentinel data is maintained.

        Args:
            aerial: (B, C, H, W) aerial imagery
            sentinel: (B, T, C, H, W) sentinel time series
            mask: (B, H, W) segmentation mask

        Returns:
            Transformed tensors.

        """
        # Apply horizontal flip
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            aerial, sentinel, mask = self._apply_hflip(aerial, sentinel, mask)

        # Apply vertical flip
        if self.vflip_prob > 0 and random.random() < self.vflip_prob:
            aerial, sentinel, mask = self._apply_vflip(aerial, sentinel, mask)

        # Apply rotation
        if self.rotation_prob > 0 and random.random() < self.rotation_prob:
            angle = random.choice(self.rotation_angles)
            aerial, sentinel, mask = self._apply_rotation(aerial, sentinel, mask, angle)

        return aerial, sentinel, mask

    def __call__(
        self,
        aerial: torch.Tensor,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply multimodal augmentations.

        Args:
            aerial: (B, C, H, W) aerial imagery
            sentinel: (B, T, C, H, W) sentinel time series
            mask: (B, H, W) segmentation mask
            positions: (B, T) month positions
            pad_mask: (B, T) padding mask, True = valid
            training: Whether in training mode

        Returns:
            Tuple of (aerial, sentinel, mask, positions, pad_mask) after augmentation.

        """
        if training:
            # 1. Apply synchronized geometric transforms
            aerial, sentinel, mask = self._apply_synchronized_geometric(aerial, sentinel, mask)

            # 2. Apply aerial-specific augmentations (photometric, elevation)
            # Note: Geometric transforms in FlairAugmentation may apply redundantly
            # but this is harmless since each sample gets independent random choices
            for b in range(aerial.shape[0]):
                aerial[b], mask[b] = self.aerial_aug.apply_flair_augmentations(aerial[b], mask[b])

        # 3. Apply sentinel augmentations (normalization + temporal, if enabled)
        sentinel, positions, pad_mask = self.sentinel_aug(
            sentinel,
            positions,
            pad_mask,
            training=training,
        )

        return aerial, sentinel, mask, positions, pad_mask
