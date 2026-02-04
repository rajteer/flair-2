"""Sentinel-2 temporal data augmentation for multimodal training.

This module provides augmentations specifically designed for Sentinel-2 time series data,
including normalization, random temporal dropout, and temporal shifts.
"""

from __future__ import annotations

import random
from typing import Any

import torch


class SentinelAugmentation:
    """Apply data augmentations to Sentinel-2 temporal data.

    Supports:
        - Channel normalization (mean/std)
        - Random temporal dropout (drop timesteps)
        - Temporal shift (shift month positions)

    """

    def __init__(self, sentinel_config: dict[str, Any]) -> None:
        """Initialize Sentinel augmentation.

        Args:
            sentinel_config: Configuration dict from data.sentinel_augmentation,
                containing:
                - enabled: bool
                - normalization: dict with mean/std lists
                - temporal.random_drop: dict with enabled, max_drop_ratio
                - temporal.temporal_shift: dict with enabled, max_shift

        """
        self.enabled = sentinel_config.get("enabled", False)

        # Parse normalization config
        norm_cfg = sentinel_config.get("normalization", {})
        self.mean = torch.tensor(norm_cfg.get("mean", []), dtype=torch.float32)
        self.std = torch.tensor(norm_cfg.get("std", []), dtype=torch.float32)
        self.apply_normalization = len(self.mean) > 0 and len(self.std) > 0

        # Parse temporal augmentation config
        temporal_cfg = sentinel_config.get("temporal", {})

        drop_cfg = temporal_cfg.get("random_drop", {})
        self.random_drop_enabled = drop_cfg.get("enabled", False)
        self.max_drop_ratio = drop_cfg.get("max_drop_ratio", 0.3)

        shift_cfg = temporal_cfg.get("temporal_shift", {})
        self.temporal_shift_enabled = shift_cfg.get("enabled", False)
        self.max_shift = shift_cfg.get("max_shift", 1)

    def normalize(self, sentinel_data: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise normalization to Sentinel data.

        Args:
            sentinel_data: Tensor of shape (B, T, C, H, W) or (T, C, H, W)

        Returns:
            Normalized tensor with same shape.

        """
        if not self.apply_normalization:
            return sentinel_data

        # Ensure mean/std are on the same device
        mean = self.mean.to(sentinel_data.device)
        std = self.std.to(sentinel_data.device)

        # Handle both batched and unbatched inputs
        if sentinel_data.ndim == 5:
            # (B, T, C, H, W) -> normalize over C dimension (index 2)
            mean = mean.view(1, 1, -1, 1, 1)
            std = std.view(1, 1, -1, 1, 1)
        else:
            # (T, C, H, W) -> normalize over C dimension (index 1)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)

        return (sentinel_data - mean) / std

    def random_temporal_drop(
        self,
        sentinel_data: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly drop timesteps from the Sentinel time series.

        This augmentation helps the model learn to handle incomplete temporal data.
        Only valid (non-padded) timesteps can be dropped. At least one timestep
        is always preserved.

        Args:
            sentinel_data: Tensor of shape (B, T, C, H, W)
            positions: Tensor of shape (B, T) with month positions
            pad_mask: Tensor of shape (B, T), True = valid timestep

        Returns:
            Tuple of (sentinel_data, positions, pad_mask) with dropped timesteps
            marked as padding (pad_mask set to False).

        """
        if not self.random_drop_enabled:
            return sentinel_data, positions, pad_mask

        batch_size = sentinel_data.shape[0]
        device = sentinel_data.device

        # Work on a copy of pad_mask
        new_pad_mask = pad_mask.clone()

        for b in range(batch_size):
            valid_indices = torch.where(pad_mask[b])[0]
            num_valid = len(valid_indices)

            if num_valid <= 1:
                continue  # Keep at least one timestep

            # Calculate number of timesteps to drop
            max_to_drop = int(num_valid * self.max_drop_ratio)
            if max_to_drop == 0:
                continue

            num_to_drop = random.randint(0, max_to_drop)
            if num_to_drop == 0:
                continue

            # Ensure we keep at least one timestep
            num_to_drop = min(num_to_drop, num_valid - 1)

            # Randomly select indices to drop
            drop_indices = random.sample(valid_indices.tolist(), num_to_drop)

            # Mark dropped timesteps as padding
            for idx in drop_indices:
                new_pad_mask[b, idx] = False

        return sentinel_data, positions, new_pad_mask

    def temporal_shift(
        self,
        positions: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Shift temporal positions to simulate temporal misalignment.

        This augmentation helps the model become robust to slight temporal
        mismatches between aerial and Sentinel acquisition dates.

        Args:
            positions: Tensor of shape (B, T) with month positions (0-11)
            pad_mask: Tensor of shape (B, T), True = valid timestep

        Returns:
            Shifted positions tensor with same shape. Shifts are applied
            per-batch with wraparound (position 0 - 1 -> 11).

        """
        if not self.temporal_shift_enabled:
            return positions

        batch_size = positions.shape[0]
        new_positions = positions.clone()

        for b in range(batch_size):
            # Random shift for this sample
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift == 0:
                continue

            # Apply shift with wraparound (months are 0-11)
            valid_mask = pad_mask[b]
            new_positions[b, valid_mask] = (positions[b, valid_mask] + shift) % 12

        return new_positions

    def __call__(
        self,
        sentinel_data: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Sentinel augmentations.

        Args:
            sentinel_data: Tensor of shape (B, T, C, H, W)
            positions: Tensor of shape (B, T) with month positions
            pad_mask: Tensor of shape (B, T), True = valid timestep
            training: Whether in training mode. Temporal augmentations
                are only applied during training.

        Returns:
            Tuple of (sentinel_data, positions, pad_mask) after augmentation.

        """
        if not self.enabled:
            return sentinel_data, positions, pad_mask

        # Normalization is always applied (train and eval)
        sentinel_data = self.normalize(sentinel_data)

        # Temporal augmentations only during training
        if training:
            sentinel_data, positions, pad_mask = self.random_temporal_drop(
                sentinel_data, positions, pad_mask
            )
            positions = self.temporal_shift(positions, pad_mask)

        return sentinel_data, positions, pad_mask
