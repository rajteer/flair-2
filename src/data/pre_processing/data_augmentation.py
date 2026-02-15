from __future__ import annotations

import random
from typing import Any

import torch


class FlairAugmentation:
    """Apply data augmentations to aerial imagery with optional elevation channel."""

    def __init__(self, data_config: dict[str, Any]) -> None:
        """Create a `FlairAugmentation` instance.

        Args:
            data_config: Full data configuration dict containing:
                - data_augmentation.clamp: Whether to clamp values
                - data_augmentation.augmentations: Dict of augmentation configs
                - normalization: Mean/std for calculating clamp bounds
                - selected_channels: List of channel indices

        """
        aug_config = data_config.get("data_augmentation", {})
        self.augmentation_config = aug_config.get("augmentations", {})
        self.clamp = aug_config.get("clamp", False)

        self.channel_clamp_min = None
        self.channel_clamp_max = None
        self.elevation_normalization_params = None

        self._init_normalization_params(data_config)
        self._init_aug_funcs()

    def _init_aug_funcs(self) -> None:
        """Initialize augmentation function lookup table with metadata."""
        self._aug_funcs = {
            "hflip": lambda img: torch.flip(img, dims=[-1]),
            "vflip": lambda img: torch.flip(img, dims=[-2]),
            "rotation": lambda img, k: torch.rot90(img, k=k, dims=(-2, -1)),
            "contrast": lambda img, val: self._adjust_contrast_tensor(img, val),
            "brightness": lambda img, val: self._adjust_brightness_tensor(img, val),
            "elevation_shift": lambda img, val: self._adjust_elevation_shift(img, val),
            "elevation_scale": lambda img, val: self._adjust_elevation_scale(img, val),
            "elevation_dropout": lambda img, val: self._adjust_elevation_dropout(img, val),
        }

        self._aug_meta = {
            "hflip": (True, None, None),
            "vflip": (True, None, None),
            "rotation": (True, "angles", [0, 90, 180, 270]),
            "contrast": (False, "range", (0.8, 1.2)),
            "brightness": (False, "range", (0.8, 1.2)),
            "elevation_shift": (False, "range", (-10.0, 10.0)),
            "elevation_scale": (False, "range", (0.9, 1.1)),
            "elevation_dropout": (False, "p", 0.1),
        }

    def _sample_params(self, aug_name: str, aug_cfg: dict) -> dict:
        """Sample random parameters for an augmentation based on its config."""
        _, param_type, default = self._aug_meta.get(aug_name, (True, None, None))

        if param_type is None:
            return {}
        if param_type == "range":
            min_val, max_val = aug_cfg.get("range", default)
            return {"val": float(random.uniform(min_val, max_val))}
        if param_type == "angles":
            angle = random.choice(aug_cfg.get("angles", default))
            k = (angle // 90) % 4
            return {"k": k} if k != 0 else {}
        if param_type == "p":
            return {"val": aug_cfg.get("p", default)}
        return {}

    def _init_normalization_params(self, data_config: dict[str, Any]) -> None:
        """Initialize clamp bounds and elevation params from normalization config."""
        norm_cfg = data_config.get("normalization")
        if norm_cfg is None:
            return

        selected_channels = data_config.get("selected_channels")
        means = norm_cfg.get("mean")
        stds = norm_cfg.get("std")

        if selected_channels and means and stds:
            n_channels = len(selected_channels)
            t_mean = torch.tensor(means[:n_channels]).view(-1, 1, 1)
            t_std = torch.tensor(stds[:n_channels]).view(-1, 1, 1)
            self.channel_clamp_min = (0.0 - t_mean) / t_std
            self.channel_clamp_max = (1.0 - t_mean) / t_std

        elev_range = norm_cfg.get("elevation_range")
        elev_idx = norm_cfg.get("elevation_channel_index")

        if (
            elev_range is not None
            and elev_idx is not None
            and selected_channels is not None
            and elev_idx in selected_channels
        ):
            pos = selected_channels.index(elev_idx)
            self.elevation_normalization_params = {
                "mean": norm_cfg["mean"][pos],
                "std": norm_cfg["std"][pos],
                "raw_range": tuple(elev_range),
            }

    def _adjust_contrast_tensor(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        orig_dtype = image.dtype
        image_f = image.to(torch.float32)

        img_optical = image_f[:4, ...]
        img_rest = image_f[4:, ...]

        mean = img_optical.mean(dim=(1, 2), keepdim=True)  # (C,1,1)
        out_optical = (img_optical - mean) * factor + mean

        out = torch.cat([out_optical, img_rest], dim=0)
        return out.to(orig_dtype)

    def _adjust_brightness_tensor(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        orig_dtype = image.dtype
        image_f = image.to(torch.float32)

        img_optical = image_f[:4, ...]
        img_rest = image_f[4:, ...]

        out_optical = img_optical * factor

        out = torch.cat([out_optical, img_rest], dim=0)
        return out.to(orig_dtype)

    def _adjust_elevation_shift(self, image: torch.Tensor, pct: float) -> torch.Tensor:
        """Vertical Shift: Simulates variations in the base terrain level.

        Args:
            image: Input tensor of shape (C, H, W).
            pct: Percentage of elevation range to shift by (e.g., 10.0 means +10%).

        Returns:
            Image with shifted elevation channel.

        """
        if image.shape[0] < 5:  # noqa: PLR2004
            return image

        elev = image[4].to(torch.float32)

        if self.elevation_normalization_params:
            params = self.elevation_normalization_params
            std = params["std"]
            elev = elev + (pct / 100.0) / (std + 1e-8)
        else:
            data_range = elev.max() - elev.min() + 1e-8
            elev = elev + (pct / 100.0) * data_range

        out = image.clone()
        out[4] = elev.to(image.dtype)
        return out

    def _adjust_elevation_scale(self, image: torch.Tensor, alpha: float) -> torch.Tensor:
        """Vertical Scaling: Simulates sensor calibration differences (Z_new = Z * alpha)."""
        if image.shape[0] < 5:  # noqa: PLR2004
            return image

        elev = image[4].to(torch.float32)

        if self.elevation_normalization_params:
            params = self.elevation_normalization_params
            raw_min, raw_max = params["raw_range"]
            raw_range = raw_max - raw_min
            mu = params["mean"]
            std = params["std"]
            shift_term = ((alpha - 1.0) / (std + 1e-8)) * (mu + (raw_min / (raw_range + 1e-8)))
            elev = alpha * elev + shift_term
        else:
            elev = elev * alpha

        out = image.clone()
        out[4] = elev.to(image.dtype)
        return out

    def _adjust_elevation_dropout(self, image: torch.Tensor, p: float) -> torch.Tensor:
        """Sensor Dropout: Simulates data acquisition failures (Z_new = 0 for some pixels)."""
        if image.shape[0] < 5:  # noqa: PLR2004
            return image

        elev = image[4].to(torch.float32)
        mask = (torch.rand(elev.shape, device=image.device) > p).float()

        if self.elevation_normalization_params:
            params = self.elevation_normalization_params
            mu = params["mean"]
            std = params["std"]
            dropout_val = -mu / (std + 1e-8)
            elev = elev * mask + dropout_val * (1.0 - mask)
        else:
            elev = elev * mask

        out = image.clone()
        out[4] = elev.to(image.dtype)
        return out

    def apply_flair_augmentations(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply configured augmentations to a single image and mask.

        Args:
            image: Tensor (C,H,W).
            mask: Tensor (C,H,W) or (H,W) - label map or probabilistic mask.

        Returns:
            Tuple of (image, mask) after augmentations.

        """
        image = image.contiguous()
        mask = mask.contiguous()

        for aug_name, func in self._aug_funcs.items():
            aug_cfg = self.augmentation_config.get(aug_name, {})
            prob = aug_cfg.get("prob", 0)
            if random.random() >= prob:
                continue

            params = self._sample_params(aug_name, aug_cfg)
            if not params and self._aug_meta[aug_name][1] == "angles":
                # rotation with k=0 means no rotation
                continue

            applies_to_mask = self._aug_meta[aug_name][0]
            if params:
                image = func(image, **params)
                if applies_to_mask:
                    mask = func(mask, **params)
            else:
                image = func(image)
                if applies_to_mask:
                    mask = func(mask)

        if self.clamp and self.channel_clamp_min is not None and self.channel_clamp_max is not None:
            cmin = self.channel_clamp_min.to(image.device)
            cmax = self.channel_clamp_max.to(image.device)
            n_channels = min(image.shape[0], cmin.shape[0])
            image[:n_channels] = torch.max(image[:n_channels], cmin[:n_channels])
            image[:n_channels] = torch.min(image[:n_channels], cmax[:n_channels])

        return image, mask

    def __call__(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.dim() == 4:  # batch (B,C,H,W)
            batch_size = images.size(0)
            out_images = []
            out_masks = []
            for i in range(batch_size):
                img, msk = self.apply_flair_augmentations(images[i], masks[i])
                out_images.append(img)
                out_masks.append(msk)
            return torch.stack(out_images), torch.stack(out_masks)
        return self.apply_flair_augmentations(images, masks)


class SentinelAugmentation:
    """Apply geometric augmentations to temporal Sentinel satellite imagery.

    Applies the same geometric transform consistently across all timesteps
    to preserve temporal coherence. Supports horizontal/vertical flip and
    90-degree rotations.

    Sentinel data shape: (T, C, H, W) where T is number of timesteps.
    Mask shape: (H, W) or (512, 512) for full resolution.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        rotation_angles: list[int] | None = None,
        channel_dropout_prob: float = 0.0,
        max_channels_drop: int = 2,
        gaussian_noise_std: float = 0.0,
    ) -> None:
        """Initialize Sentinel augmentation.

        Args:
            hflip_prob: Probability of horizontal flip.
            vflip_prob: Probability of vertical flip.
            rotation_prob: Probability of rotation.
            rotation_angles: List of rotation angles in degrees (must be 0, 90, 180, or 270).
            channel_dropout_prob: Probability of dropping random channels (bands).
            max_channels_drop: Maximum number of channels to drop (1-3).
            gaussian_noise_std: Standard deviation of Gaussian noise to add (0 = disabled).

        """
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_prob = rotation_prob
        self.rotation_angles = rotation_angles or [0, 90, 180, 270]
        self.channel_dropout_prob = channel_dropout_prob
        self.max_channels_drop = max_channels_drop
        self.gaussian_noise_std = gaussian_noise_std

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> "SentinelAugmentation":
        """Create SentinelAugmentation from data config.

        Args:
            data_config: Data configuration dict containing sentinel_augmentation section.

        Returns:
            Configured SentinelAugmentation instance.

        """
        aug_config = data_config.get("sentinel_augmentation", {})
        return cls(
            hflip_prob=aug_config.get("hflip_prob", 0.5),
            vflip_prob=aug_config.get("vflip_prob", 0.5),
            rotation_prob=aug_config.get("rotation_prob", 0.5),
            rotation_angles=aug_config.get("rotation_angles", [0, 90, 180, 270]),
        )

    def _apply_hflip(
        self,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply horizontal flip to all timesteps and mask."""
        # Sentinel: (T, C, H, W) -> flip on last dim (W)
        sentinel = torch.flip(sentinel, dims=[-1])
        # Mask: (H, W) or larger -> flip on last dim
        mask = torch.flip(mask, dims=[-1])
        return sentinel, mask

    def _apply_vflip(
        self,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply vertical flip to all timesteps and mask."""
        # Sentinel: (T, C, H, W) -> flip on second-to-last dim (H)
        sentinel = torch.flip(sentinel, dims=[-2])
        # Mask: (H, W) -> flip on second-to-last dim
        mask = torch.flip(mask, dims=[-2])
        return sentinel, mask

    def _apply_rotation(
        self,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply k*90 degree rotation to all timesteps and mask.

        Args:
            sentinel: Tensor of shape (T, C, H, W).
            mask: Tensor of shape (H, W).
            k: Number of 90-degree rotations (0, 1, 2, or 3).

        """
        if k == 0:
            return sentinel, mask

        # Rotate each timestep
        sentinel = torch.rot90(sentinel, k=k, dims=(-2, -1))
        mask = torch.rot90(mask, k=k, dims=(-2, -1))
        return sentinel, mask

    def __call__(
        self,
        sentinel: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random geometric augmentations to sentinel data and mask.

        Args:
            sentinel: Tensor of shape (T, C, H, W).
            mask: Tensor of shape (H, W).

        Returns:
            Tuple of (augmented_sentinel, augmented_mask).

        """
        sentinel = sentinel.contiguous()
        mask = mask.contiguous()

        if random.random() < self.hflip_prob:
            sentinel, mask = self._apply_hflip(sentinel, mask)

        if random.random() < self.vflip_prob:
            sentinel, mask = self._apply_vflip(sentinel, mask)

        if random.random() < self.rotation_prob:
            angle = random.choice(self.rotation_angles)
            k = (angle // 90) % 4
            sentinel, mask = self._apply_rotation(sentinel, mask, k)

        # Channel dropout - zero out random bands across all timesteps
        if self.channel_dropout_prob > 0 and random.random() < self.channel_dropout_prob:
            sentinel = self._apply_channel_dropout(sentinel)

        # Gaussian noise
        if self.gaussian_noise_std > 0:
            sentinel = self._apply_gaussian_noise(sentinel)

        return sentinel, mask

    def _apply_channel_dropout(self, sentinel: torch.Tensor) -> torch.Tensor:
        """Randomly zero out 1-max_channels_drop bands across all timesteps.

        Args:
            sentinel: Tensor of shape (T, C, H, W).

        Returns:
            Tensor with some channels zeroed out.

        """
        num_channels = sentinel.shape[1]
        n_drop = random.randint(1, min(self.max_channels_drop, num_channels - 1))
        channels_to_drop = random.sample(range(num_channels), n_drop)

        sentinel = sentinel.clone()
        for c in channels_to_drop:
            sentinel[:, c, :, :] = 0.0

        return sentinel

    def _apply_gaussian_noise(self, sentinel: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to all channels.

        Args:
            sentinel: Tensor of shape (T, C, H, W).

        Returns:
            Tensor with added noise.

        """
        noise = torch.randn_like(sentinel) * self.gaussian_noise_std
        return sentinel + noise
