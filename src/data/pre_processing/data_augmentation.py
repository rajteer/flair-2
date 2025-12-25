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
