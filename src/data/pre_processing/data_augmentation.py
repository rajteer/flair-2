import random

import torch


class FlairAugmentation:

    def __init__(
        self,
        augmentation_config: dict,
        *,
        clamp: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        """Create a `FlairAugmentation` instance.

        Example augmentation_config:

        {
          "hflip": {"prob": 0.5},
          "vflip": {"prob": 0.5},
          "rotation": {"prob": 0.5, "angles": [0,90,180,270]},
          "contrast": {"prob": 0.5, "range": (0.8, 1.2)},
          "brightness": {"prob": 0.5, "range": (0.8, 1.2)}
        }

        Args:
            augmentation_config: Configuration dict for augmentations.
            clamp: If True, clamp adjusted images to [clamp_min, clamp_max].
            clamp_min: Minimum clamp value.
            clamp_max: Maximum clamp value.

        """
        self.augmentation_config = augmentation_config
        self.clamp = clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _adjust_contrast_tensor(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        # image shape (C,H,W) - operate per-channel relative to channel mean
        # do arithmetic in float32 for stability
        orig_dtype = image.dtype
        image_f = image.to(torch.float32)
        mean = image_f.mean(dim=(1, 2), keepdim=True)  # (C,1,1)
        out = (image_f - mean) * factor + mean
        if self.clamp:
            out = out.clamp(self.clamp_min, self.clamp_max)
        return out.to(orig_dtype)

    def _adjust_brightness_tensor(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        orig_dtype = image.dtype
        image_f = image.to(torch.float32)
        out = image_f * factor
        if self.clamp:
            out = out.clamp(self.clamp_min, self.clamp_max)
        return out.to(orig_dtype)

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

        aug_configs = {
            "hflip": lambda img: torch.flip(img, dims=[-1]),  # flip width (W)
            "vflip": lambda img: torch.flip(img, dims=[-2]),  # flip height (H)
            "rotation": lambda img, k: torch.rot90(img, k=k, dims=(-2, -1)),
            "contrast": lambda img, factor: self._adjust_contrast_tensor(img, factor),
            "brightness": lambda img, factor: self._adjust_brightness_tensor(img, factor),
        }

        for aug_name, func in aug_configs.items():
            aug = self.augmentation_config.get(aug_name, {})
            prob = aug.get("prob", 0)
            if random.random() < prob:
                if aug_name == "rotation":
                    angle = random.choice(aug.get("angles", [0, 90, 180, 270]))
                    k = (angle // 90) % 4
                    if k != 0:
                        image = func(image, k=k)
                        mask = func(mask, k=k)
                elif aug_name in ["contrast", "brightness"]:
                    min_val, max_val = aug.get("range", (0.8, 1.2))
                    factor = float(random.uniform(min_val, max_val))
                    image = func(image, factor)
                else:
                    image = func(image)
                    mask = func(mask)



        return image, mask

    def __call__(self, images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # batch or single sample support
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
