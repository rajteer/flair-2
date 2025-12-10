import random
import numpy as np

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
        orig_dtype = image.dtype
        image_f = image.to(torch.float32)

        img_optical = image_f[:4, ...]
        img_rest = image_f[4:, ...]

        mean = img_optical.mean(dim=(1, 2), keepdim=True)  # (C,1,1)
        out_optical = (img_optical - mean) * factor + mean

        if self.clamp:
            out_optical = out_optical.clamp(self.clamp_min, self.clamp_max)

        out = torch.cat([out_optical, img_rest], dim=0)
        return out.to(orig_dtype)

    def _adjust_brightness_tensor(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        orig_dtype = image.dtype
        image_f = image.to(torch.float32)

        img_optical = image_f[:4, ...]
        img_rest = image_f[4:, ...]

        out_optical = img_optical * factor

        if self.clamp:
            out_optical = out_optical.clamp(self.clamp_min, self.clamp_max)

        out = torch.cat([out_optical, img_rest], dim=0)
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


class CutMix:
    """CutMix augmentation: https://arxiv.org/abs/1905.04899."""

    def __init__(self, prob: float = 0.5, beta: float = 1.0) -> None:
        """Initialize CutMix.

        Args:
            prob: Probability of applying CutMix.
            beta: Hyperparameter for Beta distribution.
        """
        self.prob = prob
        self.beta = beta

    def _rand_bbox(self, size: tuple[int, int], lam: float) -> tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = torch.tensor(int(W * cut_rat))
        cut_h = torch.tensor(int(H * cut_rat))

        cx = torch.randint(W, (1,))
        cy = torch.randint(H, (1,))

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        return int(bbx1), int(bby1), int(bbx2), int(bby2)

    def __call__(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix to a batch of images and masks."""
        if random.random() < self.prob:
            lam = np.random.beta(self.beta, self.beta)
            batch_size, _, H, W = images.shape

            rand_index = torch.randperm(batch_size)

            bbx1, bby1, bbx2, bby2 = self._rand_bbox((W, H), lam)

            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

            if masks.ndim == 3:  # (B, H, W)
                masks[:, bbx1:bbx2, bby1:bby2] = masks[rand_index, bbx1:bbx2, bby1:bby2]
            else:  # (B, C, H, W)
                masks[:, :, bbx1:bbx2, bby1:bby2] = masks[rand_index, :, bbx1:bbx2, bby1:bby2]

        return images, masks
