import random

import torch
import torchvision.transforms.functional as TF


class FlairAugmentation:
    def __init__(self, augmentation_prob: float = 0.5):
        """
        Initialize the FLAIR-style data augmentation class.

        Args:
            augmentation_prob: Probability of applying augmentations. Default is 0.5.
        """
        self.augmentation_prob = augmentation_prob

    def apply_flair_augmentations(self, image: torch.Tensor, mask: torch.Tensor,
                                  augmentation_prob: float = 0.5) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Apply FLAIR-style data augmentations to an image and its corresponding mask.

        Args:
            image: A tensor representing the input image (C, H, W).
            mask: A tensor representing the segmentation mask (C, H, W).
            augmentation_prob: Probability of applying augmentations. Default is 0.5.

        Returns:
            A tuple of augmented image and mask tensors.
        """
        if random.random() < augmentation_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < augmentation_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < augmentation_prob:
            rotation_angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, rotation_angle)
            mask = TF.rotate(mask, rotation_angle)

        return image, mask

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        return self.apply_flair_augmentations(image, mask, self.augmentation_prob)
