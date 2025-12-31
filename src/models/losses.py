"""Custom loss functions for segmentation models."""

import segmentation_models_pytorch as smp
import torch
from torch import nn


class CombinedDiceFocalLoss(nn.Module):
    """Combined loss that weights Dice and Focal losses."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        dice_kwargs: dict | None = None,
        focal_kwargs: dict | None = None,
    ) -> None:
        """Initialize the combined Dice + Focal loss module.

        Args:
            dice_weight: Weight for the Dice loss component.
            focal_weight: Weight for the Focal loss component.
            dice_kwargs: Optional kwargs for the Dice loss constructor.
            focal_kwargs: Optional kwargs for the Focal loss constructor.

        """
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        dice_params = dice_kwargs or {"mode": "multiclass"}
        self.dice_loss = smp.losses.DiceLoss(**dice_params)

        focal_params = focal_kwargs or {"mode": "multiclass"}
        self.focal_loss = smp.losses.FocalLoss(**focal_params)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss value."""
        dice_loss_value = self.dice_loss(predictions, targets)
        focal_loss_value = self.focal_loss(predictions, targets)

        return self.dice_weight * dice_loss_value + self.focal_weight * focal_loss_value
