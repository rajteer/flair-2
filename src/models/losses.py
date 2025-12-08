"""Custom loss functions for segmentation models."""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
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


class WeightedCrossEntropyDiceLoss(nn.Module):
    """Weighted Cross-Entropy + Dice loss.

    Combines a (optionally weighted) cross-entropy loss with a Dice loss.

    Notes:
    - `class_weights` may be a list, tuple or `torch.Tensor`. It will be
      converted to a tensor on the same device as the predictions at
      runtime to avoid device-mismatch issues.
    - `ignore_index` will be passed to the cross-entropy computation.

    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list | tuple | torch.Tensor | None = None,
        ignore_index: int | None = None,
        dice_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self._raw_class_weights = class_weights
        self.ignore_index = ignore_index

        # Default Dice loss parameters with epsilon for numerical stability
        dice_params = dice_kwargs or {"mode": "multiclass", "smooth": 1e-6}
        # Ensure smooth (epsilon) is set if not provided
        if "smooth" not in dice_params:
            dice_params["smooth"] = 1e-6
        self.dice_loss = smp.losses.DiceLoss(**dice_params)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted combination of cross-entropy and Dice loss.

        Args:
            predictions: Model predictions of shape (N, C, H, W) where N is batch size,
                C is number of classes, and H, W are spatial dimensions.
                **Must be raw logits (not softmax probabilities)** as both
                F.cross_entropy and smp.losses.DiceLoss apply softmax internally.
            targets: Ground truth labels of shape (N, H, W) with class indices.

        Returns:
            Combined loss value as a scalar tensor.

        Notes:
            - Cross-entropy loss applies class weights and respects ignore_index.
            - Dice loss includes epsilon (smooth parameter) for numerical stability.
            - Both losses expect raw logits and handle softmax internally.

        """
        weight = None
        if self._raw_class_weights is not None:
            if isinstance(self._raw_class_weights, torch.Tensor):
                weight = self._raw_class_weights.to(predictions.device).float()
            else:
                weight = torch.tensor(
                    self._raw_class_weights, dtype=torch.float, device=predictions.device,
                )

        ce_loss = F.cross_entropy(
            predictions, targets.long(), weight=weight, ignore_index=self.ignore_index,
        )

        dice_loss = self.dice_loss(predictions, targets)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
