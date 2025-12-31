"""Custom loss functions for segmentation models."""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


def _compute_weighted_dice_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    ignore_index: int | None = None,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute weighted Dice loss with per-class weighting and ignore_index support.

    Args:
        predictions: Raw logits of shape (N, C, H, W).
        targets: Ground truth of shape (N, H, W) with class indices.
        class_weights: Optional per-class weights tensor.
        ignore_index: Optional class index to ignore in the loss computation.
        smooth: Smoothing factor for numerical stability.

    Returns:
        Weighted Dice loss as a scalar tensor.

    """
    num_classes = predictions.shape[1]
    probs = F.softmax(predictions, dim=1)  # (N, C, H, W)

    if ignore_index is not None:
        valid_mask = targets != ignore_index  # (N, H, W)
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
    else:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        targets_clamped = targets

    targets_one_hot = F.one_hot(targets_clamped.long(), num_classes)  # (N, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

    valid_mask_expanded = valid_mask.unsqueeze(1).float()  # (N, 1, H, W)
    probs = probs * valid_mask_expanded
    targets_one_hot = targets_one_hot * valid_mask_expanded

    probs_flat = probs.view(probs.shape[0], num_classes, -1)
    targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

    intersection = (probs_flat * targets_flat).sum(dim=(0, 2))  # (C,)
    pred_sum = probs_flat.sum(dim=(0, 2))  # (C,)
    target_sum = targets_flat.sum(dim=(0, 2))  # (C,)

    dice_per_class = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    if class_weights is not None:
        weights = class_weights.to(dice_per_class.device)
        weights = weights / (weights.sum() + 1e-8)
        weighted_dice = (dice_per_class * weights).sum()
    else:
        weighted_dice = dice_per_class.mean()

    return 1.0 - weighted_dice


class CombinedDiceFocalLoss(nn.Module):
    """Combined loss that weights Dice and Focal losses with optional class weights."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        class_weights: list | tuple | torch.Tensor | None = None,
        dice_kwargs: dict | None = None,
        focal_kwargs: dict | None = None,
        smooth: float = 1e-6,
    ) -> None:
        """Initialize the combined Dice + Focal loss module.

        Args:
            dice_weight: Weight for the Dice loss component.
            focal_weight: Weight for the Focal loss component.
            class_weights: Optional per-class weights for handling class imbalance.
            dice_kwargs: Optional kwargs for the Dice loss constructor.
            focal_kwargs: Optional kwargs for the Focal loss constructor.
            smooth: Smoothing factor for Dice loss numerical stability.

        """
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            else:
                class_weights = class_weights.float()
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", None)

        dice_params = dice_kwargs or {"mode": "multiclass"}
        self.ignore_index = dice_params.get("ignore_index", None)
        self.dice_loss = smp.losses.DiceLoss(**dice_params)

        focal_params = {"mode": "multiclass"}  # Default
        if focal_kwargs:
            focal_params.update(focal_kwargs)  # Merge user params
        self.focal_loss = smp.losses.FocalLoss(**focal_params)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss value.

        Args:
            predictions: Raw logits of shape (N, C, H, W).
            targets: Ground truth of shape (N, H, W) with class indices.

        Returns:
            Combined weighted loss as a scalar tensor.

        """
        # Use weighted Dice if class weights provided, otherwise use SMP's DiceLoss
        if self.class_weights is not None:
            dice_loss_value = _compute_weighted_dice_loss(
                predictions,
                targets,
                class_weights=self.class_weights,
                ignore_index=self.ignore_index,
                smooth=self.smooth,
            )
        else:
            dice_loss_value = self.dice_loss(predictions, targets)

        focal_loss_value = self.focal_loss(predictions, targets)

        return self.dice_weight * dice_loss_value + self.focal_weight * focal_loss_value


class WeightedCrossEntropyDiceLoss(nn.Module):
    """Weighted Cross-Entropy + Weighted Dice loss.

    Combines a (optionally weighted) cross-entropy loss with a weighted Dice loss.
    Both components apply class weights to handle class imbalance.

    Notes:
    - `class_weights` may be a list, tuple or `torch.Tensor`. It will be
      converted to a tensor on the same device as the predictions at
      runtime to avoid device-mismatch issues.
    - `ce_kwargs` allows customization of the cross-entropy component.
    - The Dice component computes per-class Dice scores and weights them
      before averaging, giving minority classes more influence.

    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list | tuple | torch.Tensor | None = None,
        ce_kwargs: dict | None = None,
        dice_kwargs: dict | None = None,
        smooth: float = 1e-6,
    ) -> None:
        """Initialize the weighted Cross-Entropy + Dice loss module.

        Args:
            ce_weight: Weight for the Cross-Entropy loss component.
            dice_weight: Weight for the Dice loss component.
            class_weights: Optional per-class weights for both CE and Dice.
            ce_kwargs: Optional kwargs for the Cross-Entropy loss.
            dice_kwargs: Optional kwargs for Dice loss (used to extract ignore_index).
            smooth: Smoothing factor for Dice loss numerical stability.

        """
        super().__init__()

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = smooth

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            else:
                class_weights = class_weights.float()
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", None)

        self.ce_kwargs = ce_kwargs or {}
        dice_params = dice_kwargs or {}
        self.ignore_index = dice_params.get("ignore_index", self.ce_kwargs.get("ignore_index"))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted combination of cross-entropy and Dice loss.

        Args:
            predictions: Model predictions of shape (N, C, H, W) where N is batch size,
                C is number of classes, and H, W are spatial dimensions.
                **Must be raw logits (not softmax probabilities)** as both
                F.cross_entropy and the Dice computation apply softmax internally.
            targets: Ground truth labels of shape (N, H, W) with class indices.

        Returns:
            Combined loss value as a scalar tensor.

        Notes:
            - Cross-entropy loss applies class weights and respects ignore_index.
            - Dice loss applies class weights via weighted averaging of per-class scores.
            - Both losses expect raw logits and handle softmax internally.

        """
        ce_loss = F.cross_entropy(
            predictions,
            targets.long(),
            weight=self.class_weights,
            **self.ce_kwargs,
        )

        dice_loss = _compute_weighted_dice_loss(
            predictions,
            targets,
            class_weights=self.class_weights,
            ignore_index=self.ignore_index,
            smooth=self.smooth,
        )

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
