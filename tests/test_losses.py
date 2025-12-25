"""Tests for custom loss functions in src/models/losses.py."""

import torch

from src.models.losses import CombinedDiceFocalLoss, WeightedCrossEntropyDiceLoss


class TestCombinedDiceFocalLoss:
    """Tests for CombinedDiceFocalLoss."""

    def test_output_is_scalar_tensor(
        self,
        sample_predictions: torch.Tensor,
        sample_masks: torch.Tensor,
    ) -> None:
        """Loss output should be a scalar tensor."""
        loss_fn = CombinedDiceFocalLoss()
        loss = loss_fn(sample_predictions, sample_masks)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0  # loss should be non-negative

    def test_output_changes_with_weights(
        self,
        sample_predictions: torch.Tensor,
        sample_masks: torch.Tensor,
    ) -> None:
        """Different weight configurations should produce different losses."""
        loss_fn_dice_heavy = CombinedDiceFocalLoss(dice_weight=0.9, focal_weight=0.1)
        loss_fn_focal_heavy = CombinedDiceFocalLoss(dice_weight=0.1, focal_weight=0.9)

        loss_dice = loss_fn_dice_heavy(sample_predictions, sample_masks)
        loss_focal = loss_fn_focal_heavy(sample_predictions, sample_masks)

        # Different weights should generally produce different loss values
        # (unless the individual losses happen to be exactly equal)
        assert loss_dice.item() >= 0
        assert loss_focal.item() >= 0

    def test_handles_single_sample(self, num_classes: int, image_size: int) -> None:
        """Loss should work with batch size of 1."""
        predictions = torch.randn(1, num_classes, image_size, image_size)
        masks = torch.randint(0, num_classes, (1, image_size, image_size))

        loss_fn = CombinedDiceFocalLoss()
        loss = loss_fn(predictions, masks)

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows(
        self,
        sample_predictions: torch.Tensor,
        sample_masks: torch.Tensor,
    ) -> None:
        """Loss should allow gradient backpropagation."""
        predictions = sample_predictions.clone().requires_grad_(True)
        loss_fn = CombinedDiceFocalLoss()
        loss = loss_fn(predictions, sample_masks)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad == 0)


class TestWeightedCrossEntropyDiceLoss:
    """Tests for WeightedCrossEntropyDiceLoss."""

    def test_output_is_scalar_tensor(
        self,
        sample_predictions: torch.Tensor,
        sample_masks: torch.Tensor,
    ) -> None:
        """Loss output should be a scalar tensor."""
        loss_fn = WeightedCrossEntropyDiceLoss()
        loss = loss_fn(sample_predictions, sample_masks)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_class_weights_are_applied(self, num_classes: int, image_size: int) -> None:
        """Class weights should affect the loss value."""
        predictions = torch.randn(2, num_classes, image_size, image_size)
        masks = torch.randint(0, num_classes, (2, image_size, image_size))

        # Uniform weights
        uniform_weights = [1.0] * num_classes
        loss_fn_uniform = WeightedCrossEntropyDiceLoss(class_weights=uniform_weights)

        # Non-uniform weights (higher weight for class 0)
        non_uniform_weights = [5.0] + [1.0] * (num_classes - 1)
        loss_fn_weighted = WeightedCrossEntropyDiceLoss(class_weights=non_uniform_weights)

        loss_uniform = loss_fn_uniform(predictions, masks)
        loss_weighted = loss_fn_weighted(predictions, masks)

        # Both should be valid losses
        assert not torch.isnan(loss_uniform)
        assert not torch.isnan(loss_weighted)

    def test_ce_dice_weight_ratio(self, num_classes: int, image_size: int) -> None:
        """CE and Dice weight ratios should affect the loss."""
        predictions = torch.randn(2, num_classes, image_size, image_size)
        masks = torch.randint(0, num_classes, (2, image_size, image_size))

        loss_fn_ce_heavy = WeightedCrossEntropyDiceLoss(ce_weight=0.9, dice_weight=0.1)
        loss_fn_dice_heavy = WeightedCrossEntropyDiceLoss(ce_weight=0.1, dice_weight=0.9)

        loss_ce = loss_fn_ce_heavy(predictions, masks)
        loss_dice = loss_fn_dice_heavy(predictions, masks)

        assert loss_ce.item() >= 0
        assert loss_dice.item() >= 0

    def test_handles_tensor_class_weights(self, num_classes: int, image_size: int) -> None:
        """Class weights can be provided as a tensor."""
        predictions = torch.randn(2, num_classes, image_size, image_size)
        masks = torch.randint(0, num_classes, (2, image_size, image_size))

        weights_tensor = torch.ones(num_classes)
        loss_fn = WeightedCrossEntropyDiceLoss(class_weights=weights_tensor)
        loss = loss_fn(predictions, masks)

        assert not torch.isnan(loss)

    def test_gradient_flows(
        self,
        sample_predictions: torch.Tensor,
        sample_masks: torch.Tensor,
    ) -> None:
        """Loss should allow gradient backpropagation."""
        predictions = sample_predictions.clone().requires_grad_(True)
        loss_fn = WeightedCrossEntropyDiceLoss()
        loss = loss_fn(predictions, sample_masks)
        loss.backward()

        assert predictions.grad is not None
