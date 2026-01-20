"""Tests for model builder functions in src/models/model_builder.py."""

import pytest
import torch
from torch import nn, optim

from src.models.model_builder import (
    build_loss_function,
    build_lr_scheduler,
    build_model,
    build_optimizer,
)


def _mamba_available() -> bool:
    """Check if mamba_ssm is available."""
    try:
        import mamba_ssm  # noqa: F401

        return True
    except ImportError:
        return False


class TestBuildModel:
    """Tests for build_model function."""

    def test_builds_unet_model(self, in_channels: int, num_classes: int) -> None:
        """Should build a Unet model from SMP."""
        model = build_model(
            model_type="Unet",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
        )

        assert isinstance(model, nn.Module)

    def test_builds_fpn_model(self, in_channels: int, num_classes: int) -> None:
        """Should build an FPN model from SMP."""
        model = build_model(
            model_type="FPN",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
        )

        assert isinstance(model, nn.Module)

    def test_builds_unetformer_model(self, in_channels: int, num_classes: int) -> None:
        """Should build a UNetFormer model."""
        model = build_model(
            model_type="UNetFormer",
            encoder_name="swsl_resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
        )

        assert isinstance(model, nn.Module)

    @pytest.mark.skipif(
        not _mamba_available(),
        reason="mamba_ssm not available",
    )
    def test_builds_unetformer_with_samba(self, in_channels: int, num_classes: int) -> None:
        """Should build a UNetFormer model with Samba encoder."""
        model = build_model(
            model_type="UNetFormer",
            encoder_name="",
            in_channels=in_channels,
            n_classes=num_classes,
            model_config={
                "encoder_type": "samba",
                "samba_config": {
                    "embed_dims": [32, 64, 128, 256],
                    "depths": [1, 1, 1, 1],
                },
            },
        )

        assert isinstance(model, nn.Module)
        assert model.encoder_type == "samba"

    @pytest.mark.skipif(
        not _mamba_available(),
        reason="mamba_ssm not available",
    )
    def test_unetformer_samba_forward_pass(
        self,
        in_channels: int,
        num_classes: int,
        batch_size: int,
    ) -> None:
        """UNetFormer with Samba encoder should produce correct output shape."""
        model = build_model(
            model_type="UNetFormer",
            encoder_name="",
            in_channels=in_channels,
            n_classes=num_classes,
            model_config={
                "encoder_type": "samba",
                "samba_config": {
                    "embed_dims": [32, 64, 128, 256],
                    "depths": [1, 1, 1, 1],
                },
            },
        )
        model.eval()

        # Use smaller image size for faster testing
        h, w = 64, 64
        x = torch.randn(batch_size, in_channels, h, w)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes, h, w)

    def test_raises_for_unknown_model(self, in_channels: int, num_classes: int) -> None:
        """Should raise ValueError for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model(
                model_type="NonExistentModel",
                encoder_name="resnet18",
                in_channels=in_channels,
                n_classes=num_classes,
            )

    def test_forward_pass_produces_correct_shape(
        self,
        in_channels: int,
        num_classes: int,
        batch_size: int,
        image_size: int,
    ) -> None:
        """Model forward pass should produce output with correct shape."""
        model = build_model(
            model_type="Unet",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
        )
        model.eval()

        x = torch.randn(batch_size, in_channels, image_size, image_size)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes, image_size, image_size)

    def test_stochastic_depth_parameter(self, in_channels: int, num_classes: int) -> None:
        """Should accept stochastic_depth parameter for SMP models."""
        model = build_model(
            model_type="Unet",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
            stochastic_depth=0.1,
        )

        assert isinstance(model, nn.Module)

    def test_model_config_decoder_channels(self, in_channels: int, num_classes: int) -> None:
        """Should forward decoder_channels from model_config to SMP Unet."""
        model = build_model(
            model_type="Unet",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
            model_config={"decoder_channels": [128, 64, 32, 16, 8]},
        )

        assert isinstance(model, nn.Module)
        # Verify decoder channels were applied
        assert model.decoder.blocks[0].conv1[0].out_channels == 128

    def test_model_config_decoder_attention(self, in_channels: int, num_classes: int) -> None:
        """Should forward decoder_attention_type from model_config to SMP Unet."""
        model = build_model(
            model_type="Unet",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
            model_config={"decoder_attention_type": "scse"},
        )

        assert isinstance(model, nn.Module)
        # Verify attention blocks exist in decoder (SMP uses 'attention1' and 'attention2' for scse)
        assert hasattr(model.decoder.blocks[0], "attention1")

    def test_model_config_deeplabv3plus(self, in_channels: int, num_classes: int) -> None:
        """Should forward decoder_channels from model_config to DeepLabV3Plus."""
        model = build_model(
            model_type="DeepLabV3Plus",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
            model_config={"decoder_channels": 128},
        )

        assert isinstance(model, nn.Module)

    def test_model_config_segformer(self, in_channels: int, num_classes: int) -> None:
        """Should forward decoder_segmentation_channels from model_config to Segformer."""
        model = build_model(
            model_type="Segformer",
            encoder_name="mit_b0",
            in_channels=in_channels,
            n_classes=num_classes,
            model_config={"decoder_segmentation_channels": 384},
        )

        assert isinstance(model, nn.Module)

    def test_model_config_does_not_override_explicit_params(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        """Explicit parameters should not be overridden by model_config."""
        model = build_model(
            model_type="Unet",
            encoder_name="resnet18",
            in_channels=in_channels,
            n_classes=num_classes,
            stochastic_depth=0.2,
            model_config={"drop_path_rate": 0.5},  # Should not override stochastic_depth
        )

        assert isinstance(model, nn.Module)


class TestBuildOptimizer:
    """Tests for build_optimizer function."""

    @pytest.fixture
    def simple_model(self) -> nn.Module:
        """Create a simple model for optimizer testing."""
        return nn.Linear(10, 5)

    def test_builds_adam_optimizer(self, simple_model: nn.Module) -> None:
        """Should build Adam optimizer."""
        optimizer = build_optimizer(
            model=simple_model,
            optimizer_type="Adam",
            learning_rate=1e-3,
            betas=(0.9, 0.999),
        )

        assert isinstance(optimizer, optim.Adam)

    def test_builds_adamw_optimizer(self, simple_model: nn.Module) -> None:
        """Should build AdamW optimizer."""
        optimizer = build_optimizer(
            model=simple_model,
            optimizer_type="AdamW",
            learning_rate=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        assert isinstance(optimizer, optim.AdamW)

    def test_builds_sgd_optimizer(self, simple_model: nn.Module) -> None:
        """Should build SGD optimizer."""
        optimizer = build_optimizer(
            model=simple_model,
            optimizer_type="SGD",
            learning_rate=1e-2,
        )

        assert isinstance(optimizer, optim.SGD)

    def test_raises_for_unknown_optimizer(self, simple_model: nn.Module) -> None:
        """Should raise ValueError for unknown optimizer type."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            build_optimizer(
                model=simple_model,
                optimizer_type="NonExistentOptimizer",
                learning_rate=1e-3,
            )

    def test_applies_learning_rate(self, simple_model: nn.Module) -> None:
        """Should apply the specified learning rate."""
        lr = 0.005
        optimizer = build_optimizer(
            model=simple_model,
            optimizer_type="Adam",
            learning_rate=lr,
            betas=(0.9, 0.999),
        )

        assert optimizer.defaults["lr"] == lr


class TestBuildLRScheduler:
    """Tests for build_lr_scheduler function."""

    @pytest.fixture
    def optimizer(self) -> optim.Optimizer:
        """Create a simple optimizer for scheduler testing."""
        model = nn.Linear(10, 5)
        return optim.Adam(model.parameters(), lr=1e-3)

    def test_returns_none_when_config_is_none(self, optimizer: optim.Optimizer) -> None:
        """Should return None when scheduler_config is None."""
        scheduler = build_lr_scheduler(optimizer, scheduler_config=None)
        assert scheduler is None

    def test_builds_step_lr_scheduler(self, optimizer: optim.Optimizer) -> None:
        """Should build StepLR scheduler."""
        config = {"type": "StepLR", "args": {"step_size": 10, "gamma": 0.1}}
        scheduler = build_lr_scheduler(optimizer, scheduler_config=config)

        assert scheduler is not None
        assert isinstance(scheduler, optim.lr_scheduler.StepLR)

    def test_builds_cosine_annealing_scheduler(self, optimizer: optim.Optimizer) -> None:
        """Should build CosineAnnealingLR scheduler."""
        config = {"type": "CosineAnnealingLR", "args": {"T_max": 100}}
        scheduler = build_lr_scheduler(optimizer, scheduler_config=config)

        assert scheduler is not None
        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)

    def test_raises_for_unknown_scheduler(self, optimizer: optim.Optimizer) -> None:
        """Should raise ValueError for unknown scheduler type."""
        config = {"type": "NonExistentScheduler", "args": {}}
        with pytest.raises(ValueError, match="Unknown LR scheduler type"):
            build_lr_scheduler(optimizer, scheduler_config=config)


class TestBuildLossFunction:
    """Tests for build_loss_function function."""

    def test_builds_combined_dice_focal_loss(self) -> None:
        """Should build CombinedDiceFocalLoss."""
        loss_fn = build_loss_function("CombinedDiceFocalLoss")
        assert loss_fn is not None

    def test_builds_weighted_ce_dice_loss(self) -> None:
        """Should build WeightedCrossEntropyDiceLoss."""
        loss_fn = build_loss_function("WeightedCrossEntropyDiceLoss")
        assert loss_fn is not None

    def test_builds_smp_dice_loss(self) -> None:
        """Should build SMP DiceLoss."""
        loss_fn = build_loss_function("DiceLoss", kwargs={"mode": "multiclass"})
        assert loss_fn is not None

    def test_builds_cross_entropy_loss(self) -> None:
        """Should build PyTorch CrossEntropyLoss."""
        loss_fn = build_loss_function("CrossEntropyLoss")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_raises_for_unknown_loss(self) -> None:
        """Should raise ValueError for unknown loss type."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            build_loss_function("NonExistentLoss")

    def test_passes_kwargs_to_loss(self, num_classes: int) -> None:
        """Should pass kwargs to loss constructor."""
        weights = [1.0] * num_classes
        loss_fn = build_loss_function(
            "WeightedCrossEntropyDiceLoss",
            kwargs={"class_weights": weights},
        )
        assert loss_fn is not None


class TestBuildMultimodalModel:
    """Tests for MultimodalLateFusion model building."""

    def test_builds_multimodal_late_fusion(self, num_classes: int) -> None:
        """Should build MultimodalLateFusion model."""
        model = build_model(
            model_type="MultimodalLateFusion",
            encoder_name="resnet18",
            in_channels=5,
            n_classes=num_classes,
            model_config={
                "aerial_model_config": {"backbone_name": "resnet18"},
                "sentinel_model_config": {
                    "image_size": 10,
                    "patch_size": 2,
                    "max_seq_len": 12,
                    "dim": 64,
                },
            },
        )

        assert isinstance(model, nn.Module)
        assert hasattr(model, "aerial_model")
        assert hasattr(model, "sentinel_model")
        assert hasattr(model, "class_weights")

    def test_multimodal_forward_pass_shape(self, num_classes: int, batch_size: int) -> None:
        """Forward pass should produce correct output shape."""
        model = build_model(
            model_type="MultimodalLateFusion",
            encoder_name="resnet18",
            in_channels=5,
            n_classes=num_classes,
            model_config={
                "aerial_model_type": "Unet",  # Use SMP Unet for simpler testing
                "aerial_model_config": {"backbone_name": "resnet18"},
                "sentinel_model_config": {
                    "image_size": 10,
                    "patch_size": 2,
                    "max_seq_len": 12,
                    "dim": 64,
                },
            },
        )
        model.eval()

        h, w = 64, 64
        aerial = torch.randn(batch_size, 5, h, w)
        sentinel = torch.randn(batch_size, 8, 10, 10, 10)  # B, T, C, H, W
        positions = torch.arange(8).unsqueeze(0).repeat(batch_size, 1)

        with torch.no_grad():
            output = model(aerial, sentinel, batch_positions=positions, pad_mask=None)

        assert output.shape == (batch_size, num_classes, h, w)

    def test_per_class_weights_are_learnable(self, num_classes: int) -> None:
        """Per-class fusion weights should be nn.Parameter."""
        model = build_model(
            model_type="MultimodalLateFusion",
            encoder_name="resnet18",
            in_channels=5,
            n_classes=num_classes,
            model_config={
                "fusion_mode": "weighted",
                "aerial_model_config": {"backbone_name": "resnet18"},
                "sentinel_model_config": {
                    "image_size": 10,
                    "patch_size": 2,
                    "max_seq_len": 12,
                    "dim": 64,
                },
            },
        )

        assert isinstance(model.class_weights, nn.Parameter)
        assert model.class_weights.shape == (num_classes, 2)
        assert model.class_weights.requires_grad

    def test_frozen_encoders_have_no_grad(self, num_classes: int) -> None:
        """Encoder parameters should have requires_grad=False when frozen."""
        model = build_model(
            model_type="MultimodalLateFusion",
            encoder_name="resnet18",
            in_channels=5,
            n_classes=num_classes,
            model_config={
                "freeze_encoders": True,
                "aerial_model_config": {"backbone_name": "resnet18"},
                "sentinel_model_config": {
                    "image_size": 10,
                    "patch_size": 2,
                    "max_seq_len": 12,
                    "dim": 64,
                },
            },
        )

        # Check aerial model is frozen
        for param in model.aerial_model.parameters():
            assert not param.requires_grad

        # Check sentinel model is frozen
        for param in model.sentinel_model.parameters():
            assert not param.requires_grad

        # Check fusion weights are trainable
        assert model.class_weights.requires_grad

    def test_concat_fusion_mode(self, num_classes: int, batch_size: int) -> None:
        """Concat fusion mode should produce correct output shape."""
        model = build_model(
            model_type="MultimodalLateFusion",
            encoder_name="resnet18",
            in_channels=5,
            n_classes=num_classes,
            model_config={
                "fusion_mode": "concat",
                "aerial_model_type": "Unet",
                "aerial_model_config": {"backbone_name": "resnet18"},
                "sentinel_model_config": {
                    "image_size": 10,
                    "patch_size": 2,
                    "max_seq_len": 12,
                    "dim": 64,
                },
            },
        )
        model.eval()

        h, w = 64, 64
        aerial = torch.randn(batch_size, 5, h, w)
        sentinel = torch.randn(batch_size, 8, 10, 10, 10)
        positions = torch.arange(8).unsqueeze(0).repeat(batch_size, 1)

        with torch.no_grad():
            output = model(aerial, sentinel, batch_positions=positions, pad_mask=None)

        assert output.shape == (batch_size, num_classes, h, w)

    def test_average_fusion_mode(self, num_classes: int, batch_size: int) -> None:
        """Average fusion mode should produce correct output shape."""
        model = build_model(
            model_type="MultimodalLateFusion",
            encoder_name="resnet18",
            in_channels=5,
            n_classes=num_classes,
            model_config={
                "fusion_mode": "average",
                "aerial_model_type": "Unet",
                "aerial_model_config": {"backbone_name": "resnet18"},
                "sentinel_model_config": {
                    "image_size": 10,
                    "patch_size": 2,
                    "max_seq_len": 12,
                    "dim": 64,
                },
            },
        )
        model.eval()

        h, w = 64, 64
        aerial = torch.randn(batch_size, 5, h, w)
        sentinel = torch.randn(batch_size, 8, 10, 10, 10)
        positions = torch.arange(8).unsqueeze(0).repeat(batch_size, 1)

        with torch.no_grad():
            output = model(aerial, sentinel, batch_positions=positions, pad_mask=None)

        assert output.shape == (batch_size, num_classes, h, w)
