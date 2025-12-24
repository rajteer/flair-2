from typing import Any

import segmentation_models_pytorch as smp
from backbones.utae import UTAE
from torch import nn, optim
from torch.optim import lr_scheduler as lr_schedulers
from torch.optim.lr_scheduler import LRScheduler

from src.models.losses import CombinedDiceFocalLoss, WeightedCrossEntropyDiceLoss

try:
    from src.models.rs3mamba import RS3Mamba, load_pretrained_ckpt
except ImportError:
    RS3Mamba = None
    load_pretrained_ckpt = None
from src.models.tsvit import TSViT
from src.models.unetformer import UNetFormer


def build_model(
    model_type: str,
    encoder_name: str,
    in_channels: int,
    n_classes: int,
    encoder_weights: str | None = None,
    activation: str | None = None,
    *,
    dynamic_img_size: bool = False,
    model_config: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a segmentation model.

    Supports both SMP models and temporal models like U-TAE.

    Args:
        model_type: Type of model (e.g., 'Unet', 'FPN', 'UTAE')
        encoder_name: Name of encoder backbone
        in_channels: Number of input channels
        n_classes: Number of output classes
        encoder_weights: Pre-trained weights for encoder
        activation: Activation function for output
        dynamic_img_size: Whether to support dynamic image sizes
        model_config: Additional model-specific configuration parameters

    Returns:
        Initialized model

    """
    if model_type.upper() == "UTAE":
        utae_config = model_config or {}

        utae_params = {
            "input_dim": in_channels,
            "encoder_widths": utae_config.get("encoder_widths", [64, 64, 64, 128]),
            "decoder_widths": utae_config.get("decoder_widths", [32, 32, 64, 128]),
            "out_conv": utae_config.get("out_conv", [32, n_classes]),
            "str_conv_k": utae_config.get("str_conv_k", 4),
            "str_conv_s": utae_config.get("str_conv_s", 2),
            "str_conv_p": utae_config.get("str_conv_p", 1),
            "agg_mode": utae_config.get("agg_mode", "att_group"),
            "encoder_norm": utae_config.get("encoder_norm", "group"),
            "n_head": utae_config.get("n_head", 16),
            "d_model": utae_config.get("d_model", 256),
            "d_k": utae_config.get("d_k", 4),
            "encoder": utae_config.get("encoder", False),
            "return_maps": utae_config.get("return_maps", False),
            "pad_value": utae_config.get("pad_value", 0),
            "padding_mode": utae_config.get("padding_mode", "reflect"),
        }

        return UTAE(**utae_params)
    if model_type.upper() == "TSVIT":
        tsvit_config = model_config or {}

        image_size = tsvit_config.get("image_size", tsvit_config.get("img_res"))
        patch_size = tsvit_config.get("patch_size")
        max_seq_len = tsvit_config.get("max_seq_len")
        dim = tsvit_config.get("dim")

        missing = [
            key
            for key, value in {
                "image_size": image_size,
                "patch_size": patch_size,
                "max_seq_len": max_seq_len,
                "dim": dim,
            }.items()
            if value is None
        ]
        if missing:
            msg = f"Missing TSViT config keys: {', '.join(missing)}"
            raise ValueError(msg)

        depth_fallback = tsvit_config.get("depth", 4)
        mlp_dim = tsvit_config.get(
            "mlp_dim",
            tsvit_config.get("scale_dim", 4) * dim,
        )

        return TSViT(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=in_channels,
            num_classes=n_classes,
            max_seq_len=int(max_seq_len),
            dim=int(dim),
            temporal_depth=int(tsvit_config.get("temporal_depth", depth_fallback)),
            spatial_depth=int(tsvit_config.get("spatial_depth", depth_fallback)),
            num_heads=int(tsvit_config.get("num_heads", 4)),
            mlp_dim=int(mlp_dim),
            dropout=float(tsvit_config.get("dropout", 0.0)),
            emb_dropout=float(tsvit_config.get("emb_dropout", 0.0)),
            temporal_metadata_channels=int(tsvit_config.get("temporal_metadata_channels", 0)),
        )

    if model_type.upper() == "RS3MAMBA":
        rs3mamba_config = model_config or {}

        model = RS3Mamba(
            decode_channels=rs3mamba_config.get("decode_channels", 64),
            dropout=rs3mamba_config.get("dropout", 0.1),
            backbone_name=encoder_name or rs3mamba_config.get("backbone_name", "swsl_resnet18"),
            pretrained=encoder_weights is not None,
            window_size=rs3mamba_config.get("window_size", 8),
            num_classes=n_classes,
            in_channels=in_channels,
        )

        pretrain_path = rs3mamba_config.get("vssm_pretrain_path")
        if pretrain_path:
            model = load_pretrained_ckpt(model, pretrain_path)

        return model

    if model_type.upper() == "UNETFORMER":
        unetformer_config = model_config or {}

        return UNetFormer(
            decode_channels=unetformer_config.get("decode_channels", 64),
            dropout=unetformer_config.get("dropout", 0.1),
            backbone_name=encoder_name or unetformer_config.get("backbone_name", "swsl_resnet18"),
            pretrained=encoder_weights is not None,
            window_size=unetformer_config.get("window_size", 8),
            num_classes=n_classes,
            in_channels=in_channels,
        )

    try:
        model_class = getattr(smp, model_type)
    except AttributeError as err:
        msg = f"Unknown model type: {model_type}"
        raise ValueError(msg) from err

    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=n_classes,
        activation=activation,
    )


def build_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] | None = None,
) -> optim.Optimizer:
    """Build and return an optimizer for the given model."""
    try:
        optimizer_class = getattr(optim, optimizer_type)
    except AttributeError as err:
        msg = f"Unknown optimizer type: {optimizer_type}"
        raise ValueError(msg) from err

    return optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: dict[str, Any] | None = None,
) -> LRScheduler | None:
    """Build and return a learning rate scheduler.

    Args:
        optimizer: Optimizer to wrap with the scheduler.
        scheduler_config: Mapping containing the scheduler ``type`` and optional ``args``.

    Returns:
        Instantiated scheduler or ``None`` when configuration is missing.

    """
    if scheduler_config is None:
        return None

    scheduler_type = scheduler_config.get("type")
    scheduler_args = scheduler_config.get("args", {})

    try:
        scheduler_class = getattr(lr_schedulers, scheduler_type)
    except AttributeError as err:
        msg = f"Unknown LR scheduler type: {scheduler_type}"
        raise ValueError(msg) from err

    return scheduler_class(optimizer, **scheduler_args)


def build_loss_function(
    loss_type: str,
    kwargs: dict[str, object] | None = None,
) -> nn.Module:
    """Build and return a loss function.

    The function accepts an explicit kwargs mapping instead of a dynamic
    **kwargs to improve static typing. If kwargs is None, an empty
    mapping will be used when constructing the loss class.
    """
    if kwargs is None:
        kwargs = {}

    if loss_type == "CombinedDiceFocalLoss":
        return CombinedDiceFocalLoss(**kwargs)
    if loss_type == "WeightedCrossEntropyDiceLoss":
        return WeightedCrossEntropyDiceLoss(**kwargs)

    try:
        if hasattr(smp.losses, loss_type):
            loss_class = getattr(smp.losses, loss_type)
        else:
            loss_class = getattr(nn, loss_type)
    except AttributeError as err:
        msg = f"Unknown loss type: {loss_type}"
        raise ValueError(msg) from err

    return loss_class(**kwargs)
