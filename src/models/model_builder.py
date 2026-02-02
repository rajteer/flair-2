import logging
from typing import Any

import segmentation_models_pytorch as smp
from backbones.utae import UTAE
from torch import nn, optim
from torch.optim import lr_scheduler as lr_schedulers
from torch.optim.lr_scheduler import LRScheduler

from src.models.losses import CombinedDiceFocalLoss, WeightedCrossEntropyDiceLoss

logger = logging.getLogger(__name__)

try:
    from src.models.rs3mamba import RS3Mamba, load_pretrained_ckpt
except ImportError as e:
    RS3Mamba = None
    load_pretrained_ckpt = None
    RS3MAMBA_IMPORT_ERROR = e
from src.models.tsvit import TSViT
from src.models.unetformer import UNetFormer
from src.models.multimodal_fusion import MultimodalLateFusion, load_pretrained_multimodal
from src.models.multimodal_mid_fusion import build_multimodal_mid_fusion


def build_model(
    model_type: str,
    encoder_name: str,
    in_channels: int,
    n_classes: int,
    encoder_weights: str | None = None,
    activation: str | None = None,
    *,
    model_config: dict[str, Any] | None = None,
    stochastic_depth: float | None = None,
    decoder_norm: bool | str | dict[str, Any] | None = None,
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
        model_config: Additional model-specific configuration parameters
        stochastic_depth: Drop path rate for stochastic depth (regularization).
        decoder_norm: Decoder normalization config. Can be:
            - True: use BatchNorm (default)
            - False: no normalization
            - str: 'batchnorm', 'groupnorm', 'layernorm', 'instancenorm'
            - dict: {'type': 'groupnorm', 'num_groups': 8}

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
        if RS3Mamba is None:
            msg = (
                f"RS3Mamba could not be imported. Check dependencies (e.g. mamba_ssm, kernels). "
                f"Original error: {RS3MAMBA_IMPORT_ERROR}"
            )
            raise ImportError(msg)
        rs3mamba_config = model_config or {}

        model = RS3Mamba(
            decode_channels=rs3mamba_config.get("decode_channels", 64),
            dropout=rs3mamba_config.get("dropout", 0.1),
            backbone_name=encoder_name or rs3mamba_config.get("backbone_name", "swsl_resnet18"),
            pretrained=encoder_weights is not None,
            window_size=rs3mamba_config.get("window_size", 8),
            num_classes=n_classes,
            in_channels=in_channels,
            use_channel_attention=rs3mamba_config.get("use_channel_attention", True),
        )

        pretrain_path = rs3mamba_config.get("vssm_pretrain_path")
        if pretrain_path:
            model = load_pretrained_ckpt(model, pretrain_path)

        return model

    if model_type.upper() == "UNETFORMER":
        unetformer_config = model_config or {}

        model = UNetFormer(
            decode_channels=unetformer_config.get("decode_channels", 64),
            dropout=unetformer_config.get("dropout", 0.1),
            backbone_name=encoder_name or unetformer_config.get("backbone_name", "swsl_resnet18"),
            pretrained=encoder_weights is not None,
            window_size=unetformer_config.get("window_size", 8),
            num_classes=n_classes,
            in_channels=in_channels,
            use_aux_head=unetformer_config.get("use_aux_head", False),
            encoder_type=unetformer_config.get("encoder_type", "timm"),
            samba_config=unetformer_config.get("samba_config"),
        )
        # Store aux_loss_weight as model attribute for training loop access
        model.aux_loss_weight = unetformer_config.get("aux_loss_weight", 0.4)
        return model

    if model_type.upper() == "MULTIMODALLATEFUSION":
        multimodal_config = model_config or {}

        # Build aerial model (UNetFormer by default)
        aerial_model_type = multimodal_config.get("aerial_model_type", "UNetFormer")
        aerial_model_config = multimodal_config.get("aerial_model_config", {})
        aerial_in_channels = multimodal_config.get("aerial_in_channels", in_channels)

        aerial_model = build_model(
            model_type=aerial_model_type,
            encoder_name=aerial_model_config.get("backbone_name", encoder_name),
            in_channels=aerial_in_channels,
            n_classes=n_classes,
            encoder_weights=encoder_weights,
            model_config=aerial_model_config,
        )

        # Build Sentinel model (TSViT by default)
        sentinel_model_type = multimodal_config.get("sentinel_model_type", "TSVIT")
        sentinel_model_config = multimodal_config.get("sentinel_model_config", {})
        sentinel_in_channels = multimodal_config.get("sentinel_in_channels", 10)

        sentinel_model = build_model(
            model_type=sentinel_model_type,
            encoder_name="",
            in_channels=sentinel_in_channels,
            n_classes=n_classes,
            model_config=sentinel_model_config,
        )

        # Load pre-trained checkpoints if provided
        aerial_checkpoint = multimodal_config.get("aerial_checkpoint")
        sentinel_checkpoint = multimodal_config.get("sentinel_checkpoint")

        if aerial_checkpoint or sentinel_checkpoint:
            aerial_model, sentinel_model = load_pretrained_multimodal(
                aerial_checkpoint=aerial_checkpoint,
                sentinel_checkpoint=sentinel_checkpoint,
                aerial_model=aerial_model,
                sentinel_model=sentinel_model,
            )

        return MultimodalLateFusion(
            aerial_model=aerial_model,
            sentinel_model=sentinel_model,
            num_classes=n_classes,
            freeze_encoders=multimodal_config.get("freeze_encoders", True),
            fusion_mode=multimodal_config.get("fusion_mode", "weighted"),
            aerial_resolution=tuple(multimodal_config.get("aerial_resolution", [512, 512])),
            sentinel_resolution=tuple(multimodal_config.get("sentinel_resolution", [10, 10])),
            use_cloud_uncertainty=multimodal_config.get("use_cloud_uncertainty", False),
        )

    if model_type.upper() == "MULTIMODALMIDFUSION":
        multimodal_config = model_config or {}

        # Build UNetFormer to extract backbone and decoder separately
        unetformer_config = multimodal_config.get("aerial_model_config", {})
        aerial_in_channels = multimodal_config.get("aerial_in_channels", in_channels)

        temp_unetformer = UNetFormer(
            decode_channels=unetformer_config.get("decode_channels", 64),
            dropout=unetformer_config.get("dropout", 0.1),
            backbone_name=unetformer_config.get("backbone_name", encoder_name or "swsl_resnet18"),
            pretrained=encoder_weights is not None,
            window_size=unetformer_config.get("window_size", 8),
            num_classes=n_classes,
            in_channels=aerial_in_channels,
            use_aux_head=False,
        )
        aerial_backbone = temp_unetformer.backbone
        aerial_decoder = temp_unetformer.decoder

        # Get encoder channel info for cross-attention
        encoder_channels = aerial_backbone.feature_info.channels()
        aerial_channels = encoder_channels[-1]  # Deepest level channels

        # Build TSViT for temporal encoding
        sentinel_model_config = multimodal_config.get("sentinel_model_config", {})
        sentinel_in_channels = multimodal_config.get("sentinel_in_channels", 10)

        sentinel_encoder = TSViT(
            image_size=int(sentinel_model_config.get("image_size", 10)),
            patch_size=int(sentinel_model_config.get("patch_size", 2)),
            in_channels=sentinel_in_channels,
            num_classes=n_classes,
            max_seq_len=int(sentinel_model_config.get("max_seq_len", 12)),
            dim=int(sentinel_model_config.get("dim", 128)),
            temporal_depth=int(sentinel_model_config.get("temporal_depth", 4)),
            spatial_depth=int(sentinel_model_config.get("spatial_depth", 4)),
            num_heads=int(sentinel_model_config.get("num_heads", 4)),
            mlp_dim=int(sentinel_model_config.get("mlp_dim", 256)),
            dropout=float(sentinel_model_config.get("dropout", 0.0)),
            emb_dropout=float(sentinel_model_config.get("emb_dropout", 0.0)),
        )

        sentinel_dim = int(sentinel_model_config.get("dim", 128))
        cross_attn_config = multimodal_config.get("cross_attention_config", {})

        return build_multimodal_mid_fusion(
            aerial_backbone=aerial_backbone,
            aerial_decoder=aerial_decoder,
            sentinel_encoder=sentinel_encoder,
            num_classes=n_classes,
            sentinel_dim=sentinel_dim,
            aerial_channels=aerial_channels,
            num_heads=int(cross_attn_config.get("num_heads", 8)),
            dropout=float(cross_attn_config.get("dropout", 0.1)),
            freeze_encoders=multimodal_config.get("freeze_encoders", True),
        )

    try:
        model_class = getattr(smp, model_type)
    except AttributeError as err:
        msg = f"Unknown model type: {model_type}"
        raise ValueError(msg) from err

    kwargs = {}
    if stochastic_depth is not None:
        kwargs["drop_path_rate"] = stochastic_depth

    use_groupnorm_replacement = False
    num_groups = 32
    if isinstance(decoder_norm, dict) and decoder_norm.get("type") == "groupnorm":
        use_groupnorm_replacement = True
        num_groups = decoder_norm.get("num_groups", 32)
    elif decoder_norm == "groupnorm":
        use_groupnorm_replacement = True
    elif decoder_norm is not None:
        kwargs["decoder_use_norm"] = decoder_norm

    smp_config = model_config or {}
    for key, value in smp_config.items():
        if key == "stochastic_depth":
            continue
        if key not in kwargs:
            kwargs[key] = value

    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=n_classes,
        activation=activation,
        **kwargs,
    )

    if use_groupnorm_replacement:
        _replace_batchnorm_with_groupnorm(model.decoder, num_groups=num_groups)
        logging.getLogger(__name__).info(
            "Replaced BatchNorm2d with GroupNorm (num_groups=%d) in decoder",
            num_groups,
        )

    return model


def _replace_batchnorm_with_groupnorm(
    module: nn.Module,
    num_groups: int = 32,
) -> None:
    """Replace all BatchNorm2d layers with GroupNorm in a module (in-place).

    Args:
        module: The module to modify.
        num_groups: Number of groups for GroupNorm. Must divide num_channels.

    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            if num_channels == 0:
                continue
            effective_groups = min(num_groups, num_channels)
            while num_channels % effective_groups != 0 and effective_groups > 1:
                effective_groups -= 1

            group_norm = nn.GroupNorm(
                num_groups=effective_groups,
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine,
            )
            setattr(module, name, group_norm)
        else:
            _replace_batchnorm_with_groupnorm(child, num_groups)


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

    kwargs: dict = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if betas is not None:
        kwargs["betas"] = betas

    return optimizer_class(model.parameters(), **kwargs)


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: dict[str, Any] | None = None,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
) -> LRScheduler | None:
    """Build and return a learning rate scheduler.

    Args:
        optimizer: Optimizer to wrap with the scheduler.
        scheduler_config: Mapping containing the scheduler ``type`` and optional ``args``.
        steps_per_epoch: Number of steps per epoch (required for OneCycleLR if not in args).
        epochs: Total number of training epochs (required for OneCycleLR if not in args).

    Returns:
        Instantiated scheduler or ``None`` when configuration is missing.

    """
    if scheduler_config is None:
        return None

    scheduler_type = scheduler_config.get("type")
    scheduler_args = dict(scheduler_config.get("args", {}))

    # OneCycleLR requires steps_per_epoch and epochs
    if scheduler_type == "OneCycleLR":
        if scheduler_args.get("steps_per_epoch") is None and steps_per_epoch is not None:
            scheduler_args["steps_per_epoch"] = steps_per_epoch + 1
        if scheduler_args.get("epochs") is None and epochs is not None:
            scheduler_args["epochs"] = epochs

    try:
        scheduler_class = getattr(lr_schedulers, scheduler_type)
    except AttributeError as err:
        msg = f"Unknown LR scheduler type: {scheduler_type}"
        raise ValueError(msg) from err

    scheduler = scheduler_class(optimizer, **scheduler_args)
    logger.info(
        "Created LR scheduler: %s with args: %s",
        scheduler_type,
        scheduler_args,
    )
    initial_lr = optimizer.param_groups[0]["lr"]
    logger.info("Initial learning rate after scheduler creation: %.8f", initial_lr)
    return scheduler


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
