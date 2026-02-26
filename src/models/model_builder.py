import logging
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn, optim
from torch.optim import lr_scheduler as lr_schedulers
from torch.optim.lr_scheduler import LRScheduler

from src.models.architectures.utae_pp import UTAE
from src.training.losses import CombinedDiceFocalLoss, WeightedCrossEntropyDiceLoss

logger = logging.getLogger(__name__)

try:
    from src.models.architectures.rs3mamba import RS3Mamba, load_pretrained_ckpt
except ImportError as e:
    RS3Mamba = None
    load_pretrained_ckpt = None
    RS3MAMBA_IMPORT_ERROR = e
from src.models.architectures.multimodal_fusion import (
    MultimodalLateFusion,
    load_pretrained_multimodal,
)
from src.models.architectures.tsvit import TSViT
from src.models.architectures.tsvit_lookup import TSViTLookup
from src.models.architectures.unetformer import UNetFormer


def _resolve_attention_type(config: dict[str, Any]) -> str:
    """Resolve attention_type from config, with backward compatibility for use_cbam."""
    if "attention_type" in config:
        return config["attention_type"]
    # Backward compatibility: convert use_cbam bool to attention_type string
    use_cbam = config.get("use_cbam", True)
    return "cbam" if use_cbam else "none"


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

        n_head = utae_config.get("n_head", 16)
        d_model = utae_config.get("d_model", 256)
        d_k = utae_config.get("d_k", d_model // n_head)

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
            "n_head": n_head,
            "d_model": d_model,
            "d_k": d_k,
            "encoder": utae_config.get("encoder", False),
            "return_maps": utae_config.get("return_maps", False),
            "pad_value": utae_config.get("pad_value", 0),
            "padding_mode": utae_config.get("padding_mode", "reflect"),
            # New U-TAE++ options
            "use_convnext": utae_config.get("use_convnext", True),
            "attention_type": _resolve_attention_type(utae_config),
            "drop_path_rate": utae_config.get("drop_path_rate", 0.1),
            "deep_supervision": utae_config.get("deep_supervision", False),
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

    if model_type.upper() == "TSVIT_LOOKUP":
        tsvit_config = model_config or {}

        image_size = tsvit_config.get("image_size", tsvit_config.get("img_res"))
        patch_size = tsvit_config.get("patch_size")
        dim = tsvit_config.get("dim")

        missing = [
            key
            for key, value in {
                "image_size": image_size,
                "patch_size": patch_size,
                "dim": dim,
            }.items()
            if value is None
        ]
        if missing:
            msg = f"Missing TSViTLookup config keys: {', '.join(missing)}"
            raise ValueError(msg)

        depth_fallback = tsvit_config.get("depth", 4)
        mlp_dim = tsvit_config.get(
            "mlp_dim",
            tsvit_config.get("scale_dim", 4) * dim,
        )

        # Handle train_dates configuration (use days by default for fine-grained embeddings)
        train_dates_cfg = tsvit_config.get("train_dates", "days")
        if train_dates_cfg == "months":
            train_dates = list(range(0, 12))  # 0-11 for months (0-indexed)
            date_range = (0, 11)
        elif train_dates_cfg == "days":
            train_dates = list(range(0, 365))  # 0-364 for day-of-year (0-indexed)
            date_range = (0, 364)
        elif isinstance(train_dates_cfg, list):
            train_dates = train_dates_cfg
            date_range = tsvit_config.get("date_range", (min(train_dates), max(train_dates)))
        else:
            msg = f"train_dates must be 'months', 'days', or a list. Got: {train_dates_cfg}"
            raise ValueError(msg)

        # Allow override of date_range from config
        date_range = tsvit_config.get("date_range", date_range)

        return TSViTLookup(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=in_channels,
            num_classes=n_classes,
            train_dates=train_dates,
            date_range=tuple(date_range),
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
            img_size=unetformer_config.get("img_size"),
            use_aux_head=unetformer_config.get("use_aux_head", False),
            encoder_type=unetformer_config.get("encoder_type", "timm"),
            samba_config=unetformer_config.get("samba_config"),
            drop_path_rate=stochastic_depth or 0.0,
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
                strict=bool(multimodal_config.get("checkpoint_strict", True)),
                strip_prefixes=multimodal_config.get("checkpoint_strip_prefixes"),
            )

        sentinel_output_resolution = None
        sentinel_output_resolution_raw = multimodal_config.get("sentinel_output_resolution")
        if sentinel_output_resolution_raw is not None:
            if isinstance(sentinel_output_resolution_raw, int):
                sentinel_output_resolution = (
                    int(sentinel_output_resolution_raw),
                    int(sentinel_output_resolution_raw),
                )
            elif isinstance(sentinel_output_resolution_raw, (list, tuple)):
                if len(sentinel_output_resolution_raw) != 2:  # noqa: PLR2004
                    msg = (
                        "sentinel_output_resolution must have length 2 (H, W), "
                        f"got {sentinel_output_resolution_raw!r}"
                    )
                    raise ValueError(msg)
                sentinel_output_resolution = (
                    int(sentinel_output_resolution_raw[0]),
                    int(sentinel_output_resolution_raw[1]),
                )
            else:
                msg = (
                    "sentinel_output_resolution must be an int or a 2-item list/tuple, "
                    f"got {type(sentinel_output_resolution_raw).__name__}"
                )
                raise ValueError(msg)

        return MultimodalLateFusion(
            aerial_model=aerial_model,
            sentinel_model=sentinel_model,
            num_classes=n_classes,
            freeze_encoders=multimodal_config.get("freeze_encoders", True),
            freeze_encoder_stats=multimodal_config.get("freeze_encoder_stats"),
            fusion_mode=multimodal_config.get("fusion_mode", "weighted"),
            aerial_resolution=tuple(multimodal_config.get("aerial_resolution", [512, 512])),
            sentinel_resolution=tuple(multimodal_config.get("sentinel_resolution", [10, 10])),
            sentinel_output_resolution=sentinel_output_resolution,
            use_cloud_uncertainty=multimodal_config.get("use_cloud_uncertainty", False),
            modality_weights=multimodal_config.get("modality_weights"),
            init_class_weights=multimodal_config.get("init_class_weights"),
            gate_class_priors=multimodal_config.get("gate_class_priors"),
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


def _get_encoder_decoder_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Separate model parameters into encoder and decoder groups."""
    encoder_module = getattr(model, "backbone", None) or getattr(model, "encoder", None)
    if encoder_module is None:
        msg = "Model has no 'backbone' or 'encoder' attribute."
        raise ValueError(msg)

    encoder_param_ids = {id(p) for p in encoder_module.parameters()}
    encoder_params = list(encoder_module.parameters())
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    return encoder_params, decoder_params


def freeze_encoder(model: nn.Module) -> None:
    """Freeze encoder parameters (set requires_grad=False)."""
    encoder_module = getattr(model, "backbone", None) or getattr(model, "encoder", None)
    if encoder_module is None:
        msg = "Model has no 'backbone' or 'encoder' attribute."
        raise ValueError(msg)
    for param in encoder_module.parameters():
        param.requires_grad = False
    logger.info("Encoder frozen (%d parameters)", sum(1 for _ in encoder_module.parameters()))


def unfreeze_encoder(model: nn.Module) -> None:
    """Unfreeze encoder parameters (set requires_grad=True)."""
    encoder_module = getattr(model, "backbone", None) or getattr(model, "encoder", None)
    if encoder_module is None:
        msg = "Model has no 'backbone' or 'encoder' attribute."
        raise ValueError(msg)
    for param in encoder_module.parameters():
        param.requires_grad = True
    logger.info("Encoder unfrozen (%d parameters)", sum(1 for _ in encoder_module.parameters()))


def build_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] | None = None,
    encoder_lr_mult: float | None = None,
) -> optim.Optimizer:
    """Build and return an optimizer for the given model.

    Args:
        encoder_lr_mult: Optional multiplier for encoder LR (encoder_lr = learning_rate * mult).

    """
    try:
        optimizer_class = getattr(optim, optimizer_type)
    except AttributeError as err:
        msg = f"Unknown optimizer type: {optimizer_type}"
        raise ValueError(msg) from err

    kwargs: dict = {"weight_decay": weight_decay}
    if betas is not None:
        kwargs["betas"] = betas

    if encoder_lr_mult is not None and encoder_lr_mult != 1.0:
        encoder_params, decoder_params = _get_encoder_decoder_params(model)
        encoder_params = [p for p in encoder_params if p.requires_grad]
        decoder_params = [p for p in decoder_params if p.requires_grad]
        if not encoder_params and not decoder_params:
            dummy_param = nn.Parameter(torch.tensor([0.0]))
            return optimizer_class([dummy_param], lr=learning_rate)
        encoder_lr = learning_rate * encoder_lr_mult
        param_groups = [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": learning_rate},
        ]
        logger.info(
            "Differential LR: encoder=%.2e (mult=%.2f), decoder=%.2e",
            encoder_lr,
            encoder_lr_mult,
            learning_rate,
        )
        return optimizer_class(param_groups, **kwargs)

    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        # Keep training loop API stable when all parameters are frozen.
        dummy_param = nn.Parameter(torch.tensor([0.0]))
        return optimizer_class([dummy_param], lr=learning_rate)

    return optimizer_class(parameters, lr=learning_rate, **kwargs)


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
    if scheduler_type is None:
        return None

    scheduler_args = dict(scheduler_config.get("args", {}))

    # OneCycleLR requires steps_per_epoch and epochs
    if scheduler_type == "OneCycleLR":
        if scheduler_args.get("steps_per_epoch") is None and steps_per_epoch is not None:
            scheduler_args["steps_per_epoch"] = steps_per_epoch + 1
        if scheduler_args.get("epochs") is None and epochs is not None:
            scheduler_args["epochs"] = epochs

    # PolynomialLR requires total_iters
    if scheduler_type == "PolynomialLR":
        if (
            scheduler_args.get("total_iters") is None
            and steps_per_epoch is not None
            and epochs is not None
        ):
            scheduler_args["total_iters"] = steps_per_epoch * epochs

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
