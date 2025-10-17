from typing import Any

import segmentation_models_pytorch as smp
from torch import nn, optim
from torch.optim import lr_scheduler as lr_schedulers
from torch.optim.lr_scheduler import LRScheduler

from src.models.losses import CombinedDiceFocalLoss


def build_model(
    model_type: str,
    encoder_name: str,
    in_channels: int,
    n_classes: int,
    encoder_weights: str | None = None,
    activation: str | None = None,
) -> nn.Module:
    """Build a segmentation model from segmentation_models_pytorch."""
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
        dynamic_img_size=True,
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

    # Check for custom loss functions first
    if loss_type == "CombinedDiceFocalLoss":
        return CombinedDiceFocalLoss(**kwargs)

    try:
        if hasattr(smp.losses, loss_type):
            loss_class = getattr(smp.losses, loss_type)
        else:
            loss_class = getattr(nn, loss_type)
    except AttributeError as err:
        msg = f"Unknown loss type: {loss_type}"
        raise ValueError(msg) from err

    return loss_class(**kwargs)
