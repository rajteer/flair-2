from typing import Optional

import segmentation_models_pytorch as smp
import torch.nn as nn
from torch import optim


def build_model(model_type: str,
                encoder_name: str,
                encoder_weights: Optional[str],
                in_channels: int,
                n_classes: int,
                activation: Optional[str] = None) -> nn.Module:
    """
    Builds and returns a segmentation model from the segmentation_models_pytorch library.

    Args:
        model_type: The type of model to build (e.g., "Unet", "FPN", "DeepLabV3Plus").
        encoder_name: The name of the encoder (e.g., "resnet34", "mobilenet_v2").
        encoder_weights: The encoder weights to load (e.g., "imagenet", None).
        in_channels: The number of input image channels (e.g., 3 for RGB).
        n_classes: The number of classes to predict (including background if needed).
        activation: The activation function for the model output (e.g., "sigmoid",
        "softmax", None).

    Returns:
        nn.Module: The built PyTorch model.

    Raises:
        ValueError: If an unknown `model_type` is provided or another error
                    occurs during SMP model initialization.
    """
    try:
        model_class = getattr(smp, model_type)
    except AttributeError:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=n_classes,
        activation=activation,
        dynamic_img_size=True
    )

    return model


def build_optimizer(
        model: nn.Module,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float = 0.0
) -> optim.Optimizer:
    """
    Builds and returns an optimizer for the given model.

    Args:
        model: The PyTorch model to optimize.
        optimizer_type: The type of optimizer to use (e.g., "Adam", "SGD", "AdamW").
        learning_rate: The learning rate for the optimizer.
        weight_decay: The weight decay (L2 penalty) for the optimizer.

    Returns:
        optim.Optimizer: The configured PyTorch optimizer.

    Raises:
        ValueError: If an unknown `optimizer_type` is provided.
    """

    try:
        optimizer_class = getattr(optim, optimizer_type)
    except AttributeError:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer


def build_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Builds and returns a loss function.

    Args:
        loss_type: The type of loss function to use (e.g., "DiceLoss", "CrossEntropyLoss").
        **kwargs: Additional arguments to pass to the loss function.

    Returns:
        nn.Module: The configured loss function.

    Raises:
        ValueError: If an unknown `loss_type` is provided.
    """
    try:
        if hasattr(smp.losses, loss_type):
            loss_class = getattr(smp.losses, loss_type)
        else:
            loss_class = getattr(nn, loss_type)
    except AttributeError:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_class(**kwargs)
