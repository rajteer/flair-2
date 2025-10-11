import torch


def process_segmentation_tensor(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Standardize a segmentation tensor to shape (B, H, W) with class indices.

    Args:
        tensor (torch.Tensor): Input tensor to process.
        num_classes (int): Number of segmentation classes.

    Returns:
        torch.Tensor: Processed tensor of shape (B, H, W) with dtype torch.long.

    """
    if tensor.ndim == 4 and tensor.shape[1] == num_classes:
        tensor = torch.argmax(tensor, dim=1)
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    elif tensor.ndim != 3:
        msg = f"Expected tensor to have 3 or 4 dimensions, but got {tensor.ndim}"
        raise ValueError(msg)

    if tensor.dtype != torch.long:
        tensor = tensor.long()
    return tensor
