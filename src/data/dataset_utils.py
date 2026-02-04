"""Dataset utilities for FLAIR-2 including collate functions for variable-length sequences."""

import fnmatch
import os
import re
from pathlib import Path

import torch


def pad_tensor(
    tensor: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad a tensor along the first dimension to a target length.

    Args:
        tensor: Input tensor with shape (T, ...) where T is current length
        target_length: Desired length for the first dimension
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        Padded tensor with shape (target_length, ...)

    """
    if tensor.shape[0] >= target_length:
        return tensor[:target_length]

    pad_size = target_length - tensor.shape[0]
    pad_shape = (pad_size, *tensor.shape[1:])
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)

    return torch.cat([tensor, padding], dim=0)


def pad_collate_sentinel(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]],
    pad_value: float = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Collate function for batching Sentinel-2 samples with variable temporal dimensions.

    This function handles samples with different numbers of timesteps (T) by padding
    to the maximum temporal length in the batch. This is required because:
    - PyTorch batching requires all tensors to have the same shape
    - Cloud filtering results in different numbers of valid timesteps per sample
    - U-TAE model uses pad_mask to ignore padded positions during attention

    Args:
        batch: List of tuples (sentinel_data, mask, sample_id, month_positions) where:
            - sentinel_data: Tensor of shape (T_i, C, H, W) with T_i varying per sample
            - mask: Tensor of shape (H, W)
            - sample_id: String identifier
            - month_positions: Tensor of shape (T_i,) with month indices (0-11)
        pad_value: Value to use for padding temporal dimension (default: -1)

    Returns:
        Tuple containing:
            - padded_sentinel: Tensor of shape (B, T_max, C, H, W) where T_max is
              the maximum temporal length in the batch
            - masks: Tensor of shape (B, H, W)
            - pad_mask: Boolean tensor of shape (B, T_max) where True indicates
              a padded (invalid) timestep
            - sample_ids: List of sample identifiers
            - batch_positions: Long tensor of shape (B, T_max) with month indices (0-11)
              padded with pad_value (default: -1) for invalid positions

    """
    sentinel_data_list = [item[0] for item in batch]
    masks = torch.stack([item[1] for item in batch])
    sample_ids = [item[2] for item in batch]
    month_positions_list = [item[3] for item in batch]

    temporal_lengths = [data.shape[0] for data in sentinel_data_list]
    max_temporal_length = max(temporal_lengths)

    padded_sentinel_list = []
    padded_positions_list = []
    for data, positions in zip(sentinel_data_list, month_positions_list, strict=False):
        padded_data = pad_tensor(data, max_temporal_length, pad_value=pad_value)
        padded_sentinel_list.append(padded_data)

        padded_positions = pad_tensor(positions, max_temporal_length, pad_value=pad_value)
        padded_positions_list.append(padded_positions)

    padded_sentinel = torch.stack(padded_sentinel_list)
    batch_positions = torch.stack(padded_positions_list).long()

    pad_mask = torch.zeros(len(batch), max_temporal_length, dtype=torch.bool)
    for i, original_length in enumerate(temporal_lengths):
        if original_length < max_temporal_length:
            pad_mask[i, original_length:] = True

    return padded_sentinel, masks, pad_mask, sample_ids, batch_positions


def pad_collate_flair(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Collate function for FlairDataset with Sentinel-2 temporal data.

    When use_sentinel=True, FlairDataset returns (aerial, mask, sentinel, sample_id).
    This collate function extracts the Sentinel temporal data, pads it to a common
    length, and returns it as the model input (matching the Sentinel-only pipeline format).

    Args:
        batch: List of tuples (aerial_img, mask, sentinel_data, sample_id) where:
            - aerial_img: Tensor of shape (C, H, W) - not used as input
            - mask: Tensor of shape (H, W)
            - sentinel_data: Tensor of shape (T_i, C_s, H_s, W_s) with T_i varying
            - sample_id: String identifier
        pad_value: Value to use for padding temporal dimension (default: 0.0)

    Returns:
        Tuple containing:
            - padded_sentinel: Tensor of shape (B, T_max, C_s, H_s, W_s) - model input
            - masks: Tensor of shape (B, H, W) - targets
            - pad_mask: Boolean tensor of shape (B, T_max) where True indicates padded
            - sample_ids: List of sample identifiers
            - batch_positions: Long tensor of shape (B, T_max) for temporal attention

    """
    sentinel_data_list = [item[2] for item in batch]
    masks = torch.stack([item[1] for item in batch])
    sample_ids = [item[3] for item in batch]

    temporal_lengths = [data.shape[0] for data in sentinel_data_list]
    max_temporal_length = max(temporal_lengths)

    padded_sentinel_list = []
    for data in sentinel_data_list:
        padded_data = pad_tensor(data, max_temporal_length, pad_value=pad_value)
        padded_sentinel_list.append(padded_data)

    padded_sentinel = torch.stack(padded_sentinel_list)

    pad_mask = torch.zeros(len(batch), max_temporal_length, dtype=torch.bool)
    for i, original_length in enumerate(temporal_lengths):
        if original_length < max_temporal_length:
            pad_mask[i, original_length:] = True

    batch_positions = (
        torch.arange(max_temporal_length, dtype=torch.long).unsqueeze(0).expand(len(batch), -1)
    )

    return padded_sentinel, masks, pad_mask, sample_ids, batch_positions


def collate_multimodal(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    pad_value: float = 0.0,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[str],
    torch.Tensor,
]:
    """Collate function for multimodal models that need both aerial and sentinel inputs.

    When use_sentinel=True, FlairDataset returns (aerial, mask, sentinel, sample_id).
    This collate function returns both aerial and sentinel data as a tuple for the
    multimodal model input.

    Args:
        batch: List of tuples (aerial_img, mask, sentinel_data, sample_id) where:
            - aerial_img: Tensor of shape (C, H, W)
            - mask: Tensor of shape (H, W)
            - sentinel_data: Tensor of shape (T_i, C_s, H_s, W_s) with T_i varying
            - sample_id: String identifier
        pad_value: Value to use for padding temporal dimension (default: 0.0)

    Returns:
        Tuple containing:
            - inputs: Tuple of (aerial_images, padded_sentinel) where:
                - aerial_images: Tensor of shape (B, C, H, W)
                - padded_sentinel: Tensor of shape (B, T_max, C_s, H_s, W_s)
            - masks: Tensor of shape (B, H, W) - targets
            - pad_mask: Boolean tensor of shape (B, T_max) where True indicates padded
            - sample_ids: List of sample identifiers
            - batch_positions: Long tensor of shape (B, T_max) for temporal attention

    """
    aerial_images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    sentinel_data_list = [item[2] for item in batch]
    sample_ids = [item[3] for item in batch]

    temporal_lengths = [data.shape[0] for data in sentinel_data_list]
    max_temporal_length = max(temporal_lengths)

    padded_sentinel_list = []
    for data in sentinel_data_list:
        padded_data = pad_tensor(data, max_temporal_length, pad_value=pad_value)
        padded_sentinel_list.append(padded_data)

    padded_sentinel = torch.stack(padded_sentinel_list)

    pad_mask = torch.zeros(len(batch), max_temporal_length, dtype=torch.bool)
    for i, original_length in enumerate(temporal_lengths):
        if original_length < max_temporal_length:
            pad_mask[i, original_length:] = True

    batch_positions = (
        torch.arange(max_temporal_length, dtype=torch.long).unsqueeze(0).expand(len(batch), -1)
    )

    return (aerial_images, padded_sentinel), masks, pad_mask, sample_ids, batch_positions


def collate_standard(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, None, list[str], None]:
    """Collate function for FlairDataset without Sentinel-2 data.

    When use_sentinel=False, FlairDataset returns (aerial_img, mask, sample_id).
    This collate function stacks the tensors and returns a tuple with the same
    structure as the Sentinel collate functions for consistency.

    Args:
        batch: List of tuples (aerial_img, mask, sample_id) where:
            - aerial_img: Tensor of shape (C, H, W)
            - mask: Tensor of shape (H, W)
            - sample_id: String identifier

    Returns:
        Tuple containing:
            - images: Tensor of shape (B, C, H, W) - model input
            - masks: Tensor of shape (B, H, W) - targets
            - pad_mask: None (not used for standard models)
            - sample_ids: List of sample identifiers
            - batch_positions: None (not used for standard models)

    """
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    sample_ids = [item[2] for item in batch]

    return images, masks, None, sample_ids, None


def get_unique_id(filename: str) -> str:
    """Extract unique ID from filename.

    Args:
        filename: Name of the file (e.g., "IMG_00001.tif" or "MSK_00001.tif")

    Returns:
        Extracted numeric ID from the filename

    Raises:
        ValueError: If no numeric ID can be extracted

    """
    match = re.search(r"(\d+)", filename)
    if match is None:
        msg = f"Could not extract numeric ID from filename: {filename}"
        raise ValueError(msg)
    return match.group(1)


def get_path_mapping(directory: Path, pattern: str) -> dict[str, Path]:
    """Create a mapping from unique IDs to file paths.

    Args:
        directory: The directory to search
        pattern: The glob pattern to match files (e.g., "IMG_*.tif")

    Returns:
        A dictionary mapping unique IDs to file paths. The dictionary is sorted by ID.

    """
    mapping = {}
    directory_path = Path(directory)

    for root, _, files in os.walk(directory_path, followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                path = Path(root) / name
                try:
                    unique_id = get_unique_id(name)
                    mapping[unique_id] = path
                except ValueError:
                    continue

    return dict(sorted(mapping.items()))
