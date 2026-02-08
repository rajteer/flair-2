"""Multimodal dataset for FLAIR-2: aligned aerial and Sentinel-2 data.

This dataset loads synchronized aerial imagery and Sentinel-2 time series
for multimodal fusion experiments.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..dataset_utils import get_path_mapping
from .sentinel_utils import (
    MAX_ORIGINAL_CLASS,
    OTHER_CLASS,
    compute_monthly_averages,
    extract_domain_zone,
    extract_sentinel_patch,
    filter_cloudy_snowy_timesteps,
    load_centroids_mapping,
    load_sentinel_dates,
    load_sentinel_superpatch_paths,
    parse_sentinel_day_of_year,
)

logger = logging.getLogger(__name__)


class FlairMultimodalDataset(Dataset):
    """Multimodal dataset providing aligned aerial and Sentinel-2 data.

    This dataset is designed for multimodal fusion experiments where both
    high-resolution aerial imagery (512x512) and temporal Sentinel-2 satellite
    data (10x10) are needed together.

    Returns for each sample:
        - aerial_image: (C, H, W) aerial imagery tensor at original resolution
        - sentinel_data: (T, C, H, W) Sentinel-2 time series
        - mask: (H, W) ground truth at aerial resolution
        - sample_id: str
        - month_positions: (T,) temporal positions for Sentinel data

    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        sentinel_dir: str,
        centroids_path: str,
        num_classes: int = 13,
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        image_transform: Callable[[Tensor], Tensor] | None = None,
        selected_channels: list[int] | None = None,
        sentinel_patch_size: int = 10,
        *,
        context_size: int | None = None,
        use_monthly_average: bool = True,
        cloud_snow_cover_threshold: float = 0.6,
        cloud_snow_prob_threshold: int = 50,
        sentinel_scale_factor: float = 10000.0,
        sentinel_mean: list[float] | None = None,
        sentinel_std: list[float] | None = None,
    ) -> None:
        """Initialize the multimodal FLAIR-2 dataset.

        Args:
            image_dir: Directory containing aerial image files (IMG_*.tif)
            mask_dir: Directory containing mask files (MSK_*.tif)
            sentinel_dir: Directory containing Sentinel-2 superpatch data
            centroids_path: Path to JSON file mapping image IDs to superpatch coordinates
            num_classes: Number of segmentation classes (default: 13)
            transform: Optional transform to apply to (image, mask) pairs
            sentinel_patch_size: Size of Sentinel-2 patch to extract in Sentinel pixels
            context_size: Size of the input Sentinel-2 patch to extract (context window).
                Must be >= sentinel_patch_size. If None, defaults to sentinel_patch_size.
            use_monthly_average: Whether to compute monthly averages from cloudless dates
            cloud_snow_cover_threshold: Maximum allowed cloud/snow coverage (0-1)
            cloud_snow_prob_threshold: Minimum probability (0-100) to consider pixel cloudy
            sentinel_scale_factor: Factor to divide Sentinel-2 reflectance values by

        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.sentinel_dir = Path(sentinel_dir)
        self.num_classes = num_classes
        self.transform = transform
        self.image_transform = image_transform
        self.selected_channels = selected_channels
        self.sentinel_patch_size = sentinel_patch_size
        self.context_size = context_size if context_size is not None else sentinel_patch_size
        if self.context_size < self.sentinel_patch_size:
            msg = (
                "context_size must be >= sentinel_patch_size. "
                f"Got context_size={self.context_size}, sentinel_patch_size={self.sentinel_patch_size}."
            )
            raise ValueError(msg)
        self.use_monthly_average = use_monthly_average
        self.cloud_snow_cover_threshold = cloud_snow_cover_threshold
        self.cloud_snow_prob_threshold = cloud_snow_prob_threshold
        self.sentinel_scale_factor = sentinel_scale_factor
        if (sentinel_mean is None) != (sentinel_std is None):
            msg = "sentinel_mean and sentinel_std must be provided together or omitted."
            raise ValueError(msg)
        self.sentinel_mean = sentinel_mean
        self.sentinel_std = sentinel_std

        # Load aerial image and mask paths
        self.features_dict = get_path_mapping(self.image_dir, "IMG_*.tif")
        self.labels_dict = get_path_mapping(self.mask_dir, "MSK_*.tif")

        # Find common IDs between images and masks
        self.ids = sorted(set(self.features_dict.keys()) & set(self.labels_dict.keys()))

        if len(self.ids) == 0:
            logger.warning("No matching image-mask pairs found!")

        # Load Sentinel data structures
        self.centroids_mapping = load_centroids_mapping(centroids_path)
        (
            self.sentinel_data_dict,
            self.sentinel_masks_dict,
            self.sentinel_dates_dict,
        ) = load_sentinel_superpatch_paths(
            self.sentinel_dir,
            load_masks=True,
            load_dates=True,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ids)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """Get a multimodal sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
                - aerial_image: (C, H, W) aerial imagery tensor
                - sentinel_data: (T, C, H, W) Sentinel-2 time series
                - mask: (H, W) ground truth at aerial resolution
                - sample_id: str
                - month_positions: (T,) temporal positions

        Raises:
            IndexError: If idx is out of range

        """
        if abs(idx) >= len(self):
            msg = f"Index {idx} out of range for dataset of size {len(self)}"
            raise IndexError(msg)

        sample_id = self.ids[idx]
        feature_path = self.features_dict[sample_id]
        label_path = self.labels_dict[sample_id]

        # Load aerial image (C, H, W) - channels last to channels first
        aerial_image = tifffile.imread(feature_path)
        aerial_image = torch.from_numpy(aerial_image).permute(2, 0, 1).float()

        if self.selected_channels is not None:
            aerial_image = aerial_image[self.selected_channels, :, :]

        if self.image_transform is not None:
            aerial_image = self.image_transform(aerial_image)

        # Load mask
        mask = tifffile.imread(label_path)
        mask = np.where(mask <= MAX_ORIGINAL_CLASS, mask, OTHER_CLASS)
        mask -= 1  # Convert to 0-indexed classes
        mask = torch.from_numpy(mask).long()

        # Apply transforms if provided
        if self.transform is not None:
            aerial_image, mask = self.transform(aerial_image, mask)

        # Load Sentinel-2 data
        sentinel_data, month_positions = self._load_sentinel_patch(feature_path)

        return aerial_image, sentinel_data, mask, sample_id, month_positions

    def _normalize_sentinel(self, sentinel_tensor: torch.Tensor) -> torch.Tensor:
        if self.sentinel_mean is None or self.sentinel_std is None:
            return sentinel_tensor

        mean = torch.tensor(
            self.sentinel_mean,
            dtype=sentinel_tensor.dtype,
            device=sentinel_tensor.device,
        )
        std = torch.tensor(
            self.sentinel_std,
            dtype=sentinel_tensor.dtype,
            device=sentinel_tensor.device,
        )
        if mean.numel() != sentinel_tensor.shape[1] or std.numel() != sentinel_tensor.shape[1]:
            msg = (
                "sentinel_mean/std length must match Sentinel channels. "
                f"Got mean={mean.numel()}, std={std.numel()}, channels={sentinel_tensor.shape[1]}."
            )
            raise ValueError(msg)

        return (sentinel_tensor - mean.view(1, -1, 1, 1)) / (std.view(1, -1, 1, 1) + 1e-8)

    def _load_sentinel_patch(
        self,
        feature_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load Sentinel-2 patch with cloud filtering and monthly averaging.

        Args:
            feature_path: Path to the aerial image file (used for ID mapping)

        Returns:
            Tuple containing:
                - Sentinel-2 data tensor (T, C, H, W)
                - Month positions tensor (T,)

        """
        domain_zone = extract_domain_zone(feature_path)
        img_filename = feature_path.name
        centroid_x, centroid_y = self.centroids_mapping[img_filename]

        if domain_zone not in self.sentinel_data_dict:
            available = sorted(self.sentinel_data_dict.keys())
            preview = ", ".join(available[:5])
            msg = (
                f"Sentinel superpatch not found for domain/zone '{domain_zone}'. "
                f"Example available keys: {preview}{'...' if len(available) > 5 else ''}. "
                f"Check sentinel_dir structure and naming."
            )
            raise KeyError(msg)

        sp_data_path = self.sentinel_data_dict[domain_zone]
        sp_data = np.load(sp_data_path, mmap_mode="r")

        sentinel_patch = extract_sentinel_patch(
            sp_data,
            centroid_x,
            centroid_y,
            self.context_size,
        )

        sp_masks_path = self.sentinel_masks_dict.get(domain_zone)
        if sp_masks_path is None:
            logger.warning(
                "Sentinel masks not found for %s, skipping cloud filtering",
                domain_zone,
            )
            if self.use_monthly_average:
                num_timesteps = sentinel_patch.shape[0]
                positions = torch.arange(num_timesteps) % 12
            else:
                product_names = load_sentinel_dates(self.sentinel_dates_dict[domain_zone])
                day_of_year = [parse_sentinel_day_of_year(name) for name in product_names]
                positions = torch.tensor(day_of_year, dtype=torch.long)
            sentinel_tensor = torch.from_numpy(sentinel_patch).float()
            if self.sentinel_scale_factor != 1.0:
                sentinel_tensor = sentinel_tensor / self.sentinel_scale_factor
            sentinel_tensor = self._normalize_sentinel(sentinel_tensor)
            return sentinel_tensor, positions

        sp_masks = np.load(sp_masks_path)
        masks_patch = extract_sentinel_patch(
            sp_masks,
            centroid_x,
            centroid_y,
            self.sentinel_patch_size,
        )

        valid_timesteps = filter_cloudy_snowy_timesteps(
            masks_patch,
            self.cloud_snow_cover_threshold,
            self.cloud_snow_prob_threshold,
        )

        if len(valid_timesteps) == 0:
            logger.warning(
                "No valid timesteps found for %s, using all timesteps",
                feature_path.name,
            )
            valid_timesteps = list(range(sentinel_patch.shape[0]))
        else:
            sentinel_patch = sentinel_patch[valid_timesteps]
            masks_patch = masks_patch[valid_timesteps]

        if self.use_monthly_average:
            product_names = load_sentinel_dates(self.sentinel_dates_dict[domain_zone])
            product_names = [product_names[i] for i in valid_timesteps]
            sentinel_patch, month_indices = compute_monthly_averages(
                sentinel_patch,
                masks_patch,
                product_names,
                self.cloud_snow_cover_threshold,
                self.cloud_snow_prob_threshold,
            )
            month_positions = torch.tensor(month_indices, dtype=torch.long)
        else:
            product_names = load_sentinel_dates(self.sentinel_dates_dict[domain_zone])
            if len(valid_timesteps) > 0:
                product_names = [product_names[i] for i in valid_timesteps]
            day_of_year = [parse_sentinel_day_of_year(name) for name in product_names]
            month_positions = torch.tensor(day_of_year, dtype=torch.long)

        sentinel_tensor = torch.from_numpy(sentinel_patch).float()
        if self.sentinel_scale_factor != 1.0:
            sentinel_tensor = sentinel_tensor / self.sentinel_scale_factor
        sentinel_tensor = self._normalize_sentinel(sentinel_tensor)
        return sentinel_tensor, month_positions


def multimodal_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Collate multimodal batches with variable-length Sentinel sequences.

    Handles variable-length Sentinel-2 time series by padding to the maximum
    length in the batch and creating a padding mask.

    Args:
        batch: List of samples from FlairMultimodalDataset

    Returns:
        Tuple containing:
            - aerial_images: (B, C, H, W) batched aerial images
            - sentinel_data: (B, T_max, C, H, W) padded Sentinel time series
            - masks: (B, H, W) batched ground truth masks
            - sample_ids: List of sample ID strings
            - month_positions: (B, T_max) padded temporal positions
            - pad_mask: (B, T_max) boolean mask where True = padding (invalid), False = valid

    """
    aerial_images = []
    sentinel_list = []
    masks = []
    sample_ids = []
    positions_list = []

    for aerial, sentinel, mask, sample_id, positions in batch:
        aerial_images.append(aerial)
        sentinel_list.append(sentinel)
        masks.append(mask)
        sample_ids.append(sample_id)
        positions_list.append(positions)

    # Stack aerial images and masks (same size)
    aerial_batch = torch.stack(aerial_images, dim=0)
    mask_batch = torch.stack(masks, dim=0)

    # Pad Sentinel sequences to max length
    max_len = max(s.shape[0] for s in sentinel_list)
    batch_size = len(sentinel_list)
    channels = sentinel_list[0].shape[1]
    h, w = sentinel_list[0].shape[2], sentinel_list[0].shape[3]

    sentinel_batch = torch.zeros(batch_size, max_len, channels, h, w)
    positions_batch = torch.full((batch_size, max_len), -1, dtype=torch.long)
    pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (sentinel, positions) in enumerate(zip(sentinel_list, positions_list, strict=True)):
        seq_len = sentinel.shape[0]
        sentinel_batch[i, :seq_len] = sentinel
        positions_batch[i, :seq_len] = positions
        if seq_len < max_len:
            pad_mask[i, seq_len:] = True  # True = padding (invalid)

    return aerial_batch, sentinel_batch, mask_batch, sample_ids, positions_batch, pad_mask
