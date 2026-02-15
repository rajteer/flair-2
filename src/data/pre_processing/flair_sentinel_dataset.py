"""Sentinel-only dataset for FLAIR-2 land cover segmentation experiments."""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.dataset_utils import get_path_mapping

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
    parse_sentinel_date,
    parse_sentinel_days_from_reference,
)

logger = logging.getLogger(__name__)


class FlairSentinelDataset(Dataset):
    """Dataset for Sentinel-2 only land cover segmentation experiments.

    This dataset loads only Sentinel-2 satellite imagery and downsamples ground truth
    masks to match the Sentinel spatial resolution. It's designed for experiments that
    focus on satellite-only classification without aerial imagery.
    """

    def __init__(
        self,
        mask_dir: str,
        sentinel_dir: str,
        centroids_path: str,
        num_classes: int = 13,
        sentinel_patch_size: int = 10,
        *,
        context_size: int | None = None,
        use_monthly_average: bool = True,
        filter_clouds_snow: bool = True,
        cloud_snow_cover_threshold: float = 0.6,
        cloud_snow_prob_threshold: int = 50,
        sentinel_scale_factor: float = 10000.0,
        sentinel_mean: list[float] | None = None,
        sentinel_std: list[float] | None = None,
        date_encoding_mode: str = "days",
        downsample_masks: bool = True,
    ) -> None:
        """Initialize the Sentinel-only FLAIR-2 dataset.

        Args:
            mask_dir: Directory containing mask files (MSK_*.tif)
            sentinel_dir: Directory containing Sentinel-2 superpatch data
            centroids_path: Path to JSON file mapping image IDs to superpatch coordinates
            num_classes: Number of segmentation classes (default: 13)
            sentinel_patch_size: Size of the output Sentinel-2 patch (center crop).
            context_size: Size of the input Sentinel-2 patch to extract. Must be >= sentinel_patch_size.
                If None, defaults to sentinel_patch_size (no extra context).
                Use larger values (e.g., 20, 40) to provide spatial context during training.
            use_monthly_average: Whether to compute monthly averages from cloudless dates.
                Reduces temporal variability and produces up to 12 monthly images.
            filter_clouds_snow: Whether to filter timesteps using cloud/snow masks.
                If False, all timesteps are kept.
            cloud_snow_cover_threshold: Maximum allowed cloud/snow coverage (0-1).
                Default 0.6 (60%) as per FLAIR-2 paper.
            cloud_snow_prob_threshold: Minimum probability (0-100) to consider a pixel
                as cloudy/snowy. Default 50 (50%) as per FLAIR-2 paper.
            sentinel_scale_factor: Factor to divide Sentinel-2 reflectance values by.
                Default 10000.0 scales reflectance (0-10000) to [0, 1].
                Set to 1.0 to disable scaling (use with sentinel_mean/std).
            sentinel_mean: Per-channel means for z-score normalization (after scale_factor).
                If provided with sentinel_std, applies (x - mean) / std normalization.
            sentinel_std: Per-channel stds for z-score normalization (after scale_factor).
                If provided with sentinel_mean, applies (x - mean) / std normalization.

        """
        self.mask_dir = Path(mask_dir)
        self.sentinel_dir = Path(sentinel_dir)
        self.num_classes = num_classes
        self.sentinel_patch_size = sentinel_patch_size
        self.context_size = context_size if context_size is not None else sentinel_patch_size
        self.use_monthly_average = use_monthly_average
        self.filter_clouds_snow = filter_clouds_snow
        self.cloud_snow_cover_threshold = cloud_snow_cover_threshold
        self.cloud_snow_prob_threshold = cloud_snow_prob_threshold
        self.sentinel_scale_factor = sentinel_scale_factor
        if (sentinel_mean is None) != (sentinel_std is None):
            msg = "sentinel_mean and sentinel_std must be provided together or omitted."
            raise ValueError(msg)
        if sentinel_mean is not None and sentinel_std is not None:
            self.sentinel_mean = torch.tensor(sentinel_mean, dtype=torch.float32)
            self.sentinel_std = torch.tensor(sentinel_std, dtype=torch.float32)
        else:
            self.sentinel_mean = None
            self.sentinel_std = None

        self.date_encoding_mode = date_encoding_mode
        self.downsample_masks = downsample_masks

        self.labels_dict = get_path_mapping(self.mask_dir, "MSK_*.tif")
        self.ids = sorted(self.labels_dict.keys())

        self.features_dict = {
            id_: Path(str(path).replace("MSK_", "IMG_").replace("labels", "aerial"))
            for id_, path in self.labels_dict.items()
        }

        self.centroids_mapping = load_centroids_mapping(centroids_path)
        (
            self.sentinel_data_dict,
            self.sentinel_masks_dict,
            self.sentinel_dates_dict,
        ) = load_sentinel_superpatch_paths(
            self.sentinel_dir,
            load_masks=self.filter_clouds_snow,
            load_dates=True,
        )

        original_count = len(self.ids)
        valid_ids = []
        for sample_id in self.ids:
            feature_path = self.features_dict[sample_id]
            try:
                domain_zone = extract_domain_zone(feature_path)
                if domain_zone in self.sentinel_data_dict:
                    valid_ids.append(sample_id)
            except ValueError:
                continue

        self.ids = valid_ids
        filtered_count = original_count - len(self.ids)
        if filtered_count > 0:
            logger.warning(
                "Filtered %d samples (out of %d) without matching Sentinel data",
                filtered_count,
                original_count,
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
                - sentinel_data: Tensor with shape (M, C, H, W) where M is the number
                  of months with valid cloudless data (≤12 for single year, can be
                  larger for multi-year datasets)
                - mask: Tensor of shape (512, 512) - original FLAIR mask resolution
                - sample_id: String ID of the sample
                - month_positions: Tensor of shape (M,) with month indices (0-11)

        Raises:
            IndexError: If idx is out of range

        """
        if abs(idx) >= len(self):
            msg = f"Index {idx} out of range for dataset of size {len(self)}"
            raise IndexError(msg)

        sample_id = self.ids[idx]
        feature_path = self.features_dict[sample_id]

        sentinel_data, month_positions = self._load_sentinel_patch(feature_path)

        label_path = self.labels_dict[sample_id]
        mask = tifffile.imread(label_path)
        mask = np.where(mask <= MAX_ORIGINAL_CLASS, mask, OTHER_CLASS)
        mask -= 1
        if self.downsample_masks:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor,
                size=(self.sentinel_patch_size, self.sentinel_patch_size),
                mode="nearest",
            )
            mask = mask_tensor.squeeze().long()
        else:
            mask = torch.from_numpy(mask.copy()).long()

        return sentinel_data, mask, sample_id, month_positions

    def _load_sentinel_patch(self, feature_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Load Sentinel-2 patch with cloud filtering and monthly averaging.

        Args:
            feature_path: Path to the aerial image file (used for ID mapping)

        Returns:
            Tuple containing:
                - Sentinel-2 data tensor with shape (T, C, H, W) where T is the number
                  of valid timesteps after cloud filtering and optional monthly averaging
                - Month positions tensor with shape (T,) containing month indices (0-11)

        """
        domain_zone = extract_domain_zone(feature_path)
        img_filename = feature_path.name
        centroid_x, centroid_y = self.centroids_mapping[img_filename]

        sp_data_path = self.sentinel_data_dict[domain_zone]
        sp_data = np.load(sp_data_path, mmap_mode="r")  # Shape: (T, C, H, W)

        sentinel_patch = extract_sentinel_patch(
            sp_data,
            centroid_x,
            centroid_y,
            self.context_size,
        )

        sp_masks_path = self.sentinel_masks_dict.get(domain_zone)
        if sp_masks_path is None:
            if self.filter_clouds_snow:
                logger.warning(
                    "Sentinel masks not found for %s, skipping cloud filtering",
                    domain_zone,
                )
            # Return day-of-year or month positions as fallback when masks unavailable
            if self.use_monthly_average:
                # Use month indices (0-11) as fallback
                num_timesteps = sentinel_patch.shape[0]
                positions = torch.arange(num_timesteps) % 12
            else:
                # Use days from reference date (FLAIR-2 style)
                product_names = load_sentinel_dates(self.sentinel_dates_dict[domain_zone])
                days_from_ref = [parse_sentinel_days_from_reference(name) for name in product_names]
                positions = torch.tensor(days_from_ref, dtype=torch.long)
            sentinel_tensor = torch.from_numpy(sentinel_patch).float()
            if self.sentinel_scale_factor != 1.0:
                sentinel_tensor = sentinel_tensor / self.sentinel_scale_factor
            sentinel_tensor = self._normalize_sentinel(sentinel_tensor)
            return sentinel_tensor, positions

        sp_masks = np.load(sp_masks_path)  # Shape: (T, 2, H, W)

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
                "No valid timesteps found for %s in %s after cloud/snow "
                "filtering. Using all timesteps instead.",
                feature_path.name,
                domain_zone,
            )
            # Use all timesteps as fallback
            valid_timesteps = list(range(sentinel_patch.shape[0]))
        else:
            sentinel_patch = sentinel_patch[valid_timesteps]
            masks_patch = masks_patch[valid_timesteps]

        if self.use_monthly_average:
            product_names = load_sentinel_dates(self.sentinel_dates_dict[domain_zone])
            product_names = [product_names[i] for i in valid_timesteps]
            sentinel_patch, month_indices = compute_monthly_averages(
                sentinel_patch,
                product_names,
            )
            if self.date_encoding_mode == "month":
                # Use month indices directly (0-11) for TSViT
                month_positions = torch.tensor(month_indices, dtype=torch.long)
            else:
                # Convert month indices to days from reference (FLAIR-2 style) for UTAE
                ref_month = 5  # May
                ref_day = 15
                days_from_ref = []
                for m in month_indices:
                    actual_month = m + 1
                    ref = date(2021, ref_month, ref_day)
                    target = date(2021, actual_month, 15)
                    days_from_ref.append((ref - target).days)
                month_positions = torch.tensor(days_from_ref, dtype=torch.long)
        else:
            # Non-averaged sequences
            product_names = load_sentinel_dates(self.sentinel_dates_dict[domain_zone])
            if len(valid_timesteps) > 0:
                product_names = [product_names[i] for i in valid_timesteps]
            if self.date_encoding_mode == "month":
                # Extract month indices (0-11) for TSViT

                month_indices = [parse_sentinel_date(name)[1] - 1 for name in product_names]
                month_positions = torch.tensor(month_indices, dtype=torch.long)
            else:
                # Use days from reference for UTAE
                days_from_ref = [parse_sentinel_days_from_reference(name) for name in product_names]
                month_positions = torch.tensor(days_from_ref, dtype=torch.long)

        sentinel_tensor = torch.from_numpy(sentinel_patch).float()
        if self.sentinel_scale_factor != 1.0:
            sentinel_tensor = sentinel_tensor / self.sentinel_scale_factor
        sentinel_tensor = self._normalize_sentinel(sentinel_tensor)
        return sentinel_tensor, month_positions

    def _normalize_sentinel(self, sentinel_tensor: torch.Tensor) -> torch.Tensor:
        if self.sentinel_mean is None or self.sentinel_std is None:
            return sentinel_tensor

        mean = self.sentinel_mean.to(dtype=sentinel_tensor.dtype, device=sentinel_tensor.device)
        std = self.sentinel_std.to(dtype=sentinel_tensor.dtype, device=sentinel_tensor.device)
        if mean.numel() != sentinel_tensor.shape[1] or std.numel() != sentinel_tensor.shape[1]:
            msg = (
                "sentinel_mean/std length must match Sentinel channels. "
                f"Got mean={mean.numel()}, std={std.numel()}, channels={sentinel_tensor.shape[1]}."
            )
            raise ValueError(msg)

        return (sentinel_tensor - mean.view(1, -1, 1, 1)) / (std.view(1, -1, 1, 1) + 1e-8)

    def get_class_counts(self) -> torch.Tensor:
        """Calculate the distribution of classes in the dataset at Sentinel resolution.

        Returns:
            Tensor with count of pixels for each class.

        """
        class_counts = torch.zeros(self.num_classes, dtype=torch.int64)
        for idx in tqdm(range(len(self)), desc="Computing class counts"):
            _, mask, _, _ = self[idx]
            counts = torch.bincount(mask.flatten(), minlength=self.num_classes)
            class_counts += counts

        return class_counts
