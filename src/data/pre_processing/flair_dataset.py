import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

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
)

logger = logging.getLogger(__name__)


class MultiChannelNormalize:
    """Normalize multi-channel aerial imagery with per-channel stats.

    This transform expects an input tensor of shape (C, H, W).

    Typical usage for 5 channels [R, G, B, NIR, elevation]:

    - RGB channels use ImageNet mean/std in [0, 1]
    - NIR and elevation use user-provided statistics

    Args:
        mean: Per-channel mean values.
        std: Per-channel standard deviation values.
        scale_to_unit: Optional boolean mask (length C). For channels where the
            mask is True, values are assumed to be in [0, 255] and will be
            divided by 255 before normalization.
        elevation_range: Optional (min, max) tuple specifying raw elevation
            range. If provided, the elevation channel will be scaled to
            [0, 1] using this range before applying mean/std. This should be
            consistent with one of the channels in ``mean`` / ``std``.

    """

    def __init__(
        self,
        mean: list[float] | tuple[float, ...],
        std: list[float] | tuple[float, ...],
        scale_to_unit: list[bool] | tuple[bool, ...] | None = None,
        elevation_range: tuple[float, float] | None = None,
        elevation_channel_index: int | None = None,
    ) -> None:
        if len(mean) != len(std):
            msg = "mean and std must have the same length"
            raise ValueError(msg)

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

        if scale_to_unit is not None:
            if len(scale_to_unit) != len(mean):
                msg = "scale_to_unit must have the same length as mean/std"
                raise ValueError(msg)
            self.scale_to_unit = torch.tensor(scale_to_unit, dtype=torch.bool)
        else:
            self.scale_to_unit = None

        if (elevation_range is None) != (elevation_channel_index is None):
            msg = "Both elevation_range and elevation_channel_index must be provided together or omitted."
            raise ValueError(msg)

        self.elevation_range = elevation_range
        self.elevation_channel_index = elevation_channel_index

    def __call__(self, img: Tensor) -> Tensor:
        if img.ndim != 3:  # (C, H, W)
            msg = "MultiChannelNormalize expects tensor of shape (C, H, W)"
            raise ValueError(msg)

        img = img.float().clone()

        if self.scale_to_unit is not None:
            img[self.scale_to_unit, ...] = img[self.scale_to_unit, ...] / 255.0

        if self.elevation_range is not None and self.elevation_channel_index is not None:
            elev_min, elev_max = self.elevation_range
            if elev_max <= elev_min:
                msg = "elevation_range must satisfy max > min"
                raise ValueError(msg)
            c = self.elevation_channel_index
            img[c] = (img[c] - elev_min) / (elev_max - elev_min)

        return (img - self.mean) / self.std


class FlairDataset(Dataset):
    """FLAIR-2 dataset for aerial image semantic segmentation with optional Sentinel-2 fusion.

    This is the main dataset class for loading aerial imagery along with optional
    Sentinel-2 satellite data for multi-modal fusion experiments.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str | None = None,
        sentinel_dir: str | None = None,
        centroids_path: str | None = None,
        num_classes: int = 13,
        image_transform: Callable | None = None,
        selected_channels: list[int] | None = None,
        sentinel_patch_size: int = 40,
        *,
        use_sentinel: bool = False,
        remove_cloudy_snowy_timesteps: bool = False,
        use_monthly_average: bool = False,
        cloud_snow_cover_threshold: float = 0.6,
        cloud_snow_prob_threshold: int = 50,
    ) -> None:
        """Initialize the FLAIR-2 dataset.

        Args:
            image_dir: Directory containing aerial image files (IMG_*.tif)
            mask_dir: Directory containing mask files (MSK_*.tif). Can be None for test sets.
            sentinel_dir: Directory containing Sentinel-2 superpatch data
            centroids_path: Path to JSON file mapping image IDs to superpatch coordinates
            num_classes: Number of segmentation classes
            image_transform: Optional transform to apply to images
            selected_channels: Optional list of channels to select from aerial images
            sentinel_patch_size: Size of Sentinel-2 patch to extract (in Sentinel pixels)
            use_sentinel: Whether to load Sentinel-2 data
            remove_cloudy_snowy_timesteps: Whether to filter out timesteps with high
                cloud/snow cover
            use_monthly_average: Whether to compute monthly averages from cloudless
                dates. Returns up to 12 monthly averaged images (one per month with data).
                cloud_snow_cover_threshold: Maximum allowed cloud/snow coverage (0-1).
                Default 0.6 (60%)
            cloud_snow_prob_threshold: Minimum probability value (0-100) to consider a
                pixel as cloudy/snowy. Default 50 (50%)

        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.sentinel_dir = Path(sentinel_dir) if sentinel_dir else None
        self.num_classes = num_classes
        self.image_transform = image_transform
        self.selected_channels = selected_channels
        self.sentinel_patch_size = sentinel_patch_size
        self.use_sentinel = use_sentinel
        self.remove_cloudy_snowy_timesteps = remove_cloudy_snowy_timesteps
        self.use_monthly_average = use_monthly_average
        self.cloud_snow_cover_threshold = cloud_snow_cover_threshold
        self.cloud_snow_prob_threshold = cloud_snow_prob_threshold

        self.load_sentinel_masks = remove_cloudy_snowy_timesteps or use_monthly_average
        self.load_sentinel_dates = use_monthly_average

        if self.use_monthly_average and self.remove_cloudy_snowy_timesteps:
            logger.warning(
                "Both use_monthly_average and remove_cloudy_snowy_timesteps are enabled. "
                "Monthly averaging will be applied after filtering cloudy/snowy timesteps.",
            )

        self.features_dict = get_path_mapping(self.image_dir, "IMG_*.tif")

        if self.mask_dir:
            self.labels_dict = get_path_mapping(self.mask_dir, "MSK_*.tif")
            shared_ids = set(self.features_dict.keys()) & set(self.labels_dict.keys())
            self.ids = sorted(shared_ids)
        else:
            self.labels_dict = {}
            self.ids = sorted(self.features_dict.keys())

        if self.use_sentinel:
            if not self.sentinel_dir or not centroids_path:
                msg = "sentinel_dir and centroids_path are required when use_sentinel=True"
                raise ValueError(msg)
            self._load_sentinel_data(centroids_path)

    def _load_sentinel_data(self, centroids_path: str) -> None:
        """Load Sentinel-2 superpatch data and centroid mappings.

        Args:
            centroids_path: Path to JSON file mapping image IDs to superpatch coordinates

        """
        self.centroids_mapping = load_centroids_mapping(centroids_path)
        (
            self.sentinel_data_dict,
            self.sentinel_masks_dict,
            self.sentinel_dates_dict,
        ) = load_sentinel_superpatch_paths(
            self.sentinel_dir,
            load_masks=self.load_sentinel_masks,
            load_dates=self.load_sentinel_dates,
        )

    def __len__(self) -> int:
        """Return the number of image/mask pairs in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
                - aerial_image: Tensor of shape (C, H, W) at aerial resolution
                - mask: Tensor of shape (H, W) at aerial resolution
                - sentinel_data (optional): Tensor of shape (T, C, H, W)
                  if use_sentinel=True
                - sentinel_masks (optional): Tensor of shape (T, 2, H, W)
                  if load_sentinel_masks=True
                - sample_id: String ID of the sample

        Raises:
            IndexError: If idx is out of range

        """
        if abs(idx) >= len(self):
            msg = f"Index {idx} out of range for dataset of size {len(self)}"
            raise IndexError(msg)

        sample_id = self.ids[idx]
        feature_path = self.features_dict[sample_id]
        outputs = []

        aerial_img = tifffile.imread(feature_path)
        aerial_img = torch.from_numpy(aerial_img).float()

        if aerial_img.ndim == 3:  # noqa: PLR2004
            aerial_img = aerial_img.permute(2, 0, 1)

        if self.selected_channels is not None:
            aerial_img = aerial_img[self.selected_channels, :, :]

        if self.image_transform is not None:
            aerial_img = self.image_transform(aerial_img)

        outputs.append(aerial_img)

        if self.mask_dir:
            label_path = self.labels_dict[sample_id]
            mask = tifffile.imread(label_path)
            mask = np.where(mask <= MAX_ORIGINAL_CLASS, mask, OTHER_CLASS)
            mask -= 1
            mask = torch.from_numpy(mask).long()
            outputs.append(mask)

        if self.use_sentinel:
            sentinel_data, sentinel_masks, _ = self._load_sentinel_patch(feature_path)
            outputs.append(sentinel_data)

            if self.load_sentinel_masks:
                outputs.append(sentinel_masks)

        outputs.append(sample_id)

        return tuple(outputs)

    def _load_sentinel_patch(
        self,
        feature_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor | None, np.ndarray | None]:
        """Load Sentinel-2 patch for a given aerial image with optional cloud filtering.

        Args:
            feature_path: Path to the aerial image file

        Returns:
            Tuple of:
            - Sentinel-2 data tensor with shape (T, C, H, W)
            - Cloud/snow masks tensor with shape (T, 2, H, W) or None
            - Valid timestep indices array or None

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
            self.sentinel_patch_size,
        )

        masks_patch = None
        valid_timesteps = None

        if self.remove_cloudy_snowy_timesteps or self.load_sentinel_masks:
            sp_masks_path = self.sentinel_masks_dict.get(domain_zone)
            if sp_masks_path is None:
                msg = f"Sentinel masks not found for {domain_zone}"
                raise ValueError(msg)

            sp_masks = np.load(sp_masks_path, mmap_mode="r")  # Shape: (T, 2, H, W)

            masks_patch = extract_sentinel_patch(
                sp_masks,
                centroid_x,
                centroid_y,
                self.sentinel_patch_size,
            )

            if self.remove_cloudy_snowy_timesteps:
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

            if valid_timesteps is not None:
                product_names = [product_names[i] for i in valid_timesteps]

            sentinel_patch, masks_patch = compute_monthly_averages(
                sentinel_patch,
                masks_patch,
                product_names,
                self.cloud_snow_cover_threshold,
                self.cloud_snow_prob_threshold,
            )

        return (
            torch.from_numpy(sentinel_patch).float(),
            torch.from_numpy(masks_patch).float() if masks_patch is not None else None,
            valid_timesteps,
        )

    def get_class_counts(self) -> torch.Tensor:
        """Calculate the distribution of classes in the dataset efficiently.

        This version avoids loading images and uses torch.bincount directly
        on the integer mask representation, skipping one-hot encoding.

        Returns:
            Tensor with count of pixels for each class.

        Raises:
            ValueError: If masks are not available for the dataset

        """
        if not self.mask_dir:
            msg = "Cannot compute class counts without mask directory"
            raise ValueError(msg)

        class_counts = torch.zeros(self.num_classes, dtype=torch.int64)
        for id_ in tqdm(self.ids, desc="Computing class counts"):
            label_path = self.labels_dict.get(id_)

            y = tifffile.imread(label_path)
            y = np.where(y <= MAX_ORIGINAL_CLASS, y, OTHER_CLASS)
            y -= 1
            y_tensor = torch.from_numpy(y).long()
            y_flat = y_tensor.flatten()
            counts = torch.bincount(y_flat, minlength=self.num_classes)
            class_counts += counts

        return class_counts
