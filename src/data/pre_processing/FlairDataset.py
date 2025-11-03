import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

MAX_ORIGINAL_CLASS = 12
OTHER_CLASS = 13


class FlairDataset(Dataset):
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
        load_sentinel_masks: bool = False,
        load_sentinel_dates: bool = False,
        remove_cloudy_snowy_timesteps: bool = False,
        cloud_snow_cover_threshold: float = 0.6,
        cloud_snow_prob_threshold: float = 0.5,
    ) -> None:
        """Initialize the FLAIR-2 dataset with support for aerial and Sentinel-2 data.

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
            load_sentinel_masks: Whether to load Sentinel cloud/snow masks
            load_sentinel_dates: Whether to load Sentinel product date strings
            remove_cloudy_snowy_timesteps: Whether to filter out timesteps with high
                cloud/snow cover
            cloud_snow_cover_threshold: Maximum allowed cloud/snow coverage (0-1).
                Default 0.6 (60%)
            cloud_snow_prob_threshold: Minimum probability to consider a pixel as
                cloudy/snowy (0-1). Default 0.5

        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.sentinel_dir = Path(sentinel_dir) if sentinel_dir else None
        self.num_classes = num_classes
        self.image_transform = image_transform
        self.selected_channels = selected_channels
        self.sentinel_patch_size = sentinel_patch_size
        self.use_sentinel = use_sentinel
        self.load_sentinel_masks = load_sentinel_masks
        self.load_sentinel_dates = load_sentinel_dates
        self.remove_cloudy_snowy_timesteps = remove_cloudy_snowy_timesteps
        self.cloud_snow_cover_threshold = cloud_snow_cover_threshold
        self.cloud_snow_prob_threshold = cloud_snow_prob_threshold

        self.features_dict = self._get_path_mapping(self.image_dir, "IMG_*.tif")

        if self.mask_dir:
            self.labels_dict = self._get_path_mapping(self.mask_dir, "MSK_*.tif")
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
        with Path.open(centroids_path, "r") as f:
            self.centroids_mapping = json.load(f)

        self.sentinel_data_dict = {}
        self.sentinel_masks_dict = {}
        self.sentinel_dates_dict = {}

        for sp_data_path in self.sentinel_dir.rglob("*_data.npy"):
            # Extract domain and zone from path (e.g., D004_2021/Z14_AU)
            # Path structure: .../domain/zone/sen/SEN2_*.npy
            parts = sp_data_path.parts
            domain_zone = f"{parts[-4]}/{parts[-3]}"

            self.sentinel_data_dict[domain_zone] = sp_data_path

            if self.load_sentinel_masks:
                masks_path = sp_data_path.parent / sp_data_path.name.replace(
                    "_data.npy", "_masks.npy"
                )
                if masks_path.exists():
                    self.sentinel_masks_dict[domain_zone] = masks_path

            if self.load_sentinel_dates:
                dates_path = sp_data_path.parent / sp_data_path.name.replace(
                    "_data.npy", "_products.txt"
                )
                if dates_path.exists():
                    self.sentinel_dates_dict[domain_zone] = dates_path

    def _get_unique_id(self, filename: str) -> str:
        """Extract unique ID from filename.

        Args:
            filename: Name of the file

        Returns:
            Extracted ID from the filename

        """
        match = re.search(r"(\d+)", filename)
        if match is None:
            msg = f"Could not extract numeric ID from filename: {filename}"
            raise ValueError(msg)
        return match.group(1)

    def _get_path_mapping(self, directory: Path, pattern: str) -> dict[str, Path]:
        """Create a mapping from unique IDs to file paths.

        Args:
            directory: The directory to search
            pattern: The glob pattern to match files

        Returns:
            A dictionary mapping unique IDs to file paths

        """
        return {self._get_unique_id(path.name): path for path in sorted(directory.rglob(pattern))}

    def __len__(self) -> int:
        """Return the number of image/mask pairs in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
                - aerial_image: Tensor of aerial image (C, H, W)
                - mask: Tensor of segmentation mask (H, W) [if available]
                - sentinel_data: Tensor of Sentinel-2 data (T, C, H, W) [if use_sentinel=True]
                - sentinel_masks: Tensor of cloud/snow masks (T, 2, H, W) [if load_sentinel_masks=True]
                - sample_id: String ID of the sample

        Raises:
            IndexError: If idx is out of range

        """
        if idx >= len(self):
            msg = f"Index {idx} out of range for dataset of size {len(self)}"
            raise IndexError(msg)

        sample_id = self.ids[idx]
        feature_path = self.features_dict[sample_id]

        aerial_img = tifffile.imread(feature_path)
        aerial_img = torch.from_numpy(aerial_img).float()

        if aerial_img.ndim == 3:  # noqa: PLR2004
            aerial_img = aerial_img.permute(2, 0, 1)

        if self.selected_channels is not None:
            aerial_img = aerial_img[self.selected_channels, :, :]

        if self.image_transform is not None:
            aerial_img = self.image_transform(aerial_img)

        outputs = [aerial_img]

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
        domain_zone = self._extract_domain_zone(feature_path)

        img_filename = feature_path.name
        centroid_x, centroid_y = self.centroids_mapping[img_filename]

        sp_data_path = self.sentinel_data_dict[domain_zone]
        sp_data = np.load(sp_data_path)  # Shape: (T, C, H, W)

        half_size = self.sentinel_patch_size // 2

        y_start = max(0, centroid_y - half_size)
        y_end = min(sp_data.shape[2], centroid_y + half_size)
        x_start = max(0, centroid_x - half_size)
        x_end = min(sp_data.shape[3], centroid_x + half_size)

        sentinel_patch = sp_data[:, :, y_start:y_end, x_start:x_end]

        if (
            sentinel_patch.shape[2] != self.sentinel_patch_size
            or sentinel_patch.shape[3] != self.sentinel_patch_size
        ):
            pad_y = self.sentinel_patch_size - sentinel_patch.shape[2]
            pad_x = self.sentinel_patch_size - sentinel_patch.shape[3]

            pad_top = (centroid_y - half_size) < 0
            pad_bottom = (centroid_y + half_size) > sp_data.shape[2]
            pad_left = (centroid_x - half_size) < 0
            pad_right = (centroid_x + half_size) > sp_data.shape[3]

            padding = [
                pad_x if pad_left else 0,
                pad_x if pad_right else 0,
                pad_y if pad_top else 0,
                pad_y if pad_bottom else 0,
            ]

            sentinel_patch = np.pad(
                sentinel_patch,
                ((0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1])),
                mode="reflect",
            )

        masks_patch = None
        valid_timesteps = None

        if self.remove_cloudy_snowy_timesteps or self.load_sentinel_masks:
            sp_masks_path = self.sentinel_masks_dict.get(domain_zone)
            if sp_masks_path is None:
                msg = f"Sentinel masks not found for {domain_zone}"
                raise ValueError(msg)

            sp_masks = np.load(sp_masks_path)  # Shape: (T, 2, H, W)

            y_start = max(0, centroid_y - half_size)
            y_end = min(sp_masks.shape[2], centroid_y + half_size)
            x_start = max(0, centroid_x - half_size)
            x_end = min(sp_masks.shape[3], centroid_x + half_size)

            masks_patch = sp_masks[:, :, y_start:y_end, x_start:x_end]

            if (
                masks_patch.shape[2] != self.sentinel_patch_size
                or masks_patch.shape[3] != self.sentinel_patch_size
            ):
                masks_patch = np.pad(
                    masks_patch,
                    (
                        (0, 0),
                        (0, 0),
                        (padding[2], padding[3]),
                        (padding[0], padding[1]),
                    ),
                    mode="reflect",
                )

            if self.remove_cloudy_snowy_timesteps:
                valid_timesteps = self._filter_cloudy_snowy_timesteps(masks_patch)
                sentinel_patch = sentinel_patch[valid_timesteps]
                masks_patch = masks_patch[valid_timesteps]

        return (
            torch.from_numpy(sentinel_patch).float(),
            torch.from_numpy(masks_patch).float() if masks_patch is not None else None,
            valid_timesteps,
        )

    def _filter_cloudy_snowy_timesteps(self, masks_patch: np.ndarray) -> np.ndarray:
        """Filter timesteps with high cloud or snow cover.

        Implementation follows FLAIR-2 paper:
        "We remove satellite images with a snow or cloud cover of over 60%
        according to the meta-data (with a probability threshold of 0.5)"

        Args:
            masks_patch: Cloud/snow masks array with shape (T, 2, H, W)
                        where channel 0 is cloud probability, channel 1 is snow probability

        Returns:
            Array of timestep indices to keep

        """
        num_timesteps = masks_patch.shape[0]
        total_pixels = masks_patch.shape[2] * masks_patch.shape[3]
        valid_timesteps = []

        for t in range(num_timesteps):
            max_prob = np.maximum(masks_patch[t, 0, :, :], masks_patch[t, 1, :, :])
            covered_pixels = np.count_nonzero(max_prob >= self.cloud_snow_prob_threshold)
            coverage_ratio = covered_pixels / total_pixels

            if coverage_ratio < self.cloud_snow_cover_threshold:
                valid_timesteps.append(t)

        return np.array(valid_timesteps, dtype=np.int64)

    def _extract_domain_zone(self, file_path: Path) -> str:
        """Extract domain and zone from file path.

        Args:
            file_path: Path to the file

        Returns:
            String in format "DOMAIN/ZONE" (e.g., "D004_2021/Z14_AU")

        """
        parts = file_path.parts
        # Path structure: .../domain/zone/img/IMG_*.tif
        return f"{parts[-4]}/{parts[-3]}"

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
