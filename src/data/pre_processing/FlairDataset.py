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
    def __init__(self, image_dir: str, mask_dir: str,
                 num_classes: int,
                 image_transform: Callable | None = None,
                 selected_channels: list[int] | None = None,
                ) -> None:
        """Initialize the FLAIR dataset.

        Args:
            image_dir: Directory containing image files (IMG_*.tif)
            mask_dir: Directory containing mask files (MSK_*.tif)
            num_classes: Number of segmentation classes
            image_transform: Optional transform to apply to images
            selected_channels: Optional list of channels to select from images

        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.num_classes = num_classes
        self.image_transform = image_transform
        self.selected_channels = selected_channels

        # mapping from unique ids to file paths to image and mask files
        self.features_dict = self._get_path_mapping(self.image_dir, "IMG_*.tif")
        self.labels_dict = self._get_path_mapping(self.mask_dir, "MSK_*.tif")

        shared_ids = set(self.features_dict.keys()) & set(self.labels_dict.keys())
        self.ids = sorted(shared_ids)

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
        return {self._get_unique_id(path.name): path for path in
                sorted(directory.rglob(pattern))}

    def __len__(self) -> int:
        """Return the number of image/mask pairs in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, mask) where mask contains integer class indices

        Raises:
            IndexError: If idx is out of range

        """
        if idx >= len(self):
            msg = f"Index {idx} out of range for dataset of size {len(self)}"
            raise IndexError(msg)

        sample_id = self.ids[idx]
        feature_path = self.features_dict[sample_id]
        label_path = self.labels_dict[sample_id]

        x = tifffile.imread(feature_path)
        y = tifffile.imread(label_path)

        y = np.where(y <= MAX_ORIGINAL_CLASS, y, OTHER_CLASS)
        y -= 1

        x = torch.from_numpy(x).float()

        if x.ndim == 3:
            x = x.permute(2, 0, 1)

        if self.selected_channels is not None:
            x = x[self.selected_channels, :, :]

        y = torch.from_numpy(y).long()

        if self.image_transform is not None:
            x = self.image_transform(x)

        return x, y, sample_id

    def get_class_counts(self) -> torch.Tensor:
        """Calculate the distribution of classes in the dataset efficiently.

        This version avoids loading images and uses torch.bincount directly
        on the integer mask representation, skipping one-hot encoding.

        Returns:
            Tensor with count of pixels for each class.

        """
        class_counts = torch.zeros(self.num_classes, dtype=torch.int64)
        for id_ in tqdm(self.ids):
            label_path = self.labels_dict.get(id_)

            y = tifffile.imread(label_path)
            y = np.where(y <= MAX_ORIGINAL_CLASS, y, OTHER_CLASS)
            y -= 1
            y_tensor = torch.from_numpy(y).long()
            y_flat = y_tensor.flatten()
            counts = torch.bincount(y_flat, minlength=self.num_classes)
            class_counts += counts

        return class_counts
