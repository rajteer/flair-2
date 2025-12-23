"""Shared utilities for Sentinel-2 data processing in FLAIR-2 dataset.

This module contains functions used by both FlairDataset and FlairSentinelDataset
for loading, filtering, and processing Sentinel-2 satellite imagery.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MAX_ORIGINAL_CLASS = 12
OTHER_CLASS = 13


def load_centroids_mapping(centroids_path: str | Path) -> dict[str, tuple[int, int]]:
    """Load centroid coordinates mapping from JSON file.

    Args:
        centroids_path: Path to JSON file mapping image filenames to centroid coordinates

    Returns:
        Dictionary mapping image filenames to (x, y) centroid coordinates

    """
    with Path(centroids_path).open(encoding="utf-8") as f:
        return json.load(f)


def load_sentinel_superpatch_paths(
    sentinel_dir: Path,
    *,
    load_masks: bool = True,
    load_dates: bool = True,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Load paths to Sentinel-2 superpatch data, masks, and dates.

    Args:
        sentinel_dir: Directory containing Sentinel-2 superpatch data
        load_masks: Whether to load cloud/snow mask paths
        load_dates: Whether to load product date file paths

    Returns:
        Tuple of three dictionaries mapping domain_zone to paths:
        - sentinel_data_dict: Paths to *_data.npy files
        - sentinel_masks_dict: Paths to *_masks.npy files (empty if load_masks=False)
        - sentinel_dates_dict: Paths to *_products.txt files (empty if load_dates=False)

    """
    sentinel_data_dict = {}
    sentinel_masks_dict = {}
    sentinel_dates_dict = {}

    for sp_data_path in sentinel_dir.rglob("*_data.npy"):
        parts = sp_data_path.parts
        domain_zone = f"{parts[-4]}/{parts[-3]}"

        sentinel_data_dict[domain_zone] = sp_data_path

        if load_masks:
            masks_path = sp_data_path.parent / sp_data_path.name.replace(
                "_data.npy",
                "_masks.npy",
            )
            if masks_path.exists():
                sentinel_masks_dict[domain_zone] = masks_path

        if load_dates:
            dates_path = sp_data_path.parent / sp_data_path.name.replace(
                "_data.npy",
                "_products.txt",
            )
            if dates_path.exists():
                sentinel_dates_dict[domain_zone] = dates_path

    return sentinel_data_dict, sentinel_masks_dict, sentinel_dates_dict


def extract_domain_zone(file_path: Path) -> str:
    """Extract domain and zone from file path.

    Args:
        file_path: Path to the file (e.g., .../D004_2021/Z14_AU/img/IMG_*.tif)

    Returns:
        String in format "DOMAIN/ZONE" (e.g., "D004_2021/Z14_AU")

    """
    parts = file_path.parts
    # Path structure: .../domain/zone/img/IMG_*.tif
    return f"{parts[-4]}/{parts[-3]}"


def parse_sentinel_date(product_name: str) -> tuple[int, int]:
    """Parse year and month from Sentinel-2 product name.

    Args:
        product_name: Sentinel-2 product name (e.g., "S2B_MSIL2A_20210114T103309_...")

    Returns:
        Tuple of (year, month)

    Raises:
        ValueError: If date cannot be parsed from product name

    """
    match = re.search(r"_(\d{8})T", product_name)
    if match is None:
        msg = f"Could not extract date from product name: {product_name}"
        raise ValueError(msg)

    date_str = match.group(1)
    year = int(date_str[:4])
    month = int(date_str[4:6])
    return year, month


def parse_sentinel_day_of_year(product_name: str) -> int:
    """Parse day-of-year (0-365) from Sentinel-2 product name.

    Args:
        product_name: Sentinel-2 product name (e.g., "S2B_MSIL2A_20210114T103309_...")

    Returns:
        Day of year (0-365, where Jan 1 = 0)

    Raises:
        ValueError: If date cannot be parsed from product name

    """
    match = re.search(r"_(\d{8})T", product_name)
    if match is None:
        msg = f"Could not extract date from product name: {product_name}"
        raise ValueError(msg)

    date_str = match.group(1)
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    return date(year, month, day).timetuple().tm_yday - 1


def load_sentinel_dates(dates_path: Path) -> list[str]:
    """Load Sentinel-2 product dates for a domain/zone.

    Args:
        dates_path: Path to the dates file

    Returns:
        List of product name strings

    """
    with Path.open(dates_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def filter_cloudy_snowy_timesteps(
    masks_patch: np.ndarray,
    cloud_snow_cover_threshold: float = 0.6,
    cloud_snow_prob_threshold: int = 50,
) -> np.ndarray:
    """Filter timesteps with high cloud or snow cover.

    Args:
        masks_patch: Cloud/snow masks array with shape (T, 2, H, W)
                    where channel 0 is cloud probability (0-100),
                    channel 1 is snow probability (0-100)
        cloud_snow_cover_threshold: Maximum allowed cloud/snow coverage (0-1).
            Default 0.6 (60%)
        cloud_snow_prob_threshold: Minimum probability value (0-100) to consider a
            pixel as cloudy/snowy. Default 50 (50%)

    Returns:
        Array of timestep indices to keep. Returns empty array if no valid
        timesteps found.

    """
    num_timesteps = masks_patch.shape[0]
    total_pixels = masks_patch.shape[2] * masks_patch.shape[3]
    valid_timesteps = []

    for t in range(num_timesteps):
        max_prob = np.maximum(masks_patch[t, 0, :, :], masks_patch[t, 1, :, :])
        covered_pixels = np.count_nonzero(max_prob >= cloud_snow_prob_threshold)
        coverage_ratio = covered_pixels / total_pixels

        if coverage_ratio < cloud_snow_cover_threshold:
            valid_timesteps.append(t)

    return np.array(valid_timesteps, dtype=np.int64)


def compute_monthly_averages(
    sentinel_patch: np.ndarray,
    masks_patch: np.ndarray,
    product_names: list[str],
    cloud_snow_cover_threshold: float = 0.6,
    cloud_snow_prob_threshold: int = 50,
) -> tuple[np.ndarray, list[int]]:
    """Compute monthly averages from cloudless Sentinel-2 timesteps.

    Following the FLAIR-2 paper strategy: compute monthly average using
    cloudless dates. If no cloudless dates are available for a specific
    month, that month is skipped (resulting in fewer than 12 images).

    Args:
        sentinel_patch: Sentinel-2 data with shape (T, C, H, W)
        masks_patch: Cloud/snow masks with shape (T, 2, H, W)
        product_names: List of Sentinel-2 product names (length T)
        cloud_snow_cover_threshold: Maximum allowed cloud/snow coverage (0-1)
        cloud_snow_prob_threshold: Minimum probability (0-100) to consider pixel
            as cloudy/snowy

    Returns:
        Tuple of:
        - Monthly averaged Sentinel-2 data with shape (M, C, H, W) where M <= 12
        - List of month indices (0-11) corresponding to each averaged timestep

    """
    monthly_timesteps = defaultdict(list)
    for t, product_name in enumerate(product_names):
        year, month = parse_sentinel_date(product_name)
        monthly_timesteps[(year, month)].append(t)

    monthly_cloudless = {}

    for (year, month), timesteps in monthly_timesteps.items():
        cloudless_timesteps = []

        for t in timesteps:
            max_prob = np.maximum(
                masks_patch[t, 0, :, :],
                masks_patch[t, 1, :, :],
            )
            total_pixels = masks_patch.shape[2] * masks_patch.shape[3]
            covered_pixels = np.count_nonzero(
                max_prob >= cloud_snow_prob_threshold,
            )
            coverage_ratio = covered_pixels / total_pixels

            if coverage_ratio < cloud_snow_cover_threshold:
                cloudless_timesteps.append(t)

        if cloudless_timesteps:
            monthly_cloudless[(year, month)] = cloudless_timesteps

    # Fallback: if no cloudless timesteps found, use all timesteps grouped by month
    if not monthly_cloudless:
        monthly_cloudless = dict(monthly_timesteps)

    sorted_months = sorted(monthly_cloudless.keys())
    monthly_data_list = []
    month_indices = []

    for year_month in sorted_months:
        timesteps = monthly_cloudless[year_month]
        year, month = year_month

        month_data = np.mean(sentinel_patch[timesteps], axis=0)
        monthly_data_list.append(month_data)
        month_indices.append(month - 1)  # Convert 1-12 to 0-11

    monthly_data = np.stack(monthly_data_list, axis=0)  # Shape: (M, C, H, W)

    return monthly_data, month_indices


def extract_sentinel_patch(
    sp_data: np.ndarray,
    centroid_x: int,
    centroid_y: int,
    patch_size: int,
) -> np.ndarray:
    """Extract a patch from Sentinel superpatch around given centroid.

    Args:
        sp_data: Sentinel superpatch data with shape (T, C, H, W) or (T, 2, H, W)
        centroid_x: X coordinate of patch center
        centroid_y: Y coordinate of patch center
        patch_size: Size of patch to extract (in pixels)

    Returns:
        Extracted and padded patch with shape (T, C, patch_size, patch_size)

    """
    half_size = patch_size // 2

    y_start = max(0, centroid_y - half_size)
    y_end = min(sp_data.shape[2], centroid_y + half_size)
    x_start = max(0, centroid_x - half_size)
    x_end = min(sp_data.shape[3], centroid_x + half_size)

    patch = sp_data[:, :, y_start:y_end, x_start:x_end]

    if patch.shape[2] != patch_size or patch.shape[3] != patch_size:
        pad_y = patch_size - patch.shape[2]
        pad_x = patch_size - patch.shape[3]

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

        pad_mode = "constant" if patch.shape[2] == 0 or patch.shape[3] == 0 else "reflect"

        patch = np.pad(
            patch,
            ((0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1])),
            mode=pad_mode,
        )

    return patch
