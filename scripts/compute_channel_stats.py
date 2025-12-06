"""Compute NIR and elevation statistics from FLAIR-2 aerial imagery.

This script scans the configured aerial training images directory and computes
per-channel statistics needed for normalization:

- mean and std for the NIR channel
- mean and std for the elevation channel
- min and max for the elevation channel

It assumes aerial images are stored as TIFFs with channels ordered so that
NIR and elevation correspond to indices provided by the user.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_stats(
    images_dir: Path,
    nir_channel: int,
    elev_channel: int,
) -> None:
    """Compute and print statistics for NIR and elevation channels.

    Args:
        images_dir: Directory containing aerial image files (IMG_*.tif).
        nir_channel: Index of the NIR channel in the TIFF (0-based).
        elev_channel: Index of the elevation channel in the TIFF (0-based).
    """
    image_paths = sorted(images_dir.rglob("IMG_*.tif"))
    if not image_paths:
        msg = f"No IMG_*.tif files found in {images_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d images in %s", len(image_paths), images_dir)

    # First pass: determine raw elevation range
    elev_min = None
    elev_max = None

    for path in tqdm(image_paths, desc="Scanning for elevation range"):
        img = tifffile.imread(path)
        if img.ndim == 2:
            logger.warning("Skipping %s: expected multi-channel image", path)
            continue

        if img.shape[0] <= 8 and img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        if img.ndim != 3:
            logger.warning("Skipping %s: unsupported shape %s", path, img.shape)
            continue

        if elev_channel >= img.shape[2]:
            logger.warning("Skipping %s: not enough channels (shape %s)", path, img.shape)
            continue

        elev_raw = img[..., elev_channel].astype(np.float64)
        cur_min = float(elev_raw.min())
        cur_max = float(elev_raw.max())
        elev_min = cur_min if elev_min is None else min(elev_min, cur_min)
        elev_max = cur_max if elev_max is None else max(elev_max, cur_max)

    if elev_min is None or elev_max is None:
        msg = "Could not determine elevation range; no valid elevation pixels found."
        raise RuntimeError(msg)

    # Second pass: compute stats in the same scaled space as training
    nir_sum = 0.0
    nir_sq_sum = 0.0
    nir_count = 0

    elev_sum = 0.0
    elev_sq_sum = 0.0
    elev_count = 0
    for path in tqdm(image_paths, desc="Computing channel statistics"):
        img = tifffile.imread(path)  # (H, W, C) or (C, H, W)
        if img.ndim == 2:
            logger.warning("Skipping %s: expected multi-channel image", path)
            continue

        if img.shape[0] <= 8 and img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))

        if img.ndim != 3:
            logger.warning("Skipping %s: unsupported shape %s", path, img.shape)
            continue

        if nir_channel >= img.shape[2] or elev_channel >= img.shape[2]:
            logger.warning("Skipping %s: not enough channels (shape %s)", path, img.shape)
            continue

        nir_raw = img[..., nir_channel].astype(np.float64)
        elev_raw = img[..., elev_channel].astype(np.float64)

        nir = nir_raw / 255.0

        elev = (elev_raw - elev_min) / (elev_max - elev_min)

        nir_sum += nir.sum()
        nir_sq_sum += (nir ** 2).sum()
        nir_count += nir.size

        elev_sum += elev.sum()
        elev_sq_sum += (elev ** 2).sum()
        elev_count += elev.size

    if nir_count == 0 or elev_count == 0:
        msg = "No valid pixels found to compute statistics. Check channel indices."
        raise RuntimeError(msg)

    nir_mean = nir_sum / nir_count
    nir_var = nir_sq_sum / nir_count - nir_mean**2
    nir_std = float(np.sqrt(max(nir_var, 0.0)))

    elev_mean = elev_sum / elev_count
    elev_var = elev_sq_sum / elev_count - elev_mean**2
    elev_std = float(np.sqrt(max(elev_var, 0.0)))

    print("Computed statistics in scaled space:")
    print("  NIR assumed scaled as nir/255.0")
    print("  Elevation scaled using global raw range to [0, 1]")
    print(f"  nir_mean_scaled:  {nir_mean:.6f}")
    print(f"  nir_std_scaled:   {nir_std:.6f}")
    print(f"  elev_mean_scaled: {elev_mean:.6f}")
    print(f"  elev_std_scaled:  {elev_std:.6f}")
    print(f"  elev_min_raw:     {elev_min:.6f}")
    print(f"  elev_max_raw:     {elev_max:.6f}")

    print("\nYAML snippet for config.normalization (example for RGB+NIR+elevation):")
    print("normalization:")
    print("  mean: [0.485, 0.456, 0.406, " f"{nir_mean:.6f}, {elev_mean:.6f}]")
    print("  std:  [0.229, 0.224, 0.225, " f"{nir_std:.6f},  {elev_std:.6f}]")
    print("  scale_to_unit: [true, true, true, true, false]")
    print("  elevation_range: [" f"{elev_min:.6f}, {elev_max:.6f}]")
    print("  elevation_channel_index: 4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NIR/elevation stats for normalization.")
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to aerial training images directory (e.g. data/flair_2_toy_aerial_train)",
    )
    parser.add_argument(
        "--nir-channel",
        type=int,
        default=3,
        help="Index of NIR channel in the TIFF (0-based). Default: 3",
    )
    parser.add_argument(
        "--elev-channel",
        type=int,
        default=4,
        help="Index of elevation channel in the TIFF (0-based). Default: 4",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    images_dir = Path(args.images_dir).expanduser().resolve()
    compute_stats(images_dir=images_dir, nir_channel=args.nir_channel, elev_channel=args.elev_channel)


if __name__ == "__main__":
    main()
