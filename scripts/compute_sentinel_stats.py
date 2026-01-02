"""Compute channel-wise mean and std for Sentinel-2 data in FLAIR dataset.

This script iterates over all Sentinel-2 superpatch files in a directory,
computes per-channel means and standard deviations, and saves them to a JSON file.

Usage:
    python scripts/compute_sentinel_stats.py \
        --sentinel_dir /path/to/sentinel/data \
        --output stats/sentinel_normalization.json
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_single_file(args: tuple) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Process a single file and return channel sums, squared sums, and pixel count.

    Args:
        args: Tuple of (file_path, scale_factor)

    Returns:
        Tuple of (channel_sums, channel_sq_sums, pixel_count) or None if failed.

    """
    file_path, scale_factor = args
    try:
        data = np.load(file_path).astype(np.float64) / scale_factor
        n_channels = data.shape[1]

        # Compute sums for each channel using efficient numpy operations
        # data shape: (T, C, H, W)
        # Sum over T, H, W dimensions for each channel
        channel_sums = data.sum(axis=(0, 2, 3))  # Shape: (C,)
        channel_sq_sums = (data**2).sum(axis=(0, 2, 3))  # Shape: (C,)
        pixel_count = data.shape[0] * data.shape[2] * data.shape[3]  # T * H * W

        return channel_sums, channel_sq_sums, pixel_count
    except Exception:
        return None


def compute_sentinel_statistics_parallel(
    sentinel_dir: Path,
    sample_ratio: float = 1.0,
    scale_factor: float = 10000.0,
    num_workers: int = 8,
) -> dict[str, list[float]]:
    """Compute channel-wise mean and std using parallel processing.

    Args:
        sentinel_dir: Directory containing Sentinel-2 superpatch .npy files
        sample_ratio: Fraction of patches to sample (0-1).
        scale_factor: Factor to divide raw values by before computing stats.
        num_workers: Number of parallel workers.

    Returns:
        Dict with 'mean' and 'std' keys, each containing list of per-channel values.

    """
    data_files = list(sentinel_dir.rglob("*_data.npy"))
    logger.info("Found %d Sentinel-2 data files", len(data_files))

    if len(data_files) == 0:
        msg = f"No *_data.npy files found in {sentinel_dir}"
        raise ValueError(msg)

    if sample_ratio < 1.0:
        n_samples = max(1, int(len(data_files) * sample_ratio))
        rng = np.random.default_rng(42)
        data_files = list(rng.choice(data_files, size=n_samples, replace=False))
        logger.info("Sampled %d files for statistics computation", len(data_files))

    args_list = [(f, scale_factor) for f in data_files]

    n_channels = None
    channel_sums = None
    channel_sq_sums = None
    total_pixels = 0

    logger.info("Processing files with %d workers...", num_workers)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args[0] for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is None:
                continue

            sums, sq_sums, count = result

            if n_channels is None:
                n_channels = len(sums)
                channel_sums = np.zeros(n_channels, dtype=np.float64)
                channel_sq_sums = np.zeros(n_channels, dtype=np.float64)

            channel_sums += sums
            channel_sq_sums += sq_sums
            total_pixels += count

    if total_pixels == 0:
        msg = "No valid data samples found"
        raise ValueError(msg)

    # Compute final statistics
    mean = channel_sums / total_pixels
    variance = (channel_sq_sums / total_pixels) - (mean**2)
    std = np.sqrt(np.maximum(variance, 0))

    logger.info("Computed statistics from %d total pixels per channel", total_pixels)
    logger.info("Per-channel means: %s", mean.tolist())
    logger.info("Per-channel stds: %s", std.tolist())

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": total_pixels,
        "n_files": len(data_files),
        "scale_factor": scale_factor,
    }


def compute_sentinel_statistics_batch(
    sentinel_dir: Path,
    sample_ratio: float = 1.0,
    scale_factor: float = 10000.0,
) -> dict[str, list[float]]:
    """Compute channel-wise mean and std using batch processing (single-threaded).

    Args:
        sentinel_dir: Directory containing Sentinel-2 superpatch .npy files
        sample_ratio: Fraction of patches to sample (0-1).
        scale_factor: Factor to divide raw values by before computing stats.

    Returns:
        Dict with 'mean' and 'std' keys, each containing list of per-channel values.

    """
    data_files = list(sentinel_dir.rglob("*_data.npy"))
    logger.info("Found %d Sentinel-2 data files", len(data_files))

    if len(data_files) == 0:
        msg = f"No *_data.npy files found in {sentinel_dir}"
        raise ValueError(msg)

    if sample_ratio < 1.0:
        n_samples = max(1, int(len(data_files) * sample_ratio))
        rng = np.random.default_rng(42)
        data_files = list(rng.choice(data_files, size=n_samples, replace=False))
        logger.info("Sampled %d files for statistics computation", len(data_files))

    n_channels = None
    channel_sums = None
    channel_sq_sums = None
    total_pixels = 0

    for data_file in tqdm(data_files, desc="Computing statistics"):
        try:
            data = np.load(data_file).astype(np.float64) / scale_factor

            if n_channels is None:
                n_channels = data.shape[1]
                channel_sums = np.zeros(n_channels, dtype=np.float64)
                channel_sq_sums = np.zeros(n_channels, dtype=np.float64)

            # Efficient vectorized sum over T, H, W
            channel_sums += data.sum(axis=(0, 2, 3))
            channel_sq_sums += (data**2).sum(axis=(0, 2, 3))
            total_pixels += data.shape[0] * data.shape[2] * data.shape[3]

        except Exception as e:
            logger.warning("Failed to process %s: %s", data_file, e)
            continue

    if total_pixels == 0:
        msg = "No valid data samples found"
        raise ValueError(msg)

    mean = channel_sums / total_pixels
    variance = (channel_sq_sums / total_pixels) - (mean**2)
    std = np.sqrt(np.maximum(variance, 0))

    logger.info("Computed statistics from %d pixels per channel", total_pixels)
    logger.info("Per-channel means: %s", mean.tolist())
    logger.info("Per-channel stds: %s", std.tolist())

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": total_pixels,
        "n_files": len(data_files),
        "scale_factor": scale_factor,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute Sentinel-2 channel-wise normalization statistics"
    )
    parser.add_argument(
        "--sentinel_dir",
        type=Path,
        required=True,
        help="Directory containing Sentinel-2 superpatch data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sentinel_normalization_stats.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of files to sample (0-1). Use <1 for faster computation.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=10000.0,
        help="Factor to divide raw reflectance values by (default: 10000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (use single-threaded batch mode)",
    )

    args = parser.parse_args()

    if args.no_parallel:
        stats = compute_sentinel_statistics_batch(
            args.sentinel_dir,
            sample_ratio=args.sample_ratio,
            scale_factor=args.scale_factor,
        )
    else:
        stats = compute_sentinel_statistics_parallel(
            args.sentinel_dir,
            sample_ratio=args.sample_ratio,
            scale_factor=args.scale_factor,
            num_workers=args.workers,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Statistics saved to %s", args.output)

    print("\n" + "=" * 60)
    print("SENTINEL-2 NORMALIZATION STATISTICS")
    print("=" * 60)
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")
    print(f"Scale factor: {stats['scale_factor']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
