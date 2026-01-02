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
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_sentinel_statistics(
    sentinel_dir: Path,
    sample_ratio: float = 1.0,
    scale_factor: float = 10000.0,
) -> dict[str, list[float]]:
    """Compute channel-wise mean and std for Sentinel-2 data.

    Uses Welford's online algorithm for numerically stable computation.

    Args:
        sentinel_dir: Directory containing Sentinel-2 superpatch .npy files
        sample_ratio: Fraction of patches to sample (0-1). Use <1 for faster computation.
        scale_factor: Factor to divide raw values by before computing stats.
            Default 10000.0 normalizes reflectance to [0, 1] range.

    Returns:
        Dict with 'mean' and 'std' keys, each containing list of per-channel values.

    """
    # Find all sentinel data files
    data_files = list(sentinel_dir.rglob("*_data.npy"))
    logger.info("Found %d Sentinel-2 data files", len(data_files))

    if len(data_files) == 0:
        msg = f"No *_data.npy files found in {sentinel_dir}"
        raise ValueError(msg)

    # Sample files if requested
    if sample_ratio < 1.0:
        n_samples = max(1, int(len(data_files) * sample_ratio))
        rng = np.random.default_rng(42)
        data_files = list(rng.choice(data_files, size=n_samples, replace=False))
        logger.info("Sampled %d files for statistics computation", len(data_files))

    n_channels = None
    count = 0
    mean = None
    m2 = None

    for data_file in tqdm(data_files, desc="Computing statistics"):
        try:
            data = np.load(data_file, mmap_mode="r")

            if n_channels is None:
                n_channels = data.shape[1]
                mean = np.zeros(n_channels, dtype=np.float64)
                m2 = np.zeros(n_channels, dtype=np.float64)
                logger.info("Detected %d channels", n_channels)

            data = data.astype(np.float64) / scale_factor

            for c in range(n_channels):
                channel_data = data[:, c, :, :].flatten()

                for x in channel_data:
                    count += 1
                    delta = x - mean[c]
                    mean[c] += delta / count
                    delta2 = x - mean[c]
                    m2[c] += delta * delta2

        except Exception as e:
            logger.warning("Failed to process %s: %s", data_file, e)
            continue

    if count == 0:
        msg = "No valid data samples found"
        raise ValueError(msg)

    variance = m2 / count
    std = np.sqrt(variance)

    logger.info("Computed statistics from %d total pixels", count)
    logger.info("Per-channel means: %s", mean.tolist())
    logger.info("Per-channel stds: %s", std.tolist())

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count,
        "n_files": len(data_files),
        "scale_factor": scale_factor,
    }


def compute_sentinel_statistics_batch(
    sentinel_dir: Path,
    sample_ratio: float = 1.0,
    scale_factor: float = 10000.0,
) -> dict[str, list[float]]:
    """Compute channel-wise mean and std using batch processing (faster but uses more memory).

    This version processes entire superpatch files at once instead of pixel-by-pixel.

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

    # Collect sums and counts per channel
    n_channels = None
    channel_sums = None
    channel_sq_sums = None
    total_count = 0

    for data_file in tqdm(data_files, desc="Computing statistics (batch mode)"):
        try:
            data = np.load(data_file).astype(np.float64) / scale_factor

            if n_channels is None:
                n_channels = data.shape[1]
                channel_sums = np.zeros(n_channels, dtype=np.float64)
                channel_sq_sums = np.zeros(n_channels, dtype=np.float64)
                logger.info("Detected %d channels", n_channels)

            # Sum over all dimensions except channel
            for c in range(n_channels):
                channel_data = data[:, c, :, :]
                channel_sums[c] += channel_data.sum()
                channel_sq_sums[c] += (channel_data**2).sum()
                total_count += channel_data.size // n_channels

        except Exception as e:
            logger.warning("Failed to process %s: %s", data_file, e)
            continue

    if total_count == 0:
        msg = "No valid data samples found"
        raise ValueError(msg)

    count_per_channel = total_count

    mean = channel_sums / count_per_channel
    variance = (channel_sq_sums / count_per_channel) - (mean**2)
    std = np.sqrt(np.maximum(variance, 0))

    logger.info("Computed statistics from %d pixels per channel", count_per_channel)
    logger.info("Per-channel means: %s", mean.tolist())
    logger.info("Per-channel stds: %s", std.tolist())

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count_per_channel,
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
        "--batch",
        action="store_true",
        help="Use batch processing (faster but uses more memory)",
    )

    args = parser.parse_args()

    if args.batch:
        stats = compute_sentinel_statistics_batch(
            args.sentinel_dir,
            sample_ratio=args.sample_ratio,
            scale_factor=args.scale_factor,
        )
    else:
        stats = compute_sentinel_statistics(
            args.sentinel_dir,
            sample_ratio=args.sample_ratio,
            scale_factor=args.scale_factor,
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
