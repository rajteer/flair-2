"""Compute class pixel counts and weights for weighted cross-entropy loss.

This script computes:
1. The number of pixels belonging to each class in the training dataset
2. Class weights for weighted cross-entropy loss using inverse frequency

Usage:
    python scripts/compute_class_weights.py --mask_dir data/flair_2_toy_labels_train

The script outputs:
- Pixel counts per class
- Class weights using inverse frequency
- A sample configuration for the loss function

"""

import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_ORIGINAL_CLASS = 12
OTHER_CLASS = 13


def compute_inverse_frequency_weights(
    class_counts: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute class weights using inverse frequency.

    Args:
        class_counts: Tensor with pixel count per class
        smooth: Smoothing factor to avoid division by zero (default: 1.0)

    Returns:
        Tensor with normalized class weights

    """
    total_pixels = class_counts.sum()
    num_classes = len(class_counts)

    weights = total_pixels / (num_classes * class_counts + smooth)
    return weights / weights.sum() * num_classes


def get_mask_files(mask_dir: Path) -> list[Path]:
    """Get all mask files from the directory recursively."""
    mask_files = sorted(mask_dir.rglob("MSK_*.tif"))
    return mask_files


def compute_class_counts(mask_dir: Path, num_classes: int = 13) -> torch.Tensor:
    """Compute class pixel counts from mask files.

    Args:
        mask_dir: Directory containing mask files
        num_classes: Number of classes (default: 13)

    Returns:
        Tensor with count of pixels for each class

    """
    mask_files = get_mask_files(mask_dir)
    logger.info(f"Found {len(mask_files)} mask files")

    class_counts = torch.zeros(num_classes, dtype=torch.int64)

    for mask_path in tqdm(mask_files, desc="Computing class counts"):
        mask = tifffile.imread(mask_path)
        mask = np.where(mask <= MAX_ORIGINAL_CLASS, mask, OTHER_CLASS)
        mask -= 1
        mask_tensor = torch.from_numpy(mask).long()
        mask_flat = mask_tensor.flatten()
        counts = torch.bincount(mask_flat, minlength=num_classes)
        class_counts += counts

    return class_counts


def main(mask_dir: str, num_classes: int) -> None:
    """Compute class weights from training data."""
    mask_path = Path(mask_dir)

    if not mask_path.exists():
        logger.error(f"Mask directory not found: {mask_dir}")
        return

    # Compute class counts
    logger.info("Computing class pixel counts...")
    class_counts = compute_class_counts(mask_path, num_classes)

    # Print results
    total_pixels = class_counts.sum().item()

    print("\n" + "=" * 80)
    print("CLASS PIXEL COUNTS")
    print("=" * 80)
    print(f"{'Class':<10} {'Count':<15} {'Percentage':<12} {'Frequency':<10}")
    print("-" * 80)

    for class_idx in range(num_classes):
        count = class_counts[class_idx].item()
        percentage = 100.0 * count / total_pixels
        frequency = count / total_pixels
        print(f"{class_idx:<10} {count:<15,} {percentage:<12.4f}% {frequency:<10.6f}")

    print("-" * 80)
    print(f"{'Total':<10} {total_pixels:<15,} {100.0:<12.1f}%")
    print("=" * 80)

    # Compute inverse frequency weights
    inv_freq_weights = compute_inverse_frequency_weights(class_counts)

    print("\n" + "=" * 80)
    print("CLASS WEIGHTS (Inverse Frequency)")
    print("=" * 80)
    print(f"{'Class':<10} {'Weight':<15}")
    print("-" * 80)

    for class_idx in range(num_classes):
        weight = inv_freq_weights[class_idx].item()
        print(f"{class_idx:<10} {weight:<15.4f}")

    print("=" * 80)

    print(f"\nclass_weights: {inv_freq_weights.tolist()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute class weights for weighted cross-entropy loss",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Path to directory containing mask files (e.g., data/flair_2_toy_labels_train)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=13,
        help="Number of classes (default: 13)",
    )
    args = parser.parse_args()

    main(args.mask_dir, args.num_classes)