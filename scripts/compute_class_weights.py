"""Compute class pixel counts and weights for weighted cross-entropy loss.

This script computes:
1. The number of pixels belonging to each class in the training dataset
2. Class weights for weighted cross-entropy loss using inverse frequency

Usage:
    python -m scripts.compute_class_weights --mask_dir data/flair_2_toy_labels_train --image_dir data/flair_2_toy_aerial_train

The script outputs:
- Pixel counts per class
- Class weights using inverse frequency
- A sample configuration for the loss function

"""

import argparse
import logging
from pathlib import Path

import torch

from src.data.pre_processing.flair_dataset import FlairDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def main(mask_dir: str, image_dir: str, num_classes: int) -> None:
    """Compute class weights from training data."""
    mask_path = Path(mask_dir)

    if not mask_path.exists():
        logger.error(f"Mask directory not found: {mask_dir}")
        return
    image_path = Path(image_dir)
    if not image_path.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return

    logger.info("Loading dataset...")
    dataset = FlairDataset(
        image_dir=str(image_path),
        mask_dir=str(mask_path),
        num_classes=num_classes,
    )

    logger.info(f"Found {len(dataset)} samples")

    logger.info("Computing class pixel counts...")
    class_counts = dataset.get_class_counts()

    total_pixels = class_counts.sum().item()

    logger.info("\n" + "=" * 80)
    logger.info("CLASS PIXEL COUNTS")
    logger.info("=" * 80)
    logger.info(f"{'Class':<10} {'Count':<15} {'Percentage':<12} {'Frequency':<10}")
    logger.info("-" * 80)

    for class_idx in range(num_classes):
        count = class_counts[class_idx].item()
        percentage = 100.0 * count / total_pixels
        frequency = count / total_pixels
        logger.info(f"{class_idx:<10} {count:<15,} {percentage:<12.4f}% {frequency:<10.6f}")

    logger.info("-" * 80)
    logger.info(f"{'Total':<10} {total_pixels:<15,} {100.0:<12.1f}%")
    logger.info("=" * 80)
    inv_freq_weights = compute_inverse_frequency_weights(class_counts)

    logger.info("\n" + "=" * 80)
    logger.info("CLASS WEIGHTS (Inverse Frequency)")
    logger.info("=" * 80)
    logger.info(f"{'Class':<10} {'Weight':<15}")
    logger.info("-" * 80)

    for class_idx in range(num_classes):
        weight = inv_freq_weights[class_idx].item()
        logger.info(f"{class_idx:<10} {weight:<15.4f}")

    logger.info("=" * 80)
    logger.info(f"\nclass_counts: {class_counts.tolist()}")
    logger.info(f"\nclass_weights (Inverse Frequency): {inv_freq_weights.tolist()}\n")


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
        "--image_dir",
        type=str,
        required=True,
        help="Path to directory containing image files (auto-inferred if not provided)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=13,
        help="Number of classes (default: 13)",
    )
    args = parser.parse_args()

    main(args.mask_dir, args.image_dir, args.num_classes)
