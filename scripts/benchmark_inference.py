#!/usr/bin/env python
"""Standalone benchmark script for evaluating model inference time on the test set.

Usage:
    python scripts/benchmark_inference.py \
        --config configs/config_sentinel_tsvit.yaml \
        --checkpoint path/to/best_model.pt
"""

import argparse
import logging
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset_utils import pad_collate_sentinel
from src.data.pre_processing.flair_sentinel_dataset import FlairSentinelDataset
from src.models.model_builder import build_model
from src.utils.model_stats import compute_model_complexity
from src.utils.read_yaml import read_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_INDEX_INPUTS = 0
BATCH_INDEX_PAD_MASK = 2
BATCH_INDEX_POSITIONS = 4


def benchmark_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 10,
) -> dict[str, float]:
    """Benchmark model inference time over the entire dataset.

    Returns:
        Dictionary with timing statistics.
    """
    model.eval()
    inference_times: list[float] = []
    batch_sizes: list[int] = []

    # Warmup
    logger.info("Running %d warmup batches...", warmup_batches)
    warmup_count = 0
    for batch in data_loader:
        if warmup_count >= warmup_batches:
            break
        inputs = batch[BATCH_INDEX_INPUTS].to(device)
        pad_mask = batch[BATCH_INDEX_PAD_MASK].to(device)
        batch_positions = batch[BATCH_INDEX_POSITIONS].to(device)

        with torch.no_grad():
            _ = model(inputs, batch_positions=batch_positions, pad_mask=pad_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()
        warmup_count += 1

    logger.info("Warmup complete. Starting benchmark...")

    # Benchmark
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            inputs = batch[BATCH_INDEX_INPUTS].to(device)
            pad_mask = batch[BATCH_INDEX_PAD_MASK].to(device)
            batch_positions = batch[BATCH_INDEX_POSITIONS].to(device)

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = model(inputs, batch_positions=batch_positions, pad_mask=pad_mask)
                end_event.record()
                torch.cuda.synchronize()

                batch_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            else:
                start = time.perf_counter()
                _ = model(inputs, batch_positions=batch_positions, pad_mask=pad_mask)
                batch_time = time.perf_counter() - start

            inference_times.append(batch_time)
            batch_sizes.append(inputs.shape[0])

            if (batch_idx + 1) % 100 == 0:
                logger.info("Processed %d/%d batches", batch_idx + 1, len(data_loader))

    # Compute statistics
    total_time = sum(inference_times)
    total_images = sum(batch_sizes)
    avg_time_per_image = total_time / total_images if total_images > 0 else 0.0
    avg_time_per_batch = total_time / len(inference_times) if inference_times else 0.0

    return {
        "total_inference_time_s": total_time,
        "total_images": total_images,
        "total_batches": len(inference_times),
        "avg_time_per_image_s": avg_time_per_image,
        "avg_time_per_batch_s": avg_time_per_batch,
        "throughput_images_per_s": total_images / total_time if total_time > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model inference time")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup batches")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = read_yaml(Path(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build model
    model_type = config["model"]["model_type"].upper()
    model_config = config["model"].get("model_config", {})
    if model_type in ("TSVIT", "TSVIT_LOOKUP"):
        model_config = {**config["model"], **model_config}

    model = build_model(
        model_type=config["model"]["model_type"],
        encoder_name=config["model"].get("encoder_name", ""),
        encoder_weights=None,  # Don't load pretrained, we'll load checkpoint
        in_channels=config["model"]["in_channels"],
        n_classes=config["data"]["num_classes"],
        model_config=model_config,
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    logger.info("Loading checkpoint from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Compute model complexity
    sample_batch_size = args.batch_size or config["data"]["batch_size"]
    seq_len = 12  # Typical for monthly data
    in_channels = config["model"]["in_channels"]
    patch_size = config["data"]["sentinel_patch_size"]
    input_size = (sample_batch_size, seq_len, in_channels, patch_size, patch_size)

    logger.info("Computing model complexity for input size %s", input_size)
    complexity = compute_model_complexity(model, input_size)
    logger.info("FLOPs: %.2e", complexity["flops"])
    logger.info("MACs: %.2e", complexity["macs"])
    logger.info("Params: %.2e", complexity["params"])

    # Create test dataset
    date_encoding_mode = "month" if model_type == "TSVIT" else "days"
    pad_value = config["data"].get("pad_value", -100)

    test_dataset = FlairSentinelDataset(
        mask_dir=config["data"]["test"]["masks"],
        sentinel_dir=config["data"]["test"]["sentinel"],
        centroids_path=config["data"]["centroids_path"],
        num_classes=config["data"]["num_classes"],
        sentinel_patch_size=config["data"]["sentinel_patch_size"],
        context_size=config["data"].get("context_size"),
        use_monthly_average=config["data"].get("use_monthly_average", True),
        cloud_snow_cover_threshold=config["data"].get("cloud_snow_cover_threshold", 0.6),
        cloud_snow_prob_threshold=config["data"].get("cloud_snow_prob_threshold", 50),
        sentinel_scale_factor=config["data"].get("sentinel_scale_factor", 10000.0),
        sentinel_mean=config["data"].get("sentinel_mean"),
        sentinel_std=config["data"].get("sentinel_std"),
        date_encoding_mode=date_encoding_mode,
        downsample_masks=config["data"].get("downsample_masks", True),
    )

    batch_size = args.batch_size or config["data"]["batch_size"]
    collate_with_pad = partial(pad_collate_sentinel, pad_value=pad_value)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_with_pad,
    )

    logger.info("Test dataset: %d samples, %d batches", len(test_dataset), len(test_loader))

    # Benchmark
    stats = benchmark_model(model, test_loader, device, warmup_batches=args.warmup)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {config['model']['model_type']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)
    print(f"FLOPs:  {complexity['flops']:.2e}")
    print(f"MACs:   {complexity['macs']:.2e}")
    print(f"Params: {complexity['params']:.2e}")
    print("-" * 60)
    print(f"Total images:          {stats['total_images']}")
    print(f"Total batches:         {stats['total_batches']}")
    print(f"Total inference time:  {stats['total_inference_time_s']:.4f}s")
    print(f"Avg time per image:    {stats['avg_time_per_image_s']:.6f}s")
    print(f"Avg time per batch:    {stats['avg_time_per_batch_s']:.6f}s")
    print(f"Throughput:            {stats['throughput_images_per_s']:.2f} images/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
