"""Test multimodal pipeline flow with toy dataset."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Import the pipeline components
from src.data.pre_processing.flair_multimodal_dataset import (
    multimodal_collate_fn,
    MM_BATCH_AERIAL,
    MM_BATCH_SENTINEL,
    MM_BATCH_MASK,
    MM_BATCH_SAMPLE_IDS,
    MM_BATCH_POSITIONS,
    MM_BATCH_PAD_MASK,
)
from src.data.pre_processing.multimodal_augmentation import MultimodalAugmentation
from src.models.validation import evaluate


class ToyMultimodalDataset(Dataset):
    """Toy dataset that mimics FlairMultimodalDataset output format."""

    def __init__(self, num_samples: int = 10, num_classes: int = 14):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Aerial: (5, 512, 512) - 5 channels (RGB + NIR + elevation)
        aerial = torch.randn(5, 512, 512)

        # Sentinel: (T, 10, 32, 32) - variable timesteps, 10 bands, 32x32 patches
        num_timesteps = torch.randint(3, 12, (1,)).item()
        sentinel = torch.randn(num_timesteps, 10, 32, 32)

        # Mask: (512, 512) - class labels
        mask = torch.randint(0, self.num_classes, (512, 512))

        # Sample ID
        sample_id = f"sample_{idx}"

        # Month positions: (T,) - 1-12 for each timestep
        positions = torch.randint(1, 13, (num_timesteps,))

        return aerial, sentinel, mask, sample_id, positions


class ToyMultimodalModel(nn.Module):
    """Minimal multimodal model for testing."""

    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.num_classes = num_classes
        # Simple conv to get right output shape
        self.aerial_conv = nn.Conv2d(5, 32, 3, padding=1)
        self.output_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, aerial, sentinel, batch_positions=None, pad_mask=None):
        # Ignore sentinel for simplicity, just use aerial
        x = self.aerial_conv(aerial)
        x = torch.relu(x)
        return self.output_conv(x)


def test_collate_fn():
    """Test that collate function produces correct batch structure."""
    print("=" * 60)
    print("TEST 1: Collate Function")
    print("=" * 60)

    dataset = ToyMultimodalDataset(num_samples=4)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=multimodal_collate_fn,
        shuffle=False,
    )

    batch = next(iter(loader))

    print(f"Batch has {len(batch)} elements (expected 6)")
    print(f"  [0] Aerial shape: {batch[MM_BATCH_AERIAL].shape}")
    print(f"  [1] Sentinel shape: {batch[MM_BATCH_SENTINEL].shape}")
    print(f"  [2] Mask shape: {batch[MM_BATCH_MASK].shape}")
    print(f"  [3] Sample IDs: {batch[MM_BATCH_SAMPLE_IDS]}")
    print(f"  [4] Positions shape: {batch[MM_BATCH_POSITIONS].shape}")
    print(f"  [5] Pad mask shape: {batch[MM_BATCH_PAD_MASK].shape}")

    assert len(batch) == 6, "Batch should have 6 elements"
    assert batch[MM_BATCH_AERIAL].shape[0] == 2, "Batch size should be 2"
    print("✓ Collate function works correctly!\n")
    return batch


def test_augmentation(batch):
    """Test multimodal augmentation on a batch."""
    print("=" * 60)
    print("TEST 2: Multimodal Augmentation")
    print("=" * 60)

    config = {
        "data_augmentation": {
            "apply_augmentations": True,
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rotation_prob": 0.5,
        },
        "sentinel_augmentation": {
            "enabled": True,
            "mean": [0.0] * 10,
            "std": [1.0] * 10,
            "random_temporal_drop": True,
            "temporal_shift": True,
            "max_shift": 2,
        },
    }

    augmenter = MultimodalAugmentation(config)

    aerial = batch[MM_BATCH_AERIAL]
    sentinel = batch[MM_BATCH_SENTINEL]
    mask = batch[MM_BATCH_MASK]
    positions = batch[MM_BATCH_POSITIONS]
    pad_mask = batch[MM_BATCH_PAD_MASK]

    print(f"Before augmentation:")
    print(f"  Aerial: {aerial.shape}, Sentinel: {sentinel.shape}")

    aerial_aug, sentinel_aug, mask_aug, positions_aug, pad_mask_aug = augmenter(
        aerial, sentinel, mask, positions, pad_mask, training=True
    )

    print(f"After augmentation:")
    print(f"  Aerial: {aerial_aug.shape}, Sentinel: {sentinel_aug.shape}")

    assert aerial_aug.shape == aerial.shape, "Aerial shape should be preserved"
    assert mask_aug.shape == mask.shape, "Mask shape should be preserved"
    print("✓ Augmentation works correctly!\n")


def test_model_forward(batch):
    """Test model forward pass with batch."""
    print("=" * 60)
    print("TEST 3: Model Forward Pass")
    print("=" * 60)

    model = ToyMultimodalModel(num_classes=14)
    device = torch.device("cpu")
    model.to(device)

    aerial = batch[MM_BATCH_AERIAL].to(device)
    sentinel = batch[MM_BATCH_SENTINEL].to(device)
    positions = batch[MM_BATCH_POSITIONS].to(device)
    pad_mask = batch[MM_BATCH_PAD_MASK].to(device)

    output = model(aerial, sentinel, batch_positions=positions, pad_mask=pad_mask)

    print(f"Input aerial shape: {aerial.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 14, 512, 512)")

    assert output.shape == (2, 14, 512, 512), "Output shape mismatch"
    print("✓ Model forward pass works correctly!\n")


def test_evaluate():
    """Test evaluation function with multimodal model."""
    print("=" * 60)
    print("TEST 4: Evaluation Function")
    print("=" * 60)

    dataset = ToyMultimodalDataset(num_samples=4)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=multimodal_collate_fn,
        shuffle=False,
    )

    model = ToyMultimodalModel(num_classes=14)
    device = torch.device("cpu")

    print("Running evaluate() with is_multimodal=True...")

    metrics = evaluate(
        model=model,
        device=device,
        data_loader=loader,
        num_classes=14,
        other_class_index=13,
        log_eval_metrics=False,  # Don't log to MLflow
        log_confusion_matrix=False,
        is_multimodal=True,
        warmup_runs=0,
    )

    print(f"Metrics returned: {list(metrics.keys())}")
    print(f"  mIoU: {metrics.get('miou', 'N/A'):.4f}")
    print(f"  Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}")
    print(f"  Overall Accuracy: {metrics.get('overall_accuracy', 'N/A'):.4f}")

    assert "miou" in metrics, "Should have mIoU"
    assert "macro_f1" in metrics, "Should have macro F1"
    print("✓ Evaluation function works correctly!\n")


def main():
    print("\n" + "=" * 60)
    print("MULTIMODAL PIPELINE FLOW TEST")
    print("=" * 60 + "\n")

    # Test 1: Collate function
    batch = test_collate_fn()

    # Test 2: Augmentation
    test_augmentation(batch)

    # Test 3: Model forward
    test_model_forward(batch)

    # Test 4: Evaluation
    test_evaluate()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
