"""Simple model complexity utilities using calflops for accurate FLOPs/MACs counting."""

import logging

import torch
from calflops import calculate_flops

logger = logging.getLogger(__name__)

TEMPORAL_MODEL_NDIM = 5


def compute_model_complexity(
    model: torch.nn.Module,
    input_size: tuple[int, ...],
    batch_positions: torch.Tensor | None = None,
) -> dict[str, int]:
    """Compute MACs, FLOPs, and params for a given model and input size.

    Uses calflops for accurate counting.
    For temporal models (5D input), batch_positions must be provided.
    """
    # Check if this is a temporal model (5D input: B, T, C, H, W)
    is_temporal = len(input_size) == TEMPORAL_MODEL_NDIM
    is_multimodal = model.__class__.__name__ == "MultimodalLateFusion"

    if is_multimodal:
        # MultimodalLateFusion signature:
        # forward(aerial, sentinel, sp_centroids, sp_encs, sp_month_indices, cloud_masks=None)
        # aerial: (B, C_aerial, H, W)
        # sentinel: (B, T, C_sentinel, H, W) or (B, T, C_sentinel, h, w) depending on setup
        # sp_centroids: (B, T, 2)
        # sp_encs: (B, T, 32)
        # sp_month_indices: (B, T)

        # Infer sizes from model attributes if available, or use defaults
        aerial_in = getattr(model, "aerial_in_channels", 5)
        sentinel_in = getattr(model, "sentinel_in_channels", 10)
        # Assuming batch size 1 for complexity check
        B = 1
        H, W = input_size[-2:]  # Use aerial H, W from input_size
        T = 12  # Default temporal sequence length

        # Create dummy inputs
        aerial = torch.randn(B, aerial_in, H, W)
        sentinel = torch.randn(B, T, sentinel_in, 10, 10)  # Sentinel patches are usually 10x10
        sp_centroids = torch.randn(B, T, 2)
        sp_encs = torch.randn(B, T, 32)
        sp_month_indices = torch.randint(0, 12, (B, T))

        try:
            flops, macs, params = calculate_flops(
                model=model,
                args=[aerial, sentinel, sp_centroids, sp_encs, sp_month_indices],
                output_as_string=False,
                output_precision=0,
                print_results=False,
            )
        except Exception as e:
            logger.warning("Failed to compute FLOPs for multimodal model: %s", e)
            params = sum(p.numel() for p in model.parameters())
            return {"flops": 0, "macs": 0, "params": params}

    elif is_temporal:
        if batch_positions is None:
            # Generate dummy positions for temporal models
            batch_size, seq_len = input_size[0], input_size[1]
            batch_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # For temporal models, we need to pass actual tensors (args) instead of input_shape
        # because calflops doesn't allow both input_shape and kwargs simultaneously
        try:
            dummy_input = torch.randn(*input_size)
            flops, macs, params = calculate_flops(
                model=model,
                args=[dummy_input],
                kwargs={"batch_positions": batch_positions},
                output_as_string=False,
                output_precision=0,
                print_results=False,
            )
        except Exception as e:
            logger.warning("Failed to compute FLOPs for temporal model: %s", e)
            params = sum(p.numel() for p in model.parameters())
            return {"flops": 0, "macs": 0, "params": params}
    else:
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=input_size,
            output_as_string=False,
            output_precision=0,
            print_results=False,
        )

    return {
        "flops": int(flops),
        "macs": int(macs),
        "params": int(params),
    }
