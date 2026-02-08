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
    *,
    sentinel_input_size: tuple[int, ...] | None = None,
) -> dict[str, int]:
    """Compute MACs, FLOPs, and params for a given model and input size.

    Uses calflops for accurate counting.
    For temporal models (5D input), batch_positions must be provided.
    For multimodal late-fusion models, sentinel_input_size can be provided to
    supply the Sentinel-2 input shape.
    """
    # Check if this is a temporal model (5D input: B, T, C, H, W)
    is_temporal = len(input_size) == TEMPORAL_MODEL_NDIM
    is_multimodal = model.__class__.__name__ == "MultimodalLateFusion"

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    if is_multimodal:
        # MultimodalLateFusion signature:
        # forward(aerial, sentinel, batch_positions=None, pad_mask=None, cloud_coverage=None)
        # aerial: (B, C_aerial, H, W)
        # sentinel: (B, T, C_sentinel, H, W)
        if sentinel_input_size is None:
            sentinel_resolution = getattr(model, "sentinel_resolution", (10, 10))
            sentinel_in = getattr(model, "sentinel_in_channels", 10)
            batch_size = input_size[0] if len(input_size) > 0 else 1
            seq_len = 12
            sentinel_input_size = (
                batch_size,
                seq_len,
                sentinel_in,
                sentinel_resolution[0],
                sentinel_resolution[1],
            )

        if len(sentinel_input_size) != TEMPORAL_MODEL_NDIM:
            msg = "sentinel_input_size must be a 5D shape (B, T, C, H, W)"
            raise ValueError(msg)

        batch_size, seq_len, sentinel_in, height, width = sentinel_input_size

        aerial_input_size = input_size
        if len(aerial_input_size) == 4 and aerial_input_size[0] != batch_size:
            aerial_input_size = (batch_size, *aerial_input_size[1:])

        aerial = torch.randn(*aerial_input_size, device=model_device)
        sentinel = torch.randn(batch_size, seq_len, sentinel_in, height, width, device=model_device)

        if batch_positions is None:
            batch_positions = (
                torch.arange(seq_len, device=model_device).unsqueeze(0).expand(batch_size, -1)
            )
        else:
            batch_positions = batch_positions.to(model_device)

        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=model_device)

        try:
            flops, macs, params = calculate_flops(
                model=model,
                args=[aerial, sentinel],
                kwargs={"batch_positions": batch_positions, "pad_mask": pad_mask},
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
            batch_positions = torch.arange(seq_len, device=model_device).unsqueeze(0).expand(
                batch_size, -1
            )
        else:
            batch_positions = batch_positions.to(model_device)

        # For temporal models, we need to pass actual tensors (args) instead of input_shape
        # because calflops doesn't allow both input_shape and kwargs simultaneously
        try:
            dummy_input = torch.randn(*input_size, device=model_device)
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
        dummy_input = torch.randn(*input_size, device=model_device)
        flops, macs, params = calculate_flops(
            model=model,
            args=[dummy_input],
            output_as_string=False,
            output_precision=0,
            print_results=False,
        )

    return {
        "flops": int(flops),
        "macs": int(macs),
        "params": int(params),
    }
