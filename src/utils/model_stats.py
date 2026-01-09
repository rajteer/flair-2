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
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Check if this is a temporal model (5D input: B, T, C, H, W)
    is_temporal = len(input_size) == TEMPORAL_MODEL_NDIM

    if is_temporal:
        if batch_positions is None:
            # Generate dummy positions for temporal models
            batch_size, seq_len = input_size[0], input_size[1]
            batch_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        batch_positions = batch_positions.to(device)

        # For temporal models, we need to pass actual tensors (args) instead of input_shape
        # because calflops doesn't allow both input_shape and kwargs simultaneously
        try:
            dummy_input = torch.randn(*input_size, device=device)
            logger.info(
                "Computing FLOPs for temporal model: input_size=%s, device=%s, "
                "batch_positions.shape=%s, batch_positions.device=%s",
                input_size, device, batch_positions.shape, batch_positions.device
            )
            flops, macs, params = calculate_flops(
                model=model,
                args=[dummy_input],
                kwargs={"batch_positions": batch_positions},
                output_as_string=False,
                output_precision=0,
                print_results=False,
            )
            logger.info("FLOPs calculation succeeded: flops=%s, macs=%s, params=%s", flops, macs, params)
        except Exception as e:
            logger.warning("Failed to compute FLOPs for temporal model: %s", e, exc_info=True)
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
