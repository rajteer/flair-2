"""Simple model complexity utilities using calflops for accurate FLOPs/MACs counting."""

import torch
from calflops import calculate_flops


def compute_model_complexity(
    model: torch.nn.Module,
    input_size: tuple[int, ...],
) -> dict[str, int]:
    """Compute MACs, FLOPs, and params for a given model and input size.

    Uses calflops for accurate counting.
    """
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
