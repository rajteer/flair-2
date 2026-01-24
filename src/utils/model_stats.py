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
    canonical_seq_len: int | None = None,
) -> dict[str, int]:
    """Compute MACs, FLOPs, and params for a given model and input size.

    Uses calflops for accurate counting.
    For temporal models (5D input), batch_positions must be provided.

    Args:
        model: The model to analyze.
        input_size: Shape tuple for the input tensor.
        batch_positions: Optional temporal positions for temporal models.
        canonical_seq_len: For temporal models, use this fixed sequence length
            instead of the actual input sequence length. This ensures consistent
            FLOPs reporting across runs with varying batch sequence lengths.
            Should be set to max_seq_len from model config.

    Returns:
        Dictionary with 'flops', 'macs', and 'params' counts.

    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    is_temporal = len(input_size) == TEMPORAL_MODEL_NDIM

    if is_temporal:
        batch_size = input_size[0]
        # Use canonical_seq_len if provided, otherwise use actual input sequence length
        seq_len = canonical_seq_len if canonical_seq_len is not None else input_size[1]

        # Override input_size with canonical sequence length for consistent FLOPs
        if canonical_seq_len is not None:
            input_size = (batch_size, seq_len, *input_size[2:])
            logger.info(
                "Using canonical sequence length %d for FLOPs calculation (actual: %d)",
                canonical_seq_len,
                input_size[1] if len(input_size) > 1 else 0,
            )

        if batch_positions is None:
            batch_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        batch_positions = batch_positions.to(device)

        # Create realistic pad_mask (all False = no padding, matches real inference)
        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        try:
            dummy_input = torch.randn(*input_size, device=device)
            logger.info(
                "Computing FLOPs for temporal model: input_size=%s, device=%s, "
                "batch_positions.shape=%s, pad_mask.shape=%s",
                input_size,
                device,
                batch_positions.shape,
                pad_mask.shape,
            )
            flops, macs, params = calculate_flops(
                model=model,
                args=[dummy_input],
                kwargs={"batch_positions": batch_positions, "pad_mask": pad_mask},
                output_as_string=False,
                output_precision=0,
                print_results=False,
            )
            logger.info(
                "FLOPs calculation succeeded: flops=%s, macs=%s, params=%s",
                flops,
                macs,
                params,
            )
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
