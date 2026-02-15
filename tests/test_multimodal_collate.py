import torch

from src.data.pre_processing.flair_multimodal_dataset import multimodal_collate_fn


def test_multimodal_collate_pad_mask_marks_padding_as_true() -> None:
    aerial_0 = torch.zeros(5, 64, 64)
    aerial_1 = torch.zeros(5, 64, 64)

    # variable-length sequences
    sentinel_0 = torch.zeros(3, 10, 10, 10)  # T=3
    sentinel_1 = torch.zeros(5, 10, 10, 10)  # T=5 (max)

    mask_0 = torch.zeros(64, 64, dtype=torch.long)
    mask_1 = torch.zeros(64, 64, dtype=torch.long)

    positions_0 = torch.tensor([0, 1, 2], dtype=torch.long)
    positions_1 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    batch = [
        (aerial_0, sentinel_0, mask_0, "id0", positions_0),
        (aerial_1, sentinel_1, mask_1, "id1", positions_1),
    ]

    _, _, _, _, positions_batch, pad_mask = multimodal_collate_fn(batch)

    assert pad_mask.shape == (2, 5)
    # first sample: last 2 timesteps are padding (invalid)
    assert pad_mask[0].tolist() == [False, False, False, True, True]
    # second sample: no padding
    assert pad_mask[1].tolist() == [False, False, False, False, False]

    # padded positions are filled with -1
    assert positions_batch[0].tolist() == [0, 1, 2, -1, -1]
