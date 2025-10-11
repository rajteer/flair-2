"""Utilities for reproducible experiments: global seeding, deterministic backends.

DataLoader worker seeding, and generator creation.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs and set deterministic flags.

    Args:
        seed: Base seed to use across libraries.
        deterministic: If True, prefer deterministic algorithms and disable
            non-deterministic behaviors where possible.

    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)

    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def seed_worker(_worker_id: int) -> None:
    """Seed dataloader workers.

    Use torch.initial_seed() to derive a distinct seed per worker. The seed is
    reproducible across runs when the base seed is fixed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.default_rng(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_generator(seed: int | None = None) -> torch.Generator:
    """Return a torch.Generator seeded for reproducible DataLoader shuffling."""
    gen = torch.Generator()
    if seed is None:
        seed = 42
    gen.manual_seed(int(seed))
    return gen
