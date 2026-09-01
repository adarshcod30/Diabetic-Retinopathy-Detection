"""Determinism helpers.

Reproducibility is a stated requirement of this project (a clean clone must
reproduce the headline table), so seeding is not optional plumbing.
"""

from __future__ import annotations

import os
import random

__all__ = ["seed_everything", "worker_init_fn"]


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and torch. Returns the seed for logging.

    `deterministic=True` also disables cuDNN autotuning. On MPS most kernels are
    deterministic already, but the flag is honoured where it applies.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible seed.

    Without this, forked workers share the parent's NumPy state and every worker
    generates the *same* augmentation stream -- a subtle bug that silently
    reduces effective augmentation diversity.
    """
    import numpy as np
    import torch

    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
