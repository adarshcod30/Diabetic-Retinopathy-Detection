"""The fusion head itself: concat(CNN embedding, lesion features) -> ordinal head.

A small MLP, not another CNN -- the input is already a fixed-length vector
(the grading backbone's pooled embedding plus a handful of scalar lesion
features), not an image needing spatial processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FusionHead"]


class FusionHead(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        lesion_dim: int,
        hidden_dim: int = 128,
        num_outputs: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + lesion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
