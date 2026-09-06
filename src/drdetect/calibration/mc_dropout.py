"""MC-dropout uncertainty estimation for the grading model.

EfficientNet's classifier dropout is *functional*, not a persistent
`nn.Dropout` submodule -- timm's own `EfficientNet.forward_head` calls
`F.dropout(x, p=self.drop_rate, training=self.training)` directly. There is
no Dropout module to selectively re-enable via `.apply()`, so reactivating
it at inference means putting the WHOLE model back into `.train()` mode
(which sets the `self.training` flag `F.dropout` checks), then immediately
re-freezing BatchNorm with this project's own `freeze_batchnorm` so its
running statistics stay fixed -- leaving the dropout call as the only thing
still affected by the reactivated training flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from drdetect.grading.model import freeze_batchnorm

__all__ = ["mc_dropout_samples"]


def mc_dropout_samples(model: nn.Module, x: torch.Tensor, n_samples: int = 20) -> torch.Tensor:
    """Returns logits of shape (n_samples, batch, num_outputs), each from an
    independent stochastic forward pass with dropout active and BatchNorm
    still frozen on its pretrained running statistics."""
    model.train()
    freeze_batchnorm(model)
    with torch.no_grad():
        samples = torch.stack([model(x) for _ in range(n_samples)])
    model.eval()
    return samples
