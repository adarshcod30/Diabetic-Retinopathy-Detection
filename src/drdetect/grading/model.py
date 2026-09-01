"""DR grading backbones.

Two decisions here come from measurements rather than convention.

Frozen BatchNorm
----------------
Benchmarking on this project's target machine showed batch size is capped at 4
by available RAM (docs/05_PROTOTYPE_SCOPE.md section 6). EfficientNet uses
BatchNorm, whose running statistics are estimated per micro-batch -- at batch 4
they are noisy, and gradient accumulation does NOT help, because accumulation
batches the optimiser step, not the normalisation. Freezing BN to use the
pretrained ImageNet statistics is the standard fix for small-batch fine-tuning
and is the default here.

Ordinal-ready head
------------------
Phase 1 is a plain cross-entropy baseline, deliberately. But DR grades are
ordinal, and Phase 3 replaces the loss. `num_outputs` is therefore separated
from `num_classes` so the same backbone serves a 5-way softmax, a 1-output
regression head, or a 4-output CORAL head without rewriting the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["build_model", "freeze_batchnorm", "count_parameters"]


def freeze_batchnorm(model: nn.Module) -> int:
    """Put every BatchNorm into eval mode and stop its affine parameters training.

    Returns the number of layers frozen, for logging -- a silent no-op here
    would be indistinguishable from a working call.
    """
    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)
            frozen += 1
    return frozen


class FundusGrader(nn.Module):
    def __init__(self, backbone: nn.Module, freeze_bn: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze_bn = freeze_bn
        if freeze_bn:
            self.n_frozen_bn = freeze_batchnorm(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def train(self, mode: bool = True):
        """Keep frozen BN in eval mode even when the module is set to train.

        Without this override, `model.train()` silently re-enables running-
        statistic updates and undoes the freeze -- a bug that produces no error
        and only shows up as unstable validation metrics.
        """
        super().train(mode)
        if self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


def build_model(
    name: str = "efficientnet_b0",
    *,
    num_outputs: int = 5,
    pretrained: bool = True,
    freeze_bn: bool = True,
    drop_rate: float = 0.2,
) -> FundusGrader:
    import timm

    backbone = timm.create_model(
        name, pretrained=pretrained, num_classes=num_outputs, drop_rate=drop_rate
    )
    return FundusGrader(backbone, freeze_bn=freeze_bn)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
