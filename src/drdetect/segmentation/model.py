"""Lesion segmentation backbone.

DeepLabV3+ (Chen et al. 2018), not plain U-Net: the roadmap names both as
options, and DeepLabV3+'s atrous spatial pyramid pooling gives it multi-scale
context that plain U-Net's decoder lacks -- useful here because IDRiD's
lesion classes span very different physical sizes (a single-pixel
microaneurysm vs. a hard-exudate cluster), even though this module only
trains hard exudates so far.

Uses segmentation_models_pytorch (already a declared but, until now, unused
dependency -- same situation grad-cam and reportlab were in before Phase 2),
not a hand-rolled decoder, for the same reason drdetect.grading.model uses
timm instead of a hand-rolled EfficientNet: a maintained, widely-checked
reference implementation over a bespoke one.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["build_segmentation_model"]


def build_segmentation_model(
    encoder_name: str = "resnet34",
    *,
    pretrained: bool = True,
    classes: int = 1,
) -> nn.Module:
    """Binary (or multi-class) lesion segmentation model.

    `classes=1`: one lesion type per model, matching how this project's IDRiD
    masks ship -- a separate binary mask file per lesion type, not one
    multi-class mask. Training one model per lesion type (rather than a
    shared multi-head model) also means a lesion with very little training
    signal (soft exudates, per the roadmap's own scope-cut list) cannot drag
    down a better-supported class's gradient.
    """
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=classes,
        activation=None,  # raw logits out; loss/inference apply sigmoid explicitly
    )
