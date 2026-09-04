"""Grad-CAM evidence overlay for the DR grader.

Thin wrapper around `pytorch_grad_cam` rather than a hand-rolled hook, so the
CAM implementation matches a maintained, widely-checked reference rather than
a bespoke one. Phase 6 (docs/04_ROADMAP.md) is where CAM variants get compared
side by side and run through Adebayo's sanity checks -- this module is only
the single Grad-CAM used for Phase 2's end-to-end slice, not that comparison.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

__all__ = ["default_target_layer", "generate_cam", "overlay_cam"]


def default_target_layer(model: nn.Module) -> nn.Module:
    """The last activation before global pooling, for a timm EfficientNet.

    `bn2`, not `conv_head`. timm's EfficientNet fuses BatchNorm+SiLU into a
    single `BatchNormAct2d` module at `bn2`, so `conv_head`'s own output is
    the bare pre-activation conv -- unbounded, and roughly half-negative in
    practice. Grad-CAM/Grad-CAM++/Eigen-CAM mostly tolerate that (their
    weights can themselves be negative and partially cancel it out), but
    Score-CAM's weights are a softmax over class scores -- always positive --
    so weighting an already-negative activation map and then ReLU-ing the
    result (`base_cam.py`'s `compute_cam_per_layer`) can zero it out
    entirely. Measured directly: `conv_head` produced an all-zero Score-CAM
    heatmap on a real image; `bn2`, whose output has already passed through
    SiLU, did not. `bn2` is the correct choice for every method, not a
    Score-CAM-specific workaround -- it's simply the last point where the
    activation reflects what the network actually forward-propagates.
    """
    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "bn2"):
        raise AttributeError(
            f"{type(backbone).__name__} has no `bn2`; pass target_layer explicitly."
        )
    return backbone.bn2


def generate_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Grad-CAM heatmap for one image, as a single HxW array in [0, 1].

    Args:
        model: the grader, in eval mode. Frozen BatchNorm does not interfere
            -- Grad-CAM only needs gradients w.r.t. activations, never an
            optimiser step.
        input_tensor: 1xCxHxW, already normalised exactly as at inference.
        target_class: which class's score to explain. For a grading report
            this should be the PREDICTED class, not the true label -- the
            report is explaining what the model said, not grading the CAM.
        target_layer: defaults to `default_target_layer(model)`.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer = target_layer or default_target_layer(model)
    was_training = model.training
    model.eval()
    try:
        with GradCAM(model=model, target_layers=[layer]) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)]
            )
        return grayscale_cam[0]
    finally:
        model.train(was_training)


def overlay_cam(
    rgb_image: np.ndarray, grayscale_cam: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto its source image.

    Args:
        rgb_image: HxWx3 uint8, the exact image the CAM was computed on
            (post-preprocessing) -- overlaying on a differently-cropped raw
            image would misalign the heatmap with the lesions it is marking.
        grayscale_cam: output of `generate_cam`, same H and W as `rgb_image`.
        alpha: heatmap opacity.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image

    if grayscale_cam.shape[:2] != rgb_image.shape[:2]:
        grayscale_cam = cv2.resize(grayscale_cam, (rgb_image.shape[1], rgb_image.shape[0]))
    float_rgb = rgb_image.astype(np.float32) / 255.0
    overlay = show_cam_on_image(float_rgb, grayscale_cam, use_rgb=True, image_weight=1 - alpha)
    return overlay
