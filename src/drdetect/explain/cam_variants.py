"""Four CAM methods, one interface, so they can be compared rather than assumed equivalent.

docs/04_ROADMAP.md Phase 6 calls for Grad-CAM, Grad-CAM++, Score-CAM and
Eigen-CAM "side by side" -- comparing them requires calling them the same
way. Phase 2 shipped only Grad-CAM (drdetect.explain.gradcam); this module
adds the other three without duplicating the eval-mode/target-layer handling
already correct there.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from drdetect.explain.gradcam import default_target_layer

__all__ = ["CAM_METHODS", "generate_cam_variant"]

CAM_METHODS = ("gradcam", "gradcam++", "scorecam", "eigencam")


def _cam_class(name: str):
    from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, ScoreCAM

    return {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
    }[name]


def generate_cam_variant(
    method: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: nn.Module | None = None,
):
    """Grayscale heatmap in [0, 1] from the named method.

    Args:
        method: one of CAM_METHODS.
        target_class: explain the PREDICTED class -- see drdetect.explain.gradcam.generate_cam.

    EigenCAM and Score-CAM do not use gradients (Eigen-CAM is a pure
    activation decomposition; Score-CAM perturbs the input and reads off
    softmax scores), so they are computed under `torch.no_grad()`, unlike
    Grad-CAM/Grad-CAM++ which require gradients to reach the target layer.
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if method not in CAM_METHODS:
        raise ValueError(f"unknown CAM method {method!r}; expected one of {CAM_METHODS}")

    layer = target_layer or default_target_layer(model)
    cam_cls = _cam_class(method)
    needs_grad = method in ("gradcam", "gradcam++")

    was_training = model.training
    model.eval()
    try:
        with torch.set_grad_enabled(needs_grad), cam_cls(model=model, target_layers=[layer]) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)]
            )
        return grayscale_cam[0]
    finally:
        model.train(was_training)
