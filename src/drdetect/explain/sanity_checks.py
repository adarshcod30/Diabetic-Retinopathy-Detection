"""Adebayo-style sanity checks for saliency methods.

Adebayo et al. 2018 (NeurIPS), "Sanity Checks for Saliency Maps": several
widely-used saliency methods are empirically insensitive to the model's
learned parameters -- they produce nearly the same map whether the model is
trained or has its weights replaced with noise. A method that fails this is
not explaining the MODEL; it behaves closer to an edge detector applied to
the input, dressed up as an explanation. Nobody knew whether that applied
here until it was checked -- this project's own culture treats an unchecked
assumption as no different from a wrong one.

Two checks, both from the paper:

  Model parameter randomisation (`model_randomization_test`) -- cascading:
      progressively randomise the model from the OUTPUT layer backward
      toward the input, and measure how much each CAM method's heatmap
      changes at each step, against its own original heatmap. A method that
      PASSES should diverge quickly; a method whose heatmap barely moves as
      the whole model turns to noise is not using what the model learned.

  Data randomisation (`compare_against_shuffled_labels`) -- compare CAMs
      from the real trained model against CAMs from an otherwise-identical
      model trained on the SAME images with shuffled labels. A method that
      PASSES should look different between the two, since the two models
      learned genuinely different things; a method that looks similar
      regardless of what the underlying model actually learned is not
      sensitive to that.

Similarity is reported as both Spearman rank correlation and SSIM, matching
the two metrics the original paper used -- rank correlation alone can be
fooled by a saliency map that reorders pixels similarly but at a different
spatial scale, which SSIM is sensitive to and rank correlation is not.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from drdetect.explain.cam_variants import CAM_METHODS, generate_cam_variant

__all__ = [
    "CascadeStep",
    "cam_similarity",
    "cascading_randomization_layers",
    "model_randomization_test",
    "compare_against_shuffled_labels",
]


@dataclass(frozen=True)
class CascadeStep:
    layer_randomized: str
    spearman: dict[str, float]
    ssim: dict[str, float]


def cam_similarity(cam_a: np.ndarray, cam_b: np.ndarray) -> tuple[float, float]:
    """(Spearman rank correlation, SSIM) between two same-shape grayscale CAMs."""
    from scipy.stats import spearmanr
    from skimage.metrics import structural_similarity

    rho = float(spearmanr(cam_a.ravel(), cam_b.ravel()).statistic)
    ssim = float(structural_similarity(cam_a, cam_b, data_range=1.0))
    return rho, ssim


def cascading_randomization_layers(backbone: nn.Module) -> list[tuple[str, list[nn.Module]]]:
    """Named layer groups of a timm EfficientNet backbone, OUTPUT-to-INPUT order.

    This order is the point: Adebayo's cascading test randomises from the
    output backward, one layer group at a time, so that at step k the model
    is "the real model, except everything from the output back through
    group k is now noise." Reversed (input-to-output) would be a different,
    weaker test -- early layers dominate less of what a CAM at `bn2`
    (drdetect.explain.gradcam.default_target_layer) actually reflects.
    """
    layers = [
        ("classifier", [backbone.classifier]),
        ("conv_head+bn2", [backbone.conv_head, backbone.bn2]),
    ]
    for i in reversed(range(len(backbone.blocks))):
        layers.append((f"blocks[{i}]", [backbone.blocks[i]]))
    layers.append(("conv_stem+bn1", [backbone.conv_stem, backbone.bn1]))
    return layers


def _reset_in_place(modules: list[nn.Module]) -> None:
    for module in modules:
        for child in module.modules():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()


def model_randomization_test(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    methods: tuple[str, ...] = CAM_METHODS,
) -> list[CascadeStep]:
    """Cascading model-randomisation sanity check across every CAM method at once.

    Works on a deep copy -- the caller's trained model is never mutated.
    Every method's CAM is computed against the SAME progressively-randomised
    model at each step, so the comparison at a given step is apples to apples
    across methods, not just against each method's own baseline.
    """
    randomized = copy.deepcopy(model)
    backbone = getattr(randomized, "backbone", randomized)
    layer_groups = cascading_randomization_layers(backbone)

    originals = {
        m: generate_cam_variant(m, randomized, input_tensor, target_class) for m in methods
    }

    steps = []
    for name, modules in layer_groups:
        _reset_in_place(modules)
        spearman, ssim = {}, {}
        for method in methods:
            cam = generate_cam_variant(method, randomized, input_tensor, target_class)
            rho, s = cam_similarity(originals[method], cam)
            spearman[method] = rho
            ssim[method] = s
        steps.append(CascadeStep(layer_randomized=name, spearman=spearman, ssim=ssim))
    return steps


def compare_against_shuffled_labels(
    real_model: nn.Module,
    shuffled_model: nn.Module,
    input_tensor: torch.Tensor,
    target_class_real: int,
    target_class_shuffled: int,
    methods: tuple[str, ...] = CAM_METHODS,
) -> dict[str, tuple[float, float]]:
    """Data-randomisation check: CAM(real model) vs CAM(shuffled-label model), same image.

    Args:
        target_class_real / target_class_shuffled: each model's OWN predicted
            class for this image -- the two models were trained on different
            label associations and will usually disagree, and each CAM must
            explain what its own model actually predicted.

    Returns: {method: (spearman, ssim)}.
    """
    return {
        method: cam_similarity(
            generate_cam_variant(method, real_model, input_tensor, target_class_real),
            generate_cam_variant(method, shuffled_model, input_tensor, target_class_shuffled),
        )
        for method in methods
    }
