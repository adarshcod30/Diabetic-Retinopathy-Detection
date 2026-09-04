"""Tests for the explainability module.

Score-CAM is deliberately NOT covered here: it does one forward pass per
target-layer channel (1280 for this backbone), which takes ~15 minutes on
CPU. That cost is orthogonal to correctness -- it was still worth verifying
manually once (see docs/08_PHASE6_RESULTS.md) -- but it does not belong in
a test suite meant to run in seconds. The target-layer test below is what
actually guards against the regression Score-CAM caught: it is a fast,
direct check on the thing that broke, not a slow re-run of the symptom.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drdetect.explain.cam_variants import generate_cam_variant
from drdetect.explain.gradcam import default_target_layer, generate_cam, overlay_cam
from drdetect.grading.model import build_model


@pytest.fixture
def untrained_model():
    return build_model("efficientnet_b0", num_outputs=5, pretrained=False, freeze_bn=True)


def test_default_target_layer_is_bn2_not_conv_head(untrained_model):
    """Regression test for a real bug: conv_head is a bare pre-activation
    Conv2d, and Score-CAM's always-positive weights times a part-negative
    activation, then ReLU'd, produced an all-zero heatmap on a real image.
    bn2 is timm's fused BatchNorm+SiLU -- genuinely post-activation."""
    layer = default_target_layer(untrained_model)
    assert layer is untrained_model.backbone.bn2
    assert layer is not untrained_model.backbone.conv_head


def test_gradcam_output_contract(untrained_model):
    """Shape/range/dtype only -- NOT "is non-zero".

    An untrained, randomly-initialised model's Grad-CAM genuinely can be
    all-zero for a given seed: the channel weights come from gradients of a
    random classifier layer, which are themselves effectively random, and a
    net-negative weighted sum gets ReLU'd to zero same as it did for
    Score-CAM. That is expected on random weights, not a regression -- it is
    the exact property the model-randomisation sanity check (sanity_checks.py)
    exploits, and manually verified non-degenerate on the real trained
    checkpoint (docs/08_PHASE6_RESULTS.md). Do not "fix" this test by
    asserting non-zero; that would be asserting something false.
    """
    x = torch.randn(1, 3, 64, 64)
    cam = generate_cam(untrained_model, x, target_class=0)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("method", ["gradcam", "gradcam++", "eigencam"])
def test_cam_variants_output_contract(untrained_model, method):
    """Shape/range/dtype only -- see test_gradcam_output_contract for why
    "non-zero" is not asserted here. Score-CAM excluded -- see module docstring."""
    x = torch.randn(1, 3, 64, 64)
    cam = generate_cam_variant(method, untrained_model, x, target_class=0)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5


def test_overlay_cam_preserves_shape():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    cam = rng.random((64, 64)).astype(np.float32)
    overlay = overlay_cam(image, cam)
    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8
