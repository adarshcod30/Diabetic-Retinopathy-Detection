"""End-to-end test of the Phase 2 vertical slice: quality -> grade -> Grad-CAM -> PDF.

Uses an untrained (`pretrained=False`) model rather than a real checkpoint --
the point is to prove the plumbing (shapes, dtypes, the reject/accept branch,
a valid PDF on disk), not to check grading accuracy. Semantic correctness of
the model itself is what Phases 1/3's evaluation scripts are for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from drdetect.explain.gradcam import generate_cam
from drdetect.grading.losses import outputs_for_loss
from drdetect.grading.model import build_model
from drdetect.serve.pipeline import run_pipeline
from drdetect.serve.report import build_report_pdf


@pytest.fixture
def textured_fundus() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:600, :800]
    disc = (yy - 300) ** 2 + (xx - 420) ** 2 <= 280**2
    speckle = rng.integers(0, 255, size=(600, 800), dtype=np.uint8)
    for c, base in enumerate((150, 80, 60)):
        channel = np.clip(base + speckle.astype(int) - 128, 0, 255).astype(np.uint8)
        img[..., c] = np.where(disc, channel, 0)
    return img


@pytest.fixture
def untrained_model():
    return build_model("efficientnet_b0", num_outputs=5, pretrained=False, freeze_bn=True)


def test_pipeline_accepts_good_image(textured_fundus, untrained_model):
    result = run_pipeline(textured_fundus, untrained_model, loss_name="ce", size=224)
    assert result.quality.usable
    assert result.grade in {0, 1, 2, 3, 4}
    assert result.grade_name is not None
    assert result.referable == (result.grade >= 2)
    assert 0.0 <= result.confidence <= 1.0
    assert result.class_probs is not None
    assert len(result.class_probs) == 5
    assert abs(sum(result.class_probs) - 1.0) < 1e-4
    assert result.cam_overlay.shape == (224, 224, 3)
    assert result.cam_overlay.dtype == np.uint8


def test_pipeline_rejects_bad_image_without_grading(untrained_model):
    black = np.zeros((600, 800, 3), dtype=np.uint8)
    result = run_pipeline(black, untrained_model, loss_name="ce", size=224)
    assert not result.quality.usable
    assert result.grade is None
    assert result.cam_overlay is None


def test_pipeline_skip_quality_gate_forces_grading(untrained_model):
    black = np.zeros((600, 800, 3), dtype=np.uint8)
    result = run_pipeline(black, untrained_model, loss_name="ce", size=224, skip_quality_gate=True)
    assert not result.quality.usable  # still reports the rejection
    assert result.grade is not None  # but graded anyway


def test_report_pdf_written_for_accepted_image(tmp_path: Path, textured_fundus, untrained_model):
    result = run_pipeline(textured_fundus, untrained_model, loss_name="ce", size=224)
    out = build_report_pdf(result, "synthetic.png", tmp_path / "report.pdf")
    assert out.exists()
    assert out.read_bytes()[:5] == b"%PDF-"
    assert out.stat().st_size > 1000


def test_report_pdf_written_for_rejected_image(tmp_path: Path, untrained_model):
    black = np.zeros((600, 800, 3), dtype=np.uint8)
    result = run_pipeline(black, untrained_model, loss_name="ce", size=224)
    out = build_report_pdf(result, "synthetic.png", tmp_path / "reject.pdf")
    assert out.exists()
    assert out.read_bytes()[:5] == b"%PDF-"


class TestCamTargetIndexAcrossLossHeads:
    """Regression test for a real bug found deploying the released
    regression-loss checkpoint (docs/22): Grad-CAM's `ClassifierOutputTarget`
    indexes directly into the model's raw output. A CE/distance_ce head has 5
    outputs, so `target_class=grade` (0-4) is always valid -- but a
    regression head has exactly 1 output (`outputs_for_loss("regression")`),
    so the SAME code indexed position 2 of a 1-element tensor the moment the
    model predicted grade 2, crashing with an IndexError inside
    pytorch_grad_cam that surfaced as an UnboundLocalError in generate_cam.
    Every test in this file above used loss_name="ce" exclusively, which is
    exactly how this went undetected until a live deployment hit a real
    non-zero-grade image."""

    @pytest.mark.parametrize("loss_name", ["ce", "corn", "regression", "distance_ce"])
    def test_run_pipeline_does_not_crash_for_any_loss_head(self, textured_fundus, loss_name):
        model = build_model(
            "efficientnet_b0",
            num_outputs=outputs_for_loss(loss_name),
            pretrained=False,
            freeze_bn=True,
        )
        result = run_pipeline(textured_fundus, model, loss_name=loss_name, size=224)
        assert result.grade in {0, 1, 2, 3, 4}
        assert result.cam_overlay is not None
        assert result.cam_overlay.shape == (224, 224, 3)

    def test_regression_head_always_clamps_to_its_only_output(self):
        """The exact scenario that crashed: a 1-output head asked to explain
        a non-zero grade. Deterministic (no randomness), unlike the
        pipeline-level test above, which depends on what an untrained model
        happens to predict."""
        model = build_model(
            "efficientnet_b0",
            num_outputs=outputs_for_loss("regression"),
            pretrained=False,
            freeze_bn=True,
        )
        tensor = torch.randn(1, 3, 224, 224)
        for target_grade in range(5):  # every possible decoded grade, 0-4
            cam_target = min(target_grade, outputs_for_loss("regression") - 1)
            assert cam_target == 0  # the only valid index into a 1-output head
            generate_cam(model, tensor, target_class=cam_target)  # must not raise


class TestEnsembleInference:
    """`run_pipeline` also accepts a list of models (the live demo's 5-fold
    CNN ensemble, docs/../project memory: beats the single shipped checkpoint
    on DDR referable AUC). Averaging happens before decode for both the grade
    output and the Grad-CAM heatmap -- these check that plumbing, not
    grading accuracy (same division of concern as the rest of this file)."""

    def test_ensemble_of_one_matches_a_plain_single_model_call(self, textured_fundus):
        """A list of one model must be a true no-op: same averaged output,
        same decoded grade, same CAM -- not a special case that happens to
        look similar. `.eval()` matters here: `build_model` alone leaves the
        head's Dropout active (train mode is the nn.Module default), which
        makes even two calls on the SAME model object stochastic -- exactly
        what `load_grader` (the real caller) always fixes before inference."""
        model = build_model("efficientnet_b0", num_outputs=5, pretrained=False, freeze_bn=True)
        model.eval()
        single = run_pipeline(textured_fundus, model, loss_name="ce", size=224)
        ensembled = run_pipeline(textured_fundus, [model], loss_name="ce", size=224)
        assert ensembled.grade == single.grade
        assert ensembled.class_probs == pytest.approx(single.class_probs)
        np.testing.assert_array_equal(ensembled.cam_overlay, single.cam_overlay)

    def test_ensemble_averages_raw_outputs_before_decode(self, textured_fundus):
        """Averaging must happen on raw logits before softmax, not on softmax
        output afterward -- softmax(mean(logits)) != mean(softmax(logits))
        in general, since softmax is non-linear. This is the same
        averaging-before-decode convention already established for hflip TTA
        (evaluate.py) and cross-checkpoint ensembles (evaluate_ddr.py); an
        implementation that accidentally averaged post-softmax instead would
        still produce a valid-looking probability vector, so a plain
        "is it a valid distribution" check would not catch the bug -- this
        recomputes the expected logit average independently and compares."""
        from drdetect.data.dataset import build_transforms
        from drdetect.enhance.preprocessing import preprocess

        models = [
            build_model("efficientnet_b0", num_outputs=5, pretrained=False, freeze_bn=True).eval()
            for _ in range(2)
        ]
        result = run_pipeline(textured_fundus, models, loss_name="ce", size=224)
        assert result.grade in {0, 1, 2, 3, 4}
        assert abs(sum(result.class_probs) - 1.0) < 1e-4

        pre = preprocess(textured_fundus, size=224, use_ben_graham=True)
        tensor = build_transforms(224, train=False)(image=pre)["image"].unsqueeze(0)
        with torch.no_grad():
            expected_logits = sum(m(tensor) for m in models) / len(models)
        expected_probs = torch.softmax(expected_logits, dim=1)[0].tolist()
        assert result.class_probs == pytest.approx(expected_probs, abs=1e-5)
