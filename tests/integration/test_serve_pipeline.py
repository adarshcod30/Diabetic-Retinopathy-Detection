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
