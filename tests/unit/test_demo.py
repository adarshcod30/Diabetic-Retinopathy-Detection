"""Tests for the Gradio demo's own result-formatting logic.

`_format_summary` is pure (PredictionResult -> str), so these construct the
dataclass directly rather than running a real model through it -- fast, and
precise about which field drives which branch.
"""

from __future__ import annotations

from drdetect.quality.assessment import QualityResult
from drdetect.serve.demo import _format_summary
from drdetect.serve.pipeline import PredictionResult


class TestFormatSummary:
    def test_rejected_and_not_forced_shows_no_grade(self):
        """The genuine reject case: no grade was ever computed."""
        result = PredictionResult(
            quality=QualityResult(usable=False, reasons=["image too blurry (sharpness 0 < 15)"])
        )
        summary = _format_summary(result)
        assert "REJECTED" in summary
        assert "Grade" not in summary  # no grade heading anywhere

    def test_rejected_but_forced_still_shows_the_grade(self):
        """Regression test for a real bug found manually driving the live HF
        Space (2026-09-07, docs/04_ROADMAP.md Phase 9): checking "Grade
        anyway if quality gate rejects" made `run_pipeline` compute a real
        grade (skip_quality_gate=True), but this function used to branch on
        `not quality.usable` alone and returned the rejection text without
        ever looking at `result.grade` -- the forced grade was silently
        thrown away in the UI even though the pipeline computed it
        correctly. Must show the grade AND keep the quality warning visible,
        not pick one or the other."""
        result = PredictionResult(
            quality=QualityResult(usable=False, reasons=["underexposed (mean brightness 0)"]),
            grade=0,
            grade_name="No DR",
            referable=False,
            confidence=0.9,
            class_probs=[0.9, 0.05, 0.02, 0.02, 0.01],
        )
        summary = _format_summary(result)
        assert "Grade 0: No DR" in summary
        assert "underexposed" in summary  # the quality warning is not dropped
        assert "anyway" in summary.lower()  # flagged as an overridden gate, not a normal pass
        assert "REJECTED" not in summary  # a grade exists; this is not the no-grade case

    def test_usable_normal_grade_has_no_override_warning(self):
        result = PredictionResult(
            quality=QualityResult(
                usable=True, sharpness=500, mean_brightness=120, fov_fraction=0.8
            ),
            grade=2,
            grade_name="Moderate",
            referable=True,
            confidence=0.75,
            class_probs=[0.05, 0.1, 0.75, 0.05, 0.05],
        )
        summary = _format_summary(result)
        assert "Grade 2: Moderate" in summary
        assert "Referable (grade >= 2):** YES" in summary
        assert "anyway" not in summary.lower()
        assert "REJECTED" not in summary
