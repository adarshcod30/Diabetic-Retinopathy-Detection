"""Tests for the ICDR rationale and lesion-overlay rendering.

`extract_lesion_evidence` needs real trained checkpoints (it's exercised by
scripts directly, matching how drdetect.fusion.features's own model-running
code is tested) -- these tests cover the two deterministic, model-free
pieces: overlay rendering from a given evidence dict, and rationale text
generation from given lesion counts.
"""

from __future__ import annotations

import numpy as np

from drdetect.explain.evidence import build_icdr_rationale, render_lesion_overlay


def _empty_evidence(shape=(50, 50)):
    return {
        "od_xy": (25, 25),
        "fovea_xy": (10, 10),
        "lesions": {
            "hard_exudates": {"count": 0, "area_pct": 0.0, "instances": []},
            "soft_exudates": {"count": 0, "area_pct": 0.0, "instances": []},
            "haemorrhages": {"count": 0, "area_pct": 0.0, "instances": []},
            "microaneurysms": {"count": 0, "instances": []},
        },
        "masks": {
            "hard_exudates": np.zeros(shape, dtype=bool),
            "soft_exudates": np.zeros(shape, dtype=bool),
            "haemorrhages": np.zeros(shape, dtype=bool),
        },
    }


def test_render_lesion_overlay_draws_an_outline_for_a_masked_region():
    evidence = _empty_evidence()
    evidence["masks"]["hard_exudates"][10:20, 10:20] = True
    image = np.full((50, 50, 3), 128, dtype=np.uint8)

    overlay = render_lesion_overlay(image, evidence)

    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image)  # something was drawn


def test_render_lesion_overlay_does_not_fill_the_region():
    """Outlines, not blobs: the interior of a masked region should be
    mostly untouched, only its boundary drawn."""
    evidence = _empty_evidence()
    evidence["masks"]["hard_exudates"][10:30, 10:30] = True
    image = np.full((50, 50, 3), 128, dtype=np.uint8)

    overlay = render_lesion_overlay(image, evidence)

    center_unchanged = np.array_equal(overlay[18:22, 18:22], image[18:22, 18:22])
    assert center_unchanged


def test_render_lesion_overlay_marks_microaneurysm_points():
    evidence = _empty_evidence()
    evidence["lesions"]["microaneurysms"] = {
        "count": 1,
        "instances": [{"quadrant": "superior-foveal-side", "xy": (25.0, 25.0)}],
    }
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    overlay = render_lesion_overlay(image, evidence)

    assert overlay.sum() > 0  # a circle was drawn somewhere


def test_build_icdr_rationale_reports_no_evidence_for_a_clean_image():
    evidence = _empty_evidence()
    text = build_icdr_rationale(0, evidence)
    assert "no significant lesion evidence" in text
    assert "No apparent DR" in text


def test_build_icdr_rationale_includes_microaneurysm_count():
    evidence = _empty_evidence()
    evidence["lesions"]["microaneurysms"] = {"count": 5, "instances": []}
    text = build_icdr_rationale(1, evidence)
    assert "5 microaneurysm" in text
    assert "Mild NPDR" in text


def test_build_icdr_rationale_flags_missing_neovascularisation_evidence_at_grade_4():
    evidence = _empty_evidence()
    evidence["lesions"]["hard_exudates"]["area_pct"] = 5.0
    text = build_icdr_rationale(4, evidence)
    assert "neovascularisation" in text.lower()
    assert "Proliferative DR" in text


def test_build_icdr_rationale_does_not_mention_neovascularisation_below_grade_4():
    evidence = _empty_evidence()
    evidence["lesions"]["hard_exudates"]["area_pct"] = 5.0
    text = build_icdr_rationale(2, evidence)
    assert "neovascularisation" not in text.lower()
