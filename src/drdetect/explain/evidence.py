"""ICDR evidence table + templated natural-language rationale, and outline-based
lesion-overlay rendering for the final report (docs/04_ROADMAP.md, Phase 6).

Reuses `drdetect.fusion.features`'s model-running internals (the same tiled
segmentation inference, OD/fovea localisation, and microaneurysm candidate
scoring already built and evaluated for Phase 5's fusion head) rather than
re-implementing lesion detection a second time -- this module only adds the
per-instance/per-quadrant structure a human-readable report needs, on top of
the aggregate scalar features Phase 5's fusion head consumes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drdetect.fusion.features import (
    LOCALIZATION_SIZE,
    LesionModels,
    _predict_od_fovea,
    _tile_predict,
    lesion_centroids,
)
from drdetect.segmentation.microaneurysms import score_candidates
from drdetect.segmentation.quadrants import assign_quadrant

__all__ = [
    "extract_lesion_evidence",
    "render_lesion_overlay",
    "build_icdr_rationale",
    "LESION_COLORS_RGB",
]

_SEGMENTED_LESIONS = ["hard_exudates", "soft_exudates", "haemorrhages"]
# RGB, since render_lesion_overlay draws on an RGB array throughout this module.
LESION_COLORS_RGB = {
    "hard_exudates": (255, 215, 0),  # amber
    "soft_exudates": (0, 191, 255),  # light blue
    "haemorrhages": (255, 0, 0),  # red
    "microaneurysms": (255, 0, 255),  # magenta
}


def extract_lesion_evidence(image_path: str | Path, models: LesionModels) -> dict:
    """Per-lesion-type instance list (quadrant + size) and binary masks for
    rendering, from the same models `drdetect.fusion.features` uses for the
    fusion head's aggregate scalars."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    od_xy, fovea_xy, _diameter = _predict_od_fovea(image_rgb, models)
    scale_x = LOCALIZATION_SIZE[1] / image_rgb.shape[1]
    scale_y = LOCALIZATION_SIZE[0] / image_rgb.shape[0]

    lesions: dict[str, dict] = {}
    masks: dict[str, np.ndarray] = {}
    for lesion_type in _SEGMENTED_LESIONS:
        module = getattr(models, lesion_type)
        prob_map = _tile_predict(module, image_rgb, models.device)
        binary_mask = prob_map > 0.5
        masks[lesion_type] = binary_mask
        centroids = lesion_centroids(prob_map, threshold=0.5)
        instances = [
            {"quadrant": assign_quadrant((cx * scale_x, cy * scale_y), od_xy, fovea_xy)}
            for cx, cy in centroids
        ]
        lesions[lesion_type] = {
            "count": len(instances),
            "area_pct": float(binary_mask.mean() * 100.0),
            "instances": instances,
        }

    dummy_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    ma_result = score_candidates(
        image_bgr, dummy_mask, models.microaneurysm_classifier.model, device=str(models.device)
    )
    accepted = ma_result["scores"] > 0.95
    ma_instances = [
        {
            "quadrant": assign_quadrant((c.cx * scale_x, c.cy * scale_y), od_xy, fovea_xy),
            "xy": (float(c.cx), float(c.cy)),
        }
        for c, keep in zip(ma_result["candidates"], accepted, strict=True)
        if keep
    ]
    lesions["microaneurysms"] = {"count": len(ma_instances), "instances": ma_instances}

    return {"od_xy": od_xy, "fovea_xy": fovea_xy, "lesions": lesions, "masks": masks}


def render_lesion_overlay(image_rgb: np.ndarray, evidence: dict) -> np.ndarray:
    """Outline contours, not filled blobs -- the roadmap's own stated reason:
    outlines beat blobs for clinical legibility (a filled overlay hides the
    tissue underneath it; an outline points at it without obscuring it).

    Microaneurysms are marked with small circles rather than outlines --
    they're candidate *points* from the classifier (drdetect.segmentation.
    microaneurysms), not a segmented region, so there's no contour to draw.
    """
    overlay = image_rgb.copy()
    for lesion_type, mask in evidence["masks"].items():
        color = LESION_COLORS_RGB[lesion_type]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, thickness=3)

    ma_color = LESION_COLORS_RGB["microaneurysms"]
    for inst in evidence["lesions"].get("microaneurysms", {}).get("instances", []):
        x, y = inst["xy"]
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius=12, color=ma_color, thickness=2)

    return overlay


_ICDR_GRADE_NAMES = {
    0: "No apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


def build_icdr_rationale(grade: int, evidence: dict) -> str:
    """A templated sentence grounding the predicted ICDR grade in the actual
    detected evidence -- not a restatement of the grade, an explanation of
    what supports it (or, honestly, what doesn't fully support it).

    Neovascularisation (the feature that actually defines grade 4, PDR) has
    no segmentation model in this project (docs/01's own scope-honesty
    note: "neovascularisation has no public pixel masks... document as
    future work, do not fabricate it") -- a grade-4 rationale says so
    explicitly rather than inventing NV evidence this pipeline cannot see.
    """
    lesions = evidence["lesions"]
    ma_n = lesions.get("microaneurysms", {}).get("count", 0)
    he_n = lesions.get("haemorrhages", {}).get("count", 0)
    ex_pct = lesions.get("hard_exudates", {}).get("area_pct", 0.0)
    se_pct = lesions.get("soft_exudates", {}).get("area_pct", 0.0)

    quadrants_with_he = {
        inst["quadrant"] for inst in lesions.get("haemorrhages", {}).get("instances", [])
    }

    evidence_parts = []
    if ma_n:
        evidence_parts.append(f"{ma_n} microaneurysm candidate(s)")
    if he_n:
        evidence_parts.append(
            f"{he_n} haemorrhage region(s) across {len(quadrants_with_he)} quadrant(s)"
        )
    if ex_pct > 0.01:
        evidence_parts.append(f"hard exudates covering {ex_pct:.2f}% of the image")
    if se_pct > 0.01:
        evidence_parts.append(
            f"soft exudates (cotton-wool spots) covering {se_pct:.2f}% of the image"
        )

    grade_name = _ICDR_GRADE_NAMES.get(grade, f"grade {grade}")
    if not evidence_parts:
        return f"ICDR grade {grade} ({grade_name}): no significant lesion evidence detected."

    body = "; ".join(evidence_parts)
    note = ""
    if grade == 4:
        note = (
            " Note: proliferative DR (grade 4) is defined by neovascularisation, which this "
            "pipeline does not detect (no public pixel masks exist to train against) -- this "
            "grade is the classifier's own image-level prediction, not evidence-backed for the "
            "specific feature that defines it."
        )
    return f"ICDR grade {grade} ({grade_name}): {body}.{note}"
