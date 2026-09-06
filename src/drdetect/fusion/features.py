"""Lesion feature extraction: runs Phase 4's models as inference engines over
images that have no lesion ground truth of their own (APTOS), to build the
per-image feature vector Phase 5's fusion head concats with the grading
model's CNN embedding.

**Cross-dataset domain shift, faced honestly, not hidden**: every model used
here was trained and validated exclusively on IDRiD (2848x4288, one capture
device/protocol). APTOS images range from 1050x1050 to 3216x2136 across
several different capture devices, with no resolution normalisation applied
before inference -- a cheap sanity check (the hard-exudate model against 4
images per DR grade) showed mean predicted probability rising with grade
even so (grade 0: 0.0000-0.0025 positive-pixel fraction at threshold 0.5;
grade 3: 0.023-0.114), which is why this was judged worth building rather
than assumed broken -- but it was never validated against real APTOS lesion
ground truth (APTOS ships none), so treat these as a genuine, undemonstrated
hypothesis under the fusion head's own McNemar test, not a calibrated
measurement the way IDRiD-evaluated numbers elsewhere in this project are.

IDRiD's own grading images ("B. Disease Grading") are byte-identical to its
localisation images ("C. Localization") -- real OD/fovea/lesion ground truth
would be available there. Deliberately not used: docs/04_ROADMAP.md Phase 8
reserves IDRiD (with Messidor-2) as the locked, evaluate-once final test
set, and using it to build or tune the fusion head now would contaminate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from drdetect.segmentation.localization import (
    LocalizationModule,
    heatmap_argmax,
    working_res_od_diameter,
)
from drdetect.segmentation.microaneurysms import ClassifierModule, PatchClassifier, score_candidates
from drdetect.segmentation.model import build_segmentation_model
from drdetect.segmentation.module import SegmentationModule
from drdetect.segmentation.quadrants import assign_quadrant

__all__ = [
    "LesionModels",
    "load_lesion_models",
    "extract_lesion_features",
    "lesion_centroids",
    "FEATURE_NAMES",
    "QUADRANT_ORDER",
]

LOCALIZATION_SIZE = (512, 768)  # (height, width), matches training
SEGMENTATION_PATCH = 512
MA_TUNED_THRESHOLD = 0.95  # docs/15's val-tuned operating point, frozen there

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

QUADRANT_ORDER = [
    "superior-foveal-side",
    "inferior-foveal-side",
    "superior-disc-side",
    "inferior-disc-side",
]
FEATURE_NAMES = [
    "hard_exudate_mean_prob",
    "soft_exudate_mean_prob",
    "haemorrhage_mean_prob",
    "microaneurysm_count",
    "distance_to_fovea_mean",
    *[f"quadrant_frac_{q}" for q in QUADRANT_ORDER],
]


@dataclass
class LesionModels:
    localization: LocalizationModule
    hard_exudates: SegmentationModule
    soft_exudates: SegmentationModule
    haemorrhages: SegmentationModule
    microaneurysm_classifier: ClassifierModule
    device: torch.device


def _load_segmentation(checkpoint: str, device: torch.device) -> SegmentationModule:
    model = build_segmentation_model("resnet34", pretrained=False, classes=1)
    module = SegmentationModule.load_from_checkpoint(checkpoint, model=model, map_location=device)
    return module.eval().to(device)


def load_lesion_models(
    *,
    localization_checkpoint: str,
    hard_exudate_checkpoint: str,
    soft_exudate_checkpoint: str,
    haemorrhage_checkpoint: str,
    microaneurysm_checkpoint: str,
    device: torch.device,
) -> LesionModels:
    loc_model = build_segmentation_model("resnet34", pretrained=False, classes=2)
    localization = (
        LocalizationModule.load_from_checkpoint(
            localization_checkpoint, model=loc_model, map_location=device
        )
        .eval()
        .to(device)
    )

    ma_model = PatchClassifier()
    microaneurysm_classifier = (
        ClassifierModule.load_from_checkpoint(
            microaneurysm_checkpoint, model=ma_model, map_location=device
        )
        .eval()
        .to(device)
    )

    return LesionModels(
        localization=localization,
        hard_exudates=_load_segmentation(hard_exudate_checkpoint, device),
        soft_exudates=_load_segmentation(soft_exudate_checkpoint, device),
        haemorrhages=_load_segmentation(haemorrhage_checkpoint, device),
        microaneurysm_classifier=microaneurysm_classifier,
        device=device,
    )


def _predict_od_fovea(
    image_rgb: np.ndarray, models: LesionModels
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Returns (od_xy, fovea_xy, od_diameter), all in LOCALIZATION_SIZE's
    resized pixel space (not the input image's own resolution)."""
    resized = cv2.resize(image_rgb, (LOCALIZATION_SIZE[1], LOCALIZATION_SIZE[0]))
    rgb = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(models.device)
    with torch.no_grad():
        logits = models.localization(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    od_xy = heatmap_argmax(probs[0])
    fovea_xy = heatmap_argmax(probs[1])
    diameter = working_res_od_diameter(LOCALIZATION_SIZE)
    return od_xy, fovea_xy, diameter


def _tile_predict(
    module: SegmentationModule, image_rgb: np.ndarray, device: torch.device
) -> np.ndarray:
    """Same non-overlapping reflect-padded tiling as
    scripts/evaluate_segmentation.py's tile_predict, inlined here rather than
    imported from a script module so library code doesn't depend on scripts/."""
    patch_size = SEGMENTATION_PATCH
    h, w = image_rgb.shape[:2]
    ph = (patch_size - h % patch_size) % patch_size
    pw = (patch_size - w % patch_size) % patch_size
    padded = cv2.copyMakeBorder(image_rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    ph_full, pw_full = padded.shape[:2]

    prob_map = np.zeros((ph_full, pw_full), dtype=np.float32)
    tiles, positions = [], []
    for top in range(0, ph_full, patch_size):
        for left in range(0, pw_full, patch_size):
            tiles.append(padded[top : top + patch_size, left : left + patch_size])
            positions.append((top, left))

    batch_size = 4
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            chunk = tiles[start : start + batch_size]
            arrs = [
                ((t.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1) for t in chunk
            ]
            batch = torch.from_numpy(np.stack(arrs)).float().to(device)
            logits = module(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            for (top, left), prob in zip(positions[start : start + batch_size], probs, strict=True):
                prob_map[top : top + patch_size, left : left + patch_size] = prob
    return prob_map[:h, :w]


def lesion_centroids(prob_map: np.ndarray, threshold: float = 0.5) -> list[tuple[float, float]]:
    """Connected-component centroids of a thresholded probability map -- a
    handful of per-lesion centroids is what quadrant/distance features need,
    not per-pixel detail."""
    binary = (prob_map > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [tuple(centroids[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3]


def extract_lesion_features(image_path: str | Path, models: LesionModels) -> dict[str, float]:
    """One image -> the flat feature dict Phase 5's fusion head concats with
    the grading CNN's embedding. Every model runs at its own native working
    convention (full-resolution tiling for the three DeepLabV3+ segmentation
    models, a fixed 512x768 resize for localisation, full-resolution
    candidate generation for microaneurysms) -- see this module's own
    docstring for the cross-dataset resolution caveat that follows from that.
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    od_xy, fovea_xy, od_diameter = _predict_od_fovea(image_rgb, models)

    he_prob = _tile_predict(models.hard_exudates, image_rgb, models.device)
    se_prob = _tile_predict(models.soft_exudates, image_rgb, models.device)
    ha_prob = _tile_predict(models.haemorrhages, image_rgb, models.device)

    # Segmentation ran at the image's OWN resolution; localisation predicts in
    # LOCALIZATION_SIZE's resized space. Rescale lesion centroids into that
    # same space before computing distance/quadrant features, since od_xy/
    # fovea_xy live there.
    scale_x = LOCALIZATION_SIZE[1] / image_rgb.shape[1]
    scale_y = LOCALIZATION_SIZE[0] / image_rgb.shape[0]

    all_centroids: list[tuple[float, float]] = []
    for prob_map in (he_prob, se_prob, ha_prob):
        for cx, cy in lesion_centroids(prob_map):
            all_centroids.append((cx * scale_x, cy * scale_y))

    # score_candidates expects a ground-truth mask (it's normally used for
    # scored evaluation, see evaluate_microaneurysms.py) purely to compute
    # labels/recall bookkeeping this call never reads -- an all-zero dummy
    # mask makes that bookkeeping cheap and correct-by-construction (0
    # recovered of 0 true instances) without a separate inference-only code
    # path for a function that otherwise does exactly what's needed here.
    dummy_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    ma_result = score_candidates(
        image_bgr, dummy_mask, models.microaneurysm_classifier.model, device=str(models.device)
    )
    accepted = ma_result["scores"] > MA_TUNED_THRESHOLD
    ma_count = int(accepted.sum())
    for c, keep in zip(ma_result["candidates"], accepted, strict=True):
        if keep:
            all_centroids.append((c.cx * scale_x, c.cy * scale_y))

    if all_centroids:
        distances = [
            float(np.linalg.norm(np.array(c) - np.array(fovea_xy))) / od_diameter
            for c in all_centroids
        ]
        distance_to_fovea_mean = float(np.mean(distances))
        quadrant_counts = dict.fromkeys(QUADRANT_ORDER, 0)
        for c in all_centroids:
            quadrant_counts[assign_quadrant(c, od_xy, fovea_xy)] += 1
        total = len(all_centroids)
        quadrant_fracs = {q: quadrant_counts[q] / total for q in QUADRANT_ORDER}
    else:
        distance_to_fovea_mean = float("nan")
        quadrant_fracs = dict.fromkeys(QUADRANT_ORDER, 0.0)

    return {
        "hard_exudate_mean_prob": float(he_prob.mean()),
        "soft_exudate_mean_prob": float(se_prob.mean()),
        "haemorrhage_mean_prob": float(ha_prob.mean()),
        "microaneurysm_count": float(ma_count),
        "distance_to_fovea_mean": distance_to_fovea_mean,
        **{f"quadrant_frac_{q}": quadrant_fracs[q] for q in QUADRANT_ORDER},
    }
