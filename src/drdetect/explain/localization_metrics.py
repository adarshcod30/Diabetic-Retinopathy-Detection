"""CAM-vs-ground-truth localisation quality: pointing game + IoU.

Phase 6's own exit criterion (docs/04_ROADMAP.md): "a table of saliency
methods x (sanity-check pass/fail, pointing-game accuracy, IoU)". This
module computes the pointing-game and IoU halves of that table, against
IDRiD's real per-lesion-type pixel masks -- the sanity-check half already
exists in drdetect.explain.sanity_checks (docs/08_PHASE6_RESULTS.md).
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["upsample_cam", "pointing_game_hit", "cam_mask_iou"]


def upsample_cam(cam: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a CAM (native resolution: the model's input size, e.g. 512x512)
    up to `target_shape` (height, width) -- typically a lesion mask's own
    full resolution. A CAM is already a smooth, interpolated heatmap, so
    upsampling it preserves the mask's own pixel precision, rather than
    downsampling the mask and losing the smallest lesions -- microaneurysms
    are already known, from this project's own Phase 3 findings, to vanish
    below ~1024px."""
    h, w = target_shape
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)


def pointing_game_hit(cam: np.ndarray, mask: np.ndarray) -> bool:
    """True if the CAM's single point of maximum attention falls on a
    positive mask pixel -- the standard pointing-game definition (Zhang et
    al. 2018), threshold-free by construction."""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(mask[y, x] > 0)


def cam_mask_iou(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between the CAM thresholded at a fixed value and the binary
    lesion mask.

    Expect this to read low for genuinely sparse lesion types
    (haemorrhages and microaneurysms are well under 0.1% of image pixels,
    see drdetect.segmentation.metrics) even for a CAM correctly pointing at
    the right general region -- Grad-CAM produces a smooth, diffuse
    attention region by construction, not a pixel-precise segmentation, so
    IoU against a sparse mask is a much harder bar than the pointing game.
    NaN, not 0, when the union is empty (both CAM-positive and mask are
    empty) -- 0 would misrepresent "nothing to compare" as "total miss".
    """
    binary_cam = cam > threshold
    binary_mask = mask.astype(bool)
    intersection = np.logical_and(binary_cam, binary_mask).sum()
    union = np.logical_or(binary_cam, binary_mask).sum()
    return float(intersection / union) if union > 0 else float("nan")
