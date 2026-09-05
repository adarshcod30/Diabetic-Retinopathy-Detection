"""Pixel-level segmentation metrics.

AUPRC, not AUROC, is the metric of record here -- the roadmap says why
(docs/04_ROADMAP.md, Phase 4): "AUROC is meaningless at <0.1% positive
pixels." Measured directly on this project's own data (IDRiD_41_EX.tif):
0.068% positive. AUROC's false-positive-rate denominator is dominated by the
~99.9% true negatives regardless of model quality, so a model that is barely
better than chance can still post a high AUROC; AUPRC's precision term is not
protected by that denominator and actually penalises a flood of false
positives.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pixel_auprc", "dice_coefficient", "best_dice_threshold"]


def pixel_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision over every pixel passed in, pooled -- not averaged
    per image first. Per-image AUPRC is unstable (often undefined) for an
    image with zero or a handful of positive pixels; pooling across images
    before scoring is what IDRiD-scale lesion evaluation actually needs.

    Args:
        y_true: any shape, values in {0, 1} (or bool).
        y_score: same shape, predicted probability in [0, 1].
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel().astype(float)
    if y_true.sum() == 0:
        raise ValueError("no positive pixels in y_true -- AUPRC is undefined without any")
    return float(average_precision_score(y_true, y_score))


def dice_coefficient(y_true: np.ndarray, y_pred_binary: np.ndarray, *, eps: float = 1e-7) -> float:
    """2*|A∩B| / (|A|+|B|) at a fixed threshold -- reported alongside AUPRC,
    not instead of it, since Dice needs a threshold AUPRC does not."""
    y_true = np.asarray(y_true).ravel().astype(bool)
    y_pred = np.asarray(y_pred_binary).ravel().astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    return float((2.0 * intersection + eps) / (y_true.sum() + y_pred.sum() + eps))


def best_dice_threshold(
    y_true: np.ndarray, y_score: np.ndarray, *, thresholds: np.ndarray | None = None
) -> tuple[float, float]:
    """Threshold that maximises Dice, swept on a fixed grid (not every unique
    score value, unlike drdetect.eval.metrics.choose_threshold_for_sensitivity
    -- pixel-level score arrays here run into the hundreds of millions of
    values, and a per-unique-value sweep at that scale does not finish).

    **Call this on the VALIDATION set only** -- e.g. the internal fold's held-
    out images, never the 27 official IDRiD test images. Freeze the returned
    threshold and apply it to the test set with dice_coefficient(), the same
    selection/evaluation separation drdetect.eval.metrics already documents
    for the grading model's operating point.

    Fixed 0.5 is not a safe default here: an undertrained model's sigmoid
    outputs can sit almost entirely below 0.5 even where its pixel ranking is
    good, so Dice@0.5 can read close to zero while AUPRC (threshold-free)
    reads far higher -- measured directly on a 2-epoch checkpoint (AUPRC
    0.409, Dice@0.5 0.037) versus the fully-trained one (AUPRC 0.830, Dice@0.5
    0.703) on the identical 27 test images.

    Returns:
        (best_threshold, best_dice) from the grid.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.025, 0.975, 39)

    best_threshold, best_dice = 0.5, -1.0
    for t in thresholds:
        dice = dice_coefficient(y_true, y_score > t)
        if dice > best_dice:
            best_threshold, best_dice = float(t), dice
    return best_threshold, best_dice
