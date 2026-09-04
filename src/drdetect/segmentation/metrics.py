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

__all__ = ["pixel_auprc", "dice_coefficient"]


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
