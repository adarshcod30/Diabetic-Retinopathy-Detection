"""Evaluation metrics for DR grading.

Two things here are easy to get wrong and are therefore made explicit:

1. **The referable-DR threshold must be chosen on validation and then frozen.**
   Picking it on the test set is the most common silent cheat in this
   literature -- it converts "we achieved 90% sensitivity" into "there exists a
   threshold at which we would have". `choose_threshold_for_sensitivity` and
   `evaluate_at_threshold` are separated so the two steps cannot be conflated.

2. **A point estimate without an interval is not a result.** On a few hundred
   test images, a reported "91.3% sensitivity" can easily be consistent with
   82%. `bootstrap_ci` exists so no headline number ships bare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "REFERABLE_THRESHOLD",
    "BinaryScores",
    "quadratic_weighted_kappa",
    "referable_labels",
    "binary_scores",
    "choose_threshold_for_sensitivity",
    "sensitivity_at_specificity_floor",
    "youden_j",
    "evaluate_at_threshold",
    "bootstrap_ci",
    "expected_calibration_error",
]

# ICDR grade >= 2 (Moderate NPDR) is the clinical referral boundary.
REFERABLE_THRESHOLD = 2


@dataclass(frozen=True)
class BinaryScores:
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n_positive(self) -> int:
        return self.tp + self.fn

    @property
    def n_negative(self) -> int:
        return self.tn + self.fp


def quadratic_weighted_kappa(y_true, y_pred, *, n_classes: int = 5) -> float:
    """Cohen's kappa with quadratic weights -- the standard DR grading metric.

    Quadratic weighting is what makes this the right metric for an *ordinal*
    task: confusing grade 0 with grade 4 is penalised 16x more than confusing 0
    with 1, matching the clinical cost structure. Plain accuracy treats both as
    a single error.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    obs = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred, strict=True):
        obs[t, p] += 1

    hist_true = obs.sum(axis=1)
    hist_pred = obs.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(obs.sum(), 1)

    i, j = np.meshgrid(np.arange(n_classes), np.arange(n_classes), indexing="ij")
    weights = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    denom = (weights * expected).sum()
    if denom == 0:
        return 0.0
    return float(1.0 - (weights * obs).sum() / denom)


def referable_labels(grades) -> np.ndarray:
    """Binarise ICDR grades into referable (>=2) vs not."""
    return (np.asarray(grades) >= REFERABLE_THRESHOLD).astype(int)


def binary_scores(y_true_binary, y_pred_binary) -> BinaryScores:
    y_true = np.asarray(y_true_binary, dtype=int)
    y_pred = np.asarray(y_pred_binary, dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def safe(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    return BinaryScores(
        sensitivity=safe(tp, tp + fn),
        specificity=safe(tn, tn + fp),
        ppv=safe(tp, tp + fp),
        npv=safe(tn, tn + fn),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def choose_threshold_for_sensitivity(
    y_true_binary, y_score, *, target_sensitivity: float = 0.90
) -> float:
    """Lowest-specificity-cost threshold meeting a sensitivity floor.

    **Call this on the VALIDATION set only.** The returned threshold is then
    frozen and passed to `evaluate_at_threshold` on the test set.

    Screening is deliberately asymmetric: a missed referable case may cost
    sight, a false positive costs one ophthalmologist review. So we fix
    sensitivity and accept whatever specificity follows, rather than optimising
    a symmetric metric like accuracy or Youden's J.
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    best_threshold, best_specificity = 0.0, -1.0
    for threshold in np.unique(np.concatenate([y_score, [0.0, 1.0]])):
        scores = binary_scores(y_true, (y_score >= threshold).astype(int))
        if scores.sensitivity >= target_sensitivity and scores.specificity > best_specificity:
            best_threshold, best_specificity = float(threshold), scores.specificity
    return best_threshold


def sensitivity_at_specificity_floor(y_true_binary, y_score, *, spec_floor: float = 0.85) -> float:
    """Best sensitivity achievable while keeping specificity >= spec_floor.

    Why this exists
    ----------------
    Plain sensitivity, monitored alone with mode="max", is gameable by a
    single degenerate trick: predict referable for everyone. Checked directly
    against this project's own training logs, `val/sensitivity_referable`
    peaks at the exact epoch every other analysis in Phase 3 identified as the
    grade-2 collapse (3 of 4 CE runs checked; the fourth misses by one epoch).
    At that epoch sensitivity hits ~1.00 because the model refers almost every
    image, not because it discriminates well -- specificity there is ~0.80,
    barely above chance for the majority class. See
    docs/07_PHASE3_RESULTS.md for the measurement.

    Result 4 already showed a *different* naive fix (macro-recall selection)
    fails for the same underlying reason: a single-sided metric cannot
    distinguish genuine improvement from a degenerate shortcut.

    This function instead operationalises the project's own stated target
    (docs/01_PROJECT_ANALYSIS.md: sensitivity >90%, specificity >85%) directly:
    among all thresholds achieving specificity >= spec_floor on this data,
    return the highest sensitivity. If no threshold clears the floor, return
    `best_specificity - 1.0` -- a value guaranteed to be negative, so any
    epoch that DOES clear the floor always outranks one that does not, and
    epochs that all fail the floor are still ordered by how close they came.

    **Selection use only.** As with every other `val/*` metric monitored
    during training, this sweeps thresholds on the validation split purely to
    choose which epoch's weights to keep -- the same role val/qwk already
    plays. It is not the reported clinical operating point; that threshold is
    chosen once, on validation, and frozen before scoring the locked test set
    in `scripts/evaluate.py`, per the module-level docstring above.
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    best_sensitivity = -1.0
    best_specificity_overall = -1.0
    for threshold in np.unique(np.concatenate([y_score, [0.0, 1.0]])):
        scores = binary_scores(y_true, (y_score >= threshold).astype(int))
        if scores.specificity > best_specificity_overall:
            best_specificity_overall = scores.specificity
        if scores.specificity >= spec_floor and scores.sensitivity > best_sensitivity:
            best_sensitivity = scores.sensitivity

    if best_sensitivity >= 0.0:
        return float(best_sensitivity)
    return float(best_specificity_overall - 1.0)


def youden_j(y_true_binary, y_score, threshold: float = 0.5) -> float:
    """Youden's J = sensitivity + specificity - 1, at a fixed threshold.

    Logged alongside sensitivity_at_specificity_floor purely for comparison in
    the metrics CSV -- it is the standard symmetric alternative, included so a
    later reader can see both without re-deriving them. Not used for selection
    here: unlike the floor-constrained metric, J does not target this
    project's stated specificity requirement and can settle on a point that
    satisfies neither the sensitivity nor the specificity bar.
    """
    scores = binary_scores(y_true_binary, (np.asarray(y_score) >= threshold).astype(int))
    return float(scores.sensitivity + scores.specificity - 1.0)


def evaluate_at_threshold(y_true_binary, y_score, threshold: float) -> BinaryScores:
    """Apply a threshold frozen from validation. No tuning happens here."""
    return binary_scores(y_true_binary, (np.asarray(y_score) >= threshold).astype(int))


def bootstrap_ci(
    metric_fn,
    *arrays,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any metric over paired arrays.

    Returns (point_estimate, lower, upper). Resamples cases (not predictions),
    which is the correct unit -- the uncertainty being quantified is "which
    patients happened to be in this test set".
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same length")

    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays))

    stats = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric_fn(*[a[idx] for a in arrays])

    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return point, float("nan"), float("nan")

    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def expected_calibration_error(y_true_binary, y_prob, *, n_bins: int = 15) -> float:
    """ECE: mean |confidence - accuracy| over equal-width probability bins.

    Modern networks are systematically overconfident (Guo et al., 2017), so raw
    softmax output is not a probability. This is the number that must fall after
    temperature scaling in Phase 5.
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)
