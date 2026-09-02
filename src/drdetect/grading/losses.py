"""Ordinal losses for DR grading.

Why cross-entropy is the wrong loss here
----------------------------------------
DR grades are ordered: 0 < 1 < 2 < 3 < 4. Cross-entropy treats them as
unordered labels, so predicting 4 for a true grade 1 costs exactly as much as
predicting 2. The loss-minimising strategy under that objective is to hedge
toward the majority middle class.

The Phase 1 baseline did precisely that. From its confusion matrix, errors
collapsed onto grade 2: 29 of 74 grade-1 cases, 21 of 39 grade-3, and 22 of 59
grade-4 were all predicted Moderate. See docs/06_PHASE1_RESULTS.md.

Three alternatives are provided.

`CornLoss` -- rank-consistent ordinal regression (Shi, Cao & Raschka, 2023).
    The model emits K-1 logits; logit j models P(y > j | y > j-1). Because the
    probabilities chain multiplicatively, predictions are rank-monotonic *by
    construction*: the model cannot output "probably worse than grade 2 but not
    worse than grade 1". Each task trains only on the conditional subset, which
    is what distinguishes CORN from the earlier CORAL and removes its shared-bias
    restriction.

`OrdinalRegressionLoss` -- single continuous output, thresholds fitted after
    training. This is the approach behind most winning Kaggle DR solutions. It
    optimises QWK almost directly, at the cost of a post-hoc threshold search.

`DistanceWeightedCE` -- ordinary cross-entropy with soft targets whose mass
    decays with ordinal distance. The least invasive option: it keeps the K-way
    head and only changes the targets.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CornLoss",
    "decode_output",
    "naive_referable_cut",
    "OrdinalRegressionLoss",
    "DistanceWeightedCE",
    "corn_probabilities",
    "corn_predict",
    "corn_task_pos_weights",
    "regression_predict",
    "fit_thresholds",
    "build_loss",
    "outputs_for_loss",
]


class CornLoss(nn.Module):
    """Conditional ordinal regression, rank-consistent by construction.

    Expects `num_classes - 1` logits. Task j is trained only on samples with
    y >= j, so each binary problem is genuinely conditional.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: torch.Tensor | None = None,
        task_pos_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )
        self.register_buffer(
            "task_pos_weights",
            task_pos_weights if task_pos_weights is not None else torch.ones(num_classes - 1),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != self.num_classes - 1:
            raise ValueError(
                f"CornLoss expects {self.num_classes - 1} logits, got {logits.shape[1]}. "
                f"Build the model with num_outputs={self.num_classes - 1}."
            )

        losses, total = [], 0.0
        for j in range(self.num_classes - 1):
            subset = targets >= j  # conditional on having exceeded j-1
            if not subset.any():
                continue
            binary = (targets[subset] > j).float()
            w = self.class_weights[targets[subset]]
            bce = F.binary_cross_entropy_with_logits(
                logits[subset, j],
                binary,
                weight=w,
                pos_weight=self.task_pos_weights[j],
                reduction="sum",
            )
            losses.append(bce)
            total += float(w.sum())

        if not losses:
            return logits.sum() * 0.0  # keeps the graph connected
        return torch.stack(losses).sum() / max(total, 1.0)


def corn_task_pos_weights(labels, num_classes: int = 5) -> list[float]:
    """Per-task positive weights that rebalance CORN's conditional subsets.

    CORN trains task j only on samples with y >= j, asking whether y > j. Those
    subsets inherit the label distribution and are usually NOT balanced. Measured
    on the APTOS training split:

        task j=0   2929 samples,  50.7% positive   balanced
        task j=1   1485 samples,  80.1% positive   badly skewed
        task j=2   1189 samples,  32.8% positive   skewed
        task j=3    390 samples,  60.5% positive   balanced

    Task j=1 asks "given grade >= 1, is it worse than 1?" and is 80% positive,
    because grade 1 is only 296 of 1485 samples with grade >= 1. Unweighted, the
    loss-minimising answer is almost always yes, which pushes grade-1 cases into
    grade 2. Measured consequence: unweighted CORN sent 49 of 74 grade-1
    validation cases to grade 2, against 29 for plain cross-entropy, and grade-1
    recall fell from 0.541 to 0.297.

    Returns n_negative / n_positive per task, the standard BCE `pos_weight`,
    which makes each conditional task cost-balanced.
    """
    y = np.asarray(labels)
    weights = []
    for j in range(num_classes - 1):
        subset = y[y >= j]
        pos = int((subset > j).sum())
        neg = int(len(subset) - pos)
        weights.append(float(neg / pos) if pos > 0 else 1.0)
    return weights


def corn_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Cumulative probabilities P(y > j) from CORN logits, via the chain rule.

    The cumulative product is what guarantees monotonicity: P(y>j) can never
    exceed P(y>j-1), because it is that value multiplied by a sigmoid.
    """
    return torch.cumprod(torch.sigmoid(logits), dim=1)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predicted grade = number of thresholds exceeded."""
    return (corn_probabilities(logits) > 0.5).sum(dim=1)


def corn_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Per-class probabilities from cumulative ones, for referable scoring.

    P(y=k) = P(y>k-1) - P(y>k), with P(y>-1)=1 and P(y>K-1)=0.
    """
    cum = corn_probabilities(logits)
    ones = torch.ones_like(cum[:, :1])
    zeros = torch.zeros_like(cum[:, :1])
    upper = torch.cat([ones, cum], dim=1)
    lower = torch.cat([cum, zeros], dim=1)
    return (upper - lower).clamp(min=0.0)


class OrdinalRegressionLoss(nn.Module):
    """Single-output regression on the grade, with SmoothL1.

    SmoothL1 rather than MSE: MSE lets a handful of grade-0-predicted-as-4
    outliers dominate the gradient, which at batch 4 is destabilising.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(output.squeeze(-1), targets.float(), beta=self.beta)


def regression_predict(output: torch.Tensor, thresholds: list[float] | None = None) -> torch.Tensor:
    """Discretise a continuous prediction using cut-points."""
    if thresholds is None:
        thresholds = [0.5, 1.5, 2.5, 3.5]
    x = output.squeeze(-1)
    pred = torch.zeros_like(x, dtype=torch.long)
    for t in thresholds:
        pred += (x > t).long()
    return pred


def fit_thresholds(predictions, targets, *, n_classes: int = 5, n_rounds: int = 8) -> list[float]:
    """Coordinate-ascent search for cut-points maximising QWK.

    **Fit on validation, never on test.** The returned thresholds are part of the
    model and must be frozen alongside the weights.
    """
    from drdetect.eval.metrics import quadratic_weighted_kappa

    preds = np.asarray(predictions, dtype=float).ravel()
    targs = np.asarray(targets, dtype=int).ravel()
    thresholds = [i + 0.5 for i in range(n_classes - 1)]

    def score(th: list[float]) -> float:
        digitised = np.zeros_like(preds, dtype=int)
        for t in th:
            digitised += (preds > t).astype(int)
        return quadratic_weighted_kappa(targs, digitised, n_classes=n_classes)

    best = score(thresholds)
    for _ in range(n_rounds):
        improved = False
        for i in range(len(thresholds)):
            lo = thresholds[i - 1] + 0.05 if i > 0 else preds.min() - 0.5
            hi = thresholds[i + 1] - 0.05 if i < len(thresholds) - 1 else preds.max() + 0.5
            if hi <= lo:
                continue
            for candidate in np.linspace(lo, hi, 40):
                trial = list(thresholds)
                trial[i] = float(candidate)
                s = score(trial)
                if s > best:
                    best, thresholds, improved = s, trial, True
        if not improved:
            break
    return thresholds


class DistanceWeightedCE(nn.Module):
    """Cross-entropy against soft targets that decay with ordinal distance.

    Target mass for class k given true grade y is proportional to
    exp(-|k - y| / temperature), so predicting an adjacent grade is penalised
    far less than predicting a distant one -- the property plain CE lacks.
    """

    def __init__(
        self,
        num_classes: int = 5,
        temperature: float = 0.7,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        grid = torch.arange(num_classes).float()
        soft = torch.exp(-(grid[None, :] - grid[:, None]).abs() / temperature)
        self.register_buffer("soft_targets", soft / soft.sum(dim=1, keepdim=True))
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft = self.soft_targets[targets]
        per_sample = -(soft * F.log_softmax(logits, dim=1)).sum(dim=1)
        w = self.class_weights[targets]
        return (per_sample * w).sum() / w.sum().clamp(min=1e-8)


def outputs_for_loss(loss_name: str, num_classes: int = 5) -> int:
    """How many head outputs a given loss requires."""
    return {
        "ce": num_classes,
        "distance_ce": num_classes,
        "corn": num_classes - 1,
        "regression": 1,
    }[loss_name]


def build_loss(
    name: str,
    *,
    num_classes: int = 5,
    class_weights: list[float] | None = None,
    task_pos_weights: list[float] | None = None,
):
    w = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
    if name == "ce":
        return nn.CrossEntropyLoss(weight=w)
    if name == "corn":
        tw = (
            torch.tensor(task_pos_weights, dtype=torch.float)
            if task_pos_weights is not None
            else None
        )
        return CornLoss(num_classes, class_weights=w, task_pos_weights=tw)
    if name == "regression":
        return OrdinalRegressionLoss()
    if name == "distance_ce":
        return DistanceWeightedCE(num_classes, class_weights=w)
    raise ValueError(f"unknown loss {name!r}; expected ce, corn, regression or distance_ce")


def decode_output(output: torch.Tensor, loss_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Map a raw head output to (predicted grade, referable score).

    Single source of truth for decoding, shared by the training module and the
    evaluation script. Duplicating this logic once caused an evaluation to load a
    5-output head against a 4-output CORN checkpoint; keeping one implementation
    makes that class of mismatch impossible.

    Each loss parameterises the head differently:
      ce / distance_ce -- K-way softmax; referable = sum of P(class >= 2)
      corn             -- K-1 chained sigmoids; referable = P(y > 1) directly
      regression       -- one continuous value, which itself ranks severity

    The referable score need only be monotone in severity; threshold selection
    handles the scale.
    """
    if loss_name == "corn":
        cum = corn_probabilities(output)  # P(y > j), monotone by construction
        return (cum > 0.5).sum(dim=1).numpy(), cum[:, 1].numpy()

    if loss_name == "regression":
        return regression_predict(output).numpy(), output.squeeze(-1).numpy()

    probs = torch.softmax(output, dim=1).numpy()
    return probs.argmax(axis=1), probs[:, 2:].sum(axis=1)


def naive_referable_cut(loss_name: str) -> float:
    """The 'no calibration' cut point, whose scale depends on the head.

    Only a progress signal during training -- the reported operating point is
    chosen for target sensitivity in scripts/evaluate.py.
    """
    return 1.5 if loss_name == "regression" else 0.5
