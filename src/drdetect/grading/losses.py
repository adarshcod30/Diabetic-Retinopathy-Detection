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
    "OrdinalRegressionLoss",
    "DistanceWeightedCE",
    "corn_probabilities",
    "corn_predict",
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

    def __init__(self, num_classes: int = 5, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
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
                logits[subset, j], binary, weight=w, reduction="sum"
            )
            losses.append(bce)
            total += float(w.sum())

        if not losses:
            return logits.sum() * 0.0  # keeps the graph connected
        return torch.stack(losses).sum() / max(total, 1.0)


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


def build_loss(name: str, *, num_classes: int = 5, class_weights: list[float] | None = None):
    w = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
    if name == "ce":
        return nn.CrossEntropyLoss(weight=w)
    if name == "corn":
        return CornLoss(num_classes, class_weights=w)
    if name == "regression":
        return OrdinalRegressionLoss()
    if name == "distance_ce":
        return DistanceWeightedCE(num_classes, class_weights=w)
    raise ValueError(f"unknown loss {name!r}; expected ce, corn, regression or distance_ce")
