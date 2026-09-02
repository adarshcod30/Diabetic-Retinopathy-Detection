"""Tests for ordinal losses.

The property that matters is that these losses know grades are ORDERED --
predicting 4 for a true 1 must cost more than predicting 2. Plain cross-entropy
does not, which is why the Phase 1 baseline collapsed its errors onto grade 2.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drdetect.grading.losses import (  # noqa: E402
    CornLoss,
    DistanceWeightedCE,
    OrdinalRegressionLoss,
    build_loss,
    corn_class_probabilities,
    corn_predict,
    corn_probabilities,
    fit_thresholds,
    outputs_for_loss,
    regression_predict,
)


class TestOrdinality:
    """The defining property, tested for every ordinal loss."""

    def test_corn_penalises_distant_errors_more(self):
        loss = CornLoss(5)
        target = torch.tensor([1])
        # confident "grade 1": exceeds threshold 0, not 1,2,3
        near = torch.tensor([[3.0, -3.0, -3.0, -3.0]])
        # confident "grade 4": exceeds every threshold
        far = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
        assert loss(near, target).item() < loss(far, target).item()

    def test_distance_ce_penalises_distant_errors_more(self):
        loss = DistanceWeightedCE(5)
        target = torch.tensor([1])
        near = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]])  # says grade 2
        far = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]])  # says grade 4
        assert loss(near, target).item() < loss(far, target).item()

    def test_regression_penalises_distant_errors_more(self):
        loss = OrdinalRegressionLoss()
        target = torch.tensor([1])
        assert (
            loss(torch.tensor([[2.0]]), target).item() < loss(torch.tensor([[4.0]]), target).item()
        )

    def test_plain_ce_does_not(self):
        """Documents the deficiency being fixed: CE scores both errors alike."""
        ce = torch.nn.CrossEntropyLoss()
        target = torch.tensor([1])
        near = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]])
        far = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]])
        assert ce(near, target).item() == pytest.approx(ce(far, target).item(), abs=1e-6)


class TestCorn:
    def test_predictions_are_rank_monotonic(self):
        """The structural guarantee: P(y>j) can never exceed P(y>j-1)."""
        rng = torch.Generator().manual_seed(0)
        logits = torch.randn(400, 4, generator=rng) * 3
        cum = corn_probabilities(logits)
        assert torch.all(cum[:, :-1] >= cum[:, 1:] - 1e-6), "cumulative probabilities not monotone"

    def test_predictions_span_the_valid_range(self):
        logits = torch.randn(500, 4, generator=torch.Generator().manual_seed(1)) * 4
        preds = corn_predict(logits)
        assert preds.min() >= 0 and preds.max() <= 4

    def test_all_thresholds_exceeded_gives_top_grade(self):
        assert corn_predict(torch.tensor([[9.0, 9.0, 9.0, 9.0]])).item() == 4

    def test_no_thresholds_exceeded_gives_grade_zero(self):
        assert corn_predict(torch.tensor([[-9.0, -9.0, -9.0, -9.0]])).item() == 0

    def test_class_probabilities_sum_to_one(self):
        logits = torch.randn(100, 4, generator=torch.Generator().manual_seed(2)) * 2
        probs = corn_class_probabilities(logits)
        assert probs.shape == (100, 5)
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(100), atol=1e-5, rtol=1e-5)
        assert (probs >= 0).all()

    def test_rejects_wrong_logit_count(self):
        """5 logits is the CE shape -- a silent mismatch would train nonsense."""
        with pytest.raises(ValueError, match="expects 4 logits"):
            CornLoss(5)(torch.randn(2, 5), torch.tensor([0, 1]))

    def test_loss_decreases_as_prediction_improves(self):
        loss = CornLoss(5)
        targets = torch.tensor([0, 1, 2, 3, 4])
        good = torch.tensor(
            [
                [-5.0, -5, -5, -5],
                [5.0, -5, -5, -5],
                [5.0, 5, -5, -5],
                [5.0, 5, 5, -5],
                [5.0, 5, 5, 5],
            ]
        )
        assert loss(good, targets).item() < loss(-good, targets).item()

    def test_handles_single_class_batch(self):
        """At batch 4 a batch of identical labels is common; must not divide by zero."""
        out = CornLoss(5)(torch.randn(4, 4), torch.tensor([0, 0, 0, 0]))
        assert torch.isfinite(out)


class TestRegressionThresholds:
    def test_default_thresholds_round_to_nearest(self):
        out = torch.tensor([[0.2], [0.9], [1.6], [2.7], [3.9]])
        np.testing.assert_array_equal(regression_predict(out).numpy(), [0, 1, 2, 3, 4])

    def test_fitting_improves_or_matches_qwk(self):
        from drdetect.eval.metrics import quadratic_weighted_kappa

        rng = np.random.default_rng(0)
        targets = rng.integers(0, 5, 500)
        preds = targets + rng.normal(0.35, 0.5, 500)  # biased, so defaults are suboptimal

        def qwk_at(th):
            d = np.zeros_like(preds, dtype=int)
            for t in th:
                d += (preds > t).astype(int)
            return quadratic_weighted_kappa(targets, d)

        assert qwk_at(fit_thresholds(preds, targets)) >= qwk_at([0.5, 1.5, 2.5, 3.5])

    def test_fitted_thresholds_stay_ordered(self):
        rng = np.random.default_rng(1)
        targets = rng.integers(0, 5, 300)
        th = fit_thresholds(targets + rng.normal(0, 0.6, 300), targets)
        assert all(a < b for a, b in zip(th[:-1], th[1:], strict=True)), th


class TestFactory:
    @pytest.mark.parametrize(
        "name,expected", [("ce", 5), ("distance_ce", 5), ("corn", 4), ("regression", 1)]
    )
    def test_output_count_matches_loss(self, name, expected):
        assert outputs_for_loss(name) == expected

    @pytest.mark.parametrize("name", ["ce", "corn", "regression", "distance_ce"])
    def test_builds_and_runs(self, name):
        loss = build_loss(name)
        n = outputs_for_loss(name)
        out = loss(torch.randn(8, n), torch.randint(0, 5, (8,)))
        assert torch.isfinite(out) and out.ndim == 0

    def test_rejects_unknown_loss(self):
        with pytest.raises(ValueError, match="unknown loss"):
            build_loss("focal")

    @pytest.mark.parametrize("name", ["ce", "corn", "distance_ce"])
    def test_accepts_class_weights(self, name):
        loss = build_loss(name, class_weights=[1.0, 3.0, 1.0, 5.0, 4.0])
        out = loss(torch.randn(8, outputs_for_loss(name)), torch.randint(0, 5, (8,)))
        assert torch.isfinite(out)

    @pytest.mark.parametrize("name", ["ce", "corn", "regression", "distance_ce"])
    def test_gradients_flow(self, name):
        n = outputs_for_loss(name)
        logits = torch.randn(8, n, requires_grad=True)
        build_loss(name)(logits, torch.randint(0, 5, (8,))).backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


class TestCornTaskBalancing:
    """CORN's conditional subsets inherit the label skew.

    Measured on APTOS: task j=1 ("given grade >= 1, is it worse than 1?") is
    80.1% positive, because grade 1 is only 296 of 1485 samples with grade >= 1.
    Unweighted, that pushed 49 of 74 grade-1 validation cases into grade 2 and
    dropped grade-1 recall from 0.541 to 0.297.
    """

    def _aptos_like(self):
        from collections import Counter  # noqa: F401

        counts = {0: 1444, 1: 296, 2: 799, 3: 154, 4: 236}  # APTOS train split
        return np.repeat(list(counts), list(counts.values()))

    def test_detects_the_skewed_task(self):
        from drdetect.grading.losses import corn_task_pos_weights

        w = corn_task_pos_weights(self._aptos_like())
        assert len(w) == 4
        # task 1 is majority-positive, so its pos_weight must be well below 1
        assert w[1] < 0.4, f"task 1 weight {w[1]} does not counteract an 80% positive rate"
        # task 2 is minority-positive, so its weight must exceed 1
        assert w[2] > 1.0, f"task 2 weight {w[2]} does not counteract a 33% positive rate"

    def test_balanced_labels_give_unit_weights(self):
        from drdetect.grading.losses import corn_task_pos_weights

        # a distribution where each task is 50/50
        y = np.array([0] * 8 + [1] * 4 + [2] * 2 + [3] * 1 + [4] * 1)
        w = corn_task_pos_weights(y)
        assert all(0.4 < x < 2.6 for x in w), w

    def test_weights_change_the_loss(self):
        from drdetect.grading.losses import build_loss

        logits = torch.randn(32, 4, generator=torch.Generator().manual_seed(0))
        targets = torch.randint(0, 5, (32,), generator=torch.Generator().manual_seed(1))
        plain = build_loss("corn")(logits, targets)
        weighted = build_loss("corn", task_pos_weights=[1.0, 0.25, 2.0, 0.65])(logits, targets)
        assert not torch.isclose(plain, weighted), "task weights had no effect"

    def test_weighted_loss_still_differentiable(self):
        from drdetect.grading.losses import build_loss

        logits = torch.randn(16, 4, requires_grad=True)
        build_loss("corn", task_pos_weights=[1.0, 0.25, 2.0, 0.65])(
            logits, torch.randint(0, 5, (16,))
        ).backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_handles_absent_class(self):
        from drdetect.grading.losses import corn_task_pos_weights

        w = corn_task_pos_weights(np.array([0, 0, 0, 1, 1]))  # no grades 2-4
        assert all(np.isfinite(w))
