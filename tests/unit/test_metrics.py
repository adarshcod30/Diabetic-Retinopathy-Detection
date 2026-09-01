"""Tests for evaluation metrics.

QWK is validated against sklearn's independent implementation rather than
against hand-computed values -- if both agree, the chance of a shared bug is
negligible, and the ordinal weighting is the part most easily got wrong.
"""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from drdetect.eval.metrics import (
    binary_scores,
    bootstrap_ci,
    choose_threshold_for_sensitivity,
    evaluate_at_threshold,
    expected_calibration_error,
    quadratic_weighted_kappa,
    referable_labels,
)


class TestQWK:
    def test_perfect_agreement_is_one(self):
        y = [0, 1, 2, 3, 4, 0, 2, 4]
        assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_sklearn(self, seed):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 5, size=300)
        y_pred = np.clip(y_true + rng.integers(-1, 2, size=300), 0, 4)
        assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(
            cohen_kappa_score(y_true, y_pred, weights="quadratic"), abs=1e-9
        )

    def test_penalises_distant_errors_more(self):
        """The whole reason QWK is the DR metric: 0->4 must hurt far more than 0->1."""
        y_true = [0] * 50 + [4] * 50
        near = [0] * 50 + [3] * 50
        far = [4] * 50 + [0] * 50
        assert quadratic_weighted_kappa(y_true, near) > quadratic_weighted_kappa(y_true, far)


class TestReferable:
    def test_threshold_is_grade_2(self):
        np.testing.assert_array_equal(referable_labels([0, 1, 2, 3, 4]), [0, 0, 1, 1, 1])

    def test_binary_scores_arithmetic(self):
        #        TP  FN  TN  FP
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        s = binary_scores(y_true, y_pred)
        assert (s.tp, s.fn, s.tn, s.fp) == (2, 1, 2, 1)
        assert s.sensitivity == pytest.approx(2 / 3)
        assert s.specificity == pytest.approx(2 / 3)
        assert s.n_positive == 3 and s.n_negative == 3

    def test_empty_class_gives_nan_not_crash(self):
        s = binary_scores([0, 0, 0], [0, 0, 0])
        assert np.isnan(s.sensitivity)  # no positives exist
        assert s.specificity == 1.0


class TestThresholdSelection:
    def test_meets_sensitivity_target(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=500)
        score = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, size=500), 0, 1)
        thr = choose_threshold_for_sensitivity(y_true, score, target_sensitivity=0.90)
        assert evaluate_at_threshold(y_true, score, thr).sensitivity >= 0.90

    def test_higher_target_never_increases_threshold(self):
        """Demanding more sensitivity can only lower (or hold) the cut-point."""
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=400)
        score = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, size=400), 0, 1)
        t90 = choose_threshold_for_sensitivity(y_true, score, target_sensitivity=0.90)
        t99 = choose_threshold_for_sensitivity(y_true, score, target_sensitivity=0.99)
        assert t99 <= t90

    def test_selection_and_evaluation_are_separable(self):
        """Threshold chosen on one split must be applicable to another untouched
        split -- the API must not force test-set tuning."""
        rng = np.random.default_rng(2)
        yv, yt = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
        sv = np.clip(yv * 0.5 + rng.normal(0.25, 0.2, 300), 0, 1)
        st = np.clip(yt * 0.5 + rng.normal(0.25, 0.2, 300), 0, 1)
        thr = choose_threshold_for_sensitivity(yv, sv)
        assert 0.0 <= evaluate_at_threshold(yt, st, thr).sensitivity <= 1.0


class TestBootstrap:
    def test_interval_brackets_point_estimate(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 5, 400)
        y_pred = np.clip(y_true + rng.integers(-1, 2, 400), 0, 4)
        point, lo, hi = bootstrap_ci(quadratic_weighted_kappa, y_true, y_pred, n_resamples=300)
        assert lo <= point <= hi

    def test_smaller_n_gives_wider_interval(self):
        """The core argument for never subsampling the test set."""
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 5, 1000)
        y_pred = np.clip(y_true + rng.integers(-1, 2, 1000), 0, 4)
        _, lo_big, hi_big = bootstrap_ci(quadratic_weighted_kappa, y_true, y_pred, n_resamples=300)
        _, lo_sm, hi_sm = bootstrap_ci(
            quadratic_weighted_kappa, y_true[:100], y_pred[:100], n_resamples=300
        )
        assert (hi_sm - lo_sm) > (hi_big - lo_big)

    def test_is_reproducible(self):
        rng = np.random.default_rng(5)
        y_true, y_pred = rng.integers(0, 5, 200), rng.integers(0, 5, 200)
        a = bootstrap_ci(quadratic_weighted_kappa, y_true, y_pred, n_resamples=200, seed=7)
        b = bootstrap_ci(quadratic_weighted_kappa, y_true, y_pred, n_resamples=200, seed=7)
        assert a == b

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            bootstrap_ci(quadratic_weighted_kappa, np.zeros(10), np.zeros(5))


class TestCalibration:
    def test_perfectly_calibrated_is_near_zero(self):
        rng = np.random.default_rng(6)
        probs = rng.uniform(0, 1, 20000)
        labels = (rng.uniform(0, 1, 20000) < probs).astype(int)
        assert expected_calibration_error(labels, probs) < 0.02

    def test_overconfident_model_scores_high(self):
        """A model always claiming 0.99 while being right half the time."""
        labels = np.array([1, 0] * 500)
        probs = np.full(1000, 0.99)
        assert expected_calibration_error(labels, probs) > 0.4
