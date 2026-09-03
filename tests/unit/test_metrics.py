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
    sensitivity_at_specificity_floor,
    youden_j,
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


class TestSensitivityAtSpecificityFloor:
    """Regression tests for a real failure: plain sensitivity, monitored alone,
    peaks at exactly the grade-2 collapse epoch in 3 of 4 checked training runs
    (baseline, 768px, warmup-8 control), because predicting referable for
    almost everyone drives sensitivity to ~1.0 at the cost of specificity.
    """

    def test_degenerate_predict_everyone_positive_is_rejected(self):
        """The exact failure mode: sens=1.0 via zero discrimination."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 300)
        score = np.ones(300)  # predicts positive unconditionally -> sens=1, spec=0
        result = sensitivity_at_specificity_floor(y_true, score, spec_floor=0.85)
        assert result < 0, "a degenerate always-positive model must not score >= 0"

    def test_genuine_discriminator_clears_the_floor(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 500)
        score = np.clip(y_true * 0.6 + rng.normal(0.2, 0.15, 500), 0, 1)
        result = sensitivity_at_specificity_floor(y_true, score, spec_floor=0.85)
        assert result >= 0.0
        # cross-check: applying the returned sensitivity's threshold directly
        thr = choose_threshold_for_sensitivity(y_true, score, target_sensitivity=result - 1e-9)
        applied = evaluate_at_threshold(y_true, score, thr)
        assert applied.specificity >= 0.85 - 1e-6

    def test_failing_every_threshold_still_orders_by_closeness(self):
        """When nothing clears the floor, a closer miss must still rank higher.

        A continuous score almost always has SOME extreme threshold reaching a
        high specificity floor at near-zero sensitivity, so a merely-high floor
        like 0.99 is the wrong fixture for "genuinely unreachable" -- it is
        usually reachable. floor=1.0 with a negative tied at the maximum
        observed score is genuinely unreachable instead: since candidate
        thresholds never exceed 1.0, that negative can never be excluded, so
        specificity can never reach exactly 1.0, regardless of separation
        elsewhere.
        """
        positives = np.ones(10)
        # 9 of 10 negatives are easily excluded; one is tied at the max and
        # can never be excluded by any threshold <= 1.0.
        less_bad_negatives = np.array([0.0] * 9 + [1.0])
        # none of the 10 negatives can ever be excluded.
        worse_negatives = np.ones(10)

        y_true = np.array([1] * 10 + [0] * 10)
        less_bad = sensitivity_at_specificity_floor(
            y_true, np.concatenate([positives, less_bad_negatives]), spec_floor=1.0
        )
        worse = sensitivity_at_specificity_floor(
            y_true, np.concatenate([positives, worse_negatives]), spec_floor=1.0
        )
        assert less_bad < 0 and worse < 0, "floor of 1.0 must be unreachable in both constructions"
        assert less_bad > worse, "9-of-10 excludable must rank above 0-of-10 excludable"

    def test_reproduces_the_measured_collapse_epoch_rejection(self):
        """Concretely: replay the shape of the baseline's epoch-3 collapse
        (sens=1.000, spec=0.800) against a healthier later epoch, and confirm
        the constrained metric prefers the healthier one while plain
        sensitivity would have preferred the collapse."""
        rng = np.random.default_rng(3)
        y_true = np.array([0] * 361 + [1] * 372)

        # collapse epoch: refers almost everyone -> sens ~1.0, spec ~0.80
        collapse_score = np.concatenate([rng.uniform(0.4, 0.9, 361), rng.uniform(0.5, 1.0, 372)])
        # healthy epoch: genuine separation, sens ~0.90, spec ~0.90
        healthy_score = np.concatenate(
            [rng.normal(0.3, 0.15, 361), rng.normal(0.75, 0.15, 372)]
        ).clip(0, 1)

        plain_collapse = binary_scores(y_true, (collapse_score >= 0.5).astype(int))
        plain_healthy = binary_scores(y_true, (healthy_score >= 0.5).astype(int))
        assert plain_collapse.sensitivity > plain_healthy.sensitivity, (
            "fixture must reproduce the actual failure: plain sensitivity prefers collapse"
        )

        constrained_collapse = sensitivity_at_specificity_floor(
            y_true, collapse_score, spec_floor=0.85
        )
        constrained_healthy = sensitivity_at_specificity_floor(
            y_true, healthy_score, spec_floor=0.85
        )
        assert constrained_healthy > constrained_collapse, (
            "the constrained metric must prefer the healthy epoch where plain sensitivity did not"
        )

    def test_monotone_in_spec_floor(self):
        """A stricter floor can only lower (or hold) the achievable sensitivity."""
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, 500)
        score = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, 500), 0, 1)
        loose = sensitivity_at_specificity_floor(y_true, score, spec_floor=0.70)
        strict = sensitivity_at_specificity_floor(y_true, score, spec_floor=0.95)
        assert loose >= strict


class TestYoudenJ:
    def test_perfect_classifier_scores_one(self):
        y_true = [0, 0, 0, 1, 1, 1]
        score = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
        assert youden_j(y_true, score) == pytest.approx(1.0)

    def test_coin_flip_scores_zero(self):
        y_true = [0, 1] * 100
        score = [0.5] * 200  # everything on one side of the 0.5 threshold
        assert youden_j(y_true, score, threshold=0.5) <= 0.0 + 1e-9

    def test_degenerate_always_positive_scores_zero_not_one(self):
        """The property that motivates using the floor-constrained metric
        instead: J correctly refuses to reward a trivial always-positive
        classifier with a high score, but it also does not FAVOUR a genuine
        discriminator over it by much if the base rate is high -- it is
        symmetric, not targeted at this project's specificity requirement."""
        y_true = np.array([1] * 90 + [0] * 10)
        always_pos = np.ones(100)
        assert youden_j(y_true, always_pos) == pytest.approx(0.0, abs=1e-9)
