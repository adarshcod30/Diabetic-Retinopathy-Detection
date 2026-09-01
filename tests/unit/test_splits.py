"""Tests for leak-free splitting.

The central assertion is that a plain image-level split cannot happen by
accident. Everything else in this file supports that.
"""

import numpy as np
import pytest

from drdetect.data.splits import (
    SplitStrategy,
    assert_no_group_leakage,
    dhash,
    perceptual_hash_groups,
    stratified_group_split,
)


@pytest.fixture
def paired_eyes():
    """100 patients x 2 eyes, correlated grades -- the leakage scenario."""
    rng = np.random.default_rng(0)
    labels, groups = [], []
    for p in range(100):
        grade = int(rng.integers(0, 5))
        for _ in range(2):
            labels.append(grade)
            groups.append(f"patient_{p:03d}")
    return labels, groups


class TestGuardRail:
    def test_refuses_ungrouped_split_by_default(self):
        """The whole point: you cannot get an image-level split by omission."""
        with pytest.raises(ValueError, match="No grouping information"):
            stratified_group_split([0, 1, 2, 3] * 10, groups=None)

    def test_empty_group_strings_also_refused(self):
        with pytest.raises(ValueError):
            stratified_group_split([0, 1, 2, 3] * 10, groups=[""] * 40)

    def test_explicit_optin_warns_and_reports_strategy(self):
        with pytest.warns(UserWarning, match="IMAGE level"):
            folds, strategy = stratified_group_split(
                [0, 1, 2, 3, 4] * 20, groups=None, allow_ungrouped=True
            )
        assert strategy == SplitStrategy.IMAGE_LEVEL
        assert sum(len(f) for f in folds) == 100


class TestGroupedSplit:
    def test_no_patient_spans_two_folds(self, paired_eyes):
        labels, groups = paired_eyes
        folds, strategy = stratified_group_split(labels, groups, n_splits=5)
        assert strategy == SplitStrategy.TRUE_GROUPS
        assert_no_group_leakage(folds, groups)  # raises on failure

    def test_partition_is_complete_and_disjoint(self, paired_eyes):
        labels, groups = paired_eyes
        folds, _ = stratified_group_split(labels, groups, n_splits=5)
        allidx = np.concatenate(folds)
        assert len(allidx) == len(labels)
        assert len(set(allidx.tolist())) == len(labels)

    def test_is_reproducible(self, paired_eyes):
        labels, groups = paired_eyes
        a, _ = stratified_group_split(labels, groups, seed=42)
        b, _ = stratified_group_split(labels, groups, seed=42)
        assert [f.tolist() for f in a] == [f.tolist() for f in b]

    def test_leakage_detector_actually_fires(self, paired_eyes):
        """Guard against a vacuous guard: the assertion must fail on bad input."""
        _, groups = paired_eyes
        bad = [np.array([0, 2]), np.array([1, 3])]  # indices 0,1 are the same patient
        with pytest.raises(AssertionError, match="Group leakage"):
            assert_no_group_leakage(bad, groups)


class TestPerceptualHash:
    def _disc(self, cx=64, brightness=180, size=128):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:size, :size]
        img[(yy - 64) ** 2 + (xx - cx) ** 2 <= 40**2] = (brightness, brightness // 2, 40)
        return img

    def test_identical_images_hash_identically(self):
        img = self._disc()
        assert dhash(img) == dhash(img.copy())

    def test_hash_survives_rescaling(self):
        import cv2

        img = self._disc(size=256)
        small = cv2.resize(img, (128, 128))
        assert bin(dhash(img) ^ dhash(small)).count("1") <= 5

    def test_different_content_hashes_differently(self):
        assert dhash(self._disc(cx=40)) != dhash(self._disc(cx=90))

    def test_groups_near_duplicates_together(self):
        import cv2

        base = self._disc()
        images = [base, cv2.resize(cv2.resize(base, (64, 64)), (128, 128)), self._disc(cx=95)]
        ids = ["a", "a_rescaled", "different"]
        mapping = perceptual_hash_groups(ids, [dhash(i) for i in images])
        assert mapping["a"] == mapping["a_rescaled"]
        assert mapping["different"] != mapping["a"]

    def test_grouping_is_transitive(self):
        """Union-find: a~b and b~c implies a, b, c share a group."""
        mapping = perceptual_hash_groups(["a", "b", "c"], [0b0000, 0b0001, 0b0011])
        assert mapping["a"] == mapping["b"] == mapping["c"]

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            perceptual_hash_groups(["a", "b"], [1])


class TestAptosScenario:
    """APTOS ships no patient ids. This documents the intended workflow."""

    def test_hash_groups_feed_the_grouped_splitter(self):
        rng = np.random.default_rng(1)
        ids = [f"img_{i:04d}" for i in range(60)]
        # 30 unique eyes, each appearing twice with a 1-bit difference
        hashes = []
        for _i in range(30):
            h = int(rng.integers(0, 2**60))
            hashes += [h, h ^ 1]
        mapping = perceptual_hash_groups(ids, hashes)
        groups = [mapping[i] for i in ids]
        assert len(set(groups)) == 30

        labels = [i % 5 for i in range(60)]
        folds, strategy = stratified_group_split(labels, groups, n_splits=3)
        assert strategy == SplitStrategy.TRUE_GROUPS
        assert_no_group_leakage(folds, groups)


class TestStructuralHash:
    """Regression tests for a real bug: grouping originally hashed the
    Ben Graham-preprocessed image, which silently failed to group duplicates."""

    def _fundus(self, seed: int, size: int = 600):
        rng = np.random.default_rng(seed)
        import cv2

        img = np.zeros((size, size, 3), np.uint8)
        cy, cx = size // 2, size // 2 + int(rng.integers(-20, 20))
        yy, xx = np.ogrid[:size, :size]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 2 - 30) ** 2
        img[disc] = (170, 85, 55)
        for _ in range(6):
            cv2.line(
                img,
                (int(rng.integers(0, size)), int(rng.integers(0, size))),
                (int(rng.integers(0, size)), int(rng.integers(0, size))),
                (90, 30, 25),
                3,
            )
        return img

    def test_beats_hashing_the_preprocessed_image(self):
        """Ben Graham removes the low-frequency content dHash depends on.

        Hashing its output put a true duplicate 8 bits away (missed, threshold
        5). Hashing the structural image puts it at 0.
        """
        import cv2

        from drdetect.data.splits import structural_hash
        from drdetect.enhance.preprocessing import preprocess

        orig = self._fundus(0)
        degraded = cv2.resize(cv2.resize(orig, (300, 300)), (600, 600))

        via_preprocessed = bin(
            dhash(preprocess(orig, size=128)) ^ dhash(preprocess(degraded, size=128))
        ).count("1")
        via_structural = bin(structural_hash(orig) ^ structural_hash(degraded)).count("1")

        assert via_structural < via_preprocessed
        assert via_structural <= 5, "a true duplicate must fall inside the grouping threshold"

    def test_still_separates_unrelated_eyes(self):
        """Grouping must not collapse everything into one fold."""
        from drdetect.data.splits import structural_hash

        a, b = structural_hash(self._fundus(0)), structural_hash(self._fundus(7))
        assert bin(a ^ b).count("1") > 5

    def test_invariant_to_capture_resolution(self):
        import cv2

        from drdetect.data.splits import structural_hash

        img = self._fundus(3, size=800)
        assert (
            bin(structural_hash(img) ^ structural_hash(cv2.resize(img, (400, 400)))).count("1") <= 5
        )
