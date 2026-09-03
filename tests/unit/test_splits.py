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


class TestPercolationGuard:
    """Regression tests for the failure that produced a train=524/val=3138 split.

    Union-find takes the transitive closure, so a small number of spurious
    near-matches chains almost everything into one giant component. On APTOS a
    64-bit hash produced ~670 false edges and merged 3,138 of 3,662 images into
    a single group, which then became an entire validation fold -- while still
    reporting strategy `true_groups` and passing the leakage assertion.
    """

    def test_chain_cannot_swallow_the_dataset(self):
        """A~B~C~...~Z chain must not become one group when capped."""
        from drdetect.data.splits import perceptual_hash_groups

        # Each hash differs from its neighbour by 1 bit -> a connected chain,
        # even though the ends are 199 bits apart.
        ids = [f"img{i:04d}" for i in range(200)]
        hashes = [(1 << i) - 1 for i in range(200)]

        with pytest.warns(UserWarning, match="merge\\(s\\) refused"):
            mapping = perceptual_hash_groups(ids, hashes, max_distance=2, max_group_size=25)

        from collections import Counter

        largest = Counter(mapping.values()).most_common(1)[0][1]
        assert largest <= 25, f"cap breached: one group holds {largest} of 200"

    def test_validator_rejects_a_collapsed_grouping(self):
        """The exact APTOS shape: 86% of images in one group."""
        from drdetect.data.splits import validate_grouping

        groups = ["g000000"] * 3138 + [f"g{i:06d}" for i in range(524)]
        with pytest.raises(ValueError, match="Degenerate grouping"):
            validate_grouping(groups)

    def test_validator_rejects_wholesale_over_merging(self):
        from drdetect.data.splits import validate_grouping

        groups = [f"g{i // 20:04d}" for i in range(1000)]  # 50 groups for 1000 images
        with pytest.raises(ValueError, match="over-merging"):
            validate_grouping(groups)

    def test_validator_accepts_healthy_grouping(self):
        """~150 duplicate pairs in 3,662 images -- the real APTOS shape."""
        from drdetect.data.splits import validate_grouping

        groups = [f"g{i:05d}" for i in range(3500)] + [f"g{i:05d}" for i in range(150)]
        stats = validate_grouping(groups)
        assert stats["n_groups"] == 3500
        assert stats["largest_group"] == 2
        assert stats["merged"] == 150

    def test_a_giant_group_would_pass_the_leakage_check(self):
        """Documents WHY the validator is needed: the leakage assertion cannot
        catch this, because one group trivially never straddles folds."""
        from drdetect.data.splits import assert_no_group_leakage

        groups = ["mega"] * 100
        folds = [np.arange(0, 100)]
        assert_no_group_leakage(folds, groups)  # passes -- and the split is useless

    def test_higher_resolution_hash_separates_better(self):
        """256-bit hashing is what removes the false edges in the first place."""
        import cv2

        from drdetect.data.splits import structural_hash

        def disc(seed, size=400):
            rng = np.random.default_rng(seed)
            img = np.zeros((size, size, 3), np.uint8)
            yy, xx = np.ogrid[:size, :size]
            img[(yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= (size // 2 - 20) ** 2] = (
                165,
                80,
                50,
            )
            for _ in range(5):
                cv2.line(
                    img,
                    (int(rng.integers(0, size)), int(rng.integers(0, size))),
                    (int(rng.integers(0, size)), int(rng.integers(0, size))),
                    (85, 28, 22),
                    3,
                )
            return img

        a, b = disc(1), disc(2)
        coarse = (
            bin(
                structural_hash(a, size=128, hash_size=8)
                ^ structural_hash(b, size=128, hash_size=8)
            ).count("1")
            / 64
        )
        fine = bin(structural_hash(a) ^ structural_hash(b)).count("1") / 256
        assert fine >= coarse * 0.8, "256-bit must not separate distinct images less than 64-bit"
        assert bin(structural_hash(a) ^ structural_hash(a)).count("1") == 0


class TestGroupingDeterminism:
    """Regression: identical data must produce identical group NAMES.

    perceptual_hash_groups originally named groups by union-find root index,
    which depends on input order. preprocess.py fed it results in process-pool
    completion order, so the same dataset preprocessed twice produced identical
    partitions under different names. StratifiedGroupKFold assigns folds by
    group value, so the 512 px and 768 px manifests -- same images, same labels,
    same 3,523 groups -- yielded validation sets overlapping in only 139 of 733
    images, silently making the two runs non-comparable.
    """

    def _hashes(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        ids = [f"img{i:04d}" for i in range(n)]
        hs = []
        for _ in range(n // 2):
            h = int(rng.integers(0, 2**60))
            hs += [h, h ^ 1]  # pairs that should group together
        return ids, hs

    def test_shuffled_input_gives_identical_mapping(self):
        from drdetect.data.splits import perceptual_hash_groups

        ids, hs = self._hashes()
        a = perceptual_hash_groups(ids, hs, max_distance=6)

        order = list(range(len(ids)))
        np.random.default_rng(7).shuffle(order)
        b = perceptual_hash_groups([ids[i] for i in order], [hs[i] for i in order], max_distance=6)

        assert a == b, "group assignment changed with input order"

    def test_names_are_derived_from_members(self):
        from drdetect.data.splits import perceptual_hash_groups

        # (1 << 40) - 1 sets 40 bits, so it is far from 0. A single set bit
        # would be Hamming distance 1 and would group with everything.
        m = perceptual_hash_groups(["b", "a", "c"], [0b0000, 0b0001, (1 << 40) - 1], max_distance=2)
        assert m["a"] == m["b"] == "g_a", m  # named for the smallest member
        assert m["c"] != m["a"]

    def test_same_partition_survives_a_rebuild(self):
        """The concrete failure: two manifests, same partition, different names."""
        from drdetect.data.splits import perceptual_hash_groups

        ids, hs = self._hashes(seed=3)
        first = perceptual_hash_groups(ids, hs, max_distance=6)
        rebuilt = perceptual_hash_groups(list(reversed(ids)), list(reversed(hs)), max_distance=6)
        assert set(first.values()) == set(rebuilt.values())
        assert first == rebuilt

    def test_split_is_stable_across_rebuilds(self):
        """End to end: identical folds from an independently rebuilt grouping."""
        from drdetect.data.splits import perceptual_hash_groups, stratified_group_split

        ids, hs = self._hashes(n=100, seed=5)
        labels = [i % 5 for i in range(len(ids))]

        g1 = perceptual_hash_groups(ids, hs, max_distance=6)
        order = list(range(len(ids)))
        np.random.default_rng(11).shuffle(order)
        g2 = perceptual_hash_groups([ids[i] for i in order], [hs[i] for i in order], max_distance=6)

        f1, _ = stratified_group_split(labels, [g1[i] for i in ids], n_splits=5, seed=42)
        f2, _ = stratified_group_split(labels, [g2[i] for i in ids], n_splits=5, seed=42)
        assert [sorted(f.tolist()) for f in f1] == [sorted(f.tolist()) for f in f2]
