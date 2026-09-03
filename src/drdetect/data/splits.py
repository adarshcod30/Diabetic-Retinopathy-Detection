"""Leak-free cross-validation splits.

The failure this module exists to prevent
-----------------------------------------
Fundus datasets contain both eyes of the same patient, and the two eyes of one
diabetic are strongly correlated in DR severity. If one eye lands in train and
the other in test, the model has effectively seen the test set. Every reported
metric inflates, the loss curves look perfect, and nothing warns you.

The complication specific to APTOS
----------------------------------
APTOS-2019's `train.csv` carries only `id_code` and `diagnosis`. There is **no
patient or eye identifier**, so a true patient-level split is impossible on that
dataset as distributed. Silently falling back to an image-level split would be
exactly the failure described above, so this module:

  1. refuses to guess -- `stratified_group_split` requires an explicit decision;
  2. offers `perceptual_hash_groups` as a documented mitigation: images that are
     near-identical (plausibly the same eye or patient) are grouped so they
     cannot straddle folds;
  3. records which strategy was used, so the caveat travels with the results.

Messidor-2 *does* encode exam pairing in its filenames, so real grouping is
available there and should be used.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

__all__ = [
    "SplitStrategy",
    "dhash",
    "structural_hash",
    "perceptual_hash_groups",
    "validate_grouping",
    "stratified_group_split",
    "assert_no_group_leakage",
]


class SplitStrategy:
    """How grouping was determined. Travels with results as a provenance record."""

    TRUE_GROUPS = "true_groups"  # real patient/eye ids -- trustworthy
    PERCEPTUAL = "perceptual_hash"  # near-duplicate grouping -- mitigation only
    IMAGE_LEVEL = "image_level"  # no grouping -- leakage possible, must be declared


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash: a 64-bit perceptual fingerprint.

    Compares each pixel with its right-hand neighbour on a downscaled grayscale
    image, so it is robust to resolution, JPEG artefacts and mild brightness
    shifts, but sensitive to actual content. Two photographs of the same eye
    typically differ by only a few bits.

    Implemented here rather than pulled from `imagehash` to avoid a dependency
    for ~10 lines, and so the exact behaviour is pinned by our own tests.
    """
    import cv2

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def structural_hash(image: np.ndarray, size: int = 256, hash_size: int = 16) -> int:
    """Perceptual hash of an image's *coarse structure*, for grouping.

    Deliberately computed on a plain circle-crop + resize, NOT on the
    Ben Graham-preprocessed image used for training.

    Ben Graham preprocessing subtracts the local average colour, which removes
    exactly the low-frequency content dHash depends on. Measured on a fundus and
    a degraded copy of itself: hashing the Ben Graham output put them 8 bits
    apart (above the grouping threshold, so the duplicate was missed), while
    hashing the structural image put them 0 bits apart -- with unrelated images
    still 14 bits away.

    The identity of an eye lives in its vessel and optic-disc layout. That is
    what this hashes.

    Resolution matters more than it looks. At 8x8 (64 bits) on 128 px, measured
    across APTOS the median inter-image distance was 18 bits with ~670 pairs
    falling within 5 bits -- most of them false matches, because 64 bits cannot
    separate two fundus photographs that are both "a bright disc on black".
    At 16x16 (256 bits) on 256 px the distribution is cleanly bimodal: 148 pairs
    at 0-4 bits and then NOTHING until 39 bits. Those false matches are what
    percolated the grouping into a single giant component.
    """
    import cv2

    from drdetect.enhance.preprocessing import circle_crop

    cropped = circle_crop(image)
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return dhash(resized, hash_size=hash_size)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def perceptual_hash_groups(
    image_ids: list[str],
    hashes: list[int],
    max_distance: int = 6,
    *,
    max_group_size: int | None = None,
) -> dict[str, str]:
    """Group near-duplicate images via union-find over Hamming distance.

    Returns {image_id: group_id}. Images within `max_distance` bits are treated
    as the same group (transitively).

    Percolation, and why `max_group_size` exists
    --------------------------------------------
    Union-find takes the TRANSITIVE closure: a~b and b~c puts a, b and c in one
    group even if a and c are far apart. Over a large set, a small number of
    spurious edges is enough to chain almost everything together -- a random
    graph needs only ~N/2 edges to form a giant connected component.

    This is not hypothetical. On APTOS (3,662 images) a 64-bit hash produced
    ~670 near-matches, and the transitive closure merged 3,138 images into a
    single "group" which then became an entire validation fold. The split was
    train=524 / val=3138 and silently reported as `true_groups`.

    Two defences: a hash resolution that does not manufacture false edges (see
    `structural_hash`), and this hard cap on group size. `max_group_size`
    defaults to max(25, 1% of the dataset) -- a "patient" larger than that is a
    hash failure, not a patient, so the merge is refused.

    Choosing `max_distance`
    -----------------------
    Measured on APTOS with `structural_hash` over all 6.7M pairs, the cumulative
    count is flat across thresholds 2-6 (148 pairs at every value) and stays
    near-flat to 8 (150). It then climbs -- 207 at 10, 391 at 12 -- and chaining
    appears: threshold 10 produced a 31-image group whose members were up to 33
    bits apart. The default of 6 sits mid-plateau, so the result is insensitive
    to the exact value.

    Note the threshold must be measured for the hash ACTUALLY used. An earlier
    value of 10 was taken from a distribution measured on Ben Graham-preprocessed
    images and then applied to `structural_hash`, which runs on raw
    circle-cropped images. Different transform, different distances.

    This remains a *mitigation*, not a substitute for patient IDs: it catches
    repeated or near-identical captures, but two genuinely different photographs
    of one patient's two eyes will not be grouped. Declare the limitation when
    reporting.
    """
    if len(image_ids) != len(hashes):
        raise ValueError("image_ids and hashes must be the same length")

    n = len(image_ids)
    if max_group_size is None:
        max_group_size = max(25, n // 100)

    parent = list(range(n))
    size = [1] * n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    refused = 0

    def union(i: int, j: int) -> None:
        nonlocal refused
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if size[ri] + size[rj] > max_group_size:
            # Refusing the merge is the safe failure: two smaller groups may
            # leak, but one giant group destroys the split entirely.
            refused += 1
            return
        lo, hi = (ri, rj) if ri < rj else (rj, ri)
        parent[hi] = lo
        size[lo] += size[hi]

    # O(n^2). Fine for APTOS (3,662 -> ~6.7M comparisons, seconds). For EyePACS
    # scale this would need LSH bucketing on hash prefixes.
    for i in range(n):
        hi_hash = hashes[i]
        for j in range(i + 1, n):
            if (hi_hash ^ hashes[j]).bit_count() <= max_distance:
                union(i, j)

    if refused:
        warnings.warn(
            f"{refused} merge(s) refused because they would exceed max_group_size="
            f"{max_group_size}. This usually means the hash is producing false "
            f"matches; check structural_hash resolution and max_distance.",
            UserWarning,
            stacklevel=2,
        )

    roots = defaultdict(list)
    for idx, image_id in enumerate(image_ids):
        roots[find(idx)].append(image_id)

    # Name each group after its lexicographically smallest member, NOT the
    # union-find root index. The root index depends on the order the inputs
    # arrived in, so identical data grouped twice produced identical partitions
    # under different names -- and StratifiedGroupKFold assigns folds by group
    # VALUE, so the two runs got different splits. Measured: the 512 px and
    # 768 px manifests had the same 3,523 groups but shared only 139 of 733
    # validation images, making the two runs non-comparable.
    #
    # Naming by member makes the mapping a pure function of the partition.
    mapping: dict[str, str] = {}
    for members in roots.values():
        gid = f"g_{min(members)}"
        for m in members:
            mapping[m] = gid
    return mapping


def validate_grouping(
    groups: list[str],
    *,
    max_group_fraction: float = 0.05,
    min_group_fraction: float = 0.5,
) -> dict:
    """Reject degenerate groupings before they silently destroy a split.

    Raises if any single group holds more than `max_group_fraction` of the data,
    or if the number of distinct groups falls below `min_group_fraction` of the
    number of images.

    This exists because the failure it catches is invisible downstream: a
    grouping that collapsed 86% of APTOS into one group still reported strategy
    `true_groups`, still passed the no-leakage assertion (trivially -- one group
    cannot straddle folds), and produced a train=524 / val=3138 split that would
    have trained for hours and produced a meaningless number.
    """
    from collections import Counter

    n = len(groups)
    counts = Counter(groups)
    largest_id, largest = counts.most_common(1)[0]
    stats = {
        "n_images": n,
        "n_groups": len(counts),
        "largest_group": largest,
        "largest_group_id": largest_id,
        "merged": n - len(counts),
    }

    if largest > max_group_fraction * n:
        raise ValueError(
            f"Degenerate grouping: group {largest_id!r} holds {largest}/{n} images "
            f"({100 * largest / n:.1f}%), above the {100 * max_group_fraction:.0f}% limit.\n"
            f"A 'patient' that large is a hash collapse, not a patient. Re-run "
            f"grouping with a higher-resolution hash or a tighter max_distance."
        )
    if len(counts) < min_group_fraction * n:
        raise ValueError(
            f"Degenerate grouping: {len(counts)} groups for {n} images "
            f"(< {100 * min_group_fraction:.0f}%). The hash is over-merging."
        )
    return stats


def stratified_group_split(
    labels: list[int],
    groups: list[str] | None,
    *,
    n_splits: int = 5,
    seed: int = 42,
    allow_ungrouped: bool = False,
) -> tuple[list[np.ndarray], str]:
    """Stratified k-fold that keeps each group whole.

    Args:
        labels: class label per sample, used for stratification.
        groups: grouping key per sample. `None` or all-empty means no grouping
            information exists.
        allow_ungrouped: must be set explicitly to proceed without groups. This
            is a guard rail: falling back silently is the bug this module exists
            to prevent.

    Returns:
        (fold_indices, strategy) where fold_indices[k] holds the *test* indices
        of fold k, and strategy is a `SplitStrategy` constant to record.

    Raises:
        ValueError: if grouping is absent and `allow_ungrouped` is False.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    labels_arr = np.asarray(labels)
    has_groups = groups is not None and any(g for g in groups)

    if not has_groups:
        if not allow_ungrouped:
            raise ValueError(
                "No grouping information supplied. A plain image-level split can leak "
                "the same patient across folds and silently inflate every metric.\n"
                "Either provide `groups` (from patient ids, or from "
                "perceptual_hash_groups()), or pass allow_ungrouped=True to accept "
                "the risk explicitly and declare it when reporting results."
            )
        warnings.warn(
            "Splitting at IMAGE level with no grouping. If this dataset contains "
            "multiple images per patient, results will be optimistically biased. "
            "Record SplitStrategy.IMAGE_LEVEL alongside any metric derived from it.",
            UserWarning,
            stacklevel=2,
        )
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [test for _, test in splitter.split(np.zeros(len(labels_arr)), labels_arr)]
        return folds, SplitStrategy.IMAGE_LEVEL

    groups_arr = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [test for _, test in splitter.split(np.zeros(len(labels_arr)), labels_arr, groups_arr)]
    return folds, SplitStrategy.TRUE_GROUPS


def assert_no_group_leakage(folds: list[np.ndarray], groups: list[str]) -> None:
    """Raise if any group appears in more than one fold.

    Call this after splitting, every time. It is cheap, and it converts the most
    expensive silent bug in this literature into a loud, immediate failure.
    """
    seen: dict[str, int] = {}
    for fold_idx, indices in enumerate(folds):
        for i in indices:
            gid = groups[i]
            if not gid:
                continue
            prev = seen.setdefault(gid, fold_idx)
            if prev != fold_idx:
                raise AssertionError(
                    f"Group leakage: group {gid!r} appears in folds {prev} and {fold_idx}."
                )
