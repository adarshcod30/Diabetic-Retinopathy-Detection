"""Guard against the single most damaging bug in this literature: patient leakage.

Fundus datasets contain BOTH EYES of the same patient. If the left eye lands in
train and the right eye in test, the model has effectively seen the test set --
the two eyes of one diabetic patient are highly correlated in DR severity. Every
reported metric is then inflated, silently, and the error is invisible in the
loss curves.

This test encodes the invariant so it can never regress. It is written against a
tiny in-module reference implementation now; when `drdetect.data.splits` lands in
Phase 1, point the import at it and delete the local copy.
"""

from collections import defaultdict

import pytest


def group_disjoint_split(records, n_splits=5):
    """Assign each patient (not each image) to exactly one fold."""
    by_patient = defaultdict(list)
    for rec in records:
        by_patient[rec["patient_id"]].append(rec)

    folds = [[] for _ in range(n_splits)]
    # deterministic ordering -> reproducible folds
    for i, patient in enumerate(sorted(by_patient)):
        folds[i % n_splits].extend(by_patient[patient])
    return folds


@pytest.fixture
def two_eyes_per_patient():
    return [
        {"image_id": f"p{p:03d}_{side}", "patient_id": f"p{p:03d}", "grade": p % 5}
        for p in range(50)
        for side in ("left", "right")
    ]


def test_no_patient_spans_two_folds(two_eyes_per_patient):
    folds = group_disjoint_split(two_eyes_per_patient, n_splits=5)
    seen = {}
    for idx, fold in enumerate(folds):
        for rec in fold:
            prev = seen.setdefault(rec["patient_id"], idx)
            assert prev == idx, (
                f"LEAKAGE: patient {rec['patient_id']} appears in folds {prev} and {idx}"
            )


def test_split_is_lossless(two_eyes_per_patient):
    folds = group_disjoint_split(two_eyes_per_patient, n_splits=5)
    assert sum(len(f) for f in folds) == len(two_eyes_per_patient)


def test_both_eyes_travel_together(two_eyes_per_patient):
    folds = group_disjoint_split(two_eyes_per_patient, n_splits=5)
    for fold in folds:
        ids = {r["image_id"] for r in fold}
        for rec in fold:
            partner = (
                rec["image_id"].replace("_left", "_right")
                if rec["image_id"].endswith("_left")
                else rec["image_id"].replace("_right", "_left")
            )
            assert partner in ids, f"{rec['image_id']} separated from its fellow eye"


def test_split_is_deterministic(two_eyes_per_patient):
    a = group_disjoint_split(two_eyes_per_patient, n_splits=5)
    b = group_disjoint_split(two_eyes_per_patient, n_splits=5)
    assert [[r["image_id"] for r in f] for f in a] == [[r["image_id"] for r in f] for f in b]
