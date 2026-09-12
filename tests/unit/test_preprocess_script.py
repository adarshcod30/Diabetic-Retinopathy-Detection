"""Unit tests for scripts/preprocess.py's own label-loading functions.

APTOS/Messidor-2/IDRiD's equivalents have no dedicated unit tests (only
indirect coverage via tests/integration/test_preprocess_pipeline.py's
APTOS-shaped fixture) -- this covers EyePACS's loader directly since a CSV
column-name mismatch here would silently produce zero labelled images
rather than an error, the same silent-corruption risk class as the
Messidor-2 filename-case bug (docs/22) and the CORN task-skew bug (docs/07).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from fetch_eyepacs_subset import stratified_sample  # noqa: E402
from preprocess import find_eyepacs, load_eyepacs_labels  # noqa: E402


class TestLoadEyepacsLabels:
    def test_parses_real_column_names_and_grade_range(self, tmp_path):
        csv_path = tmp_path / "trainLabels.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "level"])
            writer.writerow(["16_left", "0"])
            writer.writerow(["16_right", "4"])
            writer.writerow(["44_left", "2"])

        labels = load_eyepacs_labels(csv_path)

        assert labels == {"16_left": 0, "16_right": 4, "44_left": 2}
        assert all(isinstance(v, int) for v in labels.values())


class TestFindEyepacs:
    def test_locates_trainlabels_and_sibling_train_dir(self, tmp_path):
        (tmp_path / "train").mkdir()
        (tmp_path / "trainLabels.csv").write_text("image,level\n16_left,0\n")

        images_dir, csv_path = find_eyepacs(tmp_path)

        assert images_dir == tmp_path / "train"
        assert csv_path == tmp_path / "trainLabels.csv"

    def test_missing_data_raises_actionable_error_not_a_traceback(self, tmp_path):
        try:
            find_eyepacs(tmp_path)
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "fetch_eyepacs_subset.py" in str(e)


class TestStratifiedSample:
    def _write_labels(self, tmp_path, counts: dict[int, int]) -> Path:
        csv_path = tmp_path / "trainLabels.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "level"])
            i = 0
            for grade, n in counts.items():
                for _ in range(n):
                    writer.writerow([f"img{i}", grade])
                    i += 1
        return csv_path

    def test_preserves_class_proportions_not_a_naive_uniform_sample(self, tmp_path):
        # Deliberately APTOS-shaped imbalance (grade 3 is the rare class),
        # matching this project's own real class-imbalance experience (docs/06)
        # rather than a synthetic 50/50 split that couldn't catch a
        # rare-class-starving bug in the sampler.
        counts = {0: 8000, 1: 1000, 2: 4000, 3: 300, 4: 700}
        csv_path = self._write_labels(tmp_path, counts)
        total = sum(counts.values())

        sampled = stratified_sample(csv_path, n_subset=1400, seed=42)

        by_grade = Counter(sampled.values())
        for grade, n in counts.items():
            expected_frac = n / total
            actual_frac = by_grade[grade] / len(sampled)
            assert abs(actual_frac - expected_frac) < 0.01, (
                f"grade {grade}: expected ~{expected_frac:.3f}, got {actual_frac:.3f}"
            )

    def test_same_seed_is_reproducible(self, tmp_path):
        csv_path = self._write_labels(tmp_path, {0: 500, 1: 500})
        a = stratified_sample(csv_path, n_subset=200, seed=7)
        b = stratified_sample(csv_path, n_subset=200, seed=7)
        assert a == b

    def test_never_samples_more_than_a_class_actually_has(self, tmp_path):
        csv_path = self._write_labels(tmp_path, {0: 100, 1: 2})
        sampled = stratified_sample(csv_path, n_subset=90, seed=42)
        by_grade = Counter(sampled.values())
        assert by_grade[1] <= 2
