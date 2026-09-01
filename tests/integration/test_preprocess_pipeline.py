"""End-to-end test of the preprocessing pipeline on a synthetic APTOS.

This exists so the pipeline is proven correct BEFORE the real 10 GB download:
it builds a fake dataset with the exact directory layout and CSV schema APTOS
ships, runs the real script as a subprocess, and asserts on the cache, the
manifest, and the grouping.
"""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from drdetect.data.manifest import read_manifest

REPO = Path(__file__).resolve().parents[2]


def make_fundus(seed: int, size: int = 600) -> np.ndarray:
    """A fundus-like image: bright off-centre disc on black, plus vessel-ish lines."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cy, cx = size // 2, size // 2 + int(rng.integers(-20, 20))
    yy, xx = np.ogrid[:size, :size]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 2 - 30) ** 2
    img[disc] = (170 + rng.integers(-20, 20), 85 + rng.integers(-15, 15), 55)
    for _ in range(6):  # vessels
        p0 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.line(img, p0, p1, (90, 30, 25), 3)
    img[disc] = np.clip(
        img[disc].astype(int) + rng.integers(-8, 8, img[disc].shape), 0, 255
    ).astype(np.uint8)
    return img


@pytest.fixture
def fake_aptos(tmp_path):
    """20 images with APTOS's exact layout: <root>/train.csv + <root>/train_images/."""
    root = tmp_path / "data" / "raw" / "aptos"
    images = root / "train_images"
    images.mkdir(parents=True)

    rows, grades = [], [0, 0, 0, 1, 2, 2, 3, 4, 0, 1, 2, 0, 2, 4, 3, 0, 1, 2, 0, 2]
    for i, grade in enumerate(grades):
        img_id = f"img{i:04d}"
        cv2.imwrite(str(images / f"{img_id}.png"), make_fundus(i))
        rows.append({"id_code": img_id, "diagnosis": grade})

    # A deliberate near-duplicate of img0000: grouping must catch this.
    dup = cv2.imread(str(images / "img0000.png"))
    cv2.imwrite(
        str(images / "img0000_dup.png"), cv2.resize(cv2.resize(dup, (300, 300)), (600, 600))
    )
    rows.append({"id_code": "img0000_dup", "diagnosis": 0})

    with open(root / "train.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id_code", "diagnosis"])
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def run_preprocess(workdir: Path, *extra: str):
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "preprocess.py"),
            "--dataset",
            "aptos",
            "--size",
            "128",
            "--workers",
            "2",
            *extra,
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestPipeline:
    def test_produces_cache_and_manifest(self, fake_aptos):
        res = run_preprocess(fake_aptos)
        assert res.returncode == 0, res.stderr

        cache = fake_aptos / "data" / "processed" / "aptos_128"
        assert len(list(cache.glob("*.jpg"))) == 21

        for img_path in list(cache.glob("*.jpg"))[:3]:
            img = cv2.imread(str(img_path))
            assert img.shape == (128, 128, 3), "every cached image must be square and uniform"

    def test_manifest_carries_labels_and_groups(self, fake_aptos):
        run_preprocess(fake_aptos)
        recs = read_manifest(fake_aptos / "data" / "manifests" / "aptos_128.csv")
        assert len(recs) == 21
        assert all(0 <= r.label <= 4 for r in recs)
        assert all(r.group_id for r in recs), "every image needs a group for leak-free splits"
        assert all(len(r.sha256) == 64 for r in recs)
        assert all(r.dataset == "aptos" for r in recs)

    def test_near_duplicate_is_grouped_with_its_original(self, fake_aptos):
        """The APTOS mitigation working: a re-scaled copy must not straddle folds."""
        run_preprocess(fake_aptos)
        recs = {
            r.image_id: r
            for r in read_manifest(fake_aptos / "data" / "manifests" / "aptos_128.csv")
        }
        assert recs["img0000_dup"].group_id == recs["img0000"].group_id

    def test_cache_is_far_smaller_than_source(self, fake_aptos):
        """The 10 GB -> 240 MB claim, in miniature."""
        run_preprocess(fake_aptos)
        raw = sum(
            f.stat().st_size for f in (fake_aptos / "data/raw/aptos/train_images").glob("*.png")
        )
        cached = sum(
            f.stat().st_size for f in (fake_aptos / "data/processed/aptos_128").glob("*.jpg")
        )
        assert cached < raw / 5

    def test_is_resumable(self, fake_aptos):
        """Interrupting a 3,662-image run must not mean starting over."""
        run_preprocess(fake_aptos)
        second = run_preprocess(fake_aptos)
        assert second.returncode == 0
        assert "21 already cached | 0 to process" in second.stdout

    def test_limit_flag_processes_subset(self, fake_aptos):
        res = run_preprocess(fake_aptos, "--limit", "5")
        assert res.returncode == 0
        assert len(list((fake_aptos / "data/processed/aptos_128").glob("*.jpg"))) == 5

    def test_missing_data_fails_cleanly(self, tmp_path):
        """A missing dataset is an expected state -- message, not traceback."""
        (tmp_path / "data" / "raw").mkdir(parents=True)
        res = run_preprocess(tmp_path)
        assert res.returncode == 1
        assert "download_data.sh" in res.stderr
        assert "Traceback" not in res.stderr


class TestSplitsOnManifest:
    def test_manifest_feeds_a_leak_free_split(self, fake_aptos):
        """The handoff that matters: manifest -> grouped split -> no leakage."""
        from drdetect.data.splits import assert_no_group_leakage, stratified_group_split

        run_preprocess(fake_aptos)
        recs = read_manifest(fake_aptos / "data" / "manifests" / "aptos_128.csv")
        labels = [r.label for r in recs]
        groups = [r.group_id for r in recs]

        folds, strategy = stratified_group_split(labels, groups, n_splits=3, seed=42)
        assert_no_group_leakage(folds, groups)
        assert sum(len(f) for f in folds) == len(recs)
        assert strategy == "true_groups"
