"""End-to-end test: synthetic APTOS -> preprocess -> train -> checkpoint.

Proves the whole Phase 1 pipeline runs before the real 10 GB download. Marked
`slow` (it downloads ImageNet weights on first run and trains a few epochs), so
CI's fast lane can skip it with `-m 'not slow'`.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("lightning")

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


def fundus(seed: int, grade: int, size: int = 400) -> np.ndarray:
    """Fundus-like image whose lesion load scales with grade.

    The signal is deliberately learnable -- the point is to exercise the
    plumbing, not to estimate accuracy. Metrics from this fixture say nothing
    about real performance.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    cy = cx = size // 2
    yy, xx = np.ogrid[:size, :size]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 2 - 20) ** 2] = (165, 80, 50)
    for _ in range(6):
        cv2.line(
            img,
            (int(rng.integers(0, size)), int(rng.integers(0, size))),
            (int(rng.integers(0, size)), int(rng.integers(0, size))),
            (85, 28, 22),
            2,
        )
    cv2.circle(img, (cx - 60, cy), 25, (220, 180, 120), -1)
    for _ in range(grade * 10):  # haemorrhage-like
        p = (int(rng.integers(40, size - 40)), int(rng.integers(40, size - 40)))
        cv2.circle(img, p, int(rng.integers(2, 4)), (140, 20, 20), -1)
    for _ in range(grade * 5):  # exudate-like
        p = (int(rng.integers(40, size - 40)), int(rng.integers(40, size - 40)))
        cv2.circle(img, p, int(rng.integers(2, 6)), (245, 230, 140), -1)
    return img


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Build a synthetic dataset, preprocess it, and train. Runs once."""
    work = tmp_path_factory.mktemp("e2e")
    images = work / "data" / "raw" / "aptos" / "train_images"
    images.mkdir(parents=True)

    rows, i = [], 0
    for grade, n in {0: 40, 1: 10, 2: 24, 3: 8, 4: 10}.items():  # APTOS-like imbalance
        for _ in range(n):
            cv2.imwrite(str(images / f"img{i:04d}.png"), fundus(i, grade))
            rows.append({"id_code": f"img{i:04d}", "diagnosis": grade})
            i += 1
    with open(work / "data/raw/aptos/train.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id_code", "diagnosis"])
        w.writeheader()
        w.writerows(rows)

    pre = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/preprocess.py"),
            "--dataset",
            "aptos",
            "--size",
            "128",
            "--workers",
            "2",
        ],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert pre.returncode == 0, pre.stderr

    train = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/train.py"),
            "--size",
            "128",
            "--batch-size",
            "8",
            "--epochs",
            "3",
            "--workers",
            "0",
            "--folds",
            "0",
        ],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert train.returncode == 0, train.stderr[-3000:]
    return work, train.stdout


class TestTrainingPipeline:
    def test_writes_checkpoint(self, trained_run):
        work, _ = trained_run
        ckpts = list((work / "models" / "checkpoints").rglob("*.ckpt"))
        assert ckpts, "no checkpoint written"
        assert ckpts[0].stat().st_size > 1_000_000

    def test_summary_records_config_and_qwk(self, trained_run):
        work, _ = trained_run
        summary = json.loads(next((work / "runs").rglob("summary.json")).read_text())
        assert "qwk_mean" in summary and -1.0 <= summary["qwk_mean"] <= 1.0
        assert summary["config"]["size"] == 128
        assert summary["folds"][0]["split_strategy"] == "true_groups"

    def test_uses_grouped_split_not_image_level(self, trained_run):
        """Grouping must flow manifest -> split. Image-level here means the
        leakage guard silently degraded."""
        _, stdout = trained_run
        assert "split strategy: true_groups" in stdout
        assert "WARNING: no grouping" not in stdout

    def test_batchnorm_is_actually_frozen(self, trained_run):
        """A freeze that silently no-ops is worse than no freeze -- it looks
        applied in the config and is absent in the model."""
        _, stdout = trained_run
        assert "frozen BatchNorm layers:" in stdout
        n = int(stdout.split("frozen BatchNorm layers:")[1].split()[0])
        assert n > 0

    def test_logs_per_class_recall(self, trained_run):
        """Aggregate QWK can look healthy while a rare class is never predicted.
        Per-class recall is the only thing that shows it."""
        work, _ = trained_run
        metrics = next((work / "runs").rglob("metrics.csv"))
        header = metrics.read_text().splitlines()[0]
        for cls in range(5):
            assert f"val/recall_{cls}_" in header

    def test_reports_referable_operating_point(self, trained_run):
        work, _ = trained_run
        header = next((work / "runs").rglob("metrics.csv")).read_text().splitlines()[0]
        assert "val/sensitivity_referable" in header
        assert "val/specificity_referable" in header


class TestModelUnits:
    def test_freeze_batchnorm_survives_train_mode(self):
        """model.train() must not silently undo the freeze -- this produces no
        error, only unstable validation metrics."""
        import torch.nn as nn

        from drdetect.grading.model import build_model

        model = build_model("resnet18", pretrained=False, freeze_bn=True)
        model.train()
        bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        assert bns, "fixture backbone has no BatchNorm"
        assert all(not m.training for m in bns), "BN re-entered train mode"
        assert all(not m.weight.requires_grad for m in bns)

    def test_unfrozen_model_trains_batchnorm(self):
        import torch.nn as nn

        from drdetect.grading.model import build_model

        model = build_model("resnet18", pretrained=False, freeze_bn=False).train()
        bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        assert all(m.training for m in bns)

    def test_output_shape_matches_num_outputs(self):
        import torch

        from drdetect.grading.model import build_model

        model = build_model("resnet18", num_outputs=5, pretrained=False).eval()
        with torch.no_grad():
            assert model(torch.randn(2, 3, 128, 128)).shape == (2, 5)
