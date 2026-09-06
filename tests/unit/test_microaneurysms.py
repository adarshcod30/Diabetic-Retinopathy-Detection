"""Tests for microaneurysm candidate generation + classification."""

from __future__ import annotations

import numpy as np
import pytest

from drdetect.segmentation.microaneurysms import (
    extract_patch,
    find_candidates,
    generation_recall,
    label_candidates,
)

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


def _synthetic_fundus(size: int = 300, background: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    image = np.full((size, size, 3), background, dtype=np.uint8)
    noise = rng.integers(-3, 4, size=image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_dark_blob(
    image: np.ndarray, cx: int, cy: int, radius: int, value: int = 40
) -> np.ndarray:
    import cv2

    out = image.copy()
    cv2.circle(out, (cx, cy), radius, (value, value, value), thickness=-1)
    return out


def test_find_candidates_detects_a_synthetic_dark_blob():
    image = _synthetic_fundus()
    image = _draw_dark_blob(image, cx=150, cy=120, radius=8)

    candidates = find_candidates(image, disk_radius=16, min_area=3, max_area=1500)

    assert len(candidates) > 0
    closest = min(candidates, key=lambda c: (c.cx - 150) ** 2 + (c.cy - 120) ** 2)
    assert closest.cx == pytest.approx(150, abs=3)
    assert closest.cy == pytest.approx(120, abs=3)


def test_find_candidates_ignores_the_black_border():
    image = _synthetic_fundus()
    image[:20, :, :] = 0  # a near-black margin, as IDRiD images carry outside the fundus
    image = _draw_dark_blob(image, cx=150, cy=120, radius=8)

    candidates = find_candidates(image, disk_radius=16, min_gray=15.0)

    for c in candidates:
        assert c.cy >= 15  # nothing proposed inside the black margin


def test_label_candidates_matches_positive_and_negative():
    from drdetect.segmentation.microaneurysms import Candidate

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:45, 40:45] = 255
    candidates = [Candidate(cx=42.0, cy=42.0, area=10), Candidate(cx=80.0, cy=80.0, area=10)]

    labels = label_candidates(candidates, mask)

    assert labels.tolist() == [True, False]


def test_generation_recall_counts_distinct_true_instances():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:15, 10:15] = 255  # instance 1
    mask[60:65, 60:65] = 255  # instance 2, far away -> distinct component
    from drdetect.segmentation.microaneurysms import Candidate

    candidates = [Candidate(cx=12.0, cy=12.0, area=10)]  # only hits instance 1

    recovered, total = generation_recall(candidates, mask)

    assert (recovered, total) == (1, 2)


def test_generation_recall_is_zero_total_when_mask_is_empty():
    mask = np.zeros((50, 50), dtype=np.uint8)
    recovered, total = generation_recall([], mask)
    assert (recovered, total) == (0, 0)


def test_extract_patch_shape_at_image_corner_uses_reflect_padding():
    image = _synthetic_fundus(size=50)
    patch = extract_patch(image, cx=0.0, cy=0.0, size=15)
    assert patch.shape == (15, 15, 3)


def test_extract_patch_is_centered_for_an_interior_point():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[50, 50] = (255, 255, 255)
    patch = extract_patch(image, cx=50.0, cy=50.0, size=11)
    assert tuple(patch[5, 5]) == (255, 255, 255)


@pytest.fixture
def synthetic_ma_pair(tmp_path):
    import cv2

    from drdetect.segmentation.dataset import LesionImagePair

    image = _synthetic_fundus(size=200)
    mask = np.zeros((200, 200), dtype=np.uint8)
    for cx, cy in [(50, 50), (150, 60), (100, 150)]:
        image = _draw_dark_blob(image, cx, cy, radius=6)
        cv2.circle(mask, (cx, cy), 6, 255, thickness=-1)

    image_path = tmp_path / "IDRiD_01.png"
    mask_path = tmp_path / "IDRiD_01_MA.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return LesionImagePair(image_id="IDRiD_01", image_path=image_path, mask_path=mask_path)


def test_build_candidate_examples_recovers_synthetic_instances(synthetic_ma_pair):
    from drdetect.segmentation.microaneurysms import build_candidate_examples

    patches, labels, stats = build_candidate_examples(
        [synthetic_ma_pair], disk_radius=16, patch_size=33, seed=0
    )

    assert patches.shape[1:] == (33, 33, 3)
    assert len(labels) == len(patches)
    assert stats["n_true_instances"] == 3
    assert stats["n_recovered_instances"] >= 1
    assert stats["n_positive_candidates"] >= 1
    assert labels.sum() == stats["n_positive_candidates"]


def test_patch_classifier_forward_returns_one_logit_per_example():
    from drdetect.segmentation.microaneurysms import PatchClassifier

    model = PatchClassifier()
    out = model(torch.randn(4, 3, 33, 33))
    assert out.shape == (4,)


def _tiny_classifier_module(**kwargs):
    from drdetect.segmentation.microaneurysms import ClassifierModule, PatchClassifier

    return ClassifierModule(PatchClassifier(), max_epochs=5, **kwargs)


def test_classifier_training_step_returns_finite_loss(monkeypatch):
    module = _tiny_classifier_module(pos_weight=5.0)
    monkeypatch.setattr(module, "log", lambda *a, **kw: None)
    x = torch.randn(8, 3, 33, 33)
    y = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    loss = module.training_step((x, y), batch_idx=0)
    assert torch.isfinite(loss)


def test_classifier_validation_epoch_end_logs_auprc(monkeypatch):
    module = _tiny_classifier_module()
    module.eval()
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.__setitem__(name, value))

    x = torch.randn(6, 3, 33, 33)
    y = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    module.validation_step((x, y), batch_idx=0)
    module.on_validation_epoch_end()

    assert "val/auprc" in logged
    assert "val/accuracy" in logged
    assert module._val_logits == []
