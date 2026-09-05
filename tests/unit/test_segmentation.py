"""Tests for the lesion segmentation dataset, metrics, and Lightning module."""

from __future__ import annotations

import numpy as np
import pytest

from drdetect.segmentation.dataset import IDRiDLesionDataset, LesionImagePair
from drdetect.segmentation.metrics import dice_coefficient, pixel_auprc

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("segmentation_models_pytorch")


@pytest.fixture
def synthetic_pair(tmp_path):
    """A 600x800 image with a small, real lesion blob -- not the flat colour
    used elsewhere, since a patch-sampling test needs texture and a lesion
    with actual spatial extent to sample around."""
    import cv2

    rng = np.random.default_rng(0)
    image = rng.integers(40, 200, size=(600, 800, 3), dtype=np.uint8)
    mask = np.zeros((600, 800), dtype=np.uint8)
    mask[500:520, 700:720] = 76  # IDRiD's own on-value, off in a corner far from center

    image_path = tmp_path / "IDRiD_01.jpg"
    mask_path = tmp_path / "IDRiD_01_EX.tif"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask)
    return LesionImagePair(image_id="IDRiD_01", image_path=image_path, mask_path=mask_path)


def test_patch_shape_and_dtype(synthetic_pair):
    ds = IDRiDLesionDataset([synthetic_pair], patch_size=256, train=True, patches_per_image=5)
    img, mask = ds[0]
    assert img.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert mask.dtype.is_floating_point
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_len_scales_with_patches_per_image(synthetic_pair):
    ds = IDRiDLesionDataset([synthetic_pair, synthetic_pair], patch_size=256, patches_per_image=7)
    assert len(ds) == 14


def test_lesion_biased_sampling_finds_the_lesion_more_than_chance(synthetic_pair):
    """The lesion occupies 20*20 / (600*800) = 0.083% of the image -- a
    uniformly random 256x256 patch would contain it on roughly (256+20)^2 /
    (600*800) ~= 12.5% of draws. With lesion_patch_prob=1.0 it must be ~100%."""
    ds = IDRiDLesionDataset(
        [synthetic_pair], patch_size=256, train=False, patches_per_image=30, lesion_patch_prob=1.0
    )
    hits = sum(1 for i in range(len(ds)) if ds[i][1].sum() > 0)
    assert hits == len(ds), "lesion_patch_prob=1.0 should hit the lesion on every draw"


def test_zero_lesion_prob_still_works_on_lesion_free_images(tmp_path):
    """An image with no lesion at all (a real, common IDRiD case -- not every
    image has every lesion type) must not crash the lesion-centering branch."""
    import cv2

    rng = np.random.default_rng(1)
    image = rng.integers(40, 200, size=(400, 400, 3), dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=np.uint8)
    image_path = tmp_path / "IDRiD_99.jpg"
    mask_path = tmp_path / "IDRiD_99_EX.tif"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask)
    pair = LesionImagePair(image_id="IDRiD_99", image_path=image_path, mask_path=mask_path)

    ds = IDRiDLesionDataset([pair], patch_size=128, patches_per_image=5, lesion_patch_prob=0.8)
    for i in range(len(ds)):
        img, mask_out = ds[i]
        assert img.shape == (3, 128, 128)


def test_pixel_auprc_perfect_score_is_one():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.0, 0.1, 0.9, 0.95, 0.05, 0.8])
    assert pixel_auprc(y_true, y_score) == pytest.approx(1.0)


def test_pixel_auprc_raises_with_no_positives():
    with pytest.raises(ValueError, match="no positive pixels"):
        pixel_auprc(np.zeros(10), np.random.rand(10))


def test_dice_coefficient_identical_masks_is_one():
    mask = np.array([[0, 1, 1], [0, 0, 1]])
    assert dice_coefficient(mask, mask) == pytest.approx(1.0)


def test_dice_coefficient_disjoint_masks_is_zero():
    a = np.array([1, 1, 0, 0])
    b = np.array([0, 0, 1, 1])
    assert dice_coefficient(a, b) == pytest.approx(0.0, abs=1e-6)


def _tiny_module(**kwargs):
    from drdetect.segmentation.model import build_segmentation_model
    from drdetect.segmentation.module import SegmentationModule

    model = build_segmentation_model("resnet18", pretrained=False, classes=1)
    return SegmentationModule(model, max_epochs=5, **kwargs)


def test_soft_dice_loss_identical_is_near_zero():
    from drdetect.segmentation.module import soft_dice_loss

    targets = torch.zeros(2, 1, 8, 8)
    targets[:, :, 2:6, 2:6] = 1.0
    logits = (targets * 2 - 1) * 20  # saturates sigmoid to ~match targets exactly
    loss = soft_dice_loss(logits, targets)
    assert loss.item() < 1e-3


def test_soft_dice_loss_disjoint_is_near_one():
    from drdetect.segmentation.module import soft_dice_loss

    targets = torch.zeros(2, 1, 8, 8)
    targets[:, :, 0:4, :] = 1.0
    pred = torch.zeros(2, 1, 8, 8)
    pred[:, :, 4:8, :] = 1.0
    logits = (pred * 2 - 1) * 20
    loss = soft_dice_loss(logits, targets)
    assert loss.item() > 0.9


def test_training_step_returns_finite_loss(monkeypatch):
    module = _tiny_module(pos_weight=5.0)
    monkeypatch.setattr(module, "log", lambda *a, **kw: None)
    x = torch.randn(2, 3, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.9).float()
    loss = module.training_step((x, y), batch_idx=0)
    assert torch.isfinite(loss)


def test_validation_epoch_end_logs_auprc_when_positives_present(monkeypatch):
    module = _tiny_module()
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.__setitem__(name, value))

    x = torch.randn(2, 3, 64, 64)
    y = torch.zeros(2, 1, 64, 64)
    y[0, 0, :4, :4] = 1.0  # guarantee at least one positive pixel in the val epoch
    module.validation_step((x, y), batch_idx=0)
    module.on_validation_epoch_end()

    assert "val/auprc" in logged
    assert "val/dice" in logged
    assert module._val_logits == []  # cleared after aggregation


def test_validation_epoch_end_skips_metrics_with_no_positives(monkeypatch):
    """An all-background validation epoch must not log a fabricated AUPRC --
    pixel_auprc itself raises on zero positives (see metrics.py), so silently
    skipping is the only honest option."""
    module = _tiny_module()
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.__setitem__(name, value))

    x = torch.randn(2, 3, 64, 64)
    y = torch.zeros(2, 1, 64, 64)
    module.validation_step((x, y), batch_idx=0)
    module.on_validation_epoch_end()

    assert "val/auprc" not in logged


def test_best_dice_threshold_finds_the_true_separating_value():
    """A score array cleanly separated at 0.6 -- the sweep should land near
    there and report the perfect Dice score achievable at that split."""
    from drdetect.segmentation.metrics import best_dice_threshold

    rng = np.random.default_rng(0)
    y_true = np.zeros(2000, dtype=int)
    y_true[:400] = 1
    y_score = np.empty(2000)
    y_score[:400] = rng.uniform(0.6, 1.0, size=400)
    y_score[400:] = rng.uniform(0.0, 0.6, size=1600)

    threshold, dice = best_dice_threshold(y_true, y_score)
    assert 0.5 < threshold < 0.7
    assert dice == pytest.approx(1.0, abs=1e-6)


def test_best_dice_threshold_beats_a_bad_fixed_threshold():
    """Reproduces the motivating case: a model whose scores sit mostly below
    0.5 even though its ranking is good. Dice@0.5 should read low; the tuned
    threshold should recover a much higher Dice from the SAME scores."""
    from drdetect.segmentation.metrics import best_dice_threshold, dice_coefficient

    rng = np.random.default_rng(1)
    y_true = np.zeros(2000, dtype=int)
    y_true[:400] = 1
    y_score = np.empty(2000)
    y_score[:400] = rng.uniform(0.20, 0.30, size=400)  # positives score low
    y_score[400:] = rng.uniform(0.0, 0.15, size=1600)  # negatives score lower

    dice_at_half = dice_coefficient(y_true, y_score > 0.5)
    threshold, _ = best_dice_threshold(y_true, y_score)
    dice_at_tuned = dice_coefficient(y_true, y_score > threshold)

    assert dice_at_half == pytest.approx(0.0, abs=1e-6)
    assert dice_at_tuned > 0.9
