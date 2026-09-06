"""Tests for the DRIVE vessel segmentation dataset, metrics, and module."""

from __future__ import annotations

import numpy as np
import pytest

from drdetect.segmentation.metrics import pixel_auroc
from drdetect.segmentation.vessels import DriveVesselDataset, DriveVesselPair, _pad_to_multiple

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("segmentation_models_pytorch")


@pytest.fixture
def synthetic_drive_pair(tmp_path):
    """A DRIVE-shaped (but tiny) image/mask/FOV triple -- 584x565 would make
    every test slow for no reason; the padding logic is resolution-independent."""
    import cv2

    rng = np.random.default_rng(0)
    h, w = 80, 70
    image = rng.integers(20, 200, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:60, 10:15] = 255  # a thin vessel-like line
    fov = np.zeros((h, w), dtype=np.uint8)
    fov[5:75, 5:65] = 255  # circular FOV approximated as a centered rectangle

    # .png here, not DRIVE's real .gif -- this OpenCV build can read gif but
    # cannot write it (fails with a dithering-kernel assertion); the dataset
    # class itself only ever calls cv2.imread on whatever path a
    # DriveVesselPair carries, so the extension used to build test fixtures
    # is independent of what production code (find_drive_vessel_pairs,
    # which does use the real .gif suffix) actually matches against.
    image_path = tmp_path / "21_training.tif"
    mask_path = tmp_path / "21_manual1.png"
    fov_path = tmp_path / "21_training_mask.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(fov_path), fov)
    return DriveVesselPair(
        image_id="21", image_path=image_path, mask_path=mask_path, fov_path=fov_path
    )


def test_pad_to_multiple_pads_up_to_the_next_multiple():
    arr = np.zeros((80, 70), dtype=np.uint8)
    out = _pad_to_multiple(arr, multiple=32, reflect=False)
    assert out.shape == (96, 96)


def test_pad_to_multiple_no_op_when_already_aligned():
    arr = np.zeros((64, 64), dtype=np.uint8)
    out = _pad_to_multiple(arr, multiple=32, reflect=False)
    assert out.shape == (64, 64)


def test_dataset_output_shapes_and_padding(synthetic_drive_pair):
    ds = DriveVesselDataset([synthetic_drive_pair], train=True)
    image, mask, fov = ds[0]
    # 80x70 pads to 96x96 (next multiple of 32 on each axis)
    assert image.shape == (3, 96, 96)
    assert mask.shape == (1, 96, 96)
    assert fov.shape == (1, 96, 96)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert set(fov.unique().tolist()) <= {0.0, 1.0}


def test_padded_region_is_excluded_from_fov(synthetic_drive_pair):
    """The whole point of zero-padding the FOV mask: padded pixels must never
    count as "inside the field of view", or the loss/metrics would train on
    synthetic border content."""
    ds = DriveVesselDataset([synthetic_drive_pair], train=False)
    _, _, fov = ds[0]
    assert (
        fov[0, -1, -1].item() == 0.0
    )  # bottom-right corner is padding for an 80x70 -> 96x96 image


def test_pixel_auroc_perfect_score_is_one():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert pixel_auroc(y_true, y_score) == pytest.approx(1.0)


def test_pixel_auroc_raises_with_only_one_class():
    with pytest.raises(ValueError, match="one class"):
        pixel_auroc(np.zeros(10), np.random.rand(10))


def _tiny_vessel_module(**kwargs):
    from drdetect.segmentation.model import build_segmentation_model
    from drdetect.segmentation.vessels import VesselSegmentationModule

    model = build_segmentation_model("resnet18", pretrained=False, classes=1)
    return VesselSegmentationModule(model, max_epochs=5, **kwargs)


def test_vessel_loss_ignores_pixels_outside_fov():
    """Two batches identical inside the FOV but different outside it must
    produce the identical loss -- otherwise the FOV mask isn't doing its job."""
    module = _tiny_vessel_module(pos_weight=3.0)
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 32, 32)
    targets = (torch.rand(1, 1, 32, 32) > 0.9).float()
    fov = torch.zeros(1, 1, 32, 32)
    fov[:, :, 8:24, 8:24] = 1.0

    targets_b = targets.clone()
    targets_b[:, :, :8, :] = 1.0  # change ground truth OUTSIDE the FOV only

    loss_a = module._loss(logits, targets, fov)
    loss_b = module._loss(logits, targets_b, fov)
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-6)


def test_vessel_training_step_returns_finite_loss(monkeypatch):
    module = _tiny_vessel_module(pos_weight=5.0)
    monkeypatch.setattr(module, "log", lambda *a, **kw: None)
    x = torch.randn(2, 3, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.9).float()
    fov = torch.ones(2, 1, 64, 64)
    loss = module.training_step((x, y, fov), batch_idx=0)
    assert torch.isfinite(loss)


def test_vessel_validation_epoch_end_logs_metrics(monkeypatch):
    module = _tiny_vessel_module()
    module.eval()  # BatchNorm in a resnet encoder needs eval mode at batch size 1
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.__setitem__(name, value))

    x = torch.randn(1, 3, 64, 64)
    y = torch.zeros(1, 1, 64, 64)
    y[0, 0, :10, :10] = 1.0
    fov = torch.ones(1, 1, 64, 64)
    module.validation_step((x, y, fov), batch_idx=0)
    module.on_validation_epoch_end()

    assert "val/auroc" in logged
    assert "val/dice" in logged
    assert module._val_probs == []
