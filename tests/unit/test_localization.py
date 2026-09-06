"""Tests for IDRiD OD/fovea heatmap-regression localisation."""

from __future__ import annotations

import numpy as np
import pytest

from drdetect.segmentation.localization import (
    IDRiDLocalizationPair,
    heatmap_argmax,
    make_gaussian_heatmap,
    working_res_od_diameter,
)

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("segmentation_models_pytorch")


def test_gaussian_heatmap_peaks_at_the_given_center():
    heatmap = make_gaussian_heatmap(h=100, w=120, cx=80.0, cy=30.0, sigma=8.0)
    assert heatmap.shape == (100, 120)
    assert heatmap.max() == pytest.approx(1.0)
    x, y = heatmap_argmax(heatmap)
    assert (x, y) == (80.0, 30.0)


def test_heatmap_argmax_round_trips_for_several_points():
    for cx, cy in [(10.0, 10.0), (50.0, 90.0), (0.0, 0.0)]:
        heatmap = make_gaussian_heatmap(h=100, w=100, cx=cx, cy=cy, sigma=5.0)
        assert heatmap_argmax(heatmap) == (cx, cy)


def test_working_res_od_diameter_scales_linearly_with_height():
    full = working_res_od_diameter((2848, 4288))
    half = working_res_od_diameter((1424, 2144))
    assert half == pytest.approx(full / 2, rel=1e-6)


@pytest.fixture
def synthetic_localization_pair(tmp_path):
    import cv2

    rng = np.random.default_rng(0)
    image = rng.integers(20, 200, size=(200, 300, 3), dtype=np.uint8)
    image_path = tmp_path / "IDRiD_001.jpg"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return IDRiDLocalizationPair(
        image_id="IDRiD_001", image_path=image_path, od_xy=(220.0, 90.0), fovea_xy=(120.0, 100.0)
    )


def test_dataset_output_shapes_and_heatmap_placement(synthetic_localization_pair):
    from drdetect.segmentation.localization import IDRiDLocalizationDataset

    ds = IDRiDLocalizationDataset(
        [synthetic_localization_pair], size=(64, 96), train=False, sigma=4.0
    )
    image, heatmaps, targets_xy = ds[0]
    assert image.shape == (3, 64, 96)
    assert heatmaps.shape == (2, 64, 96)
    assert targets_xy.shape == (2, 2)
    # no augmentation (train=False) -- resize is deterministic, so the
    # heatmap peak must land within rounding distance of the resized target
    od_peak = heatmap_argmax(heatmaps[0].numpy())
    assert od_peak == pytest.approx(tuple(targets_xy[0].tolist()), abs=1.0)


def _tiny_localization_module(**kwargs):
    from drdetect.segmentation.localization import LocalizationModule
    from drdetect.segmentation.model import build_segmentation_model

    model = build_segmentation_model("resnet18", pretrained=False, classes=2)
    return LocalizationModule(model, max_epochs=5, working_od_diameter=10.0, **kwargs)


def test_localization_training_step_returns_finite_loss(monkeypatch):
    module = _tiny_localization_module()
    monkeypatch.setattr(module, "log", lambda *a, **kw: None)
    x = torch.randn(2, 3, 64, 96)
    heatmaps = torch.rand(2, 2, 64, 96)
    targets_xy = torch.tensor([[[10.0, 10.0], [50.0, 50.0]], [[20.0, 20.0], [40.0, 40.0]]])
    loss = module.training_step((x, heatmaps, targets_xy), batch_idx=0)
    assert torch.isfinite(loss)


def test_localization_validation_epoch_end_logs_distance_metrics(monkeypatch):
    module = _tiny_localization_module()
    module.eval()
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.__setitem__(name, value))

    x = torch.randn(1, 3, 64, 96)
    heatmaps = torch.rand(1, 2, 64, 96)
    targets_xy = torch.tensor([[[10.0, 10.0], [50.0, 50.0]]])
    module.validation_step((x, heatmaps, targets_xy), batch_idx=0)
    module.on_validation_epoch_end()

    assert "val/od_error_diameters" in logged
    assert "val/fovea_error_diameters" in logged
    assert "val/mean_error_diameters" in logged
    assert logged["val/mean_error_diameters"] == pytest.approx(
        (logged["val/od_error_diameters"] + logged["val/fovea_error_diameters"]) / 2.0
    )
    assert module._val_pred_heatmaps == []
