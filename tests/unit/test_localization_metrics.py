"""Tests for CAM-vs-mask localisation metrics (pointing game + IoU)."""

from __future__ import annotations

import numpy as np
import pytest

from drdetect.explain.localization_metrics import cam_mask_iou, pointing_game_hit, upsample_cam


def test_pointing_game_hit_when_cam_peak_is_inside_the_mask():
    cam = np.zeros((10, 10))
    cam[5, 5] = 1.0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:7, 4:7] = 1

    assert pointing_game_hit(cam, mask) is True


def test_pointing_game_miss_when_cam_peak_is_outside_the_mask():
    cam = np.zeros((10, 10))
    cam[0, 0] = 1.0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[8:10, 8:10] = 1

    assert pointing_game_hit(cam, mask) is False


def test_cam_mask_iou_is_one_for_identical_regions():
    cam = np.zeros((10, 10))
    cam[2:5, 2:5] = 0.9
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1

    assert cam_mask_iou(cam, mask, threshold=0.5) == pytest.approx(1.0)


def test_cam_mask_iou_is_zero_for_disjoint_regions():
    cam = np.zeros((10, 10))
    cam[0:2, 0:2] = 0.9
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[8:10, 8:10] = 1

    assert cam_mask_iou(cam, mask, threshold=0.5) == pytest.approx(0.0)


def test_cam_mask_iou_is_nan_when_both_are_empty():
    cam = np.zeros((10, 10))
    mask = np.zeros((10, 10), dtype=np.uint8)

    assert np.isnan(cam_mask_iou(cam, mask, threshold=0.5))


def test_cam_mask_iou_partial_overlap():
    cam = np.zeros((10, 10))
    cam[0:4, 0:4] = 0.9  # 16 px
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 1  # 16 px, overlapping [2:4, 2:4] = 4 px

    # intersection 4, union 16+16-4=28
    assert cam_mask_iou(cam, mask, threshold=0.5) == pytest.approx(4 / 28)


def test_upsample_cam_resizes_to_target_shape():
    cam = np.random.default_rng(0).random((16, 24)).astype(np.float32)
    upsampled = upsample_cam(cam, target_shape=(160, 240))
    assert upsampled.shape == (160, 240)


def test_upsample_cam_preserves_the_location_of_a_peak():
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[2, 8] = 1.0  # near the top-right
    upsampled = upsample_cam(cam, target_shape=(100, 100))
    y, x = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    assert y < 40  # still in the top half
    assert x > 60  # still in the right half
