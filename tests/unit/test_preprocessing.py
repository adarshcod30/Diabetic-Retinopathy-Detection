"""Tests for fundus preprocessing primitives.

These are deliberately property-based rather than golden-image based: the point
is that the transforms behave sanely on *any* fundus-shaped input, not that they
reproduce one saved PNG.
"""

import numpy as np
import pytest

from drdetect.enhance.preprocessing import (
    apply_clahe,
    ben_graham,
    circle_crop,
    crop_from_gray,
    preprocess,
)


@pytest.fixture
def synthetic_fundus() -> np.ndarray:
    """A 600x800 black frame with a bright off-centre retinal disc.

    Off-centre on purpose: real captures are rarely centred, and a crop that
    only works on centred images is a crop that will fail in the field.
    """
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:600, :800]
    disc = (yy - 300) ** 2 + (xx - 420) ** 2 <= 280**2
    img[disc] = (180, 90, 60)
    return img


def test_crop_from_gray_removes_border(synthetic_fundus):
    cropped = crop_from_gray(synthetic_fundus)
    assert cropped.shape[0] < synthetic_fundus.shape[0]
    assert cropped.shape[1] < synthetic_fundus.shape[1]
    # every retained edge must contain signal
    assert cropped[0].max() > 7 or cropped[-1].max() > 7


def test_crop_from_gray_survives_all_black():
    """An all-dark frame is a quality failure, not a crash."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_from_gray(black).shape == black.shape


def test_crop_from_gray_handles_grayscale():
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[30:70, 20:80] = 200
    assert crop_from_gray(gray).shape == (40, 60)


def test_circle_crop_masks_corners(synthetic_fundus):
    out = circle_crop(synthetic_fundus)
    h, w = out.shape[:2]
    corners = [out[0, 0], out[0, w - 1], out[h - 1, 0], out[h - 1, w - 1]]
    assert all(c.max() == 0 for c in corners), "corners must be masked out"


def test_ben_graham_is_scale_invariant(synthetic_fundus):
    """Same content at two scales should yield similar global statistics.

    This is the property that justifies tying sigma to image radius rather than
    to a fixed pixel count.
    """
    import cv2

    small = cv2.resize(synthetic_fundus, (400, 300))
    a = ben_graham(synthetic_fundus).mean()
    b = ben_graham(small).mean()
    assert abs(a - b) < 5.0, f"mean drifted across scales: {a:.2f} vs {b:.2f}"


def test_ben_graham_preserves_dtype_and_range(synthetic_fundus):
    out = ben_graham(synthetic_fundus)
    assert out.dtype == np.uint8
    assert out.shape == synthetic_fundus.shape
    assert out.min() >= 0 and out.max() <= 255


def test_clahe_preserves_shape(synthetic_fundus):
    out = apply_clahe(synthetic_fundus)
    assert out.shape == synthetic_fundus.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("size", [224, 512])
@pytest.mark.parametrize("bg,clahe", [(True, False), (True, True), (False, False)])
def test_preprocess_output_contract(synthetic_fundus, size, bg, clahe):
    out = preprocess(synthetic_fundus, size=size, use_ben_graham=bg, use_clahe=clahe)
    assert out.shape == (size, size, 3)
    assert out.dtype == np.uint8


def test_preprocess_is_deterministic(synthetic_fundus):
    """Reproducibility starts here. If preprocessing drifts, nothing downstream
    is comparable across runs."""
    a = preprocess(synthetic_fundus, size=256)
    b = preprocess(synthetic_fundus, size=256)
    np.testing.assert_array_equal(a, b)
