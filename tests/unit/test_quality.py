"""Tests for the handcrafted quality gate.

Property-based against synthetic images, matching test_preprocessing.py: the
point is that each failure mode is actually detected, not that one saved
image scores a specific number.
"""

import cv2
import numpy as np
import pytest

from drdetect.quality.assessment import assess_quality


@pytest.fixture
def textured_fundus() -> np.ndarray:
    """A 600x800 frame with a bright, TEXTURED off-centre disc.

    Flat colour (as in test_preprocessing's fixture) has ~zero Laplacian
    variance regardless of focus, which would make every image "blurry" --
    texture is what makes sharpness measurable at all.
    """
    rng = np.random.default_rng(0)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:600, :800]
    disc = (yy - 300) ** 2 + (xx - 420) ** 2 <= 280**2
    speckle = rng.integers(0, 255, size=(600, 800), dtype=np.uint8)
    for c, base in enumerate((150, 80, 60)):
        channel = np.clip(base + speckle.astype(int) - 128, 0, 255).astype(np.uint8)
        img[..., c] = np.where(disc, channel, 0)
    return img


def test_sharp_textured_image_passes(textured_fundus):
    result = assess_quality(textured_fundus)
    assert result.usable, result.reasons


def test_blurry_image_is_rejected(textured_fundus):
    blurred = cv2.GaussianBlur(textured_fundus, (0, 0), sigmaX=15)
    result = assess_quality(blurred)
    assert not result.usable
    assert any("blurry" in r for r in result.reasons)


def test_underexposed_image_is_rejected(textured_fundus):
    dark = (textured_fundus.astype(float) * 0.05).astype(np.uint8)
    result = assess_quality(dark)
    assert not result.usable
    assert any("underexposed" in r or "near-black" in r for r in result.reasons)


def test_overexposed_image_is_rejected(textured_fundus):
    mask = textured_fundus.sum(axis=-1) > 0
    bright = textured_fundus.copy()
    bright[mask] = 255
    result = assess_quality(bright)
    assert not result.usable
    assert any("overexposed" in r or "blown out" in r for r in result.reasons)


def test_tiny_field_of_view_is_rejected():
    """A retina that only fills a small corner of the frame."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    rng = np.random.default_rng(1)
    speckle = rng.integers(0, 255, size=(60, 60), dtype=np.uint8)
    img[20:80, 20:80, 0] = speckle
    img[20:80, 20:80, 1] = speckle // 2
    result = assess_quality(img)
    assert not result.usable
    assert any("field of view" in r for r in result.reasons)


def test_all_black_image_does_not_crash():
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    result = assess_quality(black)
    assert not result.usable
    assert result.reasons


def test_reasons_lists_every_failure_not_just_first():
    """A dark AND blurry image should report both, not stop at the first hit."""
    rng = np.random.default_rng(2)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:600, :800]
    disc = (yy - 300) ** 2 + (xx - 420) ** 2 <= 280**2
    speckle = rng.integers(0, 40, size=(600, 800), dtype=np.uint8)
    img[..., 0] = np.where(disc, speckle, 0)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=15)
    result = assess_quality(blurred)
    assert not result.usable
    reasons_text = " ".join(result.reasons)
    assert "blurry" in reasons_text
    assert "underexposed" in reasons_text or "near-black" in reasons_text
