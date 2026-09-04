"""Handcrafted fundus image quality gate.

This is the "handcrafted quality features" half of Phase 2's quality stage
(docs/04_ROADMAP.md), not the learned half. The roadmap also calls for an
EfficientNet-B0 classifier trained on EyeQ; that needs a new dataset download
and a training run and is deliberately deferred. Everything here is an
uncalibrated heuristic, not a validated clinical cutoff -- the thresholds are
defensible starting points from classical blur/exposure-detection practice,
not numbers measured against this project's own data. Replace them with the
EyeQ classifier before trusting this gate on real field images, the same way
`apply_clahe`'s clip_limit is flagged as a knob to ablate, not a fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from drdetect.enhance.preprocessing import circle_crop, crop_from_gray

__all__ = ["QualityResult", "assess_quality"]

# Laplacian variance below this, on the FOV resized to a fixed working size,
# reads as blurry. Variance of the Laplacian scales with image size and FOV
# fraction, which is why the FOV is isolated and resized before measuring it
# -- otherwise the same physical sharpness would score differently depending
# on the camera's raw resolution.
#
# 80.0 (a generic blur-detection tutorial default) was the first value tried
# here, and it rejected 28 of 30 real APTOS training images -- images that
# were sharp enough for an ophthalmologist to grade. That default was tuned
# for a different normalisation scale, not this one. 15.0 is recalibrated
# against this project's own pipeline on a 60-image APTOS sample (measured
# range 7-108, with a visible cluster of genuinely-degraded outliers at 7-13
# and a continuous spread from ~17 up); it is still an eyeballed cut on a
# small sample, not the learned EyeQ classifier the roadmap calls for.
_BLUR_WORKING_SIZE = 512
_MIN_SHARPNESS = 15.0

# Mean intensity of the FOV, on a 0-255 scale. Outside this band the image is
# globally under- or over-exposed rather than locally noisy. Lower bound has
# a margin below the measured real-data minimum (36 on the same 60-image
# sample); upper bound is untested against real overexposed captures.
_MIN_MEAN_BRIGHTNESS = 30.0
_MAX_MEAN_BRIGHTNESS = 225.0

# Fraction of FOV pixels that are crushed (near-black) or blown out
# (near-white). High values mean local exposure failure even when the mean
# brightness looks fine.
_MAX_DARK_FRACTION = 0.45
_MAX_BRIGHT_FRACTION = 0.15

# Fraction of the full frame the detected FOV must occupy. Too small usually
# means the camera missed the retina (misalignment, eyelash occlusion) rather
# than a genuinely small pupil.
_MIN_FOV_FRACTION = 0.25


@dataclass(frozen=True)
class QualityResult:
    usable: bool
    reasons: list[str] = field(default_factory=list)
    sharpness: float = 0.0
    mean_brightness: float = 0.0
    dark_fraction: float = 0.0
    bright_fraction: float = 0.0
    fov_fraction: float = 0.0


def assess_quality(image: np.ndarray) -> QualityResult:
    """Gate a raw (unpreprocessed) fundus photo before grading.

    Args:
        image: HxWx3 uint8 RGB, as captured -- not yet Ben Graham-normalised.
            Judging exposure and blur on the enhanced image would be
            circular: `ben_graham` actively rewrites local contrast.

    Returns:
        QualityResult with `usable=False` and `reasons` naming every check
        that failed, not just the first -- a rejected image should tell the
        operator what to fix on recapture, not just that something is wrong.
    """
    fov = crop_from_gray(image)
    if fov.size == 0 or min(fov.shape[:2]) < 8:
        return QualityResult(usable=False, reasons=["no retinal field of view detected"])

    gray = cv2.cvtColor(fov, cv2.COLOR_RGB2GRAY)

    working = cv2.resize(
        gray, (_BLUR_WORKING_SIZE, _BLUR_WORKING_SIZE), interpolation=cv2.INTER_AREA
    )
    sharpness = float(cv2.Laplacian(working, cv2.CV_64F).var())

    mean_brightness = float(gray.mean())
    dark_fraction = float((gray < 20).mean())
    bright_fraction = float((gray > 245).mean())

    circle = circle_crop(image)
    fov_fraction = float((cv2.cvtColor(circle, cv2.COLOR_RGB2GRAY) > 0).sum()) / (
        image.shape[0] * image.shape[1]
    )

    reasons = []
    if sharpness < _MIN_SHARPNESS:
        reasons.append(f"image too blurry (sharpness {sharpness:.0f} < {_MIN_SHARPNESS:.0f})")
    if mean_brightness < _MIN_MEAN_BRIGHTNESS:
        reasons.append(f"underexposed (mean brightness {mean_brightness:.0f})")
    if mean_brightness > _MAX_MEAN_BRIGHTNESS:
        reasons.append(f"overexposed (mean brightness {mean_brightness:.0f})")
    if dark_fraction > _MAX_DARK_FRACTION:
        reasons.append(f"too much of the field is near-black ({dark_fraction:.0%})")
    if bright_fraction > _MAX_BRIGHT_FRACTION:
        reasons.append(f"too much of the field is blown out ({bright_fraction:.0%})")
    if fov_fraction < _MIN_FOV_FRACTION:
        reasons.append(f"retinal field of view too small ({fov_fraction:.0%} of frame)")

    return QualityResult(
        usable=not reasons,
        reasons=reasons,
        sharpness=sharpness,
        mean_brightness=mean_brightness,
        dark_fraction=dark_fraction,
        bright_fraction=bright_fraction,
        fov_fraction=fov_fraction,
    )
