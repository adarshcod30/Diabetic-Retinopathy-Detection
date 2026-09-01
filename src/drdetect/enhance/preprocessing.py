"""Fundus preprocessing primitives.

The pipeline order matters and is not arbitrary:

    crop_from_gray -> circle_crop -> ben_graham -> (CLAHE) -> resize

`crop_from_gray` first, because the black surround otherwise drags down every
subsequent statistic (the blur in `ben_graham`, the CLAHE histogram, the
illumination estimate). `ben_graham` before `resize`, because it is defined
relative to image radius and is cheapest to reason about at native scale.

Reference: Ben Graham's winning solution, Kaggle Diabetic Retinopathy 2015.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "crop_from_gray",
    "circle_crop",
    "ben_graham",
    "apply_clahe",
    "preprocess",
]


def crop_from_gray(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """Trim the uninformative dark border around a fundus photograph.

    Fundus cameras produce a bright circular retina on a black rectangle. The
    black region carries no signal but skews global statistics, so it goes first.

    Args:
        image: HxWx3 uint8 RGB, or HxW uint8 grayscale.
        tol: Intensity at or below which a pixel counts as background.

    Returns:
        The tightest crop containing above-threshold content. Returns the input
        unchanged if the mask is empty (a fully dark frame), rather than raising
        -- an all-dark image is a *quality* failure and is Stage 1's business,
        not this function's.
    """
    if image.ndim == 2:
        mask = image > tol
        if not mask.any():
            return image
        return image[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return image

    rows, cols = mask.any(1), mask.any(0)
    return image[np.ix_(rows, cols)]


def circle_crop(image: np.ndarray) -> np.ndarray:
    """Mask everything outside the inscribed circular field of view.

    Standardises the FOV across cameras so the model cannot learn to exploit
    corner artefacts that happen to correlate with a particular device.
    """
    image = crop_from_gray(image)
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(cx, cy)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, color=1, thickness=-1)

    out = image * mask[..., None] if image.ndim == 3 else image * mask
    return crop_from_gray(out)


def ben_graham(image: np.ndarray, sigma_scale: float = 30.0, weight: float = 4.0) -> np.ndarray:
    """Subtract local average colour (Ben Graham, Kaggle DR 2015).

        out = weight*img - weight*GaussianBlur(img, sigma) + 128

    The blur sigma is tied to *image radius*, not to a fixed pixel count, which
    makes the normalisation scale-invariant -- important because fundus cameras
    output wildly different resolutions.

    This removes the between-camera and between-patient variation in overall
    illumination and pigmentation while preserving the high-frequency detail
    that lesions live in.
    """
    radius = min(image.shape[:2]) / 2.0
    sigma = max(radius / sigma_scale, 1.0)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(image, weight, blurred, -weight, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the LAB L channel.

    Operating on L rather than per-RGB-channel avoids the colour casts that
    channel-wise CLAHE introduces.

    Note: `clip_limit` is a genuine risk knob. Too high and sensor noise is
    amplified into structures that look like microaneurysms. Ablate it; do not
    assume 2.0 is right for your camera mix.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess(
    image: np.ndarray,
    size: int = 512,
    use_ben_graham: bool = True,
    use_clahe: bool = False,
) -> np.ndarray:
    """Full preprocessing chain, resized to `size` x `size`.

    `use_clahe` defaults to False deliberately: it is a hypothesis to be tested
    in the ablation, not a default to be inherited. See docs/01_PROJECT_ANALYSIS.md
    section 6, Stage 2.
    """
    out = circle_crop(image)
    if use_ben_graham:
        out = ben_graham(out)
    if use_clahe:
        out = apply_clahe(out)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
