"""Quadrant mapping from the OD-fovea axis.

Needed for ICDR's 4-2-1 rule (severe NPDR: haemorrhages in 4 quadrants, or
venous beading in 2, or IRMA in 1), which counts lesions per retinal
quadrant. This module defines the four quadrants geometrically, as the
roadmap specifies: one dividing line through the optic disc along the
OD-fovea axis, one perpendicular to it through the same point.

**Labelling deliberately avoids true anatomical nasal/temporal names.**
Whether the "away from fovea" side of the OD is nasal or temporal depends on
which eye (OD/OS) the image is of, and neither IDRiD nor this project's
other datasets carry reliable per-image eye-laterality labels to resolve
that. The labels used here -- "foveal-side" / "disc-side" for the axis-
parallel direction, "superior" / "inferior" for the perpendicular one --
describe the geometry precisely without asserting an anatomical mapping
this project cannot verify. A deployment with real laterality metadata
could remap these to true nasal/temporal in one place (`QUADRANT_LABELS`)
without touching the geometry.
"""

from __future__ import annotations

import numpy as np

__all__ = ["quadrant_axes", "assign_quadrant", "QUADRANT_LABELS"]

QUADRANT_LABELS = {
    (1, 1): "superior-foveal-side",
    (1, -1): "inferior-foveal-side",
    (-1, 1): "superior-disc-side",
    (-1, -1): "inferior-disc-side",
}


def quadrant_axes(
    od_xy: tuple[float, float], fovea_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (along_axis, perpendicular_axis) through the OD, where
    along_axis points from OD toward the fovea. Raises if OD and fovea
    coincide (undefined axis) -- that should never happen for real data and
    signals a genuine upstream bug (e.g. a broken localisation checkpoint)
    if it does.
    """
    od = np.asarray(od_xy, dtype=float)
    fovea = np.asarray(fovea_xy, dtype=float)
    axis = fovea - od
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        raise ValueError(f"OD and fovea coincide ({od_xy} == {fovea_xy}) -- cannot define an axis.")
    along = axis / norm
    # Perpendicular, image coordinates (y grows downward): rotate -90 degrees
    # so positive `perp` points toward smaller y (up the image, superior).
    perp = np.array([along[1], -along[0]])
    return along, perp


def assign_quadrant(
    point_xy: tuple[float, float], od_xy: tuple[float, float], fovea_xy: tuple[float, float]
) -> str:
    """Which of the 4 quadrants `point_xy` falls in, relative to the OD as
    origin and the OD-fovea line as one dividing axis. See module docstring
    for why the labels describe geometry, not asserted anatomy."""
    along, perp = quadrant_axes(od_xy, fovea_xy)
    od = np.asarray(od_xy, dtype=float)
    v = np.asarray(point_xy, dtype=float) - od

    along_sign = 1 if np.dot(v, along) >= 0 else -1
    perp_sign = 1 if np.dot(v, perp) >= 0 else -1
    return QUADRANT_LABELS[(along_sign, perp_sign)]
