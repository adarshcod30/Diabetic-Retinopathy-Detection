"""Tests for OD-fovea-axis quadrant mapping."""

from __future__ import annotations

import numpy as np
import pytest

from drdetect.segmentation.quadrants import assign_quadrant, quadrant_axes


def test_quadrant_axes_are_unit_vectors_and_orthogonal():
    along, perp = quadrant_axes((100.0, 100.0), (200.0, 100.0))
    assert np.linalg.norm(along) == pytest.approx(1.0)
    assert np.linalg.norm(perp) == pytest.approx(1.0)
    assert np.dot(along, perp) == pytest.approx(0.0, abs=1e-9)


def test_quadrant_axes_raises_when_od_and_fovea_coincide():
    with pytest.raises(ValueError, match="coincide"):
        quadrant_axes((50.0, 50.0), (50.0, 50.0))


def test_along_axis_points_from_od_toward_fovea():
    along, _ = quadrant_axes((100.0, 100.0), (200.0, 100.0))
    assert along == pytest.approx([1.0, 0.0])


def test_all_four_quadrants_are_distinguishable():
    """Fovea due right of OD. Points placed unambiguously in each of the
    four geometric quadrants must all get different labels."""
    od, fovea = (100.0, 100.0), (200.0, 100.0)
    labels = {
        assign_quadrant((150.0, 50.0), od, fovea),  # foveal-side, up (smaller y)
        assign_quadrant((150.0, 150.0), od, fovea),  # foveal-side, down
        assign_quadrant((50.0, 50.0), od, fovea),  # disc-side, up
        assign_quadrant((50.0, 150.0), od, fovea),  # disc-side, down
    }
    assert len(labels) == 4


def test_assign_quadrant_matches_expected_label_for_a_known_geometry():
    od, fovea = (100.0, 100.0), (200.0, 100.0)  # axis points in +x
    # A point straight up (smaller y) and toward the fovea (+x) is
    # foveal-side + superior, given the module's stated convention that
    # smaller y (up the image) is "superior".
    assert assign_quadrant((150.0, 50.0), od, fovea) == "superior-foveal-side"
    assert assign_quadrant((150.0, 150.0), od, fovea) == "inferior-foveal-side"
    assert assign_quadrant((50.0, 50.0), od, fovea) == "superior-disc-side"
    assert assign_quadrant((50.0, 150.0), od, fovea) == "inferior-disc-side"


def test_assign_quadrant_is_rotation_consistent():
    """Same four-quadrant structure must hold regardless of the OD-fovea
    axis's absolute orientation, not just the axis-aligned case above."""
    od, fovea = (100.0, 100.0), (100.0, 200.0)  # axis points in +y this time
    labels = {
        assign_quadrant((150.0, 150.0), od, fovea),
        assign_quadrant((50.0, 150.0), od, fovea),
        assign_quadrant((150.0, 50.0), od, fovea),
        assign_quadrant((50.0, 50.0), od, fovea),
    }
    assert len(labels) == 4
