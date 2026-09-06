"""Tests for the pure aggregation logic in lesion feature extraction.

Model loading and inference (`load_lesion_models`, `extract_lesion_features`)
need real checkpoints and are exercised by scripts/extract_lesion_features.py
directly rather than mocked here -- these tests cover the deterministic,
model-free logic: candidate/centroid extraction from a probability map.
"""

from __future__ import annotations

import numpy as np
import pytest

from drdetect.fusion.features import FEATURE_NAMES, QUADRANT_ORDER, lesion_centroids


def test_lesion_centroids_finds_a_single_blob():
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[40:50, 60:70] = 0.9  # a 10x10 blob, area 100 >= the min-area floor

    centroids = lesion_centroids(prob_map, threshold=0.5)

    assert len(centroids) == 1
    cx, cy = centroids[0]
    assert cx == pytest.approx(64.5, abs=1.0)
    assert cy == pytest.approx(44.5, abs=1.0)


def test_lesion_centroids_finds_multiple_distinct_blobs():
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[10:15, 10:15] = 0.9
    prob_map[80:85, 80:85] = 0.9

    centroids = lesion_centroids(prob_map, threshold=0.5)

    assert len(centroids) == 2


def test_lesion_centroids_ignores_speckle_below_min_area():
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[50, 50] = 0.9  # a single pixel, area 1

    centroids = lesion_centroids(prob_map, threshold=0.5)

    assert centroids == []


def test_lesion_centroids_returns_empty_for_an_all_background_map():
    prob_map = np.zeros((50, 50), dtype=np.float32)
    assert lesion_centroids(prob_map, threshold=0.5) == []


def test_lesion_centroids_respects_the_threshold():
    prob_map = np.full((100, 100), 0.3, dtype=np.float32)
    prob_map[40:50, 40:50] = 0.6

    assert lesion_centroids(prob_map, threshold=0.5) != []
    assert lesion_centroids(prob_map, threshold=0.7) == []


def test_feature_names_cover_every_quadrant_exactly_once():
    quadrant_features = [f for f in FEATURE_NAMES if f.startswith("quadrant_frac_")]
    assert len(quadrant_features) == len(QUADRANT_ORDER) == 4
    assert {f.removeprefix("quadrant_frac_") for f in quadrant_features} == set(QUADRANT_ORDER)
