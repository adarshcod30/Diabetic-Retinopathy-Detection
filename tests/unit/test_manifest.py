"""Tests for dataset manifests -- the committed proof of which bytes were used."""

import pytest

from drdetect.data.manifest import (
    ImageRecord,
    read_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


@pytest.fixture
def sample_records():
    return [
        ImageRecord("b_img", "raw/b.png", "b" * 64, "aptos", 2, "g1", 512, 512),
        ImageRecord("a_img", "raw/a.png", "a" * 64, "aptos", 0, "g0", 512, 512),
    ]


def test_roundtrip_preserves_records(tmp_path, sample_records):
    path = write_manifest(sample_records, tmp_path / "m.csv")
    back = read_manifest(path)
    assert {r.image_id for r in back} == {"a_img", "b_img"}
    assert back[0].label == 0 and back[0].width == 512


def test_ordering_is_stable(tmp_path, sample_records):
    """Manifests are committed to git; unstable ordering means noisy diffs."""
    a = write_manifest(sample_records, tmp_path / "a.csv").read_text()
    b = write_manifest(list(reversed(sample_records)), tmp_path / "b.csv").read_text()
    assert a == b


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "x.bin"
    f.write_bytes(b"fundus" * 1000)
    assert sha256_file(f) == hashlib.sha256(b"fundus" * 1000).hexdigest()


def test_verify_detects_corruption_and_absence(tmp_path):
    root = tmp_path / "root"
    (root / "raw").mkdir(parents=True)
    good, bad = root / "raw" / "good.png", root / "raw" / "bad.png"
    good.write_bytes(b"intact")
    bad.write_bytes(b"intact")

    recs = [
        ImageRecord("good", "raw/good.png", sha256_file(good), "aptos"),
        ImageRecord("bad", "raw/bad.png", sha256_file(bad), "aptos"),
        ImageRecord("gone", "raw/gone.png", "0" * 64, "aptos"),
    ]
    mpath = write_manifest(recs, tmp_path / "m.csv")

    bad.write_bytes(b"CORRUPTED")  # silent bit-rot / wrong re-download
    missing, mismatched = verify_manifest(mpath, root)
    assert missing == ["gone"]
    assert mismatched == ["bad"]


def test_clean_dataset_verifies_empty(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    f = root / "a.png"
    f.write_bytes(b"data")
    mpath = write_manifest([ImageRecord("a", "a.png", sha256_file(f), "aptos")], tmp_path / "m.csv")
    assert verify_manifest(mpath, root) == ([], [])
