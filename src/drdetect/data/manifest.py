"""Dataset manifests: the committed record of exactly which bytes were used.

Raw images are never committed (licence + size). A manifest is committed instead:
one row per image with its sha256, so anyone can verify they hold byte-identical
inputs before comparing results. This is DVC's job, done with a CSV.

Schema
------
image_id   stable identifier (dataset-unique)
path       path relative to the dataset root
sha256     content hash of the raw file
dataset    aptos | idrid | messidor2 | drive | eyepacs
label      ICDR grade 0-4, or -1 when unlabelled
group_id   patient/eye grouping key for leak-free splits, or "" if unknown
phash      structural perceptual hash (decimal string), or "" -- cached so
           regrouping does not require re-decoding every source image
width      pixels
height     pixels
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

__all__ = ["ImageRecord", "sha256_file", "write_manifest", "read_manifest", "verify_manifest"]

CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    sha256: str
    dataset: str
    label: int = -1
    group_id: str = ""
    phash: str = ""
    width: int = 0
    height: int = 0


def sha256_file(path: str | Path) -> str:
    """Content hash, streamed so a 90 GB dataset does not need 90 GB of RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(records: list[ImageRecord], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(ImageRecord)]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r.image_id):  # stable ordering
            writer.writerow(asdict(rec))
    return out_path


def read_manifest(path: str | Path) -> list[ImageRecord]:
    out: list[ImageRecord] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                ImageRecord(
                    image_id=row["image_id"],
                    path=row["path"],
                    sha256=row["sha256"],
                    dataset=row["dataset"],
                    label=int(row["label"]),
                    group_id=row["group_id"],
                    phash=row.get("phash", ""),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
    return out


def verify_manifest(
    manifest_path: str | Path, root: str | Path, *, sample: int | None = None
) -> tuple[list[str], list[str]]:
    """Re-hash files on disk and compare against the manifest.

    Returns (missing, mismatched) image_ids. `sample` limits the check to the
    first N records -- full verification of a large dataset is I/O bound.
    """
    root = Path(root)
    records = read_manifest(manifest_path)
    if sample is not None:
        records = records[:sample]

    missing, mismatched = [], []
    for rec in records:
        fpath = root / rec.path
        if not fpath.exists():
            missing.append(rec.image_id)
        elif sha256_file(fpath) != rec.sha256:
            mismatched.append(rec.image_id)
    return missing, mismatched
