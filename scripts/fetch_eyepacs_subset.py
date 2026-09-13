#!/usr/bin/env python3
"""Pull a stratified EyePACS subset for pretraining -- REMOTE INSTANCE ONLY.

Why the images come from a re-hosted Kaggle DATASET, not the competition zip
-----------------------------------------------------------------------------
The original competition (`diabetic-retinopathy-detection`) ships train
images as a genuine multi-disk PKZIP split (train.zip.001-005): each
volume's local file headers are offset relative to THAT volume, not to one
continuous stream. Verified directly against the real files: a plain `cat`
of the 5 parts produces "invalid zip file with overlapped components", and
both Info-ZIP's own `-s 0`/`-s-` split-join and a custom Python multi-part
virtual stream reader correctly parse the central directory (35,127 real
entries) but fail on a genuine per-file read with a seek/offset error --
Python's zipfile module does not resolve true multi-disk offsets, and
Info-ZIP 3.0 (2008) has known Zip64-split-join bugs. Rather than hand-write
a multi-disk PKZIP offset resolver, this uses `tanlikesmath/
diabetic-retinopathy-resized` (589 votes, since 2019) -- a well-established
community re-host of the SAME images under the SAME filenames
(`<patient>_<left|right>.jpeg`), as one normally-packaged archive. The
labels still come from the original competition's own trainLabels.csv
(a small, separate, single-file download that worked fine from the start)
-- only the multi-disk-split image archive is swapped out.

Why this is a separate script, not part of download_data.sh
-------------------------------------------------------------
download_data.sh's own EyePACS branch refuses on purpose: the full
competition set is ~90GB (train+test+extras) and was never meant to touch
this project's local machine (docs/05_PROTOTYPE_SCOPE.md Sec.3.2). This
script does the thing that section actually proposed -- download the full
train set somewhere with enough disk, stratify-sample ~15k images
preserving the grade distribution, keep only that -- just on a rented GPU
instance instead of a Kaggle notebook, since the machine running this now
has its own real disk and Kaggle CLI access rather than needing the
notebook-specific workaround.

Run this ON the remote instance (via `jl run` or `jl exec`), never locally.
It refuses to run below --min-free-gb (default 30, sized for the ~7.8GB
image re-host download plus its own ~7.8GB extracted copy existing at once)
as the same kind of
guard download_data.sh already uses, so a mistaken local run fails loudly
instead of quietly filling a laptop's disk.

Usage (on the remote instance, after `jl upload`-ing this repo and its
kaggle.json):
    python scripts/fetch_eyepacs_subset.py --n-subset 15000

Leaves data/raw/eyepacs/{train/,trainLabels.csv} in exactly the layout
scripts/preprocess.py --dataset eyepacs expects, then deletes the full
downloaded/extracted set -- only the subset persists.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

COMPETITION = "diabetic-retinopathy-detection"
N_GRADES = 5


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def resolve_kaggle() -> list[str]:
    """Mirror download_data.sh's resolve_kaggle(): prefer the project's own
    venv over a global install, so this behaves the same locally and on a
    freshly-provisioned remote instance either way."""
    venv_kaggle = Path(".venv/bin/kaggle")
    if venv_kaggle.exists():
        return [str(venv_kaggle)]
    if shutil.which("kaggle"):
        return ["kaggle"]
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        check = subprocess.run([str(venv_python), "-c", "import kaggle"], capture_output=True)
        if check.returncode == 0:
            return [str(venv_python), "-m", "kaggle"]
    sys.exit(
        "ERROR: kaggle CLI not found. Install the environment first (pip install -e .) "
        "or: pip install kaggle"
    )


def find_kaggle_credentials() -> None:
    for p in (Path.home() / ".config/kaggle/kaggle.json", Path.home() / ".kaggle/kaggle.json"):
        if p.exists():
            return
    sys.exit(
        "ERROR: no Kaggle credentials found on this instance.\n"
        "Upload yours from the local machine first:\n"
        "  jl upload <id> ~/.config/kaggle/kaggle.json .config/kaggle/kaggle.json\n"
        "  jl exec <id> -- chmod 600 ~/.config/kaggle/kaggle.json"
    )


LABELS_FILE = "trainLabels.csv.zip"
IMAGES_DATASET = "tanlikesmath/diabetic-retinopathy-resized"
IMAGES_ARCHIVE = "diabetic-retinopathy-resized.zip"
# The archive's own internal layout, verified via `kaggle datasets files`
# (2026-09-12): images sit two directories deep, not flat.
IMAGES_NESTED_DIR = "resized_train/resized_train"


def download_and_extract(dest: Path, kaggle_bin: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if not (dest / "trainLabels.csv").exists():
        if not (dest / LABELS_FILE).exists():
            print(f"Downloading {LABELS_FILE} (labels, from the original competition)...")
            try:
                run(
                    [
                        *kaggle_bin,
                        "competitions",
                        "download",
                        "-c",
                        COMPETITION,
                        "-f",
                        LABELS_FILE,
                        "-p",
                        str(dest),
                    ]
                )
            except subprocess.CalledProcessError as e:
                sys.exit(
                    f"Download of {LABELS_FILE} failed (exit {e.returncode}). If this is a "
                    f"403, accept the competition rules first at "
                    f"https://www.kaggle.com/c/{COMPETITION}/rules -- a valid API token alone "
                    "is not sufficient for competition data."
                )
        run(["unzip", "-q", "-o", str(dest / LABELS_FILE), "-d", str(dest)])

    if not (dest / "train").exists():
        archive = dest / IMAGES_ARCHIVE
        if not archive.exists():
            print(f"Downloading {IMAGES_DATASET} (images, ~7.8GB re-host of the same files)...")
            try:
                run([*kaggle_bin, "datasets", "download", "-d", IMAGES_DATASET, "-p", str(dest)])
            except subprocess.CalledProcessError as e:
                sys.exit(f"Download of {IMAGES_DATASET} failed (exit {e.returncode}).")
        print("Extracting images...")
        run(["unzip", "-q", "-o", str(archive), "-d", str(dest)])
        archive.unlink()
        nested = dest / IMAGES_NESTED_DIR
        if not nested.is_dir():
            candidates = sorted(p.name for p in dest.iterdir())
            sys.exit(
                f"Expected images under {nested} after extraction.\n"
                f"What's actually there: {candidates}\n"
                f"{IMAGES_DATASET}'s own layout may have changed since this script was "
                "written -- inspect the archive and adjust IMAGES_NESTED_DIR above."
            )
        nested.rename(dest / "train")
        # resized_train/ (now empty except the moved-out subdir) and the
        # labels zip are no longer needed once train/ exists.
        (dest / "resized_train").rmdir()
        (dest / LABELS_FILE).unlink(missing_ok=True)


def stratified_sample(labels_csv: Path, n_subset: int, seed: int) -> dict[str, int]:
    """Proportional-to-class-frequency sample, same grade distribution as the
    full set -- not a naive random sample, which would under-represent the
    already-rare severe/PDR grades this project's whole APTOS experience
    (docs/06) shows are the hard classes to begin with."""
    import random

    by_grade: dict[int, list[str]] = defaultdict(list)
    with open(labels_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            by_grade[int(row["level"])].append(row["image"])

    total = sum(len(v) for v in by_grade.values())
    rng = random.Random(seed)
    sampled: dict[str, int] = {}
    print(f"Full set: {total} labelled images. Sampling {n_subset}, stratified by grade:")
    for grade in range(N_GRADES):
        pool = by_grade.get(grade, [])
        take = round(n_subset * len(pool) / total) if total else 0
        take = min(take, len(pool))
        chosen = rng.sample(pool, take)
        for image_id in chosen:
            sampled[image_id] = grade
        pct = 100 * len(pool) / max(total, 1)
        print(f"  grade {grade}: {len(pool):>6} available ({pct:4.1f}%) -> {take:>5} sampled")
    return sampled


def materialise_subset(full_images_dir: Path, sampled: dict[str, int], out_root: Path) -> None:
    out_images = out_root / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    missing = 0
    for image_id in sampled:
        src = full_images_dir / f"{image_id}.jpeg"
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, out_images / src.name)
    if missing:
        print(f"  WARNING: {missing} sampled ids had no matching .jpeg under {full_images_dir}")

    with open(out_root / "trainLabels.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "level"])
        for image_id, grade in sorted(sampled.items()):
            writer.writerow([image_id, grade])

    kept = len(list(out_images.glob("*.jpeg")))
    size_mb = sum(f.stat().st_size for f in out_images.glob("*.jpeg")) / 1024**2
    print(f"Subset materialised: {kept} images, {size_mb:.0f} MB, at {out_root}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-subset", type=int, default=15000, help="Tier-P default (docs/05 Sec.3.2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--min-free-gb", type=float, default=30.0)
    p.add_argument(
        "--keep-full-download",
        action="store_true",
        help="skip cleanup of the downloaded images archive after sampling (default: delete it)",
    )
    args = p.parse_args()

    free = free_gb(Path("."))
    if free < args.min_free_gb:
        sys.exit(
            f"ERROR: only {free:.0f}GB free, need >= {args.min_free_gb:.0f}GB.\n"
            "This downloads the FULL EyePACS competition set before sampling -- "
            "if this is your local machine, stop: this is remote-instance-only "
            "(see docs/05_PROTOTYPE_SCOPE.md Sec.3.2, docs/07_PHASE3_RESULTS.md)."
        )
    print(f"Disk check: {free:.0f}GB free (>= {args.min_free_gb:.0f}GB required) [ok]")

    kaggle_bin = resolve_kaggle()
    find_kaggle_credentials()

    full_dir = Path(args.raw_dir) / "_eyepacs_full_download"
    download_and_extract(full_dir, kaggle_bin)

    labels_csv = full_dir / "trainLabels.csv"
    if not labels_csv.exists():
        sys.exit(f"trainLabels.csv not found under {full_dir} after extraction.")

    sampled = stratified_sample(labels_csv, args.n_subset, args.seed)

    out_root = Path(args.raw_dir) / "eyepacs"
    materialise_subset(full_dir / "train", sampled, out_root)

    if not args.keep_full_download:
        freed = sum(f.stat().st_size for f in full_dir.rglob("*") if f.is_file()) / 1024**3
        print(f"Removing full download ({freed:.1f}GB) -- pass --keep-full-download to skip this.")
        shutil.rmtree(full_dir)

    print("\nNext: python scripts/preprocess.py --dataset eyepacs --size 512")
    return 0


if __name__ == "__main__":
    sys.exit(main())
