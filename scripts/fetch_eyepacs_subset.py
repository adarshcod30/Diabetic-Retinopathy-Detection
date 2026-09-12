#!/usr/bin/env python3
"""Pull a stratified EyePACS subset for pretraining -- REMOTE INSTANCE ONLY.

Why this is a separate script, not part of download_data.sh
-------------------------------------------------------------
download_data.sh's own EyePACS branch refuses on purpose: the full
competition set is ~90GB and was never meant to touch this project's local
machine (docs/05_PROTOTYPE_SCOPE.md Sec.3.2). This script does the thing
that section actually proposed -- download the full set somewhere with
enough disk, stratify-sample ~15k images preserving the grade distribution,
keep only that -- just on a rented GPU instance instead of a Kaggle
notebook, since the machine running this now has its own real disk and
Kaggle CLI access rather than needing the notebook-specific workaround.

Run this ON the remote instance (via `jl run` or `jl exec`), never locally.
It refuses to run below --min-free-gb (default 60) as the same kind of
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


def download_and_extract(dest: Path, kaggle_bin: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zip_glob = list(dest.glob(f"{COMPETITION}*.zip"))
    if not zip_glob and not (dest / "train").exists():
        print(f"Downloading {COMPETITION} (this is the full ~90GB competition set)...")
        try:
            run([*kaggle_bin, "competitions", "download", "-c", COMPETITION, "-p", str(dest)])
        except subprocess.CalledProcessError as e:
            sys.exit(
                f"Download failed (exit {e.returncode}). If this is a 403, accept the "
                f"competition rules first at https://www.kaggle.com/c/{COMPETITION}/rules "
                "-- a valid API token alone is not sufficient for competition data."
            )
        zip_glob = list(dest.glob(f"{COMPETITION}*.zip"))

    if not (dest / "train").exists():
        print(f"Extracting {len(zip_glob)} archive(s)...")
        for z in zip_glob:
            run(["unzip", "-q", "-o", str(z), "-d", str(dest)])
        # Historically this competition's train images ship as further nested
        # per-part zips (train.zip.001 etc.) inside the outer download rather
        # than flat files -- extract anything that looks like it, once, and
        # fail loudly with what was actually found if the layout has changed
        # since this was written, rather than silently proceeding on 0 images.
        inner_zips = list(dest.rglob("train*.zip*"))
        for z in inner_zips:
            if z.suffix == ".zip":
                run(["unzip", "-q", "-o", str(z), "-d", str(dest)])
        if not (dest / "train").exists():
            candidates = sorted(p.name for p in dest.iterdir())
            sys.exit(
                f"Extraction did not produce a train/ directory under {dest}.\n"
                f"What's actually there: {candidates}\n"
                "The competition's packaging may have changed since this script was "
                "written -- inspect the archive contents and adjust extraction above."
            )


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
    p.add_argument("--min-free-gb", type=float, default=60.0)
    p.add_argument(
        "--keep-full-download",
        action="store_true",
        help="skip cleanup of the full ~90GB download after sampling (default: delete it)",
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
