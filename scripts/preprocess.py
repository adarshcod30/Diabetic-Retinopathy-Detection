#!/usr/bin/env python3
"""Cache a dataset as preprocessed square images, and build its manifest.

Why this exists
---------------
Raw APTOS is ~10 GB of 3000x2000 JPEGs. Decoding those every epoch dominates
training time on a laptop, and 28 GB of free disk cannot hold several raw
datasets at once. Preprocessing ONCE to 512 px turns 10 GB into ~240 MB and
removes JPEG decode from the training loop.

It also does three things that matter for correctness later:

  * records a sha256 per source file, so results are tied to exact bytes;
  * computes a perceptual hash per image, which is how APTOS gets grouping
    information it does not ship (see drdetect.data.splits);
  * is resumable -- interrupting it and re-running skips completed files.

Usage:
    python scripts/preprocess.py --dataset aptos --size 512
    python scripts/preprocess.py --dataset aptos --size 512 --workers 4 --limit 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drdetect.data.manifest import ImageRecord, sha256_file, write_manifest  # noqa: E402
from drdetect.data.splits import perceptual_hash_groups, structural_hash  # noqa: E402
from drdetect.enhance.preprocessing import preprocess  # noqa: E402

JPEG_QUALITY = 95  # high enough that recompression does not erase microaneurysms


def find_aptos(root: Path) -> tuple[Path, Path]:
    """Locate APTOS train_images/ and train.csv, tolerating nesting from unzip."""
    for csv_path in root.rglob("train.csv"):
        images = csv_path.parent / "train_images"
        if images.is_dir():
            return images, csv_path
    raise FileNotFoundError(
        f"Could not find train.csv + train_images/ under {root}.\n"
        "Run: bash scripts/download_data.sh --datasets aptos"
    )


def load_aptos_labels(csv_path: Path) -> dict[str, int]:
    with open(csv_path, newline="") as fh:
        return {row["id_code"]: int(row["diagnosis"]) for row in csv.DictReader(fh)}


def find_messidor2(root: Path) -> tuple[Path, Path]:
    """Locate the extracted Messidor-2 images/ and the adjudicated grades CSV.

    Unlike APTOS, "download Messidor-2" is two independent requests to two
    different places (see docs/03_TECH_STACK.md Sec.5) -- the error message
    below repeats both so a missing half is never a mystery.
    """
    images = root / "images"
    csv_path = root / "messidor_data.csv"
    if images.is_dir() and csv_path.exists():
        return images, csv_path
    raise FileNotFoundError(
        f"Could not find images/ + messidor_data.csv under {root}.\n"
        "Images: request access at https://www.adcis.net/en/third-party/messidor2/ "
        "(personal-info form, manually reviewed), extract into data/raw/messidor2/images/\n"
        "Grades: kaggle datasets download -d google-brain/messidor2-dr-grades "
        "-p data/raw/messidor2, then unzip into data/raw/messidor2/"
    )


def find_idrid_grading(root: Path, split: str) -> tuple[Path, Path]:
    """Locate IDRiD's *disease-grading* images + labels (516 images total: 413
    train / 103 test) -- distinct from the 81-image segmentation subset
    already used in Phase 4/6, and sharing IDRiD's official 413/103 split
    already used for OD/fovea localisation (docs/12). 'test' is the locked
    external test set for Phase 8; do not preprocess 'train' for anything
    other than an explicitly non-tuning purpose.
    """
    folder = "a. Training Set" if split == "train" else "b. Testing Set"
    label_file = (
        "a. IDRiD_Disease Grading_Training Labels.csv"
        if split == "train"
        else "b. IDRiD_Disease Grading_Testing Labels.csv"
    )
    images = root / "B. Disease Grading" / "1. Original Images" / folder
    csv_path = root / "B. Disease Grading" / "2. Groundtruths" / label_file
    if images.is_dir() and csv_path.exists():
        return images, csv_path
    raise FileNotFoundError(
        f"Could not find {images} + {csv_path}.\n"
        "Expected the official IDRiD 'B. Disease Grading' folder under data/raw/idrid/."
    )


def load_idrid_grading_labels(csv_path: Path) -> dict[str, int]:
    """Filename stem -> ICDR grade. Source CSV has trailing empty columns and
    a trailing-space header ("Risk of macular edema "); only the grade column
    is used here."""
    labels = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Image name", "").strip()
            grade = row.get("Retinopathy grade", "").strip()
            if name and grade != "":
                labels[name.lower()] = int(grade)
    return labels


def load_messidor2_labels(csv_path: Path) -> dict[str, int]:
    """Filename stem (lower-cased) -> ICDR grade, gradable images only.

    Lower-cased because the archive mixes two real naming eras from Messidor's
    acquisition history -- 1058 images as `20051020_..._PP.png` and 690 as
    `IM004685.JPG` -- and the grades CSV spells the second group's extension
    lower-case (`.jpg`) where the archive ships it upper-case (`.JPG`). Case is
    the ONLY discrepancy: verified 1748/1748 of the CSV's rows resolve to a
    real file once case is normalised. 4 of 1748 rows have
    `adjudicated_gradable=0` and an empty grade -- excluded here rather than
    stored as label=-1, so a locked test set can never silently include an
    ungradable image under a placeholder label.
    """
    labels = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["adjudicated_gradable"] != "1":
                continue
            stem = Path(row["image_id"]).stem.lower()
            labels[stem] = int(row["adjudicated_dr_grade"])
    return labels


def process_one(args: tuple[Path, Path, int, bool, bool]) -> dict | None:
    """Worker: preprocess one image and return its record fields.

    Returns None on unreadable input rather than raising -- one corrupt file
    should not abort a 3,662-image run. Failures are counted and reported.
    """
    src, dst, size, use_ben_graham, use_clahe = args
    try:
        raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if raw is None:
            return None
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Grouping hash comes from the SOURCE structure, before Ben Graham
        # removes the low-frequency content that identifies an eye.
        group_hash = structural_hash(rgb)

        out = preprocess(rgb, size=size, use_ben_graham=use_ben_graham, use_clahe=use_clahe)

        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        return {
            "image_id": src.stem,
            "sha256": sha256_file(src),
            "width": w,
            "height": h,
            "dhash": group_hash,
        }
    except Exception as exc:  # noqa: BLE001 -- worker must not kill the pool
        return {"image_id": src.stem, "error": str(exc)}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="aptos", choices=["aptos", "messidor2", "idrid"])
    p.add_argument(
        "--idrid-split",
        default="test",
        choices=["train", "test"],
        help="IDRiD disease-grading partition (only used when --dataset idrid). "
        "'test' (103 images) is the locked external test set for Phase 8.",
    )
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--manifest-dir", default="data/manifests")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--no-ben-graham", action="store_true")
    p.add_argument("--clahe", action="store_true", help="off by default: a hypothesis to ablate")
    p.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    p.add_argument("--limit", type=int, default=None, help="process only N images (smoke test)")
    p.add_argument("--group-hash-distance", type=int, default=5)
    p.add_argument("--force", action="store_true", help="reprocess files already cached")
    args = p.parse_args()

    import os

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    raw_root = Path(args.raw_dir) / args.dataset
    if args.dataset == "messidor2":
        images_dir, csv_path = find_messidor2(raw_root)
        labels = load_messidor2_labels(csv_path)
    elif args.dataset == "idrid":
        images_dir, csv_path = find_idrid_grading(raw_root, args.idrid_split)
        labels = load_idrid_grading_labels(csv_path)
    else:
        images_dir, csv_path = find_aptos(raw_root)
        labels = load_aptos_labels(csv_path)
    print(f"Found {len(labels)} labelled images under {images_dir}")

    # IDRiD needs train vs. test disambiguated in cache/manifest filenames --
    # aptos and messidor2 have no such split, so their tag is just the dataset name.
    dataset_tag = f"idrid_{args.idrid_split}" if args.dataset == "idrid" else args.dataset

    out_dir = Path(args.out_dir) / f"{dataset_tag}_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Explicit-case globs rather than relying on a case-insensitive filesystem:
    # Messidor-2's archive ships 690 of its 1748 images as `.JPG` (upper-case),
    # and a case-sensitive filesystem (Linux CI, unlike default macOS/APFS)
    # would silently glob past all of them with a lower-case-only pattern.
    patterns = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    sources = sorted({p for pat in patterns for p in images_dir.glob(pat)})
    if args.dataset in ("messidor2", "idrid"):
        # Ungradable/unlabelled images have no label to preprocess towards --
        # and both are locked-test-set candidates, so they must be absent,
        # not present as label=-1.
        before = len(sources)
        sources = [s for s in sources if s.stem.lower() in labels]
        print(f"  excluding {before - len(sources)} unlabelled image(s) (no adjudicated grade)")
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"No images found in {images_dir}", file=sys.stderr)
        return 1

    tasks, skipped = [], 0
    for src in sources:
        dst = out_dir / f"{src.stem}.jpg"
        if dst.exists() and not args.force:
            skipped += 1
            continue
        tasks.append((src, dst, args.size, not args.no_ben_graham, args.clahe))

    print(
        f"{len(sources)} images | {skipped} already cached | {len(tasks)} to process "
        f"| {workers} workers | {args.size}px"
    )

    results, failures = [], []
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                if res is None or "error" in res:
                    failures.append(res["image_id"] if res else "unreadable")
                else:
                    results.append(res)
                if i % 200 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}  ({len(failures)} failed)", flush=True)

    # Re-derive records for already-cached files so the manifest stays complete.
    if skipped and not args.force:
        print("Re-hashing previously cached images for the manifest...")
        for src in sources:
            dst = out_dir / f"{src.stem}.jpg"
            if dst.exists() and src.stem not in {r["image_id"] for r in results}:
                # Re-read the SOURCE: the grouping hash cannot be recovered from
                # the cached image, because Ben Graham has removed the
                # low-frequency structure it depends on.
                raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if raw is None:
                    continue
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                results.append(
                    {
                        "image_id": src.stem,
                        "sha256": sha256_file(src),
                        "width": w,
                        "height": h,
                        "dhash": structural_hash(rgb),
                    }
                )

    if not results:
        print("Nothing processed.", file=sys.stderr)
        return 1

    # Sort before grouping. Results arrive from the process pool in completion
    # order, which is non-deterministic; feeding that order into grouping made
    # the split depend on scheduling. Group naming is now order-independent too,
    # but sorting keeps the manifest itself byte-stable across runs.
    results.sort(key=lambda r: r["image_id"])

    # APTOS ships no patient ids -- derive grouping from near-duplicate detection.
    print(
        f"Grouping {len(results)} images by perceptual hash (distance <= {args.group_hash_distance})..."
    )
    ids = [r["image_id"] for r in results]
    groups = perceptual_hash_groups(ids, [r["dhash"] for r in results], args.group_hash_distance)
    n_groups = len(set(groups.values()))
    print(
        f"  {len(ids)} images -> {n_groups} groups ({len(ids) - n_groups} near-duplicates merged)"
    )

    records = [
        ImageRecord(
            image_id=r["image_id"],
            path=f"{out_dir.name}/{r['image_id']}.jpg",
            sha256=r["sha256"],
            dataset=args.dataset,
            # .lower(): Messidor-2's grades CSV and its archive disagree on the
            # extension case for 690 of 1748 images (see load_messidor2_labels).
            # A no-op for APTOS, whose ids are already lower-case hex.
            label=labels.get(r["image_id"].lower(), -1),
            group_id=groups[r["image_id"]],
            phash=str(r["dhash"]),
            width=r["width"],
            height=r["height"],
        )
        for r in results
    ]
    manifest = write_manifest(records, Path(args.manifest_dir) / f"{dataset_tag}_{args.size}.csv")

    counts = np.bincount([r.label for r in records if r.label >= 0], minlength=5)
    cached_mb = sum(f.stat().st_size for f in out_dir.glob("*.jpg")) / 1024**2

    print(f"\nManifest: {manifest}")
    print(f"Cache:    {out_dir}  ({cached_mb:.0f} MB, {len(records)} images)")
    print("\nClass distribution:")
    for grade, name in enumerate(["No DR", "Mild", "Moderate", "Severe", "PDR"]):
        n = int(counts[grade])
        bar = "#" * int(40 * n / max(counts.max(), 1))
        print(f"  {grade} {name:<9} {n:>5} ({100 * n / max(counts.sum(), 1):>5.1f}%) {bar}")
    referable = int(counts[2:].sum())
    print(
        f"\n  Referable (grade >= 2): {referable} ({100 * referable / max(counts.sum(), 1):.1f}%)"
    )

    if failures:
        print(f"\n{len(failures)} images failed: {failures[:10]}", file=sys.stderr)
    if args.dataset == "messidor2" or (args.dataset == "idrid" and args.idrid_split == "test"):
        print(
            "\nThis is (part of) the LOCKED EXTERNAL TEST SET (Phase 8). Do not use it for "
            "training, threshold selection, or any tuning decision -- evaluate on it once, at the end."
        )
    else:
        print("\nNext: python scripts/train.py  (baseline)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        # Expected condition (data not downloaded yet), not a bug -- so report it
        # as a message rather than a traceback.
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None
