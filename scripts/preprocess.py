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
    p.add_argument("--dataset", default="aptos", choices=["aptos"])
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
    images_dir, csv_path = find_aptos(raw_root)
    labels = load_aptos_labels(csv_path)
    print(f"Found {len(labels)} labelled images under {images_dir}")

    out_dir = Path(args.out_dir) / f"{args.dataset}_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
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
            label=labels.get(r["image_id"], -1),
            group_id=groups[r["image_id"]],
            phash=str(r["dhash"]),
            width=r["width"],
            height=r["height"],
        )
        for r in results
    ]
    manifest = write_manifest(records, Path(args.manifest_dir) / f"{args.dataset}_{args.size}.csv")

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
