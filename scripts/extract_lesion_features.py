#!/usr/bin/env python3
"""Run Phase 4's lesion models over an APTOS subset to build Phase 5's fusion features.

APTOS has DR grade labels but no lesion ground truth of its own -- every
feature here comes from running Phase 4's IDRiD-trained models as inference
engines, not from scoring against a mask. See
src/drdetect/fusion/features.py's module docstring for the cross-dataset
domain-shift caveat this implies, and why it was still judged worth building.

Deliberately APTOS only, not IDRiD: IDRiD is Phase 8's locked, evaluate-once
final test set (docs/04_ROADMAP.md) -- using its images (even just for
feature extraction on the way to fusion-head training) would contaminate it.

Processes one image at a time with explicit cleanup between images. This
project's own microaneurysm-pipeline debugging (docs/15) found that
uncollected full-resolution image arrays across a loop can compound into
severe memory pressure well before individual peak footprints look
alarming -- the same caution applies here, over more models and (usually)
larger images than that incident involved.

Usage:
    python scripts/extract_lesion_features.py --per-grade 40
    python scripts/extract_lesion_features.py --per-grade 5   # quick check
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pick_accelerator() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", default="data/manifests/aptos_512.csv")
    p.add_argument("--image-dir", default="data/raw/aptos/train_images")
    p.add_argument("--image-ext", default=".png")
    p.add_argument("--per-grade", type=int, default=40, help="images sampled per DR grade (0-4)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--localization-checkpoint",
        default="models/checkpoints/localization_resnet34_512x768_fold0/best.ckpt",
    )
    p.add_argument(
        "--hard-exudate-checkpoint",
        default="models/checkpoints/segmentation_hard_exudates_resnet34_512px_fold0/best.ckpt",
    )
    p.add_argument(
        "--soft-exudate-checkpoint",
        default="models/checkpoints/segmentation_soft_exudates_resnet34_512px_fold0/best.ckpt",
    )
    p.add_argument(
        "--haemorrhage-checkpoint",
        default="models/checkpoints/segmentation_haemorrhages_resnet34_512px_fold0/best.ckpt",
    )
    p.add_argument(
        "--microaneurysm-checkpoint",
        default="models/checkpoints/microaneurysm_classifier_fold0/best.ckpt",
    )
    p.add_argument("--out", default="data/manifests/aptos_lesion_features.csv")
    args = p.parse_args()

    import pandas as pd
    import torch

    from drdetect.fusion.features import FEATURE_NAMES, extract_lesion_features, load_lesion_models

    manifest = pd.read_csv(args.manifest)
    sample = pd.concat(
        [
            group.sample(n=min(args.per_grade, len(group)), random_state=args.seed)
            for _, group in manifest.groupby("label")
        ]
    ).reset_index(drop=True)
    print(f"sampled {len(sample)} images across grades {sorted(sample['label'].unique())}")

    device = torch.device(pick_accelerator())
    print(f"accelerator: {device}")
    models = load_lesion_models(
        localization_checkpoint=args.localization_checkpoint,
        hard_exudate_checkpoint=args.hard_exudate_checkpoint,
        soft_exudate_checkpoint=args.soft_exudate_checkpoint,
        haemorrhage_checkpoint=args.haemorrhage_checkpoint,
        microaneurysm_checkpoint=args.microaneurysm_checkpoint,
        device=device,
    )

    rows = []
    start = time.time()
    for _i, row in sample.iterrows():
        image_path = Path(args.image_dir) / f"{row['image_id']}{args.image_ext}"
        t0 = time.time()
        features = extract_lesion_features(image_path, models)
        elapsed = time.time() - t0
        rows.append({"image_id": row["image_id"], "grade": int(row["label"]), **features})
        print(
            f"[{len(rows)}/{len(sample)}] {row['image_id']} (grade {row['label']}, "
            f"{elapsed:.1f}s): HE={features['hard_exudate_mean_prob']:.4f} "
            f"SE={features['soft_exudate_mean_prob']:.4f} "
            f"HA={features['haemorrhage_mean_prob']:.4f} "
            f"MA={features['microaneurysm_count']:.0f}"
        )
        gc.collect()

    total_elapsed = time.time() - start
    print(f"\ndone in {total_elapsed:.1f}s ({total_elapsed / len(rows):.1f}s/image)")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(
        f"Saved: {out_path} ({len(out_df)} rows, columns: {['image_id', 'grade', *FEATURE_NAMES]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
