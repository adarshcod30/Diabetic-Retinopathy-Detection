#!/usr/bin/env python3
"""Train the microaneurysm candidate classifier on IDRiD.

Unlike the other four IDRiD lesion types, microaneurysms are not trained by
scripts/train_segmentation.py -- the roadmap (docs/04_ROADMAP.md, Phase 4)
specifies a candidate-then-classify pipeline instead of plain pixel
segmentation for this lesion type. This script:

  1. Runs classical candidate generation (black top-hat, see
     drdetect.segmentation.microaneurysms) over one KFold split's train/val
     images, matching each candidate against the ground-truth mask.
  2. Reports the generation-stage recall -- the ceiling the classifier can
     never exceed, since it only accepts or rejects proposed candidates.
  3. Trains a small CNN (drdetect.segmentation.microaneurysms.PatchClassifier)
     to accept/reject candidates, early-stopped on validation AUPRC.

Single train/val split by default (`--folds 0`, not the 5-fold CV the other
lesion types started with), consistent with the scoping decision made partway
through Phase 4 to manage total time across the remaining Phase 4/5/6 items.

Usage:
    python scripts/train_microaneurysm_classifier.py
    python scripts/train_microaneurysm_classifier.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Same reasoning as scripts/train.py and scripts/train_segmentation.py: MPS
# watermarks must be set before torch import or the allocator can
# over-request unified memory. See docs/05_PROTOTYPE_SCOPE.md 6.2.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pick_accelerator() -> str:
    import torch

    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--idrid-root", default="data/raw/idrid")
    p.add_argument("--disk-radius", type=int, default=16)
    p.add_argument("--response-percentile", type=float, default=95.0)
    p.add_argument("--patch-size", type=int, default=33)
    p.add_argument("--negative-ratio", type=float, default=10.0)
    p.add_argument("--max-negatives-per-image", type=int, default=300)
    p.add_argument("--folds", default="0", help="comma-separated fold indices, e.g. 0,1,2,3,4")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "PatchDataset serves pre-extracted in-memory patches, not files -- unlike "
            "IDRiDLesionDataset there's no per-item I/O to parallelise, so worker "
            "subprocesses (each re-importing torch/lightning/cv2 on macOS's spawn "
            "start method) cost memory without buying anything. 0 is the right default here."
        ),
    )
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="models/checkpoints")
    p.add_argument("--run-name", default="microaneurysm_classifier")
    p.add_argument("--smoke", action="store_true", help="2 epochs, 4 train / 2 val images")
    args = p.parse_args()

    import lightning as L
    import numpy as np
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader

    from drdetect.segmentation.dataset import find_idrid_lesion_pairs
    from drdetect.segmentation.microaneurysms import (
        ClassifierModule,
        PatchClassifier,
        PatchDataset,
        build_candidate_examples,
    )
    from drdetect.utils.seed import seed_everything

    seed_everything(args.seed)

    train_pairs = find_idrid_lesion_pairs(args.idrid_root, "microaneurysms", "train")
    if not train_pairs:
        print(
            f"No microaneurysm training pairs found under {args.idrid_root}.",
            file=sys.stderr,
        )
        return 1

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(train_pairs))

    accelerator = pick_accelerator()
    epochs = 2 if args.smoke else args.epochs
    print(f"accelerator : {accelerator}")
    print(f"epochs      : {epochs}{'  (SMOKE)' if args.smoke else ''}")
    print("27 official IDRiD test images are held out for every fold -- never used here")

    results = []
    for fold in [int(f) for f in args.folds.split(",")]:
        print(f"\n{'=' * 60}\nFold {fold}/{args.n_splits}\n{'=' * 60}")
        tr_idx, val_idx = splits[fold]
        tr_pairs = [train_pairs[i] for i in tr_idx]
        val_pairs = [train_pairs[i] for i in val_idx]
        if args.smoke:
            tr_pairs, val_pairs = tr_pairs[:4], val_pairs[:2]
        print(f"train images: {len(tr_pairs)}  val images: {len(val_pairs)}")

        common_kwargs = dict(
            disk_radius=args.disk_radius,
            response_percentile=args.response_percentile,
            patch_size=args.patch_size,
            negative_ratio=args.negative_ratio,
            max_negatives_per_image=args.max_negatives_per_image,
        )
        train_patches, train_labels, train_stats = build_candidate_examples(
            tr_pairs, seed=args.seed, **common_kwargs
        )
        val_patches, val_labels, val_stats = build_candidate_examples(
            val_pairs, seed=args.seed + 1, **common_kwargs
        )

        print(
            f"  train candidate-generation recall: {train_stats['generation_recall']:.4f} "
            f"({train_stats['n_recovered_instances']}/{train_stats['n_true_instances']} true "
            f"instances, {train_stats['n_candidates_total']} raw candidates)"
        )
        print(
            f"  val   candidate-generation recall: {val_stats['generation_recall']:.4f} "
            f"({val_stats['n_recovered_instances']}/{val_stats['n_true_instances']} true "
            f"instances, {val_stats['n_candidates_total']} raw candidates)"
        )
        print(
            f"  train examples kept: {train_stats['n_examples_kept']} "
            f"({train_stats['n_positive_candidates']} positive)  "
            f"val examples kept: {val_stats['n_examples_kept']} "
            f"({val_stats['n_positive_candidates']} positive)"
        )

        n_pos = max(train_stats["n_positive_candidates"], 1)
        n_neg = train_stats["n_examples_kept"] - train_stats["n_positive_candidates"]
        pos_weight = n_neg / n_pos
        print(f"  empirical pos_weight (neg:pos over kept training candidates): {pos_weight:.2f}")

        train_ds = PatchDataset(train_patches, train_labels, train=True)
        val_ds = PatchDataset(val_patches, val_labels, train=False)
        common = dict(batch_size=args.batch_size, num_workers=args.workers)
        train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
        val_dl = DataLoader(val_ds, shuffle=False, **common)

        model = PatchClassifier()
        module = ClassifierModule(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=pos_weight,
            max_epochs=epochs,
        )

        run_name = f"{args.run_name}_fold{fold}"
        out_dir = Path(args.out_dir) / run_name
        if out_dir.exists() and not args.smoke:
            print(
                f"\nERROR: {out_dir} already exists. Pass --run-name to use a new name "
                "or delete the directory first.",
                file=sys.stderr,
            )
            return 1

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            precision="32-true",
            deterministic=False,
            logger=CSVLogger(save_dir="runs", name=run_name, flush_logs_every_n_steps=10),
            callbacks=[
                ModelCheckpoint(
                    dirpath=out_dir, filename="best", monitor="val/auprc", mode="max", save_top_k=1
                ),
                EarlyStopping(
                    monitor="val/auprc", mode="max", patience=args.patience, min_delta=1e-3
                ),
            ],
            log_every_n_steps=10,
            enable_progress_bar=True,
        )
        trainer.fit(module, train_dl, val_dl)

        ckpt_cb = trainer.checkpoint_callback
        best_auprc = (
            float(ckpt_cb.best_model_score)
            if ckpt_cb is not None and ckpt_cb.best_model_score is not None
            else float("nan")
        )
        fold_result = {
            "fold": fold,
            "n_train_images": len(tr_pairs),
            "n_val_images": len(val_pairs),
            "train_generation_recall": train_stats["generation_recall"],
            "val_generation_recall": val_stats["generation_recall"],
            "pos_weight": pos_weight,
            "best_val_auprc": best_auprc,
            "epochs_run": trainer.current_epoch + 1,
            "best_checkpoint": str(ckpt_cb.best_model_path) if ckpt_cb else "",
        }
        results.append(fold_result)
        print(
            f"\nFold {fold}: best val/auprc={best_auprc:.4f} ({fold_result['epochs_run']} epochs)"
        )

    summary_path = Path("runs") / args.run_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    auprcs = [r["best_val_auprc"] for r in results]
    summary = {
        "run_name": args.run_name,
        "config": vars(args),
        "n_splits": args.n_splits,
        "folds": results,
        "val_auprc_mean": float(np.nanmean(auprcs)),
        "val_auprc_std": float(np.nanstd(auprcs)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"BEST val/auprc: {auprcs[0]:.4f}  ({results[0]['epochs_run']} epochs)")
    print(f"Summary: {summary_path}")
    for r in results:
        print(
            f"Next: python scripts/evaluate_microaneurysms.py --checkpoint {r['best_checkpoint']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
