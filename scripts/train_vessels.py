#!/usr/bin/env python3
"""Train a vessel segmentation model on DRIVE.

Phase 4 target: Dice ~=0.80+, AUC ~=0.97+ (the roadmap's own numbers for this
item). DRIVE's 20 labelled ("training") images are k-fold cross-validated --
see src/drdetect/segmentation/vessels.py for why there is no separate locked
test split here the way IDRiD hard exudates has one (DRIVE's public "test"
images ship with no vessel ground truth at all).

Usage:
    python scripts/train_vessels.py                       # fold 0 only
    python scripts/train_vessels.py --folds 0,1,2,3,4       # full 5-fold CV
    python scripts/train_vessels.py --smoke                 # 2 epochs, 4 images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def estimate_pos_weight(pairs, n_samples: int) -> float:
    """Empirical neg:pos ratio over FOV-masked pixels of actual training
    images. Full images here (no patch sampling, unlike IDRiD lesions), so
    this is the real training-time ratio, not an approximation of one."""
    import cv2
    import numpy as np

    pos = 0.0
    total = 0.0
    for pair in pairs[:n_samples]:
        mask = (cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        fov = (cv2.imread(str(pair.fov_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        pos += float((mask & fov).sum())
        total += float(fov.sum())
    neg = total - pos
    if pos == 0:
        raise ValueError(
            f"0 positive pixels across {len(pairs[:n_samples])} images -- check the data."
        )
    return neg / pos


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--drive-root", default="data/raw/drive")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--folds", default="0", help="comma-separated fold indices, e.g. 0,1,2,3,4")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience in epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="models/checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true", help="2 epochs, 4 train / 2 val images")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from last.ckpt if present (optimiser, LR schedule and epoch are restored)",
    )
    args = p.parse_args()

    import lightning as L
    import numpy as np
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader

    from drdetect.segmentation.model import build_segmentation_model
    from drdetect.segmentation.vessels import (
        DriveVesselDataset,
        VesselSegmentationModule,
        find_drive_vessel_pairs,
    )
    from drdetect.utils.seed import seed_everything, worker_init_fn

    seed_everything(args.seed)

    pairs = find_drive_vessel_pairs(args.drive_root)
    if not pairs:
        print(f"No DRIVE vessel pairs found under {args.drive_root}.", file=sys.stderr)
        return 1
    if len(pairs) != 20:
        print(f"WARNING: expected 20 DRIVE training pairs, found {len(pairs)}.", file=sys.stderr)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(pairs))

    accelerator = pick_accelerator()
    epochs = 2 if args.smoke else args.epochs
    run_name = args.run_name or f"vessels_{args.encoder}"

    print(f"accelerator : {accelerator}")
    print(f"encoder     : {args.encoder}, batch {args.batch_size}")
    print(f"epochs      : {epochs}{'  (SMOKE)' if args.smoke else ''}")
    print(
        f"{len(pairs)} DRIVE images with public vessel ground truth -- no separate locked test set exists"
    )

    results = []
    for fold in [int(f) for f in args.folds.split(",")]:
        print(f"\n{'=' * 60}\nFold {fold}/{args.n_splits}\n{'=' * 60}")
        tr_idx, val_idx = splits[fold]
        tr_pairs = [pairs[i] for i in tr_idx]
        val_pairs = [pairs[i] for i in val_idx]

        if args.smoke:
            tr_pairs, val_pairs = tr_pairs[:4], val_pairs[:2]

        print(f"train images: {len(tr_pairs)}  val images: {len(val_pairs)}")

        train_ds = DriveVesselDataset(tr_pairs, train=True)
        val_ds = DriveVesselDataset(val_pairs, train=False)

        pos_weight = estimate_pos_weight(tr_pairs, len(tr_pairs))
        print(f"  empirical pos_weight (neg:pos, FOV-masked): {pos_weight:.2f}")

        common = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "worker_init_fn": worker_init_fn,
            "persistent_workers": args.workers > 0,
            "pin_memory": accelerator == "gpu",
        }
        train_dl = DataLoader(
            train_ds, shuffle=True, drop_last=len(train_ds) >= args.batch_size, **common
        )
        val_dl = DataLoader(val_ds, shuffle=False, **common)

        model = build_segmentation_model(args.encoder, pretrained=True, classes=1)
        module = VesselSegmentationModule(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=pos_weight,
            dice_weight=args.dice_weight,
            max_epochs=epochs,
        )

        out_dir = Path(args.out_dir) / f"{run_name}_fold{fold}"
        if out_dir.exists() and not args.resume and not args.smoke:
            print(
                f"\nERROR: {out_dir} already exists.\n"
                f"Pass --run-name to use a new name, --resume to continue, or delete the directory first.",
                file=sys.stderr,
            )
            return 1

        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
            precision="32-true",
            deterministic=False,
            logger=CSVLogger(
                save_dir="runs", name=run_name, version=f"fold{fold}", flush_logs_every_n_steps=5
            ),
            callbacks=[
                ModelCheckpoint(
                    dirpath=out_dir,
                    filename="best",
                    monitor="val/auroc",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                ),
                ModelCheckpoint(dirpath=out_dir, filename="latest", monitor=None, every_n_epochs=1),
                EarlyStopping(
                    monitor="val/auroc", mode="max", patience=args.patience, min_delta=1e-4
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            log_every_n_steps=5,
            enable_progress_bar=True,
        )
        resume_from = out_dir / "latest.ckpt"
        if args.resume and resume_from.exists():
            print(f"  resuming from {resume_from}")
            trainer.fit(module, train_dl, val_dl, ckpt_path=str(resume_from))
        else:
            if args.resume:
                print(f"  --resume given but {resume_from} not found; starting fresh")
            trainer.fit(module, train_dl, val_dl)

        ckpt_cb = trainer.checkpoint_callback
        best_auroc = (
            float(ckpt_cb.best_model_score)
            if ckpt_cb is not None and ckpt_cb.best_model_score is not None
            else float("nan")
        )
        fold_result = {
            "fold": fold,
            "n_train_images": len(tr_pairs),
            "n_val_images": len(val_pairs),
            "pos_weight": pos_weight,
            "best_val_auroc": best_auroc,
            "epochs_run": trainer.current_epoch + 1,
            "best_checkpoint": str(ckpt_cb.best_model_path) if ckpt_cb else "",
        }
        results.append(fold_result)
        print(
            f"\nFold {fold}: best val/auroc={best_auroc:.4f} ({fold_result['epochs_run']} epochs)"
        )

    summary_path = Path("runs") / run_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    aurocs = [r["best_val_auroc"] for r in results]
    summary = {
        "run_name": run_name,
        "config": vars(args),
        "n_splits": args.n_splits,
        "folds": results,
        "val_auroc_mean": float(np.nanmean(aurocs)),
        "val_auroc_std": float(np.nanstd(aurocs)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    if len(results) > 1:
        print(
            f"val/auroc across {len(results)} folds: {summary['val_auroc_mean']:.4f} +/- {summary['val_auroc_std']:.4f}"
        )
    else:
        print(f"BEST val/auroc: {aurocs[0]:.4f}  ({results[0]['epochs_run']} epochs)")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
