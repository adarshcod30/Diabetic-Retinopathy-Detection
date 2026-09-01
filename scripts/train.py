#!/usr/bin/env python3
"""Train a DR grading model.

Phase 1 target: an honest, un-tuned baseline QWK on APTOS to measure everything
else against.

Usage:
    python scripts/train.py                                  # baseline defaults
    python scripts/train.py --size 384 --batch-size 8 --epochs 20
    python scripts/train.py --smoke                          # 2 epochs, tiny subset
    python scripts/train.py --folds 0,1,2,3,4                # full CV
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# MPS watermarks must be set before torch is imported. Unified memory has no
# hard VRAM boundary: without these the allocator may request ~1.7x physical RAM
# and macOS honours it by swapping other applications out, exhausting the system
# instead of raising a catchable error. See docs/05_PROTOTYPE_SCOPE.md 6.2.
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
    p.add_argument("--manifest", default=None, help="default: data/manifests/aptos_<size>.csv")
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--folds", default="0", help="comma-separated fold indices")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-freeze-bn", action="store_true")
    p.add_argument("--class-weights", action="store_true", help="inverse-frequency loss weighting")
    p.add_argument("--out-dir", default="models/checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true", help="2 epochs on 40 images, for wiring checks")
    p.add_argument("--allow-ungrouped", action="store_true")
    args = p.parse_args()

    import lightning as L
    import numpy as np
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from drdetect.data.dataset import FundusDataset, build_transforms, load_split
    from drdetect.grading.model import build_model, count_parameters
    from drdetect.grading.module import GradingModule
    from drdetect.utils.seed import seed_everything, worker_init_fn

    seed_everything(args.seed)

    manifest = Path(args.manifest or f"data/manifests/aptos_{args.size}.csv")
    if not manifest.exists():
        print(
            f"Manifest not found: {manifest}\n"
            f"Run: python scripts/preprocess.py --dataset aptos --size {args.size}",
            file=sys.stderr,
        )
        return 1

    accelerator = pick_accelerator()
    epochs = 2 if args.smoke else args.epochs
    run_name = args.run_name or f"{args.backbone}_{args.size}px_bs{args.batch_size}"

    print(f"manifest    : {manifest}")
    print(f"accelerator : {accelerator}")
    print(f"backbone    : {args.backbone} @ {args.size}px, batch {args.batch_size}")
    print(f"epochs      : {epochs}{'  (SMOKE)' if args.smoke else ''}")

    results = []
    for fold in [int(f) for f in args.folds.split(",")]:
        print(f"\n{'=' * 60}\nFold {fold}\n{'=' * 60}")

        train_recs, val_recs, strategy = load_split(
            manifest,
            fold=fold,
            n_splits=args.n_splits,
            seed=args.seed,
            allow_ungrouped=args.allow_ungrouped,
        )
        if args.smoke:
            train_recs, val_recs = train_recs[:32], val_recs[:8]

        print(f"split strategy: {strategy}")
        if strategy == "image_level":
            print("  WARNING: no grouping -- results may be optimistically biased.")
        print(f"train: {len(train_recs)}  val: {len(val_recs)}")

        train_counts = np.bincount([r.label for r in train_recs], minlength=5)
        print("  train class counts:", train_counts.tolist())

        class_weights = None
        if args.class_weights:
            inv = train_counts.sum() / np.maximum(train_counts, 1)
            class_weights = (inv / inv.mean()).tolist()
            print("  class weights    :", [round(w, 2) for w in class_weights])

        train_ds = FundusDataset(train_recs, args.data_root, build_transforms(args.size, True))
        val_ds = FundusDataset(val_recs, args.data_root, build_transforms(args.size, False))

        common = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "worker_init_fn": worker_init_fn,
            "persistent_workers": args.workers > 0,
            "pin_memory": accelerator == "gpu",
        }
        train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
        val_dl = DataLoader(val_ds, shuffle=False, **common)

        model = build_model(
            args.backbone, num_outputs=5, pretrained=True, freeze_bn=not args.no_freeze_bn
        )
        total, trainable = count_parameters(model)
        print(f"  params: {total:,} total, {trainable:,} trainable")
        if not args.no_freeze_bn:
            print(f"  frozen BatchNorm layers: {model.n_frozen_bn}")

        module = GradingModule(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            max_epochs=epochs,
        )

        out_dir = Path(args.out_dir) / f"{run_name}_fold{fold}"
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            precision="32-true",  # MPS fp16 is unreliable for BN-heavy nets
            deterministic=False,  # some MPS kernels lack deterministic variants
            logger=CSVLogger(save_dir="runs", name=run_name, version=f"fold{fold}"),
            callbacks=[
                ModelCheckpoint(
                    dirpath=out_dir, filename="best", monitor="val/qwk", mode="max", save_top_k=1
                ),
                EarlyStopping(monitor="val/qwk", mode="max", patience=8, min_delta=1e-3),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            log_every_n_steps=10,
            enable_progress_bar=True,
        )
        trainer.fit(module, train_dl, val_dl)

        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        metrics.update(
            {
                "fold": fold,
                "split_strategy": strategy,
                "n_train": len(train_recs),
                "n_val": len(val_recs),
            }
        )
        results.append(metrics)
        print(f"\nFold {fold}: QWK={metrics.get('val/qwk', float('nan')):.4f}")

    summary_path = Path("runs") / run_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    qwks = [r.get("val/qwk", float("nan")) for r in results]
    summary = {
        "run_name": run_name,
        "config": vars(args),
        "folds": results,
        "qwk_mean": float(np.nanmean(qwks)),
        "qwk_std": float(np.nanstd(qwks)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"QWK across {len(qwks)} fold(s): {summary['qwk_mean']:.4f} +/- {summary['qwk_std']:.4f}")
    print(f"Summary: {summary_path}")
    print("\nThis is the BASELINE. Record it; every Phase 3 change is measured against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
