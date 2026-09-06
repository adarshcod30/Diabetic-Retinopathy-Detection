#!/usr/bin/env python3
"""Train an OD/fovea heatmap-regression model on IDRiD's localisation task.

Phase 4 target: mean localisation error < 0.5x OD diameter (the roadmap's
own number). IDRiD's localisation task ships a genuine official split (413
train / 103 test) -- unlike DRIVE vessels (no public test labels at all) or
IDRiD's own hard-exudate segmentation (81 images total), so this uses a
single internal train/val split for model selection plus one final
held-out evaluation, matching how this project's hard-exudate segmentation
started before its 5-fold CV follow-up. 5-fold CV here is explicit future
work, not attempted this pass, given how much of Phase 4 remains queued
behind it.

Usage:
    python scripts/train_localization.py
    python scripts/train_localization.py --smoke     # 2 epochs, 4 images
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


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--idrid-root", default="data/raw/idrid")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument(
        "--sigma", type=float, default=12.0, help="Gaussian heatmap sigma, working-res px"
    )
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="models/checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    import lightning as L
    import numpy as np
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from drdetect.segmentation.localization import (
        IDRiDLocalizationDataset,
        LocalizationModule,
        find_idrid_localization_pairs,
        working_res_od_diameter,
    )
    from drdetect.segmentation.model import build_segmentation_model
    from drdetect.utils.seed import seed_everything, worker_init_fn

    seed_everything(args.seed)

    train_pairs_all = find_idrid_localization_pairs(args.idrid_root, "train")
    test_pairs = find_idrid_localization_pairs(args.idrid_root, "test")
    if not train_pairs_all or not test_pairs:
        print(f"No IDRiD localisation pairs found under {args.idrid_root}.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_pairs_all))
    n_val = max(1, round(args.val_fraction * len(train_pairs_all)))
    val_idx, tr_idx = set(order[:n_val].tolist()), set(order[n_val:].tolist())
    tr_pairs = [train_pairs_all[i] for i in sorted(tr_idx)]
    val_pairs = [train_pairs_all[i] for i in sorted(val_idx)]

    if args.smoke:
        tr_pairs, val_pairs = tr_pairs[:4], val_pairs[:2]

    accelerator = pick_accelerator()
    epochs = 2 if args.smoke else args.epochs
    run_name = args.run_name or f"localization_{args.encoder}_{args.height}x{args.width}"
    size = (args.height, args.width)
    diameter = working_res_od_diameter(size)

    print(f"accelerator : {accelerator}")
    print(f"encoder     : {args.encoder} @ {args.height}x{args.width}, batch {args.batch_size}")
    print(f"epochs      : {epochs}{'  (SMOKE)' if args.smoke else ''}")
    print(
        f"train images: {len(tr_pairs)}  val images: {len(val_pairs)}  (103 official test images held out)"
    )
    print(f"working-res OD diameter (normalisation unit): {diameter:.2f}px")

    train_ds = IDRiDLocalizationDataset(tr_pairs, size=size, train=True, sigma=args.sigma)
    val_ds = IDRiDLocalizationDataset(val_pairs, size=size, train=False, sigma=args.sigma)

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

    model = build_segmentation_model(args.encoder, pretrained=True, classes=2)
    module = LocalizationModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        working_od_diameter=diameter,
        max_epochs=epochs,
    )

    out_dir = Path(args.out_dir) / f"{run_name}_fold0"
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
            save_dir="runs", name=run_name, version="fold0", flush_logs_every_n_steps=10
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=out_dir,
                filename="best",
                monitor="val/mean_error_diameters",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
            ModelCheckpoint(dirpath=out_dir, filename="latest", monitor=None, every_n_epochs=1),
            EarlyStopping(
                monitor="val/mean_error_diameters",
                mode="min",
                patience=args.patience,
                min_delta=1e-4,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
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
    best_error = (
        float(ckpt_cb.best_model_score)
        if ckpt_cb is not None and ckpt_cb.best_model_score is not None
        else float("nan")
    )
    summary = {
        "run_name": run_name,
        "config": vars(args),
        "working_od_diameter_px": diameter,
        "n_train_images": len(tr_pairs),
        "n_val_images": len(val_pairs),
        "best_val_mean_error_diameters": best_error,
        "epochs_run": trainer.current_epoch + 1,
        "best_checkpoint": str(ckpt_cb.best_model_path) if ckpt_cb else "",
    }
    summary_path = Path("runs") / run_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"BEST val/mean_error_diameters: {best_error:.4f}  ({summary['epochs_run']} epochs)")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
