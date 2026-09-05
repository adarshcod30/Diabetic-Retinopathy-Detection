#!/usr/bin/env python3
"""Train a lesion segmentation model on IDRiD.

Phase 4 target: an honest hard-exudate segmentation baseline on IDRiD's own
train/test split, scored by pixel AUPRC (see drdetect.segmentation.metrics
for why AUPRC and not AUROC -- IDRiD lesions are <0.1% positive pixels).

IDRiD's segmentation set is small (54 train / 27 test images total, see
docs/04_ROADMAP.md Phase 4). `--folds` selects which of `--n-splits` k-fold
splits of the 54 training images to run -- one at a time by default (fold 0,
matching scripts/train.py's own default), or `--folds 0,1,2,3,4` for the full
cross-validation the roadmap specifies for this task. The 27 official test
images are never touched here at all; they are reserved entirely for
scripts/evaluate_segmentation.py.

Usage:
    python scripts/train_segmentation.py                       # fold 0 only
    python scripts/train_segmentation.py --folds 0,1,2,3,4      # full 5-fold CV
    python scripts/train_segmentation.py --lesion haemorrhages
    python scripts/train_segmentation.py --smoke                # 2 epochs, 4 images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Same reasoning as scripts/train.py: MPS watermarks must be set before torch
# import or the allocator can over-request unified memory. See
# docs/05_PROTOTYPE_SCOPE.md 6.2.
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


def estimate_pos_weight(dataset, n_samples: int) -> float:
    """Empirical neg:pos pixel ratio over actually-sampled PATCHES, not over
    full images. Full-IDRiD-image imbalance is ~0.07% positive (see
    drdetect.segmentation.metrics docstring), but lesion_patch_prob biases
    the patches this model actually trains on toward lesion-containing
    crops -- reusing the full-image ratio here would badly over-weight the
    positive class relative to what a sampled patch really contains.
    """
    n_samples = min(n_samples, len(dataset))
    pos = 0.0
    total = 0.0
    for i in range(n_samples):
        _, mask = dataset[i]
        pos += float(mask.sum())
        total += float(mask.numel())
    neg = total - pos
    if pos == 0:
        raise ValueError(
            f"0 positive pixels across {n_samples} sampled patches -- "
            "lesion_patch_prob or the lesion/split combination is wrong."
        )
    return neg / pos


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--idrid-root", default="data/raw/idrid")
    p.add_argument(
        "--lesion",
        default="hard_exudates",
        choices=["microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates", "optic_disc"],
    )
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--patches-per-image", type=int, default=20)
    p.add_argument("--lesion-patch-prob", type=float, default=0.8)
    p.add_argument("--folds", default="0", help="comma-separated fold indices, e.g. 0,1,2,3,4")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument(
        "--pos-weight-samples",
        type=int,
        default=200,
        help="patches sampled to empirically estimate BCE pos_weight before training",
    )
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

    from drdetect.segmentation.dataset import IDRiDLesionDataset, find_idrid_lesion_pairs
    from drdetect.segmentation.model import build_segmentation_model
    from drdetect.segmentation.module import SegmentationModule
    from drdetect.utils.seed import seed_everything, worker_init_fn

    seed_everything(args.seed)

    train_pairs = find_idrid_lesion_pairs(args.idrid_root, args.lesion, "train")
    if not train_pairs:
        print(
            f"No {args.lesion} training pairs found under {args.idrid_root}. "
            "Check --idrid-root and that the dataset was extracted with its original folder names.",
            file=sys.stderr,
        )
        return 1

    # Computed once so every requested fold comes from the SAME partition of
    # the 54 images -- calling KFold().split() fresh per fold with the same
    # seed would happen to be equivalent here since it's deterministic, but
    # doing it once up front is what makes that a fact of implementation
    # rather than something each fold's code has to get right independently.
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(train_pairs))

    accelerator = pick_accelerator()
    epochs = 2 if args.smoke else args.epochs
    run_name = args.run_name or f"segmentation_{args.lesion}_{args.encoder}_{args.patch_size}px"

    print(f"lesion      : {args.lesion}")
    print(f"accelerator : {accelerator}")
    print(f"encoder     : {args.encoder} @ {args.patch_size}px patches, batch {args.batch_size}")
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

        train_ds = IDRiDLesionDataset(
            tr_pairs,
            patch_size=args.patch_size,
            train=True,
            patches_per_image=args.patches_per_image,
            lesion_patch_prob=args.lesion_patch_prob,
            seed=args.seed,
        )
        val_ds = IDRiDLesionDataset(
            val_pairs,
            patch_size=args.patch_size,
            train=False,
            patches_per_image=args.patches_per_image,
            lesion_patch_prob=args.lesion_patch_prob,
            seed=args.seed + 1,
        )
        print(f"  train patches/epoch: {len(train_ds)}  val patches/epoch: {len(val_ds)}")

        pos_weight = estimate_pos_weight(train_ds, args.pos_weight_samples)
        print(f"  empirical pos_weight (neg:pos over sampled patches): {pos_weight:.2f}")

        common = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "worker_init_fn": worker_init_fn,
            "persistent_workers": args.workers > 0,
            "pin_memory": accelerator == "gpu",
        }
        train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
        val_dl = DataLoader(val_ds, shuffle=False, **common)

        model = build_segmentation_model(args.encoder, pretrained=True, classes=1)
        module = SegmentationModule(
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
                f"Continuing would delete its metrics.csv and leave the OLD best.ckpt in place.\n"
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
                save_dir="runs", name=run_name, version=f"fold{fold}", flush_logs_every_n_steps=10
            ),
            callbacks=[
                ModelCheckpoint(
                    dirpath=out_dir,
                    filename="best",
                    monitor="val/auprc",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                ),
                ModelCheckpoint(dirpath=out_dir, filename="latest", monitor=None, every_n_epochs=1),
                EarlyStopping(
                    monitor="val/auprc", mode="max", patience=args.patience, min_delta=1e-3
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
        best_auprc = (
            float(ckpt_cb.best_model_score)
            if ckpt_cb is not None and ckpt_cb.best_model_score is not None
            else float("nan")
        )
        fold_result = {
            "fold": fold,
            "n_train_images": len(tr_pairs),
            "n_val_images": len(val_pairs),
            "pos_weight": pos_weight,
            "best_val_auprc": best_auprc,
            "epochs_run": trainer.current_epoch + 1,
            "best_checkpoint": str(ckpt_cb.best_model_path) if ckpt_cb else "",
        }
        results.append(fold_result)
        print(
            f"\nFold {fold}: best val/auprc={best_auprc:.4f} ({fold_result['epochs_run']} epochs)"
        )

    summary_path = Path("runs") / run_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    auprcs = [r["best_val_auprc"] for r in results]
    summary = {
        "run_name": run_name,
        "config": vars(args),
        "n_splits": args.n_splits,
        "folds": results,
        "val_auprc_mean": float(np.nanmean(auprcs)),
        "val_auprc_std": float(np.nanstd(auprcs)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    if len(results) > 1:
        print(
            f"val/auprc across {len(results)} folds: "
            f"{summary['val_auprc_mean']:.4f} +/- {summary['val_auprc_std']:.4f}"
        )
    else:
        print(f"BEST val/auprc: {auprcs[0]:.4f}  ({results[0]['epochs_run']} epochs)")
    print(f"Summary: {summary_path}")
    for r in results:
        print(f"Next: python scripts/evaluate_segmentation.py --checkpoint {r['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
