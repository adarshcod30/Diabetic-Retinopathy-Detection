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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--folds", default="0", help="comma-separated fold indices")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="LR warmup length. GradingModule has always accepted this but train.py "
        "never forwarded it, so the module default of 3 silently won. Step-matching "
        "warmup across accumulation settings requires it (see docs/07).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=8,
        help="early-stopping patience in EPOCHS. An epoch is 732/accum optimiser "
        "steps, so at --accum 4 the default buys a quarter of the updates it buys "
        "the baseline; raise it to disable when step-matching.",
    )
    p.add_argument(
        "--accum",
        type=int,
        default=1,
        help="gradient accumulation steps. Effective batch = batch-size * accum. "
        "Lets a memory-constrained machine train at a larger effective batch, and "
        "is the untested lever for the grade-2 collapse (see docs/07_PHASE3_RESULTS.md). "
        "Note it does NOT fix BatchNorm statistics, which are per micro-batch.",
    )
    p.add_argument(
        "--monitor",
        default="val/qwk",
        choices=["val/qwk", "val/macro_recall", "val/sens_at_spec85"],
        help="checkpoint/early-stopping criterion. val/qwk tracks grade-2 recall and "
        "can select a hedged epoch (Result 3); val/macro_recall is boundary-blind and "
        "unsafe for referral (Result 4); val/sensitivity_referable is NOT offered here "
        "because maximizing it alone reselects the grade-2 collapse epoch in 3 of 4 "
        "checked runs (predicting referable for nearly everyone drives it to ~1.0). "
        "val/sens_at_spec85 maximizes sensitivity subject to specificity >= 0.85, "
        "this project's own stated target -- see docs/07_PHASE3_RESULTS.md Result 9.",
    )
    p.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "corn", "regression", "distance_ce"],
        help="ce is the Phase 1 baseline; the others are ordinal (see grading/losses.py)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-freeze-bn", action="store_true")
    p.add_argument("--class-weights", action="store_true", help="inverse-frequency loss weighting")
    p.add_argument(
        "--no-corn-balance",
        action="store_true",
        help="disable per-task rebalancing for CORN. Its conditional subsets are badly "
        "skewed (task j=1 is 80%% positive on APTOS), and unweighted CORN measurably "
        "worsens grade-1 recall -- see docs/07_PHASE3_RESULTS.md",
    )
    p.add_argument("--out-dir", default="models/checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true", help="2 epochs on 40 images, for wiring checks")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from last.ckpt if present (optimiser, LR schedule and epoch are restored)",
    )
    p.add_argument("--allow-ungrouped", action="store_true")
    args = p.parse_args()

    if args.accum < 1:
        # Lightning treats a negative accum as 1 (ready %% -1 == 0 always holds),
        # so it would run silently at the wrong effective batch.
        p.error("--accum must be >= 1")
    if args.accum > 1 and (args.loss == "corn" or args.class_weights):
        # CornLoss normalises by a denominator that depends on the micro-batch's
        # label composition, and weighted CE divides by the micro-batch's target
        # weight sum, so accumulating N micro-batches is not the same function as
        # one batch of 4N. Measured relative gradient differences: 0.67-1.64.
        # Unweighted ce / distance_ce / regression are exact to ~1e-7.
        p.error(
            "--accum is not equivalent to a larger batch for CORN or class-weighted "
            "losses: they normalise per micro-batch. Use --loss ce without "
            "--class-weights, or --accum 1."
        )

    import lightning as L
    import numpy as np
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from drdetect.data.dataset import FundusDataset, build_transforms, load_split
    from drdetect.grading.losses import corn_task_pos_weights, outputs_for_loss
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
    # The default name must encode every flag that changes the experiment.
    # It previously omitted --accum, --lr and --monitor, so two different runs
    # collided on one directory: CSVLogger deletes the existing metrics.csv,
    # ModelCheckpoint keeps the OLD best.ckpt and writes best-v1.ckpt beside it
    # (so evaluate.py silently scores the wrong model), and --resume picks up the
    # other run's last.ckpt. Lightning stores no accumulate_grad_batches in the
    # checkpoint, so nothing downstream would catch it.
    _default_name = f"{args.backbone}_{args.size}px_bs{args.batch_size}_{args.loss}"
    if args.accum > 1:
        _default_name += f"_accum{args.accum}"
    if args.lr != 1e-4:
        _default_name += f"_lr{args.lr:g}"
    if args.monitor != "val/qwk":
        _default_name += "_" + args.monitor.split("/")[-1]
    run_name = args.run_name or _default_name
    n_outputs = outputs_for_loss(args.loss)

    print(f"manifest    : {manifest}")
    print(f"accelerator : {accelerator}")
    eff = args.batch_size * args.accum
    print(
        f"backbone    : {args.backbone} @ {args.size}px, batch {args.batch_size}"
        + (f" x{args.accum} accum = effective {eff}" if args.accum > 1 else "")
    )
    print(f"epochs      : {epochs}{'  (SMOKE)' if args.smoke else ''}")
    print(f"loss        : {args.loss} ({n_outputs} head outputs)")
    print(f"monitor     : {args.monitor}")

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

        task_pos_weights = None
        if args.loss == "corn" and not args.no_corn_balance:
            task_pos_weights = corn_task_pos_weights([r.label for r in train_recs])
            print(f"  CORN task weights: {[round(w, 3) for w in task_pos_weights]}")

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
        if len(train_dl) % args.accum:
            print(
                f"\nERROR: --accum {args.accum} does not divide {len(train_dl)} "
                f"micro-batches/epoch ({len(train_dl) % args.accum} left over).\n"
                f"Every epoch would end with a partial-window update at a different "
                f"effective batch. Use an accum that divides {len(train_dl)}.",
                file=sys.stderr,
            )
            return 1
        val_dl = DataLoader(val_ds, shuffle=False, **common)

        model = build_model(
            args.backbone, num_outputs=n_outputs, pretrained=True, freeze_bn=not args.no_freeze_bn
        )
        total, trainable = count_parameters(model)
        print(f"  params: {total:,} total, {trainable:,} trainable")
        if not args.no_freeze_bn:
            print(f"  frozen BatchNorm layers: {model.n_frozen_bn}")

        steps_per_epoch = len(train_dl) // args.accum
        print(
            f"  lr {args.lr:.1e}, grad-clip {args.grad_clip}, "
            f"warmup {args.warmup_epochs} epochs ({args.warmup_epochs * steps_per_epoch} steps)"
        )
        print(
            f"  {steps_per_epoch} optimiser steps/epoch, "
            f"{steps_per_epoch * epochs} over {epochs} epochs"
        )
        module = GradingModule(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            class_weights=class_weights,
            task_pos_weights=task_pos_weights,
            warmup_epochs=args.warmup_epochs,
            max_epochs=epochs,
        )

        out_dir = Path(args.out_dir) / f"{run_name}_fold{fold}"
        # Guard the collision directly, not just via naming. Without --resume,
        # an existing run directory means results are about to be clobbered.
        if out_dir.exists() and not args.resume and not args.smoke:
            print(
                f"\nERROR: {out_dir} already exists.\n"
                f"Continuing would delete its metrics.csv and leave the OLD best.ckpt in place,\n"
                f"so evaluate.py would score the wrong model. Pass --run-name to use a new name,\n"
                f"--resume to continue that run, or delete the directory first.",
                file=sys.stderr,
            )
            return 1
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=1,
            # Gradient clipping is not optional at batch 4. Without it, a single
            # noisy batch can blow the loss up by two orders of magnitude and
            # collapse the model to one class -- observed on this project.
            gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
            accumulate_grad_batches=args.accum,
            precision="32-true",  # MPS fp16 is unreliable for BN-heavy nets
            deterministic=False,  # some MPS kernels lack deterministic variants
            # Default flush is every 100 logged steps -- under twice per epoch at
            # 183 steps/epoch. A run stopped by hand would lose its tail.
            logger=CSVLogger(
                save_dir="runs",
                name=run_name,
                version=f"fold{fold}",
                flush_logs_every_n_steps=25,
            ),
            callbacks=[
                # save_last is what makes a run resumable: `best` is the
                # highest-QWK epoch, not the latest state, so it carries the
                # wrong optimiser and LR-schedule position to continue from.
                ModelCheckpoint(
                    dirpath=out_dir,
                    filename="best",
                    monitor=args.monitor,
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                ),
                # An unmonitored every-epoch checkpoint. ModelCheckpoint only writes
                # last.ckpt on epochs where a top-k save also fired, so --resume could
                # silently rewind to a much earlier epoch.
                ModelCheckpoint(dirpath=out_dir, filename="latest", monitor=None, every_n_epochs=1),
                EarlyStopping(
                    monitor=args.monitor, mode="max", patience=args.patience, min_delta=1e-3
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

        # callback_metrics holds the LAST epoch's values, but the checkpoint on
        # disk is the BEST epoch. Reporting the former while saving the latter
        # makes the summary describe a different model than the one shipped --
        # on the Phase 1 baseline that understated QWK by 0.022, which would
        # bias every later comparison in the same direction.
        ckpt_cb = trainer.checkpoint_callback
        best_monitored = (
            float(ckpt_cb.best_model_score)
            if ckpt_cb is not None and ckpt_cb.best_model_score is not None
            else float("nan")
        )
        final_qwk = float(trainer.callback_metrics.get("val/qwk", float("nan")))
        # Final value of whatever is MONITORED, so "best" and "final" are the
        # same quantity. Printing best-macro-recall beside final-QWK invites a
        # comparison between two different metrics.
        final_monitored = float(trainer.callback_metrics.get(args.monitor, float("nan")))

        metrics = {f"final/{k}": float(v) for k, v in trainer.callback_metrics.items()}
        metrics.update(
            {
                "fold": fold,
                "split_strategy": strategy,
                "n_train": len(train_recs),
                "n_val": len(val_recs),
                "loss": args.loss,
                "image_size": args.size,
                "best_qwk": best_monitored,
                "monitor": args.monitor,
                "final_qwk": final_qwk,
                "final_monitored": final_monitored,
                "epochs_run": trainer.current_epoch + 1,
                # Epochs are the misleading unit when accumulation changes what an
                # epoch costs. Record the optimisation budget explicitly.
                "accum": args.accum,
                "effective_batch": args.batch_size * args.accum,
                "steps_per_epoch": len(train_dl) // args.accum,
                "optimiser_steps": int(trainer.global_step),
                "warmup_epochs": args.warmup_epochs,
                "warmup_steps": args.warmup_epochs * (len(train_dl) // args.accum),
                "best_checkpoint": str(ckpt_cb.best_model_path) if ckpt_cb else "",
            }
        )
        results.append(metrics)
        print(
            f"\nFold {fold}: best {args.monitor}={best_monitored:.4f} "
            f"(final {final_monitored:.4f}, QWK {final_qwk:.4f}, "
            f"{metrics['epochs_run']} epochs)"
        )

    summary_path = Path("runs") / run_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Named for the metric actually monitored -- when --monitor is not QWK,
    # labelling this "QWK" reports one metric under another's name.
    monitored = [r.get("best_qwk", float("nan")) for r in results]
    summary = {
        "run_name": run_name,
        "config": vars(args),
        "folds": results,
        "monitor": args.monitor,
        "monitored_mean": float(np.nanmean(monitored)),
        "monitored_std": float(np.nanstd(monitored)),
        "qwk_mean": float(np.nanmean([r.get("final_qwk", float("nan")) for r in results])),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(
        f"BEST {args.monitor} across {len(monitored)} fold(s): "
        f"{summary['monitored_mean']:.4f} +/- {summary['monitored_std']:.4f}"
    )
    print(f"Summary: {summary_path}")
    print("\nThis is the BASELINE. Record it; every Phase 3 change is measured against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
