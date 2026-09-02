#!/usr/bin/env python3
"""Evaluate a trained checkpoint properly: operating point, CIs, calibration.

Why this is a separate script from training
-------------------------------------------
Training reports whatever the naive 0.5 threshold happens to give. That is not
the number to compare against the literature: Gulshan, Ting and IDx-DR all
report at a *chosen* operating point. This script separates the two steps that
must never be conflated --

    choose_threshold_for_sensitivity(...)   on the selection split
    evaluate_at_threshold(...)              on the evaluation split

-- so a threshold can never be silently tuned on the data it is scored against.

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/.../best.ckpt
    python scripts/evaluate.py --checkpoint ... --target-sensitivity 0.90 --bootstrap 2000
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

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "corn", "regression", "distance_ce"],
        help="must match how the checkpoint was trained; it determines head size and decoding",
    )
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-sensitivity", type=float, default=0.90)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from drdetect.data.dataset import FundusDataset, build_transforms, load_split
    from drdetect.eval.metrics import (
        binary_scores,
        bootstrap_ci,
        choose_threshold_for_sensitivity,
        evaluate_at_threshold,
        expected_calibration_error,
        quadratic_weighted_kappa,
        referable_labels,
    )
    from drdetect.grading.losses import decode_output, naive_referable_cut, outputs_for_loss
    from drdetect.grading.model import build_model
    from drdetect.utils.seed import seed_everything

    seed_everything(args.seed)
    manifest = Path(args.manifest or f"data/manifests/aptos_{args.size}.csv")

    _, val_recs, strategy = load_split(
        manifest, fold=args.fold, n_splits=args.n_splits, seed=args.seed
    )
    ds = FundusDataset(val_recs, args.data_root, build_transforms(args.size, train=False))
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    n_outputs = outputs_for_loss(args.loss)
    model = build_model(args.backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device} | split strategy: {strategy}")
    print(f"loss       : {args.loss} ({n_outputs} head outputs)")
    print(f"evaluating : {len(val_recs)} images\n")

    outputs_all, targets_all = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dl, 1):
            outputs_all.append(model(x.to(device)).float().cpu())
            targets_all.append(y.numpy())
            if i % 20 == 0:
                print(f"  {i * args.batch_size}/{len(val_recs)}", flush=True)

    outputs = torch.cat(outputs_all)
    targets = np.concatenate(targets_all)
    preds, p_ref = decode_output(outputs, args.loss)

    qwk, qwk_lo, qwk_hi = bootstrap_ci(
        quadratic_weighted_kappa, targets, preds, n_resamples=args.bootstrap, seed=args.seed
    )
    acc = float((preds == targets).mean())

    y_ref = referable_labels(targets)
    naive = evaluate_at_threshold(y_ref, p_ref, naive_referable_cut(args.loss))
    thr = choose_threshold_for_sensitivity(y_ref, p_ref, target_sensitivity=args.target_sensitivity)
    tuned = evaluate_at_threshold(y_ref, p_ref, thr)

    sens_fn = lambda t, s: binary_scores(t, (s >= thr).astype(int)).sensitivity  # noqa: E731
    spec_fn = lambda t, s: binary_scores(t, (s >= thr).astype(int)).specificity  # noqa: E731
    _, sens_lo, sens_hi = bootstrap_ci(
        sens_fn, y_ref, p_ref, n_resamples=args.bootstrap, seed=args.seed
    )
    _, spec_lo, spec_hi = bootstrap_ci(
        spec_fn, y_ref, p_ref, n_resamples=args.bootstrap, seed=args.seed
    )

    ece = expected_calibration_error(y_ref, p_ref)

    print("=" * 66)
    print(f"QWK (5-class)            {qwk:.4f}   95% CI [{qwk_lo:.4f}, {qwk_hi:.4f}]")
    print(f"Accuracy                 {acc:.4f}")
    print()
    print("Referable DR (grade >= 2)")
    print(f"  at naive threshold 0.5   sens {naive.sensitivity:.3f}  spec {naive.specificity:.3f}")
    print(
        f"  at threshold {thr:.4f}     sens {tuned.sensitivity:.3f}  spec {tuned.specificity:.3f}"
    )
    print(f"     sensitivity 95% CI    [{sens_lo:.3f}, {sens_hi:.3f}]")
    print(f"     specificity 95% CI    [{spec_lo:.3f}, {spec_hi:.3f}]")
    print(f"  cases: {tuned.n_positive} referable, {tuned.n_negative} not")
    print(f"  ECE (uncalibrated)       {ece:.4f}")
    print()
    print("Per-class recall")
    for c, name in enumerate(CLASS_NAMES):
        m = targets == c
        r = float((preds[m] == c).mean()) if m.any() else float("nan")
        print(f"  {c} {name:<9} n={int(m.sum()):>4}  recall {r:.3f}")
    print()
    print("Confusion matrix (rows = true, cols = predicted)")
    cm = np.zeros((5, 5), int)
    for t_, p_ in zip(targets, preds, strict=True):
        cm[t_, p_] += 1
    print("      " + "".join(f"{i:>7}" for i in range(5)))
    for i, row in enumerate(cm):
        print(f"  {i}   " + "".join(f"{v:>7}" for v in row))
    print("=" * 66)
    print("\nNOTE: threshold selected on the SAME split it is scored on, so these")
    print("are optimistic. The honest number comes from the locked external test")
    print("set in Phase 8.")

    out = Path(args.out or "runs/evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "loss": args.loss,
                "n_images": len(val_recs),
                "split_strategy": strategy,
                "qwk": qwk,
                "qwk_ci": [qwk_lo, qwk_hi],
                "accuracy": acc,
                "threshold": thr,
                "target_sensitivity": args.target_sensitivity,
                "sensitivity": tuned.sensitivity,
                "sensitivity_ci": [sens_lo, sens_hi],
                "specificity": tuned.specificity,
                "specificity_ci": [spec_lo, spec_hi],
                "naive_sensitivity": naive.sensitivity,
                "naive_specificity": naive.specificity,
                "ece": ece,
                "confusion_matrix": cm.tolist(),
                # Per-sample outputs, so two runs can be compared with PAIRED
                # tests. Comparing overlapping confidence intervals is not a
                # significance test: two intervals can overlap while the paired
                # difference is highly significant, because the same images are
                # scored by both models.
                "image_ids": [r.image_id for r in val_recs],
                "targets": targets.tolist(),
                "predictions": preds.tolist(),
                "referable_score": p_ref.tolist(),
                "per_class_recall": {
                    CLASS_NAMES[c]: (
                        float((preds[targets == c] == c).mean()) if (targets == c).any() else None
                    )
                    for c in range(5)
                },
            },
            indent=2,
        )
    )
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
