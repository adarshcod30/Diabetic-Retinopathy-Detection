#!/usr/bin/env python3
"""MC-dropout uncertainty + a human-escalation policy simulation.

Two roadmap items in one script (docs/04_ROADMAP.md, Phase 5) because the
second directly consumes the first's output: an uncertainty score is only
useful for escalation if it's actually shown to track real error first.

Part 1 -- is MC-dropout's uncertainty signal any good?
    20 stochastic forward passes per validation image (dropout reactivated,
    BatchNorm still frozen on its trained running stats -- see
    drdetect.calibration.mc_dropout). The std of the referable-score across
    those passes is the per-image uncertainty. Bucketed into quintiles,
    checked against actual accuracy: a useful signal should show LOWER
    accuracy in HIGHER-uncertainty buckets. Reported honestly either way --
    this project's own Phase 5 fusion-head result (docs/16) was a null, and
    an uncertainty signal that doesn't track error would be an equally
    legitimate thing to find and report, not paper over.

Part 2 -- human-escalation simulation.
    For k in {0, 5, 10, 20, 30, 50}%: route the top-k% most uncertain images
    to an assumed-perfect grader (a stated simplifying assumption, not a
    measured human accuracy -- this project has no real grader to test
    against), keep the model's own prediction for the rest, and report the
    combined system's accuracy/QWK as a function of k.

Usage:
    python scripts/evaluate_uncertainty.py --checkpoint models/checkpoints/baseline_effb0_512_fold0/best.ckpt
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
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-mc-samples", type=int, default=20)
    p.add_argument("--out", default="runs/uncertainty_evaluation.json")
    args = p.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from drdetect.calibration.mc_dropout import mc_dropout_samples
    from drdetect.data.dataset import FundusDataset, build_transforms, load_split
    from drdetect.eval.metrics import quadratic_weighted_kappa
    from drdetect.grading.losses import decode_output
    from drdetect.grading.model import build_model
    from drdetect.utils.seed import seed_everything

    seed_everything(args.seed)
    manifest = Path(args.manifest or f"data/manifests/aptos_{args.size}.csv")
    _, val_recs, strategy = load_split(
        manifest, fold=args.fold, n_splits=args.n_splits, seed=args.seed
    )
    ds = FundusDataset(val_recs, args.data_root, build_transforms(args.size, train=False))
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    print(f"val images: {len(ds)}  split strategy: {strategy}")

    device = torch.device(pick_accelerator())
    model = build_model(args.backbone, num_outputs=5, pretrained=False, freeze_bn=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.to(device).eval()

    all_targets, all_preds, all_uncertainty = [], [], []
    for x, y in dl:
        x = x.to(device)
        samples = mc_dropout_samples(model, x, n_samples=args.n_mc_samples).cpu()
        mean_logits = samples.mean(dim=0)
        preds, _referable = decode_output(mean_logits, "ce")

        per_sample_referable = np.stack(
            [decode_output(samples[i], "ce")[1] for i in range(samples.shape[0])]
        )
        uncertainty = per_sample_referable.std(axis=0)

        all_targets.append(y.numpy())
        all_preds.append(preds)
        all_uncertainty.append(uncertainty)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    uncertainty = np.concatenate(all_uncertainty)
    correct = y_pred == y_true
    print(
        f"model-alone accuracy: {correct.mean():.4f}  QWK: {quadratic_weighted_kappa(y_true, y_pred):.4f}"
    )

    # Part 1: does uncertainty track error?
    order = np.argsort(uncertainty)
    quintile_edges = np.array_split(order, 5)
    print("\nuncertainty quintile -> accuracy (low uncertainty first):")
    quintile_acc = []
    for i, idx in enumerate(quintile_edges):
        acc = correct[idx].mean()
        quintile_acc.append(float(acc))
        print(
            f"  Q{i + 1} (mean uncertainty {uncertainty[idx].mean():.4f}): accuracy {acc:.4f}  n={len(idx)}"
        )

    from scipy.stats import spearmanr

    rho, spearman_p = spearmanr(uncertainty, correct.astype(float))
    print(f"Spearman(uncertainty, correctness): rho={rho:.4f}  p={spearman_p:.4f}")

    # Part 2: human-escalation simulation (assumed-perfect grader on the
    # escalated fraction -- a stated simplification, not a measured human
    # accuracy this project has no data to provide).
    print("\nescalation-policy simulation (perfect-grader assumption on escalated cases):")
    escalation_results = []
    sorted_by_uncertainty = np.argsort(-uncertainty)  # most uncertain first
    for k_pct in [0, 5, 10, 20, 30, 50]:
        n_escalate = int(len(y_true) * k_pct / 100)
        escalated_idx = set(sorted_by_uncertainty[:n_escalate].tolist())
        combined_pred = np.array(
            [y_true[i] if i in escalated_idx else y_pred[i] for i in range(len(y_true))]
        )
        combined_acc = float((combined_pred == y_true).mean())
        combined_qwk = quadratic_weighted_kappa(y_true, combined_pred)
        escalation_results.append(
            {
                "k_pct": k_pct,
                "n_escalated": n_escalate,
                "accuracy": combined_acc,
                "qwk": combined_qwk,
            }
        )
        print(
            f"  k={k_pct:3d}%  (n={n_escalate:4d} escalated): accuracy={combined_acc:.4f}  QWK={combined_qwk:.4f}"
        )

    result = {
        "n_val": len(y_true),
        "split_strategy": strategy,
        "model_alone_accuracy": float(correct.mean()),
        "model_alone_qwk": quadratic_weighted_kappa(y_true, y_pred),
        "uncertainty_quintile_accuracy": quintile_acc,
        "spearman_uncertainty_vs_correctness": {"rho": float(rho), "p": float(spearman_p)},
        "escalation_policy": escalation_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
