#!/usr/bin/env python3
"""Fit temperature scaling on a checkpoint, report ECE before/after, plot reliability.

Phase 5 (docs/04_ROADMAP.md): "ECE < 0.05 after calibration; ... reliability
diagram + ECE before/after." This is the script that produces both numbers
and the diagram, and -- via `drdetect.calibration.temperature.save_temperature`
-- the sidecar file that makes the fitted T actually apply at inference time
in scripts/predict.py and the demo, not just live in this script's own output.

Usage:
    python scripts/calibrate.py --checkpoint models/checkpoints/baseline_effb0_512_fold0/best.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        choices=["ce", "distance_ce"],
        help="CORN/regression have no softmax to calibrate",
    )
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot-out",
        default=None,
        help="default: alongside the checkpoint, reliability_diagram.png",
    )
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from drdetect.calibration.temperature import fit_temperature, save_temperature
    from drdetect.data.dataset import FundusDataset, build_transforms, load_split
    from drdetect.eval.metrics import expected_calibration_error, referable_labels
    from drdetect.grading.losses import decode_output, outputs_for_loss
    from drdetect.grading.model import build_model
    from drdetect.utils.seed import seed_everything

    seed_everything(args.seed)
    manifest = Path(args.manifest or f"data/manifests/aptos_{args.size}.csv")

    _, val_recs, strategy = load_split(
        manifest, fold=args.fold, n_splits=args.n_splits, seed=args.seed
    )
    ds = FundusDataset(val_recs, args.data_root, build_transforms(args.size, train=False))
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    device = "cpu"  # matches scripts/predict.py's deployability constraint; calibration is a one-off, not perf-sensitive
    n_outputs = outputs_for_loss(args.loss)
    model = build_model(args.backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"checkpoint : {args.checkpoint}")
    print(f"split      : fold {args.fold}/{args.n_splits} ({strategy}), {len(val_recs)} images")

    logits_all, targets_all = [], []
    with torch.no_grad():
        for x, y in dl:
            logits_all.append(model(x.to(device)).float().cpu())
            targets_all.append(y)
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    targets_np = targets.numpy()

    # BEFORE: raw softmax, T=1.
    preds_before, p_ref_before = decode_output(logits, args.loss)
    conf_before = torch.softmax(logits, dim=1).max(dim=1).values.numpy()
    correct_before = (preds_before == targets_np).astype(int)
    y_ref = referable_labels(targets_np)

    ece_top1_before = expected_calibration_error(correct_before, conf_before)
    ece_referable_before = expected_calibration_error(y_ref, p_ref_before)

    temperature = fit_temperature(logits, targets)

    # AFTER: same logits, divided by the fitted T.
    scaled = logits / temperature
    preds_after, p_ref_after = decode_output(scaled, args.loss)
    conf_after = torch.softmax(scaled, dim=1).max(dim=1).values.numpy()
    correct_after = (preds_after == targets_np).astype(int)
    ece_top1_after = expected_calibration_error(correct_after, conf_after)
    ece_referable_after = expected_calibration_error(y_ref, p_ref_after)

    assert (preds_before == preds_after).all(), (
        "temperature scaling must not change argmax predictions"
    )

    sidecar = save_temperature(
        args.checkpoint,
        temperature,
        fitted_on=f"fold {args.fold}/{args.n_splits} val, {len(val_recs)} images",
    )

    print(f"\nfitted temperature: {temperature:.4f}")
    print(f"{'':20s} {'top-1 ECE':>12s} {'referable ECE':>15s}")
    print(f"{'before (T=1)':20s} {ece_top1_before:>12.4f} {ece_referable_before:>15.4f}")
    print(
        f"{'after (T=' + f'{temperature:.2f}' + ')':20s} {ece_top1_after:>12.4f} {ece_referable_after:>15.4f}"
    )
    print(f"\nsaved: {sidecar}")

    plot_out = Path(args.plot_out or Path(args.checkpoint).with_name("reliability_diagram.png"))
    _plot_reliability(conf_before, correct_before, conf_after, correct_after, plot_out)
    print(f"saved: {plot_out}")
    return 0


def _plot_reliability(
    conf_before, correct_before, conf_after, correct_after, out_path: Path, n_bins: int = 10
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    for ax, conf, correct, title in [
        (axes[0], conf_before, correct_before, "Before (T=1)"),
        (axes[1], conf_after, correct_after, "After temperature scaling"),
    ]:
        accs = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
            accs.append(float(correct[mask].mean()) if mask.any() else np.nan)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.bar(
            centers,
            accs,
            width=1.0 / n_bins,
            edgecolor="black",
            alpha=0.7,
            label="observed accuracy",
        )
        ax.set_xlabel("confidence")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
