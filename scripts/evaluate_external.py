#!/usr/bin/env python3
"""The Phase 8 locked-external-test evaluation. Run this exactly once.

docs/04_ROADMAP.md, Phase 8: "Evaluate on the locked Messidor-2/IDRiD test
set. Once." This script is the "once": it takes one or more ALREADY-CHOSEN
final checkpoints (chosen using internal APTOS validation only -- see
scripts/train.py runs and docs/21_PHASE8_ABLATION_RESULTS.md -- never by
looking at this data) and, in a single pass, computes every statistic the
roadmap asks for from the SAME frozen predictions: QWK, referable sens/spec
at a threshold frozen from INTERNAL validation (not re-tuned here), bootstrap
CIs, DeLong AUC comparison (if >1 checkpoint given), subgroup analysis by an
approximate quality tier, and a worst-error gallery. Running this script
again with different checkpoints, after having seen its own output, is
exactly the practice Phase 8 exists to prevent -- don't. It has ALREADY BEEN
RUN, once, for this project's own choice: docs/22_PHASE8_VALIDATION_RESULTS.md
is the result, and Messidor-2/IDRiD's grading test split is now spent.

Usage:
    python scripts/evaluate_external.py --checkpoint models/checkpoints/sweep_512_regression_fold0/best.ckpt --loss regression
    python scripts/evaluate_external.py --checkpoint A/best.ckpt --checkpoint B/best.ckpt --label A --label B --loss ce --loss regression
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
DEFAULT_BACKBONE = "efficientnet_b0"
EXTERNAL_MANIFESTS = {
    "messidor2": "data/manifests/messidor2_512.csv",
    "idrid_test": "data/manifests/idrid_test_512.csv",
}


def _score_checkpoint(checkpoint, backbone, loss_name, size, device, records, data_root):
    """One forward pass over every external-test record. Returns per-image
    arrays, kept together so every downstream statistic in this script comes
    from this single frozen scoring pass."""
    import cv2
    import numpy as np
    import torch

    from drdetect.data.dataset import build_transforms
    from drdetect.grading.losses import decode_output, outputs_for_loss
    from drdetect.grading.model import build_model
    from drdetect.quality.assessment import assess_quality

    n_outputs = outputs_for_loss(loss_name)
    model = build_model(backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.to(device).eval()

    transform = build_transforms(size, train=False)
    outputs_all, sharpness_all, fov_all = [], [], []
    for i, rec in enumerate(records, 1):
        bgr = cv2.imread(str(Path(data_root) / rec.path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        quality = assess_quality(rgb)
        sharpness_all.append(quality.sharpness)
        fov_all.append(quality.fov_fraction)
        tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs_all.append(model(tensor).cpu())
        if i % 200 == 0:
            print(f"    {i}/{len(records)}", flush=True)

    outputs = torch.cat(outputs_all)
    preds, p_ref = decode_output(outputs, loss_name)
    return {
        "preds": preds,
        "p_ref": p_ref,
        "sharpness": np.array(sharpness_all),
        "fov_fraction": np.array(fov_all),
    }


def _internal_val_threshold(checkpoint, backbone, loss_name, size, device):
    """The threshold this checkpoint would use, frozen from APTOS internal
    validation -- computed here (not re-derived from the external set) so the
    external evaluation applies an already-fixed operating point, exactly
    like scripts/evaluate.py's own internal-CV usage, just now scored on data
    that threshold has never seen."""
    import torch
    from torch.utils.data import DataLoader

    from drdetect.data.dataset import FundusDataset, build_transforms, load_split
    from drdetect.eval.metrics import choose_threshold_for_sensitivity, referable_labels
    from drdetect.grading.losses import decode_output, outputs_for_loss
    from drdetect.grading.model import build_model

    manifest = Path(f"data/manifests/aptos_{size}.csv")
    _, val_recs, _ = load_split(manifest, fold=0, n_splits=5, seed=42)
    ds = FundusDataset(val_recs, "data/processed", build_transforms(size, train=False))
    dl = DataLoader(ds, batch_size=8, num_workers=0, shuffle=False)

    n_outputs = outputs_for_loss(loss_name)
    model = build_model(backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.to(device).eval()

    outs, targets = [], []
    with torch.no_grad():
        for x, y in dl:
            outs.append(model(x.to(device)).cpu())
            targets.append(y.numpy())
    import numpy as np

    _preds, p_ref = decode_output(torch.cat(outs), loss_name)
    y_ref = referable_labels(np.concatenate(targets))
    return choose_threshold_for_sensitivity(y_ref, p_ref, target_sensitivity=0.90)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", action="append", required=True, dest="checkpoints")
    p.add_argument("--label", action="append", dest="labels", default=None)
    p.add_argument("--backbone", action="append", dest="backbones", default=None)
    p.add_argument(
        "--loss",
        action="append",
        dest="losses",
        default=None,
        choices=["ce", "corn", "regression", "distance_ce"],
        help="one per --checkpoint, same order (finalists can use different losses/backbones -- "
        "e.g. baseline is ce, a regression-loss finalist is regression). Omit entirely to default "
        "every checkpoint to ce.",
    )
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-worst", type=int, default=20)
    p.add_argument("--out", default="runs/external_evaluation.json")
    p.add_argument(
        "--i-understand-this-runs-once",
        action="store_true",
        required=True,
        help="explicit acknowledgement gate -- this script scores the locked external "
        "test set; it must not be re-run against a different model selection made "
        "after seeing a previous run's output",
    )
    args = p.parse_args()

    import numpy as np

    from drdetect.data.manifest import read_manifest
    from drdetect.eval.metrics import (
        bootstrap_ci,
        delong_test,
        evaluate_at_threshold,
        expected_calibration_error,
        quadratic_weighted_kappa,
        referable_labels,
    )

    n = len(args.checkpoints)
    labels = args.labels or [Path(c).parent.name for c in args.checkpoints]
    losses = args.losses or ["ce"] * n
    backbones = args.backbones or [DEFAULT_BACKBONE] * n
    if len(labels) != n or len(losses) != n or len(backbones) != n:
        p.error(
            f"--checkpoint given {n} times but --label/--loss/--backbone given "
            f"{len(labels)}/{len(losses)}/{len(backbones)} times -- give one of each per "
            f"--checkpoint, in the same order, or omit a flag entirely to default it for all."
        )
    device = "cpu"

    all_records = []
    dataset_of = []
    for name, path in EXTERNAL_MANIFESTS.items():
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping {name}", file=sys.stderr)
            continue
        recs = read_manifest(path)
        all_records.extend(recs)
        dataset_of.extend([name] * len(recs))
    if not all_records:
        print("No external manifests found. Run scripts/preprocess.py for messidor2/idrid first.")
        return 1
    dataset_of = np.array(dataset_of)
    targets = np.array([r.label for r in all_records])
    print(
        f"Locked external test set: {len(all_records)} images "
        f"({(dataset_of == 'messidor2').sum()} Messidor-2, "
        f"{(dataset_of == 'idrid_test').sum()} IDRiD test)"
    )

    results_per_model = {}
    for checkpoint, label, loss_name, backbone in zip(
        args.checkpoints, labels, losses, backbones, strict=True
    ):
        print(f"\n=== {label}: {checkpoint} (loss={loss_name}, backbone={backbone}) ===")
        print("  computing internal-validation threshold (APTOS val, never external data)...")
        threshold = _internal_val_threshold(checkpoint, backbone, loss_name, args.size, device)
        print(f"  frozen threshold (from internal val, target sens 0.90): {threshold:.4f}")

        print(f"  scoring {len(all_records)} external images...")
        scored = _score_checkpoint(
            checkpoint, backbone, loss_name, args.size, device, all_records, args.data_root
        )
        results_per_model[label] = {"threshold": threshold, **scored}

    out = {"n_images": len(all_records), "models": {}}
    for label, r in results_per_model.items():
        preds, p_ref, threshold = r["preds"], r["p_ref"], r["threshold"]
        y_ref = referable_labels(targets)

        qwk, qwk_lo, qwk_hi = bootstrap_ci(
            quadratic_weighted_kappa, targets, preds, n_resamples=args.bootstrap, seed=args.seed
        )
        scores = evaluate_at_threshold(y_ref, p_ref, threshold)
        # thr=threshold binds each label's own threshold at definition time --
        # without it every closure would share the loop variable and all read
        # back whichever threshold the LAST label happened to leave behind.
        sens_fn = lambda t, s, thr=threshold: evaluate_at_threshold(t, s, thr).sensitivity  # noqa: E731
        spec_fn = lambda t, s, thr=threshold: evaluate_at_threshold(t, s, thr).specificity  # noqa: E731
        _, sens_lo, sens_hi = bootstrap_ci(
            sens_fn, y_ref, p_ref, n_resamples=args.bootstrap, seed=args.seed
        )
        _, spec_lo, spec_hi = bootstrap_ci(
            spec_fn, y_ref, p_ref, n_resamples=args.bootstrap, seed=args.seed
        )
        ece = expected_calibration_error(y_ref, p_ref)

        print(
            f"\n{'=' * 70}\n{label} -- headline (n={len(all_records)}, combined Messidor-2 + IDRiD)"
        )
        print(f"{'=' * 70}")
        print(f"QWK                 {qwk:.4f}  95% CI [{qwk_lo:.4f}, {qwk_hi:.4f}]")
        print(
            f"Sensitivity         {scores.sensitivity:.3f}  95% CI [{sens_lo:.3f}, {sens_hi:.3f}]"
        )
        print(
            f"Specificity         {scores.specificity:.3f}  95% CI [{spec_lo:.3f}, {spec_hi:.3f}]"
        )
        print(f"ECE (referable)     {ece:.4f}")

        per_dataset = {}
        for ds_name in EXTERNAL_MANIFESTS:
            m = dataset_of == ds_name
            if not m.any():
                continue
            qwk_d = quadratic_weighted_kappa(targets[m], preds[m])
            sc_d = evaluate_at_threshold(y_ref[m], p_ref[m], threshold)
            per_dataset[ds_name] = {
                "n": int(m.sum()),
                "qwk": qwk_d,
                "sensitivity": sc_d.sensitivity,
                "specificity": sc_d.specificity,
                "referable_prevalence": float(y_ref[m].mean()),
            }
            print(
                f"  [{ds_name:>10}] n={int(m.sum()):4d}  QWK {qwk_d:.4f}  "
                f"sens {sc_d.sensitivity:.3f}  spec {sc_d.specificity:.3f}  "
                f"referable prevalence {y_ref[m].mean():.1%}"
            )

        # Subgroup by an approximate quality tier: assess_quality has no
        # 3-tier Good/Usable/Reject output (that classifier was never built,
        # see docs/04_ROADMAP.md Phase 2 note) -- terciled by sharpness among
        # this test set's own images instead, stated as an approximation.
        sharpness = r["sharpness"]
        tiers = np.digitize(sharpness, np.percentile(sharpness, [33.3, 66.7]))
        tier_names = {
            0: "low_sharpness_tercile",
            1: "mid_sharpness_tercile",
            2: "high_sharpness_tercile",
        }
        subgroups = {}
        for tier_idx, tier_name in tier_names.items():
            m = tiers == tier_idx
            if not m.any():
                continue
            sc_t = evaluate_at_threshold(y_ref[m], p_ref[m], threshold)
            subgroups[tier_name] = {
                "n": int(m.sum()),
                "qwk": quadratic_weighted_kappa(targets[m], preds[m]),
                "sensitivity": sc_t.sensitivity,
                "specificity": sc_t.specificity,
            }
            print(
                f"  [{tier_name:>22}] n={int(m.sum()):4d}  "
                f"QWK {subgroups[tier_name]['qwk']:.4f}  sens {sc_t.sensitivity:.3f}  "
                f"spec {sc_t.specificity:.3f}"
            )

        errors = np.abs(preds - targets)
        worst_idx = np.argsort(-errors)[: args.n_worst]
        worst = [
            {
                "image_id": all_records[i].image_id,
                "dataset": str(dataset_of[i]),
                "true_grade": int(targets[i]),
                "predicted_grade": int(preds[i]),
                "referable_score": float(p_ref[i]),
                "sharpness": float(sharpness[i]),
            }
            for i in worst_idx
        ]

        out["models"][label] = {
            "checkpoint": next(
                c for c, lab in zip(args.checkpoints, labels, strict=True) if lab == label
            ),
            "threshold": threshold,
            "qwk": qwk,
            "qwk_ci": [qwk_lo, qwk_hi],
            "sensitivity": scores.sensitivity,
            "sensitivity_ci": [sens_lo, sens_hi],
            "specificity": scores.specificity,
            "specificity_ci": [spec_lo, spec_hi],
            "ece": ece,
            "per_dataset": per_dataset,
            "quality_tier_subgroups": subgroups,
            "worst_errors": worst,
            "targets": targets.tolist(),
            "predictions": preds.tolist(),
            "referable_score": p_ref.tolist(),
            "image_ids": [rec.image_id for rec in all_records],
            "dataset_of": dataset_of.tolist(),
        }

    if len(results_per_model) > 1:
        print(f"\n{'=' * 70}\nDeLong AUC comparisons (paired, same {len(all_records)} images)")
        print(f"{'=' * 70}")
        out["delong"] = []
        label_list = list(results_per_model.keys())
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                a, b = label_list[i], label_list[j]
                y_ref = referable_labels(targets)
                dl = delong_test(
                    y_ref, results_per_model[a]["p_ref"], results_per_model[b]["p_ref"]
                )
                print(
                    f"  {a} (AUC {dl.auc_a:.4f}) vs {b} (AUC {dl.auc_b:.4f}): z={dl.z:.3f}  p={dl.p:.4f}"
                )
                out["delong"].append(
                    {"a": a, "b": b, "auc_a": dl.auc_a, "auc_b": dl.auc_b, "z": dl.z, "p": dl.p}
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")
    print(
        "\nThis script has now scored the locked external test set. Do not re-run it "
        "against a different model chosen after seeing this output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
