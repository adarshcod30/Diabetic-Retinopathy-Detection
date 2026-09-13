#!/usr/bin/env python3
"""The DDR locked-external-test evaluation. Run this exactly once.

Messidor-2/IDRiD's own locked test split (docs/22_PHASE8_VALIDATION_RESULTS.md)
has already been spent and must never be re-run against a new model choice.
Any model built after Phase 8 -- a cross-fold ensemble, a RETFound fine-tune --
needs its OWN, never-before-touched external test set to be judged honestly.
DDR (scripts/fetch_ddr_testset.py) is that set for this round of work, and this
script is its "once": every candidate that will be judged must be decided
BEFORE this runs, using APTOS internal validation only -- never by looking at
DDR first and picking the best-looking candidate afterward.

Differences from evaluate_external.py (which this mirrors structurally):
  * one dataset (DDR test), not two;
  * --checkpoint accepts a comma-separated list per model, so a cross-fold
    ensemble ("ckpt0.ckpt,ckpt1.ckpt,...,ckpt4.ckpt") is one candidate whose
    raw outputs are averaged before decoding -- the same averaging-before-
    decode approach evaluate.py's own --tta flag already uses, just across
    checkpoints instead of across an hflip;
  * --size is per-model (RETFound needs 224px, the CNN needs 512px), and
    per-backbone normalisation stats are applied via backbone_norm_stats,
    which plain evaluate_external.py has no need for (every Phase 8 finalist
    was ImageNet-normalised).

Usage:
    python scripts/evaluate_ddr.py \\
      --checkpoint models/checkpoints/hf_baseline_regression_512/best.ckpt --label shipped_cnn --loss regression --backbone efficientnet_b0 --size 512 \\
      --checkpoint "models/checkpoints/aptos_regression_512_5fold_cuda_fold0/best.ckpt,...,models/checkpoints/aptos_regression_512_5fold_cuda_fold4/best.ckpt" --label cnn_5fold_ensemble --loss regression --backbone efficientnet_b0 --size 512 \\
      --checkpoint models/checkpoints/retfound_finetune_regression_224_fold0/best.ckpt --label retfound --loss regression --backbone "hf_hub:bitfount/RETFound_MAE" --size 224 \\
      --i-understand-this-runs-once-on-ddr
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
DDR_MANIFEST = "data/manifests/ddr_512.csv"


def _score_model(checkpoints, backbone, loss_name, size, device, records, data_root):
    """One forward pass per ensemble member over every DDR test record, raw
    outputs averaged before decoding. len(checkpoints) == 1 is the plain
    single-model case; averaging one thing is a no-op."""
    import cv2
    import numpy as np
    import torch

    from drdetect.data.dataset import build_transforms
    from drdetect.grading.losses import decode_output, outputs_for_loss
    from drdetect.grading.model import backbone_norm_stats, build_model
    from drdetect.quality.assessment import assess_quality

    n_outputs = outputs_for_loss(loss_name)
    mean, std = backbone_norm_stats(backbone)
    transform = build_transforms(size, train=False, mean=mean, std=std)

    models = []
    for checkpoint in checkpoints:
        model = build_model(backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)

    outputs_all, sharpness_all, fov_all = [], [], []
    for i, rec in enumerate(records, 1):
        bgr = cv2.imread(str(Path(data_root) / rec.path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        quality = assess_quality(rgb)
        sharpness_all.append(quality.sharpness)
        fov_all.append(quality.fov_fraction)
        tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            member_outs = [m(tensor).cpu() for m in models]
        outputs_all.append(sum(member_outs) / len(member_outs))
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


def _internal_val_threshold(checkpoints, backbone, loss_name, size, device):
    """The threshold this model (single checkpoint or ensemble) would use,
    frozen from APTOS internal validation -- never re-derived from DDR."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from drdetect.data.dataset import FundusDataset, build_transforms, load_split
    from drdetect.eval.metrics import choose_threshold_for_sensitivity, referable_labels
    from drdetect.grading.losses import decode_output, outputs_for_loss
    from drdetect.grading.model import backbone_norm_stats, build_model

    manifest = Path(f"data/manifests/aptos_{size}.csv")
    _, val_recs, _ = load_split(manifest, fold=0, n_splits=5, seed=42)
    mean, std = backbone_norm_stats(backbone)
    ds = FundusDataset(
        val_recs, "data/processed", build_transforms(size, train=False, mean=mean, std=std)
    )
    dl = DataLoader(ds, batch_size=8, num_workers=0, shuffle=False)

    n_outputs = outputs_for_loss(loss_name)
    models = []
    for checkpoint in checkpoints:
        model = build_model(backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)

    outs, targets = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            member_outs = [m(x).cpu() for m in models]
            outs.append(sum(member_outs) / len(member_outs))
            targets.append(y.numpy())

    _preds, p_ref = decode_output(torch.cat(outs), loss_name)
    y_ref = referable_labels(np.concatenate(targets))
    return choose_threshold_for_sensitivity(y_ref, p_ref, target_sensitivity=0.90)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        dest="checkpoints",
        help="one checkpoint path, or a comma-separated list for an ensemble candidate",
    )
    p.add_argument("--label", action="append", dest="labels", required=True)
    p.add_argument("--backbone", action="append", dest="backbones", required=True)
    p.add_argument(
        "--loss",
        action="append",
        dest="losses",
        required=True,
        choices=["ce", "corn", "regression", "distance_ce"],
    )
    p.add_argument(
        "--size",
        action="append",
        type=int,
        dest="sizes",
        required=True,
        help="one per --checkpoint",
    )
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-worst", type=int, default=20)
    p.add_argument("--out", default="runs/ddr_evaluation.json")
    p.add_argument(
        "--i-understand-this-runs-once-on-ddr",
        action="store_true",
        required=True,
        help="explicit acknowledgement gate -- this scores DDR's test split, this "
        "project's new external test set; every candidate must be decided before "
        "this runs, not chosen after seeing a previous run's output",
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
    if not (len(args.labels) == len(args.losses) == len(args.backbones) == len(args.sizes) == n):
        p.error(
            f"--checkpoint given {n} times but --label/--loss/--backbone/--size given "
            f"{len(args.labels)}/{len(args.losses)}/{len(args.backbones)}/{len(args.sizes)} times "
            "-- give one of each per --checkpoint, in the same order."
        )
    checkpoint_lists = [c.split(",") for c in args.checkpoints]
    device = "cpu"

    if not Path(DDR_MANIFEST).exists():
        print(
            f"{DDR_MANIFEST} not found. Run scripts/preprocess.py --dataset ddr --size 512 first."
        )
        return 1
    all_records = read_manifest(DDR_MANIFEST)
    targets = np.array([r.label for r in all_records])
    print(f"DDR locked test set: {len(all_records)} images")

    results_per_model = {}
    for checkpoints, label, loss_name, backbone, size in zip(
        checkpoint_lists, args.labels, args.losses, args.backbones, args.sizes, strict=True
    ):
        tag = f"{len(checkpoints)}-member ensemble" if len(checkpoints) > 1 else "single checkpoint"
        print(f"\n=== {label}: {tag} (loss={loss_name}, backbone={backbone}, size={size}) ===")
        print("  computing internal-validation threshold (APTOS val, never DDR)...")
        threshold = _internal_val_threshold(checkpoints, backbone, loss_name, size, device)
        print(f"  frozen threshold (from internal val, target sens 0.90): {threshold:.4f}")

        print(f"  scoring {len(all_records)} DDR images...")
        scored = _score_model(
            checkpoints, backbone, loss_name, size, device, all_records, args.data_root
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
        sens_fn = lambda t, s, thr=threshold: evaluate_at_threshold(t, s, thr).sensitivity  # noqa: E731
        spec_fn = lambda t, s, thr=threshold: evaluate_at_threshold(t, s, thr).specificity  # noqa: E731
        _, sens_lo, sens_hi = bootstrap_ci(
            sens_fn, y_ref, p_ref, n_resamples=args.bootstrap, seed=args.seed
        )
        _, spec_lo, spec_hi = bootstrap_ci(
            spec_fn, y_ref, p_ref, n_resamples=args.bootstrap, seed=args.seed
        )
        ece = expected_calibration_error(y_ref, p_ref)

        print(f"\n{'=' * 70}\n{label} -- headline (n={len(all_records)}, DDR test)")
        print(f"{'=' * 70}")
        print(f"QWK                 {qwk:.4f}  95% CI [{qwk_lo:.4f}, {qwk_hi:.4f}]")
        print(
            f"Sensitivity         {scores.sensitivity:.3f}  95% CI [{sens_lo:.3f}, {sens_hi:.3f}]"
        )
        print(
            f"Specificity         {scores.specificity:.3f}  95% CI [{spec_lo:.3f}, {spec_hi:.3f}]"
        )
        print(f"ECE (referable)     {ece:.4f}")

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
                "true_grade": int(targets[i]),
                "predicted_grade": int(preds[i]),
                "referable_score": float(p_ref[i]),
                "sharpness": float(sharpness[i]),
            }
            for i in worst_idx
        ]

        out["models"][label] = {
            "checkpoints": checkpoint_lists[list(results_per_model.keys()).index(label)],
            "threshold": threshold,
            "qwk": qwk,
            "qwk_ci": [qwk_lo, qwk_hi],
            "sensitivity": scores.sensitivity,
            "sensitivity_ci": [sens_lo, sens_hi],
            "specificity": scores.specificity,
            "specificity_ci": [spec_lo, spec_hi],
            "ece": ece,
            "quality_tier_subgroups": subgroups,
            "worst_errors": worst,
            "targets": targets.tolist(),
            "predictions": preds.tolist(),
            "referable_score": p_ref.tolist(),
            "image_ids": [rec.image_id for rec in all_records],
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
        "\nThis script has now scored DDR's test split. Do not re-run it against a "
        "different model choice made after seeing this output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
