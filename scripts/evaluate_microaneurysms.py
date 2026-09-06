#!/usr/bin/env python3
"""Evaluate the microaneurysm candidate-generation + classifier pipeline.

Reports two things that pixel-segmentation evaluation (evaluate_segmentation.py)
doesn't distinguish, because for this lesion type they're genuinely different
stages that can each fail independently:

  - **Generation-stage recall**: the ceiling. A true microaneurysm the
    classical top-hat candidate generator never proposes cannot be recovered
    by the classifier, no matter how good it is.
  - **End-to-end precision/recall at a tuned operating point**, plus
    candidate-level AUPRC (threshold-free ranking quality) -- how well the
    classifier separates true microaneurysms from the much larger pool of
    spurious candidates the generator also proposes.

The operating-point threshold is tuned on the same fold's internal
validation split and frozen before touching the 27 official test images --
the same selection/evaluation separation this project uses everywhere else
(drdetect.eval.metrics, drdetect.segmentation.metrics.best_dice_threshold).

Usage:
    python scripts/evaluate_microaneurysms.py \
        --checkpoint models/checkpoints/microaneurysm_classifier_fold0/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Same reasoning as scripts/train.py and scripts/evaluate_segmentation.py: MPS
# watermarks must be set before torch import or the allocator can
# over-request unified memory. See docs/05_PROTOTYPE_SCOPE.md 6.2.
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
    p.add_argument("--idrid-root", default="data/raw/idrid")
    p.add_argument("--disk-radius", type=int, default=16)
    p.add_argument("--response-percentile", type=float, default=95.0)
    p.add_argument("--patch-size", type=int, default=33)
    p.add_argument(
        "--fold", type=int, default=0, help="must match the fold the checkpoint trained on"
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42, help="must match training's --seed")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import numpy as np
    import torch
    from sklearn.model_selection import KFold

    from drdetect.segmentation.dataset import find_idrid_lesion_pairs
    from drdetect.segmentation.metrics import best_f1_threshold, pixel_auprc
    from drdetect.segmentation.microaneurysms import (
        ClassifierModule,
        PatchClassifier,
        score_candidates,
    )

    train_pairs = find_idrid_lesion_pairs(args.idrid_root, "microaneurysms", "train")
    test_pairs = find_idrid_lesion_pairs(args.idrid_root, "microaneurysms", "test")
    if not train_pairs or not test_pairs:
        print(f"No microaneurysm pairs found under {args.idrid_root}.", file=sys.stderr)
        return 1

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    _, val_idx = list(kf.split(train_pairs))[args.fold]
    val_pairs = [train_pairs[i] for i in val_idx]
    print(
        f"val images (fold {args.fold}): {len(val_pairs)}  official test images: {len(test_pairs)}"
    )

    device = torch.device(pick_accelerator())
    model = PatchClassifier()
    module = ClassifierModule.load_from_checkpoint(
        args.checkpoint, model=model, map_location=device
    )
    module.eval().to(device)
    net = module.model

    def score_pairs(pairs, label):
        import gc

        import cv2

        all_scores, all_labels = [], []
        total_recovered, total_true = 0, 0
        per_image = []
        for pair in pairs:
            image = cv2.imread(str(pair.image_path))
            mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
            result = score_candidates(
                image,
                mask,
                net,
                device=str(device),
                disk_radius=args.disk_radius,
                response_percentile=args.response_percentile,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
            )
            all_scores.append(result["scores"])
            all_labels.append(result["labels"])
            total_recovered += result["n_recovered"]
            total_true += result["n_true"]
            per_image.append(
                {
                    "image_id": pair.image_id,
                    "n_candidates": len(result["candidates"]),
                    "n_true_instances": result["n_true"],
                    "n_recovered_instances": result["n_recovered"],
                }
            )
            print(
                f"  [{label}] {pair.image_id}: {result['n_true']} true instances, "
                f"{result['n_recovered']} recovered by generation, "
                f"{len(result['candidates'])} candidates scored"
            )
            # See build_candidate_examples' own comment: full-resolution candidate
            # generation measurably compounds across images without this.
            del image, mask, result
            gc.collect()
        recall = total_recovered / total_true if total_true else float("nan")
        print(f"  [{label}] generation-stage recall: {recall:.4f} ({total_recovered}/{total_true})")
        return (
            np.concatenate(all_labels).astype(int),
            np.concatenate(all_scores),
            recall,
            per_image,
        )

    val_labels, val_scores, val_recall, val_per_image = score_pairs(val_pairs, "val")
    tuned_threshold, val_f1_at_tuned = best_f1_threshold(val_labels, val_scores)
    print(
        f"  tuned threshold (F1-maximising on val fold {args.fold}): {tuned_threshold:.3f}"
        f"  (val F1 there: {val_f1_at_tuned:.4f})"
    )

    test_labels, test_scores, test_recall, test_per_image = score_pairs(test_pairs, "test")

    def operating_point(labels, scores, threshold, n_images):
        accepted = scores > threshold
        tp = int(np.logical_and(labels == 1, accepted).sum())
        fp = int(np.logical_and(labels == 0, accepted).sum())
        fn = int(np.logical_and(labels == 1, ~accepted).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "threshold": threshold,
            "accepted_candidates": int(accepted.sum()),
            "true_positive_candidates": tp,
            "false_positive_candidates": fp,
            "precision": precision,
            "candidate_recall": recall,
            "f1": f1,
            "avg_false_positives_per_image": fp / n_images if n_images else float("nan"),
        }

    test_auprc = pixel_auprc(test_labels, test_scores) if test_labels.sum() > 0 else float("nan")
    test_op = operating_point(test_labels, test_scores, tuned_threshold, len(test_pairs))

    print(f"\n{'=' * 60}")
    print(f"test candidate-level AUPRC       : {test_auprc:.4f}")
    print(f"test generation-stage recall      : {test_recall:.4f}")
    print(
        f"test end-to-end recall @ tuned th : {test_op['candidate_recall']:.4f}  "
        f"(precision {test_op['precision']:.4f}, "
        f"{test_op['avg_false_positives_per_image']:.1f} FP/image)"
    )

    result = {
        "checkpoint": args.checkpoint,
        "fold": args.fold,
        "disk_radius": args.disk_radius,
        "response_percentile": args.response_percentile,
        "val_generation_recall": val_recall,
        "tuned_threshold": tuned_threshold,
        "val_f1_at_tuned": val_f1_at_tuned,
        "test_generation_recall": test_recall,
        "test_candidate_auprc": test_auprc,
        "test_operating_point": test_op,
        "val_per_image": val_per_image,
        "test_per_image": test_per_image,
    }
    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "test_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
