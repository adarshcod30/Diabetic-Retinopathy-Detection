#!/usr/bin/env python3
"""Score a trained OD/fovea localisation checkpoint on IDRiD's official 103 test images.

Usage:
    python scripts/evaluate_localization.py --checkpoint runs/.../best.ckpt
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--idrid-root", default="data/raw/idrid")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import numpy as np
    import torch

    from drdetect.segmentation.localization import (
        IDRiDLocalizationDataset,
        LocalizationModule,
        euclidean,
        find_idrid_localization_pairs,
        heatmap_argmax,
        working_res_od_diameter,
    )
    from drdetect.segmentation.model import build_segmentation_model

    test_pairs = find_idrid_localization_pairs(args.idrid_root, "test")
    if not test_pairs:
        print(f"No IDRiD localisation test pairs found under {args.idrid_root}.", file=sys.stderr)
        return 1
    print(f"test images: {len(test_pairs)}")

    device = torch.device(pick_accelerator())
    model = build_segmentation_model(args.encoder, pretrained=False, classes=2)
    module = LocalizationModule.load_from_checkpoint(
        args.checkpoint, model=model, map_location=device
    )
    module.eval().to(device)

    size = (args.height, args.width)
    diameter = working_res_od_diameter(size)
    ds = IDRiDLocalizationDataset(test_pairs, size=size, train=False)

    od_errors, fovea_errors, per_image = [], [], []
    with torch.no_grad():
        for i, pair in enumerate(test_pairs):
            image, _, targets_xy = ds[i]
            logits = module(image.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            od_pred = heatmap_argmax(probs[0])
            fovea_pred = heatmap_argmax(probs[1])
            od_err = euclidean(od_pred, tuple(targets_xy[0].tolist())) / diameter
            fovea_err = euclidean(fovea_pred, tuple(targets_xy[1].tolist())) / diameter
            od_errors.append(od_err)
            fovea_errors.append(fovea_err)
            per_image.append(
                {
                    "image_id": pair.image_id,
                    "od_error_diameters": od_err,
                    "fovea_error_diameters": fovea_err,
                }
            )

    od_mean, fovea_mean = float(np.mean(od_errors)), float(np.mean(fovea_errors))
    mean_error = (od_mean + fovea_mean) / 2.0

    print(f"\n{'=' * 60}")
    print(
        f"OD error    : mean {od_mean:.4f}  median {np.median(od_errors):.4f}  max {np.max(od_errors):.4f}  (diameters)"
    )
    print(
        f"Fovea error : mean {fovea_mean:.4f}  median {np.median(fovea_errors):.4f}  max {np.max(fovea_errors):.4f}  (diameters)"
    )
    print(f"Combined mean: {mean_error:.4f}  (target < 0.5)")
    n_od_pass = sum(1 for e in od_errors if e < 0.5)
    n_fovea_pass = sum(1 for e in fovea_errors if e < 0.5)
    print(f"Images with OD error < 0.5:    {n_od_pass}/{len(test_pairs)}")
    print(f"Images with fovea error < 0.5: {n_fovea_pass}/{len(test_pairs)}")

    result = {
        "checkpoint": args.checkpoint,
        "n_test_images": len(test_pairs),
        "working_od_diameter_px": diameter,
        "od_error_mean": od_mean,
        "od_error_median": float(np.median(od_errors)),
        "od_error_max": float(np.max(od_errors)),
        "fovea_error_mean": fovea_mean,
        "fovea_error_median": float(np.median(fovea_errors)),
        "fovea_error_max": float(np.max(fovea_errors)),
        "combined_mean_error": mean_error,
        "n_od_under_target": n_od_pass,
        "n_fovea_under_target": n_fovea_pass,
        "per_image": per_image,
    }
    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "test_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
