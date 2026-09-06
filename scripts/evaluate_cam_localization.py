#!/usr/bin/env python3
"""Phase 6's exit criterion, localisation half: pointing-game + IoU vs IDRiD masks.

Runs the baseline grading checkpoint on IDRiD's 81 segmentation-subset images
(the only IDRiD images with real per-lesion pixel masks -- "B. Disease
Grading" and "C. Localization" share a different, disjoint 516-image set),
explains each prediction with 3 CAM methods, and checks each heatmap against
every lesion type that image actually has a mask for.

Score-CAM is deliberately excluded -- docs/08_PHASE6_RESULTS.md measured
~15 minutes per single Score-CAM computation on this project's hardware;
across 81 images that is many hours for one method, the same cost that
excluded it from the sanity-check cascade.

This is NOT a Phase 8 locked-test-set evaluation: it doesn't touch model
weights, thresholds, or any training decision -- it's a post-hoc
interpretability measurement on an already-frozen checkpoint, which is what
the roadmap's own Phase 6 section specifies IDRiD's masks for.

Usage:
    python scripts/evaluate_cam_localization.py --checkpoint models/checkpoints/baseline_effb0_512_fold0/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LESION_TYPES = ["microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates", "optic_disc"]
CAM_METHODS = ["gradcam", "gradcam++", "eigencam"]


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
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="cap the number of images (smoke test)")
    p.add_argument("--out", default="runs/cam_localization_evaluation.json")
    args = p.parse_args()

    import cv2
    import numpy as np
    import torch

    from drdetect.data.dataset import build_transforms
    from drdetect.explain.cam_variants import generate_cam_variant
    from drdetect.explain.localization_metrics import cam_mask_iou, pointing_game_hit, upsample_cam
    from drdetect.grading.model import build_model
    from drdetect.segmentation.dataset import find_idrid_lesion_pairs

    # Gather every IDRiD segmentation-subset image and whichever lesion masks
    # it has (not every image has every lesion type).
    images: dict[str, dict] = {}
    for lesion in LESION_TYPES:
        for split in ["train", "test"]:
            for pair in find_idrid_lesion_pairs(args.idrid_root, lesion, split):
                entry = images.setdefault(
                    pair.image_id, {"image_path": pair.image_path, "masks": {}}
                )
                entry["masks"][lesion] = pair.mask_path
    print(f"IDRiD segmentation-subset images found: {len(images)}")
    if args.limit:
        images = dict(list(images.items())[: args.limit])
        print(f"--limit applied: using {len(images)} images")

    device = torch.device(pick_accelerator())
    model = build_model(args.backbone, num_outputs=5, pretrained=False, freeze_bn=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.to(device).eval()
    transform = build_transforms(args.size, train=False)

    # hits[method][lesion] = [bool, ...]; ious[method][lesion] = [float, ...]
    hits = defaultdict(lambda: defaultdict(list))
    ious = defaultdict(lambda: defaultdict(list))
    grade_counts = defaultdict(int)
    # Chance-level pointing-game rate per lesion type: a uniformly random pixel
    # hits with probability equal to the mask's own area fraction. Needed
    # because raw pointing-game percentages are meaningless without it -- a
    # lesion covering 0.1% of the image "should" score ~0.1% by pure chance,
    # so even a low-looking raw number can be a large multiple of chance.
    chance_area_fractions = defaultdict(list)

    for i, (image_id, entry) in enumerate(images.items()):
        image_bgr = cv2.imread(str(entry["image_path"]), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        native_shape = image_rgb.shape[:2]

        tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_class = int(model(tensor).argmax(dim=1).item())
        grade_counts[pred_class] += 1

        masks = {}
        for lesion, mask_path in entry["masks"].items():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.sum() == 0:
                continue
            masks[lesion] = mask
            chance_area_fractions[lesion].append(float((mask > 0).mean()))

        for method in CAM_METHODS:
            cam = generate_cam_variant(method, model, tensor, pred_class)
            cam_full = upsample_cam(cam, native_shape)

            for lesion, mask in masks.items():
                hits[method][lesion].append(pointing_game_hit(cam_full, mask))
                iou = cam_mask_iou(cam_full, mask, threshold=args.iou_threshold)
                if not np.isnan(iou):
                    ious[method][lesion].append(iou)

        print(f"[{i + 1}/{len(images)}] {image_id}: predicted grade {pred_class}")

    print(f"\npredicted grade distribution: {dict(grade_counts)}")

    chance_rate = {lesion: float(np.mean(fracs)) for lesion, fracs in chance_area_fractions.items()}
    print("\nchance-level pointing-game rate per lesion (mean mask area fraction):")
    for lesion in LESION_TYPES:
        if lesion in chance_rate:
            print(f"  {lesion:16s} {chance_rate[lesion] * 100:.4f}%")

    result = {
        "n_images": len(images),
        "iou_threshold": args.iou_threshold,
        "chance_pointing_game_rate": chance_rate,
        "methods": {},
    }
    print(f"\n{'=' * 70}")
    for method in CAM_METHODS:
        print(f"\n{method}:")
        result["methods"][method] = {}
        for lesion in LESION_TYPES:
            h = hits[method].get(lesion, [])
            u = ious[method].get(lesion, [])
            if not h:
                continue
            pointing_acc = float(np.mean(h))
            mean_iou = float(np.mean(u)) if u else float("nan")
            chance = chance_rate.get(lesion, float("nan"))
            vs_chance = pointing_acc / chance if chance > 0 else float("nan")
            result["methods"][method][lesion] = {
                "n_images": len(h),
                "pointing_game_accuracy": pointing_acc,
                "pointing_game_vs_chance": vs_chance,
                "mean_iou": mean_iou,
            }
            print(
                f"  {lesion:16s} n={len(h):3d}  pointing-game={pointing_acc:.4f} "
                f"({vs_chance:.1f}x chance)  IoU={mean_iou:.4f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
