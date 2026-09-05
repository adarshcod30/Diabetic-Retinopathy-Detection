#!/usr/bin/env python3
"""Score a trained segmentation checkpoint on IDRiD's official held-out test images.

Full IDRiD images are 2848x4288 -- far too large to push through a
segmentation decoder whole (see drdetect.segmentation.dataset module
docstring for the measured memory reasoning). This script tiles each test
image into non-overlapping patch_size squares (reflect-padding up to the
next multiple so no tile is cropped), runs inference tile by tile, stitches
the probability map back to full resolution, then crops off the padding
before scoring -- so padded pixels never enter the metric.

Metrics are pooled across every test image's pixels (see
drdetect.segmentation.metrics.pixel_auprc docstring for why pooled, not
averaged per image: several IDRiD test images have zero lesion pixels for a
given lesion type, and per-image AUPRC is undefined there).

Dice needs a threshold; AUPRC does not. Rather than the common but unsafe
default of a fixed 0.5 (see drdetect.segmentation.metrics.best_dice_threshold
for a measured case where that reads Dice 0.037 on a model AUPRC independently
scores at 0.409), the threshold is tuned on the checkpoint's own internal
validation fold -- reconstructed here from --fold/--n-splits/--seed, which
must match how the checkpoint was trained -- and only then frozen and applied
to the 27 test images. Dice@0.5 is still reported alongside it, for
continuity with earlier results and as a sanity check on how much the tuned
threshold actually mattered.

Usage:
    python scripts/evaluate_segmentation.py --checkpoint runs/.../best.ckpt
    python scripts/evaluate_segmentation.py --checkpoint .../fold2/best.ckpt --fold 2
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


def tile_predict(model, image, patch_size: int, device, transform, batch_size: int = 4):
    """Return a float32 probability map the same H x W as `image`.

    Non-overlapping tiles over reflect-padding, run in mini-batches, stitched
    back and cropped to the original size.
    """
    import cv2
    import numpy as np
    import torch

    h, w = image.shape[:2]
    ph = (patch_size - h % patch_size) % patch_size
    pw = (patch_size - w % patch_size) % patch_size
    padded = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    ph_full, pw_full = padded.shape[:2]

    tiles, positions = [], []
    for top in range(0, ph_full, patch_size):
        for left in range(0, pw_full, patch_size):
            tiles.append(padded[top : top + patch_size, left : left + patch_size])
            positions.append((top, left))

    prob_map = np.zeros((ph_full, pw_full), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            chunk = tiles[start : start + batch_size]
            batch = torch.stack([transform(image=t)["image"] for t in chunk]).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            for (top, left), prob in zip(positions[start : start + batch_size], probs, strict=True):
                prob_map[top : top + patch_size, left : left + patch_size] = prob

    return prob_map[:h, :w]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--idrid-root", default="data/raw/idrid")
    p.add_argument(
        "--lesion",
        default="hard_exudates",
        choices=["microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates", "optic_disc"],
        help="must match the lesion type the checkpoint was trained on",
    )
    p.add_argument(
        "--encoder", default="resnet34", help="must match how the checkpoint was trained"
    )
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument(
        "--fold", type=int, default=0, help="must match the fold the checkpoint was trained on"
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42, help="must match training's --seed")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="fixed threshold to ALSO report Dice at, for continuity with earlier results",
    )
    p.add_argument("--tile-batch-size", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import cv2
    import numpy as np
    import torch
    from sklearn.model_selection import KFold

    from drdetect.segmentation.dataset import build_patch_transforms, find_idrid_lesion_pairs
    from drdetect.segmentation.metrics import best_dice_threshold, dice_coefficient, pixel_auprc
    from drdetect.segmentation.model import build_segmentation_model
    from drdetect.segmentation.module import SegmentationModule

    train_pairs = find_idrid_lesion_pairs(args.idrid_root, args.lesion, "train")
    test_pairs = find_idrid_lesion_pairs(args.idrid_root, args.lesion, "test")
    if not train_pairs or not test_pairs:
        print(f"No {args.lesion} pairs found under {args.idrid_root}.", file=sys.stderr)
        return 1

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    _, val_idx = list(kf.split(train_pairs))[args.fold]
    val_pairs = [train_pairs[i] for i in val_idx]
    print(
        f"lesion: {args.lesion}  val images (fold {args.fold}): {len(val_pairs)}  "
        f"test images: {len(test_pairs)}"
    )

    device = torch.device(pick_accelerator())
    model = build_segmentation_model(args.encoder, pretrained=False, classes=1)
    module = SegmentationModule.load_from_checkpoint(
        args.checkpoint, model=model, map_location=device
    )
    module.eval().to(device)
    transform = build_patch_transforms(args.patch_size, train=False)

    def predict_pairs(pairs, label):
        all_probs, all_targets = [], []
        per_image = []
        for pair in pairs:
            image = cv2.cvtColor(
                cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            mask = (cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            prob_map = tile_predict(
                module, image, args.patch_size, device, transform, batch_size=args.tile_batch_size
            )
            assert prob_map.shape == mask.shape, (
                f"{pair.image_id}: {prob_map.shape} vs mask {mask.shape}"
            )
            all_probs.append(prob_map.ravel())
            all_targets.append(mask.ravel())
            per_image.append(
                {
                    "image_id": pair.image_id,
                    "positive_pixels": int(mask.sum()),
                    "positive_fraction": float(mask.mean()),
                }
            )
            print(
                f"  [{label}] {pair.image_id}: {mask.sum()} positive px ({mask.mean() * 100:.3f}%)"
            )
        return np.concatenate(all_targets), np.concatenate(all_probs), per_image

    val_y_true, val_y_score, val_per_image = predict_pairs(val_pairs, "val")
    tuned_threshold, val_dice_at_tuned = best_dice_threshold(val_y_true, val_y_score)
    print(
        f"  tuned threshold (Dice-maximising on val fold {args.fold}): {tuned_threshold:.3f}"
        f"  (val Dice there: {val_dice_at_tuned:.4f})"
    )

    test_y_true, test_y_score, test_per_image = predict_pairs(test_pairs, "test")

    auprc = pixel_auprc(test_y_true, test_y_score)
    dice_fixed = dice_coefficient(test_y_true, test_y_score > args.threshold)
    dice_tuned = dice_coefficient(test_y_true, test_y_score > tuned_threshold)

    print(f"\n{'=' * 60}")
    print(
        f"pooled pixel AUPRC             : {auprc:.4f}  "
        f"(over {test_y_true.sum():,} / {len(test_y_true):,} positive px)"
    )
    print(f"pooled Dice @{args.threshold} (fixed)      : {dice_fixed:.4f}")
    print(f"pooled Dice @{tuned_threshold:.3f} (tuned on val): {dice_tuned:.4f}")

    result = {
        "checkpoint": args.checkpoint,
        "lesion": args.lesion,
        "encoder": args.encoder,
        "fold": args.fold,
        "n_splits": args.n_splits,
        "n_val_images": len(val_pairs),
        "n_test_images": len(test_pairs),
        "fixed_threshold": args.threshold,
        "dice_at_fixed_threshold": dice_fixed,
        "tuned_threshold": tuned_threshold,
        "val_dice_at_tuned_threshold": val_dice_at_tuned,
        "dice_at_tuned_threshold": dice_tuned,
        "pixel_auprc": auprc,
        "total_positive_pixels": int(test_y_true.sum()),
        "total_pixels": int(len(test_y_true)),
        "per_image": test_per_image,
        "val_per_image": val_per_image,
    }
    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "test_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
