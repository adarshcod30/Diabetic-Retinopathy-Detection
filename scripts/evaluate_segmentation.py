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

Usage:
    python scripts/evaluate_segmentation.py --checkpoint runs/.../best.ckpt
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
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tile-batch-size", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import cv2
    import numpy as np
    import torch

    from drdetect.segmentation.dataset import build_patch_transforms, find_idrid_lesion_pairs
    from drdetect.segmentation.metrics import dice_coefficient, pixel_auprc
    from drdetect.segmentation.model import build_segmentation_model
    from drdetect.segmentation.module import SegmentationModule

    test_pairs = find_idrid_lesion_pairs(args.idrid_root, args.lesion, "test")
    if not test_pairs:
        print(f"No {args.lesion} test pairs found under {args.idrid_root}.", file=sys.stderr)
        return 1
    print(f"lesion: {args.lesion}  test images: {len(test_pairs)}")

    device = torch.device(pick_accelerator())
    model = build_segmentation_model(args.encoder, pretrained=False, classes=1)
    module = SegmentationModule.load_from_checkpoint(
        args.checkpoint, model=model, map_location=device
    )
    module.eval().to(device)
    transform = build_patch_transforms(args.patch_size, train=False)

    all_probs, all_targets = [], []
    per_image = []
    for pair in test_pairs:
        image = cv2.cvtColor(cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
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
        print(f"  {pair.image_id}: {mask.sum()} positive px ({mask.mean() * 100:.3f}%)")

    y_score = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    auprc = pixel_auprc(y_true, y_score)
    dice = dice_coefficient(y_true, y_score > args.threshold)

    print(f"\n{'=' * 60}")
    print(
        f"pooled pixel AUPRC : {auprc:.4f}  (over {y_true.sum():,} / {len(y_true):,} positive px)"
    )
    print(f"pooled Dice @{args.threshold}: {dice:.4f}")

    result = {
        "checkpoint": args.checkpoint,
        "lesion": args.lesion,
        "encoder": args.encoder,
        "n_test_images": len(test_pairs),
        "threshold": args.threshold,
        "pixel_auprc": auprc,
        "dice": dice,
        "total_positive_pixels": int(y_true.sum()),
        "total_pixels": int(len(y_true)),
        "per_image": per_image,
    }
    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "test_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
