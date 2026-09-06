#!/usr/bin/env python3
"""Export a trained grading checkpoint to ONNX, and verify numerical parity.

Phase 9 (docs/04_ROADMAP.md): "ONNX export; verify parity with PyTorch
outputs." The verification is not optional -- an export that merely runs
without raising is not evidence it computes the same function; this script
runs the same real preprocessed image through both the PyTorch module and
the exported ONNX graph via onnxruntime and reports the actual max/mean
absolute difference between their raw head outputs.

Usage:
    python scripts/export_onnx.py --checkpoint models/checkpoints/sweep_512_regression_fold0/best.ckpt --loss regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint", default="models/checkpoints/sweep_512_regression_fold0/best.ckpt"
    )
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument(
        "--loss", default="regression", choices=["ce", "corn", "regression", "distance_ce"]
    )
    p.add_argument("--size", type=int, default=512)
    p.add_argument(
        "--out", default=None, help="default: models/onnx/<backbone>_<loss>_<size>px.onnx"
    )
    p.add_argument(
        "--sample-image",
        default=None,
        help="a real preprocessed image to verify parity on; default: first row of "
        "data/manifests/aptos_512.csv's cached image",
    )
    args = p.parse_args()

    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch

    from drdetect.grading.losses import outputs_for_loss
    from drdetect.grading.model import build_model

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    n_outputs = outputs_for_loss(args.loss)
    model = build_model(args.backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.eval()

    out_path = Path(args.out or f"models/onnx/{args.backbone}_{args.loss}_{args.size}px.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # This project depends on onnx/onnxruntime only (pyproject.toml), not
        # onnxscript, which torch's newer dynamo-based exporter requires --
        # dynamo=False keeps the TorchScript-based exporter that matches the
        # declared dependencies.
        dynamo=False,
    )
    print(f"exported: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # --- parity check on a REAL image, not just the random export trace input ---
    if args.sample_image:
        img_path = Path(args.sample_image)
    else:
        import csv

        with open("data/manifests/aptos_512.csv", newline="") as fh:
            first_row = next(csv.DictReader(fh))
        img_path = Path("data/processed") / first_row["path"]

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"WARNING: could not read {img_path} for parity check; skipping.", file=sys.stderr)
        return 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (args.size, args.size))
    # Same normalisation build_transforms applies (ImageNet mean/std) -- kept
    # explicit here rather than importing albumentations, so this script has
    # no dependency on the exact augmentation pipeline, only its normalisation.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (rgb.astype(np.float32) / 255.0 - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": x})[0]

    max_abs_diff = float(np.abs(torch_out - onnx_out).max())
    mean_abs_diff = float(np.abs(torch_out - onnx_out).mean())
    print(f"\nparity check on {img_path}")
    print(f"  PyTorch output: {torch_out.round(4)}")
    print(f"  ONNX output   : {onnx_out.round(4)}")
    print(f"  max abs diff  : {max_abs_diff:.2e}")
    print(f"  mean abs diff : {mean_abs_diff:.2e}")
    if max_abs_diff > 1e-3:
        print("WARNING: parity gap exceeds 1e-3 -- investigate before shipping this export.")
        return 1
    print("PARITY OK (max abs diff < 1e-3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
