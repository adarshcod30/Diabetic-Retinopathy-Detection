#!/usr/bin/env python3
"""Launch the local Gradio demo. See drdetect.serve.demo for the interface itself.

Usage:
    python scripts/demo.py --checkpoint models/checkpoints/sweep_512_regression_fold0/best.ckpt --loss regression

    # cross-fold ensemble (comma-separated), same convention as evaluate_ddr.py:
    python scripts/demo.py --checkpoint "models/checkpoints/fold0/best.ckpt,models/checkpoints/fold1/best.ckpt" --loss regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, help="one path, or a comma-separated list for an ensemble"
    )
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--loss", default="ce", choices=["ce", "corn", "regression", "distance_ce"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = p.parse_args()

    checkpoint_paths = [Path(c) for c in args.checkpoint.split(",")]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
            return 1

    from drdetect.serve.demo import build_interface

    demo = build_interface(
        checkpoint_paths, backbone=args.backbone, loss_name=args.loss, size=args.size
    )
    demo.launch(share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
