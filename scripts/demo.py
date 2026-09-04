#!/usr/bin/env python3
"""Launch the local Gradio demo. See drdetect.serve.demo for the interface itself.

Usage:
    python scripts/demo.py --checkpoint models/checkpoints/cv_baseline_fold1/best.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--loss", default="ce", choices=["ce", "corn", "regression", "distance_ce"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = p.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    from drdetect.serve.demo import build_interface

    demo = build_interface(
        checkpoint_path, backbone=args.backbone, loss_name=args.loss, size=args.size
    )
    demo.launch(share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
