#!/usr/bin/env python3
"""Grade a single fundus photo end to end: quality gate -> grade -> PDF report.

This is the Phase 2 exit criterion (docs/04_ROADMAP.md): hand it a JPEG, get a
PDF back. Deliberately CPU-only -- see `drdetect.serve.pipeline.load_grader`.

Usage:
    python scripts/predict.py --image photo.jpg --checkpoint models/checkpoints/cv_baseline_fold1/best.ckpt
    python scripts/predict.py --image photo.jpg --checkpoint best.ckpt --out reports/photo.pdf
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
        "--image", required=True, help="path to a fundus photo (any common image format)"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--loss", default="ce", choices=["ce", "corn", "regression", "distance_ce"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default=None, help="default: runs/reports/<image stem>.pdf")
    p.add_argument(
        "--force",
        action="store_true",
        help="grade even if the image fails the quality gate (report will say so)",
    )
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    import cv2

    from drdetect.calibration.temperature import load_temperature
    from drdetect.serve.pipeline import load_grader, run_pipeline
    from drdetect.serve.report import build_report_pdf

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not read {image_path} as an image.", file=sys.stderr)
        return 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"loading  : {checkpoint_path}")
    model = load_grader(checkpoint_path, backbone=args.backbone, loss_name=args.loss, device="cpu")
    temperature = load_temperature(checkpoint_path)
    calib_note = (
        f"T={temperature:.3f}"
        if temperature != 1.0
        else "none fitted -- confidence is uncalibrated"
    )
    print(f"calibration: {calib_note}")

    print(f"grading  : {image_path}")
    result = run_pipeline(
        image,
        model,
        loss_name=args.loss,
        size=args.size,
        device="cpu",
        skip_quality_gate=args.force,
        temperature=temperature,
    )

    if not result.quality.usable and not args.force:
        print("QUALITY GATE: rejected -- image not graded. Reasons:")
        for reason in result.quality.reasons:
            print(f"  - {reason}")
        print("Pass --force to grade anyway (the report will still show the rejection).")
    else:
        print(f"grade    : {result.grade} ({result.grade_name})")
        print(f"referable: {result.referable}")
        if result.confidence is not None:
            label = "temperature-scaled" if result.calibrated else "uncalibrated"
            print(f"confidence ({label}): {result.confidence:.2%}")

    out_path = Path(args.out) if args.out else Path("runs/reports") / f"{image_path.stem}.pdf"
    build_report_pdf(result, image_path.name, out_path, checkpoint_name=checkpoint_path.name)
    print(f"report   : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
