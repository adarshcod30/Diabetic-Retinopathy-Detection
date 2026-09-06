#!/usr/bin/env python3
"""Measure real end-to-end INFERENCE throughput, for Phase 7's simulation.

This is a different measurement from `scripts/benchmark_device.py` (Phase 0),
which timed *training* forward+backward passes to plan compute budget. Phase 7
needs *deployed-system* inference latency: one image at a time, eval mode,
CPU (the Phase 2 pipeline is deliberately CPU-only, see
`drdetect.serve.pipeline.load_grader`) -- because that is what actually runs
per uploaded image in the district-screening simulation, and the roadmap
explicitly says "measured from your own model... not guessed."

Two numbers are measured, because the real pipeline is two-tier:
  1. Quality gate + grading only -- runs on EVERY uploaded image.
  2. + full lesion evidence (5 extra models, tiled segmentation at full
     resolution) -- only needed for images a human will actually review, so
     it is measured separately rather than assumed to run on every image.

Usage:
    python scripts/benchmark_inference.py --checkpoint models/checkpoints/sweep_512_regression_fold0/best.ckpt --loss regression
    python scripts/benchmark_inference.py --n-images 50 --n-full-pipeline 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

IDRID_GRADING_DIR = Path("data/raw/idrid/B. Disease Grading/1. Original Images/a. Training Set")


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
    p.add_argument("--device", default="cpu", help="the deployed pipeline is CPU-only by design")
    p.add_argument("--n-images", type=int, default=40, help="images for the grading-only timing")
    p.add_argument("--n-full-pipeline", type=int, default=8, help="images for the +evidence timing")
    p.add_argument("--out", default="runs/inference_benchmark.json")
    args = p.parse_args()

    import cv2
    import torch

    from drdetect.calibration.temperature import load_temperature
    from drdetect.serve.pipeline import add_lesion_evidence, load_grader, run_pipeline

    images = sorted(IDRID_GRADING_DIR.glob("*.jpg"))
    if not images:
        print(f"No images found under {IDRID_GRADING_DIR}", file=sys.stderr)
        return 1
    file_sizes_kb = [im.stat().st_size / 1024 for im in images]
    print(f"found {len(images)} real full-resolution fundus photos in {IDRID_GRADING_DIR}")
    print(
        f"file size (KB): mean {statistics.mean(file_sizes_kb):.1f}, "
        f"median {statistics.median(file_sizes_kb):.1f}, "
        f"min {min(file_sizes_kb):.1f}, max {max(file_sizes_kb):.1f}"
    )

    checkpoint_path = Path(args.checkpoint)
    print(f"\nloading grading checkpoint: {checkpoint_path} on {args.device}")
    model = load_grader(
        checkpoint_path, backbone=args.backbone, loss_name=args.loss, device=args.device
    )
    temperature = load_temperature(checkpoint_path)

    n_grading = min(args.n_images, len(images))
    grading_ms: list[float] = []
    print(f"\ntiming grading-only pipeline over {n_grading} images...")
    for i, path in enumerate(images[:n_grading], 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        result = run_pipeline(
            rgb,
            model,
            loss_name=args.loss,
            size=args.size,
            device=args.device,
            temperature=temperature,
        )
        grading_ms.append((time.perf_counter() - t0) * 1000)
        if i % 10 == 0:
            print(f"  {i}/{n_grading}")
    del result

    print(f"\nloading 5 lesion-evidence models on {args.device} (Phase 4/6 checkpoints)...")
    from drdetect.fusion.features import load_lesion_models

    lesion_models = load_lesion_models(
        localization_checkpoint="models/checkpoints/localization_resnet34_512x768_fold0/best.ckpt",
        hard_exudate_checkpoint=(
            "models/checkpoints/segmentation_hard_exudates_resnet34_512px_fold0/best.ckpt"
        ),
        soft_exudate_checkpoint=(
            "models/checkpoints/segmentation_soft_exudates_resnet34_512px_fold0/best.ckpt"
        ),
        haemorrhage_checkpoint=(
            "models/checkpoints/segmentation_haemorrhages_resnet34_512px_fold0/best.ckpt"
        ),
        microaneurysm_checkpoint="models/checkpoints/microaneurysm_classifier_fold0/best.ckpt",
        device=torch.device(args.device),
    )

    n_full = min(args.n_full_pipeline, len(images))
    full_ms: list[float] = []
    evidence_only_ms: list[float] = []
    print(f"\ntiming +lesion-evidence pipeline over {n_full} images...")
    for i, path in enumerate(images[:n_full], 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        result = run_pipeline(
            rgb,
            model,
            loss_name=args.loss,
            size=args.size,
            device=args.device,
            temperature=temperature,
        )
        t1 = time.perf_counter()
        result = add_lesion_evidence(result, path, lesion_models)
        t2 = time.perf_counter()
        full_ms.append((t2 - t0) * 1000)
        evidence_only_ms.append((t2 - t1) * 1000)
        print(
            f"  {i}/{n_full}  grading {(t1 - t0) * 1000:.0f}ms  +evidence {(t2 - t1) * 1000:.0f}ms"
        )

    def stats(xs: list[float]) -> dict:
        return {
            "mean_ms": statistics.mean(xs),
            "median_ms": statistics.median(xs),
            "std_ms": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min_ms": min(xs),
            "max_ms": max(xs),
            "images_per_sec": 1000.0 / statistics.mean(xs),
            "n": len(xs),
        }

    grading_stats = stats(grading_ms)
    full_stats = stats(full_ms)
    evidence_stats = stats(evidence_only_ms)

    print("\n" + "=" * 66)
    print(f"device: {args.device}  |  grading size: {args.size}px  |  backbone: {args.backbone}")
    print(
        f"grading-only   : {grading_stats['mean_ms']:.0f} ms/image "
        f"({grading_stats['images_per_sec']:.2f} img/s), n={grading_stats['n']}"
    )
    print(
        f"+lesion evidence (5 extra models): {evidence_stats['mean_ms']:.0f} ms/image extra, "
        f"total {full_stats['mean_ms']:.0f} ms/image "
        f"({full_stats['images_per_sec']:.2f} img/s), n={full_stats['n']}"
    )
    print("=" * 66)

    out = {
        "device": args.device,
        "size": args.size,
        "backbone": args.backbone,
        "checkpoint": str(checkpoint_path),
        "file_size_kb": {
            "mean": statistics.mean(file_sizes_kb),
            "median": statistics.median(file_sizes_kb),
            "min": min(file_sizes_kb),
            "max": max(file_sizes_kb),
            "n": len(file_sizes_kb),
            "source": str(IDRID_GRADING_DIR),
        },
        "grading_only": grading_stats,
        "lesion_evidence_extra": evidence_stats,
        "full_pipeline": full_stats,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
