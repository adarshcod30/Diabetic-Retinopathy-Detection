#!/usr/bin/env python3
"""Measure real training throughput on this machine before committing to a scope.

The Tier-P plan in docs/05_PROTOTYPE_SCOPE.md rests on runtime *estimates*. This
script replaces them with measurements, so the scoping decision is made on data
rather than on a guess about MPS performance.

Usage:
    python scripts/benchmark_device.py
    python scripts/benchmark_device.py --backbone efficientnet_b0 --sizes 384,512,768

Reports, per (resolution, batch size): forward img/s, forward+backward img/s,
peak memory, and the projected wall-clock for one APTOS epoch and a full run.
"""

from __future__ import annotations

import argparse
import sys
import time

APTOS_TRAIN_IMAGES = 2930  # 3,662 x 0.8 train split


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Wait for async GPU work. Without this every timing is a lie."""
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_model(backbone: str, device: str):
    import torch.nn as nn

    try:
        import timm

        model = timm.create_model(backbone, pretrained=False, num_classes=5)
    except ImportError:
        from torchvision import models

        print("  (timm not installed -- falling back to torchvision resnet18)", file=sys.stderr)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
    return model.to(device)


def benchmark(backbone: str, size: int, batch: int, device: str, iters: int, warmup: int):
    import torch
    import torch.nn as nn

    model = build_model(backbone, device)
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randint(0, 5, (batch,), device=device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- forward only (inference path) ---
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync(device)
        fwd = batch * iters / (time.perf_counter() - t0)

    # --- forward + backward (training path) ---
    model.train()
    for _ in range(warmup):
        optimiser.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimiser.step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        optimiser.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimiser.step()
    sync(device)
    train = batch * iters / (time.perf_counter() - t0)

    peak_gb = 0.0
    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        peak_gb = torch.mps.current_allocated_memory() / 1024**3

    del model, optimiser, x, y
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return fwd, train, peak_gb


def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--sizes", default="384,512,768")
    p.add_argument("--batch", default="4,8,16")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40, help="epochs to project a full run")
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Run: make setup", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    print(f"torch {torch.__version__} | device: {device} | backbone: {args.backbone}")
    print(f"projecting against {APTOS_TRAIN_IMAGES} APTOS training images, {args.epochs} epochs\n")

    sizes = [int(s) for s in args.sizes.split(",")]
    batches = [int(b) for b in args.batch.split(",")]

    header = f"{'res':>5} {'batch':>6} {'fwd img/s':>11} {'train img/s':>12} {'peak GB':>9} {'epoch':>8} {'full run':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for size in sizes:
        for batch in batches:
            try:
                fwd, train, mem = benchmark(
                    args.backbone, size, batch, device, args.iters, args.warmup
                )
            except RuntimeError as exc:
                reason = "OOM" if "memory" in str(exc).lower() else type(exc).__name__
                print(f"{size:>5} {batch:>6} {reason:>11} {'-':>12} {'-':>9} {'-':>8} {'-':>9}")
                continue

            epoch_s = APTOS_TRAIN_IMAGES / train
            print(
                f"{size:>5} {batch:>6} {fwd:>11.1f} {train:>12.1f} {mem:>9.2f} "
                f"{fmt_duration(epoch_s):>8} {fmt_duration(epoch_s * args.epochs):>9}"
            )
            rows.append((size, batch, train, epoch_s * args.epochs))

    if not rows:
        print("\nNo configuration completed. Reduce --batch or --sizes.")
        return 1

    print("\nInterpretation (see docs/05_PROTOTYPE_SCOPE.md):")
    best512 = [r for r in rows if r[0] == 512]
    if best512:
        size, batch, _, full = max(best512, key=lambda r: r[2])
        print(f"  512 px, batch {batch}: one full run ~= {fmt_duration(full)}.")
        if full < 10 * 3600:
            print("  -> Overnight per experiment. Tier-P is comfortably feasible locally.")
        elif full < 24 * 3600:
            print("  -> Over a day per run. Cut folds and ablation rows; keep all training data.")
        else:
            print("  -> Too slow locally. Move training to Kaggle; keep this machine for")
            print("     preprocessing, IDRiD segmentation, explainability and the demo.")
    print("  Do NOT respond to slow numbers by subsampling APTOS -- grade 3 has only 193")
    print("  images and collapses first. Cut compute, not data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
