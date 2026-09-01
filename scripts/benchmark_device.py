#!/usr/bin/env python3
"""Measure real training throughput on this machine before committing to a scope.

The Tier-P plan in docs/05_PROTOTYPE_SCOPE.md rests on runtime *estimates*. This
script replaces them with measurements.

SAFETY NOTE -- why this script is defensive about memory
--------------------------------------------------------
On CUDA, an over-large batch raises a catchable OutOfMemoryError: one config
fails, the sweep continues. On Apple Silicon (MPS) there is no separate VRAM and
no hard boundary. PyTorch's MPS allocator defaults to a high-watermark ratio of
1.7 -- it may request ~1.7x physical RAM -- and macOS honours that by swapping
other applications out. The result is not a Python exception but system-wide
memory exhaustion and the "your system has run out of application memory"
dialog.

So `try/except RuntimeError` is NOT sufficient protection on MPS. This script
instead:

  1. sets PYTORCH_MPS_HIGH_WATERMARK_RATIO before importing torch, so the
     allocator raises instead of swallowing the machine;
  2. estimates activation memory analytically and SKIPS configs over budget
     before allocating anything;
  3. budgets against actually-free RAM, not total RAM;
  4. sweeps smallest-first and stops escalating batch size at a resolution once
     one fails, rather than pushing further into the danger zone;
  5. excludes large-resolution/large-batch configs from the defaults entirely.

Usage:
    python scripts/benchmark_device.py                 # safe defaults
    python scripts/benchmark_device.py --budget-gb 8   # raise the cap deliberately
    python scripts/benchmark_device.py --sizes 768 --batch 4 --budget-gb 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# MUST be set before torch is imported.
#   high = hard ceiling: allocations beyond this fraction of recommended max FAIL
#          (instead of overcommitting into swap and taking the machine down).
#   low  = when the allocator starts releasing cached blocks.
# PyTorch requires low <= high, and the default low is 1.4 -- so setting high
# alone raises "invalid low watermark ratio 1.4". Both must move together.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

APTOS_TRAIN_IMAGES = 2930  # 3,662 x 0.8 train split

# Peak training memory is NOT purely batch x area. There is a fixed floor
# (parameters + AdamW moments + Metal/CUDA context) that does not scale, plus an
# activation term that does. Modelling it as purely linear underestimates small
# configs badly -- measured 2.19 GB where a linear fit predicted 1.2 GB.
#
#     peak_gb ~= FIXED_OVERHEAD_GB[bb] + batch * (size/224)^2 * PER_IMAGE_GB[bb]
#
# Constants below are calibrated against measurements on Apple M4 / torch 2.13
# and are deliberately rounded UP. Under-estimating here does not cost a slow
# run, it costs the whole machine.
PER_IMAGE_GB_AT_224 = {
    "efficientnet_b0": 0.15,
    "efficientnet_b1": 0.19,
    "efficientnet_b2": 0.22,
    "efficientnetv2_s": 0.42,
    "convnext_tiny": 0.36,
    "resnet18": 0.09,
    "resnet50": 0.27,
    "swin_tiny_patch4_window7_224": 0.45,
}
FIXED_OVERHEAD_GB = {
    "efficientnet_b0": 0.5,
    "efficientnet_b1": 0.6,
    "efficientnet_b2": 0.6,
    "efficientnetv2_s": 1.0,
    "convnext_tiny": 0.9,
    "resnet18": 0.4,
    "resnet50": 0.8,
    "swin_tiny_patch4_window7_224": 1.0,
}
DEFAULT_PER_IMAGE_GB = 0.30  # unknown backbone -> assume expensive
DEFAULT_FIXED_GB = 1.0


def estimate_gb(backbone: str, size: int, batch: int) -> float:
    per_img = PER_IMAGE_GB_AT_224.get(backbone, DEFAULT_PER_IMAGE_GB)
    fixed = FIXED_OVERHEAD_GB.get(backbone, DEFAULT_FIXED_GB)
    return fixed + batch * ((size / 224.0) ** 2) * per_img


def free_ram_gb() -> float:
    """Free + inactive physical memory in GB (macOS vm_stat, else psutil)."""
    try:
        import subprocess

        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 16384
        stats = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                if v.isdigit():
                    stats[k.strip()] = int(v)
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
        pages = stats.get("Pages free", 0) + stats.get("Pages inactive", 0)
        if pages:
            return pages * page / 1024**3
    except Exception:
        pass
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return 8.0  # conservative fallback


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


def release(device: str) -> None:
    """Free cached allocator blocks. Never raises -- this runs in `finally` and
    in exception handlers, where a secondary error would mask the real one."""
    try:
        import torch

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- cleanup must be total
        pass


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

    try:
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
            peak_gb = torch.mps.driver_allocated_memory() / 1024**3
    finally:
        del model, optimiser, x, y
        release(device)

    return fwd, train, peak_gb


def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--sizes", default="384,512", help="default excludes 768 -- see --budget-gb")
    p.add_argument("--batch", default="4,8")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40, help="epochs to project a full run")
    p.add_argument(
        "--budget-gb",
        type=float,
        default=None,
        help="max estimated memory per config; defaults to 60%% of free RAM, capped at 6 GB",
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Run: make setup", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    free_gb = free_ram_gb()
    budget = args.budget_gb if args.budget_gb is not None else min(6.0, free_gb * 0.6)

    print(f"torch {torch.__version__} | device: {device} | backbone: {args.backbone}")
    print(f"free RAM: {free_gb:.1f} GB | per-config memory budget: {budget:.1f} GB")
    if device == "mps":
        print(
            f"MPS watermarks: low={os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO']} "
            f"high={os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']} "
            "(allocator refuses rather than swaps)"
        )
    print(f"projecting against {APTOS_TRAIN_IMAGES} APTOS training images, {args.epochs} epochs\n")

    sizes = sorted(int(s) for s in args.sizes.split(","))
    batches = sorted(int(b) for b in args.batch.split(","))

    header = (
        f"{'res':>5} {'batch':>6} {'est GB':>8} {'fwd img/s':>11} "
        f"{'train img/s':>12} {'peak GB':>9} {'epoch':>8} {'full run':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    skipped = []
    for size in sizes:  # smallest resolution first
        for batch in batches:  # smallest batch first
            est = estimate_gb(args.backbone, size, batch)
            if est > budget:
                print(f"{size:>5} {batch:>6} {est:>8.1f} {'SKIPPED -- over budget':>44}")
                skipped.append((size, batch, est))
                break  # larger batches at this resolution are only worse
            try:
                fwd, train, mem = benchmark(
                    args.backbone, size, batch, device, args.iters, args.warmup
                )
            except (RuntimeError, MemoryError) as exc:
                reason = "OOM" if "memory" in str(exc).lower() else type(exc).__name__
                print(f"{size:>5} {batch:>6} {est:>8.1f} {reason:>11} -- stopping this resolution")
                release(device)
                break

            epoch_s = APTOS_TRAIN_IMAGES / train
            print(
                f"{size:>5} {batch:>6} {est:>8.1f} {fwd:>11.1f} {train:>12.1f} {mem:>9.2f} "
                f"{fmt_duration(epoch_s):>8} {fmt_duration(epoch_s * args.epochs):>9}"
            )
            rows.append((size, batch, train, epoch_s * args.epochs))

    if skipped:
        print(f"\n{len(skipped)} config(s) skipped as over budget. To test one deliberately:")
        s, b, e = skipped[0]
        print(
            f"  python scripts/benchmark_device.py --sizes {s} --batch {b} --budget-gb {e + 1:.0f}"
        )
        print("  Only do this with other applications closed, and prefer measuring")
        print("  large-resolution configs on Kaggle instead.")

    if not rows:
        print("\nNo configuration completed. Lower --batch or --sizes.")
        return 1

    print("\nInterpretation (see docs/05_PROTOTYPE_SCOPE.md):")
    at512 = [r for r in rows if r[0] == 512]
    if at512:
        _, batch, _, full = max(at512, key=lambda r: r[2])
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
