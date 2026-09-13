#!/usr/bin/env python3
"""Pull DDR's official test split -- a NEW, never-touched external test set.

Why this exists
----------------
docs/22_PHASE8_VALIDATION_RESULTS.md is explicit: Messidor-2 and IDRiD's locked
grading test split has been evaluated once and "will not be run again against
a different model choice." That commitment stands. Any new model built after
Phase 8 (an ensemble, a RETFound fine-tune) needs its own, never-before-touched
external test set to be evaluated honestly -- reusing Messidor-2/IDRiD for that
purpose would be exactly the test-set-reuse this project has spent its whole
methodology avoiding.

DDR (Diabetic Retinopathy Detection dataset, Li et al. 2019, Chinese Academy of
Sciences) is a good fit: same 5-class ICDR scale as APTOS/Messidor-2/IDRiD (no
label remapping needed), a different population (multiple clinical sites across
23 provinces in China) than APTOS (Indian screening), Messidor-2 (French) or
IDRiD (Indian hospital), a pre-existing official test split (no split decision
of this project's own to second-guess), and a permissive CC BY 4.0 licence.
Re-hosted cleanly on HuggingFace as `ctmedtech/DDR-dataset` (source: Li et al.,
Information Sciences 2019; original Kaggle rehost by Maria Herrera).

This pulls ONLY the `DR_grading/test/` images and `test.txt` labels -- never
train/valid, which this project has no use for (RETFound and the existing
models are fine-tuned on APTOS's own train split; DDR's role here is purely as
a fresh, one-time external test set, not additional training data).

Usage (on the remote instance, same as fetch_eyepacs_subset.py):
    python scripts/fetch_ddr_testset.py

Leaves data/raw/ddr/{test/, test.txt} in the layout scripts/preprocess.py
--dataset ddr expects.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "ctmedtech/DDR-dataset"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--min-free-gb", type=float, default=10.0, help="test split is ~3GB of JPEGs")
    args = p.parse_args()

    free = free_gb(Path("."))
    if free < args.min_free_gb:
        sys.exit(f"ERROR: only {free:.0f}GB free, need >= {args.min_free_gb:.0f}GB.")
    print(f"Disk check: {free:.0f}GB free (>= {args.min_free_gb:.0f}GB required) [ok]")

    from huggingface_hub import snapshot_download

    out_root = Path(args.raw_dir) / "ddr"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DDR test split from {REPO_ID} (test images + labels only)...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["DR_grading/test/*", "DR_grading/test.txt"],
        local_dir=out_root / "_snapshot",
    )

    snapshot_grading = out_root / "_snapshot" / "DR_grading"
    (out_root / "test.txt").write_bytes((snapshot_grading / "test.txt").read_bytes())
    if (out_root / "test").exists():
        shutil.rmtree(out_root / "test")
    shutil.move(str(snapshot_grading / "test"), str(out_root / "test"))
    shutil.rmtree(out_root / "_snapshot")

    n_images = len(list((out_root / "test").glob("*.jpg")))
    n_labels = sum(1 for _ in (out_root / "test.txt").open())
    print(f"Done: {n_images} images, {n_labels} labelled lines, at {out_root}")
    print("\nNext: python scripts/preprocess.py --dataset ddr --size 512")
    return 0


if __name__ == "__main__":
    sys.exit(main())
