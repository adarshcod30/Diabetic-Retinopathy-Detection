#!/usr/bin/env python3
"""Write a copy of a manifest with its label column randomly permuted.

Exists for exactly one purpose: training the data-randomisation control model
for the Adebayo sanity check (docs/08_PHASE6_RESULTS.md). A model trained on
this manifest sees the same images and the same label DISTRIBUTION as the
real one, but with labels reassigned to the wrong images -- so whatever it
learns is definitionally not real DR signal, which is what makes it a valid
control to compare a real model's explanations against.

The output is not committed -- it is fully and deterministically
reproducible from this script plus the real manifest it derives from, the
same way a cache is not committed. Committing it would risk it being
mistaken for a real dataset manifest.

Usage:
    python scripts/make_shuffled_labels_manifest.py data/manifests/aptos_512.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("manifest", help="source manifest to derive from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="default: <manifest>_shuffled_labels.csv")
    args = p.parse_args()

    src = Path(args.manifest)
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    labels = [r["label"] for r in rows]
    random.Random(args.seed).shuffle(labels)
    for r, label in zip(rows, labels, strict=True):
        r["label"] = label

    out = Path(args.out) if args.out else src.with_stem(src.stem + "_shuffled_labels")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows with shuffled labels (seed {args.seed}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
