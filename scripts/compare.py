#!/usr/bin/env python3
"""Paired statistical comparison of two evaluation runs.

Why paired tests, and why overlapping CIs prove nothing
-------------------------------------------------------
Two models evaluated on the SAME images produce correlated errors. Comparing
their marginal confidence intervals throws that pairing away and is badly
underpowered: intervals can overlap substantially while the paired difference
is highly significant.

  McNemar's test  -- for the binary referable decision. Uses only the
                     DISCORDANT pairs (cases where the two models disagree),
                     which is exactly the information a marginal CI discards.
  Paired bootstrap -- for QWK. Resamples cases once and scores BOTH models on
                     the same resample, so the difference distribution accounts
                     for the shared sample.

Usage:
    python scripts/compare.py runs/eval_384.json runs/eval_512.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired correctness. Returns (b_only, a_only, p)."""
    from scipy import stats

    b_only = int((~correct_a & correct_b).sum())  # B right, A wrong
    a_only = int((correct_a & ~correct_b).sum())  # A right, B wrong
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 1.0
    # two-sided exact binomial on the discordant pairs
    p = float(min(1.0, 2 * stats.binom.cdf(min(a_only, b_only), n, 0.5)))
    return b_only, a_only, p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from drdetect.eval.metrics import quadratic_weighted_kappa, referable_labels

    a = json.loads(Path(args.run_a).read_text())
    b = json.loads(Path(args.run_b).read_text())
    la = args.label_a or Path(args.run_a).stem
    lb = args.label_b or Path(args.run_b).stem

    for run, name in ((a, la), (b, lb)):
        if "predictions" not in run:
            print(
                f"{name} has no per-sample predictions -- re-run scripts/evaluate.py",
                file=sys.stderr,
            )
            return 1

    if a["image_ids"] != b["image_ids"]:
        print("Runs cover different images; a paired test is invalid.", file=sys.stderr)
        return 1

    targets = np.array(a["targets"])
    pred_a, pred_b = np.array(a["predictions"]), np.array(b["predictions"])
    n = len(targets)

    print(f"Comparing {la}  vs  {lb}   ({n} paired images)\n")

    # --- QWK: paired bootstrap on the difference ---
    rng = np.random.default_rng(args.seed)
    qa = quadratic_weighted_kappa(targets, pred_a)
    qb = quadratic_weighted_kappa(targets, pred_b)
    diffs = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        diffs[i] = quadratic_weighted_kappa(targets[idx], pred_b[idx]) - quadratic_weighted_kappa(
            targets[idx], pred_a[idx]
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_qwk = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))

    print("QWK")
    print(f"  {la:<24} {qa:.4f}")
    print(f"  {lb:<24} {qb:.4f}")
    print(
        f"  difference (B - A)       {qb - qa:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p={p_qwk:.3f}"
    )
    print(f"  -> {'SIGNIFICANT' if p_qwk < 0.05 else 'not significant'} at alpha=0.05")

    # --- exact grade agreement: McNemar ---
    ca, cb = pred_a == targets, pred_b == targets
    b_only, a_only, p_ex = mcnemar(ca, cb)
    print("\nExact grade correctness (McNemar)")
    print(f"  {lb} right / {la} wrong   {b_only}")
    print(f"  {la} right / {lb} wrong   {a_only}")
    print(f"  p = {p_ex:.4f}  -> {'SIGNIFICANT' if p_ex < 0.05 else 'not significant'}")

    # --- referable decision at each run's own operating point: McNemar ---
    y_ref = referable_labels(targets)
    ra = (np.array(a["referable_score"]) >= a["threshold"]).astype(int)
    rb = (np.array(b["referable_score"]) >= b["threshold"]).astype(int)
    b_only, a_only, p_ref = mcnemar(ra == y_ref, rb == y_ref)
    print("\nReferable decision, each at its own chosen threshold (McNemar)")
    print(f"  {lb} right / {la} wrong   {b_only}")
    print(f"  {la} right / {lb} wrong   {a_only}")
    print(f"  p = {p_ref:.4f}  -> {'SIGNIFICANT' if p_ref < 0.05 else 'not significant'}")

    # --- per-class recall ---
    print("\nPer-class recall")
    names = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    print(f"  {'class':<12}{'n':>5}{la[:10]:>12}{lb[:10]:>12}{'delta':>9}")
    for c in range(5):
        m = targets == c
        if not m.any():
            continue
        rа, rb_ = float((pred_a[m] == c).mean()), float((pred_b[m] == c).mean())
        print(f"  {c} {names[c]:<10}{int(m.sum()):>5}{rа:>12.3f}{rb_:>12.3f}{rb_ - rа:>+9.3f}")

    print("\nNote: both models were selected and thresholded on this same split,")
    print("so absolute values are optimistic. The paired DIFFERENCE is the")
    print("quantity these tests are designed to estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
