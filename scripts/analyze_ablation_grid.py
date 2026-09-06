#!/usr/bin/env python3
"""Paired significance tests across Phase 8's ablation-grid candidates.

Reads the per-image predictions scripts/evaluate.py already saved for the
baseline and each new candidate (same fold-0 held-out APTOS images -- CLAHE's
manifest uses the same seed/n_splits/fold, so its held-out image_ids are the
same set even though the pixels differ), and reports, for each candidate vs.
the baseline: QWK with its own bootstrap CI, a paired McNemar test on
exact-grade correctness, and a paired DeLong test on referable AUC. This is
the internal-validation-only comparison that decides which checkpoint(s), if
any, are worth carrying into Phase 8's single locked external evaluation --
it does not itself touch Messidor-2 or IDRiD.

Usage:
    python scripts/analyze_ablation_grid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FILES = {
    "baseline (512px CE)": "runs/ablation_eval_baseline.json",
    "224px CE": "runs/ablation_eval_224ce.json",
    "CLAHE 512px CE": "runs/ablation_eval_clahe.json",
    "regression 512px": "runs/ablation_eval_regression.json",
    "distance_ce 512px": "runs/ablation_eval_distance_ce.json",
}
BASELINE_KEY = "baseline (512px CE)"


def main() -> int:
    import numpy as np
    from scipy import stats

    from drdetect.eval.metrics import delong_test, referable_labels

    data = {}
    for name, path in FILES.items():
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        data[name] = json.loads(Path(path).read_text())

    if BASELINE_KEY not in data:
        print("Baseline evaluation missing -- run scripts/evaluate.py for it first.")
        return 1

    print(f"{'Config':<22}{'QWK':>8}  {'95% CI':>20}  {'Sens':>7}  {'Spec':>7}")
    for name, d in data.items():
        ci = d["qwk_ci"]
        print(
            f"{name:<22}{d['qwk']:>8.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  "
            f"{d['sensitivity']:>7.3f}  {d['specificity']:>7.3f}"
        )

    base = data[BASELINE_KEY]
    base_ids = base["image_ids"]
    base_targets = np.array(base["targets"])
    base_preds = np.array(base["predictions"])
    base_pref = np.array(base["referable_score"])
    y_ref = referable_labels(base_targets)
    correct_base = base_preds == base_targets

    print("\n" + "=" * 70)
    print("Paired comparisons vs. baseline, same fold-0 held-out images")
    print("=" * 70)
    results = {}
    for name, d in data.items():
        if name == BASELINE_KEY:
            continue
        ids = d["image_ids"]
        if ids == base_ids:
            preds = np.array(d["predictions"])
            pref = np.array(d["referable_score"])
        else:
            idx = {iid: i for i, iid in enumerate(ids)}
            order = [idx[i] for i in base_ids]
            preds = np.array(d["predictions"])[order]
            pref = np.array(d["referable_score"])[order]

        correct_cand = preds == base_targets
        cand_only = int((~correct_base & correct_cand).sum())
        base_only = int((correct_base & ~correct_cand).sum())
        n_disc = cand_only + base_only
        p_mcnemar = (
            float(min(1.0, 2 * stats.binom.cdf(min(cand_only, base_only), n_disc, 0.5)))
            if n_disc
            else 1.0
        )

        dl = delong_test(y_ref, pref, base_pref)

        print(f"\n{name} vs. baseline:")
        print(
            f"  exact-grade McNemar: {name}-only-correct={cand_only}, "
            f"baseline-only-correct={base_only}, n_discordant={n_disc}, p={p_mcnemar:.4f}"
        )
        print(
            f"  referable AUC: {name}={dl.auc_a:.4f} vs baseline={dl.auc_b:.4f}, "
            f"DeLong z={dl.z:.3f}, p={dl.p:.4f}"
        )
        results[name] = {
            "qwk": data[name]["qwk"],
            "mcnemar_p": p_mcnemar,
            "mcnemar_candidate_only": cand_only,
            "mcnemar_baseline_only": base_only,
            "delong_auc_candidate": dl.auc_a,
            "delong_auc_baseline": dl.auc_b,
            "delong_p": dl.p,
        }

    out = Path("runs/ablation_grid_analysis.json")
    out.write_text(json.dumps({"baseline_qwk": base["qwk"], "candidates": results}, indent=2))
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
