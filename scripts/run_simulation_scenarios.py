#!/usr/bin/env python3
"""Run the Phase 7 scenario grid the roadmap specifies, once, and save results.

Scenarios (docs/04_ROADMAP.md, Phase 7):
  - 100,000 patients/year (headline)
  - bandwidth in {1, 5, 10} Mbps x graders in {2, 4, 8}  -- the 9-cell grid
    answering "how many graders does a district of 100k need?"
  - outage injection, on top of the headline configuration
  - sensitivity analysis on the auto-clear confidence threshold
  - (bonus, same spirit) sensitivity on the cited grader-rate range, since
    that parameter is CITED not MEASURED -- see parameters.py's docstring

Usage:
    python scripts/run_simulation_scenarios.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.simpy.district import run_simulation
from simulation.simpy.parameters import (
    AUTO_CLEAR_QUANTILE_SENSITIVITY_RANGE,
    BANDWIDTH_SCENARIOS_MBPS,
    GRADER_RATE_SENSITIVITY_RANGE,
    GRADER_SCENARIOS,
    PATIENTS_PER_YEAR_HEADLINE,
    SimulationParameters,
    load_measured_inference,
)


def base_params() -> SimulationParameters:
    params = SimulationParameters(patients_per_year=PATIENTS_PER_YEAR_HEADLINE)
    bench = Path("runs/inference_benchmark.json")
    if bench.exists():
        params = load_measured_inference(params, bench)
    else:
        print("WARNING: runs/inference_benchmark.json missing; using unmeasured defaults")
    return params


def main() -> int:
    out = {"grid": [], "outage": None, "auto_clear_sensitivity": [], "grader_rate_sensitivity": []}
    base = base_params()
    t_start = time.time()

    print("=" * 70)
    print("1) bandwidth x graders grid, 100,000 patients/year, no outage")
    print("=" * 70)
    for bw in BANDWIDTH_SCENARIOS_MBPS:
        for ng in GRADER_SCENARIOS:
            p = base_params()
            from dataclasses import replace

            p = replace(p, bandwidth_mbps=bw, n_graders=ng)
            r = run_simulation(p, seed=42)
            row = {
                "bandwidth_mbps": bw,
                "n_graders": ng,
                "grader_utilization": r.grader_utilization,
                "turnaround_p90_hours": r.turnaround_hours["p90"],
                "turnaround_p95_hours": r.turnaround_hours["p95"],
                "max_uplink_backlog": r.max_uplink_backlog,
                "cost_per_patient_inr": r.cost_per_patient_inr,
                "grader_hours_freed": r.grader_hours_freed,
            }
            out["grid"].append(row)
            print(
                f"  bw={bw:>4.0f}Mbps  graders={ng}  util={r.grader_utilization:6.1%}  "
                f"p90={r.turnaround_hours['p90']:6.2f}h  p95={r.turnaround_hours['p95']:6.2f}h  "
                f"backlog={r.max_uplink_backlog:4d}  cost/pt=Rs{r.cost_per_patient_inr:6.2f}"
            )

    print("\n" + "=" * 70)
    print("2) outage injection, headline config (5 Mbps, 4 graders)")
    print("=" * 70)
    from dataclasses import replace

    p_no_outage = replace(base, bandwidth_mbps=5.0, n_graders=4, outage_enabled=False)
    p_outage = replace(base, bandwidth_mbps=5.0, n_graders=4, outage_enabled=True)
    r_no = run_simulation(p_no_outage, seed=42)
    r_out = run_simulation(p_outage, seed=42)
    out["outage"] = {
        "no_outage": {
            "max_uplink_backlog": r_no.max_uplink_backlog,
            "turnaround_p95_hours": r_no.turnaround_hours["p95"],
        },
        "with_outage": {
            "outage_count": r_out.outage_count,
            "max_uplink_backlog": r_out.max_uplink_backlog,
            "max_backlog_during_outage": r_out.max_backlog_during_outage,
            "turnaround_p95_hours": r_out.turnaround_hours["p95"],
        },
    }
    print(
        f"  no outage  : max backlog {r_no.max_uplink_backlog}, p95 turnaround "
        f"{r_no.turnaround_hours['p95']:.2f}h"
    )
    print(
        f"  with outage: {r_out.outage_count} outages injected, max backlog "
        f"{r_out.max_uplink_backlog} (during-outage peak {r_out.max_backlog_during_outage}), "
        f"p95 turnaround {r_out.turnaround_hours['p95']:.2f}h"
    )

    print("\n" + "=" * 70)
    print("3) sensitivity: auto-clear confidence threshold (headline config)")
    print("=" * 70)
    for q in AUTO_CLEAR_QUANTILE_SENSITIVITY_RANGE:
        p = replace(base, bandwidth_mbps=5.0, n_graders=4, auto_clear_confidence_quantile=q)
        r = run_simulation(p, seed=42)
        row = {
            "auto_clear_confidence_quantile": q,
            "n_auto_clear": r.n_auto_clear,
            "n_auto_clear_wrong": r.n_auto_clear_wrong,
            "auto_clear_wrong_rate": r.n_auto_clear_wrong / max(r.n_auto_clear, 1),
            "grader_hours_freed": r.grader_hours_freed,
            "grader_utilization": r.grader_utilization,
        }
        out["auto_clear_sensitivity"].append(row)
        print(
            f"  quantile={q:.2f}  auto_clear={r.n_auto_clear:6d}  wrong={r.n_auto_clear_wrong:4d} "
            f"({row['auto_clear_wrong_rate']:.2%})  hours_freed={r.grader_hours_freed:7.1f}  "
            f"util={r.grader_utilization:.1%}"
        )

    print("\n" + "=" * 70)
    print("4) sensitivity: cited grader-rate range (headline config, 4 graders)")
    print("=" * 70)
    for rate in GRADER_RATE_SENSITIVITY_RANGE:
        p = replace(base, bandwidth_mbps=5.0, n_graders=4, grader_patients_per_hour=rate)
        r = run_simulation(p, seed=42)
        row = {
            "grader_patients_per_hour": rate,
            "grader_utilization": r.grader_utilization,
            "turnaround_p90_hours": r.turnaround_hours["p90"],
            "turnaround_p95_hours": r.turnaround_hours["p95"],
        }
        out["grader_rate_sensitivity"].append(row)
        print(
            f"  rate={rate:5.1f}/hr  util={r.grader_utilization:6.1%}  "
            f"p90={r.turnaround_hours['p90']:6.2f}h  p95={r.turnaround_hours['p95']:6.2f}h"
        )

    out_path = Path("runs/simulation_scenarios.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\ntotal wall time: {time.time() - t_start:.1f}s")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
