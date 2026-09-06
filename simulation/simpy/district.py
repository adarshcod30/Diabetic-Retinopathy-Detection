"""Discrete-event model of a district DR-screening programme.

Flow (docs/01_PROJECT_ANALYSIS.md Stage 7 / docs/04_ROADMAP.md Phase 7):

    camps -> capture -> upload -> inference -> triage -> human review

Camps are geographically separate and each has its OWN bandwidth-limited
uplink (rural connectivity is a per-camp constraint, and fails independently
per camp -- see the outage process). Inference and human grading are
CENTRALISED shared resources reached over that uplink: one inference server
(this project's own model is fast enough that it is deliberately not the
bottleneck -- see docs/20) and a pool of `n_graders` human reviewers (the
roadmap's own framing: this is "the true bottleneck").

Two-tier inference, matching what `drdetect.serve.pipeline` actually does:
every uploaded image gets the fast grading-only pass; only images that fail
to auto-clear (see `triage`) get the slower +lesion-evidence pass, since that
is only needed for a report a human will actually read.

What is simplified, stated plainly rather than silently baked in:
  - Time is continuous "camp-operating hours" (patients_per_year spread over
    working_days x working_hours), not simulated day-by-day with camps
    closing overnight. A queue that is not fully drained in one design does
    not get an artificial overnight reset. See parameters.py's docstring for
    which inputs are measured vs. cited vs. this project's own assumption.
  - Per-patient true grade is drawn from APTOS's own empirical 5-class
    distribution (itself a real Indian screening population, Aravind Eye
    Hospital) -- used as a stand-in for real screening-population prevalence,
    not a separately-cited epidemiological figure.
  - The model's per-case correctness is drawn from the REAL MC-dropout
    quintile-accuracy curve (docs/17), each patient assigned one of 5
    confidence quintiles uniformly at random. This reuses a real, already
    measured accuracy-vs-confidence relationship rather than inventing one,
    but does not reconstruct the full 5x5 confusion matrix or the true joint
    distribution of confidence and grade -- stated here as a simplification,
    not claimed as a re-validation of Phase 5's own result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simpy

from simulation.simpy.parameters import SimulationParameters, load_measured_inference

# APTOS's own real training-split class counts (docs/06_PHASE1_RESULTS.md /
# docs/05_PROTOTYPE_SCOPE.md): 1805/370/999/193/295 for grades 0-4.
APTOS_GRADE_DISTRIBUTION = (
    np.array([1805, 370, 999, 193, 295]) / np.array([1805, 370, 999, 193, 295]).sum()
)


@dataclass
class ImageEvent:
    patient_id: int
    camp_id: int
    arrival_time: float
    upload_end: float = float("nan")
    inference_end: float = float("nan")
    decision: str = ""  # "auto_clear" | "escalate"
    grader_queue_start: float = float("nan")
    done_time: float = float("nan")
    true_grade: int = -1
    predicted_grade: int = -1
    auto_clear_correct: bool | None = None


@dataclass
class Metrics:
    completed: list[ImageEvent] = field(default_factory=list)
    grader_busy_samples: list[int] = field(default_factory=list)
    grader_queue_samples: list[tuple[float, int]] = field(default_factory=list)
    uplink_queue_samples: list[tuple[float, int]] = field(default_factory=list)
    outages: list[tuple[int, float, float]] = field(default_factory=list)
    auto_clear_count: int = 0
    auto_clear_wrong_count: int = 0
    escalate_count: int = 0


class Camp:
    def __init__(self, env: simpy.Environment, camp_id: int):
        self.camp_id = camp_id
        self.uplink = simpy.Resource(env, capacity=1)
        self.up = True
        self.up_event = env.event()
        self.up_event.succeed()


def outage_process(env, camp: Camp, params: SimulationParameters, rng, metrics: Metrics):
    if not params.outage_enabled:
        return
    while True:
        yield env.timeout(rng.exponential(params.outage_mean_interval_hours))
        camp.up = False
        camp.up_event = env.event()
        start = env.now
        yield env.timeout(rng.exponential(params.outage_mean_duration_hours))
        camp.up = True
        ev, camp.up_event = camp.up_event, env.event()
        camp.up_event.succeed()
        ev.succeed()
        metrics.outages.append((camp.camp_id, start, env.now))


def _draw_prediction(true_grade: int, quintile_accuracy: tuple[float, ...], rng) -> tuple[int, int]:
    """Return (predicted_grade, quintile_index) using the real MC-dropout
    quintile-accuracy curve (docs/17): with probability quintile_accuracy[q]
    the model is exactly correct; otherwise it misses by one ordinal step
    (clipped to [0, 4]), consistent with this project's own repeated finding
    that QWK-scale errors are dominated by adjacent-grade confusions, not
    far-off ones."""
    quintile = rng.integers(0, 5)
    if rng.random() < quintile_accuracy[quintile]:
        return true_grade, quintile
    step = rng.choice([-1, 1])
    return int(np.clip(true_grade + step, 0, 4)), quintile


def image_pipeline(
    env,
    camp: Camp,
    patient_id: int,
    params: SimulationParameters,
    inference_server: simpy.Resource,
    grader_pool: simpy.Resource,
    metrics: Metrics,
    rng,
):
    ev = ImageEvent(patient_id=patient_id, camp_id=camp.camp_id, arrival_time=env.now)
    ev.true_grade = int(rng.choice(5, p=APTOS_GRADE_DISTRIBUTION))

    # --- upload: per-camp bandwidth + outage ---
    while not camp.up:
        yield camp.up_event
    with camp.uplink.request() as req:
        yield req
        while not camp.up:
            yield camp.up_event
        upload_hours = (params.image_size_kb_mean / params.bandwidth_kbytes_per_sec) / 3600.0
        yield env.timeout(upload_hours)
    ev.upload_end = env.now

    # --- central inference: grading-only pass, every image ---
    with inference_server.request() as req:
        yield req
        yield env.timeout(params.inference_ms_grading / 1000.0 / 3600.0)
    ev.inference_end = env.now

    predicted_grade, quintile = _draw_prediction(
        ev.true_grade, params.mc_dropout_quintile_accuracy, rng
    )
    ev.predicted_grade = predicted_grade
    # top-N quintiles eligible for auto-clear, N set by auto_clear_confidence_quantile
    n_eligible_quintiles = round(params.auto_clear_confidence_quantile * 5)
    confident_enough = quintile < n_eligible_quintiles

    if predicted_grade == 0 and confident_enough:
        ev.decision = "auto_clear"
        ev.auto_clear_correct = ev.true_grade == 0
        metrics.auto_clear_count += 1
        if not ev.auto_clear_correct:
            metrics.auto_clear_wrong_count += 1
    else:
        ev.decision = "escalate"
        metrics.escalate_count += 1
        # +lesion-evidence pass -- only escalated cases need the report
        with inference_server.request() as req:
            yield req
            yield env.timeout(params.inference_ms_evidence_extra / 1000.0 / 3600.0)
        with grader_pool.request() as req:
            ev.grader_queue_start = env.now
            yield req
            yield env.timeout(rng.exponential(params.grader_mean_service_hours))
    ev.done_time = env.now
    metrics.completed.append(ev)


def camp_arrivals(
    env,
    camp: Camp,
    params: SimulationParameters,
    total_hours: float,
    inference_server,
    grader_pool,
    metrics: Metrics,
    rng,
    patient_id_start: int,
):
    rate_per_hour = (params.patients_per_year / params.n_camps) / (
        params.working_days_per_year * params.working_hours_per_day
    )
    patient_id = patient_id_start
    while env.now < total_hours:
        yield env.timeout(rng.exponential(1.0 / rate_per_hour))
        for _ in range(params.images_per_patient):
            env.process(
                image_pipeline(
                    env, camp, patient_id, params, inference_server, grader_pool, metrics, rng
                )
            )
        patient_id += 1


def resource_monitor(
    env,
    grader_pool: simpy.Resource,
    camps: list[Camp],
    metrics: Metrics,
    interval_hours: float = 1.0,
):
    while True:
        metrics.grader_busy_samples.append(grader_pool.count)
        metrics.grader_queue_samples.append((env.now, len(grader_pool.queue)))
        total_uplink_queue = sum(len(c.uplink.queue) for c in camps)
        metrics.uplink_queue_samples.append((env.now, total_uplink_queue))
        yield env.timeout(interval_hours)


@dataclass
class SimulationResult:
    params: SimulationParameters
    total_hours: float
    n_completed: int
    n_auto_clear: int
    n_auto_clear_wrong: int
    n_escalate: int
    turnaround_hours: dict  # mean/median/p90/p95 for escalated (human-reviewed) cases
    grader_utilization: float
    max_uplink_backlog: int
    outage_count: int
    max_backlog_during_outage: int
    grader_hours_freed: float
    cost_per_patient_inr: float

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "params"}
        d["params"] = {
            "patients_per_year": self.params.patients_per_year,
            "bandwidth_mbps": self.params.bandwidth_mbps,
            "n_graders": self.params.n_graders,
            "outage_enabled": self.params.outage_enabled,
            "grader_patients_per_hour": self.params.grader_patients_per_hour,
            "auto_clear_confidence_quantile": self.params.auto_clear_confidence_quantile,
        }
        return d


def run_simulation(params: SimulationParameters, *, seed: int | None = None) -> SimulationResult:
    rng = np.random.default_rng(seed if seed is not None else params.seed)
    env = simpy.Environment()

    camps = [Camp(env, i) for i in range(params.n_camps)]
    inference_server = simpy.Resource(env, capacity=1)
    grader_pool = simpy.Resource(env, capacity=params.n_graders)
    metrics = Metrics()

    total_hours = params.working_days_per_year * params.working_hours_per_day

    for i, camp in enumerate(camps):
        env.process(outage_process(env, camp, params, rng, metrics))
        env.process(
            camp_arrivals(
                env,
                camp,
                params,
                total_hours,
                inference_server,
                grader_pool,
                metrics,
                rng,
                patient_id_start=i * 10_000_000,
            )
        )
    env.process(resource_monitor(env, grader_pool, camps, metrics))

    # Run past total_hours so in-flight cases (queued at close) finish and are counted.
    env.run(until=total_hours + max(24.0, 5.0 * params.grader_mean_service_hours * 50))

    completed = metrics.completed
    escalated = [e for e in completed if e.decision == "escalate" and np.isfinite(e.done_time)]
    turnaround = (
        np.array([e.done_time - e.arrival_time for e in escalated])
        if escalated
        else np.array([0.0])
    )

    grader_utilization = (
        float(np.mean(metrics.grader_busy_samples) / params.n_graders)
        if metrics.grader_busy_samples
        else 0.0
    )
    max_uplink_backlog = max((q for _, q in metrics.uplink_queue_samples), default=0)

    max_backlog_during_outage = 0
    for _camp_id, start, end in metrics.outages:
        during = [q for t, q in metrics.uplink_queue_samples if start <= t <= end + 12.0]
        if during:
            max_backlog_during_outage = max(max_backlog_during_outage, max(during))

    # Ophthalmologist-hours freed: auto-cleared cases never reach a grader: had
    # they all been escalated instead, they would have cost this many grader-hours.
    grader_hours_freed = metrics.auto_clear_count * params.grader_mean_service_hours

    n_patients = len(completed) // params.images_per_patient if completed else 0
    # Compute cost: grading pass on every image, + the extra evidence pass on
    # escalated images only (the two-tier design). Grader cost: escalate_count
    # x mean service time in expectation (matches what grader_utilization
    # already measures, rather than resampling a second random draw).
    total_compute_cost = (
        len(completed) + metrics.escalate_count
    ) * params.cost_per_image_compute_inr
    total_grader_cost = (
        metrics.escalate_count * params.grader_mean_service_hours * params.cost_per_grader_hour_inr
    )
    cost_per_patient = (total_compute_cost + total_grader_cost) / max(n_patients, 1)

    return SimulationResult(
        params=params,
        total_hours=total_hours,
        n_completed=len(completed),
        n_auto_clear=metrics.auto_clear_count,
        n_auto_clear_wrong=metrics.auto_clear_wrong_count,
        n_escalate=metrics.escalate_count,
        turnaround_hours={
            "mean": float(np.mean(turnaround)),
            "median": float(np.median(turnaround)),
            "p90": float(np.percentile(turnaround, 90)),
            "p95": float(np.percentile(turnaround, 95)),
        },
        grader_utilization=grader_utilization,
        max_uplink_backlog=int(max_uplink_backlog),
        outage_count=len(metrics.outages),
        max_backlog_during_outage=int(max_backlog_during_outage),
        grader_hours_freed=float(grader_hours_freed),
        cost_per_patient_inr=float(cost_per_patient),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patients-per-year", type=int, default=100_000)
    p.add_argument("--bandwidth-mbps", type=float, default=5.0)
    p.add_argument("--n-graders", type=int, default=4)
    p.add_argument("--outage", action="store_true")
    p.add_argument("--grader-rate", type=float, default=15.0)
    p.add_argument("--auto-clear-quantile", type=float, default=0.60)
    p.add_argument("--inference-benchmark", default="runs/inference_benchmark.json")
    p.add_argument("--out", default="runs/simulation_default.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    params = SimulationParameters(
        patients_per_year=args.patients_per_year,
        bandwidth_mbps=args.bandwidth_mbps,
        n_graders=args.n_graders,
        outage_enabled=args.outage,
        grader_patients_per_hour=args.grader_rate,
        auto_clear_confidence_quantile=args.auto_clear_quantile,
        seed=args.seed,
    )
    if Path(args.inference_benchmark).exists():
        params = load_measured_inference(params, args.inference_benchmark)
        print(f"loaded measured inference timing from {args.inference_benchmark}")
    else:
        print(f"WARNING: {args.inference_benchmark} not found; using dataclass defaults")

    print(
        f"running: {params.patients_per_year} patients/yr, {params.bandwidth_mbps} Mbps, "
        f"{params.n_graders} graders, outage={params.outage_enabled}"
    )
    result = run_simulation(params, seed=args.seed)

    print(f"\ncompleted images    : {result.n_completed}")
    print(
        f"auto-cleared        : {result.n_auto_clear} ({result.n_auto_clear_wrong} wrongly, "
        f"i.e. true grade > 0)"
    )
    print(f"escalated to grader : {result.n_escalate}")
    print(f"grader utilisation  : {result.grader_utilization:.1%}")
    print(
        f"turnaround (escalated cases, hours): mean {result.turnaround_hours['mean']:.2f}  "
        f"p90 {result.turnaround_hours['p90']:.2f}  p95 {result.turnaround_hours['p95']:.2f}"
    )
    print(f"max uplink backlog  : {result.max_uplink_backlog} images queued")
    if params.outage_enabled:
        print(
            f"outages injected    : {result.outage_count}, "
            f"max backlog during outage: {result.max_backlog_during_outage}"
        )
    print(f"grader-hours freed  : {result.grader_hours_freed:.1f}")
    print(f"cost/patient (illustrative): Rs {result.cost_per_patient_inr:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_json(), indent=2))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
