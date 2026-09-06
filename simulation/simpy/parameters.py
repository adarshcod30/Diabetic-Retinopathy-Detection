"""Simulation parameters, with provenance kept explicit.

The roadmap's own instruction for this phase is "parameterise from measured
values... no invented constants" (docs/04_ROADMAP.md, Phase 7). That is only
partly possible: this project can measure its *own* model and hardware, but
cannot measure a rural clinic's bandwidth or a grader's real review rate --
those have to come from published literature or from the roadmap's own stated
scenario values. Rather than blur the two, every field below is tagged with
where it actually comes from, and the ones this project did not measure are
swept as a range in the simulation rather than hard-coded as if precisely
known.

Provenance key
--------------
MEASURED   -- run on this project's own model/hardware/data, see the cited script.
CITED      -- a published figure, cited by source; genuinely uncertain, so the
              simulation sweeps a range around it rather than trusting one point.
SCENARIO   -- a value the roadmap itself specifies to sweep (bandwidth, grader
              count, patient volume) -- not a measurement, a requested axis.
ASSUMPTION -- this project's own explicit modelling choice, stated so it can
              be challenged, not smuggled in as if it were a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationParameters:
    # ---- SCENARIO: the roadmap's own required sweep axes ----
    patients_per_year: int = 100_000
    bandwidth_mbps: float = 5.0
    n_graders: int = 4
    outage_enabled: bool = False

    # ---- MEASURED: this project's own model, scripts/benchmark_inference.py ----
    # Defaults below are the CPU numbers from that script's own last run (see
    # runs/inference_benchmark.json, committed alongside it) -- CPU because
    # `drdetect.serve.pipeline.load_grader` is deliberately CPU-only (Phase 2),
    # matching a district server with no GPU.
    inference_ms_grading: float = 1.0  # overwritten from runs/inference_benchmark.json
    inference_ms_evidence_extra: float = 1.0  # ditto
    image_size_kb_mean: float = 401.7  # 103 real IDRiD fundus photos, see docs/20
    # Referable-DR operating point, Phase 1 baseline (docs/06_PHASE1_RESULTS.md):
    # threshold chosen on validation for target sensitivity 0.90, achieved 0.919/0.940.
    model_sensitivity: float = 0.919
    model_specificity: float = 0.940
    # MC-dropout quintile accuracy on the standard val fold (docs/17, real numbers,
    # most-confident-first): used to decide, per confidence quintile, what fraction
    # of auto-clear decisions are actually correct -- not re-measured here, reused
    # from that already-run, already-documented result.
    mc_dropout_quintile_accuracy: tuple[float, ...] = (0.9932, 0.9932, 0.7755, 0.7192, 0.6438)

    # ---- CITED: published, external, genuinely uncertain ----
    # Grader throughput has no single trustworthy number. Search turned up three
    # different real data points, none of them a direct "images/hour" figure for
    # full ICDR grading, so a range is used rather than any one of them alone:
    #   - Rapid crowdsourced binary grading averaged 25 sec/image (Brady et al.,
    #     Ophthalmology 2014, PMC4259907) -- ~72 patients/hr at 2 images/patient,
    #     but this is simplified crowdsourced grading, not full ICDR review, and
    #     is treated here as an optimistic upper bound, not the expected rate.
    #   - English NHS DESP graders assessed 500-2000 image *sets* per grader per
    #     YEAR (2012-13 workforce report) -- implies a much lower effective rate
    #     once non-grading duties and part-time grading are accounted for.
    #   - Rajalakshmi et al. 2022 (Indian tele-ophthalmology, 30 diabetes centres):
    #     25,316 patients graded by 8 retina specialists over 1 year alongside
    #     their regular clinical duties -- also implies a low full-time-equivalent
    #     rate, consistent with the NHS figure.
    # This project's own operating assumption sits between the crowdsourced upper
    # bound and the two clinical-throughput lower bounds: 15 patients/hour/grader
    # (4 min/patient) for a dedicated full-time grader doing ONLY DR grading
    # (screening-camp graders, unlike the cited studies' clinicians, are not also
    # running a clinic) -- swept 10-30/hr in the sensitivity scenarios below.
    grader_patients_per_hour: float = 15.0

    # ---- ASSUMPTION: this project's own explicit modelling choices ----
    images_per_patient: int = 2  # one field per eye, the simplest DR-screening protocol
    working_days_per_year: int = 300
    working_hours_per_day: float = 8.0
    n_camps: int = 20
    outage_mean_interval_hours: float = 48.0  # mean time between outage onsets
    outage_mean_duration_hours: float = 4.0
    # Auto-clear: predicted grade 0 AND MC-dropout confidence above this
    # quantile of the confidence distribution. Swept explicitly in Phase 7's
    # own required sensitivity analysis -- see run_sensitivity_sweep().
    auto_clear_confidence_quantile: float = 0.60
    # Illustrative only -- no real costed budget exists for this project; kept
    # separate from every measured/cited field above so it is never mistaken
    # for one. Figures are round numbers for a relative, not absolute, cost/patient.
    cost_per_image_compute_inr: float = 2.0
    cost_per_grader_hour_inr: float = 400.0
    seed: int = 42

    @property
    def bandwidth_kbytes_per_sec(self) -> float:
        return self.bandwidth_mbps * 1_000 / 8

    @property
    def patients_per_day(self) -> float:
        return self.patients_per_year / self.working_days_per_year

    @property
    def grader_mean_service_hours(self) -> float:
        return 1.0 / self.grader_patients_per_hour


def load_measured_inference(params: SimulationParameters, path: str | Path) -> SimulationParameters:
    """Overwrite the MEASURED fields from a real scripts/benchmark_inference.py run.

    Kept as an explicit, separate step (not baked into the dataclass default)
    so it is obvious in the simulation's own output whether real measured
    numbers were actually loaded, or whether the (clearly stale) dataclass
    defaults are silently in effect.
    """
    data = json.loads(Path(path).read_text())
    from dataclasses import replace

    return replace(
        params,
        inference_ms_grading=data["grading_only"]["mean_ms"],
        inference_ms_evidence_extra=data["lesion_evidence_extra"]["mean_ms"],
        image_size_kb_mean=data["file_size_kb"]["mean"],
    )


PATIENTS_PER_YEAR_HEADLINE = 100_000
BANDWIDTH_SCENARIOS_MBPS = (1.0, 5.0, 10.0)
GRADER_SCENARIOS = (2, 4, 8)
GRADER_RATE_SENSITIVITY_RANGE = (10.0, 15.0, 20.0, 30.0)
AUTO_CLEAR_QUANTILE_SENSITIVITY_RANGE = (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
