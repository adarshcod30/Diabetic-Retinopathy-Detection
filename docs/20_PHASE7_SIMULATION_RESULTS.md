# Phase 7 Results — District Screening Simulation

A SimPy discrete-event model of the flow docs/01 describes: camps -> capture -> upload ->
inference -> triage -> human review. Built to answer the roadmap's own question in numbers a
health administrator would actually use: **how many graders does a district of 100,000
patients/year need, with and without AI auto-clearing?**

## What's measured vs. cited vs. assumed — and why that distinction matters here

Unlike earlier phases, this one cannot be "measured, not guessed" all the way through: this
project can measure its own model and hardware, but not a rural clinic's bandwidth or a real
grader's review rate. Rather than blur the two, every parameter is tagged in
`simulation/simpy/parameters.py` and reported that way below. Where a number is genuinely
uncertain (grader throughput), it is swept as a range, not asserted as one point.

**Measured** (`scripts/benchmark_inference.py`, CPU -- `drdetect.serve.pipeline.load_grader` is
deliberately CPU-only, matching a district server with no GPU):

| Stage | Time | Note |
|---|---:|---|
| Quality gate + grading (every uploaded image) | **2,137 ms/image** (0.47 img/s) | EfficientNet-B0, 512px, CPU |
| + lesion evidence (5 extra models, only for escalated cases) | **+41,904 ms/image** | tiled segmentation at full resolution dominates |

The two-tier design this project already built (Phase 6: `add_lesion_evidence` as a step
separate from `run_pipeline`) turns out not to be a nice-to-have -- it is close to mandatory.
Running full lesion-evidence extraction on *every* uploaded image, not just escalated ones,
would cap throughput at 0.02 img/s (one image per 45 seconds); at 0.47 img/s, grading-only
inference is never the bottleneck for any scenario below (confirmed, not assumed -- see the
grid).

Real fundus-photo file size, measured directly from IDRiD's 413 disease-grading training
images (not the locked test split -- this benchmark only needed *some* full-resolution real
photos, so it deliberately avoided the 103 held-out test images): **mean 392.6 KB, median
389.2 KB**. docs/01's original planning estimate of "~4 MB" for a 4288x2848 JPEG was a guess,
not a measurement, and was roughly **10x too high** -- corrected here now that a real number
exists. This matters directly for the bandwidth question below: uploads are far cheaper than
originally planned for.

**Cited** (published, external, genuinely uncertain -- swept as a range rather than trusted as
one number):

Grader throughput has no single trustworthy published "images/hour" figure for full ICDR
grading. Three real data points, none of them a direct answer, bracket this project's own
working assumption:

- Rapid crowdsourced *binary* grading averaged **25 sec/image** (Brady et al., *Ophthalmology*
  2014) -- ~72 patients/hr at 2 images/patient, but this is simplified crowdsourced grading, not
  full ICDR review; treated as an optimistic upper bound.
- English NHS Diabetic Eye Screening Programme graders assessed **500-2,000 image sets per
  grader per year** (2012-13 workforce report) -- implies a much lower effective full-time rate.
- Rajalakshmi et al. 2022 (Indian tele-ophthalmology, 30 diabetes centres): 25,316 patients
  graded by 8 retina specialists over one year, alongside their regular clinical duties.

This project's own operating assumption, stated as an assumption: **15 patients/hour/grader**
(a dedicated grader doing only DR grading, unlike the cited clinicians) -- swept **10-30/hr** in
the sensitivity analysis below.

**This project's own explicit modelling assumptions** (not measurements, not citations --
see the parameters.py docstring for the full list): 20 camps, 2 images/patient (one field per
eye), 300 working days/year x 8 hours/day, outages Poisson-distributed per camp (mean interval
48h, mean duration 4h). Time is modelled as continuous camp-operating hours across the whole
year rather than simulated day-by-day with overnight resets -- a queue that does not fully drain
in one design is not given a free reset it would not get in reality. Cost figures (Rs 2/image
compute, Rs 400/grader-hour) are illustrative only; no real costed budget exists for this
project, and they are kept in a clearly separate part of the parameter file so they are never
mistaken for a measured or cited figure.

## Result 1 — bandwidth x graders grid, 100,000 patients/year, no outage

| Bandwidth | Graders | Grader utilisation | Turnaround p90 | Turnaround p95 | Cost/patient (illustrative) |
|---:|---:|---:|---:|---:|---:|
| 1 Mbps | 2 | 100.0% | **1,083.7 h** (~45 days) | 1,143.6 h | Rs 66.67 |
| 1 Mbps | 4 | 97.4% | 4.4 h | 5.2 h | Rs 45.01 |
| 1 Mbps | 8 | 49.1% | 0.22 h | 0.27 h | Rs 44.83 |
| 5 Mbps | 2 | 100.0% | 1,080.3 h | 1,140.6 h | Rs 67.04 |
| 5 Mbps | 4 | 98.4% | 6.9 h | 7.7 h | Rs 44.85 |
| 5 Mbps | 8 | 49.1% | 0.21 h | 0.26 h | Rs 44.94 |
| 10 Mbps | 2 | 100.0% | 1,085.5 h | 1,145.3 h | Rs 66.54 |
| 10 Mbps | 4 | 98.2% | 8.0 h | 8.9 h | Rs 44.90 |
| 10 Mbps | 8 | 48.8% | 0.22 h | 0.27 h | Rs 45.01 |

**The headline answer**: at this project's own base-case grader rate (15/hr), **2 graders is a
genuine collapse** (near-infinite queueing, not a slow system -- utilisation pinned at 100%
means arrivals permanently outpace service, a textbook M/M/c instability, not a rounding
artefact). **4 graders is the marginal, not comfortable, answer** -- 97-98% utilisation with
single-digit-hour turnaround, survivable but with no slack for a bad day. **8 graders is
comfortable** -- under 50% utilisation, turnaround in minutes. A district planning for 100,000
patients/year at this project's own AI auto-clear rate (~30% at the default confidence
threshold, see Result 3) should budget for **6-8 dedicated graders**, not the 2-4 a naive
patients-divided-by-published-rate calculation might suggest, because queueing delay is
non-linear near capacity.

**Bandwidth (1/5/10 Mbps) has essentially no effect on any of these numbers.** This was not
assumed going in -- docs/01 flagged human review capacity as "the true bottleneck" as a
*hypothesis*; this is the number confirming it. At 20 camps each handling ~16-17 patients/day,
even a 400KB image at 1 Mbps (~3.2 seconds) is nowhere near enough volume to saturate a per-camp
uplink under smooth arrivals. Bandwidth's real effect shows up only during outages (Result 2),
not in steady-state throughput.

## Result 2 — outage injection (headline config: 5 Mbps, 4 graders)

| | Outages injected | Max uplink backlog | Turnaround p95 |
|---|---:|---:|---:|
| No outage | 0 | 1 | 7.68 h |
| With outage | 951/year | 1 (during-outage peak: 1) | 11.20 h |

951 outages/year (mean interval 48h x 20 camps over 2,400 operating hours/year is consistent
with this by construction -- a check on the outage process itself, not a finding). Turnaround
degrades moderately (7.68h -> 11.20h at p95) but the uplink backlog itself never grows large:
individual camp outages are absorbed by the rest of the system (other camps keep flowing, the
centralised grader pool is unaffected by any one camp's connectivity) rather than compounding
into a growing queue. This is a genuinely different failure mode from Result 1's 2-grader
collapse: **outages cause a bounded, recoverable degradation; grader under-capacity causes
unbounded collapse.** A district should worry far more about grader headcount than about rural
connectivity, given this model.

## Result 3 — sensitivity: auto-clear confidence threshold

Auto-clear fires when the model predicts grade 0 *and* its MC-dropout confidence quintile falls
within the top fraction set by this threshold (0.0 = never auto-clear, 1.0 = auto-clear any
confident grade-0 prediction regardless of quintile). Per-case correctness is drawn from the
REAL quintile-accuracy curve already measured in Phase 5 (docs/17: 99.3%/99.3%/77.6%/71.9%/64.4%
across the five confidence quintiles, most-confident first) -- reused here, not re-measured.

| Quantile | Auto-cleared/year | Wrongly auto-cleared | Wrong rate | Grader-hours freed/year | Grader utilisation (4 graders) |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0.00% | 0 | 100.0% (overloaded) |
| 0.20 | 19,754 | 12 | 0.06% | 1,316.9 | 100.0% (overloaded) |
| 0.40 | 39,652 | 30 | 0.08% | 2,643.5 | 100.0% (overloaded) |
| 0.60 (default) | 57,687 | 489 | 0.85% | 3,845.8 | 98.4% |
| 0.80 | 74,897 | 1,058 | 1.41% | 4,993.1 | 84.8% |
| 1.00 | 91,222 | 1,733 | 1.90% | 6,081.5 | 73.7% |

This is the real safety/efficiency tradeoff the roadmap asks this analysis to surface, not a
single "AI saves N hours" number presented without its cost. Widening the auto-clear net from
quantile 0.6 to 1.0 frees an extra ~2,236 grader-hours/year, but more than doubles the wrong-clear
rate (0.85% -> 1.90% of auto-cleared cases) -- at 100k patients/year that is the difference
between roughly 489 and 1,733 patients per year told "no follow-up needed" when their true grade
was not 0. **Whether that tradeoff is acceptable is a clinical and policy decision this
simulation cannot make** -- it can only make the tradeoff's actual shape visible in numbers,
which is what it is for.

Also visible directly in this table: at 4 graders, the system is *overloaded* (100% utilisation)
for every quantile up to 0.6, and only clears into a stable regime at 0.8+. The "4 graders is
marginal" finding from Result 1 depends on running the default 0.60 auto-clear quantile; a more
conservative (safer, lower) auto-clear setting would need more graders, not fewer, to stay
stable -- another reason "how many graders" cannot be answered with a single number independent
of the confidence policy chosen alongside it.

## Result 4 — sensitivity: the cited grader-rate range

| Grader rate (cited range) | Utilisation (4 graders) | Turnaround p90 | Turnaround p95 |
|---:|---:|---:|---:|
| 10/hr | 100.0% (overloaded) | 707.2 h | 742.9 h |
| 15/hr (base case) | 98.4% | 6.9 h | 7.7 h |
| 20/hr | 73.7% | 0.23 h | 0.27 h |
| 30/hr | 49.1% | 0.15 h | 0.18 h |

The qualitative conclusion ("is 4 graders enough?") flips entirely across this cited-not-measured
range -- catastrophic collapse at the low end, comfortable at the high end. This is reported as
an explicit limitation, not smoothed over: **this simulation's headline "how many graders"
answer is only as solid as an externally-cited number this project could not itself measure.**
A real district deployment should replace `grader_patients_per_hour` with a measurement from its
own actual graders before trusting any of the specific hour figures above; the *methodology* --
the queueing structure, the two-tier inference split, the auto-clear tradeoff -- is the durable
part of this result, not any single number in isolation.

## What this does not model

- **Diurnal/day-boundary effects.** Time is continuous camp-operating hours, not simulated
  day-by-day with camps closing overnight and any backlog reset. A real programme where graders
  fully clear their queue every evening would show shorter tail latencies than reported here.
- **The true joint distribution of confidence and true grade.** Per-patient confidence quintile
  is drawn independently of true grade (Result 3's model), which reuses Phase 5's real *marginal*
  quintile-accuracy curve but does not reconstruct the full confusion matrix. Stated as a
  simplification in `simulation/simpy/district.py`'s own docstring, not claimed as a
  re-validation of Phase 5.
- **Camera/capture time, and any queueing at the point of capture itself** -- the model starts
  the clock at upload, not at patient arrival at the camera.
- **The Simulink mirror** the roadmap lists as optional was not built (explicitly cut, per the
  roadmap's own scope-cut list: "SimPy suffices; it is a tooling preference, not a result").

## Reproducing this

```
python scripts/benchmark_inference.py          # measures the two throughput numbers above
python scripts/run_simulation_scenarios.py     # runs every scenario in this document, ~2 minutes
```

Both write their output to `runs/` as JSON (`inference_benchmark.json`,
`simulation_scenarios.json`) alongside this document's own numbers, so the two never drift apart
silently.
