# Development Roadmap

**Assumption:** ~15–20 focused hours/week (a serious side project alongside coursework).
Total ≈ **20 weeks**. Every phase ends in something *demonstrable* — no phase is pure setup.

**Guiding principle: build the thin vertical slice first.** A mediocre end-to-end pipeline in week 5
is worth more than a brilliant segmentation model in week 14 with nothing around it. You can always
deepen a working slice; you cannot integrate three half-finished subsystems under deadline.

```mermaid
gantt
    title DR Screening Pipeline — 20-Week Plan
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section P0 Foundation
    Repo, env, data audit           :p0, 2026-09-02, 7d
    section P1 Baseline
    Preprocess + baseline grader    :p1, after p0, 14d
    section P2 Vertical Slice
    Quality gate + XAI + demo       :p2, after p1, 14d
    section P3 Grading Depth
    Resolution, ordinal, pretrain   :p3, after p2, 21d
    section P4 Segmentation
    Vessels, OD/fovea, lesions      :p4, after p3, 21d
    section P5 Integration
    Fusion + calibration            :p5, after p4, 14d
    section P6 Rigorous XAI
    Sanity checks + localisation    :p6, after p5, 14d
    section P7 Simulation
    SimPy screening model           :p7, after p6, 10d
    section P8 Validation
    Ablation, external test, stats  :p8, after p7, 14d
    section P9 Release
    Deploy, model card, README      :p9, after p8, 10d
```

---

## Phase 0 — Foundation (Week 1)

**Goal:** a repo someone else could clone and run, and total clarity on what data you actually have.

- [x] Git repo + GitHub remote
- [x] Directory scaffold, `.gitignore`, docs
- [x] Python 3.11 env; MPS confirmed available and in active use throughout training
- [x] `pyproject.toml` — installable `drdetect` package; ruff + pre-commit + pytest skeleton
- [x] Download APTOS, IDRiD, DRIVE; Messidor-2 access (images + grades) obtained and verified
- [ ] **Data audit notebook**: class distribution, image sizes, camera types, duplicate detection,
      and *check whether APTOS has patient IDs* — if not, document the leakage risk explicitly.
      The underlying audits happened (APTOS has no patient IDs; ~139 near-duplicate images found
      and de-duplicated via perceptual hashing), but as conversation output, not a committed
      notebook artifact.
- [x] Build `data/manifests/` with sha256 + labels — `image_id,path,sha256,dataset,label,group_id,phash,width,height`
- [x] **Run `make bench`** and fix the scope from measured throughput, not estimates
      (see [`05_PROTOTYPE_SCOPE.md`](05_PROTOTYPE_SCOPE.md))

**Exit criterion:** `pytest` passes on a clean clone; you can state the exact class distribution of
every dataset from memory.

> **Achieved**, except the data-audit notebook itself. Messidor-2 access (both the personal-info
> image request and the Kaggle-hosted grades) is obtained; images and grades are verified present
> under `data/raw/messidor2/` (1,748 images, `messidor_data.csv`, plus `pairs_left_right.csv` for
> patient eye-pairing — relevant to Phase 8: a patient's two eyes aren't independent samples for
> bootstrap CIs).

---

## Phase 1 — Baseline grader (Weeks 2–3)

**Goal:** a number to beat. Resist tuning.

- [x] Preprocessing module: `crop_from_gray` → `circle_crop` → Ben Graham → resize. Unit-tested.
- [x] Cache pipeline (`scripts/preprocess.py`) — parallel, resumable, builds the manifest
- [x] Split module — **APTOS ships no patient IDs**, so grouping is derived from perceptual
      hashing of source structure; `stratified_group_split` *refuses* to split ungrouped
      unless the caller opts in explicitly
- [x] Metrics module: QWK (validated against sklearn), sens/spec @ referable, bootstrap CIs, ECE
- [x] Baseline trainer (`scripts/train.py`) — EfficientNet-B0, frozen BN, cosine schedule,
      per-class recall logged every epoch
- [x] End-to-end integration test on a synthetic APTOS (76 tests total)
- [x] **Download APTOS** — 3,662 images, distribution verified against published figures
- [x] Run the real baseline — **QWK 0.8930**, see [`06_PHASE1_RESULTS.md`](06_PHASE1_RESULTS.md)
- [ ] W&B logging wired up (CSV logging works today)

**Exit criterion:** a logged baseline QWK on APTOS validation. Expect **~0.80–0.88**. Write the number
down; every future change is measured against it.

> **Achieved: QWK 0.8930** [0.8673, 0.9155], referable sens 0.919 / spec 0.940, 30 epochs, ~2 h.
> Above the expected range. Full analysis: [`06_PHASE1_RESULTS.md`](06_PHASE1_RESULTS.md).

---

## Phase 2 — Thin vertical slice (Weeks 4–5) ⭐

**Goal:** image in → graded, explained PDF out. Ugly is fine. End-to-end is the point.

- [ ] Quality model v1: EfficientNet-B0 on EyeQ, 3-class (train on Kaggle) -- deferred, needs a new
      dataset download and training run; see note below
- [x] Handcrafted quality features (Laplacian variance, illumination zones, FOV fit) + failure reason
- [x] Grad-CAM v1 on the baseline grader
- [x] PDF report generator: image, CAM overlay, grade, confidence, disclaimer
- [x] Gradio demo running locally (MPS/CUDA/CPU, whichever is available)
- [x] `scripts/predict.py` — single image, CPU, no GPU required

**Exit criterion:** you can hand someone a fundus JPEG and get a PDF back. **Demo this.** It is the
moment the project becomes real, and it de-risks everything downstream.

> **Achieved**, out of Phase 3/order: this slice was built after Phase 3's grading ablations rather
> than before them, which inverts the roadmap's own stated principle of building the vertical slice
> first. `python scripts/predict.py --image <photo> --checkpoint models/checkpoints/cv_baseline_fold1/best.ckpt`
> produces a real PDF -- verified end-to-end on raw APTOS training images, including a correct grade-2
> call with Grad-CAM localised on the visible lesion clusters. The quality gate's thresholds were
> **empirically recalibrated**, not assumed: a first pass borrowed a generic blur-detection default
> and rejected 28 of 30 real, clinically-graded APTOS images; the shipped threshold was set from this
> project's own measured sharpness distribution instead (see `src/drdetect/quality/assessment.py`).
> The EyeQ-trained quality classifier remains future work -- the handcrafted gate is a heuristic
> stand-in, not a validated replacement for it.

---

## Phase 3 — Grading depth (Weeks 6–8)

**Goal:** push the grader to publishable quality. Most accuracy gain lives here.

- [ ] **Resolution sweep: 224 / 384 / 512 / 768.** Run this first — it will dominate everything else
      (see § 2.2 of the analysis: MAs are destroyed by downscaling)
- [ ] Backbone comparison: EfficientNetV2-S vs ConvNeXt-Tiny vs Swin-Tiny
- [ ] **Ordinal loss**: regression+thresholds vs CORAL/CORN vs distance-aware label smoothing
- [ ] Class-imbalance handling: balanced sampler *or* weighted loss (not both)
- [ ] EyePACS pretraining on Kaggle → fine-tune on APTOS. *Alternative if quota is tight:* RETFound init
- [ ] Augmentation study: flips, rotation, colour jitter, CLAHE-as-augmentation
- [ ] TTA + 5-fold ensemble

**Exit criterion:** QWK ≥ **0.90** on APTOS validation (Dual-SwinOrd reports 0.937 — that is SOTA, not
your target). Referable-DR sensitivity ≥ 90 % at the chosen operating point.

> **Watch for:** grade-1 recall will be your worst class by a wide margin. Track per-class recall every
> run, not just aggregate QWK — QWK will hide a total grade-1 failure.

> **Concluded, exit criterion not met:** see [`docs/07_PHASE3_RESULTS.md`](07_PHASE3_RESULTS.md), 14
> results across 13 configurations. Baseline QWK 0.8930 stands at α = 0.05. 1024px on Kaggle
> (Result 12, QWK 0.9122) was the closest lead, but its same-backend confirmation (Result 14) found
> the compute-backend change alone — nothing else — moves QWK by 0.031 (p = 0.001) and exact-grade
> accuracy (p = 0.027), a bigger effect than the resolution change Result 12 reported; run locally,
> 1024px is null against the baseline (p = 0.374). Result 12's causal story no longer holds.
> Resolution sweep: 384/512/768/1024 all tried (224 never run), refuted as a lever — grade-1 recall
> falls monotonically through 768px. Backbone comparison: ConvNeXt-Tiny tried and found
> significantly *worse* on exact-grade accuracy, and separately refutes the hypothesis that
> batch-4 BatchNorm noise caused the recurring grade-1-into-grade-2 collapse (Result 13) —
> EfficientNetV2-S and Swin-Tiny untried. Ordinal loss: CORN tried and found not to help (Result
> 2); CORAL, plain regression, and distance-aware CE remain untested (the latter two already have
> working code via `--loss regression`/`--loss distance_ce`, just never run as ablation rows).
> Class-imbalance, augmentation study, TTA, and 5-fold ensembling are all unstarted. EyePACS/
> RETFound pretraining was investigated and found not runnable on this project's local compute
> as-is. A multi-fold confirmation *on Kaggle* remains the only unexplored piece of the original
> 1024px lead, though Result 14 means it can no longer speak to the resolution hypothesis itself.

---

## Phase 4 — Segmentation (Weeks 9–11)

**Goal:** the lesion evidence that powers both fusion and explanation.

- [ ] **Vessels** — U-Net on DRIVE, 48×48 patches. Target Dice ≈ 0.80+, AUC ≈ 0.97+
- [x] **OD & fovea** — heatmap regression on IDRiD. Target: mean localisation error < 0.5 × OD diameter
- [x] **Quadrant mapping** from OD–fovea axis (needed for ICDR's quadrant-based rules)
- [x] **Lesions** — DeepLabv3+/U-Net on IDRiD (only **81 masked images** → 5-fold CV, heavy aug,
      patch sampling at full resolution)
  - [x] Hard exudates first (easiest, highest contrast) — establishes the harness
  - [x] Haemorrhages, soft exudates
  - [x] **Microaneurysms last and separately**: morphological top-hat + matched filter candidate
        generation → small CNN classifier. Do *not* expect plain segmentation to work here.
- [x] Report **AUPRC per lesion class** (AUROC is meaningless at <0.1 % positive pixels)

**Exit criterion:** per-class AUPRC on IDRiD with cross-validated CIs, plus qualitative overlays that
a clinician would recognise.

> **Scope honesty:** neovascularisation has no public pixel masks. Detect PDR at image level; document
> NV segmentation as future work. Do not fabricate it.

> **Partially achieved:** see [`docs/10_PHASE4_RESULTS.md`](10_PHASE4_RESULTS.md) (hard exudates),
> [`docs/11_PHASE4_VESSELS_RESULTS.md`](11_PHASE4_VESSELS_RESULTS.md) (vessels), and
> [`docs/12_PHASE4_LOCALIZATION_RESULTS.md`](12_PHASE4_LOCALIZATION_RESULTS.md) (OD/fovea). Hard
> exudates trained and scored end to end on IDRiD's official split, with the 5-fold CV this item
> specifies: test-set pixel AUPRC **0.8500 ± 0.0292** across 5 folds (tiled full-resolution
> inference, 27 held-out test images, metrics pooled across pixels). Dice is reported at a
> threshold tuned per fold on that fold's own validation split, not a hardcoded 0.5 — the fixed
> threshold's fold-to-fold spread (std 0.080) was over 5x the tuned threshold's (std 0.015), on the
> identical checkpoints. Vessels: 5-fold CV on DRIVE's 20 publicly-labelled images (its official
> test split ships no vessel ground truth at all), DeepLabV3+/resnet34 on full images rather than
> U-Net on patches. AUROC lands close to target and is very stable (**0.9416 ± 0.0035** vs. the
> 0.97+ target); Dice does not (**0.662 ± 0.023** vs. 0.80+) — checked directly, only ~3.5 points
> of that gap is a fixed-threshold artifact, so this is mostly a real capacity/data-scale shortfall
> on fine vessel-branch boundaries, not a scoring artifact. **OD/fovea localisation clears its
> target outright**: heatmap regression (DeepLabV3+/resnet34, 2-channel output) on IDRiD's official
> 413/103 split scores combined mean error **0.075 OD-diameters** on the 103 held-out test images
> (target < 0.5) — 103/103 images clear the target on OD alone, 100/103 on fovea, with three real
> fovea-specific outliers (worst: 1.67 diameters on IDRiD_065) where OD localisation stays accurate
> regardless. **Quadrant mapping** is a deterministic geometric follow-on (two lines through the OD,
> one along the OD-fovea axis, one perpendicular), needing no model — verified by 6 unit tests plus
> a direct check against a real trained prediction. Labels describe geometry ("foveal-side" /
> "disc-side", "superior" / "inferior"), not asserted nasal/temporal anatomy, since eye laterality
> isn't reliably available in this project's datasets to make that mapping safely.
> **Haemorrhages** (see [`docs/13_PHASE4_HAEMORRHAGES_RESULTS.md`](13_PHASE4_HAEMORRHAGES_RESULTS.md)):
> the identical hard-exudate harness, 5-fold CV, internal val AUPRC looks respectable
> (**0.767 ± 0.033**) but drops far more on the held-out test set (**0.540 ± 0.041**) than hard
> exudates did (0.899→0.850) — a plausibly real generalisation gap (haemorrhages are morphologically
> more varied than hard exudates), not yet independently confirmed. **Soft exudates** (see
> [`docs/14_PHASE4_SOFT_EXUDATES_RESULTS.md`](14_PHASE4_SOFT_EXUDATES_RESULTS.md)): the same harness,
> single train/val split (26 training images total, the smallest lesion subset), test AUPRC
> **0.6144** vs. internal val **0.8117** — a gap between hard exudates' and haemorrhages' on the
> identical harness, roughly in line with the roadmap's own prior expectation that this lesion type
> (fewest training examples) would be hard to generalise from. Threshold tuning again helps (Dice
> 0.5546 → 0.5929). Soft exudates and everything after uses a single train/val split rather than
> 5-fold CV, a deliberate scoping decision to manage total time across the remaining Phase 4/5/6
> items. **Microaneurysms** (see
> [`docs/15_PHASE4_MICROANEURYSMS_RESULTS.md`](15_PHASE4_MICROANEURYSMS_RESULTS.md)) closes out this
> item's own required different method: candidate generation (top-hat + percentile threshold) finds
> **91.7% (val) / 96.8% (test)** of true instances — a strong ceiling — but the small CNN classifier
> that accepts/rejects candidates generalises poorly from its curated training distribution to a
> real image's true candidate flood (7,000–40,000 candidates against 10–130 true instances per
> image): test candidate-AUPRC **0.25**, end-to-end recall **0.28** at precision **0.36**. Two real
> bugs were caught and fixed en route — Otsu thresholding catastrophically failing on this response
> distribution (3/18 recall), and a whole-image-padding memory bug that drove candidate generation
> to a 39GB peak footprint before being fixed to 820MB. A plain-segmentation baseline (same harness
> as the other four lesion types, deliberately stopped early) scored val AUPRC 0.51 / Dice 0.24 —
> well below hard exudates' converged numbers even mid-training, consistent with (not a rigorous
> confirmation of) the roadmap's segmentation warning. This is Phase 4's last per-lesion item;
> Phase 5/6 remain.

---

## Phase 5 — Integration & calibration (Weeks 12–13)

**Goal:** turn two models into one system that knows what it does not know.

- [x] Lesion feature extractor: MA count, HE/EX/SE area, per-quadrant distribution, distance-to-fovea
- [x] Fusion head: `concat(CNN embedding, lesion features)` → ordinal head
- [x] Quality gating: route Reject → recapture, Usable → flag in report *(Phase 2)*
- [x] **Temperature scaling** on a held-out split; reliability diagram + ECE before/after
- [x] Uncertainty: MC-dropout or deep ensemble
- [x] **Operating point selection on validation only**, then frozen *(scripts/evaluate.py,
      since Phase 1: `choose_threshold_for_sensitivity(...)` on the selection split,
      `evaluate_at_threshold(...)` on the evaluation split -- used throughout Phase 3's own
      results, docs/07)*
- [x] Human-escalation policy: route bottom-*k* % confidence to a grader; measure the AI+human system
      *(simulated with an assumed-perfect grader -- no real reviewer to measure against exists)*

**Exit criterion:** ECE < 0.05 after calibration; the fusion model beats the grading-only model with a
significant McNemar p-value.

> **Partially achieved:** see [`docs/09_PHASE5_RESULTS.md`](09_PHASE5_RESULTS.md) (calibration) and
> [`docs/16_PHASE5_FUSION_RESULTS.md`](16_PHASE5_FUSION_RESULTS.md) (lesion features + fusion head).
> Temperature scaling is fit, wired into `scripts/predict.py`/the demo/the PDF report (a
> `temperature.json` sidecar per checkpoint, not silently applied to checkpoints never calibrated),
> and validated with a reliability diagram. The baseline checkpoint does NOT clear ECE < 0.05 after
> calibration on either metric tested (0.1347 top-1, 0.0567 referable) -- a non-monotonic
> miscalibration curve a single scalar cannot fully fix. The Result-12 1024px checkpoint does clear
> it on both (0.0462, 0.0414), and was already better-calibrated before scaling.
> **The fusion head's own exit criterion is NOT met**: on 750 APTOS images (lesion features
> extracted by running Phase 4's IDRiD-trained models on APTOS, since APTOS has no lesion ground
> truth of its own -- checked, not eliminated, via a cross-dataset sanity check first), fusion
> scored QWK 0.9362 vs. grading-alone's 0.9556 on the same 150-image val split, and the McNemar
> test on paired correctness is not significant (p=0.754) -- if anything the discordant pairs
> lean toward the baseline. A first honest negative result, not yet root-caused (candidates:
> small sample size, noisy cross-dataset lesion features -- especially the microaneurysm count,
> whose own classifier already under-generalises on IDRiD itself, see docs/15 -- or redundancy
> with what the CNN embedding already encodes). **Operating-point selection was already built** in
> `scripts/evaluate.py` since Phase 1 and used throughout Phase 3 -- not a Phase 5 gap, just not
> previously cross-referenced from this section.
> **Uncertainty + human-escalation, by contrast, work well** (see
> [`docs/17_PHASE5_UNCERTAINTY_RESULTS.md`](17_PHASE5_UNCERTAINTY_RESULTS.md)): MC-dropout on the
> baseline checkpoint (733-image val fold) shows a sharp, highly significant relationship between
> uncertainty and error -- accuracy 99.3% in the two lowest uncertainty quintiles vs. 64.4% in the
> highest (Spearman rho=-0.382, p=8.0e-27). Escalating just the most-uncertain 20% of cases to an
> assumed-perfect grader lifts QWK from 0.894 to 0.940. That perfect-grader assumption is a stated
> ceiling, not a measured human accuracy -- this project has no real reviewer to test against, the
> same constraint Phase 6's clinician timing study already flags.

---

## Phase 6 — Rigorous explainability (Weeks 14–15) ⭐ *the differentiator*

- [x] Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM side by side
- [x] **Adebayo sanity checks**: model-randomisation and data-randomisation tests. Report which methods
      pass. *A negative result is a real result* — most DR papers never run this.
- [x] **Quantified localisation vs IDRiD masks**: pointing game accuracy, CAM–lesion IoU, per-lesion-type
- [x] Lesion-overlay rendering (outlines beat blobs for clinical legibility)
- [x] ICDR evidence table → templated natural-language rationale
- [x] Final PDF report design
- [ ] **Measure the 30-second target**: time ≥20 report reviews; report mean ± SD. Recruit an MBBS
      student if possible — even n=1 clinician feedback beats none

**Exit criterion:** a table of saliency methods × (sanity-check pass/fail, pointing-game accuracy,
IoU). This is the most novel artefact in the project.

> **Partially achieved:** see [`docs/08_PHASE6_RESULTS.md`](08_PHASE6_RESULTS.md). Eigen-CAM fails
> the model-randomisation sanity check outright (its heatmap is provably unchanged by randomising
> the classifier, since it never reads the classifier at all); Grad-CAM++ passes only once enough
> of the network is randomised; plain Grad-CAM — already the Phase 2 default — passes cleanly on
> both checks. Score-CAM's sanity-check behaviour is untested (measured ~15 min/computation on
> CPU makes the full cascade impractical here); it was separately confirmed to run correctly after
> a target-layer bug fix.
> **The IoU/pointing-game item is now done** (see
> [`docs/18_PHASE6_CAM_LOCALIZATION_RESULTS.md`](18_PHASE6_CAM_LOCALIZATION_RESULTS.md)), run
> against all 81 IDRiD segmentation-subset images (Score-CAM excluded, same cost reason as above).
> Headline finding: CAM attention never once lands on a microaneurysm across all 81 images and all
> 3 methods tested (a sub-resolution lesion type, consistent with docs/07's own findings), but
> scores 3.7x-53x above chance for every other lesion type -- haemorrhages, hard exudates, soft
> exudates, and the optic disc. A second finding worth its own attention: Eigen-CAM, which
> *failed* the model-randomisation sanity check above, still posts the *highest* optic-disc
> pointing-game score of the three methods tested (18.0x chance) -- concrete evidence that a
> locally-plausible-looking heatmap is not a substitute for the sanity check, since a method that
> never reads the classifier's weights can still land on a generically-salient structure by luck
> of what fundus photos look like, independent of whether the model is any good.
> **Lesion overlays, the ICDR evidence table, and the redesigned report are also done** (see
> [`docs/19_PHASE6_REPORT_RESULTS.md`](19_PHASE6_REPORT_RESULTS.md)): outline (not filled-blob)
> renders of hard exudates/soft exudates/haemorrhages plus circle markers for accepted
> microaneurysm candidates, a templated rationale grounded in actual per-image detections (not a
> restatement of the grade), and an extended PDF with a third image panel and colour legend --
> verified end to end on a real image, not just unit-tested. That verification image (IDRiD_20,
> predicted grade 3) happened to show haemorrhages in all 4 quadrants, one of ICDR's own defining
> criteria for severe NPDR -- an unprompted, genuine alignment between detected evidence and
> predicted grade, on n=1.
> **This closes every Phase 6 item except the 30-second timing study**, which needs a human
> reviewer this project cannot supply.

---

## Phase 7 — Screening-programme simulation (Weeks 16–17)

- [ ] SimPy discrete-event model: camps → capture → upload → inference → triage → human review
- [ ] Parameterise from **measured** values: your model's real throughput, real image sizes, published
      grader rates. No invented constants.
- [ ] Scenarios: 100,000 patients/year; bandwidth 1/5/10 Mbps; 2/4/8 graders; outage injection
- [ ] Outputs: turnaround time distribution, grader utilisation, backlog under outage, cost/patient,
      **ophthalmologist-hours freed by auto-clearing confident grade-0 cases**
- [ ] Sensitivity analysis on the auto-clear threshold — this links Phase 5's calibration to programme cost
- [ ] *(Optional)* mirror in Simulink via MATLAB Online / campus licence; commit `.slx` + plots

**Exit criterion:** a chart answering "how many graders does a district of 100k need, with and without
AI?" — the number a health administrator would actually use.

---

## Phase 8 — Validation & the ablation (Weeks 18–19)

- [ ] Run the **full ablation grid** (§7 of the analysis) — 11 configurations, one factor per row
- [ ] Evaluate on the **locked** Messidor-2 / IDRiD test set. **Once.**
- [ ] Bootstrap 95 % CIs (2,000 resamples) on every metric
- [ ] DeLong for AUC comparisons; McNemar for paired sens/spec
- [ ] **Subgroup analysis** by quality tier (and camera, if metadata allows)
- [ ] Comparison table against Gulshan 2016/2019, Ting 2017, IDx-DR, Dual-SwinOrd
- [ ] Failure-mode gallery: 20 worst errors, categorised — this is where the interesting findings hide

**Exit criterion:** a results section you would be willing to defend in a viva.

> **Expect the external drop.** A faithful Gulshan reproduction fell from AUC 0.951 (EyePACS) to 0.853
> (Messidor-2). If you drop similarly, analyse the domain shift; do not hide it and do not re-tune.

---

## Phase 9 — Release (Week 20)

- [ ] ONNX export; verify parity with PyTorch outputs
- [ ] FastAPI service + Dockerfile (amd64 + arm64)
- [ ] Gradio demo on HuggingFace Spaces
- [ ] Weights on HF Hub / GitHub Releases with **research-use-only** licence
- [ ] **Model card**: intended use, training populations, measured performance *with* subgroup breakdown,
      known failure modes, "not a medical device"
- [ ] **Dataset card**: provenance, licences, what may and may not be redistributed
- [ ] Production README with the architecture diagrams and headline results
- [ ] `make setup && make evaluate` reproduces the headline table from a clean clone
- [ ] *(Optional)* write it up — this is a workshop-paper-shaped result

---

## Milestone summary

| Week | Milestone | Demonstrable |
|---|---|---|
| 1 | Foundation | Clean clone runs tests |
| 3 | Baseline | QWK number on APTOS |
| **5** | **Vertical slice** ⭐ | **JPEG in → PDF report out** |
| 8 | Strong grader | QWK ≥ 0.90, sens ≥ 90 % |
| 11 | Segmentation | Lesion overlays + AUPRC |
| 13 | Integrated & calibrated | ECE < 0.05, fusion beats baseline |
| **15** | **Rigorous XAI** ⭐ | **Sanity checks + localisation metrics** |
| 17 | Simulation | Graders-needed chart |
| 19 | Validated | Full ablation on locked test set |
| 20 | Released | Live demo + weights + model card |

---

## If you have to cut scope

Cut in this order — the top items are the ones a reviewer will actually miss:

1. ❌ Neovascularisation segmentation (no public masks — already scoped out)
2. ❌ Simulink mirror (SimPy suffices; it is a tooling preference, not a result)
3. ❌ Vessel segmentation (nice for the demo; contributes least to grading accuracy)
4. ❌ EyePACS pretraining (RETFound/ImageNet init is a defensible substitute)
5. ⚠️ Soft-exudate segmentation (fewest training examples, weakest signal)

**Never cut:** patient-level splitting, the locked external test set, calibration, the Adebayo sanity
checks, or the model card. Those are what make it *evidence-based* rather than another APTOS notebook.
