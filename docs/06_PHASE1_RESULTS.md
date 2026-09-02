# Phase 1 Results — Baseline Grader

> **This is the number to beat.** Every Phase 3 change is measured as a delta
> against this row. Deliberately un-tuned.

**Run:** `baseline_effb0_512`, fold 0, 2026-09-02 · commit `4cc7eaa`
**Config:** EfficientNet-B0 · 512 px · batch 4 · AdamW lr 1e-4 · grad-clip 1.0 ·
frozen BatchNorm · cross-entropy · cosine schedule with 3-epoch warmup
**Data:** APTOS 2019, 2,929 train / 733 validation, group-disjoint split
**Duration:** 30 of 40 epochs (early stopped, patience 8), ~2 h on Apple M4 (MPS)

---

## Headline

| Metric | Value | 95% CI | Target |
|---|---|---|---|
| **QWK (5-class)** | **0.8930** | [0.8673, 0.9155] | ≥0.90 (Phase 3) |
| Accuracy | 0.8267 | — | — |
| **Referable sensitivity** | **0.919** | [0.887, 0.949] | >0.90 |
| **Referable specificity** | **0.940** | [0.917, 0.962] | >0.85 |
| ECE (uncalibrated) | 0.0646 | — | <0.05 (Phase 5) |

Operating point: threshold 0.9430, selected on validation for ≥90 % sensitivity.
At the naive 0.5 cut the same model gives sens 0.940 / spec 0.929.

Referable cases: 298 of 733.

### How to read this honestly

- **The threshold was selected on the same split it is scored on**, and that split
  also drove early stopping and model selection. These numbers are therefore
  optimistic. The honest figure comes from the locked external test set
  (Messidor-2) in Phase 8, where a ~10-point AUC drop is normal — a faithful
  reproduction of Gulshan 2016 fell from 0.951 on EyePACS to 0.853 on Messidor-2.
- **Sensitivity's CI lower bound is 0.887**, below the 0.90 target. The point
  estimate clears the bar; the interval does not. With 298 referable cases that
  is the best precision available from one fold.
- Single fold, so there is no cross-fold variance estimate. This is a Tier-P
  choice (2 h instead of 11.5 h), not an oversight.

---

## Per-class recall

| Grade | Name | n | Recall |
|---|---|---:|---:|
| 0 | No DR | 361 | 0.981 |
| 1 | Mild (MA only) | 74 | **0.541** |
| 2 | Moderate | 200 | 0.855 |
| 3 | Severe | 39 | **0.359** |
| 4 | PDR | 59 | 0.458 |

## Confusion matrix

Rows = true, columns = predicted.

|  | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | **354** | 7 | 0 | 0 | 0 |
| **1** | 3 | **40** | 29 | 0 | 2 |
| **2** | 0 | 15 | **171** | 10 | 4 |
| **3** | 0 | 0 | 21 | **14** | 4 |
| **4** | 0 | 3 | 22 | 7 | **27** |

---

## What the confusion matrix actually says

### 1. Errors collapse toward grade 2

Grades 1, 3 and 4 all bleed into Moderate: 29/74 of grade 1, 21/39 of grade 3,
22/59 of grade 4. Grade 2 is 27 % of the training data, and plain cross-entropy
has no notion that these classes are *ordered* — it treats 1→2 and 1→4 as the
same mistake, so hedging toward the middle is the loss-minimising strategy.

**This is the direct empirical argument for the Phase 3 ordinal loss.** Recent
SOTA (Dual-SwinOrd, AOR-DR) reports the same and fixes it the same way.

### 2. The errors are mostly clinically *safe*

Of 74 grade-1 cases, **31 were over-called** (to grade 2 or 4) and only **3 were
under-called** to grade 0. Of 39 grade-3 cases, 21 went to grade 2 — still
referable. Almost every error stays inside or moves up the referral boundary.

That is why referable sensitivity is **0.919** while per-class recall for grades
1, 3 and 4 is poor: the model is bad at *grading severity* and good at *deciding
who needs an ophthalmologist*. For a screening tool, the second is the job.

### 3. Grade 1 confirms the sub-pixel microaneurysm problem

Grade 1 is defined as **microaneurysms only**. At 512 px an MA spans ~1.2 pixels
(§2.2 of the analysis). The model detects that something is abnormal — only 3 of
74 are called normal — but cannot distinguish "MAs only" from "more than MAs",
sending 29 to grade 2.

**This is the analysis's central prediction, confirmed on real data.** It is the
strongest possible motivation for the Phase 3 resolution sweep: if MAs are
physically destroyed by downscaling, no loss function or architecture recovers
them.

### 4. Grade 3 is the weakest class, and partly a sample-size artefact

Recall 0.359 on 39 cases. Each case is 2.6 percentage points, so the per-epoch
series (0.87 → 0.18 → 0.64 → 0.03 → 0.46 → 0.00 → 0.49) is substantially noise.
Severe NPDR is defined by the 4-2-1 rule — counting haemorrhages per quadrant and
spotting venous beading/IRMA — which is a *counting and localisation* task a
whole-image classifier is poorly suited to. The Phase 4 lesion branch exists for
exactly this.

---

## Training dynamics worth carrying forward

- **Best epoch 21 (QWK 0.8930); final epoch 29 was 0.8709.** Early stopping fired
  at 30/40. `summary.json` originally reported the *final* epoch while saving the
  *best* checkpoint — fixed, but it under-reported the baseline by 0.022.
- **At a fixed 0.5 threshold, sensitivity drifted down and specificity up as
  training progressed** (0.980/0.885 at ep 12 → 0.859/0.949 at ep 19). The model
  was improving throughout; the probability distribution was sharpening, making a
  fixed cut progressively more conservative. Reading a final sensitivity off 0.5
  would understate the model.
- **Rare-class recall oscillated violently between epochs**, reproducibly across
  two independent runs. Batch 4 gives 4-sample gradient estimates; with 5.3 %
  Severe and 10.1 % Mild, the rare-class boundaries lurch each epoch. Damped as
  cosine decay annealed the LR.
- **Learning rate had to drop to 1e-4 with gradient clipping.** At 3e-4 the loss
  went 0.786 → 239.35 in one epoch and the model collapsed to predicting grade 0.
  See [`05_PROTOTYPE_SCOPE.md`](05_PROTOTYPE_SCOPE.md).

---

## Comparison to published work

| Study | Task | Sens | Spec | QWK |
|---|---|---|---|---|
| **This baseline** | Referable, APTOS val | **0.919** | **0.940** | **0.893** |
| Gulshan 2016 | Referable, EyePACS-1 | 0.975 | 0.934 | — |
| Ting 2017 (SELENA) | Referable, multiethnic | 0.905 | 0.916 | — |
| Abràmoff 2018 (IDx-DR) | mtmDR, prospective | 0.872 | 0.907 | — |
| Gulshan 2019 | Referable, Aravind (India) | 0.889 | 0.922 | — |
| Dual-SwinOrd 2025 | APTOS 5-class | — | — | 0.937 |

**Do not read this as "we beat IDx-DR".** Those studies report on independent,
often prospective test sets with adjudicated labels; this is internal validation
with a threshold chosen on the same split. The comparison establishes that the
baseline is in a plausible range, nothing more.

---

## What Phase 3 must fix, in priority order

1. **Resolution sweep (384/512/768)** — grade 1 is resolution-bound. Highest
   expected gain, and now empirically justified rather than assumed.
2. **Ordinal loss** — the collapse toward grade 2 is exactly what CE incentivises.
3. **Class-balanced sampling or loss weighting** — grades 3 (5.3 %) and 4 (8.1 %).
4. **EyePACS pretraining or RETFound init** — 2,929 training images is small.
5. **Ensemble + TTA** — cheap variance reduction.

Calibration (ECE 0.0646 → target <0.05) is Phase 5. Grade-3 counting is Phase 4.

---

## Reproducing

```bash
python scripts/preprocess.py --dataset aptos --size 512
python scripts/train.py --size 512 --batch-size 4 --epochs 40 --lr 1e-4 --grad-clip 1.0 \
    --folds 0 --run-name baseline_effb0_512
python scripts/evaluate.py --checkpoint models/checkpoints/baseline_effb0_512_fold0/best.ckpt
```

Manifest sha256s are committed, so inputs are verifiable. Seed 42 throughout.
