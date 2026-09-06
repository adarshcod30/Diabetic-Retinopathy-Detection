# Phase 8 Results — The Locked External Evaluation (Run Once)

This is the number the whole project has been building toward, and the one this project committed
— in [`docs/01_PROJECT_ANALYSIS.md`](01_PROJECT_ANALYSIS.md) §8.2, before a single model was
trained — to look at exactly once: "train on APTOS, test on Messidor-2/IDRiD, and touch it exactly
once. Expect the ~10-point AUC drop the Gulshan reproduction study saw — reporting it honestly is
*stronger* than hiding it." That commitment is honoured below. **`scripts/evaluate_external.py`
has now been run against Messidor-2 and IDRiD's locked grading test split. It will not be run
again against a different model choice.**

## What was evaluated, and why two models

[`docs/21_PHASE8_ABLATION_RESULTS.md`](21_PHASE8_ABLATION_RESULTS.md) could not resolve baseline
vs. regression-loss on internal validation alone — each won on a different paired-tested metric,
and referable-AUC (arguably the metric that matters operationally) didn't distinguish them
(DeLong p=0.097). Both were committed as finalists **before** this evaluation ran, specifically so
this single external look could be the tie-breaker instead of a second round of internal tuning.

- **baseline**: `cv_baseline_fold0` (512px, EfficientNet-B0, CE loss)
- **regression**: `sweep_512_regression_fold0` (512px, EfficientNet-B0, regression loss)

Test set: **1,847 images** (1,744 Messidor-2 + 103 IDRiD official grading-test images — 1 of
Messidor-2's 1,748 falls out because `data/manifests/messidor2_512.csv` excludes the small number
of un-adjudicated images per `load_messidor2_labels`'s own filtering). Referable-DR threshold for
each model is **frozen from APTOS internal validation** (`choose_threshold_for_sensitivity`,
target sensitivity 0.90) — never chosen on, or adjusted after seeing, this data.

## Headline result

| | QWK | 95% CI | Sensitivity | 95% CI | Specificity | 95% CI | ECE | Referable AUC |
|---|---:|---|---:|---|---:|---|---:|---:|
| **baseline** | 0.6408 | [0.6033, 0.6756] | 0.415 | [0.372, 0.457] | 0.970 | [0.960, 0.978] | 0.1646 | 0.8878 |
| **regression** | **0.6995** | [0.6622, 0.7317] | 0.441 | [0.398, 0.485] | 0.976 | [0.967, 0.984] | **0.0434** | **0.9242** |

**DeLong test on referable AUC, the same 1,847 patients**: z=-6.187, **p=6.1×10⁻¹⁰**. This is not a
close call. The internal-validation ambiguity is resolved: **regression loses on internal
exact-grade correctness but wins decisively, and significantly, on the metric that actually
generalises.** QWK confirms the same direction (0.700 vs. 0.641, non-overlapping CIs) and ECE is
dramatically better for regression (0.043 vs. 0.165) — the regression model's confidence is simply
more trustworthy on data neither model was tuned on.

**Regression loss is the model this project would release**, on the strength of this one
pre-committed, paired comparison — not on the internal QWK number alone, which by itself would
have been a much weaker basis for the same conclusion (see docs/21's own caution about exactly
that trap).

## The sensitivity number needs a mechanism, not just a headline

Both models land far below the ≥90% sensitivity target (Gulshan 96.1%, Ting 90.5%, IDx-DR 87.2%,
Google/Aravind 88.9% — see the comparison table below) at their frozen operating points. Read on
its own, sensitivity 41.5%/44.1% looks like the model does not work externally at all. That is
not quite what happened, and the difference matters for what to do about it.

Checking the actual score distributions (not just the pass/fail count) shows why:

| | Frozen threshold | Median score, truly-referable external images | Median score, truly-non-referable | Sens. at the *unthresholded default* cut instead |
|---|---:|---:|---:|---:|
| baseline | 0.8613 | 0.251 | ~0 | 45.9% (cut 0.5) |
| regression | 1.7553 | 1.614 | 0.008 | 54.3% (cut 1.5) |

The frozen threshold was chosen to hit 90% sensitivity **on APTOS's own score distribution**. On
Messidor-2/IDRiD, even the median truly-positive image scores well below that threshold — for
baseline, a full order of magnitude below (0.25 vs. 0.86). This is a **threshold-transfer / model
calibration failure**, not primarily a ranking failure: the referable AUC (0.888 baseline, 0.924
regression) shows the models can still separate referable from non-referable reasonably well on
this data — 0.888 is in the same range as the Gulshan reproduction's own Messidor-2 AUC (0.853)
this project pre-registered as the expected drop. What did not survive the domain shift is the
*absolute* score level a fixed cut-point depends on, which is a well-known and expected failure
mode for a single global threshold applied across populations with different cameras, capture
protocols, and label-adjudication processes (APTOS: Indian screening population, Kaggle-competition
single-grader labels; Messidor-2: French population, panel-adjudicated; IDRiD: Indian hospital
population, different camera).

**This project is not re-tuning the threshold on this data.** Doing so now, having already seen
this result, would be exactly the "choosing the threshold on test" cheat §8.2 warns against. The
honest conclusion is stated plainly instead: **a real deployment on a new population needs its own
calibration set to choose an operating point** — the model transfers its ranking ability
reasonably; it does not transfer a specific cut-point. This is a real, documented limitation, not
a caveat to bury.

## Per-dataset and per-quality-tier breakdown

| | n | QWK | Sens | Spec | Referable prevalence |
|---|---:|---:|---:|---:|---:|
| baseline · Messidor-2 | 1,744 | 0.6159 | 0.379 | 0.970 | 26.2% |
| baseline · IDRiD test | 103 | 0.7129 | 0.672 | 0.974 | 62.1% |
| regression · Messidor-2 | 1,744 | 0.6837 | 0.398 | 0.975 | 26.2% |
| regression · IDRiD test | 103 | 0.7176 | 0.750 | 1.000 | 62.1% |

Both models score noticeably better on IDRiD than Messidor-2 despite IDRiD never being touched
during training either — a real, if modest, hint that IDRiD (Indian population, closer to APTOS's
own) transfers slightly better than Messidor-2 (French population), consistent with the
domain-shift story rather than contradicting it. Sample size (103 vs. 1,744) means this
difference is suggestive, not something this evaluation is powered to confirm on its own.

Quality-tier subgroups (an **approximation**: `assess_quality`'s gate is binary usable/reject,
not the 3-tier Good/Usable/Reject classifier the roadmap originally envisioned and Phase 2 never
built — these are sharpness terciles among this test set's own images, stated as an
approximation, not a validated quality classifier):

| | n | baseline QWK | baseline sens | regression QWK | regression sens |
|---|---:|---:|---:|---:|---:|
| low sharpness tercile | 615 | 0.6675 | 0.462 | 0.6823 | 0.431 |
| mid sharpness tercile | 617 | 0.5495 | 0.294 | 0.6533 | 0.325 |
| high sharpness tercile | 615 | 0.6748 | 0.474 | 0.7384 | 0.531 |

Both models are noticeably worse in the **middle** tercile than either the low or high tercile for
both QWK and sensitivity — not a monotonic quality effect. This is worth flagging as genuinely
unexplained rather than forcing a story onto it; a deeper look (camera type, image content, not
just sharpness) would be needed to say why, and this project does not have the metadata to pursue
it further this pass.

## Comparison against published benchmarks

| Study | Task | Data | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| Gulshan 2016 | Referable DR | EyePACS-1 | 97.5% | 93.4% | 0.991 |
| Gulshan 2016 | Referable DR | Messidor-2 | 96.1% | 93.9% | 0.990 |
| Ting 2017 | Referable DR | Singapore, multiethnic | 90.5% | 91.6% | 0.936 |
| Abràmoff 2018 (IDx-DR) | mtmDR, prospective | primary care | 87.2% | 90.7% | — |
| Gulshan 2019 | Referable DR | Aravind, India | 88.9% | 92.2% | 0.963 |
| **this project** | Referable DR | **Messidor-2 + IDRiD, external, once** | **41.5–44.1%** | **97.0–97.6%** | **0.888–0.924** |
| *Gulshan repro. (cited)* | *Referable DR* | *Messidor-2* | — | — | *0.853* |

Read honestly: this project's specificity and AUC land in a defensible range next to the
literature (AUC comparable to or above the cited Gulshan-reproduction external drop). Sensitivity
does not, for the calibration reason above — this is the axis where this project's result is
genuinely weaker than the comparators, all of which either tuned per-deployment or trained on
substantially larger, more diverse populations (EyePACS: ~128,000 images; this project's grading
model saw 3,662). Presenting these side by side rather than only the flattering columns is the
point of this table.

## Failure-mode gallery

The worst errors for both models share a striking pattern: **the single largest errors are almost
all confidently-wrong grade 0 → grade 4 calls** (a healthy retina called Proliferative DR at
near-maximum confidence — referable_score >0.98 of 1.0 for baseline, >3.5 of a 0-4 scale for
regression), not missed referable cases. This is the *opposite* direction from the dominant
sensitivity problem above: the dominant failure mode is under-confidence on true positives
(driving low sensitivity), while the rare, extreme failures are wildly over-confident false
positives on true negatives. These are two distinct, honestly separate failure modes, not one
story.

One image, `IM002441` (Messidor-2, true grade 0), appears in **both** models' top-5 worst errors,
independently — both architectures, trained with different loss functions, catastrophically
misjudge the same specific image as severe disease. This is a genuine shared blind spot worth
recording precisely because it survived a change in loss function; it was not investigated further
visually this pass (would need manual inspection against IDRiD-style lesion criteria, which
Messidor-2 does not provide pixel masks for).

## What this evaluation does and does not license

- **Licenses**: regression loss as the model this project would actually release; the specific,
  numeric case for why (§ headline result); an honest, mechanistically-grounded account of why
  sensitivity collapsed externally (calibration, not pure discrimination failure); a real,
  evidence-based limitation statement for the model card (site-specific recalibration required
  before any deployment).
- **Does not license**: re-tuning the threshold now that this result is visible; claiming this
  project clears its own ≥90%/≥85% sensitivity/specificity target (it does not, externally); using
  the "AUC is comparable to a published external drop" framing to imply the sensitivity number is
  therefore also fine (it is genuinely the weakest result in this project, and the model card must
  say so plainly).
- **Locked test set: now spent.** No future phase of this project may use Messidor-2 or IDRiD's
  grading test split to select a model, tune a threshold, or make any decision that could optimise
  toward this specific 1,847-image sample.
