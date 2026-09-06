# Phase 8 Results — Completing the Ablation Grid (Internal Validation)

The remaining untested rows of [§7's ablation table](01_PROJECT_ANALYSIS.md#7-the-ablation-that-answers-integrated--single-technique),
run on APTOS internal validation only (never the locked external set — see "A methodological note"
below for why). Phase 3 already ran 13 configurations (docs/07) and found the baseline
undefeated; this closes out the resolution floor, one preprocessing hypothesis, and two ordinal
losses Phase 3's own writeup flagged as having working code but never run as dedicated rows.

## Method

Four single-split (fold 0) training runs, same harness as Phase 3 (`scripts/train.py`, no new
code needed for three of the four — CLAHE required extending `scripts/preprocess.py` with the
`idrid` dataset case done for Phase 8's external test set, not for CLAHE itself, which already had
a `--clahe` flag sitting unused since Phase 1). Each candidate is then scored with
`scripts/evaluate.py` on the *identical* fold-0 held-out images the baseline was evaluated on
(same seed, `n_splits=5`, `fold=0`), and compared against the baseline with **both** a paired
McNemar test (exact-grade correctness) and a paired DeLong test (referable-DR AUC) —
`scripts/analyze_ablation_grid.py`. A bootstrap CI or a numerically higher point estimate alone is
not treated as a result; every claim below is backed by a paired test on the same images.

## Result

| # | Configuration | QWK | 95% CI | McNemar vs. baseline | Referable AUC (DeLong) |
|---|---|---:|---|---|---|
| — | baseline (512px, CE, ImageNet init) | 0.8967 | [0.8718, 0.9202] | — | 0.9755 |
| 1 | 224px, CE (the resolution sweep's never-run floor) | 0.8844 | [0.8588, 0.9064] | p=0.064, baseline ahead (29 vs 46 discordant) | 0.9678, p=0.181 |
| 4 | + CLAHE, 512px, CE | 0.8984 | [0.8753, 0.9222] | **p=0.899 — no real difference** | 0.9739, p=0.710 |
| 5b | + regression loss, 512px | **0.9147** | [0.8969, 0.9314] | **p=0.023, baseline ahead (32 vs 54)** | 0.9815, p=0.097 |
| 5c | + distance-aware CE loss, 512px | 0.8944 | [0.8744, 0.9134] | **p=0.0012, baseline ahead (33 vs 66)** | 0.9770, p=0.651 |

Row numbers follow §7's table; 5b/5c are the "ordinal loss" row's two untested variants (CORN was
already tried in Phase 3, Result 2, and found not to help).

### 224px: the resolution floor doesn't help either

Consistent with the rest of this project's resolution findings (docs/07: 384/512/768/1024 all
tried, none beat 512px). 224px trends toward the baseline being better (McNemar p=0.064, just
short of conventional significance) — no evidence that going *lower* than 512px recovers anything
the higher-resolution runs lost. The docs/01 §2.2 hypothesis (an MA needs ≥512px to survive
resizing) is not contradicted by this result, but 224px was the floor where an MA would be fully
destroyed, and destroying it further doesn't measurably change QWK — suggesting the model was not
leaning heavily on microaneurysm-scale detail at 512px in the first place, a point this project has
now touched from two directions (docs/07's monotonic grade-1-recall decline, and this null result).

### CLAHE: a higher number that evaporates under a paired test

QWK 0.8984 looks like a small win over 0.8967 — until McNemar (30 vs. 32 discordant pairs,
p=0.899) shows the two models are indistinguishable on the same images. This is exactly the
scenario `docs/09_PHASE5_RESULTS.md`'s own warning about overlapping-but-not-tested CIs exists to
catch, and a clean demonstration of why this project insists on paired tests rather than reading
point estimates: without the McNemar test, CLAHE would have looked like a (small, unremarkable)
win. **CLAHE does not ship as a preprocessing default based on this result.**

### Regression loss: a real, mechanistically-explained trade-off, not a clean win

This is the first configuration in this project's 17-configuration ablation history (13 in Phase 3,
4 here) to clear QWK 0.90, with a bootstrap CI that barely overlaps the baseline's. It would be a
mistake to stop reading there.

The paired McNemar test tells a different story: on the same 733 held-out images, the baseline is
exactly correct on 54 images the regression model gets wrong, while the regression model is only
exactly correct on 32 images the baseline gets wrong (p=0.023 — significant, favouring the
baseline). Raw accuracy confirms this directly: baseline 84.3% (618/733) vs. regression 81.3%
(596/733). **The regression model is measurably worse at getting the exact grade right.**

So why is its QWK higher? The confusion matrices show why. Baseline's worst errors on true PDR
(grade 4) land 18/59 cases two full grades away, at "Moderate" — QWK's quadratic penalty scores
that harshly. Regression's true-PDR errors land mostly one grade away, at "Severe" (20/59), with
fewer two-grade jumps (12/59). Regression is trading exact-match accuracy for *never being far
wrong* — a direct, expected consequence of optimising a continuous ordinal scale rather than
categorical cross-entropy, and precisely why regression-style heads are popular in QWK-scored
Kaggle competitions. It is a real, reproducible mechanism, not noise (unlike CLAHE above) — but
"which matters more, QWK or exact-grade accuracy" is a genuine trade-off this internal-validation
comparison cannot resolve on its own, especially since referable-DR AUC — arguably the single most
operationally important number, since it drives the actual refer/don't-refer decision — shows no
significant difference either way (DeLong p=0.097).

### Distance-aware CE: a real downgrade, not a wash

QWK 0.8944 looks close to baseline's 0.8967, but McNemar is unambiguous here (33 vs. 66
discordant pairs, p=0.0012) — the baseline is significantly more often exactly correct. Unlike
CLAHE's "looks different, isn't," this is "looks similar, is significantly different" — the
opposite failure mode a QWK-only read would produce, and another argument for never skipping the
paired test.

## A methodological note: resolving a tension in this project's own planning documents

[§7 of the analysis](01_PROJECT_ANALYSIS.md#7-the-ablation-that-answers-integrated--single-technique)
says its 11-row table should be "evaluated **on the locked external test set**." [§8.2 of the same
document](01_PROJECT_ANALYSIS.md#8-validation-rigour--the-five-things-that-will-otherwise-sink-this)
says to "touch it exactly once." Read literally, these conflict: scoring 11 configurations against
the locked set is scoring it (at minimum) 11 times, not once. The roadmap's own Phase 8 entry
already resolves this the right way without saying so explicitly, by listing "run the full ablation
grid" and "evaluate on the locked test set. Once." as **two separate checklist items** — this
document is the former; `docs/22_PHASE8_VALIDATION_RESULTS.md` (once run) is the latter. Every
comparison above happened on APTOS internal validation, which the baseline and every Phase 3
config were also selected on; nothing here has touched Messidor-2 or IDRiD's grading test split.
This is the same discipline Phase 5's operating-point selection already used
(`choose_threshold_for_sensitivity` on validation, `evaluate_at_threshold` on test, never
conflated) — extended here from a single hyperparameter to the model-selection decision itself.

One consequence: a true 5-fold **ensemble** (row 9's other half, alongside TTA — see
`docs/20_PHASE7_SIMULATION_RESULTS.md`'s own note and `docs/07`'s microaneurysm/fusion discussion
for related leakage reasoning) cannot be tested honestly on any APTOS split, internal or otherwise
— every fold's model was trained on the other four folds' data, so any APTOS image is "held out"
for at most one of the five fold checkpoints and "seen in training" by the other four. An ensemble
of the five `cv_baseline_fold{0-4}` checkpoints can only be evaluated without leakage on data none
of the five ever trained on — the external test set. That comparison, if made, happens in
docs/22, not here.

## Decision: two finalists carry forward, not one

Internal validation does not cleanly resolve baseline vs. regression-loss: each wins on a
different real, paired-tested metric, and the metric that plausibly matters most operationally
(referable AUC) doesn't distinguish them. Rather than pick one on ambiguous internal evidence,
**both are carried forward as the finalists into the single locked external evaluation** —
224px and distance-aware CE are dropped, since both are significantly or near-significantly
behind the baseline with no compensating strength. Whatever the external set shows for these two,
it is looked at once, per the discipline above.
