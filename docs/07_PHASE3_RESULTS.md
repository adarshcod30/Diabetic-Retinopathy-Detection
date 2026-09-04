# Phase 3 Results — Grading Depth

Running log of the ablation. Every row is one factor changed from the Phase 1
baseline, evaluated on the same fold-0 validation split (733 images, 298
referable) with **paired** significance tests.

> **Why paired tests.** Two models scored on the same images make correlated
> errors. Comparing their marginal confidence intervals discards that pairing and
> is badly underpowered — intervals can overlap while the paired difference is
> highly significant. `scripts/compare.py` uses McNemar on discordant pairs and a
> paired bootstrap for QWK.

> **Reading order note.** This is a running lab log: findings are appended near
> the section they extend, not strictly in numeric order. Result 6 (a `###`
> subsection, not `##`) sits before Result 4 physically, because it was added
> while extending the discussion Result 3 started. Use this index, not scroll
> position, to navigate by number:
>
> | # | Finding | Where |
> |---|---|---|
> | 1 | Resolution 384 vs 512 — no difference; both sub-MA | [§](#result-1--resolution-384-vs-512-no-significant-difference) |
> | 2 | Unweighted CORN actively hurts (80% positive conditional subset) | [§](#result-2--corn-does-not-help-and-unweighted-corn-actively-hurts) |
> | 3 | QWK selection discards balanced checkpoints | [§](#result-3--task-balancing-works-during-training-but-is-discarded-at-selection) |
> | 4 | Macro-recall selection is clinically unsafe (2.5× missed referrals) | [§](#result-4--macro-recall-selection-improves-class-balance-and-degrades-referral-safety) |
> | 5 | Resolution refuted outright (grade-1 recall falls monotonically) | [§](#result-5--resolution-does-not-help-and-grade-1-recall-gets-worse) |
> | 6 | Collapse is NOT learning-rate-locked (warmup control) | [§](#result-6--the-collapse-is-not-learning-rate-locked) |
> | 7 | Every effect size is inside same-seed noise (no noise floor existed) | [§](#result-7--every-phase-3-effect-size-is-inside-the-same-seed-noise) |
> | 8 | Accumulation kills the collapse; selection still picks a hedged model | [§](#result-8--accumulation-eliminates-the-sharp-collapse-but-not-at-the-checkpoint-that-gets-selected) |
> | 9 | Sensitivity-at-floor avoids the collapse but saturates; the only significant (negative) QWK result | [§](#result-9--plain-sensitivity-selection-is-unsafe-for-the-identical-reason-as-macro-recall) |
> | 10 | 5-fold CV: baseline is stable (0.8965 ± 0.0116) but QWK ≥ 0.90 is a ~40% background rate; grade-3 recall is consistently poor across every fold | [§](#result-10--5-fold-cross-validation-the-baseline-is-stable-and-090-is-a-coin-flip) |
> | 11 | Recalibrated floor (0.92) fixes the saturation from Result 9 -- span widens ~5x, QWK returns to null (not significant) instead of significantly worse | [§](#result-11--recalibrating-the-floor-to-092-fixes-the-saturation-qwk-returns-to-null) |
> | 12 | 1024px (Kaggle, T4): QWK +0.0192 but p=0.104, still not significant -- yet exact-grade accuracy is significantly better (p=0.0003). The closest to a win in the whole phase, on a different metric than the one every other row was judged on | [§](#result-12--1024-px-qwk-still-not-significant-but-exact-grade-accuracy-is) |

---

## Ablation so far

| # | Configuration | QWK | Sens | Spec | ECE | vs baseline |
|---|---|---|---|---|---|---|
| 1 | **Baseline** — EffNet-B0, 512 px, CE | **0.8930** | 0.919 | 0.940 | 0.0646 | — |
| 2 | 384 px, CE | 0.8845 | 0.906 | 0.956 | 0.0481 | p = 0.490, **n.s.** |
| 3 | 512 px, **CORN** ordinal | 0.8951 | 0.913 | 0.940 | 0.0550 | p = 0.851, **n.s.** |
| 4 | 512 px, **CORN + task balancing** | 0.8950 | 0.911 | 0.940 | 0.0391 | p = 0.734, **n.s.** |
| 5 | 512 px, CORN + macro-recall selection | 0.8790 | 0.903 | 0.959 | 0.0620 | p = 0.749, **n.s.** |
| 6 | **768 px, CE** | 0.8986 | 0.926 | 0.933 | 0.0649 | p = 0.603, **n.s.** |
| 7 | 512 px, warmup 8 (accum 1, control) | 0.8958 | 0.903 | 0.940 | 0.0605 | p = 0.910, **n.s.** vs baseline |
| 8 | 512 px, warmup 8 + **accum 4** | 0.8942 | 0.903 | 0.945 | **0.0565** | p = 0.860, **n.s.** vs control |
| 9 | 512 px, **`--monitor val/sens_at_spec85`** (floor 0.85) | 0.8557 | 0.913 | 0.945 | 0.0434 | p = 0.003, **SIGNIFICANT (worse)** |
| 10 | 512 px, **`--spec-floor 0.92`** (recalibrated) | 0.8802 | 0.909 | 0.952 | 0.0553 | p = 0.272, **n.s.** |
| 11 | **1024 px** (Kaggle T4) | 0.9122 | 0.909 | 0.961 | **0.0421** | p = 0.104, n.s. on QWK — **exact-grade accuracy p = 0.0003, SIGNIFICANT** |

**Ten experiments null on QWK, one significant negative result, one fix that
returned it to null, and one configuration that came closest to a real win —
on a metric other than the one everything else was judged on.** Result 9's
floor-constrained sensitivity metric was built to fix a real, measured flaw
in plain sensitivity selection, succeeded at that narrow goal, and then
produced a QWK result **worse than the baseline** (p = 0.003), driven by
grade-3 recall collapsing to 0.051 -- caused by the 0.85 floor saturating
(Result 9). Recalibrating that floor to 0.92 (Result 11) fixed the
saturation it was diagnosed for -- the metric's discriminating span widened
roughly 5x -- and QWK returned to a null result (p = 0.272), not a win.
1024 px (Result 12) -- the first resolution at which microaneurysms are
actually resolvable, per the size table in Result 1 -- posted the highest
QWK of the phase (0.9122) and clears QWK ≥ 0.90 outright, but the paired
test against baseline lands at p = 0.104: **not significant at α = 0.05**.
What *is* significant is a different, real measure of the same thing --
getting the exact ICDR grade right (p = 0.0003). No configuration has beaten
the baseline on QWK at α = 0.05 across eleven configurations, and that
includes the refutation of this project's central architectural hypothesis
at sub-MA resolutions (Result 5) -- but 1024 px is the first result close
enough, and the first with an independently significant finding at all, to
be worth a second look with a larger sample before calling the hypothesis
closed.

---

## Result 1 — Resolution 384 vs 512: no significant difference

```
QWK difference (512 - 384)   +0.0085   95% CI [-0.0151, +0.0341]   p = 0.490
Exact grade (McNemar)        37 vs 44 discordant                   p = 0.505
Referable decision (McNemar) 15 vs 18 discordant                   p = 0.728
```

Per-class recall:

| Grade | n | 384 px | 512 px | Δ |
|---|---:|---:|---:|---:|
| 0 No DR | 361 | 0.986 | 0.981 | −0.006 |
| 1 Mild | 74 | **0.622** | 0.541 | −0.081 |
| 2 Moderate | 200 | 0.845 | 0.855 | +0.010 |
| 3 Severe | 39 | 0.282 | **0.359** | +0.077 |
| 4 PDR | 59 | **0.525** | 0.458 | −0.068 |

### This refutes the stated hypothesis — but the experiment could not have tested it

[`01_PROJECT_ANALYSIS.md` §2.2](01_PROJECT_ANALYSIS.md) argued resolution is the
single most important hyperparameter, because grade 1 is defined by
microaneurysms and MAs are destroyed by downscaling. The prediction was that
grade-1 recall would rise with resolution. It did not: grade 1 was *better* at
384 (0.622 vs 0.541), and that difference is 6 images out of 74 — noise.

**But the comparison was mis-specified.** An MA spans roughly 0.23 % of image
width, so:

| Resolution | MA size | Resolvable? |
|---|---|---|
| 384 px | 0.9 px | no |
| 512 px | 1.2 px | no |
| 768 px | 1.8 px | marginal |
| 1024 px | 2.4 px | yes |

384 and 512 are **both below the threshold at which microaneurysms exist in the
input at all**. Comparing them tests nothing about the hypothesis; the null
result is what the hypothesis itself predicts for two sub-MA resolutions.

**Revised claim:** resolution should matter *across the MA threshold* (≥768,
ideally 1024), not uniformly. The 384-vs-512 null is consistent with that and is
not evidence against it — but neither is it evidence for it. Until 768+ is run,
the honest statement is that **resolution has not been shown to matter in the
tested range**, and the §2.2 claim is downgraded from established to untested.

### Secondary observations

- **384 is strictly cheaper**: 20 epochs to converge vs 30, 85 min vs 2.3 h, and
  peak 2.19 GB vs 2.27 GB. For equal accuracy that makes 384 the better default
  for iteration, keeping 512+ for final models.
- **384 was better calibrated** (ECE 0.0481 vs 0.0646) and more specific
  (0.956 vs 0.940) at a slightly lower sensitivity. Neither difference is
  significant, but both favour the cheaper model.
- **Grade 3 remains the weakest class in both** (0.28 / 0.36 on 39 cases). It is
  defined by the 4-2-1 rule — counting haemorrhages per quadrant, spotting venous
  beading and IRMA — which is a counting-and-localisation task a whole-image
  classifier is structurally poor at. No resolution change will fix that; the
  Phase 4 lesion branch is the right instrument.

---

## Method note

384 px was trained by downscaling the 512 px cache, which is valid: 384 requires
no detail the 512 cache lacks. 768 px required re-preprocessing from the raw
3000×2000 PNGs — training "at 768" from a 512 cache would upscale and gain
nothing, since the detail was destroyed at cache time. The 768 manifest was
verified to carry identical grades and identical 3,523 groups, so the split is
byte-identical across resolutions.


---

## Result 2 — CORN does not help, and unweighted CORN actively hurts

```
QWK difference (CORN - CE)   +0.0021   95% CI [-0.0183, +0.0232]   p = 0.851
Exact grade (McNemar)        36 vs 45 discordant                   p = 0.374
Referable decision (McNemar) 16 vs 18 discordant                   p = 0.864
```

| Grade | n | CE | CORN | Δ |
|---|---:|---:|---:|---:|
| 1 Mild | 74 | 0.541 | **0.297** | **−0.243** |
| 2 Moderate | 200 | 0.855 | **0.905** | **+0.050** |
| 3 Severe | 39 | 0.359 | 0.385 | +0.026 |

CORN was chosen to stop the collapse of errors into grade 2. It made that
collapse **worse**: 49 of 74 grade-1 cases went to grade 2, against 29 for
cross-entropy.

### The cause is structural

CORN trains task *j* only on samples with `y ≥ j`, asking whether `y > j`. Those
conditional subsets inherit the label skew. Measured on the training split:

| task | subset `{y≥j}` | positive rate | |
|---|---:|---:|---|
| j=0 | 2929 | 50.7 % | balanced |
| **j=1** | **1485** | **80.1 %** | **badly skewed** |
| j=2 | 1189 | 32.8 % | skewed |
| j=3 | 390 | 60.5 % | balanced |

Task j=1 asks *"given grade ≥ 1, is it worse than 1?"* and is 80 % positive,
because grade 1 is only 296 of the 1,485 samples with grade ≥ 1. Unweighted, the
loss-minimising answer is almost always *yes* — pushing grade 1 into grade 2.

**The finding is not that CORN fails, but that unweighted CORN cannot work on a
skewed ordinal distribution.** Its own construction guarantees it.

`corn_task_pos_weights()` now computes `n_neg / n_pos` per conditional subset —
`[0.972, 0.249, 2.049, 0.653]` here — and is enabled by default.

---

## Result 3 — Task balancing works during training but is discarded at selection

Balanced CORN reached QWK 0.8950 (p = 0.734 vs baseline, n.s.) with per-class
recall *worse* than the baseline at the selected checkpoint — grade 1 0.378,
grade 3 0.308.

But that is not what training produced. Mid-run epochs showed exactly the
intended effect: epoch 8 gave grade 1 = 0.76 and grade 3 = 0.54; epoch 18 gave
grade 1 = 0.81, the highest of any run. **Those checkpoints were discarded**,
because `ModelCheckpoint(monitor="val/qwk")` keeps only the highest-QWK epoch.

### QWK is a poor selection criterion when rare classes matter

Correlation between QWK and per-class recall, across 70 epochs of three runs:

| Run | corr(QWK, g2) | corr(QWK, g1) | corr(QWK, g3) |
|---|---:|---:|---:|
| CE baseline | −0.039 | +0.345 | +0.548 |
| CORN | +0.124 | −0.299 | +0.363 |
| **CORN-balanced** | **+0.793** | **−0.702** | +0.530 |
| **Pooled (n=70)** | **+0.512** *(p<0.0001)* | −0.232 *(p=0.053)* | +0.402 *(p=0.0006)* |

The general claim "QWK selects against rare classes" is **too strong** — QWK
correlates *positively* with grade-3 recall. The accurate statement is narrower:

- QWK tracks **grade-2** recall strongly and significantly (r = +0.51).
- For **grade 1** it is weakly negative and borderline (r = −0.23, p = 0.053).
- In the balanced-CORN run the anti-correlation was severe (g1 r = −0.70).

Concretely, in that run selecting on QWK kept epoch 12 (macro-recall 0.596) over
epoch 9 (macro-recall 0.697) — **0.100 macro-recall surrendered for 0.0097 QWK.**

`val/macro_recall` is now logged every epoch, and `--monitor` selects the
checkpoint criterion (`val/qwk`, `val/macro_recall`, or
`val/sensitivity_referable`). The next balanced-CORN run should be selected on
macro-recall, since QWK selection demonstrably discards the models the method
produces.

---

## Cross-cutting observation — CORRECTED

> ⚠️ **An earlier version of this section was wrong on two factual claims, and the
> correction matters because it is the premise for the gradient-accumulation
> experiment.** Both errors were found by re-reading the metrics CSVs the section
> itself cites. The original text is superseded by what follows.

### What was claimed, and what the data show

**Claim 1 — "rare-class recall looks balanced at epochs 1–2, then collapses."**
False. Grade-3 recall is **exactly 0.000 at epochs 0 and 1** in all three CE runs,
and grade 4 is 0.000–0.051. Only **epoch 2** is balanced.

| Run | ep0 g3 | ep1 g3 | ep2 g3 | ep3 g3 |
|---|---:|---:|---:|---:|
| 512 CE | 0.000 | 0.000 | 0.641 | 0.000 |
| 384 CE | 0.000 | 0.000 | 0.718 | 0.000 |
| 768 CE | 0.000 | 0.000 | 0.692 | 0.000 |

The real shape is **hedged → one balanced epoch → total collapse**, not
"balanced then collapse".

**Claim 2 — "four runs, three different losses, same inflection point."**
False. The epoch-3 grade-2 saturation is **cross-entropy only**:

| Run | loss | grade-2 recall @ ep3 |
|---|---|---:|
| 512 CE | CE | **1.000** |
| 384 CE | CE | **1.000** |
| 768 CE | CE | **1.000** |
| 512 CORN | CORN | 0.835 |
| 512 CORN-bal | CORN + task weights | 0.275 |
| 512 CORN-macro | CORN + task weights | 0.550 |

Three CE runs saturate exactly; no CORN run does. The phenomenon is **n = 3
across resolutions with one loss**, not n = 6 across four losses.

### What can actually be said

At epoch 3 — the first full-LR epoch — all three CE runs enter a **total
single-class state**: grade-2 recall exactly 1.000, grades 1 and 3 at 0.000–0.014,
referable sensitivity exactly 1.000 with specificity at each run's minimum. Epoch 3
is also the **only epoch whose training loss rises in all three** (0.705→0.728,
0.719→0.765, 0.673→0.708).

A training loss that *increases* precisely when the LR reaches full value, together
with collapse onto one class, is consistent with an optimisation step that is too
large. But that is now a claim about **cross-entropy at full LR**, not a
loss-invariant property.

### Result 6 — the collapse is NOT learning-rate-locked

A control run held everything at the baseline and changed only **when peak LR
arrives**, moving warmup from 3 epochs to 8 (peak LR from cumulative step 2196 to
5856). Batch, LR, loss, class prior, seed, epoch budget and steps/epoch are
identical.

| Run | collapse epoch | LR there | % of peak |
|---|---:|---:|---:|
| baseline, warmup 3 | **3** | 1.00e-4 | **100 %** |
| control, warmup 8 | **3** | 4.44e-5 | **44 %** |

The collapse did not move. Same epoch, same exact signature (grade-1 0.000,
grade-2 **1.000**, grade-3 0.000), at less than half the learning rate. Both runs
also share the run-up: one balanced epoch at 2 (grade-3 0.641 and 0.513) followed
by total collapse at 3.

**The "at full LR" conjunct is refuted.** More than doubling the warmup changed
nothing. Whatever drives the epoch-3 CE collapse, the learning-rate magnitude is
not it.

### Reaching full LR causes nothing — the other half of the refutation

The control's warmup ends at epoch 8, so that is where it first reaches peak LR:

| epoch | LR | % of peak | grade-2 recall | |
|---:|---:|---:|---:|---|
| 2 | 3.33e-5 | 33 % | 0.920 | |
| **3** | **4.44e-5** | **44 %** | **1.000** | **COLLAPSE** |
| 7 | 8.89e-5 | 89 % | 0.960 | |
| **8** | **1.00e-4** | **100 %** | **0.905** | full LR arrives — nothing happens |
| 9 | 9.98e-5 | 100 % | 0.885 | |

The original claim was that the collapse occurs "exactly when LR warmup ends and
the rate reaches full value". **Both halves are false, and this one run falsifies
both**: it fires at 44 % of peak, before warmup ends, and warmup ending produces
no collapse at all. Epoch 8 was in fact the best epoch to that point (QWK 0.883).

The control also tracks the baseline epoch-for-epoch at less than half the LR —
at epoch 4, grade-1/2/3 recall was 0.797/0.645/0.026 for the baseline and
0.838/0.650/0.026 for the control.

---

## Result 7 — every Phase 3 effect size is inside the same-seed noise

`Trainer(deterministic=False)`, because some MPS kernels have no deterministic
variant. Three runs at **identical config and seed 42** gave epoch-0 QWK of:

```
0.744    0.819    0.774        range 0.075, sd 0.038
```

Against the ablation's measured differences:

| Comparison | Δ QWK | vs same-seed range |
|---|---:|---|
| 384 vs 512 | +0.0085 | within |
| CORN vs CE | +0.0021 | within |
| CORN-balanced vs CE | +0.0020 | within |
| CORN-macro vs CE | −0.0140 | within |
| 768 vs 512 | +0.0056 | within |

**Every effect is 5–35× smaller than the variation between runs that differ in
nothing.** The five paired nulls were therefore correct but under-described: they
mean *underpowered to detect an effect this small*, not *no effect exists*.

### Caveat

The variance measured is at **epoch 0**, the noisiest point of training. The
reported figures are best-QWK over ~20–30 epochs, and a maximum over many epochs
may be considerably more stable. This is suggestive, not established — settling it
needs repeat runs at fixed config, reporting best-QWK spread.

### Consequence

Two things follow regardless of how the caveat resolves:

1. **Report a noise floor before reporting a delta.** Three repeats of the
   baseline would cost ~7 h and would tell every later comparison what it must
   exceed to mean anything. Nothing in Phase 3 had that.
2. **Prefer categorical readouts where possible.** The accumulation test below
   reads a signature at one of two epochs 13 apart, which no amount of this
   variance can move. That is why it is a sound experiment where a QWK-delta
   version would not have been.

---

### What is still confounded, and the experiment that separates it

Both runs use `--accum 1`, so epoch 3 is **2,928 optimiser steps** in both.
"Locked to optimiser steps" and "locked to epochs (data passes)" remain
indistinguishable.

Gradient accumulation separates them, and this — not the original noise argument
— is now its justification:

| Hypothesis | Prediction at `--accum 4` (183 steps/epoch) |
|---|---|
| **step-locked** (~2,928 steps) | collapse at **epoch 16** |
| **epoch-locked** (data passes) | collapse stays at **epoch 3** |

A 13-epoch separation, far outside run-to-run noise, from a categorical signature
rather than a metric delta. That is a far stronger experiment than the one
originally planned, which measured a QWK difference against ~0.075 of same-seed
variance.

Note this also means the accumulation run's earlier problem — that `--grad-clip
1.0` pins step magnitude identically in both arms (every step clipped: 0.985 at
accum 1, 1.000 at accum 4) — no longer matters. The readout is *where* the
signature appears, not how large a step was taken.

### Consequence for the accumulation experiment

The original argument was "identical across four losses ⟹ optimisation, not loss".
**That argument is dead.** Gradient accumulation is still worth testing — the
epoch-3 CE collapse is real, sharp and reproducible across three resolutions, and
lower gradient noise is a plausible cause — but the experiment must be read as a
test of *"is the CE epoch-3 collapse driven by gradient noise?"*, not as
adjudicating loss-versus-optimisation in general.

There was also a confound no accumulation run could break: full LR and epoch 3
arrived together. The warmup control (Result 6) broke it, and the answer is that
**LR magnitude is not the driver** — the collapse stayed at epoch 3 at 44 % of
peak LR.


---

## Result 4 — Macro-recall selection improves class balance and *degrades referral safety*

Selecting the checkpoint on `val/macro_recall` did what Result 3 predicted: the
rare classes improved substantially.

| Grade | n | CE | CORN-macro | Δ |
|---|---:|---:|---:|---:|
| 0 No DR | 361 | 0.981 | 0.989 | +0.008 |
| 1 Mild | 74 | 0.541 | **0.716** | **+0.176** |
| 2 Moderate | 200 | 0.855 | **0.630** | **−0.225** |
| 3 Severe | 39 | 0.359 | **0.564** | **+0.205** |
| 4 PDR | 59 | 0.458 | 0.492 | +0.034 |

**But the gain was paid for in the one place a screening tool cannot afford it.**

| | CE | CORN-macro |
|---|---:|---:|
| Grade-2 (referable) cases predicted grade 0 or 1 | 15/200 (7.5 %) | **39/200 (19.5 %)** |
| All referable cases missed at argmax | 18/298 (6.0 %) | **46/298 (15.4 %)** |

### Why this happens, and why it was foreseeable

Macro-averaged recall is the unweighted mean over classes. It therefore scores
**grade 2 → grade 1** and **grade 2 → grade 3** identically. Clinically they are
nothing alike: grade 2 is referable and grade 1 is not, so the first crosses the
referral boundary and the second stays safely inside it.

Optimising a boundary-blind metric produced a model that redistributes grade 2
in both directions, and the downward half is harmful. Grade-2 recall fell 0.225,
and 38 of those cases landed in grade 1.

### The threshold hides it

At each model's chosen operating point the two are statistically indistinguishable
— sensitivity 0.920 vs 0.903, McNemar p = 0.749 — because the threshold is applied
to the continuous referable score, not to argmax. That is precisely what makes
this dangerous: **the aggregate operating-point metrics look fine while the
underlying grade assignment is materially less safe**, and the model depends more
heavily on threshold tuning to compensate.

### Correction to Result 3's recommendation

Result 3 concluded that "the next balanced-CORN run should be selected on
macro-recall". **That recommendation was wrong.** Macro-recall is a better
selection criterion than QWK *for per-class balance*, and a worse one for the
clinical task.

The right criterion for a referral tool must respect the referral boundary.
`--monitor val/sensitivity_referable` already exists and is the better default;
a cost-weighted objective that penalises boundary crossings asymmetrically would
be better still, and is untested.

### What this says about the project's metrics generally

Three selection criteria, three different failure modes:

| Criterion | Optimises | Fails by |
|---|---|---|
| `val/qwk` | ordinal agreement | tracking grade-2 recall; discards balanced models |
| `val/macro_recall` | per-class balance | ignoring the referral boundary; 2.5× more missed referrals |
| `val/sensitivity_referable` | the clinical decision | untested here; likely to sacrifice specificity |

No single scalar captures the objective. The honest resolution is to report all
three and choose explicitly, which is what the ablation table now does.


---

## Result 5 — Resolution does not help, and grade-1 recall gets *worse*

The full sweep, all on the same fold-0 split, all QWK-selected:

| Resolution | MA size | QWK | **grade 1** | grade 2 | grade 3 | Referable sens |
|---:|---:|---:|---:|---:|---:|---:|
| 384 px | 0.9 px | 0.8845 | **0.622** | 0.845 | 0.282 | 0.906 |
| 512 px | 1.2 px | 0.8930 | **0.541** | 0.855 | 0.359 | 0.919 |
| 768 px | 1.8 px | 0.8986 | **0.419** | 0.945 | 0.282 | 0.926 |

```
768 vs 512:  QWK +0.0056  95% CI [-0.0179, +0.0287]  p = 0.603   n.s.
             exact grade (McNemar)  43 vs 32 discordant  p = 0.248  n.s.
             referable  (McNemar)   20 vs 21 discordant  p = 1.000  n.s.
```

### The central hypothesis is refuted

[`01_PROJECT_ANALYSIS.md` §2.2](01_PROJECT_ANALYSIS.md) argued that resolution is
"the single most important hyperparameter", because grade 1 is defined by
microaneurysms and MAs are destroyed by downscaling. The prediction was that
grade-1 recall would **rise** with resolution.

**Grade-1 recall falls monotonically instead: 0.622 → 0.541 → 0.419.**

Three independent runs, a clean monotonic trend, in the direction opposite to the
prediction. No QWK difference in the sweep reaches significance, so the aggregate
metric is flat; but the grade-1 trend is consistent and is the quantity the
hypothesis was specifically about.

### A coherent alternative explanation

QWK, grade-2 recall and referable sensitivity all rise with resolution while
grade 1 falls. That pattern fits a single mechanism: **more resolution gives the
model more evidence, which makes it more confident — and under cross-entropy with
49 % grade 0 and 27 % grade 2, more confidence means more commitment to the
majority classes.** Grade 1, defined by the subtlest finding and holding only
10 % of the data, absorbs the cost.

This is consistent with the earlier measurement that QWK tracks grade-2 recall
(r = +0.51, p < 0.0001, Result 3). Higher resolution appears to buy aggregate
agreement by sharpening the majority-class decision, not by revealing
microaneurysms.

### Caveats that keep this from being conclusive

- **Three points, one fold, no repeats.** The monotonic trend is suggestive, not
  established. Each point is a single run whose per-epoch grade-1 recall varied
  between 0.01 and 0.81.
- **QWK selection confounds it.** All three checkpoints were chosen on QWK, which
  Result 3 showed favours hedged models. The 768 run *reached* grade 1 = 0.66 at
  epoch 24 and 0.65 at epoch 11; the QWK-selected checkpoint has 0.42. The sweep
  therefore compares "the most grade-2-confident epoch of each run", which is
  precisely where the resolution effect would be masked.
- **1024 px was never tested.** At 2.4 px an MA is genuinely resolvable, and that
  is where the hypothesis makes its strongest claim. It needs ~9.6 GB and is
  Kaggle-only.

**What can be stated:** across 384–768 px, with QWK-based selection, higher
resolution does not improve grade-1 recall and monotonically reduces it. The
§2.2 claim is **refuted in the tested range** and remains untested at ≥1024 px.

---

## Result 8 — accumulation eliminates the sharp collapse, but not at the checkpoint that gets selected

The step-vs-epoch design (Result 6's follow-up) ran `--accum 4` against the exact
warmup-8 control, differing by that one flag. Both predictions it was built to
test — collapse at epoch 3 (epoch-locked) or epoch 16 (step-locked) — were
**both wrong**: across all 40 epochs, the collapse signature (grade-2 recall
≥0.99 with grade-1 ≤0.02 and grade-3 = 0.000) **never appeared once**. Every
accum-1 run tested (baseline, 384, 768, the warmup-8 control) hit it at epoch 3
without exception; this is the first run that does not.

```
epoch  3 (accum4): g1=0.230 g2=0.980 g3=0.051  qwk=0.8306   -- no collapse
epoch 16 (accum4): g1=0.568 g2=0.820 g3=0.231  qwk=0.8733   -- no collapse
```

### But the selected checkpoint tells a different story

Full evaluation (`scripts/evaluate.py`, bootstrap CIs) at each run's own
best-QWK epoch:

| Run | QWK | sens | spec | ECE | g1 | g2 | g3 | g4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (accum 1, warmup 3) | 0.8930 | 0.919 | 0.940 | 0.0646 | 0.541 | 0.855 | 0.359 | 0.458 |
| control (accum 1, warmup 8) | 0.8958 | 0.903 | 0.940 | 0.0605 | 0.608 | 0.835 | **0.436** | 0.525 |
| **accum 4 (warmup 8)** | 0.8942 | 0.903 | 0.945 | **0.0565** | 0.554 | **0.915** | 0.333 | 0.441 |

Paired against the baseline: QWK difference **+0.0012**, 95% CI [-0.0201, +0.0223],
**p = 0.910**. Paired against its own control (identical schedule, accum 1 vs 4):
QWK difference **-0.0016**, **p = 0.860**. Neither McNemar test (exact grade,
referable decision) reaches significance either.

**At the selected checkpoint, accum 4 has the highest grade-2 recall of any run
evaluated (0.915) and lower grade-3 recall than its own control (0.333 vs
0.436).** That is the opposite of what "eliminates the collapse" would suggest
if collapse were the thing determining final quality.

### The reconciliation: two different questions

"Does the sharp collapse happen?" and "is the selected checkpoint well-balanced?"
are not the same question, and this run answers them differently:

- **The transient collapse — answered by the trajectory.** Accumulation
  smooths the per-step gradient, and the total single-epoch wipeout genuinely
  does not occur. This is a real, reproducible difference from every accum-1 run.
- **The selected checkpoint — answered by `val/qwk` selection.** Result 3
  already showed QWK selection tracks grade-2 recall (r = +0.51) and will pick
  a hedged epoch over a balanced one when the hedged epoch scores marginally
  higher. That mechanism is untouched by accumulation: `ModelCheckpoint` still
  greedily keeps whatever epoch maximises QWK, and here that happens to be an
  epoch with 0.915 grade-2 recall.

Removing the sharp collapse did not help, because **the harm was never really
about the collapse epoch** — training recovers from it within a few epochs
regardless (see the baseline's epoch 4-8 trajectory). The harm is in which
epoch `ModelCheckpoint` keeps at the end, and that selection problem is
orthogonal to whether the loss curve is smooth or spiky along the way.

### What this changes

- The accumulation experiment is **complete and answered**: it does what the
  gradient-noise theory predicted (no more sharp collapse) without producing
  the downstream benefit that made the collapse seem like the thing to fix.
- **Every remaining lever in this ablation is downstream of checkpoint
  selection, not of training dynamics.** Resolution, loss function and
  accumulation have now all been tried; none moved QWK outside the same-seed
  noise floor (Result 7), and the one method that visibly changed per-class
  balance (macro-recall selection, Result 4) did so unsafely.
- The best-calibrated model of the whole phase is this one (ECE 0.0565,
  down from the baseline's 0.0646) — a secondary benefit worth carrying into
  Phase 5, even though QWK itself did not move.

## Result 9 — plain sensitivity selection is unsafe, for the identical reason as macro-recall

Before spending any GPU time on `--monitor val/sensitivity_referable` (the next
item on the "what to try next" list), a zero-cost check against the eight runs
already trained:

```
run                          max-sens epoch   collapse epoch(s)   same?
baseline_effb0_512                  3               [3]           True
sweep_384_ce                        2               [3]           False (off by one)
sweep_768_ce                        3               [3]           True
warmup8_accum1_lr1e-4               3               [3]           True
```

**Plain sensitivity, maximised alone, picks the exact grade-2 collapse epoch in
3 of 4 checked runs.** The mechanism is immediate once stated: at the collapse
epoch the model predicts grade ≥2 for nearly every image, so it catches every
true referable case (sensitivity → 1.000) at the cost of specificity
(0.798–0.811). Sensitivity alone cannot distinguish "confidently discriminates"
from "refers everyone" — the same failure class as macro-recall (Result 4), a
clinically-named metric that turns out to be blind to the one thing that
actually mattered.

### The fix: operationalise this project's own stated target directly

Rather than a symmetric combination (Youden's J = sens + spec − 1, logged
alongside for comparison but not used for selection), `--monitor` now offers
`val/sens_at_spec85`: **the best sensitivity achievable subject to specificity
≥ 0.85**, swept per epoch over the validation split. That 0.85 is not an
arbitrary combination weight — it is this project's own target from
[`01_PROJECT_ANALYSIS.md`](01_PROJECT_ANALYSIS.md). A degenerate always-referable
epoch cannot satisfy it (specificity ≈ 0 there), so it structurally cannot win
the way it did under plain sensitivity or, differently, under macro-recall.

This is the same category of fix as `corn_task_pos_weights` (Result 2): a naive
version was checked against real evidence before running it, found unsafe by
the same mechanism as an already-documented failure, and replaced with a
version that targets the actual requirement instead of a proxy for it.
`val/sensitivity_referable` itself remains logged for visibility; it is simply
no longer an offered `--monitor` choice, and the CLI help says why.

### The run: the fix avoided the trap it was built for, and then made a new one

One factor changed from the true Phase 1 baseline (`--monitor` only, everything
else — 512 px, batch 4, lr 1e-4, warmup 3, patience 8 — identical). Full
evaluation, bootstrap CIs, paired against baseline:

```
                    qwk      95% CI              sens    spec    ece     g3
baseline (accum1)  0.8930  [0.868, 0.916]       0.919   0.940  0.0646  0.359
sens_at_spec85     0.8557  [0.826, 0.884]       0.913   0.945  0.0434  0.051

QWK difference (B - A)  -0.0373   95% CI [-0.0622, -0.0124]   p = 0.003
-> SIGNIFICANT at alpha=0.05 -- the ONLY significant QWK result in Phase 3
```

**It did what it was designed to do**: the selected epoch (7) is nowhere near
the grade-2 collapse at epoch 3, confirming the floor constraint works exactly
as the mechanism predicts.

**It is also the single worst outcome of the entire phase.** Grade-3 recall
fell from 0.359 to **0.051** (2 of 39 cases) — an 86% relative collapse, worse
than every naive-sensitivity or naive-macro-recall failure mode this phase
uncovered, including the ones this metric was built specifically to avoid.

### Why: the floor was too easy to clear

```
sens_at_spec85 range across the run's 16 epochs:  [0.9866, 1.0000]   span 0.0134
QWK            range across the same 16 epochs:   [0.7871, 0.8930]   span 0.1059
```

An 8× compression. Once training clears the initial degenerate phase, *almost
every* epoch on this task can find some threshold hitting specificity ≥ 0.85
while sensitivity sits near the 0.99–1.00 ceiling — the referable/non-referable
ranking is comfortably good enough, epoch after epoch, that 0.85 stopped being
a meaningful bar. The metric was measuring "did training leave the collapse
basin" (yes, almost immediately) rather than "which of the many
non-collapsed epochs is best."

With the signal compressed to a 0.0134 band, `ModelCheckpoint(mode="max")`
is deciding on the 3rd–4th decimal place — effectively noise. Concretely:

```
epoch with this run's actual BEST qwk:  10   qwk=0.8930   sas=0.9899
epoch sens_at_spec85 actually SELECTED:  7   qwk=0.8557   sas=1.0000
```

**A strictly better epoch was available in the same run and was not chosen**,
because epoch 10 scored 0.0101 lower on an already-saturated metric than
epoch 7's ceiling value. This is not a case of the metric picking a
*different but defensible* tradeoff — epoch 10 dominates epoch 7 on QWK,
sensitivity, and grade-3 recall simultaneously; there is no criterion by
which epoch 7 is the better model.

### What this changes

- **0.85 was the wrong floor**, not because the floor idea is wrong, but
  because it was set at this project's *minimum acceptable* bar rather than
  near the *achievable ceiling* for a reasonably trained epoch on this task.
  A floor that saturates provides no more selection power than no floor at
  all — worse, in fact, since ties are broken by whatever noise the sweep's
  discretisation happens to produce. A tighter floor (0.92–0.95, informed by
  what healthy epochs actually achieve) would very likely restore
  discrimination; untested here.
- **This is the one result in Phase 3 large enough to matter even against the
  same-seed noise floor from Result 7** (span 0.075). At -0.0373 it is smaller
  than that raw span but was still detected by the paired test, which is more
  powerful than comparing independent same-seed runs because it controls for
  the shared validation images.
- **The general lesson generalises past this one metric.** Any selection
  criterion this project tries next should be checked for the same failure
  before trusting it: does it actually vary across epochs that differ
  substantially on QWK and per-class recall, or does it saturate? A metric
  that cannot tell epoch 7 from epoch 10 is not a safer choice than QWK just
  because it is harder to game in the degenerate case — it can still fail by
  being uninformative rather than misleading.

## Phase 3 conclusion

Six configurations, five paired comparisons, **zero significant improvements**.

| # | Configuration | QWK | vs baseline |
|---|---|---|---|
| 1 | Baseline 512 px CE | 0.8930 | — |
| 2 | 384 px | 0.8845 | p = 0.490 |
| 3 | CORN | 0.8951 | p = 0.851 |
| 4 | CORN + task balancing | 0.8950 | p = 0.734 |
| 5 | CORN + macro selection | 0.8790 | p = 0.749 |
| 6 | 768 px | 0.8986 | p = 0.603 |
| 7 | warmup 8 control (accum 1) | 0.8958 | p = 0.910 |
| 8 | warmup 8 + accum 4 | 0.8942 | p = 0.860 (vs control) |
| 9 | `--monitor val/sens_at_spec85` | 0.8557 | **p = 0.003 (significant, worse)** |

The baseline stands. What Phase 3 produced instead is six mechanisms:

1. **Sub-MA resolutions cannot test the MA hypothesis** — and above them, the
   hypothesis fails anyway (Results 1, 5).
2. **Unweighted CORN cannot work on skewed ordinal data** — its conditional
   subsets inherit the skew, and task j=1 is 80 % positive (Result 2).
3. **QWK selection discards balanced models** — it tracks grade-2 recall
   (r = +0.51) and cost 0.100 macro-recall for 0.0097 QWK (Result 3).
4. **Macro-recall selection is clinically unsafe** — boundary-blind, it produced
   2.5× more missed referrals (Result 4).
5. **A sharp CE-specific collapse, not caused by learning rate** — grade-2
   recall exactly 1.000 in all three original cross-entropy runs at epoch 3,
   *not* loss-invariant (no CORN run saturates: 0.835 / 0.275 / 0.550), and
   subsequently shown by the warmup-8 control (Result 6) to persist unmoved at
   44 % of peak LR and to *not* recur when full LR does arrive (epoch 8).
   Cause remains open; see Result 8 for where this leaves it.
6. **Eliminating the collapse does not fix the selected checkpoint** —
   `--accum 4` removes the sharp wipeout entirely across 40 epochs, but its
   QWK-selected checkpoint still ends up with the highest grade-2 recall of
   any run in this phase (0.915) and lower grade-3 recall than its own
   accum-1 control (Result 8). The problem was checkpoint selection
   (mechanism 3), not training-time instability.
7. **A safety fix can fail by being uninformative, not just by being gamed** —
   `val/sens_at_spec85` correctly refuses the degenerate always-referable
   epoch (mechanism 1's failure mode), but the 0.85 floor saturates within an
   0.0134 band across epochs whose QWK spans 0.1059. `ModelCheckpoint` then
   selects on noise: it kept an epoch 0.0373 QWK worse, on every metric that
   matters, than one available in the same run. This is Phase 3's only
   statistically significant QWK result, and it is negative (Result 9).

## Result 10 — 5-fold cross-validation: the baseline is stable, and 0.90 is a coin flip

Every number in Results 1–9, including the baseline itself, came from one fold
(`fold 0` of a `StratifiedGroupKFold(n_splits=5)`). `scripts/train.py --folds
0,1,2,3,4` runs the identical baseline configuration (512 px, batch 4, lr
1e-4, warmup 3, patience 8, CE) across all five folds in one process and
reports the aggregate — this had never been invoked with more than one fold
before.

```
fold 0   QWK 0.8967   (28 epochs)
fold 1   QWK 0.9090   (18 epochs)
fold 2   QWK 0.8955   (17 epochs)
fold 3   QWK 0.8756   (27 epochs)
fold 4   QWK 0.9056   (18 epochs)

mean 0.8965   std 0.0116
```

### This is good news for Results 1–9

The original single-fold baseline (0.8930) sits inside one standard deviation
of the 5-fold mean (0.8965 ± 0.0116). The fold this project happened to run
everything on was not a lucky or unlucky outlier — every Phase 3 comparison
that used it as the reference point was comparing against a representative
number, not a fluke. The re-run of fold 0 itself under this run (0.8967)
differs from the original baseline (0.8930) by 0.0037, consistent with the
same-seed non-determinism documented in Result 7.

### But QWK ≥ 0.90 (the Phase 3 exit criterion) is not a stable property

Two of five folds (1 and 4) cleared 0.90; three did not. A single held-out
split reporting ≥ 0.90 is not evidence the model reliably clears that bar —
it is a roughly 40% background rate at this sample size, before any deliberate
improvement. Any future claim of "QWK ≥ 0.90" on this dataset should be a
5-fold mean, not a single fold, or it risks re-litigating this exact question.

### Cross-fold per-class recall exposes instability Results 1–9 could not see

With only one fold, grade-1 and grade-3 recall each had one value; across five
they range widely: grade-1 (Mild) recall is 0.41 / 0.76 / 0.68 / 0.35 / 0.58,
and grade-3 (Severe) recall is 0.41 / 0.42 / 0.13 / 0.21 / 0.21. Grade-3 recall
in particular is consistently poor and never recovers across any fold — a
single-fold snapshot could not distinguish "this fold's grade-3 recall is
unlucky" from "this model cannot do grade-3," and the answer is the latter.

## RETFound / EyePACS pretraining — investigated, not run

The natural next lever (2,929 training images is the likeliest binding
constraint on all of Results 1–9) was investigated and is not runnable
autonomously from this machine right now, for three independent reasons:

1. **EyePACS** needs Kaggle-scale compute for the pretraining run itself; this
   project's own tech-stack doc already rules out pulling the ~90 GB dataset
   locally (`docs/03_TECH_STACK.md` §5).
2. **RETFound** ships only as a ViT-Large encoder (~300M parameters) — there
   is no smaller RETFound-branded checkpoint. Obtaining the weights requires
   registering a HuggingFace account and completing a gated-access request, which
   is a credentials/account action, not a training-design choice.
3. Even with weights in hand, ~300M parameters is a ~75× jump from the current
   4M-parameter EfficientNet-B0, on a machine that already hit OOM/system-freeze
   conditions at a small fraction of that size (docs/05_PROTOTYPE_SCOPE.md
   §6.2), and RETFound's fine-tuning is fixed at 224 px input, which conflicts
   with this project's own finding that microaneurysms need higher resolution
   to survive downscaling (Result 1, Result 5).

This matches the roadmap's own "if you have to cut scope" list
(`docs/04_ROADMAP.md`), which already names EyePACS pretraining as croppable
with ImageNet init as the accepted fallback. Recorded here as investigated
rather than silently skipped.

## Result 11 — Recalibrating the floor to 0.92 fixes the saturation; QWK returns to null

Result 9's own diagnosis named the fix: 0.85 was set at this project's minimum
acceptable bar rather than near the achievable ceiling, so raise it. Added
`--spec-floor` (it did not exist as a flag; `module.py` had 0.85 hardcoded at
the call site) and reran the identical configuration at 0.92, chosen from the
5-fold CV's own top-20-QWK epochs (13 of 20 clear 0.92 -- tight enough to
matter, not so tight the floor becomes nearly unreachable).

```
                    qwk      95% CI              sens    spec    ece     g3     g4
baseline (accum1)  0.8930  [0.868, 0.916]       0.919   0.940  0.0646  0.359  0.458
spec_floor=0.92    0.8802  [0.854, 0.905]       0.909   0.952  0.0553  0.282  0.627

QWK difference (B - A)  -0.0128   95% CI [-0.0355, +0.0098]   p = 0.272 -- NOT significant
McNemar, exact grade      38 vs 32   p = 0.550 -- not significant
McNemar, referable call   18 vs 16   p = 0.864 -- not significant
```

### The saturation is fixed

Result 9's 0.85 floor compressed 16 epochs spanning 0.106 QWK into a 0.013
`sens_at_spec85` band -- the metric could not tell a good epoch from a great
one. At 0.92, the same metric spans roughly 0.06 across this run's epochs
(0.913 at epoch 20 up to 0.973 at epochs 15 and 22, excluding the epoch-3
anomaly discussed below) -- close to a 5x wider discriminating range. The
mechanism Result 9 diagnosed is directly addressed: this floor is no longer
trivially cleared by nearly every reasonable epoch.

### But QWK did not improve -- it returned to the same null everything else got

The selected checkpoint is epoch 15 (`sens_at_spec85` 0.9732, tied with epoch
22 but selected first). It is not the run's best-QWK epoch: epoch 10 reached
QWK 0.8934 at `sens_at_spec85` 0.9698, fractionally below the peak, and so
was not selected. The gap this time (0.0132) is real but far smaller than
Result 9's (0.0373) -- consistent with a much less compressed metric doing
a more sensible job of picking among genuinely close epochs, rather than
breaking ties on noise.

The net effect: fixing the saturation stopped the metric from actively
selecting a significantly-worse epoch, but it did not make sensitivity-at-a-
floor selection better than QWK selection at this task. The result moved
from Result 9's **significant negative** (p = 0.003) to a **null result**
(p = 0.272) -- squarely alongside every other configuration in this table.

### Per-class detail: a real trade, not a uniform loss

Grade-3 recall fell again (0.359 -> 0.282), continuing the pattern that
grade-3 is this model's weakest class regardless of selection metric. But
grade-4 (PDR) recall rose substantially (0.458 -> 0.627), and ECE improved
(0.0646 -> 0.0553). Neither the QWK difference nor either McNemar test
reaches significance, so none of these per-class shifts should be read as
proven effects -- they are the same kind of noise-scale movement Result 7
already characterised, not a new finding.

### What this closes

This closes the "recalibrate the floor" branch opened by Result 9. Sensitivity-
at-a-specificity-floor selection, at either the minimum-bar floor (0.85) or a
ceiling-informed floor (0.92), does not beat `val/qwk` selection on this task.
The remaining untried levers are architectural or data-scale (1024 px, a
different backbone, more pretraining data), not further selection-metric
tuning -- three selection-metric designs (QWK, macro-recall, two sens-at-floor
variants) have now been tried, and none has produced a checkpoint
significantly better than plain QWK selection.

## Result 12 — 1024 px: QWK still not significant, but exact-grade accuracy is

The one resolution never tried, because it needs more memory than this project's local machine
has (docs/05_PROTOTYPE_SCOPE.md §6.2). Result 1's own size table says why it matters: 384 px and
512 px put a microaneurysm at 0.9–1.2 px, below what a CNN can resolve at all; 1024 px is the
first point where an MA (≈2.4 px) is actually resolvable. Trained on Kaggle (T4, after a P100
CUDA-kernel incompatibility forced a mid-run accelerator switch — see the commit history for that
detour), otherwise the identical baseline configuration (lr, warmup, patience, CE loss).

```
                    qwk      95% CI              sens    spec    ece
baseline (512px)   0.8930  [0.868, 0.916]       0.919   0.940  0.0646
1024px (Kaggle)    0.9122  [0.889, 0.935]       0.909   0.961  0.0421

QWK difference (B - A)  +0.0192   95% CI [-0.0040, +0.0428]   p = 0.104 -- NOT significant
McNemar, exact grade      60 vs 26   p = 0.0003  -- SIGNIFICANT
McNemar, referable call   28 vs 22   p = 0.480   -- not significant
```

### The highest QWK of the phase, and still not a statistically confirmed win

0.9122 clears the Phase 3 exit criterion (QWK ≥ 0.90) outright, and is the best QWK across all
eleven configurations tried, including the 5-fold CV mean from Result 10 (0.8965 ± 0.0116). But
the paired bootstrap test against the true baseline — the same test every other row in this table
was judged by — puts the improvement at p = 0.104. By this project's own α = 0.05 bar, applied
identically everywhere else in this document, **this is not a significant result.** Reporting it
as a confirmed win because the raw number looks good would be exactly the kind of unearned claim
this document exists to catch.

### What *is* significant: getting the exact grade right

McNemar's test on whether each model predicted the exact correct ICDR grade (not the ordinal-
weighted QWK, the raw hit/miss) is unambiguous: 60 images the 1024 px model got right that the
baseline missed, against 26 the other way, p = 0.0003. This is a real, independently-measured
signal that the ordinal-weighted QWK metric — built to be forgiving of near-miss errors — is
diluting. Per-class recall is consistent with a genuine, broad improvement rather than one lucky
class: Moderate +0.095, Severe +0.077, PDR +0.169, No DR and Mild essentially flat. The referable
binary decision (grade ≥ 2), the clinically operative one, does not reach significance (p = 0.480)
-- the gains are concentrated in getting the *exact* severity right among already-detected cases,
not in catching referable cases the baseline missed entirely.

### Two honest caveats before this becomes the headline number

1. **Two things changed at once, not one.** Every other resolution in this table (384/512/768)
   ran on the same local MPS backend as the baseline. This run had to move to Kaggle's CUDA/T4
   after the local-machine memory ceiling ruled it out entirely -- so "1024 px" and "different
   compute backend" are confounded here in a way they are not for any other row. Numerical
   differences between MPS and CUDA kernels are usually negligible for a converged model's
   evaluation-time predictions, but "usually negligible" is an assumption, not something measured
   in this project.
2. **Single fold, same caveat as every other row.** Result 10 already established that a single
   fold's QWK sits inside a 0.0116 standard deviation band across the 5-fold CV. 0.9122 is a
   good draw from that same distribution, not proof the *underlying* model is meaningfully better
   -- the exact-grade McNemar result is the stronger evidence here precisely because it does not
   depend on which single fold was used to select the naive 0.5 threshold.

### What this changes

1024 px is no longer the untested point of the microaneurysm hypothesis -- it is the first
resolution in this whole phase to produce an independently significant improvement on *any*
metric, and the closest QWK has come to significance across eleven tries. That is not the same as
confirming the hypothesis: QWK itself remains not significant, and the two caveats above mean this
is a genuinely promising lead, not a closed case. The natural next step, if this is worth pursuing
further, is a same-backend or multi-fold confirmation rather than treating one Kaggle run as final.

### What was tried, and what is left

The obvious next step after Result 5 — accumulation, on the theory that lower
gradient noise would fix the collapse and thereby the model — was run to
completion (Results 6 and 8). It confirmed the collapse is real and
schedule-independent, showed accumulation eliminates it, and showed that
eliminating it changes nothing that matters, because the actual failure was
always mechanism 3: `val/qwk` selection keeps whichever epoch is most
grade-2-confident, whether or not that epoch was reached via a spike or a
smooth curve. This closed the training-dynamics branch of the investigation.

The next step after that — fix the selection metric directly, since mechanism
3 was always the real target — was also run to completion (Result 9). It
correctly avoided the degenerate collapse epoch, then failed differently: the
0.85 floor saturates on this task, compressing 16 epochs of real QWK variation
(span 0.106) into a 0.013 band, and the checkpoint chosen from that noise was
significantly worse than the baseline (p = 0.003) and worse than an epoch
available in its own run. **This closes the "fix the metric with a fixed floor"
approach; a floor calibrated to the achievable ceiling rather than the minimum
acceptable bar remains untested and is now the natural next attempt.**

What remains, in order:

1. ~~`--monitor val/sens_at_spec85` with a recalibrated floor.~~ Done (Result
   11): 0.92 fixed the saturation Result 9 diagnosed, but QWK returned to
   null (p = 0.272), not an improvement. This branch is closed.
2. ~~EyePACS pretraining or RETFound init.~~ Investigated (see above) and not
   runnable autonomously from this machine: no local compute path for
   EyePACS, and RETFound's only available size is a ~300M-parameter ViT
   behind a gated HuggingFace request, fixed at 224 px input. ImageNet init
   (the current state) stands per the roadmap's own accepted fallback.
3. ~~1024 px on Kaggle.~~ Done (Result 12): highest QWK of the phase (0.9122),
   still not significant on QWK (p = 0.104), but the first independently
   significant improvement on any metric in eleven tries (exact-grade
   accuracy, p = 0.0003). Promising, not closed -- confounded with a
   backend change (MPS -> CUDA) no other row in this table has, and drawn
   from a single fold like every other row.
4. **5-fold CV, not a single fold, for any future comparison** (Result 10) —
   the baseline is stable (0.8965 ± 0.0116) but QWK ≥ 0.90 is only a ~40%
   background rate at this sample size; grade-3 recall is consistently poor
   across every fold and is a real model limitation, not fold noise.
5. **A same-backend or multi-fold confirmation of 1024 px** (Result 12) —
   the natural next step if the exact-grade-accuracy signal is worth
   pursuing further, now that it is the most promising lead in the phase.

Every selection-metric attempt so far has failed by a *different* mechanism
(QWK: tracks the wrong class; macro-recall: ignores the referral boundary;
sens-at-floor: saturates) rather than converging toward an answer. That
pattern is itself worth noticing before spending more runs on metric design.
Resolution, on the other hand, went from refuted outright (Result 5, sub-MA
sizes) to the phase's single most promising lead (Result 12, the first
MA-resolvable size) — the opposite trajectory, and a reminder that a refuted
hypothesis at the wrong operating point is not the same as a refuted
hypothesis.
