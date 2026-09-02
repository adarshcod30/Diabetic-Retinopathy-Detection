# Phase 3 Results — Grading Depth

Running log of the ablation. Every row is one factor changed from the Phase 1
baseline, evaluated on the same fold-0 validation split (733 images, 298
referable) with **paired** significance tests.

> **Why paired tests.** Two models scored on the same images make correlated
> errors. Comparing their marginal confidence intervals discards that pairing and
> is badly underpowered — intervals can overlap while the paired difference is
> highly significant. `scripts/compare.py` uses McNemar on discordant pairs and a
> paired bootstrap for QWK.

---

## Ablation so far

| # | Configuration | QWK | Sens | Spec | ECE | vs baseline |
|---|---|---|---|---|---|---|
| 1 | **Baseline** — EffNet-B0, 512 px, CE | **0.8930** | 0.919 | 0.940 | 0.0646 | — |
| 2 | 384 px, CE | 0.8845 | 0.906 | 0.956 | 0.0481 | p = 0.490, **n.s.** |
| 3 | 512 px, **CORN** ordinal | 0.8951 | 0.913 | 0.940 | 0.0550 | p = 0.851, **n.s.** |
| 4 | 512 px, **CORN + task balancing** | 0.8950 | 0.911 | 0.940 | 0.0391 | p = 0.734, **n.s.** |
| 5 | 512 px, CORN + macro-recall selection | 0.8790 | 0.903 | 0.959 | 0.0620 | p = 0.749, **n.s.** |
| 6 | 768 px, CE | *running* | | | | |

**Three experiments, three nulls.** Nothing has yet beaten the baseline at
α = 0.05. That is the honest state of the ablation. What the runs *have*
produced is three mechanistic findings that are more useful than a marginal
QWK bump would have been.

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

## Cross-cutting observation: the collapse may be optimisation, not loss

Every run so far — CE, CORN, balanced CORN — shows the same shape: rare-class
recall looks balanced at epochs 1–2, then collapses into grade 2 exactly when LR
warmup ends and the rate reaches full value. Four runs, three different losses,
same inflection point.

That consistency suggests the hedging is not primarily a property of the loss
function but of the optimisation: at full LR with 4-sample gradients, the
majority-class basin dominates however the loss is parameterised. If so, the
untested levers are a gentler schedule, a longer warmup, or gradient accumulation
for a larger effective batch — none of which change the loss at all.

This is a hypothesis from four observations, not a finding.


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
