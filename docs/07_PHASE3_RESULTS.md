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
| 3 | 768 px, CE | *pending* | | | | |
| 4 | 512 px, CORN ordinal | *running* | | | | |

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
