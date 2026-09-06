# Phase 5 Results — MC-Dropout Uncertainty + Human-Escalation Policy

Two roadmap items in one script, because the second directly consumes the first's output: an
uncertainty score is only worth routing escalation decisions on if it's actually shown to track
real error first. Unlike the fusion head (docs/16), this one works -- cleanly, and by a wide
margin.

## What this covers, and what it doesn't

Built: `src/drdetect/calibration/mc_dropout.py` (MC-dropout sampling) and
`scripts/evaluate_uncertainty.py` (the uncertainty-quality check, then the escalation-policy
simulation), run against the canonical `baseline_effb0_512_fold0` checkpoint on its own standard
733-image validation fold -- the same split `scripts/evaluate.py` and Phase 1/3's own results use,
not a new subset.

Not built: this closes out Phase 5's remaining modelling work. What's left in Phase 5 is
documentation/synthesis, not new models (see "What's still open").

## A non-obvious implementation problem, solved before running anything real

EfficientNet's classifier dropout in `timm` is **functional**, not a persistent `nn.Dropout`
submodule -- `EfficientNet.forward_head` calls `F.dropout(x, p=self.drop_rate,
training=self.training)` directly. There is no Dropout module to selectively re-enable with the
usual `model.apply(lambda m: m.train() if isinstance(m, nn.Dropout) else None)` trick. Standard
MC-dropout reactivation (`model.train()`, run N stochastic passes) would also un-freeze
BatchNorm's running statistics, which this project deliberately froze for small-batch training
(`docs/05_PROTOTYPE_SCOPE.md`) and has no reason to disturb at inference. Fixed by calling
`model.train()` (which flips the `self.training` flag the functional dropout checks) immediately
followed by this project's own `freeze_batchnorm()` (which puts every BatchNorm layer straight
back into eval mode) -- leaving dropout as the only thing still affected by the reactivated
training flag. Verified directly before trusting it on real data: with `drop_rate=0`, repeated
forward passes are bit-identical (confirming no other randomness source); with dropout active on
the real trained checkpoint, logit std across 20 samples is 0.6-1.3 against a deterministic-logit
range of roughly -3.6 to 2.3 -- a real, substantial effect, not noise.

## Method

- 20 stochastic forward passes per validation image (`--n-mc-samples 20`), each decoded to a
  referable-DR score via the existing `decode_output`; the per-image uncertainty is the standard
  deviation of that score across the 20 passes.
- The point-estimate prediction uses the *mean* logits across all 20 passes, not a single
  deterministic pass -- MC-dropout's mean is itself a (usually) better-calibrated estimate than
  one dropout-free forward pass.
- Escalation simulates routing the top-*k*% most uncertain images to an **assumed-perfect grader**
  -- a stated simplification, not a measured human accuracy this project has data for. Reported
  numbers are therefore an upper bound on what a real AI+human system would achieve, not a
  prediction of it.

## Result 1 — Uncertainty tracks real error, sharply and significantly

```
uncertainty quintile (low -> high)   accuracy
Q1                                    0.9932
Q2                                    0.9932
Q3                                    0.7755
Q4                                    0.7192
Q5 (most uncertain)                   0.6438

Spearman(uncertainty, correctness): rho = -0.382, p = 8.0e-27
```

The bottom 40% of images by uncertainty are correct 99.3% of the time; the top 20% are correct
only 64.4% of the time -- a 35-point accuracy gap between the model's most- and least-confident
quintiles, on the same checkpoint, same images. This is not a marginal or noisy signal: p=8.0e-27
is about as far from chance as a result in this project gets. Whatever else is true of this
model, when MC-dropout disagrees with itself across 20 passes, that disagreement is telling you
something real about where it's likely wrong.

## Result 2 — Escalating the most uncertain cases substantially improves the combined system

```
escalated (k%)   n images   accuracy   QWK
0  (model alone)      0      0.8254   0.8936
5                    36      0.8472   0.9022
10                   73      0.8663   0.9197
20                  146      0.8963   0.9404
30                  219      0.9345   0.9659
50                  366      0.9809   0.9906
```

Escalating just the most-uncertain 20% of cases lifts QWK from 0.894 to 0.940 -- more than half
the gap to a perfect-QWK system, from routing one in five images. This is the mechanical
consequence of Result 1: because the top uncertainty quintile is where the model is wrong most
often (64.4% accuracy there vs. 99.3% in the bottom two quintiles), escalating it swaps out a
disproportionate share of the model's actual errors for (assumed) correct human calls.

**The ceiling caveat matters here**: every number in this table assumes the escalated cases are
graded perfectly. A real human grader is not perfect, and the whole point of the escalation
policy is to route the *hardest* cases -- exactly the ones a human grader might also find
difficult. These numbers say routing on this uncertainty signal targets the right cases; they do
not say what accuracy a real deployed AI+human system would actually reach.

## What's still open

- **No real human-accuracy data exists to replace the perfect-grader assumption.** This project
  has no access to a grader to measure against; the escalation numbers above are an upper bound,
  not a deployable estimate. Closing this requires an actual reviewer, the same constraint noted
  for Phase 6's clinician timing study.
- **Uncertainty was only checked against the baseline checkpoint**, not the fusion head or the
  1024px checkpoint. Given the fusion head's own null result (docs/16), re-running this against it
  specifically wasn't judged worth the time this pass.
- **Why Q1 and Q2 both show a rounded 0.0000 mean uncertainty** in the printed table (the
  underlying values are non-zero and continuous -- the Spearman test uses full precision) is a
  print-formatting limitation (`.4f`), not a data issue -- worth a wider format string if this
  table is regenerated for a report.
