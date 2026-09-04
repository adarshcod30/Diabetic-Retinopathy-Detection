# Phase 5 Results — Calibration

Every report this project has produced until now has said "confidence (uncalibrated)" on every
single output, because there was nothing calibrated to say it about. This is the phase that closes
that disclaimer -- for the checkpoints it has actually been run against, and honestly, not for
every checkpoint by default.

## What this covers, and what it doesn't

Built: temperature scaling (Guo et al. 2017) end to end -- fitting
(`src/drdetect/calibration/temperature.py`, `scripts/calibrate.py`), a reliability diagram, and
actual wiring into `scripts/predict.py` / the Gradio demo / the PDF report, so a calibrated
checkpoint's confidence number is genuinely the post-scaling one, not a script output nobody reads.

Not built: the lesion feature extractor, the CNN+lesion fusion head, and MC-dropout/ensemble
uncertainty. All three depend on segmentation (Phase 4), which has raw data (IDRiD, DRIVE) but no
model yet. Quality gating (route reject → recapture) was already built in Phase 2. The
human-escalation policy (route low-confidence cases to a grader) is a downstream consumer of
calibrated confidence and is not implemented as a standalone policy yet.

## Method

- **No separate calibration split exists.** This project's data plan (Phase 0) reserved train/val
  per fold, not train/val/calibration -- there was never a third split to fit T on. Temperature is
  therefore fit on the same fold-0 validation set (733 images) every Phase 1/3 number came from,
  which carries the identical "optimistic, not the Phase 8 number" caveat `scripts/evaluate.py`
  already prints for the operating threshold. This is not a new methodological gap introduced
  here; it is the same one, applied consistently to a second thing fit on that split.
- **ECE measured two ways**, both reusing the one `expected_calibration_error` function that
  already existed (`src/drdetect/eval/metrics.py`, written in Phase 1 with a comment naming this
  exact phase as its purpose): top-1 accuracy-vs-confidence (the standard Guo et al. formulation)
  and referable-binary calibration (the number this project has tracked in every prior result).
- **Two checkpoints calibrated**: the canonical Phase 1/3 baseline (`baseline_effb0_512_fold0`,
  QWK 0.8930) and the Result-12 1024px model (`kaggle_1024px_ce_fold0`, QWK 0.9122) -- the two
  checkpoints most likely to actually be used going forward.

## Result 1 — Temperature scaling helps both checkpoints, but does not clear the target for either metric on the baseline

```
                          fitted T   top-1 ECE (before->after)   referable ECE (before->after)
baseline (512px)          1.573      0.1499 -> 0.1347            0.0646 -> 0.0567
1024px (Kaggle, Result 12) 1.165      0.0567 -> 0.0462            0.0421 -> 0.0414
```

Both fitted temperatures are > 1 -- both models are overconfident, the textbook direction Guo et
al. report for modern CNNs, not underconfident. Both ECE numbers improve after scaling. But the
roadmap's own exit criterion is ECE < 0.05, and the baseline does not reach it on either metric
even after calibration (0.1347, 0.0567) -- a single global scalar cannot fully fix what turns out
to be a **non-monotonic** miscalibration curve (see the reliability diagrams:
`models/checkpoints/baseline_effb0_512_fold0/reliability_diagram.png`): the baseline is
*underconfident* in its 0.4-0.5 confidence band and *overconfident* from 0.5 upward. T stretches
or shrinks every logit gap by the same multiplicative factor, so it can only ever partially
correct a curve that needs to move in different directions in different bins.

**The 1024px checkpoint tells a different story, and it already clears the target.** Its
uncalibrated ECE (0.0567 / 0.0421) starts well below the baseline's, its reliability curve is
close to monotonic (see its own `reliability_diagram.png`), and after scaling both numbers clear
0.05 (0.0462, 0.0414). This is worth flagging as a real, if uncontrolled, cross-reference to
Result 12: the same checkpoint that posted this phase's best QWK also turns out to be the
best-calibrated one out of the box. Nothing here tests *why* -- more training data effectively
seen per step (fewer, larger epochs before its own early stop), a different compute backend, and
resolution are all confounded in that one run, same as Result 12 already disclosed. It is a lead,
not a mechanism.

## Result 2 — A concrete case where scaling alone is not enough

Every claim above is an average over 733 images; here is one specific case that shows what "helps
but does not fully fix" looks like on a real prediction, not just in a bin-averaged number. Running
`scripts/predict.py` on `data/raw/aptos/train_images/000c1434d8d7.png` (true grade 2) through the
baseline checkpoint:

```
raw logits:        [-16.60,   4.58,  17.34,  -1.73,   2.98]
softmax (T=1):      essentially 100.0000% on grade 2 (softmax of a ~13-point logit gap)
softmax (T=1.573):  99.96% on grade 2
```

The correct grade, at high confidence, on a real image -- but even after applying the fitted
temperature, the reported confidence is still 99.96%, not meaningfully softer. The logit gap
between the top and second class (≈12.8) is simply too large for a single scalar divisor to close
to a well-calibrated range. This is precisely why the report and demo now print confidence at two
decimal places (`99.96%`), not one (`100.0%`) -- a genuine 99.96% rounding up to a literal "100.0%"
would silently overclaim certainty the number itself does not support, on top of whatever the
calibration itself does or doesn't fix. Caught while verifying this phase's own output, fixed
across all three places confidence is displayed (report PDF, demo, `predict.py`'s console output).

## What this changes

- `scripts/predict.py` and the demo now report **genuinely calibrated** confidence for any
  checkpoint `scripts/calibrate.py` has been run against (detected via a `temperature.json`
  sidecar next to the checkpoint -- `drdetect.calibration.temperature.load_temperature`), and
  fall back to the previous "uncalibrated" label for every checkpoint that hasn't. Both states are
  labelled explicitly; neither is silently presented as the other.
- The baseline's confidence, even calibrated, should not be read as a validated probability in the
  way the roadmap's ECE < 0.05 target implies -- 0.1347 top-1 ECE is a real, disclosed gap, not a
  rounding error.
- The 1024px checkpoint's calibration is good enough to trust more than the baseline's, which is
  one more small piece of evidence (alongside Result 12's exact-grade-accuracy finding) that it is
  the more promising checkpoint going forward, not just the one with the better headline QWK.

## What's still open

1. **A real calibration split.** Every number here shares the val set's "optimistic" caveat with
   every other Phase 1/3 result. The locked Phase 8 test set is the honest number, same as it is
   for every other metric in this project.
2. **Why does the miscalibration curve differ so much between checkpoints?** Non-monotonic for the
   baseline, close to monotonic for 1024px. Worth understanding if calibration quality turns out
   to matter operationally (e.g. for the human-escalation policy below), not yet investigated.
3. **Uncertainty beyond a point confidence** (MC-dropout or a small ensemble) -- unbuilt, per the
   roadmap.
4. **The human-escalation policy** ("route bottom-k% confidence to a grader") -- now has a
   genuinely calibrated confidence number to route on for at least two checkpoints, but the policy
   itself, and measuring the AI+human system it implies, is not built.
