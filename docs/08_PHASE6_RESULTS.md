# Phase 6 Results — Rigorous Explainability

The roadmap calls this phase "the differentiator" — most DR papers show a Grad-CAM overlay and
stop there. This phase asks the harder question first: **is the overlay actually explaining the
model, or does it just look plausible?** Adebayo et al. (2018) showed that several
widely-trusted saliency methods produce nearly the same map whether the underlying model is
trained or replaced with noise. Nobody knew whether that applied to this project's methods until
it was checked here.

## What this covers, and what it doesn't

Built: four CAM methods behind one interface (`src/drdetect/explain/cam_variants.py`) and both
Adebayo sanity checks (`src/drdetect/explain/sanity_checks.py`) — cascading model-parameter
randomisation, and a data-randomisation comparison against a model trained on deliberately
shuffled labels.

Not built yet, and not silently assumed: quantified localisation against IDRiD's lesion masks
(pointing-game accuracy, CAM–lesion IoU) and the 30-second human-review timing study. IDRiD
landed too recently in this session to have a lesion-mask dataset loader built and verified
against it, and the timing study needs an actual human reviewer, which this session cannot
supply. Both remain open roadmap items.

## Method

- **Baseline model**: `models/checkpoints/baseline_effb0_512_fold0/best.ckpt` (Phase 1, QWK 0.8930).
- **Shuffled-label control**: identical architecture and hyperparameters, trained on the same
  2,929 training images with a fixed random permutation of the label column
  (`data/manifests/aptos_512_shuffled_labels.csv`, seed 42), forced through all 40 epochs
  (`--patience 40`, no early stopping — a meaningful early-stop signal doesn't exist when the
  validation labels are equally scrambled). Converged to val/qwk 0.0339 — chance level, as it
  should for a model with nothing real to learn.
- **5 validation images**, one per ICDR grade (0–4), each explained via its own model's own
  predicted class — not the true label, since a CAM must explain what the model said, not what
  it should have said.
- **Similarity metric**: both Spearman rank correlation and SSIM between two heatmaps, matching
  the two metrics Adebayo et al. used. They can disagree, and where they do here, that
  disagreement is itself informative, not a bug to resolve away.
- **Score-CAM excluded from both experiments.** Measured cost: ~15 minutes per single CAM
  computation on this machine (one forward pass per target-layer channel, 1280 of them). The
  model-randomisation cascade alone needs 11 computations per image; the full experiment across
  5 images would cost several hours for Score-CAM alone. It was verified separately, once,
  manually, to produce a valid non-degenerate heatmap post-fix (see the `8de86c4` commit) — that
  is confirmation the method *runs correctly*, not a sanity-check verdict, and the two should not
  be conflated.

## Result 1 — Eigen-CAM fails the sharpest form of the model-randomisation test

Cascading randomisation replaces the model with noise one layer group at a time, output-first,
and measures how much each method's heatmap has moved from its own original, on the same image
and same target class.

```
mean Spearman rank correlation to each method's own un-randomised CAM, averaged over 5 images

                          gradcam   gradcam++   eigencam
after randomising:
  classifier               +0.364      +0.950     +1.000   <- classifier is the FIRST thing randomised
  conv_head+bn2             +0.078      +0.728     +0.646
  blocks[6]                 +0.243      +0.580     +0.587
  blocks[5]                 +0.063      +0.108     +0.320
  blocks[4]                 +0.012      +0.022     +0.197
  ...
  conv_stem+bn1 (fully random)  +0.011   +0.027     +0.127
```

**Eigen-CAM's first row is exactly 1.000 — perfect correlation, zero change** after the
classifier is replaced with random weights. This is not noise or a small sample-size artefact;
it is architectural. Eigen-CAM computes its heatmap from the principal component of the *target
layer's own activations* (a form of PCA on `bn2`'s feature maps) and never reads the classifier
or any gradient through it. Randomising a layer that sits *downstream* of the hook point cannot
change a method that never looks downstream. Plain Grad-CAM, by contrast, drops from a perfect
baseline correlation to +0.364 after the same single step, and to near-zero (+0.078) one step
later, at the actual hooked layer — it is doing what a sanity check wants: responding to the
model's learned parameters, not to the input alone.

Grad-CAM++ sits in between: highly resistant at the classifier step (+0.950) — it does use
gradients through the classifier, but its second/third-order weighting formula can still be
dominated by activation structure that hasn't changed yet this early in the cascade — but it
converges to near-zero (+0.027) by full randomisation, same destination as plain Grad-CAM, just
a slower path there.

**Verdict: Grad-CAM passes cleanly. Grad-CAM++ passes, but only once enough of the network is
randomised. Eigen-CAM fails the single most diagnostic step in this test outright.**

## Result 2 — The data-randomisation check corroborates it, with a metric split worth keeping

Comparing each method's CAM on the real model against its CAM on the shuffled-label model
(different images? no — the *same* image, two models that learned genuinely different things
from it):

```
                gradcam    gradcam++   eigencam
mean spearman    +0.003     -0.033      +0.008     (all ~0 -- "pass" by rank correlation alone)
mean SSIM         0.333      0.633       0.684     (gradcam clearly lower -- "pass" by structure;
                                                     gradcam++/eigencam noticeably more similar
                                                     across two differently-trained models)
```

Rank correlation alone would call this a clean pass for all three methods — none of them show a
systematic rank relationship between the real and shuffled-label explanations. SSIM disagrees for
two of the three: Grad-CAM's heatmaps are structurally quite different between the two models
(mean SSIM 0.333), while Grad-CAM++ and especially Eigen-CAM stay noticeably more similar (0.633,
0.684) despite explaining models that learned nothing in common. This is the same ordering Result
1 found from a completely different experiment (a different manipulation, a different metric) —
that convergence is what makes both results trustworthy rather than a one-off artefact of either
test.

## What this changes

| Method | Model-randomisation | Data-randomisation (SSIM) | Verdict |
|---|---|---|---|
| Grad-CAM | Passes cleanly, diverges immediately | Passes (low similarity, 0.333) | **Most trustworthy of the three tested** |
| Grad-CAM++ | Passes eventually, slow to diverge | Borderline (0.633) | Usable, less discriminative than plain Grad-CAM |
| Eigen-CAM | **Fails** the classifier-randomisation step outright | Borderline-to-fails (0.684) | Structurally insensitive to what the classifier learned |
| Score-CAM | Not tested (cost) | Not tested (cost) | Confirmed to run correctly post-fix; sanity-check status unknown |

The practical consequence for this project: **Grad-CAM, already the method used in the Phase 2
PDF report, is the right default** — this phase gives it a positive reason to be the default,
not just historical inertia. Eigen-CAM's speed (0.8–0.9s vs Grad-CAM's 1.2s) is not a good enough
reason to prefer it once its structural insensitivity to the classifier is measured, not assumed.

## What's still open

1. **CAM–lesion IoU and pointing-game accuracy against IDRiD masks** — IDRiD is downloaded and
   verified (81 segmentation images with pixel-level lesion masks), but no dataset loader or IoU
   computation exists yet for it. Natural next step now that the data is in hand.
2. **The 30-second review-time measurement** — needs an actual human reviewer; recruiting one
   (ideally an MBBS student, per the roadmap) is outside what this session can do.
3. **Score-CAM's actual sanity-check behaviour** — known to run correctly, not known to pass or
   fail either check. If it matters later, the honest way to get an answer is a single image on
   whatever hardware makes ~15 minutes acceptable, not forcing the full 5-image, 11-step cascade
   through on CPU.
4. Sample size here is 5 images, one per grade — enough to find a first-order, architecturally-
   grounded effect (which is what Eigen-CAM's failure is), not enough to report a confidence
   interval on it. A larger sample is worth doing before this becomes a paper-shaped claim rather
   than an engineering decision about which default to ship.
