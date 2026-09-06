# Phase 6 Results — CAM Localisation vs IDRiD Masks (Pointing Game + IoU)

The roadmap names this "the most novel artefact in the project" and its own exit criterion: a
table of saliency methods x (sanity-check pass/fail, pointing-game accuracy, IoU). docs/08 already
built the sanity-check column; this is the pointing-game and IoU columns, against real IDRiD pixel
masks rather than an assumed-plausible heatmap.

## What this covers, and what it doesn't

Built: `src/drdetect/explain/localization_metrics.py` (pointing-game hit, IoU, CAM upsampling) and
`scripts/evaluate_cam_localization.py`, run against the canonical `baseline_effb0_512_fold0`
checkpoint on all 81 IDRiD segmentation-subset images (the only IDRiD images with real per-lesion
pixel masks -- "B. Disease Grading" and "C. Localization" share a different, disjoint 516-image
set with the same filenames but no lesion masks).

**Not a Phase 8 locked-test-set evaluation.** This doesn't touch model weights, thresholds, or any
training decision -- it's a post-hoc interpretability measurement on an already-frozen checkpoint,
which is exactly what the roadmap's own Phase 6 section specifies IDRiD's masks for.

Not built: lesion-overlay rendering, the ICDR evidence table, and the final PDF report redesign --
next in the roadmap's own order. The 30-second clinician timing study needs a human reviewer this
project cannot supply.

## Method

- 3 CAM methods: Grad-CAM, Grad-CAM++, Eigen-CAM. **Score-CAM deliberately excluded** -- docs/08
  measured ~15 minutes per single Score-CAM computation on this project's hardware; across 81
  images that's many hours for one method, the same cost that excluded it from the sanity-check
  cascade.
- For each image: the checkpoint's own predicted grade (not the true label -- a report explains
  what the model said, matching `drdetect.explain.gradcam.generate_cam`'s existing convention),
  a CAM heatmap for that prediction at the model's 512x512 input resolution, upsampled to the
  image's full 2848x4288 resolution (a CAM is already a smooth interpolated heatmap, so upsampling
  it preserves the mask's own pixel precision rather than downsampling the mask and losing the
  smallest lesions -- already known from Phase 3 to vanish below ~1024px).
- Two metrics per (method, lesion type) pair, over every image that actually has a non-empty mask
  of that type: **pointing game** (does the CAM's single point of maximum attention fall on a
  positive mask pixel -- threshold-free, Zhang et al. 2018) and **IoU** (CAM thresholded at 0.5
  against the binary mask).
- **Chance-level baseline computed directly from the same 81 images**: a uniformly random pixel
  hits with probability equal to the lesion's own mean area fraction. Without this, a pointing-game
  score like "3.75%" reads as a near-total failure; with it (haemorrhages cover a mean 1.03% of
  the image), it's actually 3.7x better than a random guess.

## Result 1 — Localisation accuracy scales with lesion size, and it's not close

```
lesion type       chance rate   Grad-CAM         Grad-CAM++       Eigen-CAM
microaneurysms      0.104%      0.0%  (0.0x)     0.0%  (0.0x)     0.0%  (0.0x)
haemorrhages        1.025%      3.75% (3.7x)     3.75% (3.7x)     3.75% (3.7x)
hard exudates       0.900%     16.05% (17.8x)   14.81% (16.5x)   14.81% (16.5x)
soft exudates       0.378%     20.00% (53.0x)   20.00% (53.0x)   20.00% (53.0x)
optic disc          1.781%     25.93% (14.6x)   28.40% (15.9x)   32.10% (18.0x)
```

Two findings, both mechanistically explicable rather than surprising once stated:

1. **The model's attention is never once at a microaneurysm's location, across all 81 images and
   all 3 methods** -- not "rarely," never. This lines up with the rest of this project's own
   findings: microaneurysms are ~1-3px at the working resolutions grading actually trains at
   (docs/07's own resolution-hypothesis work), and a global-average-pooled classifier has no
   mechanism to preferentially attend to a feature that small relative to the whole image, however
   much it might matter for the true grade.
2. **Every other lesion type scores far above chance** -- 3.7x for haemorrhages up to 53x for soft
   exudates, and the ranking (soft exudates > hard exudates > optic disc > haemorrhages, by
   chance-multiple) doesn't simply track lesion size or area fraction in an obvious way. Read
   plainly: when the model's decision genuinely depends on visible evidence, Grad-CAM's peak
   attention lands there far more often than chance would predict -- the mechanism this whole
   explainability effort is trying to verify actually holds, for everything except the one lesion
   type already known to be sub-resolution.

## Result 2 — IoU is uniformly low, and that's the metric being too strict, not the CAM being wrong

```
lesion type       Grad-CAM IoU   Grad-CAM++ IoU   Eigen-CAM IoU
microaneurysms       0.0010          0.0008          0.0008
haemorrhages         0.0111          0.0114          0.0114
hard exudates        0.0569          0.0557          0.0568
soft exudates        0.0844          0.0911          0.0881
optic disc           0.1635          0.2033          0.2160
```

Even the best case (optic disc, the largest and most compact structure checked) tops out at
IoU 0.22. This is expected, not a failure: Grad-CAM produces a smooth, diffuse region of elevated
attention by construction, not a pixel-precise segmentation boundary -- it's the wrong tool for a
region-overlap metric, which is exactly why the pointing game (a single point, no boundary
required) is the field's standard headline localisation metric and IoU is reported here as
supporting context, not the primary number.

## Result 3 — A method that fails the sanity check can still "look" locally accurate

Eigen-CAM **failed** the model-randomisation sanity check outright (docs/08): its heatmap is
provably unchanged by randomising the classifier's weights, since Eigen-CAM never reads the
classifier at all (it's the principal component of the target layer's activations, independent of
any class score or gradient). Despite that, Eigen-CAM scores the *highest* optic-disc
pointing-game accuracy of the three methods here (18.0x chance vs. Grad-CAM's 14.6x).

This is not a contradiction -- it's the reason the sanity check exists and needs to run *before*
trusting a localisation number, not after. The optic disc is a large, generically bright, centrally
salient structure in almost any reasonably-exposed fundus photo; a method that responds to generic
image saliency rather than what the trained classifier actually learned can still land on it
often, by construction, with zero dependence on the model being any good (or even trained at all).
A naive evaluation that only checked pointing-game/IoU numbers, without docs/08's sanity checks,
would have rated Eigen-CAM as the *best* localiser of the three -- exactly backwards from what its
own randomisation test already showed. Running both checks, not just the more intuitive one, is
what this project can genuinely claim as a real methodological point most DR-explainability work
skips.

## What's still open

- **Lesion-overlay rendering, the ICDR evidence table, and the final PDF report** remain unbuilt --
  next in the roadmap's own order, and more presentation/engineering work than new findings.
- **Score-CAM's own localisation numbers** were not computed here, for the same cost reason its
  sanity-check status is still unknown (docs/08). If ever revisited on faster hardware, it would
  complete this table.
- **Why the chance-multiple ranking doesn't track area fraction monotonically** (soft exudates,
  the *rarest* lesion checked at 0.378% area, scores the *highest* chance-multiple at 53x, ahead
  of hard exudates at 0.900% area) was not investigated further -- a plausible reading is that
  soft exudates, being fewer and larger per instance than hard exudates' typically more numerous,
  smaller clusters, are simply easier for a coarse attention map to land on once at all, but this
  is speculation, not a measured mechanism.
