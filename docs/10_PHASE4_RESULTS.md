# Phase 4 Results — Segmentation (Hard Exudates)

The roadmap's own plan for Phase 4 is to do lesions "hard exudates first (easiest, highest
contrast) — establishes the harness" before touching vessels, OD/fovea, or the other lesion types.
This is that harness, run once, end to end, on real IDRiD data — and a first honest number for it.

## What this covers, and what it doesn't

Built: the full training and evaluation path for IDRiD lesion segmentation --
`src/drdetect/segmentation/{dataset,model,module,metrics}.py` and
`scripts/{train_segmentation,evaluate_segmentation}.py`. On-the-fly lesion-biased patch sampling
from full-resolution (2848x4288) images, a DeepLabV3+/resnet34 model, a combined BCE+Dice loss
with an empirically-estimated `pos_weight`, and tiled sliding-window inference for scoring
full-resolution test images a segmentation decoder could never hold whole. Trained and evaluated
hard exudates on IDRiD's own official train/test split.

Not built, per the roadmap's own Phase 4 item list: vessels (DRIVE/U-Net), OD & fovea heatmap
regression, quadrant mapping, haemorrhages, soft exudates, and microaneurysms -- which the roadmap
itself says needs a different approach entirely ("morphological top-hat + matched filter candidate
generation -> small CNN classifier. Do *not* expect plain segmentation to work here"), not this
harness. **5-fold CV**, which the roadmap explicitly suggests given how few images IDRiD's
segmentation split has (81 total), was also not run this pass -- see "What's still open" below.

## Method

- **Data**: IDRiD's official split. The 54 training images were further split 43/11
  (train/internal-val, seeded, single split, no k-fold) for model selection. The 27 official test
  images were held out entirely and touched exactly once, at final evaluation.
- **Patch sampling**: 512x512 patches, `lesion_patch_prob=0.8` biases sampling toward
  lesion-containing crops. Hard exudates are a measured <0.1% of a full image's pixels
  (`IDRiD_41_EX.tif`); an unbiased random crop would show the model an all-background patch on
  nearly every draw.
- **Loss**: `BCEWithLogitsLoss(pos_weight=12.65)` + soft Dice, summed. The pos_weight was measured
  empirically from 200 actually-sampled training patches, not derived from the whole-image
  imbalance ratio -- see Result 1.
- **Model selection**: best epoch by `val/auprc` (pooled pixel AUPRC over the internal val split),
  early-stopped at patience 8 epochs.
- **Final scoring**: non-overlapping, reflect-padded tiled inference over each of the 27 official
  test images at full 2848x4288 resolution, metrics pooled across every test pixel. Pooled, not
  averaged per image, because per-image AUPRC is undefined for a test image with zero positive
  pixels and unstable for one with only a handful -- several IDRiD test images have very few
  (`IDRiD_80`: 6,964 px, 0.057%).

## Result 1 — A first honest hard-exudate baseline: test AUPRC 0.830, Dice 0.703

```
                          internal val (11 img)   official test (27 img, held out)
pixel AUPRC                     0.8947                       0.8301
Dice @ 0.5                         --                        0.7034
```

Training ran 22 epochs (~43 minutes wall-clock on Apple Silicon MPS) before early-stopping 8
epochs past its epoch-12 peak. This is Phase 4's own first number, not a comparison against a
prior segmentation result -- none existed before this.

The val -> test drop (0.895 -> 0.830) is in the expected direction and a plausible size: val was
the model-selection target, so some optimism there is normal, and the two sets differ in lesion
burden. Positive-pixel fraction across the 27 test images ranges from 0.057% (`IDRiD_80`) to
7.510% (`IDRiD_66`) -- a >130x spread -- and pooled scoring is sensitive to which end of that range
a test set happens to contain. Pooled over the full test set, 3,577,920 of 329,730,048 pixels
(1.085%) are positive; that is a whole-test-set average, not in tension with the single-image
0.068-0.07% figure quoted in this codebase's docstrings, which was always a measurement of one
named training image, not a dataset-wide claim.

## Result 2 — The realistic pos_weight is ~12:1, not the ~1,400:1 the raw imbalance implies

Hard exudates are approximately 0.07% of a full IDRiD image's pixels -- a naive `pos_weight` set
from that ratio would be on the order of 1,400:1 (roughly `(1 - 0.0007) / 0.0007`). But the model
never trains on full, unbiased images: `lesion_patch_prob=0.8` already routes most training crops
toward lesion-containing regions, so the pixel composition a sampled *patch* actually contains is
far more balanced than the whole image is. Measuring directly, `estimate_pos_weight()` sampled 200
real training patches and found a **12.65:1** neg:pos ratio -- two orders of magnitude below the
whole-image figure. Using the whole-image ratio here would have overshot badly, pushing the loss
to over-predict the positive class relative to what the sampler was actually feeding it. This is
the same category of bug Phase 3 documented for CORN's unweighted conditional subsets
(`docs/07_PHASE3_RESULTS.md`, Result 2) -- an imbalance-weighting term computed against the wrong
distribution -- caught here before training rather than after.

## What's still open

- **5-fold CV on the 54 training images**, as the roadmap itself specifies for this exact task
  ("only 81 masked images -> 5-fold CV"). This run used a single 43/11 split to get a first number
  fastest; with only 11 images in that split, the 0.8301 test AUPRC has no confidence interval
  attached yet, and a single unlucky (or lucky) split could move it more than a real methodology
  change would. Follow the same pattern as `docs/07_PHASE3_RESULTS.md` Result 10 once this harness
  needs to be trusted for a specific number, not just a first read.
- **Vessels, OD/fovea, quadrant mapping, haemorrhages, soft exudates, microaneurysms** -- all
  explicitly out of scope for this pass, per the roadmap's own item list (see above).
- **Phase 6's CAM-vs-lesion-mask IoU and pointing-game accuracy** (`docs/04_ROADMAP.md` Phase 6)
  was blocked on "segmentation, which has raw data but no model yet." A hard-exudate model now
  exists, so this is technically unblocked for that one lesion type -- but the IoU/pointing-game
  comparison protocol itself (aligning a grading model's Grad-CAM heatmap against this
  segmentation model's masks) is not built. Noted as a lead, not attempted here.
- **Phase 5's lesion feature extractor** (`docs/04_ROADMAP.md` Phase 5: "MA count, HE/EX/SE area,
  per-quadrant distribution, distance-to-fovea") remains substantially blocked: hard-exudate area
  is now technically extractable from this model's output, but MA count needs the
  microaneurysm-specific detector and distance-to-fovea needs OD/fovea regression, neither of
  which exist yet. One input of several, not the fusion head itself.
