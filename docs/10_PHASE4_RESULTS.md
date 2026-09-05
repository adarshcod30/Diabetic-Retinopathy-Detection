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
harness.

**Update:** 5-fold CV (Result 3) and a fix for Dice's threshold (Result 4) were added in a second
pass, after Result 1 first shipped with a single ad hoc split. See those results for what changed
and why; Result 1 and Result 2 are left as originally written below.

## Method

- **Data**: IDRiD's official split. The 54 training images are split into 5 folds
  (`sklearn.model_selection.KFold(shuffle=True, random_state=42)`) for cross-validated model
  selection -- Result 1 below used one ad hoc 43/11 split instead; Result 3 replaced that with the
  proper 5-fold version. The 27 official test images are held out entirely in every fold and
  touched only at final evaluation.
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

## Result 1 — A first honest hard-exudate baseline: test AUPRC 0.830, Dice 0.703 (superseded by Result 3)

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

This finding held up under the 5-fold CV added later (Result 3): the five folds' own empirically-
measured pos_weights (fold 0-4: 15.94, 16.93, 18.55, 15.85, 15.20) shift with exactly which images
land in each fold's training set, but every one stays in the same ~15-19 range -- nowhere near the
~1,400:1 whole-image figure this result already ruled out.

## Result 3 — 5-fold CV: test AUPRC is stable at 0.850 ± 0.029; Result 1's 0.830 was representative, not lucky

```
                internal val AUPRC (own fold)   test AUPRC (fixed, 27 img, held out)
fold 0                    0.8893                            0.8584
fold 1                    0.9138                            0.8543
fold 2                    0.9017                            0.7941
fold 3                    0.8972                            0.8796
fold 4                    0.8929                            0.8633
mean +/- std           0.8990 +/- 0.0085                  0.8500 +/- 0.0292
```

Each fold trains on a different 43-44/10-11 split of the 54 IDRiD training images and is scored on
the identical fixed 27 official test images, so the test-AUPRC spread above is driven entirely by
which images happened to land in a fold's training set, not by any change in test data. Result 1's
original ad hoc single split (test AUPRC 0.8301) sits inside this range, close to the mean -- it
was not a specially lucky or unlucky draw, closing the open question that result left.

Internal val AUPRC is markedly more stable (std 0.0085) than test AUPRC (std 0.0292). That gap
itself is expected, not a red flag: each fold's val set is a held-out slice of the same 54-image
pool the model trained near, while the 27 test images are a fixed, disjoint set the model never
adapts to at all across any fold.

Fold 2 is the outlier worth naming: its val AUPRC (0.9017) is unremarkable next to the other four,
but its test AUPRC (0.7941) is the sample minimum, and it is also the fold that early-stopped
fastest by far (11 epochs vs. 22-26 for the other four, best epoch reached at epoch 1 and never
bettered) -- consistent with a less-mature checkpoint that happened to still clear a decent val
score, rather than a genuinely different model.

## Result 4 — Dice's fixed 0.5 threshold hid a 5x-larger instability than the metric itself has

```
                     fold 0   fold 1   fold 2   fold 3   fold 4   mean +/- std
Dice @ 0.5 (fixed)   0.7440   0.7465   0.5540   0.7762   0.7339   0.7109 +/- 0.0797
Dice @ tuned         0.7864   0.7773   0.7523   0.7917   0.7917   0.7799 +/- 0.0148
tuned threshold       0.900    0.925    0.925    0.625    0.925        --
```

Same 5 checkpoints, same test pixels -- the only thing that changes between the two rows is which
probability threshold turns a predicted probability into a positive/negative call. Dice@0.5 has a
std of 0.0797 across folds, more than 5x the spread AUPRC (threshold-free, Result 3) shows on the
identical checkpoints (std 0.0292). Tuning the threshold per fold -- maximising Dice on that fold's
own held-out val split, freezing it, and only then applying it to test
(`drdetect.segmentation.metrics.best_dice_threshold`, the same selection/evaluation separation
`scripts/evaluate.py` already uses for the grading model's operating point) -- collapses that
spread to 0.0148, in line with what AUPRC's own noise floor suggests is real fold-to-fold variation.
Fold 2, the AUPRC outlier from Result 3, is also where fixed-0.5 Dice does the most damage (0.554
vs. 0.752 tuned): its less-mature checkpoint's probabilities apparently sit lower on average than
the other four's, which a fixed 0.5 punishes and a per-checkpoint tuned threshold does not.

Four of five tuned thresholds land in a tight 0.900-0.925 band; fold 3 is the outlier at 0.625.
Whether that reflects something real and analysable about fold 3's checkpoint, or is simply
five-sample noise, isn't something this data resolves -- worth another look if a sixth or seventh
fold's threshold also lands away from the 0.9 cluster.

## What's still open

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
