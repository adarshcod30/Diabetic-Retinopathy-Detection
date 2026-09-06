# Phase 4 Results — Soft Exudate Segmentation (IDRiD)

Soft exudates reuse the exact same harness as hard exudates and haemorrhages --
`src/drdetect/segmentation/{dataset,model,module,metrics}.py` and
`scripts/{train_segmentation,evaluate_segmentation}.py`, switched via `--lesion soft_exudates`. No
new code was needed. Per the user's own scoping decision partway through Phase 4 (to manage total
time across the remaining Phase 4/5/6 items), this is a **single train/val split**, not 5-fold CV --
the last lesion type to use the fuller methodology was haemorrhages (`docs/13`).

## What this covers, and what it doesn't

Built: nothing new. Same DeepLabV3+/resnet34, same BCE+Dice loss with an empirically-measured
`pos_weight`, same tiled full-resolution inference and Dice-threshold-tuning methodology as
`docs/10` and `docs/13`.

Not built: microaneurysms remains -- the last IDRiD lesion type, and the roadmap is explicit that it
needs a different method entirely (morphological candidate generation + small classifier, not plain
segmentation), so it doesn't reuse this harness the way the other three lesion types did.

## Method

Soft exudates are IDRiD's smallest segmentation subset: 26 training images (further split 20
train / 6 val for this single-split run) and 14 official test images -- half of haemorrhages'
training set (53) and less than a third of hard exudates' and haemorrhages' combined test set (27
each). The roadmap itself flags this lesion type as having "the fewest training examples, weakest
signal" going in.

- **Split**: single 20/6 train/val split (no KFold loop this time -- `--folds 0` only), 15%-ish val
  fraction consistent with the OD/fovea single-split precedent.
- **pos_weight**: 18.17, empirically measured from sampled training patches (between hard exudates'
  ~12.65 and a lesion this sparse would suggest -- soft exudates lesions are less frequent per patch
  than hard exudates but more frequent than haemorrhages' dot lesions).
- **Training**: early-stopped at 26 epochs (patience 8), best epoch 16.
- **Evaluation**: tiled full-resolution inference on both the 6 val images (to tune the Dice
  threshold) and the 14 official test images (touched once), identical procedure to `docs/10` and
  `docs/13`.

## Result 1 — Test AUPRC 0.614, a val→test gap between hard exudates' and haemorrhages'

```
                        val (6 img)   test (14 img, held out)
AUPRC                     0.8117         0.6144
Dice @ 0.5 (fixed)          --           0.5546
Dice @ 0.775 (tuned)      0.5755         0.5929
```

The val -> test AUPRC drop here (0.812 -> 0.614, a fall of 0.197) sits between the two reference
points already on record on this identical harness: hard exudates dropped only 0.049 (0.899 ->
0.850, `docs/10` Result 3), haemorrhages dropped 0.227 (0.767 -> 0.540, `docs/13` Result 1). Soft
exudates -- with the fewest training images of the three -- lands closer to haemorrhages' gap than
hard exudates', consistent with the roadmap's own prior expectation that this lesion type would be
the hardest to generalise well from so little data. This is a single split, not 5-fold, so unlike
the other two there is no std-across-folds to report -- one number, not a distribution, and
correspondingly less confidence that this exact gap is stable rather than partly an artifact of
which 6 images happened to land in validation.

Threshold tuning still helps, in the same direction as every other lesion type tried so far: Dice
@0.5 (0.5546) -> Dice @ tuned 0.775 (0.5929), a gain of +0.038. Smaller in absolute terms than hard
exudates' tuning effect but directionally identical -- the default 0.5 threshold is consistently
suboptimal across all lesion types tested, and a per-fold/per-run tuned threshold consistently
recovers some of that gap.

### Reading the gap size relative to training-set size

Three lesion types, one harness, three different val->test gaps, roughly tracking training-set size
in the expected direction (fewer images -> larger gap) but not perfectly:

```
lesion            train images   val->test AUPRC drop
hard exudates          43                0.049
soft exudates          20 (of 26)        0.197
haemorrhages           43                0.227
```

Soft exudates has fewer training images than haemorrhages (20 vs 43 in-fold) yet a very slightly
smaller gap. This is weak evidence, not a contradiction -- haemorrhages' own stated hypothesis
(`docs/13`) was morphological variability (dot vs. blot haemorrhages differing considerably in size
and shape), not training-set size alone, and that hypothesis is consistent with soft exudates
(visually more homogeneous than haemorrhages, closer to hard exudates' consistency) generalising
comparably despite less data. Neither mechanism has been directly measured for any of the three
lesion types -- both remain plausible, unverified explanations.

## What's still open

- **Microaneurysms** -- the last IDRiD lesion type, and the only one requiring a different method
  (morphological top-hat/matched-filter candidate generation feeding a small classifier, not
  encoder-decoder segmentation). Not started.
- **No 5-fold CV here**, by the user's own explicit scoping decision -- a single split was used to
  get an honest first number fastest given how much of Phase 4/5/6 remained queued. The reported
  0.614 test AUPRC should be read as one data point, not a mean with a known variance, unlike hard
  exudates' and haemorrhages' CV-backed numbers.
  - **The train-set-size-vs-morphology question** (why soft exudates' gap isn't larger than
  haemorrhages' despite less data) was not directly investigated -- would need per-lesion-type
  size/shape variance measurements across both datasets to move from "plausible" to "demonstrated."
