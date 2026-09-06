# Phase 4 Results — Vessel Segmentation (DRIVE)

The roadmap's Phase 4 lists vessels first among the remaining lesion-evidence items. This is that
item: a DeepLabV3+/resnet34 model trained and cross-validated on DRIVE's 20 publicly-labelled
images, against the roadmap's own stated target (Dice ~=0.80+, AUC ~=0.97+).

## What this covers, and what it doesn't

Built: a dedicated vessel-segmentation harness --
`src/drdetect/segmentation/vessels.py`, `scripts/train_vessels.py` -- deliberately separate from
the IDRiD lesion harness (`dataset.py`/`module.py`) rather than sharing a base class with it, because
the two tasks differ in ways that matter: DRIVE images are small enough (584x565) to train on whole,
with no patch sampling; vessels are a measured ~7.5% of pixels, not IDRiD's <0.1%, so AUROC (not
AUPRC) is the roadmap's own correct choice of metric here; and every DRIVE image ships a
field-of-view mask that must exclude the black surround from both loss and metrics, a concept IDRiD's
lesion harness has no equivalent of.

Not built: OD/fovea regression, quadrant mapping, haemorrhages, soft exudates, and microaneurysms --
still queued behind this item in the roadmap's own stated order.

## Method

- **Data**: DRIVE's own public distribution ships vessel ground truth (`1st_manual/`) only for the
  20 images numbered 21-40 ("training" in DRIVE's own naming) -- confirmed directly by inspecting
  `data/raw/drive/test/`, which contains `images/` and a field-of-view `mask/` but no manual
  annotation directory at all. There is therefore no separate locked test split for this lesion
  type the way IDRiD hard exudates has one; all 20 labelled images are used for 5-fold
  cross-validation (`sklearn.model_selection.KFold`, 16 train / 4 val per fold).
- **Field-of-view masking**: both the training loss and every reported metric are computed only on
  pixels inside each image's own FOV mask. A model that trivially predicts "background" on the
  black surround around the circular retinal image would otherwise look better than it is --
  standard practice for this exact dataset, not a project-specific choice.
- **Loss**: FOV-masked `BCEWithLogitsLoss(pos_weight)` + FOV-masked soft Dice, summed. `pos_weight`
  (~6.9-7.1 across folds) is measured empirically from each fold's own FOV-masked training pixels --
  full images this time, not sampled patches, so this is the literal training-time ratio rather than
  an estimate of one.
- **A real bug caught before it wasted a training run**: `RandomRotate90` (one of this dataset's
  augmentations) swaps height and width for a 90-degree turn. DRIVE images are rectangular, so a
  batch containing both a rotated and an unrotated image produced two different tensor shapes and
  `torch.stack` refused to collate them. Padding was changed from "each axis to its own nearest
  multiple of 32" to "both axes to one shared square target," making every rotation shape-invariant.
  Caught by the smoke test, not the full run.
- **Model selection**: best epoch by `val/auroc` (the roadmap's own named target metric for this
  item), early-stopped at patience 8 epochs -- though no fold actually triggered early stopping; all
  five ran the full 40 epochs, still inching upward at the end.

## Result 1 — AUROC is stable and close to target (0.9416 ± 0.0035); Dice is not (0.662 ± 0.023)

```
fold    AUROC    Dice @ 0.5   epoch
0      0.9451      0.6641       38
1      0.9421      0.6980       37
2      0.9354      0.6241       37
3      0.9407      0.6628       39
4      0.9448      0.6614       38

mean +/- std    0.9416 +/- 0.0035        0.6621 +/- 0.0234
```

AUROC is both high and remarkably stable across folds (std 0.0035 -- tighter than the hard-exudate
segmentation CV's 0.0085, or the grading model's 5-fold QWK CV at 0.0116) and within reach of the
roadmap's 0.97+ target, though not there yet. Dice is a full 0.14 below the roadmap's 0.80+ target
and noticeably less stable across folds (std 0.0234, ~7x AUROC's relative spread) -- fold 2 in
particular sits well below the other four on both metrics, though not so far below as to look like a
degenerate outlier fold.

## Result 2 — The Dice gap is mostly real, not a fixed-threshold artifact

Given how much per-fold threshold tuning helped the IDRiD hard-exudate result (`docs/10`, Result 4:
fold 2 there went from Dice 0.554 to 0.752 once tuned), the same question needed asking here before
trusting 0.662 as a ceiling. Sweeping every threshold directly on each fold's own 4 validation
images (an upper-bound diagnostic, explicitly **not** a legitimate held-out tune -- 4 images is too
few to carve a further split from without the result being noise):

```
fold    Dice @ 0.5   best-possible Dice (same data)   threshold
0         0.6641              0.7051                    0.775
1         0.6980              0.7185                    0.700
2         0.6241              0.6676                    0.750
3         0.6628              0.6931                    0.725
4         0.6614              0.7038                    0.750

mean       0.6621 +/- 0.0234    0.6976 +/- 0.0170
```

Threshold tuning recovers about 3.5 points (0.662 -> 0.698) -- real, but nowhere near IDRiD's ~20
point swing on its worst fold, and nowhere near closing the 0.14 gap to the roadmap's target even at
the best threshold each fold's own data could possibly support. This is a meaningfully different
finding from Result 4 in `docs/10`: there, the fixed threshold was hiding most of the story; here, it
is not -- the shortfall is mostly a real property of the model, not a scoring artifact.

The likely mechanism is scale, in both senses of the word. Sixteen training images per fold is thin
even by this project's own standards (IDRiD hard exudates had 43-44 per fold); and Dice is
disproportionately punishing for thin structures specifically -- a vessel branch one pixel wide loses
a large fraction of its overlap score from a one-pixel misalignment along its length, in a way
AUROC's ranking-based formulation does not penalise nearly as heavily. Both point toward the same
conclusion: this model ranks vessel-vs-background correctly most of the time (hence strong, stable
AUROC), but its pixel-exact boundary localisation on fine capillary-scale structures is the part that
is genuinely still short of the target, not an artifact of how the number was read off.

## What's still open

- **Closing the Dice gap itself** was not attempted here -- candidates worth trying, in rough order
  of cost: more epochs with a lower learning rate late in training (all five folds were still
  improving, however slowly, at epoch 39); a deeper or higher-resolution-preserving encoder;
  boundary-aware loss terms (e.g. a distance-transform weighting) that penalise thin-structure
  misalignment less harshly than plain Dice. None of these were run this pass.
- **OD/fovea, quadrant mapping, haemorrhages, soft exudates, microaneurysms** -- all next in the
  roadmap's own stated Phase 4 order, still entirely unstarted.
- **A qualitative overlay check** (rendering predicted vessel maps against the real images) was not
  done this pass -- the roadmap's Phase 4 exit criterion calls for "qualitative overlays that a
  clinician would recognise" alongside the quantitative numbers.
