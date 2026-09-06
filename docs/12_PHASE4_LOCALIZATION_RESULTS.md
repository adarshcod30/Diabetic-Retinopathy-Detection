# Phase 4 Results — OD/Fovea Localisation & Quadrant Mapping (IDRiD)

The roadmap's exit criterion for OD/fovea: mean localisation error < 0.5x OD diameter. Unlike
hard-exudate segmentation or vessels, this one clears its target outright, on a genuine held-out
official test set. Quadrant mapping, the roadmap's next item and a direct consumer of this one's
output, is also covered here (Result 3) since it's a small deterministic follow-on rather than a
separate model.

## What this covers, and what it doesn't

Built: a heatmap-regression harness --
`src/drdetect/segmentation/localization.py`, `scripts/{train_localization,evaluate_localization}.py`
-- plus `src/drdetect/segmentation/quadrants.py` for quadrant mapping. A DeepLabV3+/resnet34 model
with a 2-channel output (optic disc, fovea), each channel trained against a 2D Gaussian centred on
the true point, with the predicted point read back off as each channel's argmax at inference.

Not built: haemorrhages, soft exudates, and microaneurysms -- next in the roadmap's own stated
order for the remaining IDRiD lesion types.

## Method

- **Data**: IDRiD's "C. Localization" task ships a genuine official split -- 413 training / 103
  test images, each with (x, y) pixel coordinates for both the optic disc and fovea centre, at full
  (2848x4288) resolution. This is the largest, cleanest-labelled IDRiD subset used in this project
  (the hard-exudate segmentation harness has 81 images total; this one has 516). A 15% internal
  validation split was carved from the 413 training images for model selection; the 103 official
  test images were touched exactly once, at final evaluation.
- **Working resolution**: 512x768 (matching IDRiD's real ~1.505:1 aspect ratio to within 0.3%),
  chosen because heatmap-regression localisation doesn't need native 2848x4288 resolution the way
  pixel-exact segmentation does. Coordinates and heatmaps are generated in this working space, and
  the OD-diameter normalisation constant is rescaled to match (see below) so no full-resolution
  coordinate transform is needed anywhere in training or evaluation.
- **OD-diameter normalisation is a cross-subset proxy, not a per-image ground truth.** The
  roadmap's success unit is "0.5x OD diameter", but per-image OD diameter isn't available for the
  localisation task itself -- only the disjoint 81-image segmentation subset ships OD masks, and
  the two subsets use different filename numbering (`IDRiD_01..54` vs `IDRiD_001..413/103`), so
  they cannot be joined by image ID. The mean OD diameter measured directly from the 54 real
  segmentation masks (527.7px at full resolution, equivalent circular diameter from measured mask
  area) is used as a fixed constant instead of a per-image value or an invented one.
- **Loss**: MSE between sigmoid-activated predictions and the target Gaussian heatmaps, standard
  for this style of landmark regression.
- **Model selection**: best epoch by `val/mean_error_diameters` (OD and fovea error averaged),
  early-stopped at patience 8 -- though training ran the full 40 epochs regardless, still finding
  marginal new bests near the end.

## Result 1 — Target cleared: combined mean error 0.075 diameters on the official test set (target < 0.5)

```
                    internal val (62 img)   official test (103 img, held out)
OD error (mean)             --                     0.0463
Fovea error (mean)          --                     0.1029
Combined mean             0.0587                   0.0746
```

103 of 103 test images clear the 0.5-diameter target on OD alone; 100 of 103 clear it on fovea. The
val -> test gap (0.059 -> 0.075) is small and in the expected direction. This is the first Phase 4
sub-item to clearly meet its roadmap-stated target -- unlike hard-exudate segmentation (partial,
CV-confirmed but the phase's overall AUPRC target was never explicitly numeric) or vessel
segmentation (AUROC close but Dice well short), OD/fovea localisation has an unambiguous numeric
bar and clears it by a wide margin (0.075 actual vs. 0.5 target, roughly 6.7x margin).

## Result 2 — OD localisation is reliable everywhere; fovea has three real outliers

```
image        OD error   fovea error
IDRiD_065     0.037        1.665
IDRiD_005     0.037        0.969
IDRiD_067     0.012        0.838
IDRiD_017     0.045        0.357
IDRiD_042     0.026        0.317
```

The five worst fovea cases all have excellent OD localisation (0.01-0.05 diameters) -- the model
never loses track of the optic disc, even on images where it completely misses the fovea. This
points at fovea-specific failure rather than a general localisation weakness: the fovea is a subtle,
low-contrast feature (no sharp boundary the way the optic disc has), and on three test images
(IDRiD_065, IDRiD_005, IDRiD_067) the model's predicted heatmap peak lands far enough from the true
fovea to fail the 0.5-diameter bar outright, most severely on IDRiD_065 at 1.665 diameters -- over
three times the target. Median fovea error (0.053) is far below the mean (0.103), confirming these
are a small number of real outliers dragging the mean up, not a systematically noisy metric.

## Result 3 — Quadrant mapping: a deterministic geometric follow-on, no model needed

`src/drdetect/segmentation/quadrants.py` divides the retina into four quadrants using two lines
through the OD -- one along the OD-fovea axis, one perpendicular to it -- exactly as the roadmap
specifies. No training involved; the only inputs are the OD and fovea (x, y) points this phase's
own model already produces. Verified two ways: 6 unit tests (axis orthogonality/unit-length,
correct handling of an OD-fovea axis at an arbitrary orientation, all four quadrants distinguishable
and consistent), and a direct check against a real trained prediction (`IDRiD_001`, OD=(651, 1453),
Fovea=(1914, 1617)) confirming two diagonally-opposite sample points resolve to diagonally-opposite
quadrant labels.

**One deliberate scope limit, stated plainly rather than left implicit**: quadrant labels here
describe geometry ("foveal-side" / "disc-side", "superior" / "inferior"), not asserted anatomy
("nasal" / "temporal"). Which side of the OD is nasal vs. temporal depends on which eye (OD/OS) an
image is of, and neither IDRiD nor this project's other datasets carry reliable per-image
eye-laterality labels to resolve that safely. A deployment with real laterality metadata could
remap these four geometric labels to true anatomical quadrants in one place
(`QUADRANT_LABELS`) without touching the underlying axis math.

## What's still open

- **The three fovea outliers** were not investigated further (e.g. whether they share a quality or
  pathology characteristic) -- worth a look if fovea-specific accuracy becomes load-bearing for a
  downstream feature (e.g. distance-to-fovea in Phase 5's lesion feature extractor).
- **5-fold CV** was not run here, matching the same scoping decision made for vessels this pass --
  a single internal-val/official-test split was used to get a first, honestly-evaluated number
  fastest, given how much of Phase 4 remains queued behind this item.
- **Haemorrhages, soft exudates, microaneurysms** -- still entirely unstarted.
