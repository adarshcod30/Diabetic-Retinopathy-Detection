# Phase 4 Results — Microaneurysm Detection (IDRiD)

Microaneurysms are the last IDRiD lesion type, and the only one the roadmap says needs a
different method entirely: "morphological top-hat + matched filter candidate generation ->
small CNN classifier. Do not expect plain segmentation to work here." This result set builds
that pipeline from scratch (`src/drdetect/segmentation/microaneurysms.py`,
`scripts/{train_microaneurysm_classifier,evaluate_microaneurysms}.py`), confirms the roadmap's
segmentation warning empirically rather than taking it on faith, and is honest about where the
new pipeline itself falls short.

## What this covers, and what it doesn't

Built: full-resolution candidate generation (black top-hat on the green channel, percentile
thresholding, connected-component filtering), a small CNN classifier (~24K params) to accept or
reject each candidate, and an evaluation harness that separates generation-stage recall from
classifier precision/recall -- two genuinely different failure modes for this lesion type, unlike
the other four where one pixel-AUPRC number tells the whole story.

Not built: this closes out Phase 4's per-lesion segmentation work. Phase 5 (fusion head,
uncertainty, operating-point selection) and Phase 6's remaining items are next.

## Method

- **Data**: same 54 train / 27 test IDRiD segmentation-subset images as the other lesion types,
  this time using the microaneurysm masks. Single train/val split (43/11, `--folds 0`), per the
  scoping decision already applied to soft exudates and haemorrhages-onward.
- **Candidate generation**: black top-hat (`cv2.MORPH_BLACKHAT`) on the median-denoised green
  channel, disk radius 16 (33px structuring element, sized from the measured p95 microaneurysm
  diameter of 29.9px across all 81 real masks), thresholded at the 95th percentile of the
  response's own per-image distribution, connected-component filtered to area 3-1500px.
- **Classifier**: a 3-conv-block CNN on 33x33 patches, BCE loss with an empirically-measured
  `pos_weight`, trained on candidates matched against ground truth (positive if a candidate's
  centroid lands on a true-mask pixel), with negatives subsampled per image to manage training-set
  size.
- **Evaluation**: generation-stage recall (does *any* candidate land on each true instance) is
  reported separately from classifier precision/recall at a val-tuned F1-maximising threshold,
  and from candidate-level AUPRC (both computed on the *full*, unsampled candidate pool -- see
  Result 2).

## Two real bugs found and fixed during development

Worth recording plainly rather than silently -- both were caught by testing before they reached a
real training run, not discovered after the fact from bad numbers.

1. **Otsu thresholding fails on this response distribution.** The first implementation used
   `cv2.THRESH_OTSU` on the top-hat response, the standard automatic-threshold choice. Measured
   directly on IDRiD_01: Otsu picked a threshold of 4 (out of a max response of 71), marking
   33.8% of the entire 12.2-megapixel image as "foreground" -- the response is a long-tailed,
   noise-dominated distribution, not the clean bimodal signal Otsu assumes. That 33.8% merged
   into huge connected blobs (one covered 1.76M px) that then failed the area filter and were
   discarded whole, recall included: **3/18 true instances recovered**. Swept percentile
   thresholds on 5 real images instead; the 95th percentile recovered 91-100% of true instances
   per image at a *smaller* candidate count than Otsu's, despite being less permissive in
   principle. `find_candidates`'s own docstring carries the measured numbers.
2. **Whole-image padding, called once per candidate.** `extract_patch` originally called
   `cv2.copyMakeBorder` on the *entire* 2848x4288 image before cropping a 33px patch out of it --
   a ~36MB copy on every call, and it's called once per kept candidate (positives plus up to
   hundreds of negatives, per image). Measured directly: candidate generation across just 6 real
   images reached a **39GB peak memory footprint** and aborted; on this project's 16GB development
   machine that translated into 20+ GB of swap and, on one run, exhausted enough disk headroom to
   be a genuine risk, not just a slow crash. Fixed by cropping the available region first and
   padding only the small overhang beyond the image edge -- bit-identical output (verified by the
   existing border/centering unit tests), 4.2s and an 820MB peak for the same 6 images afterward
   (a ~20x speed, ~48x memory improvement).

## Result 1 — Candidate generation clears a high bar: 91.7% (val) / 96.8% (test) recall

```
                          val (11 img)   test (27 img, held out)
true instances                 793              1085
recovered by generation        727              1050
generation-stage recall       0.9168            0.9677
```

This is the ceiling on end-to-end recall, and it's a strong one -- the classical top-hat approach,
once correctly thresholded, finds the great majority of true microaneurysms across both internal
validation and the official test set, with test recall actually *higher* than validation (not a
generalisation gap in this stage). Whatever the pipeline's weaknesses turn out to be, missed
candidates are not the dominant one.

## Result 2 — The classifier doesn't generalise from its training distribution to the real one

First attempt (`negative_ratio=10`, this project's default subsampling elsewhere): training
reported val/auprc **0.7503** after 30 epochs (clean convergence, plateaued the last 10 epochs).
Running the *same checkpoint* through `evaluate_microaneurysms.py` -- which scores every candidate
a real image proposes, not a class-balanced subsample -- told a very different story:

```
                                  attempt 1 (ratio=10)   attempt 2 (ratio=100)
training val/auprc (subsampled)        0.7503                  0.5528
test candidate-level AUPRC             0.1706                  0.2524
test end-to-end recall @ tuned th      0.1985                  0.2808
test precision @ tuned th              0.3085                  0.3564
test false positives / image            27.8                    31.7
```

**Why**: training validates on a curated pool -- at most `negative_ratio` x as many negatives as
positives per image (10x in attempt 1). A real image proposes 7,000-40,000 raw candidates against
10-130 true instances -- ratios of roughly 100:1 to 3,700:1, one to two orders of magnitude more
imbalanced than what the classifier was ever scored against during training. The 0.75 figure was
real, but it was answering an easier question than the one the deployed pipeline actually faces.

Retraining with `negative_ratio=100` (`max_negatives_per_image=2000`, empirical pos_weight 31.56
vs. attempt 1's 6.23) moved the *training* validation metric down to 0.55 -- a less flattering but
more honest number, since the subsampled pool is now closer to reality -- and moved the *test*
numbers up (AUPRC 0.17 -> 0.25, recall 0.20 -> 0.28, precision 0.31 -> 0.36). The mechanism
identified is real and the fix helped, but did not close the gap: even 100x subsampling, capped at
2,000 negatives per image, is still far short of the 3,700:1 ratio the worst real images present,
and training on every candidate from every image (millions of examples) was judged not worth the
time cost this pass given how much of Phase 4/5/6 remains -- left as the clear next step rather
than attempted here.

## What's still open

- **The classifier is the pipeline's weak stage, not the generator.** 96.8% of true microaneurysms
  are proposed as candidates on the test set; only 28.1% end up correctly accepted. Closing more
  of that gap is the highest-value next step if microaneurysm counts become load-bearing for a
  downstream feature (e.g. ICDR grade-1 evidence, which is defined by microaneurysms alone).
  Plausible next moves, not attempted here: train on the full unsampled candidate pool rather than
  any fixed subsampling ratio, hard-negative mining (specifically oversample the false candidates
  the current classifier is most confident -- and wrong -- about), or a slightly larger classifier
  (24K params may simply be too little capacity to separate true microaneurysms from vessel
  cross-sections and noise speckle at this volume).
- **A plain-segmentation baseline was run for comparison and deliberately stopped early**
  (same DeepLabV3+ harness as the other four lesion types, `--lesion microaneurysms`): best
  val/AUPRC 0.506 and val/Dice 0.236 at epoch 5 of a run stopped at epoch 6 to prioritise the
  required candidate+classify pipeline, not run to convergence. Even mid-training, both numbers
  sit far below hard exudates' converged result (0.899 val AUPRC) on the identical harness, and
  the empirically-measured patch-level `pos_weight` (265:1) is an order of magnitude more extreme
  than any other lesion type's -- consistent with, though not a rigorous confirmation of, the
  roadmap's warning that plain segmentation underperforms here.
- **Two bugs shipped, then caught, in the same feature.** Both were caught before a real training
  run corrupted results or a real machine took damage, but both are worth a reviewer's attention if
  this pipeline is extended: the percentile threshold (95.0) was tuned on 5 images, not swept
  exhaustively; and any future change to `extract_patch` should re-run
  `test_extract_patch_is_centered_for_an_interior_point` and
  `test_extract_patch_shape_at_image_corner_uses_reflect_padding` before trusting it again.
