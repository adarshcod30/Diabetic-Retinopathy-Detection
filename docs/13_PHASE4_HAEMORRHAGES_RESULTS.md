# Phase 4 Results — Haemorrhage Segmentation (IDRiD)

Haemorrhages reuse the exact hard-exudate harness unchanged --
`src/drdetect/segmentation/{dataset,model,module,metrics}.py` and
`scripts/{train_segmentation,evaluate_segmentation}.py` already generalise across lesion types via
`--lesion`. No new code was needed; only new runs. The headline finding here is a genuinely
different, less reassuring one than hard exudates produced: internal validation looks strong, but
it doesn't carry over to the held-out test set nearly as well.

## What this covers, and what it doesn't

Built: nothing new. Same DeepLabV3+/resnet34, same BCE+Dice loss with an empirically-measured
`pos_weight`, same 5-fold CV over IDRiD's 53 haemorrhage-labelled training images, same tiled
full-resolution inference and Dice-threshold-tuning methodology as `docs/10_PHASE4_RESULTS.md`.

Not built: soft exudates and microaneurysms remain, next in the roadmap's own order.

## Method

Identical to hard exudates (`docs/10`), substituting `--lesion haemorrhages`. 53 training images (27
official test images, held out and touched once), 5-fold CV, tiled inference, Dice tuned per fold
on that fold's own validation split.

**One infrastructure incident worth recording honestly**: this run was interrupted mid-fold-4 by a
SIGTERM at a session boundary (an environment teardown/reconnect, not a code or data problem --
confirmed via the log's own `Received SIGTERM: 15` line). `latest.ckpt` had been saved moments
earlier, so fold 4 was resumed from it with `--resume` and finished separately rather than
re-trained from scratch. `ModelCheckpoint`'s and `EarlyStopping`'s internal state both survived the
resume correctly -- the resumed run picked up mid-patience-count and stopped at the same best epoch
(0.7295 AUPRC, originally observed live before the interruption) it would have reached uninterrupted.
The one real side effect: Lightning's CSVLogger deletes a fold's own prior log on resume, so fold
4's `metrics.csv` only shows post-resume epochs -- the full 5-fold `summary.json` was reconstructed
by hand from all five folds' own recorded results rather than regenerated automatically.

## Result 1 — Internal validation is respectable (0.767 ± 0.033); the test set is a different story (0.540 ± 0.041)

```
fold    val AUPRC   test AUPRC   Dice @ 0.5   Dice @ tuned   tuned threshold
0        0.8112       0.5514       0.3968        0.5380           0.975
1        0.7644       0.5375       0.4979        0.5416           0.850
2        0.7334       0.5844       0.4957        0.5603           0.825
3        0.7981       0.5628       0.4935        0.5340           0.975
4        0.7295       0.4649       0.4088        0.4869           0.950

mean +/- std
val AUPRC    : 0.7673 +/- 0.0330
test AUPRC   : 0.5402 +/- 0.0407
Dice @ 0.5   : 0.4585 +/- 0.0457
Dice @ tuned : 0.5322 +/- 0.0244
```

The val -> test drop here (0.767 -> 0.540, a fall of 0.227) is far larger than hard exudates showed
on the identical harness (0.899 -> 0.850, a fall of 0.049, `docs/10` Result 3). Every fold drops
substantially and consistently -- this isn't one bad fold dragging the mean down, all five fall in
the same direction by a similar amount. Threshold tuning still helps and still reduces variance
(Dice std 0.0457 -> 0.0244, the same direction as Result 4's hard-exudate finding), but it cannot
close a gap this size; even the best achievable threshold on each fold's own data leaves test
performance well below internal validation.

### A plausible mechanism, not a proven one

Haemorrhages are morphologically more varied than hard exudates -- small dot haemorrhages and
larger blot haemorrhages differ considerably in size and shape, where hard exudates are more
visually consistent (bright, sharply-bordered, similar texture across instances). A 43-image
training fold may simply not cover enough of that variation to generalise as cleanly to the 27 test
images as it did for the more homogeneous lesion type. This is a reasonable explanation given what's
already known about the two lesion types' appearance, but it has not been separately verified here
(e.g. by measuring haemorrhage size/shape variance directly) -- stated as a plausible mechanism, not
a demonstrated one.

## What's still open

- **The mechanism behind the large val-test gap** was not directly investigated (e.g. quantifying
  haemorrhage size/shape variance in train vs. test, or checking whether specific test images drive
  the drop). Worth a look if haemorrhage segmentation becomes load-bearing for a downstream feature.
- **Soft exudates and microaneurysms** -- next in the roadmap's own order. Soft exudates in
  particular has only 26 training images (half of haemorrhages'), so a similar or worse
  generalisation gap would not be surprising there either.
- Per the user's own scoping decision partway through this phase, **soft exudates and everything
  after uses a single train/val split**, not 5-fold CV, to manage total time budget across the
  remaining Phase 4/5/6 items -- this run was already committed to full CV before that decision and
  was carried through to completion rather than abandoned partway.
