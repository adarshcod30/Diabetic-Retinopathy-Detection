# Phase 5 Results — Lesion Feature Extraction + Fusion Head

Phase 4's per-lesion models made this item possible for the first time: `concat(CNN embedding,
lesion features) -> ordinal head`. This result set builds both the feature extractor and the
fusion head, and reports an honest negative result -- on this first attempt, at this sample size,
fusion does not beat the grading-only baseline.

## What this covers, and what it doesn't

Built: `src/drdetect/fusion/features.py` (runs Phase 4's five lesion models as inference engines
over APTOS images, since APTOS carries DR grades but no lesion ground truth of its own),
`src/drdetect/fusion/embedding.py` (extracts the grading CNN's 1280-d pre-classifier embedding),
`src/drdetect/fusion/model.py` (`FusionHead`, a small MLP), and
`scripts/{extract_lesion_features,train_fusion}.py`. The fusion head reuses `GradingModule`
unchanged -- it is already loss-agnostic, so the only new model-specific code is the small MLP
itself.

Not built: uncertainty estimation (MC-dropout/ensemble) and the human-escalation policy -- next in
the roadmap's own order for Phase 5's remaining items. The frozen operating-point selection the
roadmap also lists was already built in `scripts/evaluate.py` back in Phase 1 (not a Phase 5 gap;
it just hadn't been cross-referenced from this section of the roadmap before).

## Two deliberate scope decisions, stated plainly

- **APTOS, not IDRiD.** IDRiD's grading images are byte-identical to its localisation images
  (verified by SHA-256) -- tempting, since it would give real OD/fovea/lesion ground truth instead
  of model-inferred features. Not used: `docs/04_ROADMAP.md` Phase 8 reserves IDRiD (with
  Messidor-2) as the locked, evaluate-once final test set, and using it here would contaminate it.
  So every lesion feature in this result comes from running IDRiD-trained models on APTOS images
  they were never validated against -- a real, acknowledged domain-shift risk, checked (not
  eliminated) by a cheap sanity check before committing to building the rest of the pipeline: the
  hard-exudate model's mean predicted probability rose with DR grade across a small stratified
  APTOS sample (grade 0: 0.0000-0.0025 positive-pixel fraction at threshold 0.5; grade 3:
  0.023-0.114) despite having never seen an APTOS image during training.
- **`baseline_effb0_512_fold0` (QWK 0.8930), not the 1024px checkpoint.** Phase 3's own conclusion
  was "the baseline stands," and Result 14 (`docs/07_PHASE3_RESULTS.md`) retracted the 1024px
  checkpoint's causal story -- its edge over the baseline is now suspected to be a backend
  artifact. Building Phase 5 on a result the project already walked back would undercut that
  finding.

## Method

- 750 APTOS images (150 per DR grade, stratified), lesion features extracted via full-resolution
  tiled inference (hard exudates, soft exudates, haemorrhages) plus fixed-resolution localisation
  and full-resolution microaneurysm candidate generation -- 9 features total: mean predicted
  probability for 3 lesion types, accepted microaneurysm count, mean distance-to-fovea (normalised
  by OD diameter), and 4 per-quadrant lesion-count fractions.
- CNN embedding: the baseline checkpoint's 1280-d pre-classifier pooled feature
  (`model.backbone.forward_head(features, pre_logits=True)`), extracted from the same
  Ben-Graham-preprocessed cache the grading model was actually trained on (lesion features, by
  contrast, use the raw APTOS images -- the Phase 4 models were trained on IDRiD's un-enhanced
  originals, so each model gets the image distribution it actually learned from).
- Single train/val split (600/150), grouped by APTOS's own perceptual-hash near-duplicate groups
  (`stratified_group_split`, `SplitStrategy.TRUE_GROUPS` -- confirmed no group leaks across the
  split), per the scoping decision applied to every Phase 4/5/6 item since soft exudates.
- Lesion features standardised on train-split statistics only (they range from ~0-0.2 for mean
  probabilities to 0-80+ for microaneurysm count); the embedding is left as-is.
- `FusionHead`: a 2-layer MLP (1289 -> 128 -> 4), trained with `CornLoss` (rank-consistent ordinal
  regression, this project's existing implementation) via the unmodified `GradingModule` --
  identical training/decoding/QWK-logging code path as the main grading pipeline.

## Result 1 — Fusion does not beat grading-alone on this first attempt

```
                          grading-alone (baseline)   fusion head
val QWK (150 img)               0.9556                  0.9362
val accuracy                    0.9067                  0.8933

McNemar (paired, same 150 val images):
  fusion-correct, baseline-wrong  : 4
  baseline-correct, fusion-wrong  : 6
  p = 0.754 (not significant)
```

The roadmap's own exit criterion for this item -- "the fusion model beats the grading-only model
with a significant McNemar p-value" -- is not met. If anything the numbers point the other way:
lower QWK, lower accuracy, and more discordant pairs favouring the baseline than the fusion head.
None of that difference is statistically distinguishable from noise (p=0.754, and only 10
discordant pairs total out of 150 images -- this comparison has very little statistical power at
this sample size, in either direction).

**This baseline QWK (0.9556) is not the project's canonical 0.8930** -- it's the same checkpoint,
scored on a different, much smaller (150-image) val slice specific to this experiment's own split,
not Phase 3's full validation set. The two numbers are not in tension; they're different
measurements of the same model on different data.

### Plausible (not verified) reasons fusion didn't help here

- **Sample size**: 600 training images is small for a 165K-parameter head to find a reliable
  correction on top of an already-strong embedding, and 150 val images give only 10 discordant
  pairs to test on -- likely too little power to detect a modest real effect even if one exists.
- **Lesion feature quality**: every feature here comes from IDRiD-trained models running,
  unvalidated, on a different dataset (see the domain-shift caveat above), and the microaneurysm
  count specifically comes from a classifier already shown (docs/15) to generalise poorly from its
  own training distribution to a real image's candidate flood -- noisy input features cap how much
  a fusion head can extract from them regardless of architecture.
- **Redundancy, not complementarity**: the 1280-d CNN embedding may already implicitly encode most
  of what these 9 scalar lesion features make explicit, leaving little genuinely new signal for
  the fusion head to add.

These are candidate explanations consistent with what's already known about this pipeline, not
independently demonstrated causes -- distinguishing between them (e.g. an ablation training the
fusion head on lesion features alone, or scaling to more images) is future work, not attempted
here given how much of Phase 5/6 remains.

## What's still open

- **This is a null result on the first attempt, not a closed question.** The roadmap's exit
  criterion is unmet, but nothing here rules out fusion helping with more data, cleaner lesion
  features (particularly a better microaneurysm classifier, docs/15's own stated next step), or a
  different fusion architecture.
- **Uncertainty estimation and the human-escalation policy** remain unbuilt -- next in the
  roadmap's own order. (Operating-point selection is not on this list -- see above.)
- **A larger APTOS subset** than 750 images was not attempted this pass, given the ~46 minutes the
  750-image extraction already took (5 models per image) and the amount of Phase 5/6 work still
  queued behind this item.
