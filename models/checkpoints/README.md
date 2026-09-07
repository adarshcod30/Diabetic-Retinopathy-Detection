# Checkpoints — archived here, deleted from disk on 2026-09-08

This directory held 43 experiment checkpoints (119 `.ckpt` files, ~17 GB) — the output of
every grading ablation (Phase 3, 13 configs; Phase 8, 4 more), every segmentation model
(Phase 4: hard exudates and haemorrhages ×5 folds each, vessels ×5 folds, soft exudates,
microaneurysm-classifier, OD/fovea localisation, one abandoned plain-segmentation baseline),
the ConvNeXt-Tiny ablation run, the data-randomisation sanity-check model (Phase 6), and the
fusion head (Phase 5). All 43 were deleted on **2026-09-08** to reclaim local disk space.

**Nothing here was load-bearing for reproducing the project's findings** — every one of these
checkpoints' results is already fully reported, with real numbers and paired significance
tests, in the corresponding `docs/NN_PHASE*_RESULTS.md` file. The checkpoint files themselves
were only useful for further inference or fine-tuning from that exact point, and any of them
can be regenerated from the documented config + fixed seed (`seed=42` throughout) once a GPU
and the source data (see `../../data/raw/README.md`) are available again.

## The one that actually matters is not local — it's already released

`sweep_512_regression_fold0/best.ckpt` — the checkpoint this project would actually ship — is
**not lost**. It is published, verified present immediately before this cleanup, at:

- **HuggingFace Hub**: <https://huggingface.co/adarshcod30/drdetect-dr-screening> (`best.ckpt` +
  an ONNX export, parity-verified to 2.4×10⁻⁷ max abs diff)
- **GitHub Releases**: [v1.0](https://github.com/adarshcod30/Diabetic-Retinopathy-Detection/releases/tag/v1.0)
  (same two files, mirrored)

`app.py` (this repo's HuggingFace Spaces entry point) already fetches it from HF Hub at
runtime via `hf_hub_download` — it was never read from this local directory in the deployed
path. To use it locally:

```bash
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('adarshcod30/drdetect-dr-screening', 'best.ckpt'))"
```

See [`../../MODEL_CARD.md`](../../MODEL_CARD.md) for why this specific checkpoint (regression
loss, not the internally-stronger CE baseline) is the one released, and its documented
limitations.

## Regenerating any of the others

Every checkpoint's exact training command is in its own results doc's "Method" or
"Reproducing" section — for example the original baseline:

```bash
python scripts/train.py --size 512 --batch-size 4 --epochs 40 --lr 1e-4 --grad-clip 1.0 \
    --folds 0 --run-name baseline_effb0_512
```

`docs/07_PHASE3_RESULTS.md`, `docs/10`–`docs/15`, and `docs/21_PHASE8_ABLATION_RESULTS.md`
have the rest, one per configuration.
