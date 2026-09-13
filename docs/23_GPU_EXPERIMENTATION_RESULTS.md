# GPU Experimentation Results — Two Nulls, One Real Win

This isn't a numbered roadmap phase. [`docs/04_ROADMAP.md`](04_ROADMAP.md) ends at Phase 9
(Released), and every phase before it used only local CPU/MPS compute by deliberate scoping
decision ([`docs/05_PROTOTYPE_SCOPE.md`](05_PROTOTYPE_SCOPE.md) §6). This project's contact at
JarvisLabs.ai arranged sponsored GPU credits after that release was already live — the first time
real GPU training was available at all. What follows is what came out of using it: two things that
looked promising on paper and didn't pan out, one thing that did, and a training-backend effect
that had to be understood before any of the other three numbers could be trusted.

Every result below follows this project's own discipline: measured, not estimated; nulls reported
as nulls; and the two locked external test sets this project has already spent
([`docs/22_PHASE8_VALIDATION_RESULTS.md`](22_PHASE8_VALIDATION_RESULTS.md) — Messidor-2 + IDRiD)
were **not** touched again. A new one (DDR) was sourced instead, for the one candidate that
actually earned a look outside APTOS.

## A GPU changes the number by itself — control for backend before anything else

The very first run on the rented A30 — the shipped config, 512px EfficientNet-B0, regression
loss, same seed (42), same recipe as the released checkpoint — came back at **0.9158 QWK**,
meaningfully below what was already published. Before concluding anything about EyePACS or
resolution, that gap itself needed an explanation:

| Hypothesis | Test | Result |
|---|---|---|
| TF32 (Ampere reduced precision) | disabled explicitly | 0.9160 vs 0.9158 — no change |
| Data pipeline drift | byte-diffed local vs. remote manifests | identical |
| Package version drift | pinned torch/torchvision to exact local versions | 0.9174 vs 0.9158 — no change |
| Unlucky seed | 3 seeds tried on the GPU | 0.894–0.916 — seed 42 was the *best* of the three |

None of it explained the gap. **Conclusion: this is real, and the root cause was never found.**
It isn't a new mystery, either — it's the third time this exact phenomenon has shown up in this
project's history. [`docs/07_PHASE3_RESULTS.md`](07_PHASE3_RESULTS.md)'s own Result 14 found that
swapping MPS→Kaggle-CUDA alone moved QWK by 0.031 (p=0.001) with nothing else changed. This
session found a similar-or-larger gap (~0.04) on a third backend pairing (local MPS vs. this
JarvisLabs A30/A100). **Any comparison between a locally-trained and a cloud-GPU-trained
checkpoint on this project must control for backend, or the comparison isn't valid** — which is
exactly why every result below is measured against a same-A30 control trained fresh alongside it,
never against the historical published number directly.

## EyePACS pretraining: null, once backend is controlled for

With the 0.9158 same-A30 control in hand, a two-stage warm-start (pretrain on a 15k-image
stratified EyePACS subset, then fine-tune on APTOS) scored **0.9124 — p=0.590**. Not a real
difference.

Getting the EyePACS subset at all took a real detour: the original plan (`scripts/download_data.sh`,
per [`docs/05_PROTOTYPE_SCOPE.md`](05_PROTOTYPE_SCOPE.md) §3.2) assumed the Kaggle competition's
own `train.zip.001`–`.005` could be joined and extracted normally. It can't — verified directly,
not assumed: it's a genuine multi-disk PKZIP split where each volume's local file headers are
offset relative to that volume alone. A plain `cat` of the five parts produces "invalid zip file
with overlapped components," and neither Info-ZIP's own split-join nor a hand-written multi-part
stream reader gets past the central directory to an actual per-file read. `scripts/fetch_eyepacs_subset.py`
now pulls from `tanlikesmath/diabetic-retinopathy-resized`, a long-established community re-host
of the same images under the same filenames, packaged as one normal ~7.8GB archive — the labels
still come from the competition's own `trainLabels.csv`, which was never the broken part. The
fetch works now; the pretraining it enabled still didn't move the number.

## 1024px resolution: null, for the fourth time

Same 0.9158 same-A30 control, this time against a 1024px run: **p=0.575**, not a real difference.
This is the fourth time this exact hypothesis has been tested somewhere in this project's history
— Phase 3 on Kaggle, Phase 3's own backend-confound experiment (Result 14, which refuted an
earlier apparent 1024px lead as a backend artifact — [`docs/07_PHASE3_RESULTS.md`](07_PHASE3_RESULTS.md)
Result 12 vs. 14), and now this A30 — and it has never once held up under a controlled comparison.

## A proper 5-fold CV for the shipped config, for the first time

Phase 8 only ever validated the production regression-loss config on a single fold. This phase
ran the full 5-fold CV for it for the first time: **0.9094 ± 0.0079**, against the old CE
baseline's own historical 5-fold CV of 0.8965 ± 0.0116
([`docs/07_PHASE3_RESULTS.md`](07_PHASE3_RESULTS.md) Result 10). Regression loss's advantage
holds up across folds and across backends, not a single-fold artifact — and this number, not the
single-fold Phase 8 estimate, is the real bar every candidate below is measured against.

## RETFound: given a genuinely fair shot, still behind

RETFound (Zhou et al., *Nature* 2023) is a ViT-Large, 303M parameters, MAE-pretrained on 1.6M
retinal images — the obvious foundation-model stretch goal once real GPU compute was available.

**First attempt** used a flat learning rate across the whole backbone: 5-fold CV came back at
**0.8877 ± 0.0140**. Reading RETFound's own reference implementation
(`github.com/rmaphoh/RETFound_MAE`, `util/lr_decay.py`) afterward showed this wasn't actually
their documented recipe — it specifies layer-wise learning-rate decay (`layer_decay=0.65`),
`weight_decay=0.05`, `drop_path=0.2`, and a batch-size-scaled base LR. Implemented properly in
`layer_wise_param_groups`/`_vit_layer_id` (`src/drdetect/grading/model.py`), mirroring the
reference implementation's own parameter-naming and layer-counting logic exactly.

**Proper-recipe 5-fold CV: 0.8917 ± 0.0115** (fold QWKs: 0.9092, 0.8926, 0.8798, 0.8830, 0.8937).
That's only +0.004 over the flat-LR attempt — well inside the noise of either run. **The recipe
fix genuinely closes the question: RETFound wasn't held back by an unfair training setup.** Given
a correct fine-tuning recipe, it still doesn't clear the CNN's 0.9094 ± 0.0079 — the two ranges
(CNN 0.9015–0.9173, RETFound 0.8802–0.9032) barely overlap.

**Decision: RETFound is closed out.** Per this project's own discipline — finalists get decided on
internal validation, external test sets are for tie-breaking, not hopeful long shots — it will not
be sent to DDR or any future external evaluation unless a materially different attempt someday
clears the internal bar first.

## The one real win: a 5-fold ensemble, proven on a new external test set

[`docs/21_PHASE8_ABLATION_RESULTS.md`](21_PHASE8_ABLATION_RESULTS.md) flagged this possibility
without being able to test it: a true ensemble of the 5-fold CV checkpoints can't be evaluated
honestly on any APTOS split, since every fold's model was trained on the other four folds' data —
every APTOS image is "held out" for at most one member and "seen in training" by the other four.
It can only be judged on data none of the five ever trained on. Messidor-2 and IDRiD are already
spent, so this needed a genuinely new external set.

**DDR** (Li et al., *Information Sciences* 2019, Chinese Academy of Sciences, multiple clinical
sites across 23 provinces) fits: the same 5-class ICDR scale as APTOS (no relabelling), a
different population than APTOS/Messidor-2/IDRiD, a pre-existing official test split, and a
permissive CC BY 4.0 licence via the clean re-host `ctmedtech/DDR-dataset`. One real gotcha: DDR's
own label scale is 0–5, not 0–4 — grade 5 is "ungradable," not a severity beyond PDR, and had to
be excluded (`load_ddr_labels`) exactly like `load_messidor2_labels` already excludes Messidor-2's
own non-gradable rows. Final test set: **3,759 images**.

The candidate pool was decided before this ran, per the same discipline as docs/22: the single
shipped checkpoint, and a 5-fold ensemble (raw regression outputs averaged before decoding, the
same principle as this project's own hflip TTA — RETFound was excluded, having already failed to
clear internal validation above).

| | QWK | Sensitivity | Specificity | Referable AUC |
|---|---:|---:|---:|---:|
| **shipped\_cnn** (the single checkpoint deployed at the time) | **0.6399** | **0.340** | 0.995 | 0.8906 |
| **cnn\_5fold\_ensemble** | 0.6362 | 0.305 | **0.998** | **0.8990** |

**DeLong test on referable AUC, same 3,759 images: z=−2.071, p=0.0384.** QWK and sensitivity at
the frozen threshold are roughly a wash — arguably slightly in the single model's favour — but
referable AUC, the metric that matters most for a screening tool (it doesn't depend on where a
threshold happens to sit), is a real, significant win for the ensemble. This is the one genuinely
new, externally-proven improvement from this whole GPU phase.

## What changed as a result

- The live demo now serves the 5-fold ensemble instead of the single checkpoint — `app.py`
  downloads all 5 fold checkpoints and `run_pipeline` averages their outputs and Grad-CAM
  heatmaps before decoding. See the README's "Shipping the live demo" section for how.
- RETFound is closed out (above) — no further GPU time will go toward it on this dataset.
- EyePACS pretraining and 1024px resolution are both closed out as confirmed nulls, not
  unresolved gaps — `scripts/fetch_eyepacs_subset.py`'s fix means the EyePACS path itself now
  actually works, for whenever a future idea wants to reuse pretraining on it.
- DDR (`scripts/fetch_ddr_testset.py`, `scripts/evaluate_ddr.py`) is now this project's second
  external test set, evaluated once, with the same "will not be re-run against a different model
  choice" commitment as Messidor-2/IDRiD.
