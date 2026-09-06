# Model Card — drdetect DR grading model

**Weights**: [huggingface.co/adarshcod30/drdetect-dr-screening](https://huggingface.co/adarshcod30/drdetect-dr-screening)
· **Release**: [GitHub v1.0](https://github.com/adarshcod30/Diabetic-Retinopathy-Detection/releases/tag/v1.0)
(checkpoint + ONNX export, both mirrors of the same file)

Following the structure of Mitchell et al., ["Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993)
(FAT* 2019). Companion to [`DATASET_CARD.md`](DATASET_CARD.md) (what data went in, and its
licensing) and every `docs/NN_PHASE*_RESULTS.md` file this project produced (the full evidence
trail behind every number below).

## Model details

- **Developed by**: Adarsh Dwivedi, as a portfolio/research project — not by or for a company,
  hospital, or regulatory-cleared product line.
- **Architecture**: EfficientNet-B0 backbone (`timm`), ImageNet-initialised, frozen BatchNorm,
  **regression-loss ordinal head** (a single scalar output, thresholded into 5 ICDR grades and a
  referable-DR binary decision).
- **Why regression loss, specifically**: this is not the model with the best *internal*
  validation exact-grade accuracy — a plain cross-entropy baseline scored higher there
  (84.3% vs. 81.3% raw accuracy, [`docs/21_PHASE8_ABLATION_RESULTS.md`](docs/21_PHASE8_ABLATION_RESULTS.md)).
  It is the model that won the **locked external evaluation**, decisively, on referable-DR AUC
  (0.9242 vs. 0.8878, DeLong p=6.1×10⁻¹⁰) — see [`docs/22_PHASE8_VALIDATION_RESULTS.md`](docs/22_PHASE8_VALIDATION_RESULTS.md).
  That single, pre-committed, one-time external comparison is the basis for this choice, not the
  internal number.
- **Input**: a single fundus photograph, RGB, any common format, any resolution (resized to
  512×512 internally after `crop_from_gray → circle_crop → Ben Graham` preprocessing — CLAHE was
  tested and found to make no significant difference, so it is not part of the pipeline).
- **Output**: an ICDR grade (0–4), a referable-DR flag (grade ≥2), a confidence value (raw softmax
  is NOT calibrated for this checkpoint — see "Calibration" below), and optionally (via
  `add_lesion_evidence`) a lesion overlay, an ICDR evidence table, and a templated rationale.
- **Released as**: PyTorch checkpoint + ONNX export (parity-verified, max abs diff 2.4×10⁻⁷),
  under a **research-use-only** license (see "License" below) — distinct from this repository's
  own code, which is MIT.
- **Version**: v1.0, corresponding to `models/checkpoints/sweep_512_regression_fold0/best.ckpt`
  as of this project's Phase 9 (2026-09-07).

## Intended use

- **Primary intended use**: a research prototype and portfolio artefact demonstrating an
  end-to-end, explainable, quality-aware DR screening pipeline for a rural-India screening-camp
  context — not a diagnostic tool.
- **Primary intended users**: researchers, students, and reviewers evaluating this project's
  methodology; a downstream developer building on this codebase.
- **Out-of-scope uses**: **this model must never be used to make or influence an actual clinical
  decision about a real patient without a qualified ophthalmologist reviewing every case.** It has
  not been through any regulatory clearance process (FDA, CDSCO, or otherwise), has not been
  prospectively validated, and — per the results below — does **not** clear this project's own
  referable-DR sensitivity target on external data. It is not validated for any population outside
  the three represented in its training/evaluation data (Indian screening/hospital populations via
  APTOS and IDRiD, French population via Messidor-2), any camera hardware other than what those
  datasets used, or any DR-adjacent condition (e.g. it does not detect diabetic macular edema,
  glaucoma, or other retinal disease).

## Training data

APTOS 2019 (3,662 images, Aravind Eye Hospital, India) — see [`DATASET_CARD.md`](DATASET_CARD.md)
for full provenance. Class distribution: 49.3% grade 0, 10.1% grade 1, 27.3% grade 2, 5.3% grade
3, 8.1% grade 4. Single-grader labels (not panel-adjudicated). No EyePACS/RETFound pretraining —
ImageNet initialisation only (investigated and found not runnable on this project's local compute,
see `docs/07_PHASE3_RESULTS.md`).

## Evaluation data

**Locked external test set, evaluated exactly once**: 1,847 images (1,744 Messidor-2, France,
panel-adjudicated grades; 103 IDRiD official grading-test split, India, hospital population).
Never used for any training, threshold selection, or model-choice decision before this evaluation
ran. See [`docs/22_PHASE8_VALIDATION_RESULTS.md`](docs/22_PHASE8_VALIDATION_RESULTS.md) for the
full protocol.

## Metrics

Quadratic Weighted Kappa (QWK, the standard ordinal-agreement metric for 5-class ICDR grading),
sensitivity/specificity at a referable-DR (grade≥2) operating point **frozen from APTOS internal
validation** (never chosen on the evaluation data), AUROC via DeLong's test for paired comparison,
Expected Calibration Error (ECE), and bootstrap 95% CIs (2,000 resamples) throughout.

## Quantitative analysis — the real, locked-external numbers

| | QWK | 95% CI | Sensitivity | Specificity | ECE | Referable AUC |
|---|---:|---|---:|---:|---:|---:|
| **This model, external (once)** | 0.6995 | [0.6622, 0.7317] | **0.441** | 0.976 | 0.043 | 0.9242 |
| Internal validation (APTOS, same fold) | 0.9147 | [0.8969, 0.9314] | 0.940* | 0.925* | — | 0.9815 |

*internal sensitivity/specificity are reported at the SAME split the threshold was chosen on, so
they are optimistic by construction (`scripts/evaluate.py`'s own standing caveat) — the external
row is the honest number.

**Per-dataset**: Messidor-2 QWK 0.684, sens 0.398, spec 0.975 (n=1,744, referable prevalence
26.2%). IDRiD test QWK 0.718, sens 0.750, spec 1.000 (n=103, referable prevalence 62.1%).

**Per (approximate) quality tier** — sharpness terciles among the test set's own images, NOT a
validated 3-tier quality classifier (see `DATASET_CARD.md`/roadmap Phase 2 note — that classifier
was never built): low-sharpness QWK 0.682/sens 0.431 (n=615); mid-sharpness QWK 0.653/sens 0.325
(n=617); high-sharpness QWK 0.738/sens 0.531 (n=615). Not a monotonic quality effect — the middle
tercile is worse than either extreme for both metrics, on both models tested, and this project
does not have an explanation for why.

**Comparison to published work** (same task, different data/populations — not a like-for-like
benchmark, included for context):

| Study | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|
| Gulshan 2016, Messidor-2 | 96.1% | 93.9% | 0.990 |
| Ting 2017, Singapore | 90.5% | 91.6% | 0.936 |
| IDx-DR 2018, prospective primary care | 87.2% | 90.7% | — |
| Gulshan 2019, Aravind India | 88.9% | 92.2% | 0.963 |
| Gulshan reproduction (cited), Messidor-2 | — | — | 0.853 |
| **this model, Messidor-2 + IDRiD** | **44.1%** | **97.6%** | **0.924** |

## The headline limitation, diagnosed not just disclosed

**Referable-DR sensitivity is 44.1% externally, against a >=90% target.** This is the single
weakest number in this project and must not be minimised. It is, however, a **specific,
diagnosed** failure, not an unexplained one: the operating threshold was frozen from APTOS
internal validation and never re-tuned on Messidor-2/IDRiD (re-tuning after seeing this result
would itself have been a validity violation this project chose not to commit). Checking the
underlying score distributions shows the model's referable-score for truly-referable external
patients sits, on median, an order of magnitude below that frozen threshold — while referable AUC
(0.924) stayed in a defensible range close to published external-validation drops. **This is best
read as a threshold-transfer/calibration failure rather than a collapse in the model's underlying
ability to rank patients by risk.** A deployment on any new population MUST fit its own operating
threshold on a local calibration set before this model's binary referable/non-referable output is
trusted for any decision — using the shipped threshold as-is on a new population is very likely to
under-refer real cases, exactly as it did here.

## Calibration

This checkpoint has **not** been temperature-scaled (Phase 5's calibration work, `docs/09`, was
run on the CE baseline before the Phase 8 model choice was made; ECE is reported above using raw
output). `scripts/calibrate.py` exists and should be re-run against this checkpoint before its
confidence values are shown to any user as if they were probabilities.

## Failure modes

The single worst errors on the external set, for this model and the superseded baseline alike,
are confidently-wrong **grade 0 → grade 4** calls (a healthy retina scored as Proliferative DR at
near-maximum confidence) — the opposite direction from the dominant sensitivity problem above.
One specific Messidor-2 image (`IM002441`) is independently misjudged this way by both models
despite different loss functions — a genuine shared blind spot, not investigated further this
pass. The candidate-generation stage for microaneurysm detection has strong recall (91.7–96.8%,
`docs/15`) but its accept/reject classifier does not generalise well from its curated training
distribution to a real image's true candidate flood (test candidate-AUPRC 0.25) — microaneurysm
evidence in any generated report should be read as under-sensitive, not exhaustive. Grad-CAM
attention never lands on a microaneurysm in any of 81 tested IDRiD images across 3 CAM methods
(`docs/18`) — the visual explanation is not informative for grade-1-defining lesions specifically,
even where the grade itself may be correct.

## Ethical considerations

- **Not a medical device.** No regulatory clearance in any jurisdiction. This notice appears in
  every generated PDF report (`src/drdetect/serve/report.py`'s `_DISCLAIMER`) and every FastAPI
  response (`src/drdetect/serve/api.py`'s `NOT_A_MEDICAL_DEVICE`) — two independently-written but
  consistent disclaimers, not a single shared constant.
- **Population representation**: APTOS and IDRiD are Indian; Messidor-2 is French. This model's
  behaviour on any other population — different ethnicity, camera hardware, or DR prevalence — is
  unverified and should not be assumed to transfer, especially given the calibration failure
  documented above.
- **No patient-identifying data** is processed, stored, or required by this pipeline; EXIF data is
  not read from uploaded images.
- **Human-in-the-loop is mandatory, not optional**, for exactly the reason this card's headline
  limitation describes: a system with 44% sensitivity at its shipped operating point would miss
  more than half of real referable cases if deployed autonomously.

## License

**Code** (this repository): MIT — see [`LICENSE`](LICENSE).

**Model weights**: **research use only**. The weights are derived from training data under mixed
licenses, several of which explicitly restrict redistribution (APTOS/Kaggle competition rules;
Messidor-2's ADCIS terms) — see [`DATASET_CARD.md`](DATASET_CARD.md) for the exact terms per
dataset. Weights may be used and modified for research and educational purposes; they may not be
used in any product offered for clinical, diagnostic, or commercial use, and downstream
redistribution should carry this same restriction and this same card.

## Recommendations for anyone extending this work

1. **Recalibrate the threshold, and ideally temperature-scale, on any new deployment population**
   before trusting the referable-DR binary output — do not reuse this checkpoint's shipped
   threshold on a different camera/population without a local calibration set.
2. **Do not present the calibrated confidence as validated probability** until `scripts/calibrate.py`
   has actually been run against this specific checkpoint.
3. If pursuing regulatory-grade performance, prioritise closing the external-sensitivity gap over
   any further internal-QWK optimisation — this project's own evidence (docs/21) is that the two
   do not always move together, and it is the external number that failed the target.
4. Treat microaneurysm-derived evidence (candidate counts, grade-1 rationale) as a lower bound,
   not an exhaustive count, per the failure-mode note above.
