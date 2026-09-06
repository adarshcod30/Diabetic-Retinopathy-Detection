# Dataset Card — drdetect training and evaluation data

This project trains and evaluates on five public fundus-image datasets. **None of the raw
images are redistributed by this repository** — only code, download instructions
(`scripts/download_data.sh`, `scripts/preprocess.py`), and derived manifests (`data/manifests/*.csv`:
filenames, sha256 hashes, ICDR labels, perceptual-hash grouping — no pixel data). This card exists so
a downstream user knows exactly what they may and may not do with each dataset before they request it
themselves.

## Summary table

| Dataset | Images used | Role in this project | Licence | Redistribution |
|---|---:|---|---|---|
| [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | 3,662 | Primary grading train/val set (Phases 1, 3, 5) | Kaggle competition rules | Research use; do not redistribute images |
| [IDRiD](https://idrid.grand-challenge.org/) | 516 (413 train / 103 test grading) + 81 (segmentation subset) | Lesion segmentation, OD/fovea localisation, CAM localisation ground truth, **locked external test set** (grading, Phase 8) | CC BY 4.0 | Free with attribution |
| [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) | 1,748 | **Locked external test set** (grading, Phase 8) | ADCIS terms (images) + Kaggle-hosted adjudicated grades | Do not redistribute images |
| [DRIVE](https://drive.grand-challenge.org/) | 40 | Vessel segmentation (Phase 4) | Research use, registration required | Do not redistribute images |
| EyePACS / EyeQ | 0 (planned, not used) | Would have been pretraining / quality-model data | Kaggle competition rules | N/A — never downloaded (see below) |

## Why EyePACS/EyeQ were never downloaded

Both are large (EyePACS ~90 GB, EyeQ ~29 GB derived from it) and did not fit this project's
28 GB free-disk budget on a 16 GB development machine (see
[`docs/05_PROTOTYPE_SCOPE.md`](docs/05_PROTOTYPE_SCOPE.md)). EyePACS pretraining was investigated in
Phase 3 and found not runnable on local compute as-is; documented as a real, unclosed gap rather than
silently dropped (see [`docs/07_PHASE3_RESULTS.md`](docs/07_PHASE3_RESULTS.md)). Every model in this
project is trained from ImageNet initialisation, not EyePACS- or RETFound-pretrained weights.

## Per-dataset detail

### APTOS 2019 Blindness Detection

- **Provenance:** fundus photographs collected by Aravind Eye Hospital, India, across multiple
  clinics and camera types, released for the 2019 Kaggle APTOS competition.
- **Labels:** 5-class ICDR grade (0–4), single grader per image (not adjudicated by a panel).
- **Class distribution (this project's training split):** 1805 / 370 / 999 / 193 / 295 for grades
  0–4 (49.3% grade 0) — see [`docs/05_PROTOTYPE_SCOPE.md`](docs/05_PROTOTYPE_SCOPE.md).
- **No patient IDs are shipped.** This project derives grouping for a leakage-safe split via
  perceptual hashing of image structure (`src/drdetect/data/splits.py`), and found ~139
  near-duplicate images this way — documented, not silently ignored.
- **Licence:** Kaggle competition rules — research/competition use; do not redistribute the images
  themselves. This repository ships only `data/manifests/aptos_*.csv` (hashes + labels), never the
  cached JPEGs.

### IDRiD (Indian Diabetic Retinopathy Image Dataset)

- **Provenance:** Nanded, Maharashtra, India — one of the few public datasets with **pixel-level**
  lesion masks (microaneurysms, haemorrhages, hard exudates, soft exudates) in addition to grades.
- **Three sub-tasks used in this project**, each with its own official split:
  - **A. Segmentation** (81 images: 54 train / 27 test) — Phase 4 lesion models, Phase 6 CAM
    localisation. This subset also carries OD/fovea coordinates, used for Phase 4's localisation model.
  - **B. Disease Grading** (516 images: 413 train / 103 test) — the **103-image official test split**
    is this project's IDRiD contribution to the Phase 8 locked external test set. The 413-image
    training split was **not used** by this project (would have required combining two different
    grading populations mid-training, an unplanned change this project did not make).
- **Licence:** CC BY 4.0 — free to use with attribution, including redistribution. This project still
  ships only manifests, not images, for consistency with how the other datasets here are handled.

### Messidor-2

- **Provenance:** Messidor programme, France (LaTIM, Brest / Faculté de Médecine, Angers). Distinct
  from the original Messidor dataset. Requesting it is **two separate steps**: the images (personal-info
  form at ADCIS, manually reviewed) and the adjudicated ICDR grades (hosted separately on Kaggle,
  `google-brain/messidor2-dr-grades`) — see [`docs/03_TECH_STACK.md`](docs/03_TECH_STACK.md) §5.
- **Labels:** adjudicated ICDR grade + DME risk + a `gradable` flag (this project excludes the 4/1748
  images marked ungradable, per `load_messidor2_labels` in `scripts/preprocess.py` — they carry no
  grade to evaluate against, and this is a locked test set, so they are absent, not present as a
  placeholder label).
- **This project's designated locked external test set** (alongside IDRiD's grading test split) —
  **never used for training, threshold selection, or any tuning decision**, evaluated exactly once,
  in Phase 8 (see [`docs/21_PHASE8_VALIDATION_RESULTS.md`](docs/21_PHASE8_VALIDATION_RESULTS.md)).
- **Licence:** ADCIS third-party terms — **do not redistribute the images.** This repository ships
  only `data/manifests/messidor2_512.csv`.

### DRIVE (Digital Retinal Images for Vessel Extraction)

- **Provenance:** a diabetic-retinopathy screening programme in the Netherlands.
- **Labels:** binary vessel segmentation masks, 20 publicly labelled images (the official DRIVE test
  split ships no ground truth at all — this project 5-fold cross-validates on the 20 labelled images
  instead, see [`docs/11_PHASE4_VESSELS_RESULTS.md`](docs/11_PHASE4_VESSELS_RESULTS.md)).
- **Licence:** research use, registration required with the dataset maintainers.

## What downstream users may and may not do

- **May:** use this repository's code, manifests (hashes/labels only), and trained model weights
  (see [`MODEL_CARD.md`](MODEL_CARD.md) for the weights' own research-use-only licence) to reproduce
  or extend this project, provided they independently obtain each raw dataset under its own licence.
- **May not:** obtain raw images through this repository — none are included or linked as direct
  downloads; **may not** redistribute APTOS, Messidor-2, or DRIVE images obtained via this project's
  download scripts; **may** redistribute IDRiD images with attribution (CC BY 4.0), independent of
  this project.
- **Populations represented:** APTOS and IDRiD are Indian screening populations (this project's
  actual deployment target); Messidor-2 is French; DRIVE is Dutch. Performance on any other
  population — different camera hardware, different retinopathy prevalence, different ethnicity — is
  **not evaluated by this project** and should not be assumed to transfer. See
  [`MODEL_CARD.md`](MODEL_CARD.md) for how this shapes the model's own intended-use statement.
