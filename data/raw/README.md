# Raw datasets — archived here, deleted from disk on 2026-09-08

The four public datasets below sat in this directory throughout Phases 0–8. They were deleted
on **2026-09-08** to reclaim local disk space (`data/raw/` + `data/processed/` +
`data/processed_clahe/` were together consuming ~16 GB, and `models/checkpoints/` another
~17 GB, on a machine whose free disk had dropped from ~60 GB to ~15 GB). Nothing was lost:

- Every dataset here is public and independently re-downloadable from its original source.
- The released model does **not** need this data — it is already published on
  [HuggingFace Hub](https://huggingface.co/adarshcod30/drdetect-dr-screening) and
  [GitHub Releases](https://github.com/adarshcod30/Diabetic-Retinopathy-Detection/releases/tag/v1.0),
  verified present on both immediately before this cleanup.
- This data is only needed to retrain, re-run an ablation, or extend this project — which
  needs a GPU this project does not currently have local access to. It will be re-downloaded
  when that changes.
- The **sha256 hash of every image that was here is still committed** in `../manifests/*.csv`
  (never deleted — see `.gitignore`'s explicit `!data/manifests/*.csv` exception). A
  re-download can be verified byte-identical against what every result in `docs/06` through
  `docs/22` actually trained and evaluated on.

Full provenance, licensing terms, and per-dataset structural detail:
[`../../DATASET_CARD.md`](../../DATASET_CARD.md). Full download commands with context:
[`../../docs/03_TECH_STACK.md`](../../docs/03_TECH_STACK.md) §5.

## What was here

| Dataset | Size on disk | Images | Role in this project | How to get it back |
|---|---:|---:|---|---|
| **APTOS 2019** | 9.5 GB | 3,662 | Primary grading train/val (Phases 1, 3, 5, 8) | `kaggle competitions download -c aptos2019-blindness-detection -p data/raw/aptos` |
| **IDRiD** | 966 MB | 516 grading (413 train/103 test) + 81 segmentation (54 train/27 test) | Lesion segmentation, OD/fovea localisation, CAM ground truth, and (its 103-image grading-test split) part of the **locked external test set** | Free IEEE DataPort account, no manual review, immediate download: <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid> (DOI 10.21227/H25W98) |
| **Messidor-2** | 2.3 GB | 1,748 (4 excluded as ungradable) | **Locked external test set** — evaluated exactly once (`docs/22`), now permanently spent | Two separate steps: images via a personal-info form at <https://www.adcis.net/en/third-party/messidor2/> (manually reviewed, ADCIS terms — do not redistribute); adjudicated grades via `kaggle datasets download -d google-brain/messidor2-dr-grades` |
| **DRIVE** | 29 MB | 40 (20 publicly labelled) | Vessel segmentation, 5-fold CV | Free grand-challenge.org account, register for the DRIVE challenge specifically: <https://drive.grand-challenge.org/> |

`data/processed/` (2.8 GB) and `data/processed_clahe/` (522 MB) were preprocessing caches —
Ben Graham + circle-crop, and a CLAHE variant used for one Phase 8 ablation row (`docs/21`).
Pure derived data, not source data; regenerate once the raw images above are back:

```bash
python scripts/preprocess.py --dataset aptos --size 512
python scripts/preprocess.py --dataset idrid --size 512
python scripts/preprocess.py --dataset idrid --idrid-split test --size 512
python scripts/preprocess.py --dataset messidor2 --size 512
```

## Verifying a re-download matches what this project actually trained on

```bash
sha256sum data/raw/aptos/train_images/*.png | sort > /tmp/new_hashes.txt
# compare image_id -> sha256 against the corresponding column already committed
# in data/manifests/aptos_512.csv (or aptos_224.csv / aptos_768.csv / aptos_1024.csv)
```

**Never subsample any of these when re-downloading.** `docs/05_PROTOTYPE_SCOPE.md` §1.4 is the
recorded reason why — a locked test set below ~200 referable cases has a confidence interval
wide enough to make a sensitivity claim unmakeable, and grade 3 (Severe) has only 193 images in
all of APTOS to begin with.
