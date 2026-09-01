# Prototype Scope — Scaling Down Without Breaking the Science

> **Decision record.** Considered: subsampling the datasets to 0.1–0.5× to fit local
> compute and disk. **Rejected in that form.** Adopted instead: keep all task data,
> scale *compute*. This document records the arithmetic behind that call so it is not
> relitigated later.

---

## 1. The proposal and why it was re-aimed

The original idea — "train on 10–20k of 140k, like OpenForensics" — is sound practice in
general. It does not transfer here, for three measurable reasons.

### 1.1 APTOS is already prototype-sized

OpenForensics has 140,000 images. **APTOS has 3,662.** It is already ~1/38th the scale.
The property that made the OpenForensics cut safe — abundant redundant examples per class
— is absent.

### 1.2 Disk was never the constraint for the task datasets

Preprocessing caches to 512 px JPEG (~70 KB/image), applied once:

| Dataset | Images | Raw | Cached @512 px | Shrink |
|---|---:|---:|---:|---:|
| APTOS | 3,662 | 10.0 GB | **0.24 GB** | 41× |
| IDRiD | 516 | 2.5 GB | **0.03 GB** | 73× |
| Messidor-2 | 1,748 | 5.0 GB | **0.12 GB** | 43× |
| DRIVE | 40 | 0.1 GB | **0.00 GB** | 37× |
| **Total** | | **17.6 GB** | **0.40 GB** | |

Raw fits in the 28 GB budget (tightly). After caching, delete raw and **27+ GB stays free**.

**Only EyePACS (88,702 imgs / ~90 GB) and EyeQ (28,792, a subset of it) genuinely do not
fit.** They are pretraining and auxiliary data — not task data.

### 1.3 Rare classes collapse before the dataset looks small

APTOS class counts are 1805 / 370 / 999 / 193 / 295 for grades 0–4. Training split (80 %):

| Scale | Grade 1 | Grade 3 | Grade 4 | Verdict |
|---|---:|---:|---:|---|
| 1.00× | 296 | 154 | 236 | fine |
| 0.50× | 148 | **77** | 118 | marginal |
| 0.25× | 74 | **38** | 59 | not viable |
| 0.10× | 29 | **15** | 23 | not viable |

Grade 3 (Severe NPDR) is where the 4-2-1 rule lives — the clinically urgent cases. Training
it on 15–38 examples produces a number, not a model.

### 1.4 The test set has a hard statistical floor

Wilson 95 % CI for an observed sensitivity of 0.90:

| Referable cases in test | 95 % CI | Half-width | Verdict |
|---:|---|---:|---|
| 50 | [78.6, 95.7] | ±8.5 pp | useless |
| 100 | [82.6, 94.5] | ±6.0 pp | weak |
| 200 | [85.1, 93.4] | ±4.2 pp | defensible |
| ~450 (full Messidor-2) | [86.9, 92.4] | ±2.8 pp | strong |

Shrinking the test set does not weaken the claim — it makes it **unmakeable**. Test data is
inference-only and costs minutes. **Never subsample it.**

---

## 2. The governing principle

For a fixed budget `C = images × epochs × cost_per_image`, more unique images with fewer
epochs beats fewer images with more epochs — especially with 193 irreplaceable grade-3
examples.

> **Subsampling discards data permanently. Cutting resolution, backbone width, and fold
> count discards compute you can buy back with one Kaggle run.** Same speedup, reversible
> instead of destructive.

---

## 3. Tier-P — the prototype specification

| Dimension | Full plan | **Tier-P** | Speedup | Reversible? |
|---|---|---|---:|---|
| APTOS train | 3,662 | **3,662 (unchanged)** | 1× | — |
| IDRiD (516 / 81 masks) | full | **unchanged** — already at floor | 1× | — |
| DRIVE (40) | full | **unchanged** — already at floor | 1× | — |
| Messidor-2 test | 1,748 | **unchanged** — CI floor | 1× | — |
| **EyePACS pretrain** | 88,702 | **12–15k stratified, cached on Kaggle** | — | ✅ |
| **EyeQ quality** | 28,792 | **~8k stratified subset** | — | ✅ |
| Backbone | EfficientNetV2-S / RETFound | **EfficientNet-B0** | 3× | ✅ |
| Input resolution | 768 px | **512 px** | 2.25× | ✅ |
| Cross-validation | 5-fold + full TTA | **single stratified split + hflip TTA** | 5× | ✅ |
| Lesion classes | MA, HE, SE, EX | **EX + HE first**, MA detector if time | ~2× | ✅ |
| Ablation rows | 11 | **6** | ~2× | ✅ |

**Net ≈ 34× cheaper on the grading arm, with zero task data discarded.**

### 3.1 Why 512 px is a floor, not a preference

A microaneurysm is ~10 px on a 4288 px image; at 512 px it is ~1.2 px, at 384 px ~0.9 px,
at 224 px it is gone. ICDR grade 1 is *defined* as microaneurysms only. Dropping below
512 px does not make grade 1 harder — it removes it from the input. See
[`01_PROJECT_ANALYSIS.md` § 2.2](01_PROJECT_ANALYSIS.md).

### 3.2 How to get EyePACS without downloading 90 GB

1. Open a Kaggle notebook with the EyePACS competition data attached (already mounted, no download).
2. Stratified-sample ~15k images preserving the grade distribution.
3. Apply `preprocess(size=512)` and write out the cache (~1 GB).
4. Publish as a private Kaggle Dataset; download that.

Local disk touched: ~1 GB instead of 90 GB.

**Alternative:** community pre-resized EyePACS datasets already exist on Kaggle. Verify the
resize method before use — a naive resize to 224 px has already destroyed the microaneurysms.

---

## 4. What Tier-P can and cannot claim

### Survives the scale-down

These are **paired** comparisons on the same test set. McNemar's test operates on discordant
pairs only, so it is far more sample-efficient than an absolute proportion CI — the ablation
stays statistically valid at prototype scale.

- Does quality gating improve downstream metrics? ✅
- Does lesion-feature fusion beat the grading-only baseline? ✅
- Does temperature scaling reduce ECE? ✅
- Does Ben Graham preprocessing help? ✅
- **Grad-CAM localisation vs IDRiD masks** ✅ — depends on IDRiD's 81 masks, which are
  unchanged. The project's main differentiator is entirely unaffected.
- The integrated-vs-single-technique ablation ✅

### Does not survive

- "Matches Gulshan's 97.5 % sensitivity" — absolute claims need the full test set and CIs
- Confident Grade 3 vs Grade 4 discrimination — too few examples either way
- Any claim about camera types or populations not in the training data

### The honest framing

> A pilot study demonstrating an integrated, explainable screening pipeline. Relative effects
> of each pipeline component are established with paired significance tests; absolute
> performance is reported with confidence intervals and treated as a lower bound pending
> full-scale training.

That is a legitimate and publishable position. It is *stronger* than an inflated absolute
number, because it is defensible under scrutiny.

---

## 5. Scale-up path

Tier-P is a set of config overrides, not a fork. Every cut reverses on Kaggle:

```bash
# Tier-P (local, overnight on M4)
python scripts/train.py experiment=grading model=effnet_b0 data.image_size=512 data.folds=1

# Tier-F (Kaggle, full) -- same code, same data, different config
python scripts/train.py experiment=grading model=effnetv2_s data.image_size=768 data.folds=5
```

Because no data was discarded, Tier-F requires no re-collection and no re-splitting — the
same manifests and the same locked test set carry over, so results remain directly comparable.

---

## 6. First action: measure, do not estimate

Runtime figures above are estimates. Before committing to Tier-P, benchmark the real
throughput on this machine:

```bash
python scripts/benchmark_device.py --backbone efficientnet_b0 --sizes 384,512,768 --batch 4,8,16
```

Report images/sec for forward and forward+backward on MPS and CPU, plus peak memory. If MPS
throughput at 512 px turns out worse than expected, the next cut is **folds and ablation rows**,
then backbone — **not** training data.
