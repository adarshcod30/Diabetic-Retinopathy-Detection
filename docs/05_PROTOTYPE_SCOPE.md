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

## 6. Measured throughput (Apple M4, 16 GB, torch 2.13, MPS)

Run 2026-09-02 via `scripts/benchmark_device.py`, EfficientNet-B0, ~6.1 GB RAM free
(Chrome and other apps open), projecting 2,930 APTOS training images x 40 epochs:

| Resolution | Batch | Fwd img/s | Train img/s | Peak GB | Per epoch | **Full run** |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 4 | 99.1 | 22.8 | 2.19 | 2.1 min | **86 min** |
| 384 | 8 | — | — | — | — | skipped, over budget |
| **512** | **4** | **59.5** | **14.1** | **2.27** | **3.5 min** | **2.3 h** |
| 512 | 8 | — | — | — | — | skipped, over budget |

**Verdict: Tier-P is comfortably feasible locally.** A full 40-epoch run at the
Tier-P configuration (512 px, EfficientNet-B0) takes **~2.3 hours** — an afternoon,
not an overnight job. Six ablation rows is roughly 14 hours of compute, easily
spread across a few days.

### 6.1 Two constraints the measurement exposed

**Batch size is capped at 4 by free RAM, not by the GPU.** Free RAM during the run
was 6.1 GB of 16 GB — the rest was Chrome, the editor, and other applications.
Closing them raises the budget and allows batch 8. The GPU is not the bottleneck;
*everything else running on the machine* is.

**Batch 4 + BatchNorm needs handling.** EfficientNet uses BatchNorm, whose running
statistics are estimated per micro-batch. At batch 4 those statistics are noisy, and
gradient accumulation does **not** fix it (accumulation batches the optimiser step,
not the normalisation). Options, in order of preference:

1. **Freeze BN** in the pretrained backbone and use ImageNet running statistics —
   simplest, standard for small-batch fine-tuning.
2. **ConvNeXt-Tiny instead of EfficientNet** — LayerNorm has no batch dependence at
   all, so batch 4 is not a statistical problem. Costs more memory per image.
3. Close other applications and train at batch 8–16.

This is a genuine design input that only appeared once throughput was measured.

### 6.2 A note on the memory guard

The first version of this benchmark swept up to 768 px x batch 16 and exhausted
system memory, triggering macOS's "out of application memory" dialog. The cause is
worth recording, because it will recur in training code:

> On CUDA, an over-large batch raises a catchable `OutOfMemoryError`. On MPS there
> is no separate VRAM; the allocator's default high-watermark ratio is **1.7**, so
> it may request ~1.7x physical RAM and macOS honours that by swapping other
> applications out. No Python exception is raised — the *system* fails instead.
> `try/except RuntimeError` is not protection on Apple Silicon.

The script now sets `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7` and
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` before importing torch (both are required —
PyTorch rejects `high < low`, and the default low is 1.4), estimates memory
analytically as `fixed_overhead + batch x (size/224)^2 x per_image`, and skips
configs over budget before allocating anything.

**Apply the same watermark settings in `scripts/train.py`.** A long training run
that quietly swaps is worse than one that fails fast.

---

## 7. First action: measure, do not estimate

Runtime figures above are estimates. Before committing to Tier-P, benchmark the real
throughput on this machine:

```bash
make bench
```

Defaults are deliberately conservative (384/512 px, batch 4/8, budget = 60 % of free RAM
capped at 6 GB). Large configurations are opt-in via `--budget-gb`, and should preferably be
measured on Kaggle rather than locally.

If throughput at 512 px is worse than expected, the cut order is **folds → ablation rows →
backbone** — **never** training data.
