"""Microaneurysm detection: candidate generation + small-CNN classification.

The roadmap (docs/04_ROADMAP.md, Phase 4) is explicit that this lesion type
does *not* use the same DeepLabV3+ pixel-segmentation harness as the other
four: "morphological top-hat + matched filter candidate generation -> small
CNN classifier. Do not expect plain segmentation to work here." Measured
directly from all 81 IDRiD microaneurysm masks (train+test): 3,497 instances,
equivalent-diameter median 17.3px, p5 10.0px, p95 29.9px at full (2848x4288)
resolution -- resolvable at full res, but this project's own Phase 3 findings
(docs/07_PHASE3_RESULTS.md) already established that lesions this small are
destroyed by the working resolutions (384-1024px) used elsewhere in this
project, and a patch-sampled decoder trained at 512px (the other lesions'
harness) would be learning from images where a microaneurysm is ~2px wide.

Candidate generation runs a black top-hat (highlights small dark blobs
against a brighter surround -- microaneurysms are dark red dots) on the green
channel at full resolution, where the classes of interest are still several
pixels wide. This is a classical, deterministic, untrained step -- it proposes
candidates, it does not decide; the small CNN afterwards is what actually
distinguishes a true microaneurysm from a vessel cross-section, noise speckle,
or the countless other things a black top-hat also lights up.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = [
    "Candidate",
    "tophat_response",
    "find_candidates",
    "label_candidates",
    "generation_recall",
    "extract_patch",
    "build_candidate_examples",
    "PatchDataset",
    "PatchClassifier",
    "score_candidates",
    "ClassifierModule",
]

DEFAULT_PATCH_SIZE = 33


@dataclass(frozen=True)
class Candidate:
    cx: float
    cy: float
    area: int


def tophat_response(image_bgr: np.ndarray, *, disk_radius: int = 16) -> np.ndarray:
    """Black top-hat (closing - image) on the median-denoised green channel.

    `disk_radius=16` (33px-diameter structuring element): the disk must be
    larger than a microaneurysm for the closing step to erase it and leave a
    peak in the top-hat response -- 33px comfortably exceeds the measured p95
    equivalent diameter (29.9px) without being so large it also erases small
    real vessel gaps, which would flood the candidate set with vessel debris.
    """
    green = image_bgr[:, :, 1]
    green = cv2.medianBlur(green, 3)
    ksize = 2 * disk_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)


def find_candidates(
    image_bgr: np.ndarray,
    *,
    disk_radius: int = 16,
    response_percentile: float = 95.0,
    min_area: int = 3,
    max_area: int = 1500,
    min_gray: float = 15.0,
) -> list[Candidate]:
    """Percentile-threshold the top-hat response, then connected-component it
    into candidate blobs.

    **Not Otsu, measured, not assumed**: Otsu was the first thing tried here,
    and it fails badly on this response. Checked directly on IDRiD_01: Otsu
    picks a threshold of 4 (out of a max response of 71), marking 33.8% of
    the *entire image* as foreground -- because the top-hat response over a
    12.2-megapixel fundus image is a skewed, long-tailed distribution
    dominated by low-level texture and JPEG noise, not the clean bimodal
    signal Otsu assumes. The 33.8% "foreground" merges into huge connected
    blobs (one component covered 1.76M px), which then fail `max_area` and
    get discarded entirely -- taking any real microaneurysm sitting inside
    them down with it. Measured recall on IDRiD_01 with Otsu: 3/18 true
    instances. A fixed percentile of the response's own distribution sidesteps
    the bimodality assumption entirely and adapts per-image; swept on 5 real
    training images (IDRiD_01/02/10/20/33), the 95th percentile recovered
    91-100% of true instances per image (100%, 97.6%, 92.1%, 91.3%, 93.7%)
    vs. Otsu's 16.7% on IDRiD_01 (3/18) -- at a candidate count Otsu actually
    made *worse*, not better, despite being far more permissive.

    `min_area=3, max_area=1500`: measured true-instance area at full res
    ranges 13-2231px (median 234, p95 704) -- the lower bound is set below
    the smallest real instance to avoid discarding true microaneurysms before
    the classifier even sees them (a generation-stage miss can never be
    recovered later; a false positive can still be rejected), the upper
    bound well above p95 to keep merged/clustered annotations reachable while
    still rejecting obviously-too-large blobs (a stray haemorrhage or vessel
    segment surviving the top-hat).

    `min_gray=15.0`: IDRiD images carry a near-black margin outside the
    circular fundus; without this check, JPEG ringing at that boundary
    produces top-hat "candidates" that are pure imaging artifact, not tissue.
    """
    response = tophat_response(image_bgr, disk_radius=disk_radius)
    threshold = float(np.percentile(response, response_percentile))
    binary = (response > threshold).astype(np.uint8) * 255
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    candidates = []
    for i in range(1, n):  # 0 is the background component
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        if gray[int(round(cy)), int(round(cx))] < min_gray:
            continue
        candidates.append(Candidate(cx=float(cx), cy=float(cy), area=area))
    return candidates


def label_candidates(candidates: list[Candidate], mask: np.ndarray) -> np.ndarray:
    """True where a candidate's centroid lands on a positive mask pixel."""
    labels = np.zeros(len(candidates), dtype=bool)
    h, w = mask.shape[:2]
    for i, c in enumerate(candidates):
        x, y = int(round(c.cx)), int(round(c.cy))
        if 0 <= y < h and 0 <= x < w:
            labels[i] = mask[y, x] > 0
    return labels


def generation_recall(candidates: list[Candidate], mask: np.ndarray) -> tuple[int, int]:
    """(instances with >=1 candidate landing inside them, total true instances).

    This is the ceiling end-to-end recall can ever reach: the classifier only
    accepts or rejects candidates the generator already proposed, so a true
    microaneurysm the generator never proposed is unrecoverable downstream.
    Reported separately from the classifier's own recall so a low end-to-end
    number can be attributed to the right stage.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    n_components, comp_labels = cv2.connectedComponents(binary_mask, connectivity=8)
    n_true = n_components - 1
    if n_true == 0:
        return 0, 0
    h, w = mask.shape[:2]
    recovered_ids: set[int] = set()
    for c in candidates:
        x, y = int(round(c.cx)), int(round(c.cy))
        if 0 <= y < h and 0 <= x < w:
            comp_id = int(comp_labels[y, x])
            if comp_id > 0:
                recovered_ids.add(comp_id)
    return len(recovered_ids), n_true


def extract_patch(image_bgr: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """A `size`x`size` crop centred on (cx, cy), reflect-padded at borders
    (a candidate near the image edge is common -- fundus lesions don't avoid
    the periphery -- and reflect padding keeps the patch's local texture
    statistics sane instead of introducing a hard black edge a real
    microaneurysm patch would never have).

    Pads only the small overhang beyond the image edge, not the whole image.
    An earlier version called `cv2.copyMakeBorder` on the full `image_bgr`
    first and cropped afterwards -- measured directly, that allocates a full
    ~36MB padded copy of the *entire* 2848x4288 image on every single call,
    and this is called once per kept candidate (positives plus up to
    hundreds of negatives, per image). Across even a handful of images that
    drove this project's development machine to a 39GB peak memory
    footprint and an aborted process; cropping first and padding only the
    (at most `size`-pixel) overhang produces bit-identical output at a
    fraction of a percent of the cost.
    """
    h, w = image_bgr.shape[:2]
    half = size // 2
    x, y = int(round(cx)), int(round(cy))
    top, left = y - half, x - half
    bottom, right = top + size, left + size

    top_clip, left_clip = max(top, 0), max(left, 0)
    bottom_clip, right_clip = min(bottom, h), min(right, w)
    crop = image_bgr[top_clip:bottom_clip, left_clip:right_clip]

    pad_top, pad_left = top_clip - top, left_clip - left
    pad_bottom, pad_right = bottom - bottom_clip, right - right_clip
    if pad_top or pad_left or pad_bottom or pad_right:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
    return crop


def build_candidate_examples(
    pairs,
    *,
    disk_radius: int = 16,
    response_percentile: float = 95.0,
    patch_size: int = DEFAULT_PATCH_SIZE,
    negative_ratio: float = 10.0,
    max_negatives_per_image: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run candidate generation + ground-truth matching across a list of
    `LesionImagePair`s (see drdetect.segmentation.dataset), returning
    `(patches[N,size,size,3] uint8 BGR, labels[N] int64, stats)`.

    Negatives are heavily subsampled per image: a full-resolution top-hat
    routinely proposes far more spurious blobs than there are true
    microaneurysms in an image, and keeping every one would both blow up
    memory and swamp the true positives the way naive whole-image BCE
    already does for pixel segmentation (drdetect.segmentation.module). Kept
    at `negative_ratio` times that image's positive count, floored at 50 so
    an image with zero recovered positives still contributes hard-negative
    signal rather than none, capped at `max_negatives_per_image` so no single
    image dominates the pool.
    """
    rng = np.random.default_rng(seed)
    all_patches: list[np.ndarray] = []
    all_labels: list[bool] = []
    total_recovered, total_true = 0, 0
    per_image = []

    for pair in pairs:
        image = cv2.imread(str(pair.image_path))
        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
        candidates = find_candidates(
            image, disk_radius=disk_radius, response_percentile=response_percentile
        )
        labels = label_candidates(candidates, mask)
        recovered, n_true = generation_recall(candidates, mask)
        total_recovered += recovered
        total_true += n_true

        pos_idx = np.flatnonzero(labels)
        neg_idx = np.flatnonzero(~labels)
        n_neg_keep = min(len(neg_idx), max(50, int(len(pos_idx) * negative_ratio)))
        n_neg_keep = min(n_neg_keep, max_negatives_per_image)
        neg_keep = (
            rng.choice(neg_idx, size=n_neg_keep, replace=False) if n_neg_keep > 0 else neg_idx[:0]
        )
        keep_idx = np.concatenate([pos_idx, neg_keep])

        for idx in keep_idx:
            c = candidates[idx]
            all_patches.append(extract_patch(image, c.cx, c.cy, patch_size))
            all_labels.append(bool(labels[idx]))

        per_image.append(
            {
                "image_id": pair.image_id,
                "n_candidates": len(candidates),
                "n_positive_candidates": int(labels.sum()),
                "n_recovered_instances": recovered,
                "n_true_instances": n_true,
                "n_examples_kept": int(len(keep_idx)),
            }
        )
        # Measured directly: candidate generation on one full-resolution
        # (2848x4288) IDRiD image peaks around 640MB (connectedComponentsWithStats'
        # internal bookkeeping over tens of thousands of components, on top of
        # the image/response/binary/labels/gray arrays it holds at once). That
        # is fine for one image; left to Python's own GC timing across a full
        # train+val pair list it measurably compounded into severe system-wide
        # memory pressure on this project's 16GB development machine. An
        # explicit del + collect between images is cheap insurance against that.
        del image, mask, candidates, labels
        gc.collect()

    if all_patches:
        patches = np.stack(all_patches).astype(np.uint8)
    else:
        patches = np.zeros((0, patch_size, patch_size, 3), dtype=np.uint8)
    labels_arr = np.array(all_labels, dtype=np.int64)

    stats = {
        "n_images": len(pairs),
        "n_true_instances": total_true,
        "n_recovered_instances": total_recovered,
        "generation_recall": (total_recovered / total_true) if total_true else float("nan"),
        "n_candidates_total": int(sum(p["n_candidates"] for p in per_image)),
        "n_positive_candidates": int(labels_arr.sum()),
        "n_examples_kept": int(len(labels_arr)),
        "per_image": per_image,
    }
    return patches, labels_arr, stats


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PatchDataset(Dataset):
    """Pre-extracted candidate patches, normalised on the fly.

    Patches are tiny (33x33x3 uint8 by default, ~3KB each) and already
    extracted by `build_candidate_examples` -- unlike `IDRiDLesionDataset`,
    there's no full-resolution image I/O left to amortise per `__getitem__`,
    so a plain in-memory array is simpler than another on-the-fly sampler.
    """

    def __init__(self, patches: np.ndarray, labels: np.ndarray, *, train: bool = True):
        self.patches = patches
        self.labels = labels
        self.train = train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        if self.train:
            if np.random.rand() < 0.5:
                patch = patch[:, ::-1]
            if np.random.rand() < 0.5:
                patch = patch[::-1, :]
            patch = np.rot90(patch, k=int(np.random.randint(4)))
        rgb = np.ascontiguousarray(patch[:, :, ::-1]).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return tensor, label


class PatchClassifier(nn.Module):
    """Small CNN: a candidate patch in, one accept/reject logit out.

    Deliberately tiny (~26K params) next to the 22.4M-param DeepLabV3+ used
    for pixel segmentation elsewhere in this project (drdetect.segmentation.
    model) -- the roadmap specifies a "small CNN classifier" because the
    input here is a fixed, already-localised, tiny patch needing only a yes/no
    answer, not a full image needing spatial decoding back to pixel masks.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features).squeeze(-1)


def score_candidates(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    model: nn.Module,
    *,
    device: str = "cpu",
    disk_radius: int = 16,
    response_percentile: float = 95.0,
    patch_size: int = DEFAULT_PATCH_SIZE,
    batch_size: int = 256,
) -> dict:
    """Run generation + classification on one image end to end, in batches --
    used by scripts/evaluate_microaneurysms.py once per official test image.

    Unlike `build_candidate_examples` (which subsamples negatives for
    training), this keeps *every* candidate: evaluation needs the true,
    unsampled precision/recall/false-positive-rate the deployed pipeline
    would actually see, not a class-balanced training view of it. Patches are
    extracted and scored one batch at a time rather than materialising every
    candidate patch for the image up front -- a single full-resolution IDRiD
    image can propose tens of thousands of candidates.
    """
    candidates = find_candidates(
        image_bgr, disk_radius=disk_radius, response_percentile=response_percentile
    )
    labels = label_candidates(candidates, mask)
    recovered, n_true = generation_recall(candidates, mask)

    scores = np.zeros(len(candidates), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            patches = np.stack([extract_patch(image_bgr, c.cx, c.cy, patch_size) for c in batch])
            rgb = patches[:, :, :, ::-1].astype(np.float32) / 255.0
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).float().to(device)
            logits = model(tensor)
            scores[start : start + len(batch)] = torch.sigmoid(logits).cpu().numpy()

    return {
        "candidates": candidates,
        "scores": scores,
        "labels": labels,
        "n_recovered": recovered,
        "n_true": n_true,
    }


class ClassifierModule(L.LightningModule):
    """BCE-only (no Dice term): unlike pixel segmentation, there is no
    spatial overlap to score here -- each example is one candidate, one
    label, so plain weighted BCE is the whole loss, with `pos_weight` set
    from the same-style empirical measurement used throughout this project
    (the true ratio in the actually-sampled, negative-subsampled candidate
    pool, not a theoretical one)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        max_epochs: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.register_buffer("_pos_weight", torch.tensor(pos_weight))
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self._pos_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Classifier loss is {loss.item()} at step {batch_idx}. Diverged.")
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self._pos_weight)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_logits.append(logits.detach().float().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets)
        probs = torch.sigmoid(logits).numpy()
        targets_np = targets.numpy()

        if targets_np.sum() > 0:
            # pixel_auprc is a generic ravel + average_precision_score despite
            # its name -- reused as-is rather than duplicated for candidates.
            from drdetect.segmentation.metrics import pixel_auprc

            self.log("val/auprc", pixel_auprc(targets_np, probs), prog_bar=True)
            self.log("val/accuracy", float(((probs > 0.5) == (targets_np > 0.5)).mean()))
        self._val_logits.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
