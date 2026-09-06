"""DRIVE vessel segmentation: dataset, model, and FOV-aware Lightning module.

Deliberately separate from dataset.py/module.py (the IDRiD lesion harness),
not sharing a base class with them, for reasons that are architectural, not
just organisational:

  * DRIVE images are small and uniform (584x565) -- full images are trained
    on directly, no patch sampling.
  * Vessels are a measured ~7.5% of pixels, not IDRiD's <0.1% -- AUROC is a
    meaningful metric here (see metrics.pixel_auroc), where it is not for
    lesions.
  * Every DRIVE image has a field-of-view mask, and pixels outside it (the
    black surround around the circular retinal image) are not tissue at
    all -- they must be excluded from both loss and metrics, or a model that
    trivially gets the black border right looks better than it is. IDRiD's
    lesion harness has no equivalent concept.

DRIVE's own public distribution ships vessel ground truth (`1st_manual/`)
only for the 20 "training" images -- the 20 "test" images have no publicly
released annotations (confirmed directly: `data/raw/drive/test/` contains
only `images/` and `mask/`, no manual-annotation directory). All 20 labelled
images are therefore used for k-fold cross-validation; there is no separate
locked test split for this lesion type the way IDRiD has one for hard
exudates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from drdetect.segmentation.metrics import dice_coefficient, pixel_auroc

__all__ = [
    "DriveVesselPair",
    "find_drive_vessel_pairs",
    "DriveVesselDataset",
    "VesselSegmentationModule",
]


@dataclass(frozen=True)
class DriveVesselPair:
    image_id: str
    image_path: Path
    mask_path: Path
    fov_path: Path


def find_drive_vessel_pairs(drive_root: str | Path) -> list[DriveVesselPair]:
    """The 20 DRIVE "training" images -- the only ones with public vessel
    ground truth. Numbered 21-40 in DRIVE's own convention."""
    root = Path(drive_root) / "training"
    image_dir, mask_dir, fov_dir = root / "images", root / "1st_manual", root / "mask"
    pairs = []
    for image_path in sorted(image_dir.glob("*.tif")):
        image_id = image_path.stem.split("_")[0]
        mask_path = mask_dir / f"{image_id}_manual1.gif"
        fov_path = fov_dir / f"{image_id}_training_mask.gif"
        if mask_path.exists() and fov_path.exists():
            pairs.append(
                DriveVesselPair(
                    image_id=image_id, image_path=image_path, mask_path=mask_path, fov_path=fov_path
                )
            )
    return pairs


def _pad_to_multiple(arr: np.ndarray, multiple: int = 32, *, reflect: bool) -> np.ndarray:
    """Pad to a SQUARE size (both axes padded to the same target, from the
    larger of the two), not just each axis to its own nearest multiple.

    DRIVE images are rectangular (584x565) and `RandomRotate90` is one of
    this dataset's augmentations -- a 90 degree turn swaps height and width,
    so a rectangular pad target produces a *different* output shape for a
    rotated image than an unrotated one in the same batch, and
    `torch.stack` cannot collate a batch of mismatched shapes. Padding both
    axes to one shared square target makes every rotation shape-invariant.
    """
    h, w = arr.shape[:2]
    target = -(-max(h, w) // multiple) * multiple  # round up to the next multiple
    ph, pw = target - h, target - w
    border = cv2.BORDER_REFLECT_101 if reflect else cv2.BORDER_CONSTANT
    return cv2.copyMakeBorder(arr, 0, ph, 0, pw, border, value=0)


def build_vessel_transforms(train: bool) -> A.Compose:
    """Same reasoning as the grading and lesion pipelines: flips/rotation are
    label-preserving for a fundus with no canonical orientation; no blur,
    since vessels include capillary-scale structures a blur would erase."""
    normalise = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if not train:
        return A.Compose([normalise, ToTensorV2()], additional_targets={"fov": "mask"})
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            normalise,
            ToTensorV2(),
        ],
        additional_targets={"fov": "mask"},
    )


class DriveVesselDataset(Dataset):
    """One full (padded) DRIVE image per item -- no patch sampling, unlike
    the IDRiD lesion dataset, since these images are already small enough
    to train on whole."""

    def __init__(self, pairs: list[DriveVesselPair], *, train: bool = True):
        self.pairs = pairs
        self.transform = build_vessel_transforms(train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        fov = (cv2.imread(str(pair.fov_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

        # Reflect-pad the image (real content), zero-pad mask/FOV (synthetic
        # border pixels are never vessel and never inside the field of view,
        # so this makes the padding automatically excluded everywhere the
        # FOV mask is already used to exclude the real black surround).
        image = _pad_to_multiple(image, reflect=True)
        mask = _pad_to_multiple(mask, reflect=False)
        fov = _pad_to_multiple(fov, reflect=False)

        out = self.transform(image=image, mask=mask, fov=fov)
        return out["image"], out["mask"].float().unsqueeze(0), out["fov"].float().unsqueeze(0)


class VesselSegmentationModule(L.LightningModule):
    def __init__(
        self,
        model,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        dice_weight: float = 1.0,
        max_epochs: int = 40,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.register_buffer("_pos_weight", torch.tensor(pos_weight))
        self._val_probs: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._val_fov: list[torch.Tensor] = []

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self._pos_weight, reduction="none"
        )
        bce = (bce_per_pixel * fov).sum() / fov.sum().clamp(min=1.0)

        probs = torch.sigmoid(logits) * fov
        targets_in_fov = targets * fov
        intersection = (probs * targets_in_fov).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets_in_fov.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()

        return bce + self.hparams.dice_weight * dice_loss

    def training_step(self, batch, batch_idx):
        x, y, fov = batch
        logits = self(x)
        loss = self._loss(logits, y, fov)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Vessel segmentation loss is {loss.item()} at step {batch_idx}. Diverged."
            )
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, fov = batch
        logits = self(x)
        loss = self._loss(logits, y, fov)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_probs.append(torch.sigmoid(logits).detach().float().cpu())
        self._val_targets.append(y.detach().cpu())
        self._val_fov.append(fov.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_probs:
            return
        probs = torch.cat(self._val_probs).numpy()
        targets = torch.cat(self._val_targets).numpy()
        fov = torch.cat(self._val_fov).numpy().astype(bool)

        in_fov_probs = probs[fov]
        in_fov_targets = targets[fov]
        if in_fov_targets.sum() > 0:
            auroc = pixel_auroc(in_fov_targets, in_fov_probs)
            self.log("val/auroc", auroc, prog_bar=True)
            dice = dice_coefficient(in_fov_targets, in_fov_probs > 0.5)
            self.log("val/dice", dice, prog_bar=True)

        self._val_probs.clear()
        self._val_targets.clear()
        self._val_fov.clear()

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
