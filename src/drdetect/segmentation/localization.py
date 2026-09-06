"""IDRiD optic disc / fovea localisation: heatmap regression, not segmentation.

A genuinely different task from lesion or vessel segmentation -- the target
is a single (x, y) point per structure, not a per-pixel mask. Following the
roadmap's own specified approach ("heatmap regression"), each target point
becomes a 2D Gaussian blob at working resolution; the model predicts a
2-channel heatmap (OD, fovea) and inference reads the point back off as the
argmax of each channel.

IDRiD's "C. Localization" task ships a genuine official split -- 413
training / 103 test images, each in its own CSV with (x, y) pixel
coordinates at full (2848x4288) resolution, for both the optic disc and the
fovea. This is the largest, cleanest-labelled subset of IDRiD used in this
project so far (segmentation's hard-exudate harness has 81 images total;
this one has 516).

OD-diameter normalisation: the roadmap's success criterion is "mean
localisation error < 0.5 x OD diameter", but per-image OD diameter isn't
available for the localisation set itself -- only the disjoint 81-image
segmentation set ships OD masks, and the two subsets use different filename
numbering (IDRiD_01..54 vs IDRiD_001..413/103), so they cannot be joined by
ID. The mean OD diameter measured directly from the 54 real segmentation
masks (527.7px at full 2848x4288 resolution) is used as a fixed proxy
constant instead of a per-image value or an invented one.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

__all__ = [
    "IDRiDLocalizationPair",
    "find_idrid_localization_pairs",
    "build_localization_transforms",
    "make_gaussian_heatmap",
    "heatmap_argmax",
    "IDRiDLocalizationDataset",
    "LocalizationModule",
    "MEAN_OD_DIAMETER_PX_FULL_RES",
    "working_res_od_diameter",
    "euclidean",
]

# Measured directly from the 54 real IDRiD segmentation-task OD masks (see
# module docstring for why this is a cross-subset proxy, not a per-image
# ground truth value).
MEAN_OD_DIAMETER_PX_FULL_RES = 527.7
_FULL_RES_H = 2848


@dataclass(frozen=True)
class IDRiDLocalizationPair:
    image_id: str
    image_path: Path
    od_xy: tuple[float, float]
    fovea_xy: tuple[float, float]


def _read_coordinate_csv(path: Path) -> dict[str, tuple[float, float]]:
    """IDRiD's markup CSVs ship ~50 trailing empty columns per row and a
    header row with inconsistent spacing ("X- Coordinate", "Y - Coordinate")
    -- read positionally (columns 0-2), not by header name."""
    coords = {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        coords[row[0].strip()] = (float(row[1]), float(row[2]))
    return coords


def find_idrid_localization_pairs(
    idrid_root: str | Path, split: str
) -> list[IDRiDLocalizationPair]:
    """Args: idrid_root e.g. "data/raw/idrid"; split "train" or "test"."""
    root = Path(idrid_root) / "C. Localization"
    split_dir = "a. Training Set" if split == "train" else "b. Testing Set"
    split_label = "Training Set" if split == "train" else "Testing Set"
    image_dir = root / "1. Original Images" / split_dir

    od_csv = (
        root
        / "2. Groundtruths"
        / "1. Optic Disc Center Location"
        / f"{'a' if split == 'train' else 'b'}. IDRiD_OD_Center_{split_label}_Markups.csv"
    )
    fovea_csv = (
        root
        / "2. Groundtruths"
        / "2. Fovea Center Location"
        / f"IDRiD_Fovea_Center_{split_label}_Markups.csv"
    )
    od_coords = _read_coordinate_csv(od_csv)
    fovea_coords = _read_coordinate_csv(fovea_csv)

    pairs = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = image_path.stem
        if image_id in od_coords and image_id in fovea_coords:
            pairs.append(
                IDRiDLocalizationPair(
                    image_id=image_id,
                    image_path=image_path,
                    od_xy=od_coords[image_id],
                    fovea_xy=fovea_coords[image_id],
                )
            )
    return pairs


def build_localization_transforms(size: tuple[int, int], train: bool) -> A.Compose:
    """`size` is (height, width). Flips are included, unlike segmentation's
    IDRiD transforms sharing the same rotation-invariance reasoning does NOT
    apply here -- flipping DOES change which anatomical side is which, but
    that's fine: the (x, y) keypoints are flipped consistently with the
    image by albumentations' own keypoint handling, and the model is trained
    to predict two independent points, not a rotation-invariant blob."""
    normalise = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    if not train:
        return A.Compose(
            [A.Resize(*size), normalise, ToTensorV2()], keypoint_params=keypoint_params
        )
    return A.Compose(
        [
            A.Resize(*size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            normalise,
            ToTensorV2(),
        ],
        keypoint_params=keypoint_params,
    )


def make_gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """A single-channel float32 Gaussian blob peaking at 1.0 at (cx, cy)."""
    ys, xs = np.mgrid[0:h, 0:w]
    heatmap = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2)))
    return heatmap.astype(np.float32)


def heatmap_argmax(heatmap: np.ndarray) -> tuple[float, float]:
    """(x, y) of a 2D array's peak."""
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(x), float(y)


class IDRiDLocalizationDataset(Dataset):
    def __init__(
        self,
        pairs: list[IDRiDLocalizationPair],
        *,
        size: tuple[int, int] = (512, 768),
        train: bool = True,
        sigma: float = 12.0,
    ):
        self.pairs = pairs
        self.size = size
        self.sigma = sigma
        self.transform = build_localization_transforms(size, train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        keypoints = [pair.od_xy, pair.fovea_xy]

        out = self.transform(image=image, keypoints=keypoints)
        h, w = self.size
        (od_x, od_y), (fov_x, fov_y) = out["keypoints"]

        od_heatmap = make_gaussian_heatmap(h, w, od_x, od_y, self.sigma)
        fovea_heatmap = make_gaussian_heatmap(h, w, fov_x, fov_y, self.sigma)
        heatmaps = torch.from_numpy(np.stack([od_heatmap, fovea_heatmap]))

        targets_xy = torch.tensor([[od_x, od_y], [fov_x, fov_y]], dtype=torch.float32)
        return out["image"], heatmaps, targets_xy


def working_res_od_diameter(size: tuple[int, int]) -> float:
    """MEAN_OD_DIAMETER_PX_FULL_RES rescaled to a `size=(h, w)` working
    resolution, assuming the uniform-scale resize `build_localization_transforms`
    actually performs (height ratio == width ratio, since the working
    aspect ratio is chosen to match IDRiD's real one)."""
    scale = size[0] / _FULL_RES_H
    return MEAN_OD_DIAMETER_PX_FULL_RES * scale


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class LocalizationModule(L.LightningModule):
    """MSE between predicted (sigmoid) heatmaps and target Gaussians --
    standard for heatmap-regression landmark detection. Validation reads the
    point back off each predicted channel's argmax and reports Euclidean
    error in OD-diameter units, the roadmap's own stated success unit.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        working_od_diameter: float = 1.0,
        max_epochs: int = 40,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self._val_pred_heatmaps: list[torch.Tensor] = []
        self._val_target_xy: list[torch.Tensor] = []

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return nn.functional.mse_loss(probs, target_heatmaps)

    def training_step(self, batch, batch_idx):
        x, heatmaps, _ = batch
        logits = self(x)
        loss = self._loss(logits, heatmaps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Localisation loss is {loss.item()} at step {batch_idx}. Diverged.")
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, heatmaps, targets_xy = batch
        logits = self(x)
        loss = self._loss(logits, heatmaps)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_pred_heatmaps.append(torch.sigmoid(logits).detach().float().cpu())
        self._val_target_xy.append(targets_xy.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_pred_heatmaps:
            return
        preds = torch.cat(self._val_pred_heatmaps).numpy()
        targets = torch.cat(self._val_target_xy).numpy()

        od_errors, fovea_errors = [], []
        diameter = self.hparams.working_od_diameter
        for i in range(preds.shape[0]):
            od_pred = heatmap_argmax(preds[i, 0])
            fovea_pred = heatmap_argmax(preds[i, 1])
            od_errors.append(euclidean(od_pred, tuple(targets[i, 0])) / diameter)
            fovea_errors.append(euclidean(fovea_pred, tuple(targets[i, 1])) / diameter)

        od_mean = float(np.mean(od_errors))
        fovea_mean = float(np.mean(fovea_errors))
        self.log("val/od_error_diameters", od_mean, prog_bar=True)
        self.log("val/fovea_error_diameters", fovea_mean, prog_bar=True)
        self.log("val/mean_error_diameters", (od_mean + fovea_mean) / 2.0, prog_bar=True)

        self._val_pred_heatmaps.clear()
        self._val_target_xy.clear()

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
