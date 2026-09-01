"""Torch dataset over the preprocessed cache.

Reads the 512 px cache written by scripts/preprocess.py, not the raw images.
The expensive work (crop, Ben Graham, resize) happened once; the training loop
only decodes a small JPEG and augments.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from drdetect.data.manifest import ImageRecord, read_manifest

__all__ = ["FundusDataset", "build_transforms", "load_split"]


class FundusDataset(Dataset):
    """Fundus images with ICDR grades.

    Args:
        records: manifest rows for this split.
        root: directory the manifest's `path` fields are relative to.
        transform: Albumentations transform. Must include normalisation and
            ToTensorV2 -- this class does no implicit conversion, so what the
            model receives is always explicit in the config.
    """

    def __init__(self, records: list[ImageRecord], root: str | Path, transform=None):
        self.records = records
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = self.root / rec.path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"Could not read {path}. Re-run scripts/preprocess.py, or verify the "
                f"cache against the manifest with drdetect.data.manifest.verify_manifest."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, rec.label

    @property
    def labels(self) -> np.ndarray:
        return np.array([r.label for r in self.records])

    @property
    def groups(self) -> list[str]:
        return [r.group_id for r in self.records]


def build_transforms(size: int, train: bool):
    """Augmentations.

    Train-time choices and why:
      * flips and rotation -- a fundus has no canonical orientation, and left vs
        right eyes are mirror images, so these are label-preserving;
      * mild brightness/contrast -- models the illumination variation that
        portable cameras genuinely produce in the field;
      * NO heavy blur or noise -- a microaneurysm is 1-3 px at this resolution,
        and those augmentations would erase the very feature that defines
        grade 1.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # ImageNet statistics, because the backbone is ImageNet-pretrained.
    normalise = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    if not train:
        return A.Compose([A.Resize(size, size), normalise, ToTensorV2()])

    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            normalise,
            ToTensorV2(),
        ]
    )


def load_split(
    manifest_path: str | Path,
    *,
    fold: int = 0,
    n_splits: int = 5,
    seed: int = 42,
    allow_ungrouped: bool = False,
) -> tuple[list[ImageRecord], list[ImageRecord], str]:
    """Split a manifest into (train, val, strategy) for one fold.

    Grouping comes from the manifest's `group_id`, which scripts/preprocess.py
    populates by perceptual hashing. Leakage is asserted, not assumed.
    """
    from drdetect.data.splits import assert_no_group_leakage, stratified_group_split

    records = read_manifest(manifest_path)
    labels = [r.label for r in records]
    groups = [r.group_id for r in records]

    folds, strategy = stratified_group_split(
        labels, groups, n_splits=n_splits, seed=seed, allow_ungrouped=allow_ungrouped
    )
    assert_no_group_leakage(folds, groups)

    val_idx = set(folds[fold].tolist())
    train_records = [r for i, r in enumerate(records) if i not in val_idx]
    val_records = [r for i, r in enumerate(records) if i in val_idx]
    return train_records, val_records, strategy
