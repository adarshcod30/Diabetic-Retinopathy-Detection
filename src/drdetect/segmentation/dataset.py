"""IDRiD lesion segmentation dataset: on-the-fly patch sampling from full-resolution images.

IDRiD images are 2848x4288 -- about 17x the pixel count of the 1024px grading
images this project trains at elsewhere, and holding even one at full
resolution through a segmentation decoder is not something this project's
measured memory budget (docs/05_PROTOTYPE_SCOPE.md) supports. The roadmap's
own plan is "patch sampling at full resolution" (docs/04_ROADMAP.md, Phase 4),
not downscaling -- a hard exudate here is 0.07% of the image by pixel count
(measured directly, IDRiD_41_EX.tif), so shrinking the image shrinks the
lesion out of existence before the model ever sees it.

Patches are drawn fresh every `__getitem__` call, not pre-extracted once,
biased toward lesion-containing regions (`lesion_patch_prob`) -- a uniformly
random 512x512 crop of a 2848x4288 image with <0.1% positive pixels would
return an all-background patch almost every time, and a model trained on
"predict background" batches learns exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

__all__ = [
    "LesionImagePair",
    "find_idrid_lesion_pairs",
    "IDRiDLesionDataset",
    "build_patch_transforms",
]

LESION_FOLDERS = {
    "microaneurysms": "1. Microaneurysms",
    "haemorrhages": "2. Haemorrhages",
    "hard_exudates": "3. Hard Exudates",
    "soft_exudates": "4. Soft Exudates",
    "optic_disc": "5. Optic Disc",
}
# IDRiD's own mask filename suffixes, per lesion type -- not derivable from
# the folder name alone (e.g. "Hard Exudates" -> "_EX", not "_HardExudates").
LESION_SUFFIXES = {
    "microaneurysms": "_MA",
    "haemorrhages": "_HE",
    "hard_exudates": "_EX",
    "soft_exudates": "_SE",
    "optic_disc": "_OD",
}


@dataclass(frozen=True)
class LesionImagePair:
    image_id: str
    image_path: Path
    mask_path: Path


def find_idrid_lesion_pairs(
    idrid_root: str | Path, lesion: str, split: str
) -> list[LesionImagePair]:
    """Match IDRiD images to their mask for one lesion type and split.

    Args:
        idrid_root: e.g. "data/raw/idrid".
        lesion: a key of LESION_FOLDERS.
        split: "train" or "test".
    """
    if lesion not in LESION_FOLDERS:
        raise ValueError(f"unknown lesion {lesion!r}; expected one of {sorted(LESION_FOLDERS)}")
    root = Path(idrid_root)
    split_dir = "a. Training Set" if split == "train" else "b. Testing Set"
    image_dir = root / "A. Segmentation" / "1. Original Images" / split_dir
    mask_dir = (
        root
        / "A. Segmentation"
        / "2. All Segmentation Groundtruths"
        / split_dir
        / LESION_FOLDERS[lesion]
    )

    pairs = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{image_path.stem}{LESION_SUFFIXES[lesion]}.tif"
        if mask_path.exists():
            pairs.append(
                LesionImagePair(
                    image_id=image_path.stem, image_path=image_path, mask_path=mask_path
                )
            )
    return pairs


def build_patch_transforms(patch_size: int, train: bool) -> A.Compose:
    """Same reasoning as drdetect.data.dataset.build_transforms: flips/rotation
    are label-preserving for a fundus with no canonical orientation, and no
    blur/noise, since it would erase small lesions the same way it would
    erase microaneurysms in the grading pipeline."""
    normalise = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if not train:
        return A.Compose([normalise, ToTensorV2()])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            normalise,
            ToTensorV2(),
        ]
    )


class IDRiDLesionDataset(Dataset):
    """One lesion type, patch-sampled on the fly.

    `patches_per_image` fixes an epoch's length to something meaningful for a
    54-image training set -- without it, `len(dataset) == len(pairs)` would
    make one epoch see 54 patches total, most of a 2848x4288 image never
    sampled at all.
    """

    def __init__(
        self,
        pairs: list[LesionImagePair],
        *,
        patch_size: int = 512,
        train: bool = True,
        patches_per_image: int = 20,
        lesion_patch_prob: float = 0.8,
        seed: int = 42,
    ):
        self.pairs = pairs
        self.patch_size = patch_size
        self.train = train
        self.patches_per_image = patches_per_image
        self.lesion_patch_prob = lesion_patch_prob
        self.transform = build_patch_transforms(patch_size, train)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def _load(self, pair: LesionImagePair) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        return image, mask

    def _sample_patch(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        p = self.patch_size
        ys, xs = np.where(mask > 0)
        want_lesion_patch = len(ys) > 0 and self._rng.random() < self.lesion_patch_prob
        if want_lesion_patch:
            i = self._rng.integers(0, len(ys))
            cy, cx = int(ys[i]), int(xs[i])
            top = int(np.clip(cy - p // 2 + self._rng.integers(-p // 4, p // 4 + 1), 0, h - p))
            left = int(np.clip(cx - p // 2 + self._rng.integers(-p // 4, p // 4 + 1), 0, w - p))
        else:
            top = int(self._rng.integers(0, max(h - p, 1)))
            left = int(self._rng.integers(0, max(w - p, 1)))
        return image[top : top + p, left : left + p], mask[top : top + p, left : left + p]

    def __getitem__(self, idx: int):
        pair = self.pairs[idx % len(self.pairs)]
        image, mask = self._load(pair)
        patch_img, patch_mask = self._sample_patch(image, mask)
        out = self.transform(image=patch_img, mask=patch_mask)
        return out["image"], out["mask"].float().unsqueeze(0)
