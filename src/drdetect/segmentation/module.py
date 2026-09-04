"""Lightning module for lesion segmentation.

BCE + Dice, not plain BCE: at <0.1% positive pixels (measured, see
dataset.py), unweighted BCE's gradient is dominated by the trivial
"predict background everywhere" solution long before it ever learns the
lesion class -- the same class-imbalance failure mode this project already
diagnosed once for CORN's conditional subsets (docs/07_PHASE3_RESULTS.md,
Result 2), showing up again in a different part of the pipeline. Dice loss
is imbalance-robust by construction (it scores overlap ratio, not per-pixel
classification), so combining it with BCE keeps BCE's well-behaved gradients
while Dice supplies the signal BCE alone would drown out.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from drdetect.segmentation.metrics import dice_coefficient, pixel_auprc

__all__ = ["SegmentationModule", "soft_dice_loss"]


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, eps: float = 1.0
) -> torch.Tensor:
    """Differentiable Dice loss on probabilities, not thresholded predictions.

    `eps=1.0` (not 1e-7): with this few positive pixels per patch, a tiny eps
    lets a patch with zero true positives and zero predicted positives produce
    a huge, meaningless gradient from floating-point noise in the ratio. A
    Dice-smoothing epsilon of 1.0 is standard practice for exactly this
    regime, not an arbitrary choice.
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
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

        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self._pos_weight)
        dice = soft_dice_loss(logits, targets)
        return bce + self.hparams.dice_weight * dice

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Segmentation loss is {loss.item()} at step {batch_idx}. Diverged.")
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
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
            auprc = pixel_auprc(targets_np, probs)
            self.log("val/auprc", auprc, prog_bar=True)
            dice = dice_coefficient(targets_np, probs > 0.5)
            self.log("val/dice", dice, prog_bar=True)
        else:
            # A validation split with zero positive patches this epoch --
            # log nothing rather than a fabricated 0.0 that would look like a
            # real (bad) score instead of an undefined one.
            pass

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
