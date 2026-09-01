"""Lightning module for DR grading.

Phase 1 is a plain cross-entropy baseline. It is deliberately un-tuned: its job
is to produce a number to beat, and every later improvement is measured as a
delta against it.

Per-class recall is logged every epoch, not only aggregate QWK. On APTOS, grade
3 is 5.3% of the data and grade 1 is 10.1%; a model can post a respectable QWK
while completely failing those classes, and aggregate metrics will not say so.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from drdetect.eval.metrics import (
    binary_scores,
    quadratic_weighted_kappa,
    referable_labels,
)

__all__ = ["GradingModule"]

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


class GradingModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        num_classes: int = 5,
        class_weights: list[float] | None = None,
        warmup_epochs: int = 1,
        max_epochs: int = 40,
    ):
        super().__init__()
        # `model` is a live module; saving it into the checkpoint hyperparameters
        # would serialise the whole network twice.
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes

        weight = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_logits.append(logits.detach().float().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets).numpy()
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)

        self.log("val/qwk", quadratic_weighted_kappa(targets, preds), prog_bar=True)
        self.log("val/acc", float((preds == targets).mean()))

        # Referable DR (grade >= 2) at the default 0.5 operating point. The
        # calibrated, sensitivity-targeted threshold is chosen in Phase 5 -- this
        # is only a progress signal, not the reported operating point.
        y_ref = referable_labels(targets)
        p_ref = probs[:, 2:].sum(axis=1)
        scores = binary_scores(y_ref, (p_ref >= 0.5).astype(int))
        self.log("val/sensitivity_referable", scores.sensitivity, prog_bar=True)
        self.log("val/specificity_referable", scores.specificity)

        # Per-class recall: aggregate QWK can look fine while a rare class is
        # never predicted at all.
        for cls in range(self.num_classes):
            mask = targets == cls
            recall = float((preds[mask] == cls).mean()) if mask.any() else float("nan")
            self.log(f"val/recall_{cls}_{CLASS_NAMES[cls].replace(' ', '')}", recall)

        self._val_logits.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        # Only parameters that actually require grad -- frozen BN affine weights
        # must not be handed to the optimiser, or AdamW allocates moment buffers
        # for tensors it will never update.
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        warmup = max(self.hparams.warmup_epochs, 0)
        total = max(self.hparams.max_epochs, warmup + 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / (warmup + 1)
            progress = (epoch - warmup) / max(total - warmup, 1)
            return float(0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
