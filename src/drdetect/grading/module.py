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
from drdetect.grading.losses import (
    build_loss,
    corn_probabilities,
    regression_predict,
)

__all__ = ["GradingModule"]

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


class GradingModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_classes: int = 5,
        loss_name: str = "ce",
        class_weights: list[float] | None = None,
        warmup_epochs: int = 3,
        max_epochs: int = 40,
    ):
        super().__init__()
        # `model` is a live module; saving it into the checkpoint hyperparameters
        # would serialise the whole network twice.
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.criterion = build_loss(loss_name, num_classes=num_classes, class_weights=class_weights)

        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._first_epoch_loss: float | None = None

    @property
    def _epoch(self) -> int:
        """current_epoch without requiring an attached trainer.

        Used only in error messages. A guard that raises AttributeError while
        constructing its own diagnostic is worse than no guard.
        """
        try:
            return int(self.current_epoch)
        except (AttributeError, RuntimeError):
            return -1

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Training loss is {loss.item()} at epoch {self._epoch}, step "
                f"{batch_idx}. Diverged -- lower --lr or check the data pipeline."
            )
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """Abort a diverged run immediately rather than after hours.

        Small-batch training (batch 4, forced here by available RAM) produces
        noisy gradients; with too high a learning rate the loss can explode by
        two orders of magnitude in a single epoch. Measured on this project:
        train loss went 0.786 -> 239.35 when warmup ended at lr 3e-4, and the
        model collapsed to predicting grade 0 for everything. Detecting that in
        one epoch instead of forty is the difference between five minutes and
        two and a half hours.
        """
        loss = self.trainer.callback_metrics.get("train/loss")
        if loss is None:
            return
        loss = float(loss)
        if self._first_epoch_loss is None:
            self._first_epoch_loss = loss
            return
        if loss > max(10.0 * self._first_epoch_loss, 5.0):
            raise RuntimeError(
                f"Training diverged: loss {loss:.2f} at epoch {self._epoch} vs "
                f"{self._first_epoch_loss:.3f} at epoch 0.\n"
                f"With batch size 4 this usually means the learning rate is too high. "
                f"Try --lr 5e-5, and confirm gradient clipping is enabled."
            )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_logits.append(logits.detach().float().cpu())
        self._val_targets.append(y.detach().cpu())

    def decode(self, output: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Map raw model output to (predicted grade, referable score).

        Each loss parameterises the head differently, so decoding must match:
          ce / distance_ce -> K-way softmax; referable = sum of P(class >= 2)
          corn             -> K-1 chained sigmoids; referable = P(y > 1) directly
          regression       -> one continuous value; the value itself ranks severity

        Referable score only needs to be monotone in severity -- threshold
        selection handles the scale.
        """
        if self.loss_name == "corn":
            cum = corn_probabilities(output)  # P(y > j)
            preds = (cum > 0.5).sum(dim=1).numpy()
            referable = cum[:, 1].numpy()  # P(y > 1) == P(grade >= 2)
            return preds, referable

        if self.loss_name == "regression":
            preds = regression_predict(output).numpy()
            referable = output.squeeze(-1).numpy()
            return preds, referable

        probs = torch.softmax(output, dim=1).numpy()
        return probs.argmax(axis=1), probs[:, 2:].sum(axis=1)

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets).numpy()
        preds, p_ref = self.decode(logits)

        self.log("val/qwk", quadratic_weighted_kappa(targets, preds), prog_bar=True)
        self.log("val/acc", float((preds == targets).mean()))

        # Referable DR (grade >= 2) at the default 0.5 operating point. The
        # calibrated, sensitivity-targeted threshold is chosen in Phase 5 -- this
        # is only a progress signal, not the reported operating point.
        y_ref = referable_labels(targets)
        # 0.5 is a naive cut and is only a progress signal; the reported
        # operating point is chosen for target sensitivity in scripts/evaluate.py.
        cut = 1.5 if self.loss_name == "regression" else 0.5
        scores = binary_scores(y_ref, (p_ref >= cut).astype(int))
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
