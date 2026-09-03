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
from drdetect.grading.losses import build_loss, decode_output, naive_referable_cut

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
        task_pos_weights: list[float] | None = None,
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
        self.criterion = build_loss(
            loss_name,
            num_classes=num_classes,
            class_weights=class_weights,
            task_pos_weights=task_pos_weights,
        )

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

    def on_before_optimizer_step(self, optimizer):
        """Log the PRE-clip gradient norm.

        This hook fires before configure_gradient_clipping, so it sees the
        unclipped gradient. It is the direct instrument for the epoch-3 question:
        clipping acts on the accumulation MEAN, and the mean of 16 samples has a
        smaller norm than a 4-sample gradient, so clip 1.0 binds far less often at
        higher accumulation. Without this, "the clip bound less" and "the noise
        was lower" are indistinguishable in the logs.
        """
        from lightning.pytorch.utilities import grad_norm

        norms = grad_norm(self, norm_type=2)
        total = norms.get("grad_2.0_norm_total")
        if total is not None:
            self.log("train/grad_norm", total, on_step=False, on_epoch=True)
            clip = getattr(self.trainer, "gradient_clip_val", None)
            if clip:
                self.log(
                    "train/grad_clipped_frac",
                    float(total > clip),
                    on_step=False,
                    on_epoch=True,
                )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_logits.append(logits.detach().float().cpu())
        self._val_targets.append(y.detach().cpu())

    def decode(self, output: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Delegate to the shared decoder so training and evaluation cannot drift."""
        return decode_output(output, self.loss_name)

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
        cut = naive_referable_cut(self.loss_name)
        scores = binary_scores(y_ref, (p_ref >= cut).astype(int))
        self.log("val/sensitivity_referable", scores.sensitivity, prog_bar=True)
        self.log("val/specificity_referable", scores.specificity)

        # Per-class recall: aggregate QWK can look fine while a rare class is
        # never predicted at all.
        for cls in range(self.num_classes):
            mask = targets == cls
            recall = float((preds[mask] == cls).mean()) if mask.any() else float("nan")
            self.log(f"val/recall_{cls}_{CLASS_NAMES[cls].replace(' ', '')}", recall)

        # Macro-averaged recall: the unweighted mean over classes, so a rare
        # class counts as much as a common one. Logged as a selection
        # alternative to QWK, which on this data tracks grade-2 recall
        # (r=+0.51 over 70 epochs) and, for balanced CORN, was strongly
        # ANTI-correlated with grade-1 recall (r=-0.70). Selecting that run on
        # QWK cost 0.100 macro-recall to gain 0.0097 QWK.
        recalls = [
            float((preds[targets == c] == c).mean())
            for c in range(self.num_classes)
            if (targets == c).any()
        ]
        self.log("val/macro_recall", float(np.mean(recalls)) if recalls else 0.0)

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
