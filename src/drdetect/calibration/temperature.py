"""Temperature scaling (Guo et al. 2017) for post-hoc confidence calibration.

Modern networks are systematically overconfident: raw softmax output is not a
probability, even when the model's argmax is highly accurate. This project's
own Phase 2 report has said so explicitly on every single output --
"Model confidence (uncalibrated)" -- since there was nothing calibrated to
report. This module is what closes that disclaimer, for checkpoints it has
actually been run against.

Temperature scaling fixes overconfidence with one learned scalar T, applied
AFTER training with every other weight frozen:

    calibrated_probs = softmax(logits / T)

T > 1 softens (reduces) confidence; T < 1 sharpens it. It cannot change which
class wins (dividing every logit by the same positive number does not
reorder them), only how confident the model is entitled to sound about it.

Only meaningful for a softmax head (`ce` / `distance_ce`). CORN's logits are
per-task sigmoids, not a single softmax, and regression has no probability
distribution to calibrate at all -- both are out of scope here, not silently
mishandled.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemperatureScaler", "fit_temperature", "save_temperature", "load_temperature"]


class TemperatureScaler(nn.Module):
    """Wraps a frozen classifier's logits; only `log_temperature` is ever trained."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        # exp(), not a raw Parameter: keeps T > 0 for every value the
        # optimiser can reach. logits / T is undefined at T=0 and inverted
        # (sharpens instead of softens) for T<0 -- a plain Parameter could
        # wander into either during optimisation; exp() cannot reach them.
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, *, max_iter: int = 50) -> float:
    """Fit T on held-out (logits, targets) by minimising NLL. Returns the fitted T.

    `logits` must come from a split the model was not trained OR early-stopped
    on. This project has only a train/val split per fold (no separate
    calibration split was reserved at Phase 0), so in practice this is fit on
    the same fold-0 validation set every other Phase 1/3 number came from --
    the identical "optimistic, not yet the Phase 8 number" caveat already
    disclosed for the operating threshold applies here too, not a new one.

    LBFGS, not Adam/SGD: the standard choice for this exact problem in the
    original paper's own reference implementation -- one parameter, a
    well-behaved objective, converges in a handful of iterations.
    """
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.item())


def save_temperature(checkpoint_path: str | Path, temperature: float, *, fitted_on: str) -> Path:
    """Write a sidecar JSON next to `checkpoint_path`, not inside the .ckpt.

    Keeping it separate means load_temperature() can no-op cleanly for every
    checkpoint that has never been calibrated, instead of every existing
    checkpoint needing to carry a temperature=1.0 placeholder it was never
    actually fit for.
    """
    out = Path(checkpoint_path).with_name("temperature.json")
    out.write_text(json.dumps({"temperature": temperature, "fitted_on": fitted_on}, indent=2))
    return out


def load_temperature(checkpoint_path: str | Path) -> float:
    """1.0 (uncalibrated, a no-op divisor) if this checkpoint has no fitted temperature."""
    sidecar = Path(checkpoint_path).with_name("temperature.json")
    if not sidecar.exists():
        return 1.0
    return float(json.loads(sidecar.read_text())["temperature"])
