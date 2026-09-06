"""Tests for MC-dropout uncertainty sampling.

EfficientNet's classifier dropout is functional (`F.dropout(..., training=
self.training)`, not a persistent nn.Dropout submodule -- see
drdetect.calibration.mc_dropout's own docstring), so these tests exist to
directly verify the somewhat unusual train()-then-refreeze-BatchNorm
reactivation actually behaves as intended, not just that it runs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")


def _tiny_dropout_model():
    import torch.nn as nn

    class TinyDropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(8)
            self.drop = nn.Dropout(p=0.9)
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(self.drop(self.bn(x)))

    return TinyDropoutModel()


def test_mc_dropout_samples_vary_when_dropout_is_reactivated():
    from drdetect.calibration.mc_dropout import mc_dropout_samples

    model = _tiny_dropout_model()
    model.eval()
    x = torch.randn(4, 8)

    samples = mc_dropout_samples(model, x, n_samples=10)

    assert samples.shape == (10, 4, 3)
    assert samples.std(dim=0).mean().item() > 0


def test_mc_dropout_leaves_the_model_in_eval_mode_afterward():
    from drdetect.calibration.mc_dropout import mc_dropout_samples

    model = _tiny_dropout_model()
    model.eval()
    mc_dropout_samples(model, torch.randn(4, 8), n_samples=5)

    assert model.training is False


def test_mc_dropout_does_not_update_batchnorm_running_stats():
    from drdetect.calibration.mc_dropout import mc_dropout_samples

    model = _tiny_dropout_model()
    model.eval()
    before = model.bn.running_mean.clone()

    mc_dropout_samples(model, torch.randn(4, 8) * 100 + 50, n_samples=5)

    assert torch.equal(model.bn.running_mean, before)


def test_mc_dropout_samples_are_deterministic_with_dropout_disabled():
    """p=0 dropout should give identical samples -- confirms the variation
    seen elsewhere comes from dropout, not some other source of randomness."""
    import torch.nn as nn

    class NoDropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(x)

    from drdetect.calibration.mc_dropout import mc_dropout_samples

    model = NoDropoutModel()
    model.eval()
    samples = mc_dropout_samples(model, torch.randn(4, 8), n_samples=5)

    assert samples.std(dim=0).max().item() == pytest.approx(0.0, abs=1e-6)
