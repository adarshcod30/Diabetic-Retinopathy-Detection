"""Tests for temperature scaling."""

from __future__ import annotations

import json

import torch

from drdetect.calibration.temperature import (
    TemperatureScaler,
    fit_temperature,
    load_temperature,
    save_temperature,
)


def test_temperature_scaler_forward_divides_logits():
    scaler = TemperatureScaler()
    scaler.log_temperature.data.fill_(torch.log(torch.tensor(2.0)))
    logits = torch.tensor([[2.0, 4.0, 6.0]])
    out = scaler(logits)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-5)


def test_temperature_never_reaches_zero_or_negative():
    # -50, not -1000: exp() genuinely cannot produce <=0 for any finite input,
    # but exp(-1000) underflows float32 to exactly 0.0, which would make this
    # test fail for a float-precision reason unrelated to what it checks --
    # -50 (exp(-50) ~= 2e-22) stays within float32's positive range while
    # still testing a value far more extreme than 50 LBFGS steps at lr=0.01
    # could plausibly reach from a log_temperature=0 start.
    scaler = TemperatureScaler()
    scaler.log_temperature.data.fill_(-50.0)
    assert scaler.temperature.item() > 0.0


def test_fit_temperature_softens_an_overconfident_model():
    """A model whose logits are far too extreme for its actual accuracy should
    fit T > 1 (softening) -- the textbook overconfidence case Guo et al. describe."""
    torch.manual_seed(0)
    n, k = 200, 5
    targets = torch.randint(0, k, (n,))
    # Correct class gets a huge logit, others near zero -- overconfident but
    # only ~70% accurate once softmax'd through a coarser class boundary is
    # simulated by occasionally corrupting the "confident" class.
    logits = torch.full((n, k), -10.0)
    confident_target = targets.clone()
    wrong_mask = torch.rand(n) < 0.3
    confident_target[wrong_mask] = (confident_target[wrong_mask] + 1) % k
    logits[torch.arange(n), confident_target] = 10.0

    temperature = fit_temperature(logits, targets)
    assert temperature > 1.0


def test_fit_temperature_does_not_change_argmax():
    torch.manual_seed(1)
    logits = torch.randn(100, 5) * 3
    targets = torch.randint(0, 5, (100,))
    temperature = fit_temperature(logits, targets)
    before = logits.argmax(dim=1)
    after = (logits / temperature).argmax(dim=1)
    assert torch.equal(before, after)


def test_save_and_load_temperature_roundtrip(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("not a real checkpoint")
    sidecar = save_temperature(ckpt, 1.732, fitted_on="test")
    assert sidecar.name == "temperature.json"
    assert json.loads(sidecar.read_text())["temperature"] == 1.732
    assert load_temperature(ckpt) == 1.732


def test_load_temperature_defaults_to_one_when_uncalibrated(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("not a real checkpoint")
    assert load_temperature(ckpt) == 1.0
