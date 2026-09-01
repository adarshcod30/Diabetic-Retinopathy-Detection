"""Regression tests for the training divergence guard.

Real failure this pins: on APTOS at batch 4, train loss went 0.786 -> 239.35 in
a single epoch when LR warmup ended at 3e-4, and the model collapsed to
predicting grade 0 for every image (QWK 0.0000, referable sensitivity 0.000).
Without a guard that costs 2.5 hours to discover; with one, two epochs.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


def _module(lr=1e-4):
    from drdetect.grading.model import build_model
    from drdetect.grading.module import GradingModule

    return GradingModule(build_model("resnet18", pretrained=False), lr=lr, max_epochs=10)


class _FakeTrainer:
    def __init__(self, loss):
        self.callback_metrics = {"train/loss": torch.tensor(loss)}


class TestDivergenceGuard:
    def test_raises_on_exploding_epoch_loss(self):
        m = _module()
        m.trainer = _FakeTrainer(0.78)
        m.on_train_epoch_end()  # establishes the baseline
        m.trainer = _FakeTrainer(239.35)  # the real observed value
        with pytest.raises(RuntimeError, match="Training diverged"):
            m.on_train_epoch_end()

    def test_tolerates_normal_fluctuation(self):
        """Loss rising modestly between epochs is ordinary and must not abort."""
        m = _module()
        for loss in (0.78, 0.91, 0.85, 1.2, 0.7):
            m.trainer = _FakeTrainer(loss)
            m.on_train_epoch_end()

    def test_absolute_floor_prevents_false_alarms_on_tiny_losses(self):
        """10x a very small loss is still small; only abort past an absolute floor."""
        m = _module()
        m.trainer = _FakeTrainer(0.01)
        m.on_train_epoch_end()
        m.trainer = _FakeTrainer(0.4)  # 40x, but nowhere near diverged
        m.on_train_epoch_end()

    def test_nan_loss_aborts_immediately(self):
        m = _module()
        x = torch.randn(2, 3, 64, 64)
        y = torch.randint(0, 5, (2,))
        del m._modules["criterion"]  # nn.Module rejects replacing a submodule
        m.criterion = lambda *_: torch.tensor(float("nan"))
        with pytest.raises(RuntimeError, match="Diverged"):
            m.training_step((x, y), 0)

    def test_no_trainer_metrics_is_not_an_error(self):
        m = _module()
        m.trainer = _FakeTrainer(0.5)
        m.trainer.callback_metrics = {}
        m.on_train_epoch_end()


class TestSafeDefaults:
    def test_default_lr_is_small_batch_appropriate(self):
        """3e-4 diverged at batch 4; the default must be below it."""
        assert _module().hparams.lr <= 1e-4

    def test_warmup_is_long_enough_to_matter(self):
        assert _module().hparams.warmup_epochs >= 3

    def test_train_script_enables_gradient_clipping_by_default(self):
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            [sys.executable, str(repo / "scripts/train.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
        assert "--grad-clip" in out
        src = (repo / "scripts/train.py").read_text()
        assert "gradient_clip_val" in src
        assert "default=1.0" in src.split("--grad-clip")[1][:120]
