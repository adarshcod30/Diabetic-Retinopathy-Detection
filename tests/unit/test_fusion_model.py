"""Tests for the fusion head model and its GradingModule wiring."""

from __future__ import annotations

import pytest

from drdetect.fusion.model import FusionHead

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


def test_fusion_head_output_shape_matches_num_outputs():
    head = FusionHead(embedding_dim=1280, lesion_dim=9, num_outputs=4)
    x = torch.randn(6, 1280 + 9)
    out = head(x)
    assert out.shape == (6, 4)


def test_fusion_head_accepts_arbitrary_batch_size():
    head = FusionHead(embedding_dim=16, lesion_dim=4, num_outputs=4)
    for batch_size in (1, 3, 32):
        out = head(torch.randn(batch_size, 20))
        assert out.shape == (batch_size, 4)


def test_grading_module_trains_the_fusion_head_with_corn_loss(monkeypatch):
    from drdetect.grading.losses import corn_task_pos_weights
    from drdetect.grading.module import GradingModule

    head = FusionHead(embedding_dim=16, lesion_dim=4, num_outputs=4)
    labels = [0, 1, 2, 3, 4, 0, 1, 2]
    module = GradingModule(
        head,
        loss_name="corn",
        task_pos_weights=corn_task_pos_weights(labels),
        max_epochs=5,
    )
    monkeypatch.setattr(module, "log", lambda *a, **kw: None)

    x = torch.randn(8, 20)
    y = torch.tensor(labels)
    loss = module.training_step((x, y), batch_idx=0)

    assert torch.isfinite(loss)


def test_grading_module_decodes_fusion_logits_to_valid_grades():
    from drdetect.grading.losses import decode_output

    head = FusionHead(embedding_dim=16, lesion_dim=4, num_outputs=4)
    x = torch.randn(10, 20)
    with torch.no_grad():
        logits = head(x)

    preds, referable = decode_output(logits, "corn")

    assert preds.shape == (10,)
    assert preds.min() >= 0 and preds.max() <= 4
    assert referable.shape == (10,)
