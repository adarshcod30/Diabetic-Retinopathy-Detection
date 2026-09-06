"""End-to-end test of the Phase 9 FastAPI service.

Same strategy as test_serve_pipeline.py: an untrained model, saved to a real
on-disk checkpoint so the API's own startup path (DRDETECT_CHECKPOINT env var
-> load_grader) is genuinely exercised, not bypassed -- the point is to prove
the HTTP plumbing (upload -> decode -> run_pipeline -> JSON response), not
grading accuracy.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from drdetect.grading.model import build_model


@pytest.fixture
def untrained_checkpoint(tmp_path: Path) -> Path:
    model = build_model("efficientnet_b0", num_outputs=5, pretrained=False, freeze_bn=True)
    ckpt_path = tmp_path / "untrained.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, ckpt_path)
    return ckpt_path


@pytest.fixture
def textured_fundus_png_bytes() -> bytes:
    rng = np.random.default_rng(0)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:600, :800]
    disc = (yy - 300) ** 2 + (xx - 420) ** 2 <= 280**2
    speckle = rng.integers(0, 255, size=(600, 800), dtype=np.uint8)
    for c, base in enumerate((150, 80, 60)):
        channel = np.clip(base + speckle.astype(int) - 128, 0, 255).astype(np.uint8)
        img[..., c] = np.where(disc, channel, 0)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(untrained_checkpoint, monkeypatch):
    monkeypatch.setenv("DRDETECT_CHECKPOINT", str(untrained_checkpoint))
    monkeypatch.setenv("DRDETECT_SIZE", "224")
    from fastapi.testclient import TestClient

    from drdetect.serve.api import app

    with TestClient(app) as c:
        yield c


def test_health_reports_loaded_checkpoint(client, untrained_checkpoint):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checkpoint"] == str(untrained_checkpoint)
    assert "not a medical device" in body["disclaimer"].lower()


def test_grade_accepts_a_usable_image(client, textured_fundus_png_bytes):
    resp = client.post(
        "/grade",
        files={"image": ("fundus.png", io.BytesIO(textured_fundus_png_bytes), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["usable"] is True
    assert body["grade"] in {0, 1, 2, 3, 4}
    assert body["referable"] == (body["grade"] >= 2)
    assert body["class_probs"] is not None
    assert len(body["class_probs"]) == 5
    assert abs(sum(body["class_probs"]) - 1.0) < 1e-4
    assert "not a medical device" in body["disclaimer"].lower()


def test_grade_rejects_an_unusable_image(client):
    black = np.zeros((600, 800, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", black)
    assert ok
    resp = client.post(
        "/grade", files={"image": ("black.png", io.BytesIO(buf.tobytes()), "image/png")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["usable"] is False
    assert body["grade"] is None
    assert len(body["quality_reasons"]) > 0


def test_grade_rejects_undecodable_upload(client):
    resp = client.post(
        "/grade", files={"image": ("not_an_image.png", io.BytesIO(b"not an image"), "image/png")}
    )
    assert resp.status_code == 400
