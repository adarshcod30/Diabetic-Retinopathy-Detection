"""FastAPI service wrapping the Phase 2 pipeline (docs/04_ROADMAP.md, Phase 9).

CPU-only, single global model instance loaded once at startup -- the same
design constraint `scripts/predict.py` and `load_grader` already state
explicitly: a district server is not assumed to have a GPU. This is a
research prototype, not a medical device (docs/01_PROJECT_ANALYSIS.md §11);
every response carries that disclaimer rather than only the PDF report doing so.

Run locally:
    uvicorn drdetect.serve.api:app --host 0.0.0.0 --port 8000

Configuration is via environment variables (not CLI flags) because this is
meant to run inside a container (Dockerfile) where env vars are the natural
knob, not argv:
    DRDETECT_CHECKPOINT   path to the grading checkpoint (required)
    DRDETECT_BACKBONE     default: efficientnet_b0
    DRDETECT_LOSS         default: ce
    DRDETECT_SIZE         default: 512
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from drdetect.calibration.temperature import load_temperature
from drdetect.serve.pipeline import load_grader, run_pipeline

NOT_A_MEDICAL_DEVICE = (
    "Research prototype only. NOT a medical device and NOT a substitute for "
    "clinical judgement. Every result requires confirmation by a qualified "
    "ophthalmologist before any care decision is made."
)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.environ.get("DRDETECT_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).exists():
        raise RuntimeError(
            "DRDETECT_CHECKPOINT env var must point to an existing checkpoint file. "
            f"Got: {checkpoint!r}"
        )
    backbone = os.environ.get("DRDETECT_BACKBONE", "efficientnet_b0")
    loss_name = os.environ.get("DRDETECT_LOSS", "ce")
    _state["size"] = int(os.environ.get("DRDETECT_SIZE", "512"))
    _state["loss_name"] = loss_name
    _state["checkpoint"] = checkpoint
    _state["model"] = load_grader(checkpoint, backbone=backbone, loss_name=loss_name, device="cpu")
    _state["temperature"] = load_temperature(checkpoint)
    yield
    _state.clear()


app = FastAPI(
    title="drdetect: DR screening API",
    description=NOT_A_MEDICAL_DEVICE,
    version="0.1.0",
    lifespan=lifespan,
)


class GradeResponse(BaseModel):
    usable: bool
    quality_reasons: list[str]
    grade: int | None = None
    grade_name: str | None = None
    referable: bool | None = None
    confidence: float | None = None
    calibrated: bool = False
    class_probs: list[float] | None = None
    disclaimer: str = NOT_A_MEDICAL_DEVICE


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "checkpoint": _state.get("checkpoint"),
        "disclaimer": NOT_A_MEDICAL_DEVICE,
    }


@app.post("/grade", response_model=GradeResponse)
async def grade(image: UploadFile = File(...)) -> GradeResponse:  # noqa: B008 -- FastAPI's own required-file-upload idiom
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_bytes = await image.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = run_pipeline(
        rgb,
        _state["model"],
        loss_name=_state["loss_name"],
        size=_state["size"],
        device="cpu",
        temperature=_state["temperature"],
    )

    return GradeResponse(
        usable=result.quality.usable,
        quality_reasons=result.quality.reasons,
        grade=result.grade,
        grade_name=result.grade_name,
        referable=result.referable,
        confidence=result.confidence,
        calibrated=result.calibrated,
        class_probs=result.class_probs,
    )
