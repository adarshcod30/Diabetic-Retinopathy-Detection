"""End-to-end single-image inference: quality gate -> grade -> Grad-CAM.

This is the Phase 2 vertical slice (docs/04_ROADMAP.md): image in, graded and
explained result out. `scripts/predict.py` and the Gradio demo are both thin
callers of `run_pipeline` so the two entry points cannot drift the way
training and evaluation once did over `decode_output` (docs/07_PHASE3_RESULTS.md).

Confidence is calibrated ONLY when the caller passes a `temperature` fitted
by `scripts/calibrate.py` (see `drdetect.calibration.temperature`) -- the
default, 1.0, is a no-op divisor that reports raw softmax output. Check
`PredictionResult.calibrated` before presenting a confidence number as a
validated probability rather than a raw model output; the two look
identical in this dataclass except for that flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from drdetect.data.dataset import build_transforms
from drdetect.enhance.preprocessing import preprocess
from drdetect.eval.metrics import REFERABLE_THRESHOLD
from drdetect.explain.gradcam import generate_cam, overlay_cam
from drdetect.grading.losses import decode_output, outputs_for_loss
from drdetect.grading.model import build_model
from drdetect.grading.module import CLASS_NAMES
from drdetect.quality.assessment import QualityResult, assess_quality

__all__ = ["PredictionResult", "load_grader", "run_pipeline"]


@dataclass(frozen=True)
class PredictionResult:
    quality: QualityResult
    preprocessed: np.ndarray | None = None
    grade: int | None = None
    grade_name: str | None = None
    referable: bool | None = None
    confidence: float | None = None
    class_probs: list[float] | None = None
    cam_overlay: np.ndarray | None = None
    calibrated: bool = False


def load_grader(
    checkpoint: str | Path,
    *,
    backbone: str = "efficientnet_b0",
    loss_name: str = "ce",
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a trained grader for inference, on CPU by default.

    CPU-only is deliberate, not a fallback: the roadmap's exit criterion for
    this script is "single image, CPU, no GPU required" -- a district
    screening kiosk is not assumed to have one.
    """
    n_outputs = outputs_for_loss(loss_name)
    model = build_model(backbone, num_outputs=n_outputs, pretrained=False, freeze_bn=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def run_pipeline(
    raw_image_rgb: np.ndarray,
    model: torch.nn.Module,
    *,
    loss_name: str = "ce",
    size: int = 512,
    device: str = "cpu",
    skip_quality_gate: bool = False,
    temperature: float = 1.0,
) -> PredictionResult:
    """Run the full quality -> grade -> explain chain on one raw image.

    Args:
        raw_image_rgb: HxWx3 uint8 RGB, as captured -- not preprocessed.
        model: from `load_grader`.
        skip_quality_gate: force grading through even on a rejected image.
            Used by the demo so a user can see *why* an image was flagged
            instead of only being refused a result.
        temperature: from `drdetect.calibration.temperature.load_temperature`
            for this checkpoint, or the default 1.0 (a no-op divisor) if it
            has never been calibrated. Dividing logits by T before softmax
            changes reported confidence, never the predicted grade (argmax
            is invariant to a positive rescaling) -- see
            scripts/calibrate.py for what temperature scaling is and why.
    """
    quality = assess_quality(raw_image_rgb)
    if not quality.usable and not skip_quality_gate:
        return PredictionResult(quality=quality)

    pre = preprocess(raw_image_rgb, size=size, use_ben_graham=True)
    tensor = build_transforms(size, train=False)(image=pre)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
    preds, _p_ref = decode_output(output.cpu(), loss_name)
    grade = int(preds[0])

    class_probs = None
    confidence = None
    if loss_name in ("ce", "distance_ce"):
        probs = torch.softmax(output / temperature, dim=1)[0].detach().cpu().tolist()
        class_probs = probs
        confidence = float(probs[grade])

    cam = generate_cam(model, tensor, target_class=grade)
    overlay = overlay_cam(pre, cam)

    return PredictionResult(
        quality=quality,
        preprocessed=pre,
        grade=grade,
        grade_name=CLASS_NAMES[grade],
        referable=grade >= REFERABLE_THRESHOLD,
        confidence=confidence,
        class_probs=class_probs,
        cam_overlay=overlay,
        calibrated=temperature != 1.0,
    )
