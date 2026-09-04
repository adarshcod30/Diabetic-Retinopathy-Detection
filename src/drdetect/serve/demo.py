"""Local Gradio demo: upload a fundus photo, see the graded result live.

Unlike `scripts/predict.py` this runs on whatever accelerator is available
(MPS/CUDA/CPU) -- the CPU-only constraint is specifically about the
deployable CLI path, not an interactive local demo (docs/04_ROADMAP.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drdetect.serve.pipeline import PredictionResult, load_grader, run_pipeline

__all__ = ["build_interface"]


def _format_summary(result: PredictionResult) -> str:
    q = result.quality
    lines = [
        f"**Quality:** sharpness {q.sharpness:.0f} | brightness {q.mean_brightness:.0f} | "
        f"field-of-view {q.fov_fraction:.0%}"
    ]
    if not q.usable:
        lines.append("\n**REJECTED -- recapture requested.** Reasons:")
        lines.extend(f"- {r}" for r in q.reasons)
        return "\n".join(lines)

    lines.append(f"\n### Grade {result.grade}: {result.grade_name}")
    lines.append(f"**Referable (grade >= 2):** {'YES' if result.referable else 'no'}")
    if result.confidence is not None:
        lines.append(f"**Confidence (uncalibrated):** {result.confidence:.1%}")
    if result.class_probs:
        from drdetect.grading.module import CLASS_NAMES

        lines.append("\n| Grade | Probability |\n|---|---|")
        for name, p in zip(CLASS_NAMES, result.class_probs, strict=True):
            lines.append(f"| {name} | {p:.1%} |")
    lines.append(
        "\n*Research prototype only. Not a medical device. "
        "Every case must be reviewed by a qualified clinician.*"
    )
    return "\n".join(lines)


def build_interface(
    checkpoint: str | Path,
    *,
    backbone: str = "efficientnet_b0",
    loss_name: str = "ce",
    size: int = 512,
):
    import gradio as gr
    import torch

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_grader(checkpoint, backbone=backbone, loss_name=loss_name, device=device)

    def infer(image: np.ndarray, force: bool):
        if image is None:
            return None, "Upload a fundus photo to begin."
        result = run_pipeline(
            image, model, loss_name=loss_name, size=size, device=device, skip_quality_gate=force
        )
        overlay = result.cam_overlay if result.cam_overlay is not None else image
        return overlay, _format_summary(result)

    with gr.Blocks(title="DR Screening Demo") as demo:
        gr.Markdown(
            "# Diabetic Retinopathy Screening -- Local Demo\n"
            "Upload a fundus photo. This is a research prototype: outputs are not "
            "calibrated and this is not a diagnosis."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="numpy", label="Fundus photo")
                force = gr.Checkbox(label="Grade anyway if quality gate rejects", value=False)
                run_btn = gr.Button("Grade", variant="primary")
            with gr.Column():
                image_out = gr.Image(label="Grad-CAM evidence")
                summary_out = gr.Markdown()

        run_btn.click(infer, inputs=[image_in, force], outputs=[image_out, summary_out])

    return demo
