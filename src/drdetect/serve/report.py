"""One-page PDF report: image, Grad-CAM evidence, grade, and a plain disclaimer.

The 30-second review target (docs/04_ROADMAP.md, Phase 6) is a property of
this layout that has not been measured yet -- Phase 6 is where it gets timed
against real reviewers. This module only produces the artefact to be timed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from drdetect.serve.pipeline import PredictionResult

__all__ = ["build_report_pdf"]

_DISCLAIMER = (
    "Research prototype output only. NOT a medical device and NOT a diagnosis. "
    "Every flagged case must be reviewed by a qualified clinician before any "
    "action is taken."
)


def _draw_lesion_legend(c: canvas.Canvas, x: float, y: float) -> float:
    """A small colour key for the lesion-overlay panel -- an outline is only
    legible if the viewer knows what each colour means."""
    from drdetect.explain.evidence import LESION_COLORS_RGB

    labels = {
        "hard_exudates": "Hard exudates",
        "soft_exudates": "Soft exudates",
        "haemorrhages": "Haemorrhages",
        "microaneurysms": "Microaneurysms",
    }
    c.setFont("Helvetica", 8)
    for lesion_type, label in labels.items():
        r, g, b = LESION_COLORS_RGB[lesion_type]
        c.setFillColorRGB(r / 255, g / 255, b / 255)
        c.rect(x, y - 7, 8, 8, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(x + 11, y - 6, label)
        x += 11 + len(label) * 4.3 + 14
    return y - 14


def _draw_wrapped(
    c: canvas.Canvas, text: str, x: float, y: float, width: float, font_size: int = 9
) -> float:
    """Draw `text` wrapped to `width` points, returning the y below the last line."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    c.setFont("Helvetica", font_size)
    words = text.split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if stringWidth(trial, "Helvetica", font_size) > width and line:
            c.drawString(x, y, line)
            y -= font_size + 2
            line = word
        else:
            line = trial
    if line:
        c.drawString(x, y, line)
        y -= font_size + 2
    return y


def build_report_pdf(
    result: PredictionResult,
    image_name: str,
    out_path: str | Path,
    *,
    checkpoint_name: str = "",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = A4
    margin = 18 * mm
    c = canvas.Canvas(str(out_path), pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_h - margin, "Diabetic Retinopathy Screening Report")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawString(margin, page_h - margin - 14, f"Image: {image_name}")
    if checkpoint_name:
        c.drawString(margin, page_h - margin - 26, f"Model checkpoint: {checkpoint_name}")
    c.setFillColor(colors.black)

    y = page_h - margin - 45

    q = result.quality
    if not q.usable:
        c.setFillColor(colors.red)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "IMAGE REJECTED -- RECAPTURE REQUIRED")
        c.setFillColor(colors.black)
        y -= 22
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, "Reasons:")
        y -= 14
        for reason in q.reasons:
            y = _draw_wrapped(c, f"- {reason}", margin, y, page_w - 2 * margin)
        y -= 10
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, "Quality metrics (for reference):")
        y -= 14
    else:
        panels = [
            (Image.fromarray(result.preprocessed.astype(np.uint8)), "Preprocessed image"),
            (
                Image.fromarray(result.cam_overlay.astype(np.uint8)),
                "Grad-CAM evidence (predicted grade)",
            ),
        ]
        if result.lesion_overlay is not None:
            panels.append(
                (
                    Image.fromarray(result.lesion_overlay.astype(np.uint8)),
                    "Lesion evidence (outlines)",
                )
            )

        gap = 10
        img_w = (page_w - 2 * margin - gap * (len(panels) - 1)) / len(panels)
        img_h = img_w  # preprocessed images are square (circle_crop -> resize)

        for i, (panel_img, label) in enumerate(panels):
            x = margin + i * (img_w + gap)
            c.drawImage(_pil_reader(panel_img), x, y - img_h, width=img_w, height=img_h)
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.grey)
            c.drawCentredString(x + img_w / 2, y - img_h - 12, label)
        c.setFillColor(colors.black)
        y = y - img_h - 26

        if result.lesion_overlay is not None:
            y = _draw_lesion_legend(c, margin, y)
        y -= 6

        c.setFont("Helvetica-Bold", 14)
        referable_tag = "REFERABLE" if result.referable else "not referable"
        c.drawString(
            margin, y, f"ICDR grade {result.grade}: {result.grade_name}  ({referable_tag})"
        )
        y -= 20

        if result.icdr_rationale:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, "Evidence:")
            y -= 14
            c.setFont("Helvetica", 9)
            y = _draw_wrapped(c, result.icdr_rationale, margin + 8, y, page_w - 2 * margin - 8)
            y -= 6

        calib_label = "temperature-scaled" if result.calibrated else "uncalibrated"
        c.setFont("Helvetica", 10)
        if result.confidence is not None:
            c.drawString(margin, y, f"Model confidence ({calib_label}): {result.confidence:.2%}")
            y -= 16

        if result.class_probs:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, f"Per-grade probability ({calib_label}):")
            y -= 14
            c.setFont("Helvetica", 9)
            from drdetect.grading.module import CLASS_NAMES

            for name, p in zip(CLASS_NAMES, result.class_probs, strict=True):
                c.drawString(margin + 8, y, f"{name:<10} {p:.2%}")
                y -= 12
            y -= 6

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, "Image quality (passed):")
        y -= 14

    c.setFont("Helvetica", 9)
    c.drawString(
        margin + 8,
        y,
        f"sharpness {q.sharpness:.0f}  |  mean brightness {q.mean_brightness:.0f}  |  "
        f"field-of-view {q.fov_fraction:.0%} of frame",
    )
    y -= 24

    c.setStrokeColor(colors.grey)
    c.line(margin, y, page_w - margin, y)
    y -= 16
    c.setFillColor(colors.grey)
    y = _draw_wrapped(c, _DISCLAIMER, margin, y, page_w - 2 * margin, font_size=8)

    c.showPage()
    c.save()
    return out_path


def _pil_reader(image: Image.Image):
    from reportlab.lib.utils import ImageReader

    return ImageReader(image)
