"""HuggingFace Spaces entry point for the Gradio demo (docs/04_ROADMAP.md, Phase 9).

Not used for local development -- `scripts/demo.py` is the local launcher,
with a required `--checkpoint` flag pointing anywhere on disk. A Space has no
such flag: HF Spaces runs `python app.py` with no arguments, so this file
resolves the checkpoint from a fixed, documented location instead
(`checkpoint/best.ckpt`, uploaded to the Space's own repo alongside this
file -- research-use-only weights, see MODEL_CARD.md, never committed to the
main GitHub repo, which ships code only).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

CHECKPOINT = Path(__file__).resolve().parent / "checkpoint" / "best.ckpt"

if __name__ == "__main__":
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT} not found. Upload the chosen checkpoint (see MODEL_CARD.md) to "
            "this Space's own repo at checkpoint/best.ckpt before it will serve requests."
        )

    from drdetect.serve.demo import build_interface

    demo = build_interface(CHECKPOINT, backbone="efficientnet_b0", loss_name="ce", size=512)
    demo.launch()
