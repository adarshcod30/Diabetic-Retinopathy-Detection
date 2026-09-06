"""HuggingFace Spaces entry point for the Gradio demo (docs/04_ROADMAP.md, Phase 9).

Not used for local development -- `scripts/demo.py` is the local launcher,
with a required `--checkpoint` flag pointing anywhere on disk. A Space has no
such flag: HF Spaces runs `python app.py` with no arguments, so this file
fetches the checkpoint from the separate model repo
(huggingface.co/adarshcod30/drdetect-dr-screening -- research-use-only, see
MODEL_CARD.md there) via `hf_hub_download` instead of bundling a 46MB weight
file into this Space's own repo. `hf_hub_download` caches the file locally
after the first fetch, so a Space restart does not re-download it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

MODEL_REPO = "adarshcod30/drdetect-dr-screening"

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename="best.ckpt")

    from drdetect.serve.demo import build_interface

    # regression loss, not ce: it won the locked external evaluation decisively
    # (docs/22_PHASE8_VALIDATION_RESULTS.md, referable AUC 0.924 vs 0.888, DeLong p=6.1e-10)
    demo = build_interface(checkpoint, backbone="efficientnet_b0", loss_name="regression", size=512)
    demo.launch()
