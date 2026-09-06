"""HuggingFace Spaces entry point for the Gradio demo (docs/04_ROADMAP.md, Phase 9).

Not used for local development -- `scripts/demo.py` is the local launcher,
with a required `--checkpoint` flag pointing anywhere on disk. A Space has no
such flag: HF Spaces runs `python app.py` with no arguments, so this file
fetches the checkpoint from the separate model repo
(huggingface.co/adarshcod30/drdetect-dr-screening -- research-use-only, see
MODEL_CARD.md there) via `hf_hub_download` instead of bundling a 46MB weight
file into this Space's own repo. `hf_hub_download` caches the file locally
after the first fetch, so a Space restart does not re-download it.

This project's model is CPU-only by design (see
`drdetect.serve.pipeline.load_grader`'s own docstring: "a district screening
kiosk is not assumed to have [a GPU]"), but the Space is hosted on HF's free
ZeroGPU hardware tier (free accounts can no longer create free CPU-basic
Gradio Spaces). Two ZeroGPU-specific fixes were needed, found by reading this
Space's own runtime error, not guessed:

1. `import spaces` early -- without it, HF's supervisor stops the app
   shortly after a clean startup ("Stopping Node.js server..." with no other
   error visible in the application log).
2. **`_zerogpu_registration` below**: even with (1), the platform's own
   `get_space_runtime` reported the real reason plainly --
   `"No @spaces.GPU function detected during startup"` -- ZeroGPU spaces
   require at least one such function to exist, whether or not it is ever
   called. This function is never invoked by the demo; grading always runs
   on CPU exactly as the rest of this project does. It exists solely to
   satisfy that platform check without changing this project's actual
   CPU-only inference path.
3. **`device="cpu"` passed explicitly to `build_interface` below**: its
   default auto-detection (`drdetect.serve.demo`) checks
   `torch.cuda.is_available()`, which ZeroGPU makes report `True` even
   outside an actual `@spaces.GPU` grant -- auto-detection then picked
   "cuda" and crashed the first real request with `RuntimeError: Low-level
   CUDA init... reached` the moment a tensor was moved to it, since nothing
   in the demo path is wrapped in `@spaces.GPU`. Forcing "cpu" here matches
   this project's actual design (CPU-only inference throughout) rather than
   trusting a detection heuristic that ZeroGPU's emulation invalidates.

Both are wrapped so this file still runs unmodified outside a Space
(`spaces` is not, and should not be, a dependency of the main `drdetect`
package).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import spaces

    @spaces.GPU
    def _zerogpu_registration() -> None:
        """Never called -- exists only so ZeroGPU's startup check finds a
        decorated function. Real inference always runs on CPU (see module
        docstring point 2)."""
        return None
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

MODEL_REPO = "adarshcod30/drdetect-dr-screening"

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename="best.ckpt")

    from drdetect.serve.demo import build_interface

    # regression loss, not ce: it won the locked external evaluation decisively
    # (docs/22_PHASE8_VALIDATION_RESULTS.md, referable AUC 0.924 vs 0.888, DeLong p=6.1e-10)
    demo = build_interface(
        checkpoint, backbone="efficientnet_b0", loss_name="regression", size=512, device="cpu"
    )
    demo.launch(ssr_mode=False)
