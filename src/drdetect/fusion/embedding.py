"""CNN embedding extraction from the canonical grading checkpoint, for
Phase 5's fusion head.

Uses `baseline_effb0_512_fold0` (QWK 0.8930), not the 1024px checkpoint --
Phase 3's own conclusion was "the baseline stands" (docs/07_PHASE3_RESULTS.md),
and Result 14 retracted the 1024px checkpoint's causal story (its edge over
the baseline is now suspected to be a backend artifact, not resolution).
Building the fusion head on a result the project already walked back would
undercut that finding.

Reuses the checkpoint-loading convention scripts/evaluate.py and
scripts/calibrate.py already established (strip the Lightning `model.`
prefix, load into a bare `FundusGrader`) rather than inventing a new one.

**Images come from the Ben-Graham-preprocessed cache
(`data/processed/aptos_512/`), not the raw APTOS files** -- unlike
`drdetect.fusion.features`, which deliberately uses raw images because the
Phase 4 lesion models were trained on IDRiD's un-enhanced originals. The
grading model was trained on the preprocessed cache, so its embedding is
only meaningful computed on the same distribution it actually learned from.
Two different "the right image for this model" answers for two different
models feeding the same fusion vector -- not an inconsistency.
"""

from __future__ import annotations

from pathlib import Path

import torch

from drdetect.data.dataset import build_transforms
from drdetect.grading.model import FundusGrader, build_model

__all__ = ["load_grading_backbone", "extract_cnn_embedding", "EMBEDDING_DIM"]

EMBEDDING_DIM = 1280  # efficientnet_b0's pre-classifier feature width


def load_grading_backbone(
    checkpoint: str, *, backbone: str = "efficientnet_b0", device: torch.device
) -> FundusGrader:
    model = build_model(backbone, num_outputs=5, pretrained=False, freeze_bn=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state)
    return model.to(device).eval()


def extract_cnn_embedding(
    image_path: str | Path, model: FundusGrader, device: torch.device, *, size: int = 512
) -> torch.Tensor:
    """The 1280-d pre-classifier pooled feature (`forward_head(...,
    pre_logits=True)`), not the 5-way logits `model(x)` would give."""
    embedding, _logits = extract_cnn_embedding_and_logits(image_path, model, device, size=size)
    return embedding


def extract_cnn_embedding_and_logits(
    image_path: str | Path, model: FundusGrader, device: torch.device, *, size: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both the fusion head's input (the embedding) and the baseline
    grading-only prediction (the 5-way logits) from one forward pass --
    used when comparing fusion against grading-alone on the same images, so
    the baseline's own prediction doesn't need a second pass over the model."""
    import cv2

    transform = build_transforms(size, train=False)
    image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.backbone.forward_features(tensor)
        embedding = model.backbone.forward_head(features, pre_logits=True)
        logits = model.backbone.forward_head(features)
    return embedding.squeeze(0).cpu(), logits.squeeze(0).cpu()
