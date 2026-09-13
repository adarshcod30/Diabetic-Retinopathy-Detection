"""DR grading backbones.

Two decisions here come from measurements rather than convention.

Frozen BatchNorm
----------------
Benchmarking on this project's target machine showed batch size is capped at 4
by available RAM (docs/05_PROTOTYPE_SCOPE.md section 6). EfficientNet uses
BatchNorm, whose running statistics are estimated per micro-batch -- at batch 4
they are noisy, and gradient accumulation does NOT help, because accumulation
batches the optimiser step, not the normalisation. Freezing BN to use the
pretrained ImageNet statistics is the standard fix for small-batch fine-tuning
and is the default here.

Ordinal-ready head
------------------
Phase 1 is a plain cross-entropy baseline, deliberately. But DR grades are
ordinal, and Phase 3 replaces the loss. `num_outputs` is therefore separated
from `num_classes` so the same backbone serves a 5-way softmax, a 1-output
regression head, or a 4-output CORAL head without rewriting the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "build_model",
    "freeze_batchnorm",
    "count_parameters",
    "backbone_norm_stats",
    "layer_wise_param_groups",
]

# Per-backbone pretrained normalisation stats, for backbones whose own
# pretrained_cfg differs from plain ImageNet (the default everywhere else in
# this project). RETFound's own config.json specifies 0.5/0.5/0.5, not the
# ImageNet mean/std baked into build_transforms' default.
_RETFOUND_MEAN_STD = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
_IMAGENET_MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def backbone_norm_stats(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(mean, std) a backbone was pretrained with, for build_transforms."""
    if "retfound" in name.lower():
        return _RETFOUND_MEAN_STD
    return _IMAGENET_MEAN_STD


def freeze_batchnorm(model: nn.Module) -> int:
    """Put every BatchNorm into eval mode and stop its affine parameters training.

    Returns the number of layers frozen, for logging -- a silent no-op here
    would be indistinguishable from a working call.
    """
    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)
            frozen += 1
    return frozen


class FundusGrader(nn.Module):
    def __init__(self, backbone: nn.Module, freeze_bn: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze_bn = freeze_bn
        if freeze_bn:
            self.n_frozen_bn = freeze_batchnorm(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def train(self, mode: bool = True):
        """Keep frozen BN in eval mode even when the module is set to train.

        Without this override, `model.train()` silently re-enables running-
        statistic updates and undoes the freeze -- a bug that produces no error
        and only shows up as unstable validation metrics.
        """
        super().train(mode)
        if self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


def build_model(
    name: str = "efficientnet_b0",
    *,
    num_outputs: int = 5,
    pretrained: bool = True,
    freeze_bn: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.0,
) -> FundusGrader:
    kwargs = {"drop_rate": drop_rate}
    if drop_path_rate:
        kwargs["drop_path_rate"] = drop_path_rate

    if name.startswith("hf_hub:"):
        backbone = _build_hf_hub_vit(name.removeprefix("hf_hub:"), num_outputs, pretrained, kwargs)
    else:
        import timm

        backbone = timm.create_model(name, pretrained=pretrained, num_classes=num_outputs, **kwargs)
    return FundusGrader(backbone, freeze_bn=freeze_bn)


def _hf_hub_download_retry(repo_id: str, filename: str, attempts: int = 12):
    """A cached file first, hf_hub_download() (retried with backoff) only if
    nothing local exists. Access to this specific repo has proven genuinely
    intermittent from this machine -- back-to-back calls with identical
    arguments sometimes 404 and sometimes succeed, unrelated to auth, import
    order, or cache state, and repeated calls appear to make it *more* likely
    to fail, not less (consistent with rate-limiting that a short retry loop
    cannot outlast). Since this file has already been pulled successfully at
    least once whenever the local snapshot exists, reading it directly avoids
    the network path (and its validation call) entirely rather than trusting
    huggingface_hub's own local-cache fallback, which has its own separate bug
    in HF_HUB_OFFLINE mode against this exact cache layout."""
    import glob
    import os
    import time

    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE

    org, name = repo_id.split("/")
    repo_cache = os.path.join(HF_HUB_CACHE, f"models--{org}--{name}")

    # Resolve the commit hash from refs/main, then try to actually OPEN the
    # candidate file rather than stat-checking it first with os.path.exists():
    # a separate existence check against this instance's storage has proven
    # unreliable in exactly the "check says no, but the file is right there"
    # way a stat/listing call can be on some overlay or network filesystems,
    # while directly attempting the read either plainly succeeds or fails.
    try:
        with open(os.path.join(repo_cache, "refs", "main")) as f:
            commit_hash = f.read().strip()
        candidate = os.path.join(repo_cache, "snapshots", commit_hash, filename)
        with open(candidate, "rb"):
            pass
        return candidate
    except OSError:
        pass

    for candidate in glob.glob(os.path.join(repo_cache, "snapshots", "*", filename)):
        try:
            with open(candidate, "rb"):
                pass
            return candidate
        except OSError:
            continue

    last_exc = None
    for attempt in range(attempts):
        try:
            path = hf_hub_download(repo_id, filename)
            if attempt:
                print(f"  {repo_id}/{filename}: succeeded on retry {attempt + 1}/{attempts}")
            return path
        except Exception as exc:  # noqa: BLE001 -- retry regardless of exact exception type
            last_exc = exc
            if attempt < attempts - 1:
                wait = min(2**attempt, 30)
                print(
                    f"  {repo_id}/{filename}: attempt {attempt + 1}/{attempts} failed "
                    f"({type(exc).__name__}), retrying in {wait}s"
                )
                time.sleep(wait)
    raise last_exc


def _build_hf_hub_vit(repo_id: str, num_outputs: int, pretrained: bool, kwargs: dict):
    """timm's own `hf_hub:` loading (timm.create_model's internal
    download_from_hf) fails in this project's exact timm/huggingface_hub
    version pairing -- it 404s even with a valid, working token, while
    huggingface_hub's own hf_hub_download() works correctly for the identical
    file, called directly. Confirmed by testing both against the same repo
    side by side. Rather than chase a dependency-version bug further, this
    fetches the same config.json/weights via the function that's actually
    proven to work, and builds the model timm's own way once the raw
    architecture name is known.

    HF_HUB_VIT_LOCAL_DIR: set this to a local directory already containing
    config.json/pytorch_model.bin (e.g. after manually placing files fetched
    from elsewhere) to skip huggingface_hub entirely. Added because this
    specific repo's reachability from this machine has proven bad enough,
    even with retries, that a fully local escape hatch is worth having --
    the network path below already retries hard on its own; this is for when
    even that isn't enough."""
    import json
    import os

    import timm
    import torch

    local_dir = os.environ.get("HF_HUB_VIT_LOCAL_DIR")
    if local_dir:

        def fetch(filename: str) -> str:
            return os.path.join(local_dir, filename)
    else:
        fetch = lambda filename: _hf_hub_download_retry(repo_id, filename)  # noqa: E731

    config_path = fetch("config.json")
    with open(config_path) as f:
        config = json.load(f)
    global_pool = config.get("pretrained_cfg", {}).get("global_pool", "token")

    backbone = timm.create_model(
        config["architecture"],
        pretrained=False,
        num_classes=num_outputs,
        global_pool=global_pool,
        **kwargs,
    )

    if pretrained:
        weights_path = fetch("pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        # the hub checkpoint's head is for num_classes=1000 (its own pretraining
        # task), not this project's num_outputs -- drop it rather than error on
        # a shape mismatch, exactly like build_model's own --init-checkpoint path
        # already does for this project's own checkpoints.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if set(missing) - {"head.weight", "head.bias"} or unexpected:
            print(f"  WARNING: hf_hub_vit load missing={missing} unexpected={unexpected}")

    return backbone


def _vit_layer_id(name: str, num_layers: int) -> int:
    """Which of a ViT's num_layers depth-groups a parameter belongs to.
    Mirrors RETFound's own reference fine-tuning recipe (util/lr_decay.py,
    itself following BEiT): patch_embed/cls_token/pos_embed are layer 0 (the
    earliest, most general features), blocks.N is layer N+1, and everything
    else (the final norm and the classification head) is the last layer --
    full learning rate, since that's what actually needs to adapt to a new task."""
    if name in ("cls_token", "pos_embed", "dist_token", "reg_token"):
        return 0
    if name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers


def layer_wise_param_groups(
    backbone, lr: float, weight_decay: float, layer_decay: float
) -> list[dict]:
    """Per-layer parameter groups for fine-tuning a pretrained ViT, following
    RETFound's own documented recipe (github.com/rmaphoh/RETFound_MAE,
    util/lr_decay.py) rather than the single flat learning rate this project
    otherwise uses for its CNN backbones. A pretrained transformer's early
    layers already encode general, broadly-useful features from its own
    pretraining; a single learning rate for the whole network is simultaneously
    too aggressive for those layers (risking destroying them) and too timid for
    the later layers that actually need to adapt -- layer_decay < 1 gives early
    layers a much smaller effective learning rate than late ones.

    Requires a ViT-style backbone (patch_embed/cls_token/pos_embed/blocks.N.*
    naming, timm's own convention) -- not meant for the CNN backbones, which
    keep configure_optimizers' existing flat-lr path.
    """
    num_layers = len(backbone.blocks) + 1
    no_decay_names = backbone.no_weight_decay() if hasattr(backbone, "no_weight_decay") else set()

    groups: dict[str, dict] = {}
    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        decay = 0.0 if (param.ndim == 1 or name in no_decay_names) else weight_decay
        layer_id = _vit_layer_id(name, num_layers)
        key = f"layer_{layer_id}_{'decay' if decay else 'no_decay'}"
        if key not in groups:
            scale = layer_decay ** (num_layers - layer_id)
            groups[key] = {"params": [], "lr": lr * scale, "weight_decay": decay}
        groups[key]["params"].append(param)
    return list(groups.values())


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
