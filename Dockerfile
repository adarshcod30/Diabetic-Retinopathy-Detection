# drdetect DR-screening API (docs/04_ROADMAP.md, Phase 9).
#
# CPU-only image, deliberately -- the whole serving path
# (drdetect.serve.pipeline.load_grader) is designed to run without a GPU, so
# a district server does not need one either. No architecture-specific base
# image or binary is pinned, so this builds for both amd64 and arm64 via:
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t drdetect-api .
#
# A single-arch local build is just `docker build -t drdetect-api .`
#
# Known simplification, stated rather than hidden: this installs the full
# `drdetect` package dependency set (pyproject.toml has one dependency list,
# not separate train/serve extras), which pulls in training-only packages
# (wandb, kaggle, hydra-core) the API never imports. Splitting that into a
# `[project.optional-dependencies].serve` group would shrink this image
# meaningfully; not done here to avoid restructuring a dependency file the
# rest of the project already depends on, this late in the project.
FROM python:3.11-slim AS base

# libgl1/libglib2.0-0: opencv-python-headless still needs these two shared
# libs at import time on Debian slim, even though "headless" means no GUI.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Model weights are NOT baked into the image (Messidor-2/APTOS licences
# restrict redistribution, and IDRiD/DRIVE weights are research-use only --
# see MODEL_CARD.md): mount a checkpoint at runtime instead.
ENV DRDETECT_CHECKPOINT=/models/checkpoint.ckpt
ENV DRDETECT_BACKBONE=efficientnet_b0
ENV DRDETECT_LOSS=ce
ENV DRDETECT_SIZE=512

EXPOSE 8000

CMD ["uvicorn", "drdetect.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
