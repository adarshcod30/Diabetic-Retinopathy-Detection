.PHONY: help setup test test-all lint format clean bench data preprocess train evaluate demo sim

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install the package with dev extras (idempotent)
	uv venv --python 3.11 .venv --allow-existing
	uv pip install --python .venv/bin/python -e ".[dev]"
	.venv/bin/pre-commit install

test:  ## Run the fast test suite (no training)
	.venv/bin/python -m pytest tests/ -v -m "not slow"

test-all:  ## Run every test, including the end-to-end training run
	.venv/bin/python -m pytest tests/ -v

lint:  ## Lint and format-check
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:  ## Auto-format
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

bench:  ## Measure real training throughput before fixing the scope
	.venv/bin/python scripts/benchmark_device.py

data:  ## Download the datasets that fit locally (NOT EyePACS)
	bash scripts/download_data.sh --datasets aptos,idrid,drive

preprocess:  ## Cache APTOS at 512px (10GB -> ~200MB), then Messidor-2 and IDRiD's locked test split
	.venv/bin/python scripts/preprocess.py --dataset aptos --size 512
	.venv/bin/python scripts/preprocess.py --dataset messidor2 --size 512
	.venv/bin/python scripts/preprocess.py --dataset idrid --idrid-split test --size 512

train:  ## Run a training experiment (override any flag with ARGS="--size 384 --loss corn")
	.venv/bin/python scripts/train.py $(ARGS)

evaluate:  ## Reproduce the headline table on the locked external test set (Messidor-2 + IDRiD)
	.venv/bin/python scripts/evaluate_external.py \
		--checkpoint $(or $(CHECKPOINT),models/checkpoints/cv_baseline_fold1/best.ckpt) \
		--bootstrap 2000 --i-understand-this-runs-once

demo:  ## Launch the Gradio demo (override checkpoint with CHECKPOINT=...)
	.venv/bin/python scripts/demo.py --checkpoint $(or $(CHECKPOINT),models/checkpoints/cv_baseline_fold1/best.ckpt)

sim:  ## Run the district screening simulation
	.venv/bin/python -m simulation.simpy.district --patients-per-year 100000

clean:  ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
