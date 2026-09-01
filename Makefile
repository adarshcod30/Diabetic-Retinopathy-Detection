.PHONY: help setup test lint format clean bench data preprocess train evaluate demo sim

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install the package with dev extras (idempotent)
	uv venv --python 3.11 .venv --allow-existing
	uv pip install --python .venv/bin/python -e ".[dev]"
	.venv/bin/pre-commit install

test:  ## Run the test suite
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

preprocess:  ## Cache APTOS at 512px (10GB -> ~200MB)
	.venv/bin/python scripts/preprocess.py --dataset aptos --size 512 --pipeline bengraham

train:  ## Run a training experiment (override with EXP=...)
	.venv/bin/python scripts/train.py experiment=$(or $(EXP),grading_baseline)

evaluate:  ## Evaluate on the locked external test set
	.venv/bin/python scripts/evaluate.py --split external_test --bootstrap 2000

demo:  ## Launch the Gradio demo
	.venv/bin/python -m drdetect.serve.demo

sim:  ## Run the district screening simulation
	.venv/bin/python -m simulation.simpy.district --patients-per-year 100000

clean:  ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
