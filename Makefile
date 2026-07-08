.PHONY: help install sample ingest features cluster validate search pipeline test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package + dev tooling in editable mode
	pip install -e ".[search,dev]"

sample:  ## Generate a synthetic master parquet (no real data needed)
	python scripts/make_sample_dataset.py

ingest:  ## Build master parquet from raw .mat files
	python scripts/run_ingest.py

features:  ## Build the model-ready feature table
	python scripts/run_features.py

cluster:  ## Fit UMAP + HDBSCAN and label the data
	python scripts/run_clustering.py

validate:  ## Score the clustering + preservation metrics
	python scripts/run_validation.py

search:  ## Run the hyperparameter search
	python scripts/run_validation.py --search

pipeline: sample features cluster validate  ## Full demo run on synthetic data

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Format with black + ruff --fix
	black src tests scripts
	ruff check --fix src tests

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache outputs
	find . -type d -name __pycache__ -exec rm -rf {} +
