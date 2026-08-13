# Common commands for the credit default pipeline project. Run `make help`
# (or just `make`, since it's the first target) to list them.

.PHONY: help format lint test up down pull extract population seed migration \
        analytics train rag agent api ui pipeline_software \
        pipeline_software_alternative

PYTHON := uv run python

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-31s\033[0m %s\n", $$1, $$2}'

## --- Code quality ------------------------------------------------------

format: ## Format the codebase with Ruff
	uvx ruff format .

lint: ## Lint the codebase with Ruff, auto-fixing what it can
	uvx ruff check . --fix

test: ## Run the test suite with coverage
	uv run pytest tests --cov=. --cov-report=term-missing

## --- Docker --------------------------------------------------------------

up: ## Build and start the full stack (postgres-db, api, ui) in the background
	docker compose up --build -d

down: ## Stop and remove the stack's containers
	docker compose down

## --- Data pipeline -------------------------------------------------------

pull: ## Pull DVC-tracked data (parquet extracts) from remote storage
	uv run dvc pull

extract: ## Extract seeded OLTP data into parquet files for DVC (one-off, already done)
	$(PYTHON) -m analytics.ingestion.extract

population: ## Restore the OLTP database from DVC-tracked parquet extracts
	$(PYTHON) -m analytics.ingestion.restore

seed: ## Generate synthetic OLTP data from scratch with Faker (slow, ~2h)
	$(PYTHON) -m simulation.seed

migration: ## Apply Alembic migrations to bring the OLTP schema up to date
	uv run alembic upgrade head

dbt_setup: ## Set up direnv for interactive dbt work (one-off; requires direnv installed)
	cd analytics/dbt_project && cp -n .envrc.example .envrc && direnv allow

analytics: ## Build the dbt OLAP star schema
	cd analytics/dbt_project && set -a && . ../../.env && set +a && uv run dbt deps && uv run dbt build && cd ../..

## --- ML & agent ------------------------------------------------------

train: ## Train and evaluate every model family, logging to MLflow
	$(PYTHON) -m ml.run_training

rag: ## Extract project documentation and (re)build the agent's ChromaDB index
	$(PYTHON) -m agent.rag.extract_docs
	$(PYTHON) -m agent.rag.ingest

agent: ## Run the interactive CLI agent
	$(PYTHON) -m agent.run_agent

## --- Serving ------------------------------------------------------------

api: ## Run the FastAPI backend locally, with auto-reload
	uv run uvicorn api.main:app --reload

# Run `make api` in a separate terminal first — the UI is a thin HTTP
# client for the API and does nothing useful without it running.
ui: ## Run the Streamlit UI locally (requires `make api` running elsewhere)
	uv run streamlit run ui/app.py

## --- End-to-end pipelines ------------------------------------------------

software: ## Rebuild everything from DVC-tracked data, then start the stack
	docker compose up -d postgres-db
	@echo "Waiting for postgres-db to accept connections..."
	@until docker compose exec -T postgres-db pg_isready > /dev/null 2>&1; do sleep 1; done
	uv run alembic upgrade head
	uv run dvc pull
	$(PYTHON) -m analytics.ingestion.restore
	cd analytics/dbt_project && set -a && . ../../.env && set +a && uv run dbt deps && uv run dbt build && cd ../..
	$(PYTHON) -m ml.run_training
	$(PYTHON) -m agent.rag.extract_docs
	$(PYTHON) -m agent.rag.ingest
	docker compose up --build -d

# WARNING: this regenerates the OLTP data from scratch with Faker instead
# of restoring it from DVC, and has taken 2 hours to run just the seeding.
software_alternative: ## Rebuild everything from scratch (re-seeds data, slow), then start the stack
	docker compose up -d postgres-db
	@echo "Waiting for postgres-db to accept connections..."
	@until docker compose exec -T postgres-db pg_isready > /dev/null 2>&1; do sleep 1; done
	uv run alembic upgrade head
	$(PYTHON) -m simulation.seed
	cd analytics/dbt_project && set -a && . ../../.env && set +a && uv run dbt deps && uv run dbt build && cd ../..
	$(PYTHON) -m ml.run_training
	$(PYTHON) -m agent.rag.extract_docs
	$(PYTHON) -m agent.rag.ingest
	docker compose up --build -d
