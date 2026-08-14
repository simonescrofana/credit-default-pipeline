# Enterprise Credit Default Pipeline & GenAI Analyst

An end-to-end, production-grade MLOps and Data Engineering pipeline designed to predict corporate insolvency on highly imbalanced financial and transactional data. Features a decoupled storage architecture, automated experiment tracking, perimetric data validation, and a Generative AI assistant agent integrated with Explainable AI (xAI) metrics.

---

## 🏗️ System Architecture

The system is engineered using a strictly decoupled, multi-layered architecture to process data securely from ingestion to intelligent, explainable inference:

1. **Transactional Layer (OLTP):** Containerized PostgreSQL instance simulating a production core-banking system managed via SQLAlchemy ORM and tracked through Alembic migrations.
2. **Analytical Layer (OLAP):** Dimensional Data Warehouse modeled into a Star Schema driven by dbt Core over historical immutable ledgers, with point-in-time correctness enforced through SCD Type 2 dimensions and as-of temporal joins.
3. **MLOps & Lifecycle Layer:** Data version control implemented with DVC. Multi-model training pipeline (a cost-sensitive baseline, Gradient Boosted Trees, and a PyTorch Neural Network) integrated with MLflow for artifact logging, hyperparameter tracking, and model registry.
4. **Explainable AI (xAI) Module:** Interpretability extraction utilizing SHAP to ensure credit scoring compliance and transparency.
5. **Generative AI Layer:** An agent system built via LangGraph acting as an autonomous financial analyst, querying a ChromaDB vector store, running local inference, and validated by an LLM-as-a-Judge node.
6. **Application & Serving Layer:** A FastAPI backend exposing validated REST endpoints for predictions and chat interactions, paired with a Streamlit interface acting as a live, interactive demo of the full pipeline.

---

## 🚀 Getting Started

This section walks through everything needed to go from a fresh clone to the full stack running, on a clean Ubuntu/WSL2 machine.

### 1. Install prerequisites

* **[`uv`](https://docs.astral.sh/uv/)**, the project's package and environment manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
* **Docker Engine**, via the official convenience script rather than `apt install docker` (that package name doesn't resolve on Ubuntu):
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER   # then reboot the WSL/Ubuntu for this to take effect
```
* `make` and `git`, usually already present on Ubuntu/WSL2; if not, `sudo apt install -y make git`.
* **[`direnv`](https://direnv.net/)** (optional — only needed for working inside `analytics/dbt_project/` interactively, not for anything under `make`):
```bash
sudo apt install -y direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc && source ~/.bashrc
```

### 2. Clone and configure

```bash
git clone https://github.com/simonescrofana/credit-default-pipeline
cd credit-default-pipeline
cp .env.example .env   # then fill it in with your own values
```

* **DVC remote credentials** — this project's DVC remote (DagsHub) requires authentication that is never stored in the repo. Set it once, locally:
```bash
uv run dvc remote modify --local origin auth basic
uv run dvc remote modify --local origin user <your-dagshub-username>
uv run dvc remote modify --local origin password <your-dagshub-token>
```
  Generate a token under your DagsHub profile → Settings → Tokens.
* **dbt profile** — dbt reads connection details from `~/.dbt/profiles.yml`, a per-machine file that is never part of the repo either:
```bash
mkdir -p ~/.dbt
cat > ~/.dbt/profiles.yml << 'EOF'
dbt_project:
  outputs:
    dev:
      dbname: "{{ env_var('POSTGRES_DB', 'insolvency_db') }}"
      host: "{{ env_var('POSTGRES_HOST', 'localhost') }}"
      pass: "{{ env_var('POSTGRES_PASSWORD') }}"
      port: "{{ env_var('POSTGRES_PORT', '5433') | as_number }}"
      schema: public
      threads: 1
      type: postgres
      user: "{{ env_var('POSTGRES_USER') }}"
  target: dev
EOF
```

#### 3. Run the pipeline

`make` (or `make help`) lists every available command. Two end-to-end pipelines rebuild the whole project and bring up the full stack (Postgres, API, UI) in one shot:
⚠️ WARNING: Running the first fast pipeline took almost 1 hour on my hardware AMD Ryzen 7 8845HS, 16GB RAM, 512 GB SSD M.2 PCIe Gen4.

```bash
make software              # restores data from DVC - fast
```
```bash
make software_alternative  # regenerates data from scratch with Faker - slow, ~2h+ just for seeding
```

Either one runs migrations, loads or generates the OLTP data, builds the dbt star schema, trains and evaluates every model, builds the agent's documentation index, and finally starts `postgres-db`, `api`, and `ui` via Docker Compose. Once it finishes, the API is at `http://localhost:8000/docs` and the UI at `http://localhost:8501`, exactly as described above under Try the API and Try the UI.

For working on one piece at a time instead, see `make help` for the individual steps (`migration`, `population`/`seed`, `analytics`, `train`, `rag`, `up`/`down`, ...) that the two pipelines above are built from.

#### Troubleshooting: containers can't reach the internet

If a running container (e.g. `api` failing to reach Groq or Logfire, timing out instead of erroring quickly) can't reach the outside world even though the host itself has working internet access, this is a known Docker-on-WSL2 networking issue (native Docker Engine, not Docker Desktop) — it can show up the first time Docker starts after installation, or after the host's network state changes (e.g. sleep/resume, reconnecting Wi-Fi). Restarting the Docker service resolves it:

```bash
sudo service docker restart
docker compose down
docker compose up -d
```

---

## 🤖 Agent Graph

The Generative AI layer is built as an explicit LangGraph state machine. A router node classifies every incoming request into one of four routes; all four converge on a shared LLM response node. The two prediction routes (existing or ad hoc company) both pass through a shared extraction and prediction step first. A generic, unrouted question skips validation entirely and returns directly; every other response is checked by an LLM-as-a-Judge node, which can send it back to be regenerated up to a fixed retry cap, past which a static fallback message is returned instead:

```mermaid
graph TD
    START([User Prompt]) --> ROUTER{Router}

    ROUTER -->|Prediction for existing company| EXTRACTOR_A[Extractor<br/>Case A]
    ROUTER -->|Prediction from ad hoc data| EXTRACTOR_B[Extractor<br/>Case B]
    ROUTER -->|Documentation/project question| RAG[RAG Node]
    ROUTER -->|Generic question| RISPOSTA[LLM Response Node]

    EXTRACTOR_A --> PREDICTOR[Predictor Node]
    EXTRACTOR_B --> PREDICTOR
    PREDICTOR --> RISPOSTA
    RAG --> RISPOSTA

    RISPOSTA -->|Generic question| END([Final Response])
    RISPOSTA -->|Everything else| JUDGE{LLM-as-a-Judge}
    JUDGE -->|Approve| END
    JUDGE -->|Reject, under retry cap| RISPOSTA
    JUDGE -->|Reject, retry cap reached| FALLBACK[Fixed Fallback Message]
    FALLBACK --> END
```

---

## 💬 Try the Agent

Ready-to-use example prompts for every route the agent handles are available under [`docs/prompts/`](docs/prompts/), in both Italian and English. Run the agent with:
 
```bash
uv run python -m agent.run_agent
```

---

## 🚀 Try the API

The same agent and model are also served over HTTP, alongside two direct, LLM-independent prediction endpoints. Start the server with:

```bash
uv run uvicorn api.main:app --reload
```

Then browse the interactive, auto-generated documentation at [`http://localhost:8000/docs`](http://localhost:8000/docs) (Swagger UI, supports sending real requests directly from the browser) or [`http://localhost:8000/redoc`](http://localhost:8000/redoc), or exercise the endpoints directly with `curl`. The example company used below, `"Sistemi Tamburello S.r.l."`, is part of the seeded database.

⚠️ The commands below pipe the response through `jq` for pretty-printed JSON. Install it first (`sudo apt install jq` / `brew install jq`), or drop `| jq` from any command to see the raw response instead.

```bash
# Health check
curl http://localhost:8000/health | jq

# Chat - first turn, no Session-Id header yet
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qual è il rischio di insolvenza di Sistemi Tamburello S.r.l.?"}' | jq

# Chat - reuse the session_id returned above to continue the conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Session-Id: <session_id from the previous response>" \
  -d '{"message": "Quali sono le ragioni principali che hanno portato a questa predizione?"}' | jq

# Direct prediction on ad hoc data (not in the database)
curl -X POST http://localhost:8000/predict/ad-hoc \
  -H "Content-Type: application/json" \
  -d '{
    "unpaid_ratio_trailing_90d": 0.15,
    "total_outstanding_debt": 12000,
    "foundation_date": "2015-03-01",
    "industry_sector": "manufacturing",
    "registered_office_region": "Lazio"
  }' | jq

# Direct prediction on an existing company, looked up by legal name or VAT number
curl -X POST http://localhost:8000/predict/company \
  -H "Content-Type: application/json" \
  -d '{"identifier": "Sistemi Tamburello S.r.l."}' | jq
```

---

## 🖥️ Try the UI

A Streamlit interface wraps the conversational endpoint for terminal-free, non-technical use: a sidebar lists past conversations from the current browser session, and the main panel renders the agent's Markdown-formatted answers instead of raw text. Only the conversational endpoint is surfaced here; direct prediction targets callers who already have structured data and are expected to use the API directly, as shown above.

With the API still running (see Try the API above), start the UI in a separate terminal:

```bash
uv run streamlit run ui/app.py
```

It opens at [`http://localhost:8501`](http://localhost:8501) by default.

---

## 🛠️ Tech Stack

* **Infrastructure & DevOps:** Docker, Docker Compose, GitHub Actions (CI)
* **Environment & Package Management:** Python, uv
* **Data Engineering & Storage:** PostgreSQL, SQLAlchemy, Alembic, dbt Core
* **Data Versioning:** DVC
* **Machine Learning Engines:** PyTorch, Scikit-Learn, XGBoost
* **Explainable AI (xAI):** SHAP
* **MLOps & Model Tracking:** MLflow
* **QA & Enterprise Validation:** Pytest, Pydantic Validation
* **Configuration Management:** Pydantic Settings
* **Observability & Logging:** Pydantic Logfire, Ruff
* **Generative AI Infrastructure:** LangGraph, ChromaDB, Groq
* **Application Layer & UI:** FastAPI, Streamlit

---

## 📂 Project Structure (up to this moment)

```text
insolvency_prediction_project/
├── .dvc/
│   ├── .gitignore
│   └── config
├── .github/
│   ├── workflows/
│   │   └── main.yml
│   └── pull_request_template.md
├── agent/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_judge.py
│   │   └── llm_responder.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   ├── judge.py
│   │   ├── predictor_node.py
│   │   ├── responder.py
│   │   ├── retriever_node.py
│   │   └── router.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── extractor_prompt.py
│   │   ├── judge_prompt.py
│   │   ├── responder_prompt.py
│   │   └── router_prompt.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── extract_docs.py
│   │   ├── ingest.py
│   │   └── retrieval.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── llm_utils.py
│   ├── __init__.py
│   ├── graph.py
│   ├── run_agent.py
│   └── state.py
├── analytics/
│   ├── dbt_project/
│   │   ├── analyses
│   │   │   └── .gitkeep
│   │   ├── macros
│   │   │   └── generate_company_key.sql
│   │   ├── models/
│   │   │   ├── intermediate/
│   │   │   │   ├── int_billing_trailing_90d.sql
│   │   │   │   ├── int_companies_scd_resolved.sql
│   │   │   │   ├── int_company_date_spine.sql
│   │   │   │   ├── int_contracts_asof.sql
│   │   │   │   ├── int_financial_asof.sql
│   │   │   │   ├── int_insolvency_label.sql
│   │   │   │   ├── int_logins_trailing.sql
│   │   │   │   ├── int_tickets_trailing.sql
│   │   │   │   └── schema.yml
│   │   │   ├── marts/
│   │   │   │   ├── dim_companies.sql
│   │   │   │   ├── dim_date.sql
│   │   │   │   ├── fct_company_credit_profile.sql
│   │   │   │   └── schema.yml
│   │   │   └── staging/
│   │   │       ├── schema.yml
│   │   │       ├── sources.yml
│   │   │       ├── stg_companies.sql
│   │   │       ├── stg_crm_support_tickets.sql
│   │   │       ├── stg_energy_contracts.sql
│   │   │       ├── stg_financial_statements.sql
│   │   │       ├── stg_invoices.sql
│   │   │       ├── stg_payments.sql
│   │   │       └── stg_user_web_logins.sql
│   │   ├── seeds/
│   │   │   └── .gitkeep
│   │   ├── snapshots/
│   │   │   ├── companies_snapshot.sql
│   │   │   └── energy_contracts_snapshot.sql
│   │   ├── tests/
│   │   │   ├── intermediate/
│   │   │   │   ├── int_billing_trailing_90d/
│   │   │   │   │   ├── int_billing_trailing_90d_debt_ratio_match.sql
│   │   │   │   │   ├── int_billing_trailing_90d_dpd_consistency.sql
│   │   │   │   │   └── int_billing_trailing_90d_no_future_leakage.sql
│   │   │   │   ├── int_companies_scd_resolved/
│   │   │   │   │   ├── int_companies_scd_resolved_chronology.sql
│   │   │   │   │   ├── int_companies_scd_resolved_expired_versions.sql
│   │   │   │   │   └── int_companies_scd_resolved_leakage.sql
│   │   │   │   ├── int_company_date_spine/
│   │   │   │   │   ├── int_company_date_spine_no_dates_before_foundation.sql
│   │   │   │   │   ├── int_company_date_spine_no_future_dates.sql
│   │   │   │   │   ├── int_company_date_spine_np_gaps.sql
│   │   │   │   │   └── int_company_date_spine_respects_valid_to.sql
│   │   │   │   ├── int_contracts_asof/
│   │   │   │   │   ├── int_contracts_asof_count_flag_consistency.sql
│   │   │   │   │   └── int_contracts_asof_no_future_leakage.sql
│   │   │   │   ├── int_financial_asof/
│   │   │   │   │   ├── int_financial_asof_publication_delay_leakage.sql
│   │   │   │   │   └── int_financial_asof_rank_recency.sql
│   │   │   │   ├── int_insolvency_label/
│   │   │   │   │   ├── int_insolvency_label_false_negative.sql
│   │   │   │   │   └── int_insolvency_label_false_positive.sql
│   │   │   │   ├── int_logins_trailing/
│   │   │   │   │   ├── int_logins_trailing_null_consistency.sql
│   │   │   │   │   ├── int_logins_trailing_recency_boundary.sql
│   │   │   │   │   └── int_logins_trailing_velocity_coherence.sql
│   │   │   │   └── int_tickets_trailing_90d/
│   │   │   │       └── int_tickets_trailing_90d_no_future_leakage.sql
│   │   │   └── marts/
│   │   │       ├── dim_companies/
│   │   │       │   ├── dim_companies_no_overlapping_windows.sql
│   │   │       │   └── dim_companies_single_current_version.sql
│   │   │       ├── dim_date/
│   │   │       │   └── dim_date_no_gaps.sql
│   │   │       └── fct_company_credit_profile/
│   │   │           ├── fct_company_key_temporal_correctness.sql
│   │   │           ├── fct_no_dropped_spine_rows.sql
│   │   │           └── fct_no_unexpected_nulls.sql
│   │   ├── .envrc
│   │   ├── .envrc.example
│   │   ├── .gitignore
│   │   ├── dbt_project.yml
│   │   ├── package-lock.yml
│   │   ├── packages.yml
│   │   └── README.md
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── extract.py
│   │   └── restore.py
│   └── __init__.py
├── api/ 
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   └── predict.py
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   └── session_store.py
├── data/
│   ├── .gitignore
│   └── raw.dvc
├── database/
│   ├── migrations/
│   │   ├── versions/
│   │   │   ├── 4c84a2bf5287_feat_create_database_structure.py
│   │   │   ├── 7f2797ec0404_feat_create_database_structure.py
│   │   │   └── c1cf595229f7_feat_create_database_structure_really_.py
│   │   ├── env.py
│   │   ├── README
│   │   └── script.py.mako
│   ├── __init__.py
│   ├── base.py
│   ├── connection.py
│   ├── credit-default-database.sql
│   ├── models.py
│   └── types.py
├── docs/
│   ├── images/
│   │   ├── credit-default-database.pdf
│   │   ├── credit-default-DFM.pdf
│   │   ├── credit-default-star-schema.pdf
│   │   └── dag-dbt.jpg
│   ├── prompts/
│   │   ├── case_a/
│   │   │   ├── case_a_en.md
│   │   │   └── case_a_it.md
│   │   ├── case_b/
│   │   │   ├── case_b_en.md
│   │   │   └── case_b_it.md
│   │   ├── direct/
│   │   │   ├── direct_en.md
│   │   │   └── direct_it.md
│   │   └── rag/
│   │       ├── rag_en.md
│   │       └── rag_it.md
│   └── schema/
│       ├── credit-default-DFM.sql
│       ├── credit-default-star-schema.sql
│       └── database_structure.sql
├── ml/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── split.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── explainability.py
│   │   ├── metrics.py
│   │   └── plots.ipynb
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── predictor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── mlp.py
│   │   ├── protocol.py
│   │   └── xgboost_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── mlflow_utils.py
│   │   └── trainer.py
│   ├── tuning/
│   │   ├── results/
│   │   │   ├── baseline_results.csv
│   │   │   ├── mlp_results.csv
│   │   │   ├── xgboost_fine_results.csv
│   │   │   └── xgboost_results.csv
│   │   └── scripts
│   │       ├── fine_tune_xgb.sh
│   │       ├── tune_baseline.sh
│   │       ├── tune_mlp.sh
│   │       └── tune_xgb.sh
│   ├── __init__.py
│   └── run_training.py
├── schemas/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── extraction_validation.py
│   │   ├── judge_validation.py
│   │   ├── route_validation.py
│   │   └── types.py
│   ├── api/ 
│   │   ├── routers/ 
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   └── predict.py
│   │   ├── __init__.py
│   │   └── session_validation.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── models_validation.py
│   │   └── types.py
│   ├── ml/
│   │   ├── __init__.py
│   │   └── insolvency_prediction.py
│   └── __init__.py
├── simulation/
│   ├── __init__.py
│   ├── profiles.py
│   └── seed.py
├── tests/
│   ├── agent/
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── test_llm_judge.py
│   │   │   └── test_llm_responder.py
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── test_extractor.py
│   │   │   ├── test_judge.py
│   │   │   ├── test_predictor_node.py
│   │   │   ├── test_responder.py
│   │   │   ├── test_retriever_node.py
│   │   │   └── test_router.py
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── test_extract_docs.py
│   │   │   ├── test_ingest.py
│   │   │   └── test_retrieval.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   └── test_llm_utils.py
│   │   ├── __init__.py
│   │   ├── test_graph.py
│   │   └── test_run_agent.py
│   ├── analytics/
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── test_extract.py
│   │   │   └── test_restore.py
│   │   └── __init__.py
│   ├── api/
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── test_chat.py
│   │   │   └── test_predict.py
│   │   ├── __init__.py
│   │   ├── test_dependencies.py
│   │   ├── test_main.py
│   │   └── test_session_store.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── test_connection.py
│   │   └── test_models.py
│   ├── ml/
│   │   ├── dataset/
│   │   │   ├── __init__.py
│   │   │   ├── test_loader.py
│   │   │   ├── test_preprocessing.py
│   │   │   └── test_split.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── test_explainability.py
│   │   │   └── test_metrics.py
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── test_model_loader.py
│   │   │   └── test_predictor.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── test_baseline.py
│   │   │   └── test_mlp.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   └── test_trainer.py
│   │   ├── __init__.py
│   │   └── test_run_training.py
│   ├── schemas/
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   └── test_models_validation.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   └── test_insolvency_prediction.py
│   │   └── __init__.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── test_seed.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── test_app.py
│   │   ├── test_chat_registry.py
│   │   └── test_client.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── test_date_validation.py
│   │   └── test_timezone_utils.py
│   ├── __init__.py
│   └── conftest.py
├── ui/
│   ├── __init__.py
│   ├── app.py
│   ├── chat_registry.py
│   └── client.py 
├── utils/
│   ├── __init__.py
│   ├── date_validation.py
│   ├── logging_utils.py
│   ├── queries.py
│   └── timezone_utils.py
├── .dvcignore
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── alembic.ini
├── config.py
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 📈 Pipeline Highlights & Implementation Details

#### 1. Robust Data Ingestion & Semantic Modeling
* Modeled transactional entities using clean Object-Relational Mapping (SQLAlchemy) to enforce structural constraints at the application layer.
* Decoupled structural evolutions using database migration tracking (Alembic) to avoid manual, destructive changes on raw schemas.
* Orchestrated data warehouse compilation using dbt Core, shifting processing from raw ledgers into a fully optimized star schema layout (`fct_company_credit_profile`), with embedded enterprise data-quality constraints (`unique`, `not_null`, `relationships`, plus custom leakage and temporal-correctness tests) running at the analytical border.

#### 2. Multi-Model Benchmark & MLOps Governance
To solve the severe class imbalance typical of credit default data, the orchestration engine isolates and profiles three alternative algorithms:
* **Baseline Model:** Logistic Regression optimized via cost-sensitive class weighting.
* **Tree-based Ensemble:** Optimized XGBoost engines tuned for highly skewed feature split distributions, leveraging native handling of missing values.
* **Deep Learning Neural Network:** A custom PyTorch Multi-Layer Perceptron (funnel architecture with dropout regularization) trained with a class-weighted `BCEWithLogitsLoss` and `AdamW`, exposing a `fit`/`predict_proba` interface so the training orchestrator treats it identically to the scikit-learn-based models.
* **Model-Agnostic Orchestration:** A structural `Estimator` protocol (`fit`/`predict_proba`) — rather than a shared base class — lets the training pipeline remain entirely unaware of whether a given model is scikit-learn- or PyTorch-based. The orchestrator additionally inspects each model's `fit` signature at runtime to opportunistically pass a validation split for per-epoch monitoring where supported (the PyTorch model), without requiring every model family to accept it.
* **Imbalance Strategy:** Class weighting is applied consistently across all three model families, combined with decision threshold tuning at evaluation time to align the precision/recall trade-off with the real-world cost asymmetry of missed insolvencies. The simulated dataset carries a substantial positive rate (~16%), measured directly from the populated star schema rather than assumed, and treated as a genuine benchmark condition rather than an artificially rebalanced one.
* **Governance:** Operational artifacts, model weights, validation graphs, and metrics (Precision-Recall AUC, ROC curves, confusion matrix counts) are automatically serialized and registered into **MLflow**, with cross-validation runs nested under a parent run per model family and a size-weighted aggregation of per-fold metrics. The fitted preprocessing artifacts (encoder, scaler) are persisted alongside each final model, and per-epoch training/validation loss is logged for the neural network to support manual early-stopping analysis. Raw datasets are locked and version-controlled via **DVC**.
* **Data Quality Discovery:** While validating the ML feature set end-to-end, a bug was traced to the `int_billing_trailing_90d` dbt model: the 90-day trailing window used to select candidate invoices excluded unpaid invoices right before their days-past-due value could reach the 90-day insolvency threshold, silently producing zero labeled insolvencies across the entire mart. The window was widened to 120 days — the minimum needed to reliably capture the threshold under monthly snapshots — without altering the underlying days-past-due calculation.
* **Near-Leakage Feature Removal:** An unexpectedly high test-set AUC-PR (0.988 for XGBoost) prompted a targeted investigation rather than acceptance at face value. Removing `avg_dpd_trailing_90d` from the imported data from the central fact of the star schema alone dropped AUC-PR to 0.691 (XGBoost), 0.633 (baseline), and 0.499 (MLP) - a consistent, comparable drop across all three model families - confirming that a single feature, strongly correlated (Pearson 0.80) with the deterministic label-generating column `max_dpd_trailing_90d`, was dominating the shared predictive signal rather than the model families learning it independently. The feature was removed from the training set entirely, trading a superficially higher score for a result that reflects the model's actual ability to generalize from independent risk signals.
* **Expected Test Warnings:** The test suite intentionally exercises a validation fold with no positive examples, a realistic edge case under severe class imbalance. This triggers `UndefinedMetricWarning`/`UserWarning` from scikit-learn's AUC-ROC and AUC-PR implementations, which are mathematically undefined when only one class is present. These warnings are expected and left unsuppressed by design, so they remain visible as a reminder of the underlying edge case rather than being silently filtered out.

#### 3. Explainable AI & Agentic Conversational Loops
* Instead of rendering opaque black-box credit predictions, the pipeline processes validation instances through **SHAP** (Shapley Additive exPlanations) to map exact feature attributions for every prediction.
* Conversational interaction is driven by a stateful **LangGraph** engine, built router-first: every request is classified into exactly one of four routes before any downstream work happens, rather than always retrieving a company profile or querying the vector store "just in case." A request about a specific or hypothetical company retrieves its credit profile and prediction with full SHAP attributions; a documentation or methodology question instead retrieves contextual text chunks from a **ChromaDB** vector store; a general conversational message skips both entirely.
* To secure enterprise responses, an autonomous **LLM-as-a-Judge** node, hosted separately from the response-generating model, scores the generated reply against the exact material it was based on to catch and mitigate hallucinations before the payload reaches the UI, within a bounded retry loop that falls back to a fixed message rather than looping indefinitely.

#### 4. Inference Layer
* Serving reuses the exact fitted encoder and scaler produced during training — loaded together with the model from the same MLflow run — instead of refitting preprocessing transformations on new data, preventing training/serving skew.
* Scoring an existing company retrieves its most recent star schema snapshot through a parameterized query (never string-interpolated), keeping the same feature-selection logic used during training as the single source of truth.
* Scoring a company not present in the database (arbitrary user/agent-supplied data) is validated through a dedicated Pydantic schema before it reaches the model, see the SHAP-driven validation entry under Architectural Decisions below, then shares the same encoding, scoring, and SHAP explanation path as an existing company.

#### 5. Serving Layer
* The agent and the model are exposed over HTTP through a REST API, sitting alongside the existing command-line interface rather than replacing it: a conversational endpoint forwards free-text messages through the full agent graph, while two direct endpoints call the model without going through the LLM at all, for callers who already have structured data or want a deterministic, low-latency path.
* Every request and response body is validated through dedicated schemas, the same discipline already applied to the agent's own internal state, and the API's auto-generated interactive documentation is built directly from those same schemas, kept accurate by construction rather than maintained separately by hand.
* Conversations are tracked per client-supplied session, isolating concurrent users from one another, while the agent and the loaded model are built once at startup and reused across every request rather than reconstructed on each call.
* A terminal-free interface consumes the conversational endpoint alone, rendering the agent's Markdown-formatted answers properly instead of as raw text, and keeps a lightweight, client-side index of past conversations for its own sidebar, always reading the underlying messages back from the server rather than keeping a second, independent copy of them.

---

## 🧠 Architectural Decisions & Rationale

#### Modern Python Tooling (Python 3.13 + `uv` + Ruff)
* **Choice:** Moving away from standard `pip`/`venv` and selecting `uv` as the exclusive dependency manager alongside Ruff for quality gating.
* **Justification:** `uv` provides blazing-fast environment synchronization and strict, deterministic lockfile management, eliminating the "it works on my machine" anti-pattern in production containers. Ruff guarantees lightning-fast code linting and formatting compliance natively during pre-commit and CI stages.

#### Complete Isolation of OLTP and OLAP Store
* **Choice:** Running an analytical dbt layer over isolated analytical tables instead of running feature engineering queries straight against live application tables.
* **Justification:** Aggregation queries on production ledgers introduce lock contention and severely degrade user experience. Isolating data access ensures transactional low-latency uptime while enabling heavy-duty relational processing inside an optimized data warehouse.

#### Feature Engineering Delegated to the Analytical Layer
* **Choice:** Performing the heavy feature engineering (trailing-window aggregations, as-of temporal joins, SCD2 resolution) inside dbt/SQL, keeping the Python ML layer focused on feature *preparation* (encoding, imputation, scaling) rather than feature *engineering*.
* **Justification:** Keeping a single source of truth for business logic in SQL/dbt — already covered by dedicated data-quality and leakage tests — avoids duplicating transformation logic across languages and reduces the risk of training/serving skew.

#### Temporal Integrity as a First-Class Constraint
* **Choice:** Treating the feature mart as panel data (one row per company per snapshot date) and designing every split, validation, and dimension-resolution strategy around a point-in-time cutoff instead of random shuffling.
* **Justification:** Credit risk data is inherently sequential. A random split would leak future information into training and produce metrics that look strong but collapse in production; SCD Type 2 dimensions and as-of joins ensure that every feature reflects only information that was actually available at the snapshot date.

#### F2-Optimal Decision Threshold per Model Family
* **Choice:** Instead of using the default 0.5 probability cutoff to convert predicted probabilities into a binary class, each model family's decision threshold is separately tuned by maximizing the F2-score on the aggregated cross-validation folds, then wired into the final training run.
* **Justification:** Under substantial class imbalance (~16% positive rate), 0.5 is an arbitrary cutoff with no particular statistical justification, and each model family produces probabilities on a different scale (e.g. the neural network's optimal threshold is far below 0.5). F2 weighs recall more heavily than precision, appropriate for insolvency prediction where missing an actual default is costlier than a false alarm; tuning it per model family, rather than sharing one global threshold, respects each family's own calibration instead of forcing a shared assumption onto all three.

#### Manual Hyperparameter Tuning via Shell Scripts
* **Choice:** Hyperparameter search is performed as a manual grid search driven by dedicated shell scripts, one per model family (plus a narrower `fine_tune_xgb.sh` second pass for XGBoost, once the full grid pointed to a promising region). Each script temporarily patches the relevant defaults via `sed`, runs the training pipeline for that combination, and appends the resulting cross-validation metrics to a CSV, keeping full per-run logs for inspection. The original files are always restored on exit (success, failure, or interruption).
* **Justification:** With only 2-3 hyperparameters explored per model family, a manual grid search keeps the process reproducible and inspectable without adding a new dependency or learning curve. A shared quirk had to be worked around: `setup_logging()` registers both a plain `StreamHandler` and Logfire's handler, and Logfire also echoes the same line to stdout independently, so each log line is captured twice, the parsing step takes only the first occurrence to avoid corrupting the results CSV.

#### SHAP-Driven Input Validation for Ad Hoc Predictions
* **Choice:** XGBoost is the only model family explained via SHAP (`ml/evaluation/explainability.py`, built on `TreeExplainer`), since only the best-performing model is ever deployed, and prior benchmarking already pointed to XGBoost, investing explainability effort in the baseline or MLP would not translate into production value. That same SHAP analysis (see the beeswarm plot in `ml/evaluation/plots.ipynb`) directly drives which fields are required versus optional in the `InsolvencyPredictionRequest` Pydantic schema used to validate ad hoc predictions for companies not yet in the database: the two dominant features (`unpaid_ratio_trailing_90d`, `total_outstanding_debt`) are mandatory, while the rest are optional and left as NaN when omitted.
* **Justification:** This ties the validation layer's strictness to actual, measured feature importance rather than an arbitrary judgment call about which fields "feel" necessary, and lets XGBoost's native missing-value handling (the model is trained with `handle_nan=False`) degrade gracefully on the fields it relies on least.

#### Informative Missing Values over Blind Imputation or Row Dropping
* **Choice:** Nullable feature groups (financial statement ratios, login activity, support ticket satisfaction) are handled with a per-group binary flag marking whether the value was observed, followed by a group-specific fallback constant, rather than dropping incomplete rows or imputing with a statistical average.
* **Justification:** A diagnostic query showed that a naive `dropna` would have discarded roughly 77% of the dataset, disproportionately removing companies with no recent support tickets — a systematic selection bias, not a random one. The missingness itself is informative (e.g. no resolved tickets recently, no published financial statement yet), so it is preserved as a signal instead of being discarded or disguised as an invented average.

#### Size-Weighted Cross-Validation Metric Aggregation
* **Choice:** Aggregating per-fold validation metrics with a mean weighted by each fold's validation set size, while confusion matrix counts (true/false positives/negatives) are summed rather than averaged.
* **Justification:** `TimeSeriesSplit`'s expanding window produces folds of unequal size, so an unweighted mean would give a small, less reliable early fold the same influence as a large, more reliable later one. Confusion matrix counts are absolute quantities tied to fold size; averaging them (weighted or not) yields a number with no clear interpretation, while summing preserves their meaning as a total across the full cross-validation run, since each row is scored exactly once.

#### Structural Typing over a Shared Model Base Class
* **Choice:** Defining an `Estimator` structural protocol (`fit`/`predict_proba`) that scikit-learn estimators and the custom PyTorch wrapper both satisfy implicitly, rather than forcing a shared base class or inheritance hierarchy across model families.
* **Justification:** scikit-learn and PyTorch models have fundamentally different internals (a declarative `.fit()` call versus a manual training loop); requiring a common base class would either constrain scikit-learn's own class hierarchy or force an artificial wrapper on it. Structural typing lets the training orchestrator depend only on the shape of the interface, keeping both model families and the orchestrator itself decoupled from one another.

#### Rigid Perimetric Validation over Dict Passing
* **Choice:** Utilizing **Pydantic v2** models to validate every incoming and outgoing payload across FastAPI endpoints and LangGraph state nodes.
* **Justification:** Untyped dictionary structures cause fragile software architectures. Applying the *fail-fast* principle at the application boundary guarantees that bad data types or unauthorized input structures are dropped instantly, keeping the pipeline secure and debugging predictable.

#### State Machine Graph Architectures over Sequential Prompt Chains
* **Choice:** Building agent logic with LangGraph instead of linear pipeline chains.
* **Justification:** Financial workflows are inherently cyclical, requiring validation loops, user clarification, and data retrieval backtracking. LangGraph treats the conversation as a formal state machine, maintaining deterministic state consistency across complex, asynchronous reasoning steps.

#### Router-First Agent Design
* **Choice:** The agent's graph classifies every incoming request into one of four explicit routes (company-specific prediction, ad hoc prediction, documentation retrieval, or a direct conversational reply) before any downstream node runs, rather than always invoking retrieval "just in case" and falling back when nothing relevant comes back.
* **Justification:** Retrieval is not free: every ChromaDB query adds latency and risks injecting irrelevant context into the prompt for requests that don't need it, such as a greeting or an out-of-scope question. An explicit router keeps each path in the graph doing only the work it actually needs, at the cost of depending on the router's own classification accuracy.

#### Single Source of Truth for the Agent's Routing Type
* **Choice:** The set of valid routes is defined once as a shared type, reused both by the schema constraining the router's structured LLM output and by the graph's own state, rather than duplicating the same set of literal values in both places.
* **Justification:** The two schemas serve different purposes — one shapes what the LLM is allowed to return, the other is the graph's source of truth — but they must always agree on the same set of valid routes. Defining that set once and importing it in both places removes the possibility of the two drifting out of sync if a new route is ever added.

#### Extraction, Not Query or Feature Generation
* **Choice:** The extraction step of the agent never lets the LLM produce SQL or a ready-to-use feature set directly. For a request about a specific company, the LLM only extracts the free-text identifier mentioned (legal name or VAT number); resolving it into a database record is done by a parameterized, code-written query, never one the model itself constructs. For an ad hoc, hypothetical company, the LLM extracts data into a permissive intermediate representation, every field optional and unconstrained, which is then validated for real against the same schema used to validate any prediction request; a validation failure is reported to the user rather than silently discarded or guessed around.
* **Justification:** Letting an LLM generate SQL from free text opens the door to injection risk that has nothing to do with a malicious user, a model producing a malformed or unsafe query is enough on its own. Separating what the user actually said from whether it is sufficient and valid also prevents an LLM extraction quirk, such as turning a vague qualitative claim into an invented number, from silently masquerading as real data feeding a financial prediction model.

#### Single Source of Truth for the Decision Threshold
* **Choice:** The prediction functions no longer accept a decision threshold from the caller; every prediction is scored against the threshold selected during training for the served model, read directly from the model's own tracked run when it is loaded.
* **Justification:** With the threshold as a separate, caller-supplied argument, nothing prevented a prediction from being scored against a value inconsistent with the one the model was actually tuned against. Tying it to the loaded model itself removes that possibility entirely, at the cost of failing fast if a run has no threshold recorded, a deliberate trade-off, since a run without one isn't ready to serve predictions in the first place.

#### Documentation Extraction Without Importing the Project
* **Choice:** The project's own documentation, every Python docstring and every dbt model and column description, is read directly from source text rather than by importing the project's modules to inspect them at runtime.
* **Justification:** Several modules open real side effects on import, such as a database connection, or pull in heavy machine learning dependencies not needed just to read a docstring. Reading source text directly avoids triggering any of that, so building the documentation index never risks side effects from the very modules it's documenting.

#### Idempotent Documentation Indexing
* **Choice:** The vector index backing the agent's documentation retrieval is cleared and rebuilt from scratch on every indexing run, rather than only adding newly found content to what's already there.
* **Justification:** Only adding new content would leave outdated material in the index forever once its source, a deleted section, a removed docstring, disappears from the project. Rebuilding from a clean slate each time means the index can always be safely refreshed after a documentation or code change, with no manual cleanup step.

#### Matching Prompt Format to the Kind of Material Being Injected
* **Choice:** The response node formats the material it works from differently depending on what it is, not uniformly as plain text: prediction results are passed as compact, structured records, while retrieved documentation is passed as plain prose.
* **Justification:** Prediction results are precise, multi-field numeric records the model must cite exactly without mixing up figures across companies or fields, exactly the case where a structured format outperforms prose. Retrieved documentation, by contrast, is prose the model should synthesize freely, where a structured format would only add noise. The format handed to a language model shapes how it reasons about the content, not just how that content looks, so the format is chosen to fit the material rather than applied uniformly.

#### Retrying a Known Flakiness in the Hosted Model Provider
* **Choice:** A call to the LLM that requires a structured, schema-constrained response is retried, with a higher retry allowance for the judge than for earlier steps, if the model responds with plain text instead of the required structured call. Once every attempt is exhausted, the response is treated as approved by default, logged as a notable event, rather than surfacing a raw error to the user.
* **Justification:** This is a documented flakiness of the specific hosted model family in use under a strict structured-output requirement, not a sign of a malformed request, so a short retry resolves it in practice. Defaulting to approved after exhausting every attempt reflects that the user has, in every case observed, already received a valid response from the model before judging even begins; an unverifiable verdict is a worse outcome for them than an unverified one.

#### Feature Order and Presence Checked Against the Model Itself
* **Choice:** Before scoring, the feature set is reordered to match the exact column order the model was trained on, read directly from the model rather than assumed from how each entry point happens to construct its own data.
* **Justification:** The underlying gradient-boosting library validates column order strictly, not just column names, and nothing else in the pipeline guaranteed the two would agree. Reading the order from the model itself removes that assumption entirely, rather than relying on it staying correct by convention.

#### Canonical Company Name Carried Through the Prediction
* **Choice:** A prediction for an existing company now carries its canonical legal name alongside its numeric result, and the response node also receives the company name or identifier as the user actually typed it.
* **Justification:** Without this, a response could only refer to a company by an internal identifier, with no way to confirm which company that identifier actually corresponded to, an inconsistency invisible in isolated testing but immediately apparent the first time a real conversation referred to a company by a name that didn't match verbatim.

#### A Deliberately Small API Surface
* **Choice:** The API exposes only two areas: a conversational endpoint routed through the agent, and two direct prediction endpoints that call the model without going through the LLM at all. It exposes no endpoint to query the transactional database or the analytical warehouse directly.
* **Justification:** The database and the datamart are implementation details of how the agent and the model arrive at an answer, not something an end user needs, or should be able to, query on their own. Keeping the public surface limited to a conversational entry point and a direct scoring path avoids turning a portfolio inference service into a general-purpose database query API, which was never the goal.

#### Two Prediction Endpoints, Not One Branching Internally
* **Choice:** Direct prediction is split into two distinct endpoints, one for a fully-specified profile not present in the database, and one for an existing company looked up by legal name or VAT number, rather than a single endpoint that inspects the request body to decide which case applies.
* **Justification:** This mirrors the same existing-company/ad-hoc-data distinction already used throughout the rest of the project, rather than reintroducing it as an implicit branch at the HTTP layer. The two also fail differently in ways that are easier to reason about as separate endpoints: a lookup by name or VAT number can fail to resolve to any company, a case that simply doesn't exist for a fully ad hoc, hypothetical request, which is always valid as long as it passes schema validation.

#### No Internal Database Identifier Ever Exposed to the Client
* **Choice:** Looking up an existing company happens by legal name or VAT number, resolved to an internal database identifier entirely server-side; that identifier is never included in the response sent back to the client.
* **Justification:** A surrogate database key is an implementation detail of the analytical warehouse, not a piece of information the caller supplied or has any independent way to obtain, since no endpoint exists to look one up in the first place. Returning it would expose an internal identifier with no actionable use on the client side, the same reasoning that kept the transactional and analytical schemas unexposed to begin with.

#### The Agent and the Model Are Built Once, Not Per Request
* **Choice:** Both the compiled agent graph and the loaded model bundle are built exactly once, at server startup, cached for the lifetime of the process, and reused across every request rather than rebuilt each time a request comes in.
* **Justification:** Both are expensive to construct, one wires together every node of a multi-step graph, the other loads a full model bundle from the tracking backend. Rebuilding either on every request would add unnecessary latency and load to every single call; building once and reusing the result keeps that cost entirely off the request path.

#### Conversation History Isolated per Session, Not Shared Globally
* **Choice:** Conversation history is keyed by a client-supplied session identifier, generated server-side and handed back to the client on the very first message if none was sent, rather than kept as a single shared history the way the interactive command-line version of the agent does.
* **Justification:** A single shared history is only valid for a tool that serves one user, one conversation, at a time. An HTTP server is stateless between requests and may serve several concurrent users at once, so history has to be isolated per conversation rather than shared globally. Storage is in-memory rather than persistent, an accepted limitation for a self-contained, no-cloud first version: state is lost on restart and isn't shared across multiple running instances.

#### One Shared Company-Resolution Query, Not Duplicated
* **Choice:** The parameterized query that resolves a free-text company identifier into a database record is defined once, in a shared location, and used both by the agent's own extraction step and by the API's direct lookup endpoint, rather than being duplicated between the two or having the API reach into the agent's own internals to reuse it.
* **Justification:** Both call sites need the exact same resolution logic, so duplicating it would risk the two drifting out of sync if the query ever changed. Reaching into the agent's own module to reuse it instead would couple the API to internals, prompts, and dependencies it has no reason to depend on, for the sake of a single shared query.

#### A Terminal-Free Interface That Only Talks to the Conversational Endpoint
* **Choice:** The UI calls the conversational endpoint and its history-reading counterpart exclusively; it never calls either direct prediction endpoint.
* **Justification:** Anyone with structured data ready for a direct prediction endpoint already knows how to make an HTTP call, documented above under Try the API. The interface's purpose is to make the conversational agent usable without a terminal, not to duplicate the direct prediction path behind a form.

#### A Lightweight, Client-Side Conversation Index, Never a Second Copy of the History Itself
* **Choice:** The interface keeps a small, session-scoped index of which past conversations exist and a short preview of each, so its sidebar can list them, but never stores the underlying messages themselves. Reopening a past conversation always re-reads its messages from the server rather than from a locally kept copy.
* **Justification:** The server is already the single source of truth for conversation history; keeping a second, client-side copy risks the two drifting apart, for instance after a server restart, which clears server-side history but would leave stale messages behind on the client. The index itself lives only for the current browser session and is lost when it closes, the same in-memory, no-cloud limitation already accepted for conversation history on the server side.

#### Trained Artifacts Are a Prerequisite of the Containerized Stack, Not Something It Produces
* **Choice:** The model tracking database and artifact store already produced by training on the host are mounted into the API's container at startup, rather than baked into its image at build time or regenerated by training inside the container on every run.
* **Justification:** Bringing up the containerized stack orchestrates an already-trained project, it doesn't train one from scratch, the same expectation already in place for its configuration. Mounting them rather than copying them in also means a freshly retrained model becomes visible on a restart without rebuilding anything, and keeps potentially large artifacts out of the image itself.

#### A Correction Loop That Actually Corrects, Not Just Retries Blindly
* **Choice:** When a response is rejected, the reason for that rejection is now fed back into the next attempt as an explicit correction hint, rather than the next attempt being generated exactly the same way as the one just rejected.
* **Justification:** Only the graph's own routing logic ever looked at a rejection verdict before, to decide whether to loop back at all, the step that actually generates the response never saw why its previous attempt had failed. In practice this meant a retry regenerated essentially the same answer and was rejected again for the same reason, exhausting the retry budget and falling back to a generic message even when the underlying issue was easily fixable. Closing that loop was already the intent behind recording a reason alongside every verdict, just not yet wired through to where it could act on it.

#### Known Limitation: Cross-Lingual Retrieval Is Unreliable
* **Choice:** This limitation is documented rather than worked around for this first version — no query translation step and no switch to a larger multilingual embedding model has been added yet.
* **Justification:** The embedding model behind documentation retrieval runs locally on CPU rather than through a hosted API, chosen for being free and adding no network round-trip per query, at the cost of weaker cross-lingual alignment than a larger or hosted model would offer. In practice, a question asked in Italian about English-language project documentation can retrieve much weaker matches than the same question asked in English, occasionally missing the relevant material entirely, an outcome no amount of instructing the response or verification step can fix once retrieval has already returned the wrong material to work with. A larger or hosted multilingual embedding model would very likely resolve this, at the cost of the constraints that led to the current choice in the first place.

#### A Single Command Rebuilds the Whole Project, Rather Than a Sequence to Remember
* **Choice:** Two end-to-end commands, one restoring data from version control and one regenerating it from scratch, chain every step of a full rebuild, migrations through bringing the containerized stack up, behind a single entry point each.
* **Justification:** Rebuilding the project from a clean checkout touches several independent tools, each with its own prerequisites and ordering constraints; collapsing that into one command per realistic starting point removes the chance of a step being skipped or run out of order, rather than leaving that sequencing to be remembered and repeated correctly by hand.

#### Database Readiness Is Actively Checked, Not Assumed
* **Choice:** Before running migrations against a freshly started database container, the rebuild process actively polls it for readiness rather than assuming it can already accept connections as soon as the container itself has started.
* **Justification:** A brand new database volume takes a few seconds to finish its own initialization, and simply waiting for a container to have started is not the same as the service inside it being ready, running migrations immediately after start-up intermittently raced against that initialization and failed. Polling for actual readiness removes that race outright, instead of masking it with a fixed delay that would be too short on a slower machine or wasted time on a faster one.

---

## 📊 Model Performance

**XGBoost is the production model**, chosen after benchmarking it against a Logistic Regression baseline and a PyTorch MLP on the same final holdout test set (never seen during training, cross-validation, or hyperparameter tuning), each evaluated at its own F2-optimal decision threshold:

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (Logistic Regression) | 0.8742 | 0.5026 | 0.3911 | 0.9578 | 0.5555 |
| MLP (PyTorch) | 0.8890 | 0.6030 | 0.3838 | 1.0000 | 0.5547 |
| **XGBoost** | **0.9198** | **0.6951** | **0.4351** | **0.9794** | **0.6026** |

XGBoost is the strongest model on every metric except recall, where the MLP reaches a perfect 1.0. This is not attributable to a well-chosen decision threshold, the same result holds regardless of the threshold used, suggesting the MLP's predicted probabilities are clustered in a narrow, mostly-high range rather than genuinely separating the two classes. Combined with its lower AUC-ROC and AUC-PR, this points to weaker discrimination overall rather than superior performance. XGBoost is the model served in production, consistent with it being the only model family benchmarked and explained in depth (see the SHAP section below and `ml/evaluation/plots.ipynb`).
