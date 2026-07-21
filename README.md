# Enterprise Credit Default Pipeline & GenAI Analyst

An end-to-end, production-grade MLOps and Data Engineering pipeline designed to predict credit default on highly imbalanced financial and transactional data. Features a decoupled storage architecture, automated experiment tracking, perimetric data validation, and a Generative AI assistant agent integrated with Explainable AI (xAI) metrics.

---

## 🚧 Project Status & Roadmap

This project is actively developed to simulate an enterprise-grade AI infrastructure deployment. 

* **[x] Phase 1:** Infrastructure Setup (Docker, PostgreSQL, GitHub Actions).
* **[x] Phase 2:** OLTP Core Banking Database Setup & Schema Design (SQLAlchemy ORM + Alembic Migrations).
* **[x] Phase 3:** MLOps Data Versioning (DVC Data Tracking) & OLAP Warehouse Transformation (dbt Core, Star Schema).
* **[ ] Phase 4:** Machine Learning Benchmark Suite (Sklearn, XGBoost, PyTorch) & Experiment Tracking (MLflow).
* **[ ] Phase 5:** Explainable AI (SHAP) Integration & Agentic GenAI Layer (LangGraph + ChromaDB).
* **[ ] Phase 6:** Production Exposure (FastAPI App) & Live Monitoring/Observability UI (Streamlit + Pydantic + Logfire).

---

## 🏗️ System Architecture

The system is engineered using a strictly decoupled, multi-layered architecture to process data securely from ingestion to intelligent, explainable inference:

1. **Transactional Layer (OLTP):** Containerized PostgreSQL instance simulating a production core-banking system managed via SQLAlchemy ORM and tracked through Alembic migrations.
2. **Analytical Layer (OLAP):** Dimensional Data Warehouse modeled into a Star Schema driven by dbt Core over historical immutable ledgers.
3. **MLOps & Lifecyle Layer:** Data version control implemented with DVC. Dual-engine training pipeline (Gradient Boosted Trees & PyTorch Neural Architectures) integrated with MLflow for artifact logging, hyperparameter tracking, and model registry.
4. **Explainable AI (xAI) Module:** Interpretability extraction utilizing SHAP to ensure credit scoring compliance and transparency.
5. **Generative AI Layer:** An agent system built via LangGraph acting as an autonomous financial analyst, querying a ChromaDB vector store, running local inference, and validated by an LLM-as-a-Judge node.
6. **Application & Serving Layer:** A FastAPI backend exposing validated REST endpoints for predictions and chat interactions, paired with a Streamlit interface acting as a live, interactive demo of the full pipeline.

---
 
## 🧠 Architectural Decisions & Rationale
 
#### Complete Isolation of OLTP and OLAP Store
* **Choice:** Running an analytical dbt layer over isolated analytical tables instead of running feature engineering queries straight against live application tables.
* **Justification:** Aggregation queries on production ledgers introduce lock contention and severely degrade user experience. Isolating data access ensures transactional low-latency uptime while enabling heavy-duty relational processing inside an optimized data warehouse.

#### Feature Engineering Delegated to the Analytical Layer
* **Choice:** Performing the heavy feature engineering (trailing-window aggregations, as-of temporal joins, SCD2 resolution) inside dbt/SQL, keeping the Python ML layer focused on feature *preparation* (encoding, imputation, scaling) rather than feature *engineering*.
* **Justification:** Keeping a single source of truth for business logic in SQL/dbt — already covered by dedicated data-quality and leakage tests — avoids duplicating transformation logic across languages and reduces the risk of training/serving skew.

#### Temporal Integrity as a First-Class Constraint
* **Choice:** Treating the feature mart as panel data (one row per company per snapshot date) and designing every split, validation, and dimension-resolution strategy around a point-in-time cutoff instead of random shuffling.
* **Justification:** Credit risk data is inherently sequential. A random split would leak future information into training and produce metrics that look strong but collapse in production; SCD Type 2 dimensions and as-of joins ensure that every feature reflects only information that was actually available at the snapshot date.

#### Modern Python Tooling (`uv` + Ruff)
* **Choice:** Moving away from standard `pip`/`venv` and selecting `uv` as the exclusive dependency manager alongside Ruff for quality gating.
* **Justification:** `uv` provides blazing-fast environment synchronization and strict, deterministic lockfile management, eliminating the "it works on my machine" anti-pattern in production containers. Ruff guarantees lightning-fast code linting and formatting compliance natively during pre-commit and CI stages.

---

## 🛠️ Tech Stack

* **Infrastructure & DevOps:** Docker, Docker Compose, GitHub Actions (CI/CD)
* **Environment & Package Management:** Python, uv
* **Data Engineering & Storage:** PostgreSQL, SQLAlchemy, Alembic, dbt Core
* **Data Versioning:** DVC
* **Machine Learning Engines:** PyTorch, Scikit-Learn, XGBoost
* **Explainable AI (xAI):** SHAP
* **MLOps & Model Tracking:** MLflow
* **QA & Enterprise Validation:** Pytest, Pydantic Validation
* **Observability & Logging:** Pydantic Logfire, Ruff
* **Generative AI Infrastructure:** LangGraph, ChromaDB
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
│   │   │   ├── marts/
│   │   │   │   ├── dim_companies/
│   │   │   │   │   ├── dim_companies_no_overlapping_windows.sql
│   │   │   │   │   └── dim_companies_single_current_version.sql
│   │   │   │   ├── dim_date/
│   │   │   │   │   └── dim_date_no_gaps.sql
│   │   │   │   └── fct_company_credit_profile/
│   │   │   │       ├── fct_company_key_temporal_correctness.sql
│   │   │   │       ├── fct_no_dropped_spine_rows.sql
│   │   │   │       └── fct_no_unexpected_nulls.sql
│   │   │   └── intermediate/
│   │   │       ├── int_billing_trailing_90d/
│   │   │       │   ├── int_billing_trailing_90d_debt_ratio_match.sql
│   │   │       │   ├── int_billing_trailing_90d_dpd_consistency.sql
│   │   │       │   └── int_billing_trailing_90d_no_future_leakage.sql
│   │   │       ├── int_companies_scd_resolved/
│   │   │       │   ├── int_companies_scd_resolved_chronology.sql
│   │   │       │   ├── int_companies_scd_resolved_expired_versions.sql
│   │   │       │   └── int_companies_scd_resolved_leakage.sql
│   │   │       ├── int_company_date_spine/
│   │   │       │   ├── int_company_date_spine_no_dates_before_foundation.sql
│   │   │       │   ├── int_company_date_spine_no_future_dates.sql
│   │   │       │   ├── int_company_date_spine_np_gaps.sql
│   │   │       │   └── int_company_date_spine_respects_valid_to.sql
│   │   │       ├── int_contracts_asof/
│   │   │       │   ├── int_contracts_asof_count_flag_consistency.sql
│   │   │       │   └── int_contracts_asof_no_future_leakage.sql
│   │   │       ├── int_financial_asof/
│   │   │       │   ├── int_financial_asof_publication_delay_leakage.sql
│   │   │       │   └── int_financial_asof_rank_recency.sql
│   │   │       ├── int_insolvency_label/
│   │   │       │   ├── int_insolvency_label_false_negative.sql
│   │   │       │   └── int_insolvency_label_false_positive.sql
│   │   │       ├── int_logins_trailing/
│   │   │       │   ├── int_logins_trailing_null_consistency.sql
│   │   │       │   ├── int_logins_trailing_recency_boundary.sql
│   │   │       │   └── int_logins_trailing_velocity_coherence.sql
│   │   │       └── int_tickets_trailing_90d/
│   │   │           └── int_tickets_trailing_90d_no_future_leakage.sql
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
│   └── schema/
│       ├── credit-default-DFM.sql
│       ├── credit-default-star-schema.sql
│       └── database_structure.sql
├── ml/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── split.py
│   └── __init__.py
├── schemas/
│   ├── __init__.py
│   ├── base.py
│   ├── models_validation.py
│   └── types.py
├── simulation/
│   ├── __init__.py
│   ├── profiles.py
│   └── seed.py
├── tests/
│   ├── analytics/
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── test_extract.py
│   │   │   └── test_restore.py
│   │   └── __init__.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── test_connection.py
│   │   └── test_models.py
│   ├── ml/
│   │   ├── dataset/
│   │   │   ├── __init__.py
│   │   │   ├── test_loader.py
│   │   │   └── test_split.py
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── test_models_validation.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── test_seed.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── test_timezone_utils.py
│   ├── __init__.py
│   └── conftest.py
├── ui/
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   └── timezone_utils.py
├── .dvcignore
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── alembic.ini
├── config.py
├── docker-compose.yml
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```
