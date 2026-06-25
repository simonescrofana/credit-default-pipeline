# Enterprise Credit Default Pipeline & GenAI Analyst

An end-to-end, production-grade MLOps and Data Engineering pipeline designed to predict credit default on highly imbalanced financial and transactional data. Features a decoupled storage architecture, automated experiment tracking, perimetric data validation, and a Generative AI assistant agent integrated with Explainable AI (xAI) metrics.

---

## 🚧 Project Status & Roadmap

This project is actively developed to simulate an enterprise-grade AI infrastructure deployment. 

* **[x] Phase 1:** Infrastructure Setup (Docker, PostgreSQL, GitHub Actions).
* **[ ] Phase 2:** OLTP Core Banking Database Setup & Schema Design (SQLAlchemy ORM + Alembic Migrations).
* **[ ] Phase 3:** MLOps Data Versioning (DVC Data Tracking) & OLAP Warehouse Transformation (dbt Core, Star Schema).
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
├── .github/workflows/
│   └── main.yml
├── agent/
├── database/
├── dbt_project/
├── pipeline/
├── schemas/
├── tests/
├── ui/
├── .env
├── .env.example
├── .gitignore
├── config.py
├── docker-compose.yml
├── LICENSE
├── pyproject.toml
└── README.md
```
