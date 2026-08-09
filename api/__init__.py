"""HTTP API layer for the credit default pipeline.

Exposes the LangGraph agent and the underlying XGBoost model over HTTP, so
the project can be used as a standalone service rather than only through the
interactive CLI. Two endpoints are exposed:

- POST /chat: free-text conversation, routed through the agent (case_a,
    case_b, rag, or direct).
- POST /predict: a direct, deterministic prediction on a fully-specified
    company profile, bypassing the LLM.

The database, the OLAP star schema, and the ML training pipeline are not
exposed directly; they remain internal, used by the agent and the inference
layer behind the scenes.

"""
