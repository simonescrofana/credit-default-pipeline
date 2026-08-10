"""HTTP API layer for the credit default pipeline.

Exposes the LangGraph agent and the underlying XGBoost model over HTTP, so
the project can be used as a standalone service rather than only through the
interactive CLI. Two endpoint areas are exposed:

- POST /chat: free-text conversation, routed through the agent (case_a,
    case_b, rag, or direct).
- POST /predict/ad-hoc and POST /predict/company: direct, deterministic
    predictions bypassing the LLM, on ad hoc data or on a company already
    in the database, respectively.

The database, the OLAP star schema, and the ML training pipeline are not
exposed directly; they remain internal, used by the agent and the inference
layer behind the scenes.

"""
