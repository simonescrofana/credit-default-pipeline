"""Streamlit UI layer for the credit default pipeline.

Exposes the LangGraph agent as a browsable, non-technical interface, via
`POST /chat` on the FastAPI backend. Only the conversational endpoint is
surfaced here, direct prediction (`/predict/ad-hoc`, `/predict/company`)
targets callers who already have structured data are expected to use the
API directly, documented in the project's README.

"""
