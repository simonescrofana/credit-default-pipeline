"""Agentic layer for the credit default pipeline.

Implements a LangGraph-based conversational agent acting as an autonomous
financial analyst. The agent routes user requests across four paths:

- case_a: insolvency prediction for one or more existing companies
    (via ml.inference.predictor)
- case_b: insolvency prediction on user-provided data, validated through Pydantic
- rag: retrieval-augmented answers grounded in project documentation (ChromaDB)
- direct: unrouted conversational responses requiring no external grounding

Predictions are always accompanied by SHAP-based explanations produced by
the existing ML inference layer.

"""
