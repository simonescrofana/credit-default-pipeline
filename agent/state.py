"""Shared state definition for the LangGraph agent.

Define the single Pydantic model threaded through every node of the graph.
Each node reads the fields it needs and returns a partial update, following
the router-first design: the router classifies the incoming request into
one of four routes (case_a, case_b, rag, direct) before any downstream node
runs, so that retrieval and prediction are only triggered when actually
relevant to the request.

"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Represent the full state shared between LangGraph nodes.

    Attributes:
        user_input (str): The raw, most recent message from the user.
        messages (list[dict]): The running conversation history, as
            role/content dicts.
        route (str | None): The path chosen by the router node, one of
            "case_a" (prediction on an existing company), "case_b"
            (prediction on user-supplied data), "rag" (documentation
            retrieval), or "direct" (no external grounding needed).
        company_identifiers (list[str] | None): Company identifiers (e.g.
            VAT number or legal name) extracted from the prompt for case_a.
            Supports multiple companies in a single request.
        raw_prediction_input (dict | None): Raw, unvalidated feature dict
            extracted from the prompt for case_b, to be validated by
            `schemas.insolvency_prediction.InsolvencyPredictionRequest`
            before being passed to the predictor.
        prediction_results (list[dict]): One entry per successfully scored
            company, shared by case_a and case_b. Each entry carries the
            predicted probability, class, and SHAP-based explanation
            produced by `ml.inference.predictor`.
        prediction_errors (list[str]): Operational failures that prevented
            a prediction from being produced (e.g. company not found,
            Pydantic validation failed), not to be confused with an
            unfavorable prediction, which is a valid, explained outcome.
        retrieved_context (list[dict] | None): Chunks retrieved from
            ChromaDB for the rag route, together with their metadata and
            similarity score.
        final_answer (str | None): The natural-language response returned
            to the user.
        judge_verdict (dict | None): Output of the LLM-as-a-Judge node,
            once introduced.

    """

    # Raw input and conversation history
    user_input: str
    messages: list[dict[str, str]] = Field(default_factory=list)

    # Router output
    route: Optional[Literal["case_a", "case_b", "rag", "direct"]] = None

    # Case A: prediction for one or more existing companies in the DB
    company_identifiers: Optional[list[str]] = None

    # Case B: prediction on user-provided data
    raw_prediction_input: Optional[dict[str, Any]] = None

    # Shared output for case A and case B (via predictor_node)
    prediction_results: list[dict[str, Any]] = Field(default_factory=list)
    prediction_errors: list[str] = Field(default_factory=list)

    # RAG
    retrieved_context: Optional[list[dict[str, Any]]] = None

    # Final output
    final_answer: Optional[str] = None

    # Judge (to be evaluated later)
    judge_verdict: Optional[dict[str, Any]] = None
