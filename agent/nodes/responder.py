"""Responder node: turns the graph's accumulated state into a final reply.

A single parametric system prompt (see `agent.prompts.responder_prompt`)
covers every route, the responder's role doesn't change across case_a,
case_b, rag, and direct, only the material it has to work with does. That
material is formatted differently depending on what it is:

- Prediction results and errors (case_a, case_b): formatted as minimal
  JSON, wrapped in an XML-style tag. This is structured, multi-field
  numeric data the model must cite precisely without mixing up figures
  across companies or fields, exactly the case where a keyed format
  outperforms prose.
- Retrieved context (rag): formatted as plain text, wrapped in its own
  tag. This is prose the model should synthesize freely, where a keyed
  format would add syntactic noise without benefit.
- direct: no material is injected at all.

"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.models.llm_responder import get_responder_llm
from agent.prompts.responder_prompt import RESPONDER_SYSTEM_PROMPT
from agent.state import AgentState

logger = logging.getLogger(__name__)


def format_prediction_material(state: AgentState) -> str:
    """Format prediction_results and prediction_errors as tagged minimal JSON.

    Args:
        state (AgentState): The current graph state.

    Returns:
        str: The formatted material, or an empty string if there is
            nothing to report.

    """
    if not state.prediction_results and not state.prediction_errors:
        return ""

    parts = []
    if state.prediction_results:
        parts.append(
            "<prediction_results>\n"
            + json.dumps(state.prediction_results, separators=(",", ":"))
            + "\n</prediction_results>"
        )
    if state.prediction_errors:
        parts.append(
            "<prediction_errors>\n"
            + json.dumps(state.prediction_errors, separators=(",", ":"))
            + "\n</prediction_errors>"
        )
    return "\n".join(parts)


def format_rag_material(state: AgentState) -> str:
    """Format retrieved_context as tagged plain text.

    Args:
        state (AgentState): The current graph state.

    Returns:
        str: The formatted material, or an empty string if nothing was
            retrieved.

    """
    if not state.retrieved_context:
        return ""

    chunks = "\n\n".join(chunk["text"] for chunk in state.retrieved_context)
    return f"<retrieved_context>\n{chunks}\n</retrieved_context>"


def build_material(state: AgentState) -> str:
    """Dispatch to the right formatter for state.route.

    Args:
        state (AgentState): The current graph state.

    Returns:
        str: The formatted material to inject into the user message, or
            an empty string for the direct route (no external grounding).

    """
    if state.route in ("case_a", "case_b"):
        return format_prediction_material(state)
    if state.route == "rag":
        return format_rag_material(state)
    return ""


def responder_node(state: AgentState) -> dict:
    """Generate the final natural-language reply for the user.

    Args:
        state (AgentState): The current graph state. Reads `route`,
            `user_input`, and, depending on the route,
            `prediction_results`/`prediction_errors` or
            `retrieved_context`.

    Returns:
        dict: A partial state update setting `final_answer` to the
            generated reply.

    """
    llm = get_responder_llm()

    material = build_material(state)
    user_content = f"{material}\n\n{state.user_input}" if material else state.user_input

    messages = [
        SystemMessage(content=RESPONDER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    logger.info("Generating response for route '%s'...", state.route)
    response = llm.invoke(messages)

    return {"final_answer": response.content}
