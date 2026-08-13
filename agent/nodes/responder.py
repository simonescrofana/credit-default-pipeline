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
    if (
        not state.company_identifiers
        and not state.prediction_results
        and not state.prediction_errors
    ):
        return ""

    parts = []
    if state.company_identifiers:
        # The free-text identifiers as extracted from the prompt (e.g. a
        # legal name or VAT number), so the model can match them against
        # prediction_results' company_id/company_name — prediction_results
        # alone never carries the user's original wording.
        parts.append(
            "<company_identifiers>\n"
            + json.dumps(state.company_identifiers, separators=(",", ":"))
            + "\n</company_identifiers>"
        )
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

    # On a retry, the judge already rejected a prior attempt: feed its
    # reason back in as a correction hint, so this attempt can actually
    # fix what was wrong instead of blindly regenerating the same answer
    # and getting rejected again for the same reason.
    if state.judge_verdict and not state.judge_verdict.get("approved"):
        user_content = (
            f"{user_content}\n\n"
            "<correction_needed>\n"
            "Your previous answer to this request was rejected. Reason:\n"
            f"{state.judge_verdict.get('reason', '')}\n"
            "Write a new answer that fixes this issue.\n"
            "</correction_needed>"
        )

    messages = [
        SystemMessage(content=RESPONDER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    logger.info("Generating response for route '%s'...", state.route)
    response = llm.invoke(messages)

    return {"final_answer": response.content}
