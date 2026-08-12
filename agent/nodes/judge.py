"""Judge node: evaluates a generated response against its source material.

Reuses `responder._build_material` to format the same material the
responder based its answer on (prediction results/errors for case_a and
case_b, retrieved context for rag), so the judge evaluates the response
against exactly what the responder actually saw, not a re-derived or
inconsistent version of it. The direct route has no external material to
check a response against, so it is never routed to this node at all —
that dispatch is `graph.py`'s responsibility, not this module's.

"""

import logging

from groq import BadRequestError
from langchain_core.messages import HumanMessage, SystemMessage

from agent.models.llm_judge import get_judge_llm
from agent.nodes.responder import build_material
from agent.prompts.judge_prompt import JUDGE_SYSTEM_PROMPT
from agent.state import AgentState
from agent.utils.llm_utils import MAX_RETRIES, invoke_with_retry
from schemas.agent.judge_validation import JudgeVerdict

logger = logging.getLogger(__name__)


def judge_node(state: AgentState) -> dict:
    """Evaluate the responder's answer against its source material.

    Args:
        state (AgentState): The current graph state. Reads `route`,
            `user_input`, `final_answer`, and, depending on the route,
            `prediction_results`/`prediction_errors` or
            `retrieved_context`.

    Returns:
        dict: A partial state update setting `judge_verdict` to the
            evaluation produced by the judge LLM.

    """
    llm = get_judge_llm()
    structured_llm = llm.with_structured_output(JudgeVerdict)

    material = build_material(state)
    user_content = (
        f"User's original request:\n{state.user_input}\n\n"
        f"{material}\n\n"
        f"Response to evaluate:\n{state.final_answer}"
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    logger.info("Judging response for route '%s'...", state.route)
    try:
        verdict: JudgeVerdict = invoke_with_retry(
            structured_llm, messages, max_retries=MAX_RETRIES
        )
        logger.info(
            "Judge verdict: approved=%s. Reason: %s", verdict.approved, verdict.reason
        )
    except BadRequestError:
        logger.warning(
            "Judge LLM failed to produce a verdict after %d attempts; "
            "defaulting to approved.",
            MAX_RETRIES,
        )
        verdict = JudgeVerdict(
            approved=True,
            reason="Judge LLM unavailable after retries; response passed "
            "through unverified.",
        )

    return {"judge_verdict": verdict.model_dump()}
