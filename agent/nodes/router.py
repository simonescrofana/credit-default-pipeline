"""Router node: classifies the incoming request into one of four routes.

The router is the entry point of the graph. It reads the user's request
and decides which path it should follow, case_a, case_b, rag, or direct,
by delegating the classification to the responder LLM constrained to
`RouterDecision`. It does not extract data, resolve company identifiers,
or answer the request itself; those responsibilities belong to downstream
nodes.

"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.models.llm_responder import get_responder_llm
from agent.prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from agent.state import AgentState
from schemas.agent.route_validation import RouterDecision

logger = logging.getLogger(__name__)


def router_node(state: AgentState) -> dict:
    """Classify the user's request and route it to the appropriate path.

    Args:
        state (AgentState): The current graph state. Reads `user_input`
            together with `messages` as conversational context, so the
            router can account for references to earlier turns.

    Returns:
        dict: A partial state update setting `route` to the classification
            produced by the LLM.

    """
    llm = get_responder_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RouterDecision)

    history = [
        HumanMessage(content=turn["content"])
        if turn["role"] == "user"
        else AIMessage(content=turn["content"])
        for turn in state.messages
    ]

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=state.user_input),
    ]

    logger.info("Routing user request...")
    decision: RouterDecision = structured_llm.invoke(messages)
    logger.info("Request routed to '%s'.", decision.route)

    return {"route": decision.route}
