"""Assembles the agent's LangGraph.

Wires together every node written under `agent/nodes/` into the graph's
topology:

    router -> (case_a | case_b) -> extractor -> predictor_node -> responder
    router -> rag -> retriever_node -> responder
    router -> direct -> responder

    responder -> (case_a | case_b | rag) -> judge
    responder -> direct -> END (never judged: no external material to
        check a response against)

    judge -> approved -> END
    judge -> rejected, under the retry cap -> responder (regenerate)
    judge -> rejected, retry cap reached -> a fixed fallback message -> END

The retry cap exists so a response the judge keeps rejecting can never
loop indefinitely; past it, the graph stops trying and returns a fixed,
non-LLM-generated message instead.

"""

import logging

from langgraph.graph import END, START, StateGraph

from agent.nodes.extractor import extractor_node
from agent.nodes.judge import judge_node
from agent.nodes.predictor_node import predictor_node
from agent.nodes.responder import responder_node
from agent.nodes.retriever_node import retriever_node
from agent.nodes.router import router_node
from agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
FALLBACK_MESSAGE = "Sorry, I can not help you with that."


def route_after_router(state: AgentState) -> str:
    """Pick the next node based on the router's classification.

    Args:
        state (AgentState): The current graph state. Reads `route`.

    Returns:
        str: The name of the next node to run.

    """
    if state.route in ("case_a", "case_b"):
        return "extractor"
    if state.route == "rag":
        return "retriever_node"
    return "responder"


def route_after_responder(state: AgentState) -> str:
    """Decide whether a generated response needs judging.

    Args:
        state (AgentState): The current graph state. Reads `route`.

    Returns:
        str: The name of the next node to run.

    """
    if state.route == "direct":
        return END
    return "judge"


def route_after_judge(state: AgentState) -> str:
    """Decide whether to accept, retry, or give up on a judged response.

    Args:
        state (AgentState): The current graph state. Reads `judge_verdict`
            and `retry_count`.

    Returns:
        str: The name of the next node to run.

    """
    if state.judge_verdict and state.judge_verdict.get("approved"):
        return END
    if state.retry_count < MAX_RETRIES:
        return "responder"
    return "fallback"


def increment_retry_count(state: AgentState) -> dict:
    """Increment retry_count before looping back to the responder.

    Args:
        state (AgentState): The current graph state. Reads `retry_count`.

    Returns:
        dict: A partial state update incrementing `retry_count` by one.

    """
    return {"retry_count": state.retry_count + 1}


def fallback_node(state: AgentState) -> dict:
    """Return the fixed fallback message once the retry cap is reached.

    Args:
        state (AgentState): The current graph state. Unused, present only
            to match the node function signature every other node uses.

    Returns:
        dict: A partial state update setting `final_answer` to
            `FALLBACK_MESSAGE`.

    """
    logger.warning("Retry cap reached; returning the fallback message.")
    return {"final_answer": FALLBACK_MESSAGE}


def build_agent():
    """Build and compile the agent's LangGraph.

    Returns:
        CompiledStateGraph: The compiled graph, ready to be invoked with
            an initial `AgentState`.

    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("predictor_node", predictor_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("responder", responder_node)
    graph.add_node("judge", judge_node)
    graph.add_node("increment_retry_count", increment_retry_count)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "extractor": "extractor",
            "retriever_node": "retriever_node",
            "responder": "responder",
        },
    )

    graph.add_edge("extractor", "predictor_node")
    graph.add_edge("predictor_node", "responder")
    graph.add_edge("retriever_node", "responder")

    graph.add_conditional_edges(
        "responder",
        route_after_responder,
        {"judge": "judge", END: END},
    )

    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {"responder": "increment_retry_count", "fallback": "fallback", END: END},
    )
    graph.add_edge("increment_retry_count", "responder")
    graph.add_edge("fallback", END)

    return graph.compile()
