"""Test suite for the agent's graph assembly.

Covers every routing function's branches (`route_after_router`,
`route_after_responder`, `route_after_judge`, including the defensive
case of a missing judge_verdict), the two small state-mutating nodes
(`increment_retry_count`, `fallback_node`), and that `build_agent`
compiles successfully with every expected node present.

"""

from langgraph.graph import END

from agent.graph import (
    FALLBACK_MESSAGE,
    MAX_RETRIES,
    build_agent,
    fallback_node,
    increment_retry_count,
    route_after_judge,
    route_after_responder,
    route_after_router,
)
from agent.state import AgentState


def test_route_after_router_case_a_goes_to_extractor() -> None:
    """Verify case_a routes to the extractor."""
    state = AgentState(user_input="...", route="case_a")
    assert route_after_router(state) == "extractor"


def test_route_after_router_case_b_goes_to_extractor() -> None:
    """Verify case_b routes to the extractor."""
    state = AgentState(user_input="...", route="case_b")
    assert route_after_router(state) == "extractor"


def test_route_after_router_rag_goes_to_retriever() -> None:
    """Verify rag routes to the retriever node."""
    state = AgentState(user_input="...", route="rag")
    assert route_after_router(state) == "retriever_node"


def test_route_after_router_direct_goes_to_responder() -> None:
    """Verify direct routes straight to the responder."""
    state = AgentState(user_input="...", route="direct")
    assert route_after_router(state) == "responder"


def test_route_after_responder_direct_skips_judge() -> None:
    """Verify a direct response goes straight to END, never judged."""
    state = AgentState(user_input="...", route="direct")
    assert route_after_responder(state) == END


def test_route_after_responder_other_routes_go_to_judge() -> None:
    """Verify case_a/case_b/rag responses are always judged."""
    for route in ("case_a", "case_b", "rag"):
        state = AgentState(user_input="...", route=route)
        assert route_after_responder(state) == "judge"


def test_route_after_judge_approved_goes_to_end() -> None:
    """Verify an approved verdict ends the graph."""
    state = AgentState(
        user_input="...", judge_verdict={"approved": True, "reason": "ok"}
    )
    assert route_after_judge(state) == END


def test_route_after_judge_rejected_under_cap_retries() -> None:
    """Verify a rejected verdict under the retry cap routes back to responder."""
    state = AgentState(
        user_input="...",
        judge_verdict={"approved": False, "reason": "not faithful"},
        retry_count=MAX_RETRIES - 1,
    )
    assert route_after_judge(state) == "responder"


def test_route_after_judge_rejected_at_cap_falls_back() -> None:
    """Verify a rejected verdict at the retry cap routes to the fallback."""
    state = AgentState(
        user_input="...",
        judge_verdict={"approved": False, "reason": "not faithful"},
        retry_count=MAX_RETRIES,
    )
    assert route_after_judge(state) == "fallback"


def test_route_after_judge_missing_verdict_retries_defensively() -> None:
    """Verify a missing judge_verdict is treated as not approved, not a crash."""
    state = AgentState(user_input="...", judge_verdict=None, retry_count=0)
    assert route_after_judge(state) == "responder"


def test_increment_retry_count_happy_path() -> None:
    """Verify increment_retry_count increases retry_count by one."""
    state = AgentState(user_input="...", retry_count=2)
    result = increment_retry_count(state)
    assert result == {"retry_count": 3}


def test_fallback_node_happy_path() -> None:
    """Verify fallback_node returns the fixed fallback message."""
    state = AgentState(user_input="...")
    result = fallback_node(state)
    assert result == {"final_answer": FALLBACK_MESSAGE}


def test_build_agent_compiles_with_every_node() -> None:
    """Verify build_agent compiles and includes every expected node."""
    compiled = build_agent()

    node_names = set(compiled.get_graph().nodes.keys())
    expected_nodes = {
        "router",
        "extractor",
        "predictor_node",
        "retriever_node",
        "responder",
        "judge",
        "increment_retry_count",
        "fallback",
    }
    assert expected_nodes.issubset(node_names)
