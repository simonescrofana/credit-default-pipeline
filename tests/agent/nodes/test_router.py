"""Test suite for the router node.

Cover the happy path of classifying a request with no prior history, the
construction of the message payload when `state.messages` holds a mix of
user/assistant turns, the edge case of an empty history, and that the
system prompt and current input are placed correctly around it.

"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes.router import router_node
from agent.state import AgentState
from schemas.agent.route_validation import RouterDecision


@patch("agent.nodes.router.get_responder_llm")
def test_router_node_happy_path_no_history(mock_get_responder_llm) -> None:
    """Verify router_node returns the route classified by the LLM."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = RouterDecision(route="case_a")
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(user_input="Rischia default Rossi SRL?")
    result = router_node(state)

    assert result == {"route": "case_a"}
    mock_get_responder_llm.assert_called_once_with(temperature=0.0)
    mock_llm.with_structured_output.assert_called_once_with(RouterDecision)


@patch("agent.nodes.router.get_responder_llm")
def test_router_node_builds_mixed_history_correctly(mock_get_responder_llm) -> None:
    """Verify user/assistant turns in state.messages map to the right message type."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = RouterDecision(route="direct")
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(
        user_input="E per l'azienda che ti ho chiesto prima?",
        messages=[
            {"role": "user", "content": "Ciao"},
            {"role": "assistant", "content": "Ciao, come posso aiutarti?"},
        ],
    )
    router_node(state)

    sent_messages = mock_structured_llm.invoke.call_args.args[0]

    assert isinstance(sent_messages[1], HumanMessage)
    assert sent_messages[1].content == "Ciao"
    assert isinstance(sent_messages[2], AIMessage)
    assert sent_messages[2].content == "Ciao, come posso aiutarti?"


@patch("agent.nodes.router.get_responder_llm")
def test_router_node_empty_history(mock_get_responder_llm) -> None:
    """Verify an empty state.messages produces no history messages."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = RouterDecision(route="rag")
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(user_input="Cosa significa leverage_ratio?")
    router_node(state)

    sent_messages = mock_structured_llm.invoke.call_args.args[0]

    assert len(sent_messages) == 2


@patch("agent.nodes.router.get_responder_llm")
def test_router_node_message_order(mock_get_responder_llm) -> None:
    """Verify the system prompt is first and the current user input is last."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = RouterDecision(route="case_b")
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(
        user_input="E se avesse leva 3.2?",
        messages=[{"role": "user", "content": "Ciao"}],
    )
    router_node(state)

    sent_messages = mock_structured_llm.invoke.call_args.args[0]

    assert isinstance(sent_messages[0], SystemMessage)
    assert isinstance(sent_messages[-1], HumanMessage)
    assert sent_messages[-1].content == "E se avesse leva 3.2?"
