"""Test suite for the responder node.

Covers one happy path per route: case_a/case_b inject prediction results as
tagged JSON, rag injects retrieved context as tagged plain text, and direct
injects no material at all, sending only the user's input.

"""

from unittest.mock import MagicMock, patch

from agent.nodes.responder import responder_node
from agent.state import AgentState


@patch("agent.nodes.responder.get_responder_llm")
def test_responder_node_case_a_happy_path(mock_get_responder_llm) -> None:
    """Verify case_a injects prediction_results as tagged JSON."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="Rossi SRL has a low default risk."
    )
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(
        user_input="Rischia default Rossi SRL?",
        route="case_a",
        prediction_results=[
            {
                "company_id": 1,
                "probability": 0.12,
                "predicted_class": 0,
                "explanation": {},
            }
        ],
    )
    result = responder_node(state)

    assert result == {"final_answer": "Rossi SRL has a low default risk."}
    sent_messages = mock_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<prediction_results>" in user_message
    assert '"probability":0.12' in user_message
    assert "Rischia default Rossi SRL?" in user_message


@patch("agent.nodes.responder.get_responder_llm")
def test_responder_node_case_b_happy_path(mock_get_responder_llm) -> None:
    """Verify case_b injects prediction_results the same way as case_a."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="This hypothetical company is risky."
    )
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(
        user_input="What if a company had high debt?",
        route="case_b",
        prediction_results=[
            {
                "company_id": None,
                "probability": 0.8,
                "predicted_class": 1,
                "explanation": {},
            }
        ],
    )
    result = responder_node(state)

    assert result == {"final_answer": "This hypothetical company is risky."}
    sent_messages = mock_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<prediction_results>" in user_message
    assert '"predicted_class":1' in user_message


@patch("agent.nodes.responder.get_responder_llm")
def test_responder_node_rag_happy_path(mock_get_responder_llm) -> None:
    """Verify rag injects retrieved_context as tagged plain text."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="Groq was chosen for its free tier."
    )
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(
        user_input="Why was Groq chosen?",
        route="rag",
        retrieved_context=[
            {
                "text": "Groq offers a genuinely free tier.",
                "distance": 0.1,
                "metadata": {},
            }
        ],
    )
    result = responder_node(state)

    assert result == {"final_answer": "Groq was chosen for its free tier."}
    sent_messages = mock_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<retrieved_context>" in user_message
    assert "Groq offers a genuinely free tier." in user_message


@patch("agent.nodes.responder.get_responder_llm")
def test_responder_node_direct_happy_path(mock_get_responder_llm) -> None:
    """Verify direct injects no material, sending only the user's input."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Hi there! How can I help?")
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(user_input="Ciao!", route="direct")
    result = responder_node(state)

    assert result == {"final_answer": "Hi there! How can I help?"}
    sent_messages = mock_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert user_message == "Ciao!"


@patch("agent.nodes.responder.get_responder_llm")
def test_responder_node_rag_with_no_retrieved_context(mock_get_responder_llm) -> None:
    """Verify rag with an empty retrieved_context injects no material."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="I don't have information on that."
    )
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(user_input="What is X?", route="rag", retrieved_context=[])
    responder_node(state)

    sent_messages = mock_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<retrieved_context>" not in user_message
    assert user_message == "What is X?"


@patch("agent.nodes.responder.get_responder_llm")
def test_responder_node_case_a_with_no_results_or_errors(
    mock_get_responder_llm,
) -> None:
    """Verify case_a with empty results and errors injects no material."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="I have nothing to report.")
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(user_input="Rischia default Rossi SRL?", route="case_a")
    responder_node(state)

    sent_messages = mock_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<prediction_results>" not in user_message
    assert "<prediction_errors>" not in user_message
    assert user_message == "Rischia default Rossi SRL?"


@patch("agent.nodes.responder.get_responder_llm")
def test_responder_node_case_a_with_only_errors(mock_get_responder_llm) -> None:
    """Verify case_a with only prediction_errors (no results) injects just that tag."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Rossi SRL was not found.")
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(
        user_input="Rischia default Rossi SRL?",
        route="case_a",
        prediction_errors=["Company 'Rossi SRL' not found in the database."],
    )
    responder_node(state)

    sent_messages = mock_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<prediction_results>" not in user_message
    assert "<prediction_errors>" in user_message
    assert "not found in the database" in user_message
