"""Test suite for the judge node.

Covers one happy path per route the judge evaluates: case_a and case_b
(judged against prediction results/errors) and rag (judged against
retrieved context). Each verifies the judge builds the same material the
responder used, includes the response to evaluate, and returns the
judge's verdict as a dict in the state update.

"""

from unittest.mock import MagicMock, patch

from agent.nodes.judge import judge_node
from agent.state import AgentState
from schemas.agent.judge_validation import JudgeVerdict


@patch("agent.nodes.judge.get_judge_llm")
def test_judge_node_case_a_happy_path(mock_get_judge_llm) -> None:
    """Verify judge_node evaluates a case_a response against prediction results."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = JudgeVerdict(
        approved=True, reason="The response accurately reflects the prediction results."
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_judge_llm.return_value = mock_llm

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
        final_answer="Rossi SRL has a low default risk (probability 12%).",
    )
    result = judge_node(state)

    assert result == {
        "judge_verdict": {
            "approved": True,
            "reason": "The response accurately reflects the prediction results.",
        }
    }
    mock_llm.with_structured_output.assert_called_once_with(JudgeVerdict)

    sent_messages = mock_structured_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "Rischia default Rossi SRL?" in user_message
    assert "<prediction_results>" in user_message
    assert "Rossi SRL has a low default risk (probability 12%)." in user_message


@patch("agent.nodes.judge.get_judge_llm")
def test_judge_node_case_b_happy_path(mock_get_judge_llm) -> None:
    """Verify judge_node evaluates a case_b response against prediction results."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = JudgeVerdict(
        approved=False,
        reason="The response states a probability of 90%% but the prediction "
        "results show 0.8.",
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_judge_llm.return_value = mock_llm

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
        final_answer="This hypothetical company has a 90%% chance of defaulting.",
    )
    result = judge_node(state)

    assert result["judge_verdict"]["approved"] is False
    assert "0.8" in result["judge_verdict"]["reason"]

    sent_messages = mock_structured_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<prediction_results>" in user_message
    assert '"probability":0.8' in user_message
    assert "90% chance of defaulting" in user_message


@patch("agent.nodes.judge.get_judge_llm")
def test_judge_node_rag_happy_path(mock_get_judge_llm) -> None:
    """Verify judge_node evaluates a rag response against retrieved context."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = JudgeVerdict(
        approved=True, reason="The response is faithful to the retrieved context."
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_judge_llm.return_value = mock_llm

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
        final_answer="Groq was chosen for its free tier.",
    )
    result = judge_node(state)

    assert result == {
        "judge_verdict": {
            "approved": True,
            "reason": "The response is faithful to the retrieved context.",
        }
    }

    sent_messages = mock_structured_llm.invoke.call_args.args[0]
    user_message = sent_messages[1].content
    assert "<retrieved_context>" in user_message
    assert "Groq offers a genuinely free tier." in user_message
    assert "Groq was chosen for its free tier." in user_message
