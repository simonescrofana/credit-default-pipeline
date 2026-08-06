"""Test suite for the retriever node.

Covers the happy path: retriever_node passes the user's input to
retrieve_context and writes its result to retrieved_context in the state
update, unchanged.

"""

from unittest.mock import patch

from agent.nodes.retriever_node import retriever_node
from agent.state import AgentState


@patch("agent.nodes.retriever_node.retrieve_context")
def test_retriever_node_happy_path(mock_retrieve_context) -> None:
    """Verify retriever_node queries with user_input and returns retrieved_context."""
    fake_context = [
        {"text": "chunk one", "distance": 0.1, "metadata": {"name": "a"}},
        {"text": "chunk two", "distance": 0.3, "metadata": {"name": "b"}},
    ]
    mock_retrieve_context.return_value = fake_context

    state = AgentState(user_input="what LLM does the agent use?")
    result = retriever_node(state)

    mock_retrieve_context.assert_called_once_with(query="what LLM does the agent use?")
    assert result == {"retrieved_context": fake_context}
