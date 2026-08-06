"""Retriever node: fetches relevant documentation for the rag route.

Reads the user's request, searches the project's documentation index via
`agent.rag.retrieval.retrieve_context`, and writes the result to
`AgentState.retrieved_context` for the responder node to ground its answer
in.

"""

import logging

from agent.rag.retrieval import retrieve_context
from agent.state import AgentState

logger = logging.getLogger(__name__)


def retriever_node(state: AgentState) -> dict:
    """Retrieve documentation chunks relevant to the user's request.

    Args:
        state (AgentState): The current graph state. Reads `user_input`.

    Returns:
        dict: A partial state update setting `retrieved_context` to the
            chunks retrieved for the request.

    """
    context = retrieve_context(query=state.user_input)
    return {"retrieved_context": context}
