"""Free-text conversation endpoint, routed through the LangGraph agent.

Exposes a single endpoint, `POST /chat`, that forwards the user's message
to the compiled agent graph (case_a, case_b, rag, or direct routing handled
internally by the graph) and returns its natural-language answer.

"""

import uuid

from fastapi import APIRouter, Depends, Header
from langgraph.graph.state import CompiledStateGraph

from api.dependencies import get_agent, get_session_store
from api.session_store import SessionStore
from schemas.api.routers.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    session_id: str | None = Header(default=None),
    agent: CompiledStateGraph = Depends(get_agent),
    session_store: SessionStore = Depends(get_session_store),
) -> ChatResponse:
    """Send a free-text message to the agent and return its answer.

    A new `session_id` is generated when the client does not send one
    (e.g. the very first message of a conversation), and echoed back in
    the response so the client can reuse it on subsequent turns to
    continue the same conversation.

    Args:
        request (ChatRequest): The user's free-text message.
        session_id (str | None): The conversation identifier, read from
            the `Session-Id` request header. `None` if the client did not
            send one, in which case a new one is generated.
        agent (CompiledStateGraph): The compiled LangGraph agent, injected
            via `Depends(get_agent)`.
        session_store (SessionStore): The shared, in-memory session store,
            injected via `Depends(get_session_store)`.

    Returns:
        ChatResponse: The agent's natural-language answer, together with
            the `session_id` this turn belongs to.

    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    history = session_store.get_history(session_id)
    messages = [message.model_dump() for message in history]

    result = agent.invoke({"user_input": request.message, "messages": messages})
    final_answer = result.get("final_answer", "")

    session_store.append_turn(session_id, request.message, final_answer)

    return ChatResponse(final_answer=final_answer, session_id=session_id)
