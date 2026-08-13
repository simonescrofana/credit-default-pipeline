"""Pydantic schemas for the /chat endpoints' request and response bodies.

`ChatRequest` carries the user's free-text message; `ChatResponse` carries
the agent's natural-language answer plus the `session_id` the client should
reuse on subsequent turns, so a new conversation can start without the
client having to invent an identifier up front. `ChatHistoryResponse`
carries the full message history of an existing conversation, read back via
`GET /chat/{session_id}/history`.

"""

from pydantic import BaseModel, Field

from schemas.api.session_validation import Message


class ChatRequest(BaseModel):
    """Request body for a single chat turn.

    Attributes:
        message (str): The user's free-text message to the agent.

    """

    message: str = Field(
        ...,
        min_length=1,
        description="Free-text message to send to the agent (e.g. a "
        "question about a company's insolvency risk, or a request for "
        "prediction on ad hoc data).",
    )


class ChatResponse(BaseModel):
    """Response body for a single chat turn.

    Attributes:
        final_answer (str): The agent's natural-language answer.
        session_id (str): The identifier of the conversation this turn
            belongs to. Echoes the `Session-Id` header sent by the client,
            or a newly generated one if the client did not send it, so the
            client can reuse it on subsequent turns to continue the same
            conversation.

    """

    final_answer: str = Field(
        ...,
        description="The agent's natural-language answer to the user's message.",
    )
    session_id: str = Field(
        ...,
        description="Identifier of the conversation this turn belongs to. "
        "Send it back in the Session-Id header on the next request to "
        "continue the same conversation.",
    )


class ChatHistoryResponse(BaseModel):
    """Response body for reading back an existing conversation's history.

    Attributes:
        session_id (str): The identifier of the conversation this history
            belongs to, echoed back for confirmation.
        messages (list[Message]): The conversation's messages so far, in
            chronological order. In practice never empty for a session
            that exists, since a session is only ever created by
            appending its first turn.

    """

    session_id: str = Field(
        ...,
        description="Identifier of the conversation this history belongs to.",
    )
    messages: list[Message] = Field(
        ...,
        description="The conversation's messages so far, in chronological order.",
    )
