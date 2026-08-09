"""Pydantic models used to validate chat session data.

These models define the shape of a single conversation turn as stored and
exchanged by the API's session store and chat endpoint.

"""

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """A single turn in a conversation, matching the agent's expected shape.

    Attributes:
        role: who produced the message, either the end user or the agent.
        content: the raw text of the message.

    """

    role: Literal["user", "assistant"]
    content: str
