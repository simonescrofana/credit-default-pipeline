"""Shared utilities for calling LLMs with structured output reliably.

Provides `invoke_with_retry`, wrapping a structured-output LLM's `.invoke`
call with a retry for a known, intermittent Groq failure mode: on the
`openai/gpt-oss-*` model family, `tool_choice="required"` sometimes causes
the model to respond with plain text instead of the required tool call,
raising `groq.BadRequestError` with code `tool_use_failed` and message
"Tool choice is required, but model did not call a tool" — documented
across multiple independent reports (Groq's own community forum, the
LangChain and pydantic-ai issue trackers) as a transport-level flakiness in
the model/provider combination itself, not a sign of a malformed prompt or
schema. A short retry resolves it in practice, since the failure is
intermittent rather than deterministic.

"""

import logging

from groq import BadRequestError
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _is_tool_use_failed(exc: BadRequestError) -> bool:
    """Check whether a BadRequestError is the known tool_use_failed flakiness.

    Args:
        exc (BadRequestError): The exception raised by the Groq client.

    Returns:
        bool: True if the error body's code is "tool_use_failed", False
            otherwise (e.g. a genuine schema violation, which should not
            be retried and instead surface immediately).

    """
    body = getattr(exc, "body", None) or {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    return error.get("code") == "tool_use_failed"


def invoke_with_retry(
    structured_llm: Runnable,
    messages: list[BaseMessage],
    max_retries: int = MAX_RETRIES,
) -> BaseModel:
    """Invoke a structured-output LLM, retrying on tool_use_failed errors.

    Args:
        structured_llm (Runnable): An LLM already wrapped via
            `with_structured_output(...)`.
        messages (list[BaseMessage]): The messages to send.
        max_retries (int, optional): How many attempts to make before
            giving up and re-raising. Defaults to `MAX_RETRIES`.

    Returns:
        BaseModel: The validated structured output.

    Raises:
        BadRequestError: If every attempt fails with tool_use_failed, or
            immediately for any other kind of error (never retried).

    """
    last_exception: BadRequestError | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return structured_llm.invoke(messages)
        except BadRequestError as exc:
            if not _is_tool_use_failed(exc):
                raise
            last_exception = exc
            logger.warning(
                "tool_use_failed on attempt %d/%d, retrying...", attempt, max_retries
            )

    raise last_exception
