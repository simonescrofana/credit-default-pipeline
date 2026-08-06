"""Instantiate the primary LLM used by the agent's conversational nodes.

Provide a single entry point for obtaining the chat model backing the
router, extractor, and responder nodes, hosted on Groq. Centralizing
instantiation here means the model choice can change without touching any
node's logic.

"""

import logging

from langchain_groq import ChatGroq

from config import settings

logger = logging.getLogger(__name__)

RESPONDER_MODEL_NAME = "openai/gpt-oss-120b"


def get_responder_llm(temperature: float = 0.7) -> ChatGroq:
    """Instantiate the Groq-hosted chat model used for agent responses.

    Args:
        temperature (float, optional): Sampling temperature. Defaults to
            `0.7`, appropriate for free-form conversational responses;
            callers using this model for classification or extraction nodes
            should pass a lower value (e.g. `0.0`) for deterministic output.

    Returns:
        ChatGroq: A configured chat model instance, ready to be used
            directly or wrapped via `with_structured_output`.

    """
    logger.info("Instantiating responder LLM '%s' via Groq...", RESPONDER_MODEL_NAME)
    return ChatGroq(
        model=RESPONDER_MODEL_NAME,
        temperature=temperature,
        api_key=settings.GROQ_API_KEY.get_secret_value(),
    )
