"""Instantiate the judge LLM used by the agent's LLM-as-a-Judge node.

Provides a single entry point for obtaining the chat model backing the
judge node, hosted on Groq like the responder LLM but from a different
model family, evaluating a response with the same model that generated
it risks the judge sharing (and failing to catch) the same blind spots.

"""

import logging

from langchain_groq import ChatGroq

from config import settings

logger = logging.getLogger(__name__)

JUDGE_MODEL_NAME = "qwen/qwen3.6-27b"


def get_judge_llm(temperature: float = 0.0) -> ChatGroq:
    """Instantiate the Groq-hosted chat model used to judge agent responses.

    Args:
        temperature (float, optional): Sampling temperature. Defaults to
            `0.0`, appropriate for a judgment task, which calls for
            consistent, deterministic evaluation rather than creative
            variation.

    Returns:
        ChatGroq: A configured chat model instance, ready to be used
            directly or wrapped via `with_structured_output`.

    """
    logger.info("Instantiating judge LLM '%s' via Groq...", JUDGE_MODEL_NAME)
    return ChatGroq(
        model=JUDGE_MODEL_NAME,
        temperature=temperature,
        api_key=settings.GROQ_API_KEY.get_secret_value(),
    )
