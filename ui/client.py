"""Thin HTTP client for the FastAPI backend, used by the Streamlit UI.

Wraps the single call the UI needs today, `POST /chat`. Deliberately does
not import any Pydantic model from `schemas.api`: the UI is a separate
process (and, once containerized, a separate image) from the API, so it
treats the response as plain JSON rather than sharing server-side types
across a process boundary.

"""

import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def send_chat_message(message: str, session_id: str | None) -> dict:
    """Send a free-text message to the agent via POST /chat.

    Args:
        message (str): The user's free-text message.
        session_id (str | None): The conversation identifier to send in
            the `Session-Id` header, or `None` to start a new
            conversation (the API generates one and returns it in the
            response).

    Returns:
        dict: The parsed JSON response body, with `final_answer` and
            `session_id` keys, as returned by `POST /chat`.

    Raises:
        requests.HTTPError: If the API responds with a non-2xx status.

    """
    headers = {}
    if session_id is not None:
        headers["Session-Id"] = session_id

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"message": message},
        headers=headers,
    )
    response.raise_for_status()

    return response.json()
