"""Test suite for the Streamlit UI's HTTP client (ui.client).

Covers the happy path for `send_chat_message`, both without a session_id
(a new conversation, no `Session-Id` header sent) and with one (the header
is sent and forwarded correctly), plus the `requests.HTTPError` raised when
the API responds with a non-2xx status. Also covers `get_chat_history`: its
happy path, and the same `requests.HTTPError` propagation (e.g. on a 404
for an unknown session). `requests.post`/`requests.get` are mocked, so no
real API server is needed.

"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from ui.client import API_BASE_URL, get_chat_history, send_chat_message


@patch("ui.client.requests.post")
def test_send_chat_message_without_session_id_sends_no_header(
    mock_post,
) -> None:
    """Test send_chat_message omits the Session-Id header when session_id is None."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "final_answer": "Il rischio è basso.",
        "session_id": "generated-id",
    }
    mock_post.return_value = mock_response

    result = send_chat_message("Ciao!", None)

    assert result == {
        "final_answer": "Il rischio è basso.",
        "session_id": "generated-id",
    }
    mock_post.assert_called_once_with(
        f"{API_BASE_URL}/chat",
        json={"message": "Ciao!"},
        headers={},
    )


@patch("ui.client.requests.post")
def test_send_chat_message_with_session_id_sends_header(
    mock_post: MagicMock,
) -> None:
    """Test send_chat_message forwards session_id in the Session-Id header."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "final_answer": "Il rischio è alto.",
        "session_id": "abc-123",
    }
    mock_post.return_value = mock_response

    result = send_chat_message("E ora?", "abc-123")

    assert result == {"final_answer": "Il rischio è alto.", "session_id": "abc-123"}
    mock_post.assert_called_once_with(
        f"{API_BASE_URL}/chat",
        json={"message": "E ora?"},
        headers={"Session-Id": "abc-123"},
    )


@patch("ui.client.requests.post")
def test_send_chat_message_raises_on_http_error(mock_post: MagicMock) -> None:
    """Test send_chat_message propagates requests.HTTPError on a non-2xx response."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("422 Client Error")
    mock_post.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        send_chat_message("", "abc-123")


@patch("ui.client.requests.get")
def test_get_chat_history_returns_parsed_response(mock_get) -> None:
    """Test get_chat_history returns the parsed JSON response for a known session."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "session_id": "abc-123",
        "messages": [
            {"role": "user", "content": "ciao"},
            {"role": "assistant", "content": "ciao a te"},
        ],
    }
    mock_get.return_value = mock_response

    result = get_chat_history("abc-123")

    assert result == {
        "session_id": "abc-123",
        "messages": [
            {"role": "user", "content": "ciao"},
            {"role": "assistant", "content": "ciao a te"},
        ],
    }
    mock_get.assert_called_once_with(f"{API_BASE_URL}/chat/abc-123/history")


@patch("ui.client.requests.get")
def test_get_chat_history_raises_on_http_error(mock_get: MagicMock) -> None:
    """Test get_chat_history propagates requests.HTTPError on a non-2xx response."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        get_chat_history("unknown-session")
