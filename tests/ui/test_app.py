"""Test suite for the Streamlit chat UI's entry point (ui.app).

Covers every conditional branch in the script: sending a first message
(new conversation, registered on success), sending a message in an
existing conversation (session_id kept, no re-registration), a network
error while sending a message, switching to a past conversation from the
sidebar (both the happy path and the case where its history fails to
load), the "New chat" button, and submitting no input at all.

Every test patches `ui.client.requests.post` / `ui.client.requests.get`
directly, one level below the re-imported names, so no real API server
is needed.

"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "ui" / "app.py"


def _run_app() -> AppTest:
    """Load and run ui/app.py from a fresh state.

    Returns:
        AppTest: The app after its first run, ready for widget interaction.

    """
    at = AppTest.from_file(str(APP_PATH))
    at.run()

    return at


@patch("ui.client.requests.post")
def test_sending_first_message_registers_a_new_conversation(
    mock_post,
) -> None:
    """Test sending the first message shows the answer recording the conversation."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "final_answer": "Il rischio è basso.",
        "session_id": "new-session-id",
    }
    mock_post.return_value = mock_response

    at = _run_app()
    at.chat_input[0].set_value("Qual è il rischio di Rossi?").run()

    assert at.get("exception") == []
    markdown_values = [element.value for element in at.markdown]
    assert "Qual è il rischio di Rossi?" in markdown_values
    assert "Il rischio è basso." in markdown_values

    at.run()
    button_labels = [button.label for button in at.button]
    assert any("Qual è il rischio di Rossi?" in label for label in button_labels)


@patch("ui.client.requests.post")
def test_sending_message_in_existing_conversation_keeps_same_session(
    mock_post,
) -> None:
    """Test sending a second message in an active conversation reuses its session_id."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "final_answer": "Prima risposta.",
        "session_id": "session-1",
    }
    mock_post.return_value = mock_response

    at = _run_app()
    at.chat_input[0].set_value("Primo messaggio").run()

    mock_response.json.return_value = {
        "final_answer": "Seconda risposta.",
        "session_id": "session-1",
    }
    at.chat_input[0].set_value("Secondo messaggio").run()

    assert at.get("exception") == []
    conversation_buttons = [
        button for button in at.button if button.label != "New chat"
    ]
    assert len(conversation_buttons) == 1
    assert conversation_buttons[0].label == "Primo messaggio"


@patch("ui.client.requests.post")
def test_sending_message_shows_error_on_network_failure(
    mock_post,
) -> None:
    """Test a network error while sending a message does not cause crashing."""
    mock_post.side_effect = requests.ConnectionError("Failed to establish a connection")

    at = _run_app()
    at.chat_input[0].set_value("Qual è il rischio di Rossi?").run()

    assert at.get("exception") == []
    assert len(at.error) == 1
    assert "went wrong" in at.error[0].value.lower()


@patch("ui.client.requests.get")
@patch("ui.client.requests.post")
def test_switching_to_past_conversation_loads_its_history(
    mock_post,
    mock_get,
) -> None:
    """Test clicking a conversation in the sidebar loads and displays its history."""
    mock_post_response = MagicMock()
    mock_post_response.json.return_value = {
        "final_answer": "Prima risposta.",
        "session_id": "session-1",
    }
    mock_post.return_value = mock_post_response

    at = _run_app()
    at.chat_input[0].set_value("Primo messaggio").run()
    at.run()

    mock_get_response = MagicMock()
    mock_get_response.json.return_value = {
        "session_id": "session-1",
        "messages": [
            {"role": "user", "content": "Primo messaggio"},
            {"role": "assistant", "content": "Prima risposta."},
        ],
    }
    mock_get.return_value = mock_get_response

    conversation_button = next(
        button for button in at.button if button.label == "Primo messaggio"
    )
    conversation_button.click().run()

    assert at.get("exception") == []
    mock_get.assert_called_once()
    markdown_values = [element.value for element in at.markdown]
    assert "Primo messaggio" in markdown_values
    assert "Prima risposta." in markdown_values


@patch("ui.client.requests.get")
@patch("ui.client.requests.post")
def test_switching_to_past_conversation_shows_error_if_history_fails_to_load(
    mock_post,
    mock_get,
) -> None:
    """Test a failed history load shows an error and clears the displayed messages."""
    mock_post_response = MagicMock()
    mock_post_response.json.return_value = {
        "final_answer": "Prima risposta.",
        "session_id": "session-1",
    }
    mock_post.return_value = mock_post_response

    at = _run_app()
    at.chat_input[0].set_value("Primo messaggio").run()
    at.run()

    mock_get.side_effect = requests.ConnectionError("Failed to establish a connection")

    conversation_button = [
        button for button in at.button if button.label == "Primo messaggio"
    ]
    conversation_button[0].click().run()

    assert at.get("exception") == []
    assert len(at.error) == 1
    assert "history" in at.error[0].value.lower()
    assert at.chat_message == []


@patch("ui.client.requests.post")
def test_new_chat_button_clears_the_active_conversation(mock_post) -> None:
    """Test clicking "New chat" clears the active conversation and its messages."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "final_answer": "Prima risposta.",
        "session_id": "session-1",
    }
    mock_post.return_value = mock_response

    at = _run_app()
    at.chat_input[0].set_value("Primo messaggio").run()

    new_chat_button = next(button for button in at.button if button.label == "New chat")
    new_chat_button.click().run()

    assert at.get("exception") == []
    assert at.chat_message == []


def test_no_input_renders_nothing_extra() -> None:
    """Test running the app with no submitted input shows no chat messages."""
    at = _run_app()

    assert at.get("exception") == []
    assert at.chat_message == []
