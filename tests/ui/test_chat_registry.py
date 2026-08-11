"""Test suite for the client-side conversation registry (ui.chat_registry).

Covers the happy path for each public function, plus the one conditional
worth testing on its own: `register_conversation` must not overwrite an
existing conversation's preview when called again for the same
`session_id`. Since the module reads and writes `st.session_state`, which
only exists inside a running Streamlit script, every test runs its
assertions through `streamlit.testing.v1.AppTest`, reading results back via
`st.text(...)` elements rather than `print(...)`, since `AppTest` captures
rendered Streamlit elements, not raw stdout.

"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

SCRIPT_HEADER = """
import sys
sys.path.insert(0, "{project_root}")

import streamlit as st
from ui.chat_registry import (
    get_active_conversation_id,
    list_conversations,
    register_conversation,
    set_active_conversation,
)
"""


def _run_script(body: str) -> AppTest:
    """Run a Streamlit script made of the shared imports plus a test body.

    Args:
        body (str): The test-specific statements to run after the shared
            imports, typically ending in one or more `st.text(...)` calls
            whose rendered values the test asserts on.

    Returns:
        AppTest: The finished app run, ready for `.get("exception")` and
            `.text` inspection.

    """
    project_root = str(Path(__file__).resolve().parents[2])
    script = SCRIPT_HEADER.format(project_root=project_root) + body

    at = AppTest.from_string(script)
    at.run()

    return at


def test_get_active_conversation_id_returns_none_when_empty() -> None:
    """Test get_active_conversation_id returns None with no conversations registered."""
    at = _run_script("st.text(str(get_active_conversation_id()))")

    assert at.get("exception") == []
    assert at.text[0].value == "None"


def test_register_conversation_adds_it_and_makes_it_active() -> None:
    """Test register_conversation adds a new conversation and marks it active."""
    at = _run_script(
        'register_conversation("session-1", "Qual è il rischio di Rossi?")\n'
        "st.text(get_active_conversation_id())\n"
        "conversations = list_conversations()\n"
        "st.text(str(len(conversations)))\n"
        "st.text(conversations[0].session_id)\n"
        "st.text(conversations[0].preview)\n"
    )

    assert at.get("exception") == []
    values = [element.value for element in at.text]
    assert values == [
        "session-1",
        "1",
        "session-1",
        "Qual è il rischio di Rossi?",
    ]


def test_register_conversation_truncates_long_first_messages() -> None:
    """Test register_conversation truncates a preview longer than the max length."""
    long_message = "A" * 80

    at = _run_script(
        f'register_conversation("session-1", "{long_message}")\n'
        "st.text(list_conversations()[0].preview)\n"
    )

    assert at.get("exception") == []
    assert at.text[0].value == "A" * 50 + "..."


def test_register_conversation_does_not_overwrite_existing_preview() -> None:
    """Test recording chat again with same session_id keeps the original preview."""
    at = _run_script(
        'register_conversation("session-1", "Primo messaggio")\n'
        'register_conversation("session-1", "Messaggio diverso, '
        'non dovrebbe cambiare nulla")\n'
        "conversations = list_conversations()\n"
        "st.text(str(len(conversations)))\n"
        "st.text(conversations[0].preview)\n"
    )

    assert at.get("exception") == []
    values = [element.value for element in at.text]
    assert values == ["1", "Primo messaggio"]


def test_set_active_conversation_switches_the_active_session() -> None:
    """Test set_active_conversation changes which conversation is active."""
    at = _run_script(
        'register_conversation("session-1", "Prima chat")\n'
        'register_conversation("session-2", "Seconda chat")\n'
        'set_active_conversation("session-1")\n'
        "st.text(get_active_conversation_id())\n"
    )

    assert at.get("exception") == []
    assert at.text[0].value == "session-1"


def test_list_conversations_returns_all_registered_conversations_in_order() -> None:
    """Test list_conversations returns every registered conversation."""
    at = _run_script(
        'register_conversation("session-1", "Prima chat")\n'
        'register_conversation("session-2", "Seconda chat")\n'
        'register_conversation("session-3", "Terza chat")\n'
        "st.text(str([c.session_id for c in list_conversations()]))\n"
    )

    assert at.get("exception") == []
    assert at.text[0].value == "['session-1', 'session-2', 'session-3']"
