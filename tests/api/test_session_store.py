"""Test suite for the in-memory chat session store.

Covers the happy path for each `SessionStore` method: reading the history of
a new session, reading the history of an existing session, appending a
user/assistant turn, and checking whether a session exists (including that
the check itself creates no side effect).

"""

from api.session_store import SessionStore
from schemas.api.session_validation import Message


def test_get_history_returns_empty_list_for_new_session() -> None:
    """Test a session that has never been touched has an empty history."""
    store = SessionStore()

    history = store.get_history("new-session")

    assert history == []


def test_get_history_returns_existing_messages() -> None:
    """Test an existing session's history is returned as stored."""
    store = SessionStore()
    store.append_turn("session-1", user_input="Hello", final_answer="Hi there")

    history = store.get_history("session-1")

    assert history == [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
    ]


def test_append_turn_adds_user_and_assistant_messages_in_order() -> None:
    """Test appending a turn adds a user message followed by an assistant message."""
    store = SessionStore()

    store.append_turn(
        "session-1", user_input="What is the risk?", final_answer="Low risk"
    )

    history = store.get_history("session-1")
    assert len(history) == 2
    assert history[0] == Message(role="user", content="What is the risk?")
    assert history[1] == Message(role="assistant", content="Low risk")


def test_append_turn_accumulates_across_multiple_calls() -> None:
    """Test multiple turns on the same session accumulate in chronological order."""
    store = SessionStore()

    store.append_turn(
        "session-1", user_input="First question", final_answer="First answer"
    )
    store.append_turn(
        "session-1", user_input="Second question", final_answer="Second answer"
    )

    history = store.get_history("session-1")
    assert len(history) == 4
    assert [message.content for message in history] == [
        "First question",
        "First answer",
        "Second question",
        "Second answer",
    ]


def test_sessions_are_isolated_from_each_other() -> None:
    """Test turns appended to one session do not affect another session's history."""
    store = SessionStore()

    store.append_turn("session-1", user_input="Hello", final_answer="Hi")
    store.append_turn("session-2", user_input="Hey", final_answer="Hello there")

    assert len(store.get_history("session-1")) == 2
    assert len(store.get_history("session-2")) == 2
    assert store.get_history("session-1")[0].content == "Hello"
    assert store.get_history("session-2")[0].content == "Hey"


def test_has_session_returns_false_for_unknown_session() -> None:
    """Test has_session returns False for a session that was never touched."""
    store = SessionStore()

    assert store.has_session("unknown-session") is False


def test_has_session_returns_true_after_append_turn() -> None:
    """Test has_session returns True once a turn has been appended."""
    store = SessionStore()
    store.append_turn("session-1", user_input="Hello", final_answer="Hi there")

    assert store.has_session("session-1") is True


def test_has_session_does_not_create_a_session_as_a_side_effect() -> None:
    """Test calling has_session does not create an entry for a unknown session."""
    store = SessionStore()

    store.has_session("unknown-session")

    assert store.has_session("unknown-session") is False
