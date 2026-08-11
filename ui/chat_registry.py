"""Client-side registry of known conversations, for the Streamlit sidebar.

Keeps track of which `session_id`s the user has started in this browser
session, and a short preview of each, so the sidebar can list past
conversations the way Claude.ai or ChatGPT do. Only the `session_id` and a
preview are kept here: the actual message history is never duplicated
client-side, and is instead read back from the server (the single source
of truth, via `GET /chat/{session_id}/history`) whenever the user switches
to a past conversation.

Storage is `st.session_state`, so it is scoped to a single browser tab and
lost when that tab is closed, exactly like `api/session_store.py` on the
server side is lost on server restart.

"""

import streamlit as st

PREVIEW_MAX_LENGTH = 50
REGISTRY_KEY = "conversations"
ACTIVE_SESSION_KEY = "active_session_id"


class Conversation:
    """A single conversation known to this browser session.

    Attributes:
        session_id (str): The conversation's identifier, as used by the
            API's `Session-Id` header and `session_id` path parameter.
        preview (str): A short, truncated preview of the conversation's
            first user message, for display in the sidebar.

    """

    def __init__(self, session_id: str, preview: str) -> None:
        """Initialize a Conversation with its identifier and preview.

        Args:
            session_id (str): The conversation's identifier.
            preview (str): A short, truncated preview of the
                conversation's first user message.

        """
        self.session_id = session_id
        self.preview = preview


def ensure_registry_initialized() -> None:
    """Initialize the registry's session_state entries on first use.

    Streamlit reruns the whole script on every interaction, so this must
    be idempotent: it only sets up `st.session_state` the first time it is
    called in a given browser session, leaving it untouched afterwards.

    """
    if REGISTRY_KEY not in st.session_state:
        st.session_state[REGISTRY_KEY] = {}
    if ACTIVE_SESSION_KEY not in st.session_state:
        st.session_state[ACTIVE_SESSION_KEY] = None


def register_conversation(session_id: str, first_message: str) -> None:
    """Add a new conversation to the registry, and make it the active one.

    Does nothing if `session_id` is already registered, so calling this
    again for an existing conversation (e.g. on a later turn of the same
    chat) never overwrites its original preview.

    Args:
        session_id (str): The conversation's identifier.
        first_message (str): The user's first message in this
            conversation, used to build the sidebar preview.

    """
    ensure_registry_initialized()

    if session_id not in st.session_state[REGISTRY_KEY]:
        preview = first_message.strip().replace("\n", " ")
        if len(preview) > PREVIEW_MAX_LENGTH:
            preview = preview[:PREVIEW_MAX_LENGTH].rstrip() + "..."

        st.session_state[REGISTRY_KEY][session_id] = Conversation(
            session_id=session_id, preview=preview
        )

    st.session_state[ACTIVE_SESSION_KEY] = session_id


def set_active_conversation(session_id: str) -> None:
    """Mark an already-registered conversation as the active one.

    Args:
        session_id (str): The identifier of the conversation to activate.

    """
    ensure_registry_initialized()

    st.session_state[ACTIVE_SESSION_KEY] = session_id


def get_active_conversation_id() -> str | None:
    """Return the currently active conversation's session_id, if any.

    Returns:
        str | None: The active `session_id`, or `None` if no conversation
            has been started yet in this browser session.

    """
    ensure_registry_initialized()

    return st.session_state[ACTIVE_SESSION_KEY]


def list_conversations() -> list[Conversation]:
    """Return every conversation known to this browser session.

    Returns:
        list[Conversation]: All registered conversations, in the order
            they were first registered.

    """
    ensure_registry_initialized()

    return list(st.session_state[REGISTRY_KEY].values())
