"""Streamlit entry point for the credit default pipeline's chat UI.

Assembles `ui.client` (HTTP calls to the FastAPI backend) and
`ui.chat_registry` (the client-side index of known conversations) into a
single-page chat interface: a sidebar to switch between conversations, and
a main area with the active conversation's messages and an input box.

Run with:
    streamlit run ui/app.py

"""

import sys
from pathlib import Path

# `streamlit run` executes this file as a standalone script, adding only
# its own directory (`ui/`) to sys.path, not the project root, so
# `ui.chat_registry` and `ui.client` would not be importable as a package
# without this. `uv run` alone does not fix this, since it only manages
# the environment/interpreter, not how Streamlit resolves the script it is
# told to execute as a file path rather than a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from ui.chat_registry import (
    get_active_conversation_id,
    list_conversations,
    register_conversation,
    set_active_conversation,
)
from ui.client import get_chat_history, send_chat_message

_MESSAGES_KEY = "messages"


def _get_displayed_messages() -> list[dict]:
    """Return the messages currently shown in the main area.

    Returns:
        list[dict]: The active conversation's messages so far, each a
            `{"role": ..., "content": ...}` dict. Empty if no conversation
            is active yet.

    """
    if _MESSAGES_KEY not in st.session_state:
        st.session_state[_MESSAGES_KEY] = []

    return st.session_state[_MESSAGES_KEY]


def _switch_to_conversation(session_id: str) -> None:
    """Make an existing conversation active and load its history from the API.

    Args:
        session_id (str): The identifier of the conversation to switch to.

    Returns:
        bool: `True` if the history was loaded successfully, `False` if it
            could not be loaded (in which case an error is already shown
            and the caller should not immediately rerun, or the error
            would be wiped before the user ever sees it).

    """
    set_active_conversation(session_id)

    try:
        history = get_chat_history(session_id)
    except requests.RequestException:
        st.error("Could not load this conversation's history.")
        st.session_state[_MESSAGES_KEY] = []
        return False

    st.session_state[_MESSAGES_KEY] = history["messages"]
    return True


st.set_page_config(page_title="Credit Default Pipeline", page_icon="💬")

with st.sidebar:
    st.header("Conversations")

    if st.button("New chat", use_container_width=True):
        set_active_conversation(None)
        st.session_state[_MESSAGES_KEY] = []

    for conversation in list_conversations():
        is_active = conversation.session_id == get_active_conversation_id()
        if st.button(
            conversation.preview,
            key=f"conversation-{conversation.session_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if _switch_to_conversation(conversation.session_id):
                st.rerun()

st.title("Credit Default Pipeline")

for message in _get_displayed_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about a company's insolvency risk...")

if user_input:
    displayed_messages = _get_displayed_messages()
    displayed_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = send_chat_message(user_input, get_active_conversation_id())
            except requests.RequestException:
                st.error(
                    "Something went wrong while contacting the agent. Please try again."
                )
            else:
                final_answer = result["final_answer"]
                session_id = result["session_id"]

                st.markdown(final_answer)
                displayed_messages.append(
                    {"role": "assistant", "content": final_answer}
                )

                if get_active_conversation_id() is None:
                    register_conversation(session_id, user_input)
                else:
                    set_active_conversation(session_id)
