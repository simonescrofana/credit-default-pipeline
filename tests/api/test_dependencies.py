"""Test suite for the FastAPI dependency-injection helpers.

Covers the happy path for each dependency: `get_agent` reading the compiled
graph off the request's app state, and `get_session_store` returning the
shared session store singleton.

"""

from unittest.mock import MagicMock

from fastapi import Request

from api.dependencies import get_agent, get_session_store
from api.session_store import session_store


def test_get_agent_returns_agent_from_request_app_state() -> None:
    """Test get_agent returns the compiled graph stored on app.state.agent."""
    fake_agent = MagicMock()
    request = MagicMock(spec=Request)
    request.app.state.agent = fake_agent

    result = get_agent(request)

    assert result is fake_agent


def test_get_session_store_returns_shared_singleton() -> None:
    """Test get_session_store returns the module-level session_store instance."""
    result = get_session_store()

    assert result is session_store
