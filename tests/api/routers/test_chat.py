"""Test suite for the /chat endpoint.

Covers the happy path when the client sends a `Session-Id` header (also
verifying the prior message history is correctly converted and passed to
the agent), and the case where no header is sent, which must generate a
new, valid session_id. Dependency injection is bypassed via
`app.dependency_overrides`; the agent and session store are mocked, so no
real graph or storage is needed.

"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_agent, get_session_store
from api.routers.chat import router
from schemas.api.session_validation import Message


@pytest.fixture
def fake_agent() -> MagicMock:
    """Build a mocked compiled agent returning a fixed final_answer.

    Returns:
        MagicMock: A mock whose `invoke` method returns a fixed
            `final_answer`, regardless of the state passed to it.

    """
    agent = MagicMock()
    agent.invoke.return_value = {"final_answer": "Il rischio è basso."}

    return agent


@pytest.fixture
def fake_session_store() -> MagicMock:
    """Build a mocked session store with a two-message prior history.

    Returns:
        MagicMock: A mock whose `get_history` returns one prior user/
            assistant turn, so tests can verify it is correctly forwarded
            to the agent.

    """
    store = MagicMock()
    store.get_history.return_value = [
        Message(role="user", content="ciao"),
        Message(role="assistant", content="ciao, come posso aiutarti?"),
    ]

    return store


@pytest.fixture
def client(fake_agent: MagicMock, fake_session_store: MagicMock) -> TestClient:
    """Build a TestClient with get_agent and get_session_store overridden.

    Args:
        fake_agent (MagicMock): The mocked agent to inject.
        fake_session_store (MagicMock): The mocked session store to inject.

    Returns:
        TestClient: A client for the isolated `chat` router, with its
            agent and session store dependencies replaced by mocks.

    """
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_agent] = lambda: fake_agent
    app.dependency_overrides[get_session_store] = lambda: fake_session_store

    return TestClient(app)


def test_chat_returns_answer_and_forwards_history(
    client: TestClient,
    fake_agent: MagicMock,
    fake_session_store: MagicMock,
) -> None:
    """Test POST /chat returns the agent's answer and forwards the prior history."""
    response = client.post(
        "/chat",
        json={"message": "Qual è il rischio di Rossi S.r.l.?"},
        headers={"Session-Id": "abc-123"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "final_answer": "Il rischio è basso.",
        "session_id": "abc-123",
    }

    fake_agent.invoke.assert_called_once_with(
        {
            "user_input": "Qual è il rischio di Rossi S.r.l.?",
            "messages": [
                {"role": "user", "content": "ciao"},
                {"role": "assistant", "content": "ciao, come posso aiutarti?"},
            ],
        }
    )
    fake_session_store.append_turn.assert_called_once_with(
        "abc-123",
        "Qual è il rischio di Rossi S.r.l.?",
        "Il rischio è basso.",
    )


def test_chat_generates_session_id_when_header_missing(
    client: TestClient,
    fake_session_store: MagicMock,
) -> None:
    """Test POST /chat generates a valid, new session_id when no header is sent."""
    response = client.post("/chat", json={"message": "Ciao!"})

    assert response.status_code == 200

    generated_session_id = response.json()["session_id"]

    fake_session_store.append_turn.assert_called_once_with(
        generated_session_id,
        "Ciao!",
        "Il rischio è basso.",
    )

    # A second call without a header must get its own, different
    # session_id: catches a regeneration bug where the "new" id is
    # accidentally fixed or memoized instead of freshly random each time.
    second_response = client.post("/chat", json={"message": "Ancora ciao!"})
    assert second_response.json()["session_id"] != generated_session_id
