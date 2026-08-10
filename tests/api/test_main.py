"""Test suite for the FastAPI application entrypoint (api.main).

Covers the two pieces of actual logic in this module: the `GET /health`
endpoint, and the lifespan, which must build the agent once at startup and
attach it to `app.state.agent`. Mounting the two routers is not tested
here, since that only exercises FastAPI's own `include_router`, not any
logic of this project; the routers themselves are already tested in
`tests.api.routers`.

"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app


def test_health_returns_ok() -> None:
    """Test GET /health returns a 200 with the fixed ok status payload."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.main.setup_logging")
@patch("api.main.build_agent")
def test_lifespan_builds_agent_and_configures_logging(
    mock_build_agent,
    mock_setup_logging,
) -> None:
    """Test the lifespan configures logging and attaches the agent to app.state."""
    fake_agent = MagicMock()
    mock_build_agent.return_value = fake_agent

    with TestClient(app):
        assert app.state.agent is fake_agent

    mock_setup_logging.assert_called_once_with(log_level="INFO")
    mock_build_agent.assert_called_once_with()
