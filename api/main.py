"""FastAPI application entrypoint for the credit default pipeline.

Builds the compiled LangGraph agent once at startup (never per-request,
since building it is expensive), includes the `chat` and `predict` routers,
and exposes a `GET /health` endpoint for container orchestration.

"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.graph import build_agent
from api.routers.chat import router as chat_router
from api.routers.predict import router as predict_router
from utils.logging_utils import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Build the agent once at startup and attach it to app.state.

    Runs once when the server starts, before any request is served, and
    once more when it shuts down. `agent.graph.build_agent()` is not cheap
    (it wires together every node of the graph), so it must not run on
    every request; storing it on `app.state.agent` makes it reachable from
    every endpoint via `api.dependencies.get_agent`.

    Args:
        app (FastAPI): The application instance being started.

    Yields:
        None: Control is handed back to FastAPI to serve requests; nothing
            is torn down explicitly on shutdown, as the agent holds no
            external resources of its own to release.

    """
    setup_logging(log_level="INFO")
    app.state.agent = build_agent()

    yield


app = FastAPI(
    title="Credit Default Prediction API",
    description="HTTP API for the credit default insolvency prediction "
    "pipeline: a free-text chat endpoint routed through a LangGraph "
    "agent, and direct prediction endpoints bypassing the LLM.",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(predict_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the service is up and ready to serve requests.

    Used by container orchestration (e.g. a Docker Compose healthcheck) to
    tell a running-but-not-yet-ready container (still building the agent
    in its lifespan) apart from one that has finished starting up.

    Returns:
        dict[str, str]: A fixed `{"status": "ok"}` payload.

    """
    return {"status": "ok"}
