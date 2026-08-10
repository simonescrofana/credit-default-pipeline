"""Dependency-injection helpers for the FastAPI endpoints.

Every function here is meant to be used with `Depends(...)` in endpoints, so
shared resources (agent, session store, loaded model, database session) are
not recreated on every request and can be swapped out in tests via
`app.dependency_overrides`.

"""

from functools import cache

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from api.session_store import SessionStore, session_store
from ml.inference.model_loader import LoadedModel, load_model


def get_agent(request: Request) -> CompiledStateGraph:
    """Return the LangGraph agent built once in the lifespan.

    The compiled graph only exists at runtime, created by `build_agent()`
    in the app's lifespan and stored on `app.state.agent`; there is no
    module-level instance to import directly, so this function is the only
    way to reach it from an endpoint.

    Args:
        request: the current request; FastAPI injects it automatically when
            this function is used with `Depends(get_agent)`.

    Returns:
        CompiledStateGraph: the compiled graph, ready to be invoked.

    """
    return request.app.state.agent


def get_session_store() -> SessionStore:
    """Return the module-level session store singleton.

    Unlike the agent, the session store is not built in the lifespan: it is
    a single instance already created at import time in
    `api.session_store`. Wrapping it in a function is still useful so
    endpoints can depend on it via `Depends(get_session_store)` and tests
    can swap it out with `app.dependency_overrides`.

    Returns:
        SessionStore: the shared, in-memory conversation session store.

    """
    return session_store


@cache
def get_loaded_model() -> LoadedModel:
    """Load and cache the production model bundle for the lifetime of the process.

    Mirrors the same `@cache` pattern used by the agent's predictor node
    (`agent.nodes.predictor_node.get_loaded_model`): loading the model
    queries MLflow and reads artifacts from disk, so it is done once per
    process and reused across every `/predict` request.

    Returns:
        LoadedModel: The model, encoder, scaler, explainer, and decision
            threshold bundle, loaded once and reused across every request.

    """
    return load_model()  # pragma: no cover
