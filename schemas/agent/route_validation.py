"""Pydantic schema for the router node's structured LLM output.

Validates the classification produced by the router node before it is
merged into `AgentState`. Kept intentionally minimal, the router's only
responsibility is choosing a route, not extracting or processing any
other data, so the schema mirrors that single-field scope rather than
reusing the full `AgentState` model. Also defines `AgentRoute`, the shared
type alias for the four possible routes, reused by `agent.state.AgentState`
so the set of valid routes is defined once, not duplicated.

"""

from typing import Literal

from pydantic import BaseModel, Field

from schemas.agent.types import AgentRoute


class RouterDecision(BaseModel):
    """Represent the router node's classification of a user request.

    Attributes:
        route (AgentRoute): The chosen path for the request, one of
            "case_a" (prediction for a named, existing company), "case_b"
            (prediction on ad hoc user-supplied data), "rag" (documentation
            retrieval), or "direct" (no external grounding needed).

    """

    route: AgentRoute = Field(
        description=(
            "The classification of the user's request into one of the "
            "agent's four handling routes."
        )
    )
