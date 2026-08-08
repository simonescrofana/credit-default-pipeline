"""Shared type aliases for the agentic layer's validation schemas.

Collects reusable types shared across multiple schemas in `schemas.agent`.

"""

from typing import Literal

AgentRoute = Literal["case_a", "case_b", "rag", "direct"]
