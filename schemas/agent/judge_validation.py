"""Pydantic schema for the judge node's structured LLM output.

Validates the verdict produced by the judge node before it is merged into
`AgentState`. The decision itself is strictly binary — a judge can never
approve and reject at once — but `reason` is free text, detailed enough to
name which criterion failed and why, so it can be fed back into the
responder's prompt on a retry as a correction hint, not just logged.

"""

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Represent the judge node's evaluation of a generated response.

    Attributes:
        approved (bool): Whether the response passed evaluation and can be
            returned to the user as-is.
        reason (str): An explanation of the verdict — which criterion
            (e.g. faithfulness to the prediction results, faithfulness to
            the retrieved context, answer relevancy) was or wasn't met,
            and why. Written to be useful both as a log entry and as a
            correction hint re-injected into the responder's prompt on a
            retry.

    """

    approved: bool = Field(
        description="Whether the response is approved and can be returned "
        "to the user as-is."
    )
    reason: str = Field(
        description="Explanation of the verdict, naming which criterion "
        "was or wasn't met and why."
    )
