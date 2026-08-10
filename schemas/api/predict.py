"""Pydantic schemas for the /predict endpoints' request and response bodies.

`PredictionResponse` is the shared response model for both `/predict` (ad
hoc data, case_b) and `/predict/existing` (a company already in the
database, case_a): it mirrors `ml.inference.predictor.PredictionResult`
field for field, except for the company id which should not be exposed,
but as a Pydantic model rather than a `NamedTuple`, so FastAPI can generate
response validation and OpenAPI documentation from it without coupling the
API layer to the ML layer's internal return type.

`ExistingCompanyRequest` is the request body for `/predict/existing`: a
single free-form identifier (legal name or VAT number), resolved
server-side the same way `agent.nodes.extractor.extract_case_a` resolves
it, via `utils.queries.RESOLVE_COMPANY_ID_QUERY`.

"""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """The outcome of a single company's insolvency prediction, as returned by the API.

    Attributes:
        company_name (str | None): The company's canonical legal name from
            the database, or `None` for ad hoc data (from `/predict`),
            which has no database record to draw a canonical name from.
        probability (float): The predicted probability of insolvency.
        predicted_class (int): The binary prediction (0 or 1), obtained by
            applying the decision threshold to `probability`.
        explanation (dict): A SHAP-based explanation of the prediction,
            mapping each feature to its original (encoded) value and
            Shapley contribution, sorted by descending absolute impact.

    """

    company_name: str | None = Field(
        default=None,
        description="Canonical legal name of the scored company. Always "
        "null for predictions made from ad hoc data, since there is no "
        "matching database record.",
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Predicted probability of insolvency, as a fraction "
        "between 0 and 1.",
    )
    predicted_class: int = Field(
        ...,
        description="Binary prediction (0: not insolvent, 1: insolvent), "
        "obtained by applying the model's decision threshold to "
        "`probability`.",
    )
    explanation: dict = Field(
        ...,
        description="SHAP-based explanation of the prediction, mapping "
        "each feature to its original value and its contribution to the "
        "prediction, sorted by descending absolute impact.",
    )


class ExistingCompanyRequest(BaseModel):
    """Request body for scoring a company already present in the database.

    Attributes:
        identifier (str): The company's legal name or VAT number, used to
            look it up in the database before scoring.

    """

    identifier: str = Field(
        ...,
        min_length=1,
        description="The company's legal name or VAT number, exactly as "
        "recorded in the database (e.g. 'Rossi S.r.l.' or its VAT "
        "number).",
    )
