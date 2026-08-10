"""Pydantic request/response schemas for each router in api.routers.

One module per router, mirroring `api.routers`:

- chat: ChatRequest, ChatResponse, the /chat endpoint's message body and
    the agent's answer plus session_id.
- predict: ExistingCompanyRequest, PredictionResponse, the /predict/ad-hoc
    and /predict/company endpoints' request and shared response bodies.

"""
