"""FastAPI routers exposed by the API layer.

One module per endpoint area, each defining an `APIRouter` included by
`api.main`:

- chat: POST /chat, free-text conversation routed through the agent.
- predict: POST /predict/ad-hoc and POST /predict/company, direct
    predictions bypassing the LLM.

"""
