"""Pydantic validation schemas for the HTTP API layer.

Collects the request/response models used by the `api` package's endpoints
and session store — such as chat turns and prediction responses — kept
separate from the domain validation schemas used by the OLTP, ML, and agent
layers.

"""
