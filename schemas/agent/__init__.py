# schemas/agent/__init__.py
"""Pydantic validation schemas for the agentic layer.

Collects the request/response models used by the agent's LLM-backed
nodes — such as the router's structured classification output — kept
separate from the domain validation schemas used by the OLTP and ML
layers.

"""
