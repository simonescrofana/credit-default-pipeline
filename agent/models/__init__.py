"""LLM instantiation for the agentic layer.

Centralizes the construction of every chat model used by the agent,
keeping provider and model choices isolated from node logic. Currently
provides the responder LLM (routing, extraction, and conversational
responses); a separate judge LLM is expected to be added here once the
LLM-as-a-Judge node is introduced.

"""
