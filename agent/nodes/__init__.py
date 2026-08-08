# agent/nodes/__init__.py
"""LangGraph node implementations for the agent.

Each module defines one node function of the agent's graph (router,
extractor, predictor, retriever, responder, judge), following the
LangGraph convention of a function that reads the shared AgentState and
returns a partial state update.

"""
