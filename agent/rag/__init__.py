# agent/rag/__init__.py
"""Retrieval-augmented generation layer for the agent.

Builds and serves the documentation index that the agent's rag route
queries: `extract_docs.py` collects docstrings and dbt descriptions
from the project into a single JSON file, and `ingest.py` embeds and
loads them into ChromaDB for retrieval.

"""
