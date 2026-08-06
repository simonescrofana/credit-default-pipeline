"""Similarity search over the project's ChromaDB documentation index.

Provides the single function `retrieve_context`, used by `retriever_node`
to turn a user's request into a list of relevant documentation chunks. No
similarity-score threshold is applied here: the top `N_RESULTS` chunks are
always returned as-is, and it is the judge node's responsibility, later in
the graph, to catch a response that isn't actually supported by the
retrieved context, rather than guessing a cutoff score now without reald
usage data to calibrate it against.

"""

import logging
from pathlib import Path
from typing import Any

import chromadb

from agent.rag.ingest import CHROMA_PERSIST_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)

N_RESULTS = 6


def retrieve_context(
    query: str,
    n_results: int = N_RESULTS,
    persist_path: Path = CHROMA_PERSIST_PATH,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant documentation chunks for a query.

    Args:
        query (str): The text to search against the indexed documentation
            (typically the user's request).
        n_results (int, optional): How many chunks to retrieve. Defaults
            to `N_RESULTS`. Kept relatively high (rather than a stricter
            top-1/top-2) since a single request can bundle several
            sub-questions, or ask broadly about the project.
        persist_path (Path, optional): Directory where the ChromaDB index
            is persisted. Defaults to the project's actual path.

    Returns:
        list[dict[str, Any]]: One entry per retrieved chunk, each with
            `text` (the chunk's content), `distance` (cosine distance to
            the query — lower means more similar), and `metadata` (the
            chunk's `source`, `kind`, `name`, and `file`, as written by
            `ingest.py`).

    """
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_collection(name=COLLECTION_NAME)

    logger.info("Retrieving context for query...")
    result = collection.query(query_texts=[query], n_results=n_results)

    documents = result["documents"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]

    context = [
        {"text": doc, "distance": dist, "metadata": meta}
        for doc, dist, meta in zip(documents, distances, metadatas)
    ]

    logger.info("Retrieved %d chunk(s).", len(context))
    return context
