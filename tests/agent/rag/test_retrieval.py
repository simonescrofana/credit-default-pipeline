"""Test suite for the similarity search over the documentation index.

Covers the happy path of `retrieve_context`, verifying both that
the returned chunks of `collection.query` correctly pair each
document with its own distance and metadata (not misaligned across
the three parallel lists ChromaDB returns).

"""

from unittest.mock import MagicMock, patch

from agent.rag.retrieval import N_RESULTS, retrieve_context


@patch("agent.rag.retrieval.chromadb.PersistentClient")
def test_retrieve_context_happy_path(mock_client_class) -> None:
    """Verify retrieve_context queries the collection and shapes the result."""
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["chunk one", "chunk two"]],
        "distances": [[0.1, 0.4]],
        "metadatas": [
            [
                {"source": "python", "kind": "module", "name": "a", "file": "a.py"},
                {
                    "source": "readme",
                    "kind": "section",
                    "name": "Tech Stack",
                    "file": "README.md",
                },
            ]
        ],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    result = retrieve_context(query="what LLM does the agent use?")

    mock_client.get_collection.assert_called_once_with(name="project_docs")
    mock_collection.query.assert_called_once_with(
        query_texts=["what LLM does the agent use?"], n_results=N_RESULTS
    )

    assert len(result) == 2
    assert result[0] == {
        "text": "chunk one",
        "distance": 0.1,
        "metadata": {"source": "python", "kind": "module", "name": "a", "file": "a.py"},
    }
    assert result[1] == {
        "text": "chunk two",
        "distance": 0.4,
        "metadata": {
            "source": "readme",
            "kind": "section",
            "name": "Tech Stack",
            "file": "README.md",
        },
    }
