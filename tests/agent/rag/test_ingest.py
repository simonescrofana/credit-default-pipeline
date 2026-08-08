"""Test suite for the ChromaDB ingestion script.

Covers `iter_docs_json` (happy path), `chunk_readme` (a plain `##`
section stays whole, a `##` with `####` children is skipped as a whole
chunk but any intro text before the first child is still captured, a `##`
with children and no intro text yields no spurious empty chunk, and a
`####` section stops at the next header of equal or shallower level),
`_make_id` (happy path, uniqueness across differing fields), and `ingest`
(happy path with a mocked ChromaDB client, the idempotent-clear branch, and
the first-run branch where the collection doesn't exist yet).

"""

from unittest.mock import MagicMock, call, patch

from chromadb.errors import NotFoundError

from agent.rag.ingest import _make_id, chunk_readme, ingest, iter_docs_json


def test_iter_docs_json_happy_path(tmp_path) -> None:
    """Verify iter_docs_json yields every document from the JSON file."""
    docs_file = tmp_path / "docs_data.json"
    docs_file.write_text(
        '[{"source": "python", "kind": "module", "name": "a", "file": "a.py", '
        '"text": "Doc A."}, '
        '{"source": "dbt", "kind": "model", "name": "b", "file": "b.yml", '
        '"text": "Doc B."}]'
    )

    docs = list(iter_docs_json(docs_file))

    assert len(docs) == 2
    assert docs[0]["name"] == "a"
    assert docs[1]["name"] == "b"


def test_chunk_readme_plain_section_stays_whole(tmp_path) -> None:
    """Verify a '##' section with no '####' children is one full chunk."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Tech Stack\n"
        "Python, LangGraph, ChromaDB.\n"
        "\n"
        "## Results\n"
        "Some metrics here.\n"
    )

    chunks = list(chunk_readme(readme))

    assert len(chunks) == 2
    assert chunks[0]["name"] == "Tech Stack"
    assert "Python, LangGraph, ChromaDB." in chunks[0]["text"]
    assert chunks[1]["name"] == "Results"
    assert "Some metrics here." in chunks[1]["text"]


def test_chunk_readme_section_with_children_and_intro_captures_intro(tmp_path) -> None:
    """Verify intro text before the first '####' child becomes its own chunk."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Rationale\n"
        "Intro text before any sub-section starts.\n"
        "\n"
        "#### First Decision\n"
        "* Choice: A\n"
        "\n"
        "#### Second Decision\n"
        "* Choice: B\n"
    )

    chunks = list(chunk_readme(readme))

    names = [c["name"] for c in chunks]
    assert names == ["Rationale", "First Decision", "Second Decision"]
    assert "Intro text before any sub-section starts." in chunks[0]["text"]
    assert "#### First Decision" not in chunks[0]["text"]


def test_chunk_readme_section_with_children_and_no_intro_emits_no_spurious_chunk(
    tmp_path,
) -> None:
    """Verify a '##' immediately followed by '####' yields no empty intro chunk."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Rationale\n"
        "\n"
        "#### First Decision\n"
        "* Choice: A\n"
        "\n"
        "#### Second Decision\n"
        "* Choice: B\n"
    )

    chunks = list(chunk_readme(readme))

    names = [c["name"] for c in chunks]
    assert names == ["First Decision", "Second Decision"]


def test_chunk_readme_subsection_stops_at_next_header(tmp_path) -> None:
    """Verify a '####' chunk's text stops before the next header, not beyond it."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Rationale\n"
        "\n"
        "#### First Decision\n"
        "First decision body.\n"
        "\n"
        "#### Second Decision\n"
        "Second decision body.\n"
        "\n"
        "## Next Section\n"
        "Unrelated content.\n"
    )

    chunks = list(chunk_readme(readme))

    first = next(c for c in chunks if c["name"] == "First Decision")
    assert "First decision body." in first["text"]
    assert "Second decision body." not in first["text"]
    assert "Unrelated content." not in first["text"]


def test_make_id_is_stable_and_unique() -> None:
    """Verify _make_id builds a distinct id per distinct document/index."""
    doc_a = {"source": "python", "file": "a.py", "name": "foo"}
    doc_b = {"source": "python", "file": "b.py", "name": "foo"}

    id_a = _make_id(doc_a, 0)
    id_b = _make_id(doc_b, 0)
    id_a_different_index = _make_id(doc_a, 1)

    assert id_a != id_b
    assert id_a != id_a_different_index


@patch("agent.rag.ingest.chromadb.PersistentClient")
def test_ingest_happy_path(mock_client_class, tmp_path) -> None:
    """Verify ingest embeds every chunk from both sources into the collection."""
    docs_file = tmp_path / "docs_data.json"
    docs_file.write_text(
        '[{"source": "python", "kind": "module", "name": "a", "file": "a.py", '
        '"text": "Doc A."}]'
    )
    readme_file = tmp_path / "README.md"
    readme_file.write_text("## Tech Stack\nSome content.\n")

    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    total = ingest(
        docs_json_path=docs_file,
        readme_path=readme_file,
        persist_path=tmp_path / "chroma_db",
    )

    assert total == 2
    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args.kwargs
    assert len(call_kwargs["ids"]) == 2
    assert len(call_kwargs["documents"]) == 2
    assert len(call_kwargs["metadatas"]) == 2


@patch("agent.rag.ingest.chromadb.PersistentClient")
def test_ingest_clears_existing_collection_first(mock_client_class, tmp_path) -> None:
    """Verify ingest deletes the existing collection before recreating it."""
    docs_file = tmp_path / "docs_data.json"
    docs_file.write_text("[]")
    readme_file = tmp_path / "README.md"
    readme_file.write_text("## Section\nContent.\n")

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = MagicMock()
    mock_client_class.return_value = mock_client

    ingest(
        docs_json_path=docs_file,
        readme_path=readme_file,
        persist_path=tmp_path / "chroma_db",
    )

    mock_client.delete_collection.assert_called_once()
    # deletion must happen before the collection is (re)created
    assert mock_client.mock_calls.index(
        call.delete_collection(name="project_docs")
    ) < mock_client.mock_calls.index(
        call.get_or_create_collection(
            name="project_docs", metadata={"hnsw:space": "cosine"}
        )
    )


@patch("agent.rag.ingest.chromadb.PersistentClient")
def test_ingest_first_run_with_no_existing_collection(
    mock_client_class, tmp_path
) -> None:
    """Verify ingest proceeds normally when the collection doesn't exist yet."""
    docs_file = tmp_path / "docs_data.json"
    docs_file.write_text("[]")
    readme_file = tmp_path / "README.md"
    readme_file.write_text("## Section\nContent.\n")

    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = NotFoundError(
        "Collection [project_docs] does not exist"
    )
    mock_client.get_or_create_collection.return_value = MagicMock()
    mock_client_class.return_value = mock_client

    total = ingest(
        docs_json_path=docs_file,
        readme_path=readme_file,
        persist_path=tmp_path / "chroma_db",
    )

    assert total == 1
    mock_client.get_or_create_collection.assert_called_once()
