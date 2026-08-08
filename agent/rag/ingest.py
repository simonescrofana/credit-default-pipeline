"""Build the ChromaDB index the agent's rag route queries.

Two sources are embedded into a single persistent collection:

- Every entry in `agent/data/docs_data.json` (produced by `extract_docs.py`)
  — a module/class/function docstring or a dbt model/column description.
  Each is already a self-contained unit of text, so no further chunking is
  applied.
- The project's README, chunked by Markdown header rather than treated as
  a single document: `##` sections split it into large pieces (Roadmap,
  Results, Tech Stack, ...), and where a section is itself broken into
  `####` sub-sections (the Architectural Decisions & Rationale bullets),
  those become the chunk instead, since each is already a self-contained
  Choice/Justification unit — splitting there gives a finer, still
  coherent chunk than the whole Rationale section at once.

Uses ChromaDB's default local embedding function (all-MiniLM-L6-v2, via
sentence-transformers), consistent with the project's CPU-only hardware:
no API key, no network round-trip per chunk, and it runs once here at
ingestion time rather than per request.

"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import chromadb
from chromadb.errors import NotFoundError

from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_JSON_PATH = PROJECT_ROOT / "agent" / "data" / "docs_data.json"
README_PATH = PROJECT_ROOT / "README.md"
CHROMA_PERSIST_PATH = PROJECT_ROOT / "agent" / "data" / "chroma_db"

COLLECTION_NAME = "project_docs"
BATCH_SIZE = 100

# matches a Markdown header line, capturing its level (number of '#') and title
HEADER_PATTERN = re.compile(r"^(#{2,4})\s+(.*)$", re.MULTILINE)


def iter_docs_json(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each document from docs_data.json one at a time.

    Args:
        path (Path): Path to the JSON file produced by `extract_docs.py`.

    Yields:
        dict[str, Any]: One document (source, kind, name, file, text) at
            a time, so the caller never needs the full list in memory at
            once beyond what `json.load` itself requires to parse it.

    """
    documents = json.loads(path.read_text(encoding="utf-8"))
    yield from documents


def chunk_readme(path: Path) -> Iterator[dict[str, Any]]:
    """Split the README into chunks along its Markdown header structure.

    A `##` section becomes its own chunk unless it contains `####`
    sub-sections, in which case each sub-section becomes a chunk instead
    (finer-grained, and each is already a self-contained unit in this
    project's README style).

    Args:
        path (Path): Path to the README file.

    Yields:
        dict[str, Any]: One chunk at a time, with `source` ("readme"),
            `kind` ("section"), `name` (the header title), `file` (the
            README's path relative to the project root), and `text` (the
            header plus its body, up to the next header of the same or
            higher level).

    """
    content = path.read_text(encoding="utf-8")
    relative_path = path.name

    matches = list(HEADER_PATTERN.finditer(content))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # a section's body runs until the next header of the same or a
        # shallower level (e.g. a '##' section stops at the next '##',
        # but not at a '####' nested inside it)
        end = len(content)
        for next_match in matches[i + 1 :]:
            if len(next_match.group(1)) <= level:
                end = next_match.start()
                break

        # if a '##' section contains '####' sub-sections, don't yield the
        # whole span as one chunk (the sub-sections will each be yielded on
        # their own turn), but still yield any intro text between the '##'
        # header and its first '####' child, so it isn't silently dropped
        if level == 2:
            first_child = next(
                (
                    m
                    for m in matches[i + 1 :]
                    if m.start() < end and len(m.group(1)) > level
                ),
                None,
            )
            if first_child is not None:
                intro_text = content[match.start() : first_child.start()].strip()
                if intro_text != match.group(0).strip():  # skips spaces-only lines
                    yield {
                        "source": "readme",
                        "kind": "section",
                        "name": title,
                        "file": relative_path,
                        "text": intro_text,
                    }
                continue

        text = content[match.start() : end].strip()

        yield {
            "source": "readme",
            "kind": "section",
            "name": title,
            "file": relative_path,
            "text": text,
        }


def _make_id(document: dict[str, Any], index: int) -> str:
    """Build a stable, unique id for a document to use as its ChromaDB id."""
    return f"{document['source']}:{document.get('file', '')}:{document['name']}:{index}"


def ingest(
    docs_json_path: Path = DOCS_JSON_PATH,
    readme_path: Path = README_PATH,
    persist_path: Path = CHROMA_PERSIST_PATH,
) -> int:
    """Embed and load every document and README chunk into ChromaDB.

    Args:
        docs_json_path (Path, optional): Path to docs_data.json. Defaults
            to the project's actual path.
        readme_path (Path, optional): Path to the README. Defaults to the
            project's actual path.
        persist_path (Path, optional): Directory where the ChromaDB index
            is persisted. Defaults to the project's actual path.

    Returns:
        int: The total number of chunks added to the collection.

    """
    client = chromadb.PersistentClient(path=str(persist_path))

    # start from a clean collection on every run: this makes ingestion
    # idempotent and ensures a chunk removed from a source (e.g. a deleted
    # README section or docstring) doesn't linger in the index forever
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except NotFoundError:
        pass  # first run: the collection doesn't exist yet, nothing to clear

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = list(iter_docs_json(docs_json_path)) + list(chunk_readme(readme_path))

    total = 0
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]

        collection.upsert(
            ids=[_make_id(doc, batch_start + i) for i, doc in enumerate(batch)],
            documents=[doc["text"] for doc in batch],
            metadatas=[
                {
                    "source": doc["source"],
                    "kind": doc["kind"],
                    "name": doc["name"],
                    "file": doc.get("file", ""),
                }
                for doc in batch
            ],
        )
        total += len(batch)

    logger.info(
        "Ingested %d chunk(s) into collection '%s' at %s.",
        total,
        COLLECTION_NAME,
        persist_path,
    )
    return total


if __name__ == "__main__":
    setup_logging("INFO")
    ingest()
