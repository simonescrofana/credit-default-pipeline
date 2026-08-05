"""Extract project documentation into a single indexable JSON file.

Walks the project tree and collects two kinds of documentation, without
ever importing or executing any project code:

- Every Python docstring (module, class, and function level), read via the
  `ast` module rather than `inspect`, so a file can be read even if
  importing it would trigger side effects (e.g. a database connection
  opened at module level) or require heavy dependencies not needed just
  to read a docstring.
- Every `description` field in the dbt project's `schema.yml` files, at
  both the model and column level.

The result is written to `agent/data/docs_data.json`, re-run only when the
underlying documentation changes (e.g. new modules, or the project README),
rather than on every agent startup. `ingest.py` reads this file to build
the ChromaDB index.

"""

import ast
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "agent" / "data" / "docs_data.json"

EXCLUDED_DIR_NAMES = {
    "tests",
    "__pycache__",
    "mlruns",
    "logs",
    "target",  # dbt build artifacts, including compiled schema.yml-named dirs
}


def _is_excluded(path: Path) -> bool:
    """Check whether any parent directory of path is hidden or excluded.

    A directory is excluded if its name is in EXCLUDED_DIR_NAMES, or if it
    starts with a dot — covering version control, CI, and tool caches
    (.git, .github, .dvc, .venv, .pytest_cache, .ruff_cache, and any future
    one) without needing to name each one individually.

    """
    return any(
        part in EXCLUDED_DIR_NAMES or part.startswith(".") for part in path.parts
    )


def _extract_python_docs(root: Path) -> list[dict[str, Any]]:
    """Extract module, class, and function docstrings from every .py file.

    Args:
        root (Path): The project root to walk.

    Returns:
        list[dict[str, Any]]: One entry per non-empty docstring found, each
            with `source` ("python"), `kind` ("module", "class", or
            "function"), `name` (qualified name, e.g. "router.router_node"),
            `file` (path relative to root), and `text` (the docstring).

    """
    documents: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*.py")):
        if _is_excluded(path):
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            logger.warning("Skipping %s: could not be parsed.", path)
            continue

        relative_path = path.relative_to(root).as_posix()
        module_name = relative_path.removesuffix(".py").replace("/", ".")

        module_doc = ast.get_docstring(tree)
        if module_doc:
            documents.append(
                {
                    "source": "python",
                    "kind": "module",
                    "name": module_name,
                    "file": relative_path,
                    "text": module_doc,
                }
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if not doc:
                    continue
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                documents.append(
                    {
                        "source": "python",
                        "kind": kind,
                        "name": f"{module_name}.{node.name}",
                        "file": relative_path,
                        "text": doc,
                    }
                )

    logger.info("Extracted %d Python docstring(s).", len(documents))
    return documents


def _extract_dbt_docs(root: Path) -> list[dict[str, Any]]:
    """Extract model and column descriptions from every dbt schema.yml file.

    Args:
        root (Path): The project root to walk.

    Returns:
        list[dict[str, Any]]: One entry per non-empty description found,
            each with `source` ("dbt"), `kind` ("model" or "column"),
            `name` (e.g. "fct_company_credit_profile" or
            "fct_company_credit_profile.unpaid_ratio_trailing_90d"), `file`
            (path relative to root), and `text` (the description).

    """
    documents: list[dict[str, Any]] = []

    for path in sorted(root.rglob("schema.yml")):
        if _is_excluded(path) or not path.is_file():
            continue

        relative_path = path.relative_to(root).as_posix()
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        for model in content.get("models", []):
            model_name = model.get("name")
            model_description = model.get("description")
            if model_description:
                documents.append(
                    {
                        "source": "dbt",
                        "kind": "model",
                        "name": model_name,
                        "file": relative_path,
                        "text": model_description,
                    }
                )

            for column in model.get("columns", []):
                column_description = column.get("description")
                if column_description:
                    documents.append(
                        {
                            "source": "dbt",
                            "kind": "column",
                            "name": f"{model_name}.{column.get('name')}",
                            "file": relative_path,
                            "text": column_description,
                        }
                    )

    logger.info("Extracted %d dbt description(s).", len(documents))
    return documents


def extract_docs(root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Extract every documentation source and return the combined list.

    Args:
        root (Path, optional): The project root to walk. Defaults to the
            actual project root.

    Returns:
        list[dict[str, Any]]: The combined Python docstring and dbt
            description documents.

    """
    return _extract_python_docs(root) + _extract_dbt_docs(root)


def main() -> None:
    """Extract every documentation source and write it to OUTPUT_PATH."""
    setup_logging("INFO")
    documents = extract_docs()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(documents, indent=2, ensure_ascii=False))

    logger.info("Wrote %d document(s) to %s.", len(documents), OUTPUT_PATH)


if __name__ == "__main__":
    main()
