"""Test suite for the documentation extraction script.

Covers `_is_excluded` (a clean path passes, every excluded category is
caught), `_extract_python_docs` (module/class/function docstrings, a file
that fails to parse is skipped rather than raised, excluded directories
are never walked into), `_extract_dbt_docs` (model/column descriptions,
excluded directories and non-file schema.yml paths are skipped), and
`extract_docs` combining both sources. Every test builds a throwaway file
tree under pytest's `tmp_path`, never touching the real project tree.

"""

import json
from unittest.mock import patch

from agent.rag.extract_docs import (
    _extract_dbt_docs,
    _extract_python_docs,
    _is_excluded,
    extract_docs,
    main,
)


def test_is_excluded_clean_path_is_not_excluded(tmp_path) -> None:
    """Verify a path with no excluded segment returns False."""
    path = tmp_path / "agent" / "nodes" / "router.py"
    assert _is_excluded(path) is False


def test_is_excluded_catches_dot_directories(tmp_path) -> None:
    """Verify any dot-prefixed directory (git, venv, caches, ...) is excluded."""
    for dot_dir in [".git", ".venv", ".pytest_cache", ".ruff_cache", ".github"]:
        path = tmp_path / dot_dir / "some_file.py"
        assert _is_excluded(path) is True, f"{dot_dir} should be excluded"


def test_is_excluded_catches_named_directories(tmp_path) -> None:
    """Verify every explicitly named excluded directory is caught."""
    for named_dir in ["tests", "__pycache__", "mlruns", "logs", "target"]:
        path = tmp_path / "some" / named_dir / "file.py"
        assert _is_excluded(path) is True, f"{named_dir} should be excluded"


def test_extract_python_docs_happy_path(tmp_path) -> None:
    """Verify module, class, and function docstrings are all extracted."""
    module_file = tmp_path / "sample.py"
    module_file.write_text(
        '"""Module-level docstring."""\n'
        "\n"
        "class Sample:\n"
        '    """Class-level docstring."""\n'
        "\n"
        "    def method(self):\n"
        '        """Method-level docstring."""\n'
        "        pass\n"
        "\n"
        "def top_level_func():\n"
        '    """Function-level docstring."""\n'
        "    pass\n"
    )

    docs = _extract_python_docs(tmp_path)

    kinds = {(d["kind"], d["name"]) for d in docs}
    assert ("module", "sample") in kinds
    assert ("class", "sample.Sample") in kinds
    assert ("function", "sample.method") in kinds
    assert ("function", "sample.top_level_func") in kinds
    assert all(d["source"] == "python" for d in docs)


def test_extract_python_docs_skips_files_with_no_docstrings(tmp_path) -> None:
    """Verify a file with no docstrings at all produces no documents."""
    (tmp_path / "empty.py").write_text("x = 1\n")

    docs = _extract_python_docs(tmp_path)

    assert docs == []


def test_extract_python_docs_skips_unparseable_file(tmp_path) -> None:
    """Verify a file with a syntax error is skipped, not raised."""
    (tmp_path / "broken.py").write_text("def broken(:\n    this is not valid python\n")
    (tmp_path / "valid.py").write_text('"""A valid module."""\n')

    docs = _extract_python_docs(tmp_path)

    assert len(docs) == 1
    assert docs[0]["name"] == "valid"


def test_extract_python_docs_skips_undocumented_class_and_function(tmp_path) -> None:
    """Verify a class/function with no docstring produces no document for it."""
    module_file = tmp_path / "sample.py"
    module_file.write_text(
        '"""Module-level docstring."""\n'
        "\n"
        "class DocumentedClass:\n"
        '    """Has a docstring."""\n'
        "    pass\n"
        "\n"
        "class UndocumentedClass:\n"
        "    pass\n"
        "\n"
        "def documented_func():\n"
        '    """Has a docstring."""\n'
        "    pass\n"
        "\n"
        "def undocumented_func():\n"
        "    pass\n"
    )

    docs = _extract_python_docs(tmp_path)

    names = {d["name"] for d in docs}
    assert "sample.DocumentedClass" in names
    assert "sample.documented_func" in names
    assert "sample.UndocumentedClass" not in names
    assert "sample.undocumented_func" not in names


def test_extract_python_docs_respects_exclusions(tmp_path) -> None:
    """Verify files under an excluded directory are never included."""
    excluded_dir = tmp_path / "tests"
    excluded_dir.mkdir()
    (excluded_dir / "test_something.py").write_text('"""A test module docstring."""\n')

    (tmp_path / "real_module.py").write_text('"""A real module docstring."""\n')

    docs = _extract_python_docs(tmp_path)

    assert len(docs) == 1
    assert docs[0]["name"] == "real_module"


def test_extract_dbt_docs_happy_path(tmp_path) -> None:
    """Verify model and column descriptions are extracted from schema.yml."""
    schema_file = tmp_path / "schema.yml"
    schema_file.write_text(
        "version: 2\n"
        "models:\n"
        "  - name: dim_companies\n"
        '    description: "SCD2 company dimension."\n'
        "    columns:\n"
        "      - name: company_id\n"
        '        description: "Business key."\n'
    )

    docs = _extract_dbt_docs(tmp_path)

    kinds = {(d["kind"], d["name"], d["text"]) for d in docs}
    assert ("model", "dim_companies", "SCD2 company dimension.") in kinds
    assert ("column", "dim_companies.company_id", "Business key.") in kinds
    assert all(d["source"] == "dbt" for d in docs)


def test_extract_dbt_docs_skips_entries_without_description(tmp_path) -> None:
    """Verify a model/column with no description produces no document."""
    schema_file = tmp_path / "schema.yml"
    schema_file.write_text(
        "version: 2\n"
        "models:\n"
        "  - name: undocumented_model\n"
        "    columns:\n"
        "      - name: some_column\n"
    )

    docs = _extract_dbt_docs(tmp_path)

    assert docs == []


def test_extract_dbt_docs_respects_exclusions(tmp_path) -> None:
    """Verify schema.yml files under an excluded directory are skipped."""
    excluded_dir = tmp_path / "target" / "compiled"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "schema.yml").write_text(
        'models:\n  - name: compiled_model\n    description: "Should be skipped."\n'
    )

    (tmp_path / "schema.yml").write_text(
        'models:\n  - name: real_model\n    description: "Should be kept."\n'
    )

    docs = _extract_dbt_docs(tmp_path)

    assert len(docs) == 1
    assert docs[0]["name"] == "real_model"


def test_extract_dbt_docs_skips_directory_named_schema_yml(tmp_path) -> None:
    """Test an schema.yml that is a directory (dbt build artifact) is skipped."""
    fake_schema_dir = tmp_path / "schema.yml"
    fake_schema_dir.mkdir()

    (tmp_path / "real" / "schema.yml").parent.mkdir(parents=True)
    (tmp_path / "real" / "schema.yml").write_text(
        'models:\n  - name: real_model\n    description: "A real model."\n'
    )

    docs = _extract_dbt_docs(tmp_path)

    assert len(docs) == 1
    assert docs[0]["name"] == "real_model"


def test_extract_docs_combines_both_sources(tmp_path) -> None:
    """Verify extract_docs returns the union of Python and dbt documents."""
    (tmp_path / "sample.py").write_text('"""A module."""\n')
    (tmp_path / "schema.yml").write_text(
        'models:\n  - name: a_model\n    description: "A model."\n'
    )

    docs = extract_docs(tmp_path)

    sources = {d["source"] for d in docs}
    assert sources == {"python", "dbt"}
    assert len(docs) == 2


@patch("agent.rag.extract_docs.extract_docs")
def test_main_writes_extracted_docs_to_output_path(mock_extract_docs, tmp_path) -> None:
    """Verify main() writes extract_docs' result to OUTPUT_PATH as JSON."""
    fake_docs = [
        {
            "source": "python",
            "kind": "module",
            "name": "sample",
            "file": "sample.py",
            "text": "A module docstring.",
        }
    ]
    mock_extract_docs.return_value = fake_docs

    output_path = tmp_path / "agent" / "data" / "docs_data.json"
    with patch("agent.rag.extract_docs.OUTPUT_PATH", output_path):
        main()

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == fake_docs
