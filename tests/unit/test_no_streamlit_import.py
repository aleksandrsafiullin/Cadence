"""No file under ``src`` imports Streamlit."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_python_files() -> list[Path]:
    return sorted((REPO_ROOT / "src").rglob("*.py"))


def _imports_streamlit(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "streamlit" or alias.name.startswith("streamlit."):
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module == "streamlit" or module.startswith("streamlit."):
                return True
    return False


def test_comments_and_strings_are_not_streamlit_imports() -> None:
    source = """\
# import streamlit
# from streamlit import sidebar
note = "import streamlit"
value = 1
"""
    assert _imports_streamlit(source) is False


def test_streamlit_import_forms_are_detected() -> None:
    assert _imports_streamlit("import streamlit\n")
    assert _imports_streamlit("import streamlit as st\n")
    assert _imports_streamlit("from streamlit import cache_data\n")
    assert _imports_streamlit("from streamlit.runtime import get_instance\n")


def test_src_does_not_import_streamlit() -> None:
    offenders: list[str] = []
    for path in _project_python_files():
        text = path.read_text(encoding="utf-8")
        if _imports_streamlit(text):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []
