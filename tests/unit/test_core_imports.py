"""Core modules must not import the page server, Jinja2, or Streamlit."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_LAB = REPO_ROOT / "src" / "excavation_lab"
_PACKAGE_DIRS = ("media", "datasets", "models", "tracking", "analytics")
_MODULE_FILES = ("jobs.py", "storage.py", "hardware.py", "schemas.py")
_FORBIDDEN = frozenset({"starlette", "jinja2", "streamlit"})


def _core_python_files() -> list[Path]:
    files: list[Path] = []
    for name in _PACKAGE_DIRS:
        root = _LAB / name
        if not root.is_dir():
            raise AssertionError(f"missing core package: {root}")
        files.extend(sorted(root.rglob("*.py")))
    for name in _MODULE_FILES:
        path = _LAB / name
        if not path.is_file():
            raise AssertionError(f"missing core module: {path}")
        files.append(path)
    return files


def _forbidden_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN:
                    found.append(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _FORBIDDEN:
                found.append(root)
    return found


def test_comments_and_strings_are_not_web_imports() -> None:
    source = """\
# import starlette
# from jinja2 import Template
# from streamlit import sidebar
note = "import starlette"
value = 1
"""
    assert _forbidden_imports(source) == []


def test_web_import_forms_are_detected() -> None:
    assert _forbidden_imports("import starlette\n") == ["starlette"]
    assert _forbidden_imports("import jinja2 as templates\n") == ["jinja2"]
    assert _forbidden_imports("import streamlit as st\n") == ["streamlit"]
    assert _forbidden_imports("from starlette.routing import Route\n") == ["starlette"]
    assert _forbidden_imports("from jinja2 import Template\n") == ["jinja2"]
    assert _forbidden_imports("from streamlit import cache_data\n") == ["streamlit"]


def test_core_modules_do_not_import_web_packages() -> None:
    offenders: list[str] = []
    for path in _core_python_files():
        text = path.read_text(encoding="utf-8")
        hits = _forbidden_imports(text)
        if hits:
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel}: {', '.join(hits)}")
    assert offenders == []
