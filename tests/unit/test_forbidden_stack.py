"""Fail when src or app.py uses the stacks this lab does not ship."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_MODULES = frozenset({"ultralytics", "fastapi", "mlflow"})


def _project_python_files() -> list[Path]:
    files = sorted((REPO_ROOT / "src").rglob("*.py"))
    app = REPO_ROOT / "app.py"
    if app.is_file():
        files.append(app)
    return files


def _is_sv_bytetrack(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "ByteTrack"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sv"
    )


def _has_shell_true(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    for keyword in node.keywords:
        if keyword.arg != "shell":
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and value.value is True:
            return True
    return False


def _is_torch_compile_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "compile"
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def violations(source: str, filename: str = "<snippet>") -> list[str]:
    """Return forbidden uses. Comments and string literals are ignored."""
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_MODULES:
                    found.append(f"{filename}:{node.lineno}: import {root}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _FORBIDDEN_MODULES:
                found.append(f"{filename}:{node.lineno}: import {root}")
        elif _is_sv_bytetrack(node):
            found.append(f"{filename}:{node.lineno}: sv.ByteTrack")
        elif _has_shell_true(node):
            found.append(f"{filename}:{node.lineno}: shell=True")
        elif _is_torch_compile_call(node):
            found.append(f"{filename}:{node.lineno}: torch.compile")
    return found


def test_comments_and_strings_do_not_match() -> None:
    source = """\
# import ultralytics
# from fastapi import FastAPI
# import mlflow
# sv.ByteTrack()
# subprocess.run(cmd, shell=True)
# torch.compile(model)
note = '''
import ultralytics
from fastapi import FastAPI
import mlflow
sv.ByteTrack()
subprocess.run(cmd, shell=True)
torch.compile(model)
'''
value = "import mlflow; sv.ByteTrack(); shell=True; torch.compile(model)"
"""
    assert violations(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "import ultralytics\n",
        "import ultralytics.nn\n",
        "from fastapi import FastAPI\n",
        "from mlflow.tracking import MlflowClient\n",
        "import sv\nsv.ByteTrack()\n",
        "import subprocess\nsubprocess.run(['ffprobe'], shell=True)\n",
        "import torch\ntorch.compile(model)\n",
    ],
)
def test_real_forbidden_uses_are_detected(source: str) -> None:
    assert violations(source)


def test_src_and_app_have_no_forbidden_stack() -> None:
    problems: list[str] = []
    for path in _project_python_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT).as_posix()
        problems.extend(violations(text, relative))
    assert problems == []


def test_web_rejects_fastapi_and_streamlit() -> None:
    web = REPO_ROOT / "src" / "excavation_lab" / "web"
    assert web.is_dir()
    offenders: list[str] = []
    for path in sorted(web.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT).as_posix()
        offenders.extend(violations(text, relative))
        tree = ast.parse(text)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", 1)[0]]
            if "fastapi" in names or "streamlit" in names:
                offenders.append(f"{relative}: import")
    assert offenders == []
