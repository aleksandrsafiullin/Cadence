"""The default suite must not launch pilot-small training.

``resolve_train_profile("pilot-small")`` only builds a config.
``tests/unit/test_train_excludes_test.py`` drives the CLI through CliRunner
and a trainer double, so a second job is rejected. That is not a launch.
A module with ``pytestmark = pytest.mark.manual`` is skipped. ``pyproject.toml``
already collects ``-m 'not manual'``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
_SUBPROCESS_ATTRS = frozenset({"run", "Popen", "call", "check_call", "check_output"})
_EXEMPT = _TESTS / "unit" / "test_train_excludes_test.py"


def launch_sites(source: str, filename: str = "<snippet>") -> list[str]:
    """Return launch sites. A manual module returns none."""
    tree = ast.parse(source, filename=filename)
    if _module_is_manual(tree):
        return []
    scanner = _Scanner(filename)
    scanner.visit(tree)
    return scanner.hits


def test_suite_does_not_launch_pilot_small() -> None:
    hits: list[str] = []
    for path in sorted(_TESTS.rglob("*.py")):
        hits.extend(launch_sites(path.read_text(encoding="utf-8"), str(path)))
    assert hits == []


def test_train_excludes_test_is_not_scanned_as_a_launch() -> None:
    """The file invokes the CLI. It does not shell out or call RFDETR.train."""
    source = _EXEMPT.read_text(encoding="utf-8")
    assert launch_sites(source, str(_EXEMPT)) == []
    assert "resolve_train_profile" in source
    assert "pilot-small" in source


def test_config_and_cli_runner_are_not_launches() -> None:
    source = """
from typer.testing import CliRunner

def test_it():
    resolve_train_profile("pilot-small")
    CliRunner().invoke(app, ["train", "--project", "pilot", "--profile", "pilot-small"])
"""
    assert launch_sites(source) == []


def test_shell_and_unmocked_train_are_launches() -> None:
    source = """
import os
import subprocess

def test_shell():
    command = ["uv", "run", "excavation", "train", "--profile", "pilot-small"]
    subprocess.run(command)
    subprocess.Popen(["excavation", "train", "--profile", "pilot-small"])
    os.system("excavation train --profile pilot-small")

def test_api():
    profile = "pilot-small"
    enqueue_train(workspace, "pilot", "ds", profile)
    RFDETR.train(profile="pilot-small")
    RFDETRNano().train(profile="pilot-small")
"""
    hits = launch_sites(source, "bad.py")
    assert len(hits) == 6
    assert all(item.startswith("bad.py:") for item in hits)


def test_mocked_train_call_is_not_a_launch() -> None:
    source = """
def test_it(monkeypatch):
    monkeypatch.setattr(train_mod, "enqueue_train", fake)
    train_mod.enqueue_train(workspace, "pilot", "ds", profile="pilot-small")
    monkeypatch.setattr("rfdetr.detr.RFDETR.train", fail)
    RFDETR.train(profile="pilot-small")
"""
    assert launch_sites(source) == []


def test_train_call_before_the_mock_is_a_launch() -> None:
    source = """
def test_it(monkeypatch):
    enqueue_train(workspace, "pilot", "ds", "pilot-small")
    monkeypatch.setattr(train_mod, "enqueue_train", fake)
"""
    assert len(launch_sites(source)) == 1


def test_manual_module_is_skipped() -> None:
    source = """
import pytest
import subprocess

pytestmark = pytest.mark.manual

def test_it():
    subprocess.run(["excavation", "train", "--profile", "pilot-small"])
"""
    assert launch_sites(source) == []


def test_ffprobe_subprocess_is_not_a_launch() -> None:
    source = """
import subprocess

def test_it(path):
    command = ["ffprobe", "-show_format", str(path)]
    subprocess.run(command, capture_output=True, check=False)
"""
    assert launch_sites(source) == []


class _Scanner(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bindings: list[dict[str, str]] = [{}]
        self.enqueue_mocked = [False]
        self.rfdetr_mocked = [False]
        self.hits: list[str] = []

    def lookup(self, name: str) -> str | None:
        for frame in reversed(self.bindings):
            if name in frame:
                return frame[name]
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bindings.append({})
        self.enqueue_mocked.append(self.enqueue_mocked[-1])
        self.rfdetr_mocked.append(self.rfdetr_mocked[-1])
        for stmt in node.body:
            self.visit(stmt)
        self.bindings.pop()
        self.enqueue_mocked.pop()
        self.rfdetr_mocked.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind(node.target, node.value)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        enqueue = False
        rfdetr = False
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Call):
                kind = _patch_kind(expr)
                enqueue = enqueue or kind == "enqueue"
                rfdetr = rfdetr or kind == "rfdetr"
        if not enqueue and not rfdetr:
            self.generic_visit(node)
            return
        self.enqueue_mocked.append(self.enqueue_mocked[-1] or enqueue)
        self.rfdetr_mocked.append(self.rfdetr_mocked[-1] or rfdetr)
        self.generic_visit(node)
        self.enqueue_mocked.pop()
        self.rfdetr_mocked.pop()

    def visit_Call(self, node: ast.Call) -> None:
        kind = _patch_kind(node)
        if kind == "enqueue":
            self.enqueue_mocked[-1] = True
        elif kind == "rfdetr":
            self.rfdetr_mocked[-1] = True
        elif _is_shell(node) and _command_has_pilot_small_train(node, self.lookup):
            self.hits.append(f"{self.filename}:{node.lineno}: shell")
        elif (
            _is_enqueue(node)
            and _profile_is(node, self.lookup, positional=3)
            and not self.enqueue_mocked[-1]
        ):
            self.hits.append(f"{self.filename}:{node.lineno}: enqueue_train")
        elif (
            _is_rfdetr_train(node)
            and _profile_is(node, self.lookup, positional=None)
            and not self.rfdetr_mocked[-1]
        ):
            self.hits.append(f"{self.filename}:{node.lineno}: RFDETR.train")
        self.generic_visit(node)

    def _enter_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        dec_enqueue, dec_rfdetr = _decorator_patches(node)
        self.bindings.append({})
        self.enqueue_mocked.append(self.enqueue_mocked[-1] or dec_enqueue)
        self.rfdetr_mocked.append(self.rfdetr_mocked[-1] or dec_rfdetr)
        for stmt in node.body:
            self.visit(stmt)
        self.bindings.pop()
        self.enqueue_mocked.pop()
        self.rfdetr_mocked.pop()

    def _bind(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        parts = _parts(value, self.lookup)
        if parts:
            self.bindings[-1][target.id] = " ".join(parts)


def _module_is_manual(tree: ast.Module) -> bool:
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            named = any(
                isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
            )
            if named:
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "pytestmark":
                value = node.value
        if value is not None and _mark_manual(value):
            return True
    return False


def _mark_manual(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "manual":
        return True
    if isinstance(node, ast.Call):
        return _mark_manual(node.func)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_mark_manual(elt) for elt in node.elts)
    return False


def _decorator_patches(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, bool]:
    enqueue = False
    rfdetr = False
    for dec in fn.decorator_list:
        text = _static_text(dec)
        if "enqueue_train" in text:
            enqueue = True
        if "RFDETR" in text and "train" in text:
            rfdetr = True
    return enqueue, rfdetr


def _patch_kind(call: ast.Call) -> str | None:
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    if name not in {"setattr", "patch"}:
        return None
    blob = " ".join(_static_text(arg) for arg in call.args)
    if "enqueue_train" in blob:
        return "enqueue"
    if "RFDETR" in blob and "train" in blob:
        return "rfdetr"
    return None


def _is_shell(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name) and func.id in {"Popen", "system"}:
        return True
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "system" and isinstance(func.value, ast.Name) and func.value.id == "os":
        return True
    return func.attr in _SUBPROCESS_ATTRS and _chain_has(func.value, "subprocess")


def _is_enqueue(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "enqueue_train"
    return isinstance(func, ast.Attribute) and func.attr == "enqueue_train"


def _is_rfdetr_train(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "train":
        return False
    return _chain_has_prefix(func.value, "RFDETR")


def _profile_is(
    call: ast.Call,
    lookup,
    *,
    positional: int | None,
) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "profile" and _string_arg(keyword.value, lookup) == "pilot-small":
            return True
    if positional is not None and len(call.args) > positional:
        return _string_arg(call.args[positional], lookup) == "pilot-small"
    return False


def _command_has_pilot_small_train(call: ast.Call, lookup) -> bool:
    node = call.args[0] if call.args else None
    for keyword in call.keywords:
        if keyword.arg in {"args", "command"}:
            node = keyword.value
    if node is None:
        return False
    blob = " ".join(_parts(node, lookup))
    return "train" in blob and "pilot-small" in blob


def _string_arg(node: ast.AST, lookup) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return lookup(node.id)
    return None


def _parts(node: ast.AST, lookup) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Name):
        found = lookup(node.id)
        return [found] if found else []
    if isinstance(node, (ast.List, ast.Tuple)):
        parts: list[str] = []
        for elt in node.elts:
            parts.extend(_parts(elt, lookup))
        return parts
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _parts(node.left, lookup) + _parts(node.right, lookup)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            parts.extend(_parts(value, lookup))
        return parts
    if isinstance(node, ast.FormattedValue):
        return _parts(node.value, lookup)
    return []


def _static_text(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_static_text(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        parts = [_static_text(node.func)]
        parts.extend(_static_text(arg) for arg in node.args)
        return " ".join(parts)
    if isinstance(node, (ast.List, ast.Tuple)):
        return " ".join(_static_text(elt) for elt in node.elts)
    return ""


def _chain_has(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.Attribute):
        return node.attr == name or _chain_has(node.value, name)
    return False


def _chain_has_prefix(node: ast.AST, prefix: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id.startswith(prefix)
    if isinstance(node, ast.Attribute):
        return node.attr.startswith(prefix) or _chain_has_prefix(node.value, prefix)
    if isinstance(node, ast.Call):
        return _chain_has_prefix(node.func, prefix)
    return False
