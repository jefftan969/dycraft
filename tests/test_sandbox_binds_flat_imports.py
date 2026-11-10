"""Every repo module the Blender worker imports by its flat name is bound into the sandbox's `/tools`.

Inside the sandbox only `/tools` is on the path, so `worker.py`, `verify.py`, `chunking.py`, and the modules they pull
in import each other by bare name (`from collision import ...`). A module that is imported that way but missing from
`runtime.command`'s bind list fails only when Blender runs it. This finds every such import, including the ones
inside functions, and checks each against the binds, on the host.
"""
import ast
from pathlib import Path

import pytest

from dycraft.animation.blender import runtime

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")

REPO = Path(__file__).resolve().parents[1]
#: Where a flat-imported module can live: the directories `runtime.command` binds files from.
SOURCES = [REPO / "animation" / "blender", REPO / "verify", REPO / "render", REPO / "animation" / "mocap"]
ENTRY = ["worker", "verify", "chunking"]


def _module_file(name):
    for directory in SOURCES:
        path = directory / f"{name}.py"
        if path.is_file():
            return path
    return None


def _flat_imports(path):
    """Top-level names this file imports without a leading dot, at any depth of its code."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _reachable():
    """The repo modules the entry points reach through flat imports, transitively."""
    seen, todo = set(), list(ENTRY)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        path = _module_file(name)
        if path is None:
            continue
        seen.add(name)
        todo.extend(n for n in _flat_imports(path) if _module_file(n) is not None)
    return seen


def test_every_flat_imported_module_is_bound(tmp_path, monkeypatch):
    if runtime.blender_python() is None and runtime.blender_binary() is None:
        pytest.skip("needs a configured Blender")
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path / "home"))
    ws, out = tmp_path / "ws", tmp_path / "out"
    ws.mkdir(), out.mkdir()
    argv = runtime.command(ws, out, None, serve=True)
    bound = {Path(dest).stem for flag, src, dest in zip(argv, argv[1:], argv[2:])
             if flag == "--ro-bind" and dest.startswith("/tools/") and dest.endswith(".py")}
    reachable = _reachable()
    assert set(ENTRY) <= reachable
    assert sorted(reachable - bound) == []
