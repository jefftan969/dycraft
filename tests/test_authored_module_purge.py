"""A resident Blender re-reads an edited helper, however it is imported."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]

SPELLINGS = ("import helper as h",
             "import animation_src.helper as h",
             "from animation_src import helper as h")


def _load_worker(monkeypatch, workspace):
    """The real worker module with `bpy` stubbed, as `test_physics_bake_audit`."""
    monkeypatch.setenv("DYCRAFT_BLENDER_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils",
                        SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    monkeypatch.setitem(sys.modules, "_blender_child", SimpleNamespace())
    spec = importlib.util.spec_from_file_location(
        "purge_worker", REPO / "animation/blender/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_an_edited_helper_is_re_read_on_the_next_call(tmp_path, monkeypatch, spelling):
    """`animation_src` has no `__init__.py`, so it imports as a namespace
    package with no `__file__`, and the purge still covers it: otherwise the
    package keeps its `helper` attribute and `from animation_src import helper`
    returns the module from before the edit, with no error."""
    worker = _load_worker(monkeypatch, tmp_path)
    (tmp_path / "animation_src").mkdir()
    helpers = (tmp_path / "helper.py", tmp_path / "animation_src/helper.py")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    names = ("helper", "animation_src", "animation_src.helper")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        for path in helpers:
            path.write_text("N = 24\n")
        found = {}
        exec(spelling, found)
        assert found["h"].N == 24
        for path in helpers:
            path.write_text("N = 26\n")
            os.utime(path, (1e9 + 10, 1e9 + 10))
        worker.purge_authored_modules([tmp_path])
        importlib.invalidate_caches()
        exec(spelling, found)
        assert found["h"].N == 26
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_a_namespace_package_outside_the_authored_roots_is_kept(tmp_path, monkeypatch):
    worker = _load_worker(monkeypatch, tmp_path / "workspace")
    shared = SimpleNamespace(__path__=[str(tmp_path / "site/shared")])
    mixed = SimpleNamespace(__path__=[str(tmp_path / "workspace/mixed"),
                                      str(tmp_path / "site/mixed")])
    monkeypatch.setitem(sys.modules, "_purge_shared", shared)
    monkeypatch.setitem(sys.modules, "_purge_mixed", mixed)
    worker.purge_authored_modules([tmp_path / "workspace"])
    assert sys.modules["_purge_shared"] is shared and sys.modules["_purge_mixed"] is mixed
