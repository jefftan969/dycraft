"""A sandboxed interpreter starts with its own packages, whatever layout holds them.

`runtime.command` asks the interpreter what it needs bound
(`_interpreters.sandbox_view`). Its fallback, binding the resolved
interpreter's grandparent, is right for a conda env and wrong for every other
common layout: a venv's `bin/python` resolves to its *base* interpreter and
starts without the venv; a `--copies` venv cannot find its standard library at
all; a user-site or PYTHONPATH install is outside every bind.

Each case builds a throwaway, stdlib-only layout holding a marker module, runs
the real `runtime.command` argv with only its final program swapped, and
imports the marker inside the sandbox -- the same process shape the collision
child and the `bpy` worker run in. Each also checks the fallback fails, so the
case can tell a layout the view is needed for from one that never needed it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft import _interpreters as I
from dycraft.animation.blender import runtime

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")

MARKER = "dycraft_layout_marker"
SYSTEM_PYTHON = next((p for p in ("/usr/bin/python3",) if Path(p).is_file()), None)


def _run(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, **kw)


def _plant(site: Path, label: str) -> None:
    site.mkdir(parents=True, exist_ok=True)
    (site / f"{MARKER}.py").write_text(f"WHERE = {label!r}\n")


def _purelib(python: str, env=None) -> Path:
    out = _run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
               env=env)
    return Path(out.stdout.strip())


def _venv(root: Path, base: str, *flags: str) -> str:
    made = _run([base, "-m", "venv", "--without-pip", *flags, str(root)])
    if made.returncode:
        pytest.skip(f"{base} cannot make a venv: {made.stderr.strip()[-200:]}")
    return str(root / "bin" / "python")


def _layout(name: str, tmp: Path, monkeypatch) -> str:
    """Build layout `name` under `tmp`; return the interpreter that sees the marker."""
    if name == "venv":
        py = _venv(tmp / "venv", sys.executable)
    elif name == "venv-copies":
        py = _venv(tmp / "venv", sys.executable, "--copies")
    elif name == "uv-venv":
        uv = shutil.which("uv")
        if not uv:
            pytest.skip("uv is not installed")
        made = _run([uv, "venv", "-q", "-p", sys.executable, str(tmp / "venv")])
        if made.returncode:
            pytest.skip(f"uv venv failed: {made.stderr.strip()[-200:]}")
        py = str(tmp / "venv" / "bin" / "python")
    elif name == "uv-alias-venv":
        # uv's own layout: a venv whose `home` names a uv-managed base through its minor-version alias
        # (`cpython-3.13-…` → `cpython-3.13.18-…`), so Python looks for the standard library under the alias. A
        # uv-managed build cannot fall back to the prefix it was built for (`/install`), as conda's relocated one can.
        uv = shutil.which("uv")
        found = _run([uv, "python", "dir"]) if uv else None
        root = Path(found.stdout.strip()) if found and found.returncode == 0 else None
        real = sorted((p for p in root.glob("cpython-3.*") if p.is_dir() and not p.is_symlink()
                       and list(p.glob("bin/python3.*"))), key=str) if root and root.is_dir() else []
        if not real:
            pytest.skip("no uv-managed CPython installed")
        tmp.mkdir(parents=True, exist_ok=True)
        alias = tmp / "cpython-alias"
        alias.symlink_to(real[-1], target_is_directory=True)
        base = next(p for p in sorted((alias / "bin").glob("python3.*")) if not p.name.endswith("-config"))
        py = _venv(tmp / "venv", str(base))
        if f"home = {alias / 'bin'}" not in (tmp / "venv" / "pyvenv.cfg").read_text():
            pytest.skip("venv did not record the alias as home")
    elif name == "system-venv":
        if not SYSTEM_PYTHON:
            pytest.skip("no /usr/bin/python3")
        py = _venv(tmp / "venv", SYSTEM_PYTHON)
    elif name == "user-site":
        if not SYSTEM_PYTHON:
            pytest.skip("no /usr/bin/python3")
        monkeypatch.setenv("PYTHONUSERBASE", str(tmp / "userbase"))
        site = _run([SYSTEM_PYTHON, "-c", "import site; print(site.getusersitepackages())"])
        _plant(Path(site.stdout.strip()), name)
        return SYSTEM_PYTHON
    elif name == "pythonpath":
        monkeypatch.setenv("PYTHONPATH", str(tmp / "tree"))
        _plant(tmp / "tree", name)
        return sys.executable
    else:
        raise AssertionError(name)
    _plant(_purelib(py), name)
    return py


LAYOUTS = ("venv", "venv-copies", "uv-venv", "uv-alias-venv", "system-venv", "user-site", "pythonpath")


@pytest.fixture
def dirs(tmp_path):
    for d in ("ws", "out"):
        (tmp_path / d).mkdir()
    (tmp_path / "job.json").write_text("{}")
    return tmp_path


def _spliced(cmd: list[str], program: list[str]) -> list[str]:
    """`cmd` with everything from its final `--chdir` replaced by `program`."""
    cut = len(cmd) - cmd[::-1].index("--chdir") - 1
    return [*cmd[:cut], "--chdir", "/", *program]


def _require_sandbox(cmd):
    probe = _run(_spliced(cmd, ["/usr/bin/true"]), env=runtime.worker_env())
    if probe.returncode:
        pytest.skip(f"no usable bubblewrap sandbox here: {probe.stderr.strip()[-200:]}")


def _fallback(monkeypatch, dirs):
    """The argv `command` builds when it cannot ask the interpreter.

    `_python_view` falls back to that recipe when `sandbox_view` cannot answer,
    so forcing the failure reproduces it faithfully.
    """
    def unanswerable(*_a, **_k):
        raise RuntimeError("forced for the fallback comparison")
    monkeypatch.setattr(runtime, "sandbox_view", unanswerable)
    monkeypatch.setattr(runtime, "_VIEW_WARNED", set())
    try:
        return runtime.command(dirs / "ws", dirs / "out", dirs / "job.json")
    finally:
        monkeypatch.setattr(runtime, "sandbox_view", I.sandbox_view)


def _setenv(cmd, name):
    pairs = {cmd[i + 1]: cmd[i + 2] for i, a in enumerate(cmd) if a == "--setenv"}
    return pairs.get(name)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_collision_interpreter_keeps_its_packages(layout, dirs, monkeypatch):
    python = _layout(layout, dirs / "layout", monkeypatch)
    monkeypatch.setenv("DYCRAFT_COAL_PYTHON", python)
    monkeypatch.setattr(runtime, "COAL_IMPORTS", ("hashlib", MARKER))
    monkeypatch.setattr(runtime, "blender_binary", lambda: "/usr/bin/true")
    cmd = runtime.command(dirs / "ws", dirs / "out", dirs / "job.json")
    _require_sandbox(cmd)

    given = _setenv(cmd, "DYCRAFT_COAL_PYTHON")
    # What `CoalClient` runs: `-I`, then `coal_worker.serve` re-adds the roots
    # the isolated start cannot see.
    child = ("import sys; sys.path.insert(0, '/tools'); import coal_worker; "
             f"coal_worker.extend_path(); import {MARKER}; print({MARKER}.WHERE)")
    env = runtime.worker_env()
    ran = _run(_spliced(cmd, [given, "-I", "-B", "-c", child]), env=env)
    assert ran.returncode == 0, ran.stderr[-800:]
    assert ran.stdout.strip() == layout

    # Not vacuous: the fallback fails on this layout.
    old = _fallback(monkeypatch, dirs)
    ran = _run(_spliced(old, [_setenv(old, "DYCRAFT_COAL_PYTHON"), "-I", "-B", "-c", child]),
               env=env)
    assert ran.returncode != 0, f"{layout} also works with the fallback"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_bpy_interpreter_keeps_its_packages(layout, dirs, monkeypatch):
    python = _layout(layout, dirs / "layout", monkeypatch)
    monkeypatch.setattr(runtime, "blender_binary", lambda: None)
    monkeypatch.setattr(runtime, "blender_python", lambda: python)
    monkeypatch.setattr(runtime, "BPY_IMPORTS", (MARKER,))
    # Keep the collision interpreter's binds out of the comparison: the layouts
    # are built from this interpreter, so its env root would otherwise hand a
    # `--copies` venv the base standard library it is missing.
    monkeypatch.setattr(runtime, "COAL_IMPORTS", ("hashlib",))
    if SYSTEM_PYTHON:
        monkeypatch.setenv("DYCRAFT_COAL_PYTHON", SYSTEM_PYTHON)
    elif layout == "venv-copies":
        pytest.skip("needs a collision interpreter outside this env")
    cmd = runtime.command(dirs / "ws", dirs / "out", dirs / "job.json")
    assert cmd[-2:] == ["-B", "/tools/worker.py"]
    launch = cmd[-3]
    _require_sandbox(cmd)
    child = f"import {MARKER}; print({MARKER}.WHERE)"
    ran = _run(_spliced(cmd, [launch, "-B", "-c", child]), env=runtime.worker_env())
    assert ran.returncode == 0, ran.stderr[-800:]
    assert ran.stdout.strip() == layout

    old = _fallback(monkeypatch, dirs)
    ran = _run(_spliced(old, [old[-3], "-B", "-c", child]), env=runtime.worker_env())
    assert ran.returncode != 0, f"{layout} also works with the fallback"


#: The shape of the `bpy` wheel, which is what broke the first version of the
#: view: its top-level package inserts an inner directory into `sys.path` and
#: re-imports itself from there, and the inner package only works when the
#: outer loader has registered its compiled core first (`_bpy`).
_RELOCATING_OUTER = '''
import os, sys
sys.modules["_relocating_core"] = type(sys)("_relocating_core")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inner"))
del sys.modules[__name__]
import dycraft_relocating
'''
_RELOCATING_INNER = "import _relocating_core\nWHERE = 'relocated'\n"


def test_a_module_that_rewrites_sys_path_is_not_handed_its_inner_root(dirs, monkeypatch):
    """The root a module adds to `sys.path` while loading is not an import root.

    Read *after* the import, the bpy wheel's `scripts/modules` would look like
    a missing root and go onto the sandbox's PYTHONPATH ahead of site-packages,
    and the inner package would then load without its core: `No module named
    '_bpy'`.
    """
    python = _venv(dirs / "venv", sys.executable)
    site = _purelib(python)
    (site / "dycraft_relocating" / "inner" / "dycraft_relocating").mkdir(parents=True)
    (site / "dycraft_relocating" / "__init__.py").write_text(_RELOCATING_OUTER)
    (site / "dycraft_relocating" / "inner" / "dycraft_relocating" / "__init__.py").write_text(
        _RELOCATING_INNER)
    assert I.sandbox_view(python, ("dycraft_relocating",)).sitedirs == ()

    monkeypatch.setattr(runtime, "blender_binary", lambda: None)
    monkeypatch.setattr(runtime, "blender_python", lambda: python)
    monkeypatch.setattr(runtime, "BPY_IMPORTS", ("dycraft_relocating",))
    cmd = runtime.command(dirs / "ws", dirs / "out", dirs / "job.json")
    _require_sandbox(cmd)
    child = "import dycraft_relocating; print(dycraft_relocating.WHERE)"
    ran = _run(_spliced(cmd, [cmd[-3], "-B", "-c", child]), env=runtime.worker_env())
    assert ran.returncode == 0, ran.stderr[-800:]
    assert ran.stdout.strip() == "relocated"


def test_a_conda_env_binds_only_its_root(dirs, monkeypatch):
    """A conda env needs one bind, the env root, and nothing else."""
    monkeypatch.delenv("DYCRAFT_COAL_PYTHON", raising=False)
    view = I.sandbox_view(sys.executable, runtime.COAL_IMPORTS)
    if not (Path(sys.prefix) / "conda-meta").is_dir():
        pytest.skip("the test interpreter is not a conda env")
    root = str(Path(sys.executable).resolve().parents[1])
    assert [dest for _, dest in view.binds] == [root]
    assert view.sitedirs == ()


def test_an_interpreter_that_cannot_import_falls_back_and_says_so(dirs, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "COAL_IMPORTS", ("dycraft_no_such_module",))
    monkeypatch.setattr(runtime, "_VIEW_WARNED", set())
    monkeypatch.setattr(runtime, "blender_binary", lambda: "/usr/bin/true")
    monkeypatch.setenv("DYCRAFT_COAL_PYTHON", sys.executable)
    for _ in range(2):
        cmd = runtime.command(dirs / "ws", dirs / "out", dirs / "job.json")
    assert _setenv(cmd, "DYCRAFT_COAL_PYTHON") == str(Path(sys.executable).resolve())
    err = capsys.readouterr().err
    assert err.count("falling back") == 1, "warned once, not per spawn"
    assert "dycraft_no_such_module" in err


def test_no_bind_exposes_the_repository_or_home():
    repo = Path(I.__file__).resolve().parent
    assert I._refused(repo)
    assert I._refused(repo.parent)
    assert I._refused(Path.home().resolve())
    assert I._refused(Path("/"))
    assert I._refused(repo / ".venv") is None, "a repo-local venv is the uv default"


def test_collapse_drops_covered_and_system_binds():
    pairs = [("/opt/py", "/opt/py"), ("/opt/py/lib", "/opt/py/lib"),
             ("/usr/lib/x", "/usr/lib/x"), ("/real/env", "/alias/env")]
    assert I._collapse(pairs) == (("/real/env", "/alias/env"), ("/opt/py", "/opt/py"))


def test_the_check_starts_the_collision_interpreter_in_the_sandbox(dirs, monkeypatch):
    """`sandbox.coal_view` starts the interpreter as `CoalClient` does, so a layout whose binds look complete but
    which cannot start inside (uv's alias `home`, before its base prefix was bound there) fails the plain check, not
    the first collision check of a run."""
    from types import SimpleNamespace
    from dycraft import environment_test as E
    python = _layout("uv-alias-venv", dirs / "layout", monkeypatch)
    monkeypatch.setenv("DYCRAFT_COAL_PYTHON", python)
    monkeypatch.setattr(runtime, "COAL_IMPORTS", ("hashlib", MARKER))
    monkeypatch.setattr(runtime, "blender_binary", lambda: "/usr/bin/true")
    _require_sandbox(runtime.command(dirs / "ws", dirs / "out", dirs / "job.json"))
    result = E.check_coal_view(SimpleNamespace(tmp=dirs / "check", facts={}))
    assert result.status == E.PASS, (result.summary, result.detail)
    assert "starts and imports coal inside the sandbox" in result.summary
    # Without the bind at the alias, the check says why it cannot start.
    I._sandbox_view.cache_clear()
    monkeypatch.setattr(I, "_venv_home", lambda root: None)
    result = E.check_coal_view(SimpleNamespace(tmp=dirs / "check2", facts={}))
    I._sandbox_view.cache_clear()
    assert result.status == E.FAIL
    assert result.summary == f"{python} does not start inside the sandbox: Fatal Python error: Failed to import " \
                             "encodings module"
