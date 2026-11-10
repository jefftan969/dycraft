"""Shared pytest configuration.

Two jobs: bind `dycraft` to the checkout these tests live in, and register the
`expensive` marker.

**The binding matters.** The repository directory is not necessarily named
`dycraft` (a second checkout is conventionally `dycraft-dev`), and an editable
install may map the name somewhere else entirely. Without an explicit binding a
test can import a *different* copy of the package than the one it is meant to
exercise, and the only symptom is results that do not match the code in front of
you. `_bind_dycraft()` resolves the package from this file's own location, so
the tests are self-locating and depend on nothing about where the repo sits or
what its directory is called.

Subprocess tests need the same guarantee. A directory holding a single `dycraft`
symlink to this checkout is put first on `PYTHONPATH` for the whole session, so
every child's `python -m dycraft…` imports this checkout, and the session refuses
to start if a child would still import another copy. The `pkgroot` fixture
returns that directory, for a test that builds a child's environment from scratch.

`dycraft_bootstrap()` is the same binding for a *child* program: several tests
run a snippet under the CAD interpreter, where dycraft is not installed, and
each one needs the package on `sys.modules` before it can import anything from
it. One definition, so a fix to the bootstrap cannot land in only three of four
copies.

Expensive tests are the slow ones (Blender, CAD builds, >30 s); none starts an agent CLI or spends usage. They are
skipped unless `--run-expensive` is passed or `DYCRAFT_RUN_EXPENSIVE=1` is set — the env-var form is convenient for
CI configurations that don't expose pytest options easily.
"""
from __future__ import annotations

import atexit
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# `resolve()` is deliberate: it follows any symlink used to expose this repo
# under the importable name, landing on the real checkout.
REPO = Path(__file__).resolve().parent.parent


def _bind_dycraft() -> None:
    """Make `import dycraft` resolve to the checkout this file belongs to."""
    existing = getattr(sys.modules.get("dycraft"), "__file__", None)
    if existing is not None:
        if Path(existing).resolve() != (REPO / "__init__.py").resolve():
            raise RuntimeError(
                f"`dycraft` was already imported from {existing}, but these "
                f"tests belong to {REPO}. Run pytest with no conflicting "
                f"PYTHONPATH so the tests exercise their own checkout.")
        return
    spec = importlib.util.spec_from_file_location(
        "dycraft", REPO / "__init__.py",
        submodule_search_locations=[str(REPO)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["dycraft"] = module
    spec.loader.exec_module(module)


_bind_dycraft()


def _export_child_pkgroot() -> Path:
    """Put a `dycraft` symlink to this checkout first on every child's `PYTHONPATH`.

    The package is installed from a development checkout, so without this a
    child's `python -m dycraft…` imports that copy, and a subprocess test passes
    or fails against code that is not the tree under test. Done at import, before
    any test can start a child; the parent's own `sys.path` is unaffected.
    """
    root = Path(tempfile.mkdtemp(prefix="dycraft_pkgroot_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    (root / "dycraft").symlink_to(REPO, target_is_directory=True)
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(root)] + ([previous] if previous else []))
    return root


_CHILD_PKGROOT = _export_child_pkgroot()


@pytest.fixture(scope="session")
def sibling_test_module():
    """Import another test file's helpers by name, e.g. `test_import_gltf`.

    `tests/` is not a package and this directory is not on `sys.path` under the
    documented invocation (`pytest dycraft/tests`, one level above the checkout),
    so neither `import test_import_gltf` nor `from tests.test_import_gltf
    import ...` resolves — the second one only works when the cwd happens to be
    the repo root, which is exactly the kind of hidden dependency `_bind_dycraft`
    exists to remove. Loading by path off `REPO` is the same self-locating trick,
    and having it in one place is what keeps a fixture shared between two test
    files from being a collection error in half the ways the suite is run.
    """
    cache: dict = {}

    def load(name: str):
        if name not in cache:
            path = REPO / "tests" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(
                f"_dycraft_sibling_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cache[name] = mod
        return cache[name]

    return load


@pytest.fixture(scope="session")
def cad_bootstrap() -> str:
    """Python source that binds `dycraft` to this checkout, for a child program.

    Prepended to a snippet run under an interpreter that has build123d but no
    dycraft install (see `_cad_env.b123d_python`). Binds by file path, so it
    needs no PYTHONPATH, no symlink and no install — the same mechanism
    `_object_worker_child` uses in production.
    """
    return (
        "import sys, importlib.util as _u\n"
        f"_REPO = {str(REPO)!r}\n"
        "_sp = _u.spec_from_file_location('dycraft', _REPO + '/__init__.py',\n"
        "                                submodule_search_locations=[_REPO])\n"
        "_m = _u.module_from_spec(_sp); sys.modules['dycraft'] = _m\n"
        "_sp.loader.exec_module(_m)\n"
    )


@pytest.fixture(scope="session")
def pkgroot() -> Path:
    """A directory whose only entry is a `dycraft` symlink to this checkout.

    Already first on `PYTHONPATH` for every child (`_export_child_pkgroot`); pass
    it explicitly to a child whose environment is built from scratch. A child
    cannot use the in-process binding above, and the package name it needs may
    not match this repo's directory name, so the symlink supplies the name
    without copying anything.
    """
    return _CHILD_PKGROOT


@pytest.fixture(autouse=True)
def _no_viewer_flat_pass(request, monkeypatch):
    """Keep the viewer's shader colour pass (`viewer/derived._flat_pass`) from starting Blender under test.

    Every URDF request for a run with a shader part starts it, so any test that serves such a run would start a
    Blender process that outlives the test. Under test the pass fails at once instead, which leaves those parts grey.
    A test of the pass patches `_flat_pass` itself; an expensive one runs the real thing.
    """
    if request.node.get_closest_marker("expensive"):
        yield
        return
    from dycraft.render import RenderUnavailable
    from dycraft.viewer import derived

    def _refuse(run_dir, entries):
        raise RenderUnavailable("the shader colour pass does not run under test")

    monkeypatch.setattr(derived, "_flat_pass", _refuse)
    yield


@pytest.fixture(autouse=True)
def _clear_compile_cache():
    """Drop the per-part compile memo between tests.

    `_solid_compile.compile_entry` memoizes on the geometry handed to it, and
    that memo deliberately outlives a build — a scene-layer `scene.py` recompiles
    every object it authors on every build, and the ones the author did not
    touch reconstruct byte-identically. The premise is that the function is a
    pure function of its entry, which holds in production and does not hold in
    a test that monkeypatches a stage inside it: the next test compiling the
    same fixture geometry would be served the result from before the patch.

    Autouse and unconditional rather than opt-in per test, because the tests it
    protects are the ones that do not know they need it.
    """
    yield
    try:
        from dycraft.agent.cad._solid_compile import clear_compile_cache
    except ImportError:
        return
    clear_compile_cache()


def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Run @pytest.mark.expensive tests (full claude/codex end-to-end "
             "runs, external API calls). Equivalent to "
             "DYCRAFT_RUN_EXPENSIVE=1.",
    )


#: The viewer's asset cache for the whole session (bakes, shader colors, trajectory payloads), so no test writes into
#: the user's home. Set once before any test, so every test's environment snapshot (`_no_environ_leak`) agrees; a test
#: that names its own cache sets `DYCRAFT_VIEWER_ASSET_CACHE_DIR` itself, which wins.
_VIEWER_CACHE_SET: list[str] = []


def pytest_configure(config):
    if "DYCRAFT_VIEWER_ASSET_CACHE_DIR" not in os.environ:
        import tempfile
        cache = tempfile.mkdtemp(prefix="dycraft-test-viewer-cache-")
        os.environ["DYCRAFT_VIEWER_ASSET_CACHE_DIR"] = cache
        _VIEWER_CACHE_SET.append(cache)
    # `python -m dycraft.cli` probes this machine's interpreters, Blender and CLIs before it starts anything
    # (`cli._environment_preflight`): seconds per launch, and an answer about this machine rather than the code a
    # test drives. Set for the whole session, so it is part of every test's environment baseline;
    # `test_launch_preflight.py` unsets it to test the checks themselves.
    os.environ.setdefault("DYCRAFT_SKIP_PREFLIGHT", "1")
    # The GL libraries `install tools` put in place (`.gl_prefix`), on the loader path of every child a test starts,
    # as `python -m dycraft.*` puts them there (`_home.reexec_for_loader`).
    from dycraft import _home
    _home.gl_on_path()
    # Refuse to run if a child process would still import another dycraft
    # (e.g. a `.pth` entry or a `PYTHONPATH` the export above cannot outrank).
    found = subprocess.run(
        [sys.executable, "-c", "import dycraft; print(dycraft.__file__)"],
        capture_output=True, text=True, timeout=60).stdout.strip()
    if not found or Path(found).resolve() != (REPO / "__init__.py").resolve():
        raise pytest.UsageError(
            f"a child process imports dycraft from {found or '(nowhere)'}, not "
            f"from the checkout under test ({REPO}); subprocess tests would "
            f"exercise the wrong code.")
    config.addinivalue_line(
        "markers",
        "expensive: slow tests (Blender, CAD builds, >30 s wall time); none "
        "starts an agent CLI. Skipped by default; pass --run-expensive "
        "or set DYCRAFT_RUN_EXPENSIVE=1 to include.",
    )
    config.addinivalue_line(
        "markers",
        "cold_pool: the test's claim depends on CAD worker-pool state — it "
        "compares a warm child against a cold one, reads the pool's live/idle "
        "accounting, or exercises the cap, the idle TTL or the non-pooled "
        "path. `tests/test_object_worker_pool.py` tears the pool down around "
        "these and only these; every other test there may inherit a warm "
        "child, which is what keeps the file from paying `import build123d` "
        "forty times.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-expensive"):
        return
    if os.environ.get("DYCRAFT_RUN_EXPENSIVE") == "1":
        return
    skip = pytest.mark.skip(
        reason="needs --run-expensive (or DYCRAFT_RUN_EXPENSIVE=1)")
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Environment isolation (audit E4)
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_environ(monkeypatch):
    """Swap `os.environ` for a throwaway copy for the duration of one test.

    For tests that drive `cli.main()`. On its way to `execvpe` that function
    exports into `os.environ` — the CAD interpreter and
    `DYCRAFT_RENDER_BACKEND` among others — which is
    correct in production, where the exec happens microseconds later and
    replaces the process. Under test the exec is stubbed, so those writes
    outlive the test.

    A leaked `DYCRAFT_RENDER_BACKEND=blender`, for example, makes later tests
    render through Cycles and fail with `rendering unavailable` where no
    Blender interpreter is configured, while passing in isolation.

    A plain dict keeps `main`'s writes visible to the code under test
    (everything reads `os.environ` at call time) and lets monkeypatch put the
    real mapping back afterwards. Note that it does **not** reach subprocesses,
    which inherit the real environment — so a test that spawns a child and
    expects it to see an exported value wants `monkeypatch.setenv`, not this.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    return os.environ


#: Environment variables whose value decides how a *later* test behaves — the
#: ones a leak actually damages. Deliberately not "every variable": importing
#: PySide, torch or decord sets `LD_LIBRARY_PATH` and `QT_QPA_*` as a
#: process-wide side effect of the first import, which no test can undo and
#: none of them causes.
_LEAK_GUARDED_PREFIXES = ("DYCRAFT_", "CODEX_HOME", "CLAUDE_CONFIG_DIR",
                          "ANTHROPIC_", "OPENAI_", "OPENROUTER_", "GEMINI_",
                          "GOOGLE_API_KEY", "QWEN_")


def _guarded_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if k.startswith(_LEAK_GUARDED_PREFIXES)}


@pytest.fixture(autouse=True)
def _no_environ_leak():
    """Fail a test that permanently changes a run-shaping environment variable.

    The point is the *next* test, not this one: a leaked variable makes an
    unrelated file fail later, in a way that passes when run alone and is
    therefore attributed to anything but its actual cause. Catching it at the
    boundary names the culprit instead of the victim.

    Runs after `monkeypatch` undoes its own `setenv`/`delenv` — an autouse
    conftest fixture sets up before the test's own fixtures, so it tears down
    after them. Tests that legitimately need `main()` to write env should take
    the `isolated_environ` fixture.
    """
    def snapshot():
        return _guarded_env()

    before = snapshot()
    yield
    after = snapshot()
    if after == before:
        return
    added = {k: after[k] for k in after.keys() - before.keys()}
    removed = sorted(before.keys() - after.keys())
    changed = sorted(k for k in before.keys() & after.keys()
                     if before[k] != after[k])
    for k in added:                      # restore, so one leak is not many
        os.environ.pop(k, None)
    for k in removed:
        os.environ[k] = before[k]
    for k in changed:
        os.environ[k] = before[k]
    raise AssertionError(
        "this test leaked os.environ into every test after it — "
        f"added={sorted(added)} removed={removed} changed={changed}. "
        "Use the `isolated_environ` fixture (for cli.main) or "
        "monkeypatch.setenv (for a single variable).")


def pytest_unconfigure(config):
    for cache in _VIEWER_CACHE_SET:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
        os.environ.pop("DYCRAFT_VIEWER_ASSET_CACHE_DIR", None)


@pytest.fixture
def namespaces_sandbox(monkeypatch):
    """For tests of the bubblewrap argv itself: `runtime.command` builds it whatever this machine's own Blender
    sandbox is (`runtime.sandbox_mode`; a host that cannot make namespaces gets Landlock's launcher instead)."""
    from dycraft.animation.blender import runtime
    monkeypatch.delenv(runtime.SANDBOX_ENV, raising=False)
    monkeypatch.setattr(runtime, "_namespaces_problem", lambda: "")


@pytest.fixture(autouse=True)
def _forget_sandbox_probes():
    """`runtime` remembers whether this machine can make the worker's namespaces. A test that patches
    `shutil.which` (or the probes) must not leave its answer to the next, which would then build another sandbox."""
    yield
    runtime = sys.modules.get("dycraft.animation.blender.runtime")
    if runtime is not None:
        runtime._NAMESPACES_PROBLEM = None
        runtime._USERNS_PREFIX = None


@pytest.fixture
def apparmor(monkeypatch, tmp_path):
    """This machine's AppArmor policy on user namespaces as a test sets it (`_userns`), so advice that depends on it
    does not depend on the machine running the tests: `apparmor(restricted=, bwrap=, profiled=)`."""
    from dycraft import _userns

    def pin(*, restricted=True, bwrap=True, profiled=False):
        monkeypatch.setattr(_userns, "restricted", lambda: restricted)
        monkeypatch.setattr(_userns, "SYSTEM_BWRAP", Path("/bin/sh") if bwrap else tmp_path / "no-system-bwrap")
        monkeypatch.setattr(_userns, "bwrap_profiled", lambda: profiled)
    return pin
