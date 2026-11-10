"""Discovery of the frozen build123d interpreter, and the env its children need.

The single resolution chain for "which Python has the CAD kernel":
``DYCRAFT_B123D_PYTHON`` > the ``.b123d_python`` pointer file > conventional
``dycraft-b123d`` conda prefixes. Setting the env var to the empty string is a
deliberate "no interpreter" sentinel that suppresses every fallback.

Consumers: ``object_worker`` (which launches ``object.py``),
``verify._exact_adjudicator`` (which launches the collision child), and the CLI
preflight. Keeping one chain here is what stops authoring and adjudication from
disagreeing about which environment has the kernel.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from ... import _home, _oom
except ImportError:
    # Loaded by file path, outside the package: a driver that runs under any
    # python may do this on purpose, and relies on this module being
    # stdlib-only. `_oom` and `_home` are stdlib-only too, so load
    # them the same way rather than leave the probes below unpriced.
    import importlib.util as _ilu

    def _load_sibling(name):
        spec = _ilu.spec_from_file_location(
            name, Path(__file__).resolve().parents[2] / f"{name}.py")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _oom = _load_sibling("_oom")
    _home = _load_sibling("_home")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_POINTER_FILE = _home.state_path(".b123d_python")
_ENV_VAR = "DYCRAFT_B123D_PYTHON"


def _usable(cand: str | None, *, require_exec: bool = False) -> str | None:
    """Return `cand` resolved to an interpreter path, or None.

    `require_exec` is for auto-discovered candidates only: when guessing we must
    not settle on a non-executable file. For an interpreter the caller named
    explicitly we deliberately do NOT require the exec bit — handing it on lets
    the launch fail with `launch_failed`, which says "your interpreter is
    broken", rather than `unavailable`, which says "nothing was configured".
    """
    if not cand:
        return None
    if os.path.isfile(cand) and (os.access(cand, os.X_OK) or not require_exec):
        return cand
    return shutil.which(cand)


def _pointer_file_python() -> str | None:
    try:
        for line in _PYTHON_POINTER_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        pass
    return None


# DyCraft is frozen at build123d==0.11.1 (see `_b123d_version.REQUIRED_BUILD123D`
# and environment-b123d.yml). The pinned env is searched FIRST; the unsuffixed
# name is searched after it, and the child's version check refuses an
# interpreter there that does not carry the pinned build123d.
_ENV_NAMES = ("dycraft-b123d-0.11.1", "dycraft-b123d")


def _conventional_pythons() -> list[str]:
    """Conventional install prefixes for the build123d env, most-likely first."""
    prefixes = [Path.home(), _REPO_ROOT.parent, _REPO_ROOT.parent.parent]
    return [str(prefix / "conda_envs" / env / "bin" / "python")
            for env in _ENV_NAMES
            for prefix in prefixes]


def _lib_prepended_env(interp: str, base_env: dict | None = None) -> dict:
    """`base_env` with `interp`'s own ``lib/`` ahead of ``LD_LIBRARY_PATH``.

    Split out of `b123d_child_env` so the ambient-interpreter probe can build a
    child env for a *candidate* without asking `_resolve_b123d_python` which
    interpreter to use — that question is what the probe is answering, and
    calling back into it would recurse forever.
    """
    env = dict(base_env if base_env is not None else os.environ)
    lib = Path(interp).resolve().parent.parent / "lib"
    if lib.is_dir():
        prior = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (f"{lib}{os.pathsep}{prior}" if prior
                                  else str(lib))
    return env


def _ambient_pythons() -> list[str]:
    """Interpreters to consider as a last resort, most trustworthy first.

    `sys.executable` is the interpreter already running DyCraft. It is preferred
    over a bare ``python`` from ``PATH`` because it is absolute and immune to
    PATH ordering — and the eval prepends a sandbox bin dir to ``PATH`` inside
    every cell, so ``which("python")`` there can resolve somewhere nobody
    intended.
    """
    return [sys.executable, "python3", "python"]


@functools.lru_cache(maxsize=16)
def _has_pinned_build123d(cand: str) -> bool:
    """Whether `cand` has *the pinned* build123d, asked of the interpreter.

    Presence is not the test; the version is. The rollback env in
    `_ENV_NAMES` carries 0.9.1 precisely so a deliberate downgrade can be
    selected, and the worker child refuses anything but
    `REQUIRED_BUILD123D` with `build123d_version_mismatch`. Accepting "has
    build123d at all" here would trade one clear startup failure for a
    confusing failure on every compile.

    Cached per path: this costs a subprocess, and `b123d_child_env` runs on
    every child launch.
    """
    from ._b123d_version import REQUIRED_BUILD123D
    try:
        proc = subprocess.run(
            _oom.tiered([cand, "-c",
             "import importlib.metadata as m; print(m.version('build123d'))"], _oom.WORKER),
            capture_output=True, text=True, timeout=60,
            env=_lib_prepended_env(cand))
    except BaseException:  # noqa: BLE001 — unusable interpreter is "no"
        return False
    return proc.returncode == 0 and proc.stdout.strip() == REQUIRED_BUILD123D


def b123d_child_env(base_env: dict | None = None) -> dict:
    """Environment for a child launched under the CAD interpreter.

    Prepends the interpreter's own ``lib/`` to ``LD_LIBRARY_PATH``.

    This is not cosmetic. The ``cadquery-ocp`` wheel links the *system*
    ``libexpat`` (2.2 on this host), and once OCP loads it into the process the
    conda-built ``pyexpat`` — compiled against expat 2.8 — fails with
    ``undefined symbol: XML_SetAllocTrackerActivationThreshold``. build123d
    0.11.1 imports IPython at module scope, which reaches ``xml.dom.minidom``,
    so *every* import of build123d hits it. Putting the env's own lib directory
    ahead of the system loader path makes the conda expat win.

    Harmless for an interpreter that does not need it (the 0.9.1 rollback env
    prepends its own lib dir and behaves exactly as before).
    """
    interp = _resolve_b123d_python()
    if not interp:
        return dict(base_env if base_env is not None else os.environ)
    return _lib_prepended_env(interp, base_env)


def _resolve_b123d_python(explicit: str | None = None) -> str | None:
    """Resolve the CAD interpreter to an absolute path/command, or None.

    Precedence: explicit arg > ``DYCRAFT_B123D_PYTHON`` > the ``.b123d_python``
    pointer file > conventional conda prefixes > the ambient interpreter, if it
    carries the pinned build123d. An env var set to the empty string is an
    explicit "no interpreter" sentinel that suppresses every fallback, the
    ambient one included.

    The ambient step is deliberately last and deliberately *gated*. It exists
    because the earlier candidates are all repo-relative or host-conventional
    and can therefore be invisible in a context that is otherwise perfectly
    configured: the pointer file lives under ``_home.home()``, which for a
    source tree is the tree itself, so inside a git worktree it points at the
    worktree, where the untracked ``.b123d_python`` does not exist. A revision
    ablation hit exactly that — every cell died in
    0.3s with ``no build123d interpreter found`` while the interpreter running
    the eval had 0.11.1 installed the whole time.

    An explicitly named interpreter is *not* gated, and must not be: this
    function's contract is that naming one and having it fail produces
    ``launch_failed`` ("your interpreter is broken"), never a silent fall
    through to somewhere else. Falling through would mean a typo in
    ``DYCRAFT_B123D_PYTHON`` produces a *successful* run against an interpreter
    nobody chose, which is worse than the error it replaces.
    """
    if explicit:
        return _usable(explicit)
    env = os.environ.get(_ENV_VAR)
    if env is not None and not env.strip():
        return None          # set-but-empty -> deliberately unconfigured
    if env:
        return _usable(env)
    for cand in [_pointer_file_python(), *_conventional_pythons()]:
        found = _usable(cand, require_exec=True)
        if found:
            return found
    for cand in _ambient_pythons():
        found = _usable(cand, require_exec=True)
        if found and _has_pinned_build123d(found):
            return found
    return None


def b123d_python() -> str | None:
    """Public preflight: the resolved build123d interpreter, or None.

    Without an up-front check the shortfall is silent, because the object layer
    is offered unconditionally and a run with no interpreter would otherwise
    reach the point of committing geometry before discovering it cannot build
    any.
    """
    return _resolve_b123d_python()


def probe_build123d_version(interp: str | None = None,
                            timeout: float = 60.0) -> str | None:
    """The build123d version installed in `interp`, or None if there is none.

    Resolving an interpreter path is not the same as finding a usable CAD
    environment: a `.b123d_python` recorded months ago can point at a Python
    that has since lost build123d, or never had it. Path resolution alone then
    reports the environment healthy and every compile fails later, inside the
    worker, where a human is no longer watching.

    So this asks the interpreter itself, which is the only source that can
    answer. One subprocess, paid once at launch by the CLI preflight; the worker
    child does its own strict check per run (`check_build123d_version`).

    None covers every way the question can fail to produce a version — no
    interpreter, build123d absent, the interpreter refusing to start, a hang —
    because the caller's next move is the same for all of them.
    """
    interp = interp or _resolve_b123d_python()
    if not interp:
        return None
    try:
        proc = subprocess.run(
            _oom.tiered([interp, "-c",
             "import importlib.metadata as m; print(m.version('build123d'))"], _oom.WORKER),
            capture_output=True, text=True, timeout=timeout,
            env=b123d_child_env())
    except BaseException:  # noqa: BLE001 — an unusable interpreter is "no version"
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def probe_ocp(interp: str | None = None,
              timeout: float = 60.0) -> bool:
    """Whether `import OCP` resolves in `interp`.

    Asked separately from the build123d version because it is a separate way
    for the same environment to be wrong, and `object.py` is given `OCP` as a
    global: an interpreter with build123d and no OCP fails at namespace
    construction, before any authored line runs.

    `find_spec`, not a real import — OCP costs a second and hundreds of
    megabytes to load, and the question here is only whether it is there.
    """
    interp = interp or _resolve_b123d_python()
    if not interp:
        return False
    try:
        proc = subprocess.run(
            _oom.tiered([interp, "-c",
             "from importlib.util import find_spec; "
             "print(find_spec('OCP') is not None)"], _oom.WORKER),
            capture_output=True, text=True, timeout=timeout,
            env=b123d_child_env())
    except BaseException:  # noqa: BLE001 — an unusable interpreter is "no OCP"
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "True"
