"""Discovery of the Blender (`bpy`) interpreter, and the env its child needs.

The single resolution chain for "which Python has Blender":
``DYCRAFT_BLENDER_PYTHON`` > the ``.blender_python`` pointer file >
conventional ``dycraft-blender`` conda prefixes. Setting the env var to the
empty string is a deliberate "no interpreter" sentinel that suppresses every
fallback.

Deliberately the same shape as ``agent/cad/_cad_env.py`` — this repo
already resolves other out-of-process interpreters (build123d, cuRobo) and one
more that invented its own precedence would be one more thing to remember. If
you change the rule, change it in both.

Why a separate interpreter at all: the ``bpy`` wheel ships an entire Blender
inside a Python package and pins the minor version it was built for, so it
cannot share the harness env. The subprocess boundary also buys back something
the Open3D backend pays dearly for — Filament's thread-ownership aborts (see
``open3d_renderer``'s "Filament thread ownership" note) simply do not exist for
a renderer that lives and dies in its own process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .. import _home, _oom

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_POINTER_FILE = _home.state_path(".blender_python")
_ENV_VAR = "DYCRAFT_BLENDER_PYTHON"

#: Conventional env names, most-likely first.
_ENV_NAMES = ("dycraft-blender",)


def _usable(cand: str | None, *, require_exec: bool = False) -> str | None:
    """Return `cand` resolved to an interpreter path, or None.

    `require_exec` is for auto-discovered candidates only: when guessing we must
    not settle on a non-executable file. For an interpreter the caller named
    explicitly we deliberately do NOT require the exec bit, so a broken pin
    fails loudly at launch rather than silently degrading to "not configured".
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


def _conventional_pythons() -> list[str]:
    """Conventional install prefixes for the Blender env, most-likely first."""
    prefixes = [Path.home(), _REPO_ROOT.parent, _REPO_ROOT.parent.parent]
    return [str(prefix / "conda_envs" / env / "bin" / "python")
            for env in _ENV_NAMES
            for prefix in prefixes]


def _resolve_blender_python(explicit: str | None = None) -> str | None:
    """Resolve the Blender interpreter to an absolute path/command, or None.

    Precedence: explicit arg > ``DYCRAFT_BLENDER_PYTHON`` > the
    ``.blender_python`` pointer file > conventional conda prefixes. An env var
    set to the empty string is an explicit "no interpreter" sentinel.
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
    return None


def blender_python() -> str | None:
    """Public preflight: the resolved Blender interpreter, or None."""
    return _resolve_blender_python()


#: Pointer file for an official Blender *binary*, resolved the same way the
#: interpreter is. Written by hand or by a setup script; one line, a path.
_BINARY_POINTER_FILE = _home.state_path(".blender_binary")
_BINARY_ENV_VAR = "DYCRAFT_BLENDER_BINARY"
#: Conventional install roots, most-likely first. A directory here is expected
#: to be an extracted blender.org tarball, i.e. to contain a `blender` binary.
_BINARY_CONVENTIONAL = (
    Path.home() / "blender",
)


def blender_binary() -> str | None:
    """The official Blender executable to run workers under, or None.

    Why this exists alongside `blender_python`: the `bpy` PyPI wheel is built
    with ``WITH_PYTHON_MODULE=ON``, and in that configuration Mantaflow's
    Python bindings do not work — baking any fluid domain, liquid or gas,
    aborts inside the solver with missing methods on manta's own grid and
    particle classes (bpy 4.5.13, 4.5.14 and 5.0.1), where an official
    blender.org build of the same 4.5 LTS line bakes the identical scene to
    ``FINISHED``. Fluid is therefore only available when the worker runs under
    a real binary, which is what this resolves.

    Precedence mirrors `_resolve_blender_python`, with one addition:
    ``DYCRAFT_BLENDER_BINARY`` > the ``.blender_binary`` pointer file >
    conventional install roots > ``blender`` on ``PATH``. Set the env var to
    the empty string to force "no binary" and fall back to the `bpy` module
    path.
    """
    raw = os.environ.get(_BINARY_ENV_VAR)
    if raw is not None:
        return _usable_binary(raw)
    try:
        if _BINARY_POINTER_FILE.is_file():
            found = _usable_binary(_BINARY_POINTER_FILE.read_text().strip())
            if found:
                return found
    except OSError:
        pass
    for root in _BINARY_CONVENTIONAL:
        try:
            if not root.is_dir():
                continue
            # Newest first, so an added build wins without editing anything.
            for child in sorted(root.iterdir(), reverse=True):
                found = _usable_binary(child / "blender")
                if found:
                    return found
        except OSError:
            continue
    # `PATH` last, and only as a backstop. It is the least specific answer -- a
    # distribution package could be an older line than the wheel, and this
    # cannot tell -- but it is the difference between finding a usable build and
    # silently falling back to a `bpy` wheel that cannot bake fluid at all. A
    # machine with Blender installed anywhere normal now gets fluid without
    # anyone remembering to write a pointer file, and the three
    # higher-precedence sources still override it. Measured: without this, a
    # host resolved None with Blender 4.5.13 LTS sitting on `PATH`, and every
    # fluid bake failed inside Mantaflow with no indication that the build was
    # the reason.
    return _usable_binary(shutil.which("blender"))


def _usable_binary(cand) -> str | None:
    """`cand` resolved to an executable Blender binary, or None."""
    if not cand:
        return None
    path = Path(str(cand)).expanduser()
    if path.is_dir():
        path = path / "blender"
    if not path.is_file() or not os.access(path, os.X_OK):
        return None
    return str(path.resolve())


def is_available() -> bool:
    return blender_python() is not None


def blender_child_env(base_env: dict | None = None) -> dict:
    """Environment for the render child.

    Strips the software-GL pins the launcher sets for Filament
    (`cli.py::build_env_dict` exports `LIBGL_ALWAYS_SOFTWARE=1`,
    `MESA_LOADER_DRIVER_OVERRIDE=llvmpipe`, `EGL_PLATFORM=surfaceless`). Those
    exist because Open3D's Filament renderer must come up on a headless box with
    no GPU driver, and they are actively wrong here: Cycles reaches the GPU
    through CUDA/OptiX, not GL, but Blender still probes GL at startup and an
    llvmpipe pin makes that probe slow and noisy for a renderer that will not
    use it.

    Also drops `PYTHONPATH` and `PYTHONHOME`. The child must import the `bpy`
    wheel from its own env and nothing else — a `PYTHONPATH` inherited from the
    harness env would put the *harness's* numpy ahead of Blender's, and a
    mismatched numpy ABI inside bpy is an import-time crash rather than a
    graceful `unavailable`.
    """
    env = dict(base_env if base_env is not None else os.environ)
    for key in ("LIBGL_ALWAYS_SOFTWARE", "MESA_LOADER_DRIVER_OVERRIDE",
                "EGL_PLATFORM", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    return env


def probe_blender_version(interp: str | None = None,
                          timeout: float = 120.0) -> str | None:
    """`bpy.app.version_string` under `interp`, or None if it will not import.

    A probe rather than an assumption, for the reason the CAD env has one: a
    pointer file recorded months ago can point at an interpreter whose wheel was
    since removed or upgraded, and the failure mode without a probe is a render
    that dies mid-run instead of a backend that reports itself unavailable.

    Slow (importing bpy is seconds), so callers should cache it rather than ask
    per render.
    """
    interp = interp or _resolve_blender_python()
    if not interp:
        return None
    try:
        out = subprocess.run(
            _oom.tiered([interp, "-c", "import bpy; print(bpy.app.version_string)"],
                        _oom.WORKER),
            capture_output=True, text=True, timeout=timeout,
            env=blender_child_env())
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None
