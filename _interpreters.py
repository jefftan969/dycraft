"""Resolving the out-of-process interpreters DyCraft shells out to.

`h.ik` runs agent-authored code under an interpreter that has cuRobo, which the
main environment does not. It is optional: without it the op reports
`error_kind="unavailable"` and the layer keeps working. `resolve` finds that
interpreter: an explicit argument, then an env override, then the pointer
file the setup wrote, then conventional install prefixes.

A caller that names an interpreter gets it resolved through `PATH` if it is
not a path, and gets no exec-bit check; a candidate we *guessed* must be an
executable file before we settle on it. The reasoning for that asymmetry is
in `_usable`.

Stdlib only. Imported by the parent half (`animation/ik/ik.py`), never by a
child runner — a child runs under the foreign
interpreter and may not import `dycraft.*` at all.

`sandbox_view` answers the question that comes *after* resolution: what a
bubblewrap sandbox has to contain for an interpreter to start and import what
it needs. See its docstring for why that cannot be read off the path.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

try:
    from . import _oom
except ImportError:
    # Loaded by file path, outside the package, the way `_cad_env` allows for.
    # `_oom` is stdlib-only as well, so the probes below stay priced.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_oom", Path(__file__).with_name("_oom.py"))
    _oom = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_oom)

__all__ = ["read_pointer", "resolve", "sandbox_view", "SandboxView"]


def _usable(cand: str | None, *, require_exec: bool = False) -> str | None:
    """`cand` resolved to an interpreter path or command, or None.

    `require_exec` is for auto-discovered candidates only: when guessing we
    must not settle on a non-executable file. For an interpreter the caller
    named explicitly we deliberately do **not** require the exec bit — handing
    it on lets the launch fail with `launch_failed`, which says "your
    interpreter is broken", rather than `unavailable`, which says "nothing was
    configured". Those are different problems and the operator needs to be told
    which one they have.
    """
    if not cand:
        return None
    if os.path.isfile(cand) and (os.access(cand, os.X_OK) or not require_exec):
        return cand
    return shutil.which(cand)


def read_pointer(path) -> str | None:
    """First non-comment, non-blank line of a setup script's pointer file.

    A pointer file (`.ik_python`) is gitignored operator state: one
    interpreter path, written by the setup (`python -m dycraft.install ik`),
    with a `#` header saying what wrote it and who reads it. A missing or unreadable file is not an error — it is the ordinary state
    of a checkout where that optional subsystem was never set up.
    """
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        return None
    return None


def resolve(*, env_var: str, pointer, conventional=(),
            explicit: str | None = None) -> str | None:
    """The interpreter for one optional subsystem, or None for "unavailable".

    Precedence, highest first:

      1. `explicit` — an argument the caller passed through from its own API.
      2. `env_var`. **Set-but-empty forces None**, suppressing every fallback.
         That is the sentinel a test uses to exercise the unavailable path
         without uninstalling anything, and it has to beat the pointer file or
         a developer machine with a working setup could not reach that path.
      3. `pointer` — the path of the pointer file its setup script writes.
      4. `conventional` — install prefixes to guess, most-likely first.

    None is a *supported* outcome, not an error. Every caller of this function
    reports it as `error_kind="unavailable"` and carries on without the
    dependency; the layer stays usable and the agent is told the shortfall is
    not a failure of its code.
    """
    if explicit:
        return _usable(explicit)

    env = os.environ.get(env_var)
    if env is not None and not env.strip():
        return None                      # set-but-empty → deliberately off
    if env:
        return _usable(env)

    for cand in [read_pointer(pointer), *(str(c) for c in conventional)]:
        found = _usable(cand, require_exec=True)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# what a sandbox must contain for an interpreter to run in it
# ---------------------------------------------------------------------------

#: Bound by every sandbox `animation/blender/runtime.py::command` builds, so a
#: view never repeats them (and never needs a directory created inside them,
#: which a read-only bind would refuse).
SYSTEM_BINDS = ("/usr", "/lib", "/lib64")

#: Run under the interpreter being described. Imports what the child will
#: import, then reports where everything came from — including every file the
#: loader mapped, which is the only reliable account of the shared libraries an
#: extension module needs (a conda `libcrypto`, a wheel's `numpy.libs`, a
#: Homebrew glibc) without parsing ELF.
_VIEW_PROBE = r'''
import json, os, sys, sysconfig
# Captured before anything is imported: a module may rewrite `sys.path` as it
# loads -- the bpy wheel inserts `bpy/<ver>/scripts/modules` and swaps itself
# for the package found there -- and a root that only exists *because* of an
# import is not a root the child should be handed.
start = list(sys.path)
mods = {n: getattr(__import__(n), "__file__", None) for n in sys.argv[1:]}
maps = set()
with open("/proc/self/maps") as f:
    for line in f:
        parts = line.split(None, 5)
        name = parts[5].strip() if len(parts) == 6 else ""
        # Regular files only: `/memfd:...`, `(deleted)` and device mappings
        # name nothing a bind could or should reproduce.
        if (name.startswith("/") and os.path.isfile(name)
                and not name.startswith(("/dev/", "/proc/", "/sys/"))):
            maps.add(name)
print(json.dumps({"executable": sys.executable, "prefix": sys.prefix,
                  "base_prefix": sys.base_prefix,
                  "stdlib": sysconfig.get_path("stdlib"),
                  "platstdlib": sysconfig.get_path("platstdlib"),
                  "path": start, "modules": mods, "maps": sorted(maps)}))
'''


class SandboxView(NamedTuple):
    """How to run one interpreter inside a bubblewrap sandbox.

    `launch` is the path to exec, `binds` the `(source, destination)` pairs to
    `--ro-bind`, and `sitedirs` the import roots an *isolated* start (`-I`, or
    the closed `worker_env`) would not find on its own — a user site, a
    `PYTHONPATH` install, a module-system tree — which the child has to be told
    about explicitly.
    """
    launch: str
    binds: tuple
    sitedirs: tuple


def _symlink_hops(path: Path) -> list[Path]:
    """`path` and every hop its symlink chain takes, as spelled at each hop.

    Each hop's directory has to exist inside the sandbox *under that spelling*,
    or the kernel cannot follow the link — which is how a venv's `bin/python`
    reaches the base interpreter it points at.
    """
    hops = [path]
    while hops[-1].is_symlink() and len(hops) < 40:
        target = Path(os.readlink(hops[-1]))
        hops.append(Path(os.path.normpath(
            target if target.is_absolute() else hops[-1].parent / target)))
    return hops


def _venv_home(root: Path) -> Path | None:
    """The `home` a venv's `pyvenv.cfg` names (its base interpreter's directory), as spelled there."""
    try:
        text = (root / "pyvenv.cfg").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "home" and value.strip():
            return Path(os.path.normpath(value.strip()))
    return None


def _load_home():
    """`dycraft._home`, loaded by file path: this module may run outside the
    package (a driver may load it by path from any python), and `_home` is
    stdlib-only for that reason."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_dycraft_home", Path(__file__).resolve().parent / "_home.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _refused(dest: Path) -> str | None:
    """Why `dest` must not be bound into the sandbox, or None.

    The sandbox exists to keep agent-authored Blender code away from the
    project and the operator's files, so no bind may contain either: `/`, the
    user-state root (runs, pointer files, assets), `$HOME`, or, for a source
    tree, the repository. A path *inside* them is fine — `.venv` in the repo
    root is the ordinary uv layout, and binding it exposes the venv, not the
    checkout.

    An installed package directory is not guarded: it sits inside the
    interpreter's own environment, which has to be bound, and holds no user
    state.
    """
    home = _load_home()
    guards = [("the user-state root", home.home()), ("$HOME", Path.home())]
    if not home.installed():
        guards.append(("the repository", Path(__file__).resolve().parent))
    for label, guarded in guards:
        try:
            guarded = guarded.resolve()
        except OSError:
            continue
        if dest == guarded or dest in guarded.parents:
            return f"it would expose {label} ({guarded})"
    if dest == Path("/"):
        return "it would expose the whole filesystem"
    return None


def _collapse(pairs) -> tuple:
    """`(source, destination)` pairs minus the ones another pair already covers."""
    system = [Path(p) for p in SYSTEM_BINDS]
    kept: list[tuple[Path, Path]] = []
    for src, dest in sorted({(Path(s), Path(d)) for s, d in pairs},
                            key=lambda p: (len(p[1].parts), str(p[1]))):
        if src == dest and any(dest == s or s in dest.parents for s in system):
            continue
        if any((dest == d or d in dest.parents)
               and src == s / dest.relative_to(d) for s, d in kept):
            continue
        kept.append((src, dest))
    return tuple((str(s), str(d)) for s, d in kept)


def _run_json(argv, *, timeout: float) -> dict:
    proc = subprocess.run(_oom.tiered(argv, _oom.WORKER), capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode:
        tail = (proc.stderr.strip().splitlines() or ["(no output)"])[-1]
        raise RuntimeError(tail)
    return json.loads(proc.stdout.strip().splitlines()[-1])


@functools.lru_cache(maxsize=32)
def _sandbox_view(python: str, modules: tuple, _env_key: tuple) -> SandboxView:
    exe = python if os.path.isfile(python) else (shutil.which(python) or python)
    try:
        info = _run_json([exe, "-c", _VIEW_PROBE, *modules], timeout=180)
        isolated = _run_json([exe, "-I", "-c",
                              "import json, sys; print(json.dumps(sys.path))"],
                             timeout=60)
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        wanted = f" import {', '.join(modules)}" if modules else " start"
        raise RuntimeError(f"{python} cannot{wanted}: {exc}") from exc

    # The directory is resolved and the name is not. Resolving the directory is
    # what keeps an aliased mount (a home directory reached through a
    # symlink, as on some clusters) out of the sandbox, where the alias does
    # not exist. Keeping the name is what keeps a venv a venv: `bin/python` is
    # a symlink to the base interpreter, and Python finds `pyvenv.cfg` next to
    # the path it was *invoked* as, so exec'ing the resolved target silently
    # drops every package the venv holds.
    named = Path(info["executable"])
    launch = named.parent.resolve() / named.name

    root = next((d for d in (launch.parent, launch.parent.parent)
                 if (d / "pyvenv.cfg").is_file()), None)
    wanted = {root or Path(info["prefix"]),
              Path(info["stdlib"]), Path(info["platstdlib"])}
    # A venv's base prefix as its `pyvenv.cfg` spells it. Python looks for the
    # standard library next to that `home`, under that spelling. When the
    # spelling goes through a symlinked directory (uv's minor-version alias,
    # `cpython-3.13-…` → `cpython-3.13.18-…`), the sandbox has no symlink there,
    # only the directories bound under it, so the base prefix is bound at that
    # spelling as well as its own. Without it the interpreter cannot start:
    # "Failed to import encodings module".
    home = _venv_home(root) if root else None
    if home is not None and home.parent.exists() and home.parent.resolve() != home.parent \
            and not _refused(home.parent):
        wanted.add(home.parent)
    wanted |= {hop.parent for hop in _symlink_hops(launch)}
    found = [f for f in info["modules"].values() if f]
    homes = []
    for f in found:
        entry = max((e for e in info["path"]
                     if e and f.startswith(e.rstrip("/") + "/")),
                    key=len, default=None)
        if entry and entry not in homes:
            homes.append(entry)
    wanted |= {Path(h) for h in homes}
    wanted |= {Path(m).parent for m in info["maps"]}

    pairs = []
    for dest in wanted:
        dest = Path(os.path.normpath(dest))
        if not dest.exists():
            continue
        src = dest.resolve()
        pairs.append((src, dest))
        if src != dest:
            # Bound under both spellings: the real one for anything already
            # resolved, the given one for a symlink hop that names it.
            pairs.append((src, src))
    binds = _collapse(pairs)
    for _, dest in binds:
        why = _refused(Path(dest))
        if why:
            raise RuntimeError(
                f"refusing to bind {dest} into the Blender sandbox for "
                f"{python}: {why}. Install the interpreter's packages outside "
                "that directory (a venv, conda env, or user site all work).")
    sitedirs = tuple(h for h in homes if h not in isolated)
    return SandboxView(str(launch), binds, sitedirs)


def sandbox_view(python, modules=()) -> SandboxView:
    """What a bubblewrap sandbox needs so `python` can start and import `modules`.

    **Why the path alone does not answer this.** The obvious recipe — resolve
    the interpreter, bind its grandparent, exec the resolved path — is right
    for exactly one layout, a conda env, and wrong for every other common one:

    * a venv (python -m venv, virtualenv, uv, poetry, pipenv, hatch): the
      resolved path is the *base* interpreter, which starts without the venv's
      site-packages;
    * a `--copies` venv, or any venv whose base is not under `/usr`: the base
      interpreter's standard library is outside every bind and Python cannot
      start at all;
    * a user-site or `PYTHONPATH` install: the packages are neither under the
      interpreter's prefix nor visible to an isolated start.

    So the interpreter is asked. One probe imports `modules` and reports its
    invoked executable, its standard library, the `sys.path` entry each module
    came from, and every file the loader mapped; a second reports what an
    isolated start (`-I`) finds. The binds cover exactly those, `launch` is the
    venv's own `bin/python` with only its directory resolved, and `sitedirs`
    lists the import roots the isolated start misses.

    Cached per interpreter, module list and import-shaping environment: the
    probes cost a fraction of a second, and `command` asks on every worker
    spawn. Raises RuntimeError when the interpreter cannot import `modules`, or
    when covering them would bind `/`, `$HOME` or the repository.
    """
    env_key = tuple(os.environ.get(k) for k in
                    ("PYTHONPATH", "PYTHONUSERBASE", "PYTHONNOUSERSITE",
                     "PYTHONHOME", "VIRTUAL_ENV"))
    return _sandbox_view(str(python), tuple(modules), env_key)
