#!/usr/bin/env python3
"""Is this machine ready to run DyCraft? Answers, and says exactly why not.

    python environment_test.py                 # static checks: seconds, no network, no model calls
    python environment_test.py --deep          # + every subsystem exercised through its real code path
    python environment_test.py --e2e           # + the animation pipeline end to end, minus the model
    python environment_test.py --backend codex1 --model gpt-6-astra          # + live probes of a harness
    python environment_test.py --backend claude3 --model claude-opus-5-5 --capabilities
    python environment_test.py --harness-spec newcli.json --model M --capabilities
    python environment_test.py --write-env     # regenerate environment-main.yml + requirements-main.txt
    python environment_test.py --python PATH   # check another interpreter (re-execs under it)

Run it with the interpreter you launch DyCraft with. `--deep` and `--e2e` spend
minutes, not money; `--backend` spends a handful of model requests on the
account it names, and `--capabilities` a few more.

**The rule this is built on: no silent failures.** DyCraft degrades on purpose
in many places, so that one flaky dependency cannot wedge a five-hour run — a
missing renderer returns blank stills; `h.ik` reports `unavailable`; the bpy
wheel cannot bake fluid; the GPU falls back to CPU with a line on stderr. Each of
those is correct inside a run and dangerous before one, because a run that
completes looks the same as a run that was checked. So every graceful path in
the code is a *flag* here, never a pass, and every check proves itself against
a known answer — a render must contain the colour it was given, a harness must
return arithmetic that appears nowhere in its prompt — rather than merely not
raising.

Statuses:

    ✔ pass          works
    ✖ fail          a run fails, or produces results that mean nothing
    ◐ degraded      a run completes but silently loses checking or quality
    ⚠ warning       an optional feature is missing or misbehaving, or a hazard to
                    avoid (containment, a shadowing checkout); results are
                    unaffected
    ◌ unavailable   an optional capability is absent; the agent is told so
    – skipped       a prerequisite did not pass
    · info          a fact worth knowing, not a verdict

Exit code: 1 if anything failed; 2 with `--strict` if anything is degraded;
0 otherwise. Warnings and unavailable optional capabilities never change the
exit code, and each gets its own block at the end so neither can scroll past.

**Only the standard library at import**, so it runs in the broken environment
it is diagnosing. Every third-party import happens in a subprocess, so an import
that crashes the interpreter (OCP and open3d both can) is reported against the
package that did it. Where the code already answers a question — which
interpreter a run will pick, what the sandbox binds, how a backend finds its
account — this asks the code rather than re-deriving it, so the answer is what
a run will find.

**Adding a harness.** `--harness-spec FILE.json` describes a coding CLI that has
no `cli/backends/` class yet (see `_SPEC_FIELDS`), and `--capabilities` then
runs the same checklist the claude and codex backends pass, so a port's cost is
known before any backend code is written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _load_oom():
    """`_oom`, loaded by path: stdlib-only, so it works before `dycraft` imports.

    Every probe this spawns is priced as a disposable worker, so if a check pushes
    the machine out of memory the kernel takes the probe, not the session that
    launched it.
    """
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("_dycraft_oom", REPO / "_oom.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:   # noqa: BLE001 — a missing _oom.py must not stop the diagnosis
        return None


_oom = _load_oom()


def _load_home():
    """`_home`, loaded by path for the same reason as `_oom`: it says where user state lives."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_dycraft_home", REPO / "_home.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_home = _load_home()


#: True for an installed copy (under site-packages): the checks about a checkout (git, the environment files,
#: another checkout shadowing this one) do not apply to it.
INSTALLED = _home.installed()


def state_path(name: str) -> Path:
    """Where the runtime reads the pointer file `name`: $DYCRAFT_HOME, this checkout, or, for an installed
    package, the directory this runs from."""
    return _home.state_path(name)

# ===========================================================================
# the harness env's requirements -- the one place they are written down
# ===========================================================================


@dataclass(frozen=True)
class Req:
    dist: str
    module: str
    pin: str                 # the tested version; a mismatch is reported, not failed
    group: str               # "main" | "cad" | "viewer"
    rule: str = ""           # the only constraint the spec files carry: a version the code enforces ("==x", ">=x").
                             # Anything a dependency already declares (e.g. NumPy 2) is left to pip
    marker: str = ""         # PEP 508 marker for requirements-main.txt
    why: str = ""
    lost: str = ""


REQUIREMENTS = (
    Req("numpy", "numpy", "2.2.6", "main",
        why="geometry everywhere. build123d 0.11.1 and coal 3.x (through cmeel-boost) declare NumPy 2, "
            "so pip resolves it; this table adds no constraint of its own",
        lost="nothing in the harness imports"),
    Req("scipy", "scipy", "1.15.3", "main",
        why="motion solver consults (animation/blender/solvers.py) and the convex hulls "
            "bare-arm staging computes",
        lost="solver consults and bare-arm staging raise"),
    Req("pillow", "PIL", "12.1.0", "main",
        why="every render and snapshot",
        lost="renders cannot be written or read"),
    Req("open3d", "open3d", "0.19.0", "main",
        why="the default stills renderer (DYCRAFT_RENDER_BACKEND=open3d)",
        lost="snapshots fail on the default backend"),
    Req("huggingface_hub", "huggingface_hub", "0.36.2", "main",
        why="`python -m dycraft.install library` and `films`, which download the asset library and the paper's "
            "films from the Hub (imported only by those two steps)",
        lost="the asset library and the films cannot be downloaded"),
    Req("coal", "coal", "3.0.3", "main", ">=3.0.1",
        why="narrow-phase collision for the scene and animation layers, run from this "
            "interpreter inside the Blender sandbox (coal_worker.py refuses versions before 3.0.1)",
        lost="scene and animation runs raise on their first collision check"),
    Req("tomli", "tomli", "2.4.0", "main", marker='python_version < "3.11"',
        why="reads codex's config.toml on Python 3.10 (tomllib is stdlib from 3.11)",
        lost="the codex backend cannot resolve its model or start"),
    Req("build123d", "build123d", "0.11.1", "cad", "==0.11.1",
        why="authors every object; the CAD child refuses any other version",
        lost="object, scene, and animation runs refuse to start"),
    Req("fastapi", "fastapi", "0.129.0", "viewer",
        why="the results viewer (python -m dycraft.viewer)",
        lost="the viewer cannot start; results can still be generated"),
    Req("uvicorn", "uvicorn", "0.41.0", "viewer",
        why="serves the viewer", lost="the viewer cannot start"),
    Req("websockets", "websockets", "16.0", "viewer",
        why="uvicorn's WebSocket transport, which the viewer streams frames over",
        lost="the viewer starts but cannot stream animations"),
)

_GENERATED = ("# GENERATED by `python environment_test.py --write-env` from its REQUIREMENTS\n"
              "# table. Edit the table, not this file: the checker flags any drift.\n")


def _pins(groups, *, markers: bool) -> list[str]:
    lines = []
    for req in REQUIREMENTS:
        if req.group in groups:
            # As loose as works: the tested version is what the check reports against, not a requirement.
            line = f"{req.dist}{req.rule}"
            if markers and req.marker:
                line += f"; {req.marker}"
            lines.append(line)
    return lines


def render_environment_yml() -> str:
    main = "\n".join(f"      - {p}" for p in _pins(("main", "cad"), markers=False))
    viewer = "\n".join(f"      - {p}" for p in _pins(("viewer",), markers=False))
    return f"""{_GENERATED}#
# The harness env: launcher, daemon, verifiers, renderer, the collision child
# the Blender sandbox calls back into, and the build123d CAD interpreter. Blender
# is NOT here: bpy 4.5 needs Python 3.11 and NumPy < 2, so it lives in its own env
# beside the official 4.5 LTS build (python -m dycraft.install blender).
#
#   conda env create -f environment-main.yml -n dycraft && conda activate dycraft
#   pip install -e .                      # from the clone (or: pip install .)
#   python -m dycraft.install cad         # this env carries build123d 0.11.1
#   python -m dycraft.install check --deep
#
# Every package is pip-installed; conda contributes only the interpreter. Only
# the versions the code enforces are constrained (build123d exactly, coal from
# 3.0.1); everything else is left to pip, and `python -m dycraft.install check`
# reports where an installed version differs from the one validated. No conda?
# requirements-main.txt is the same set for a venv, uv or a system Python.

name: dycraft

channels:
  - conda-forge
  - nodefaults

dependencies:
  # Validated on 3.10, 3.13 and 3.14. 3.15 has no open3d, coal or OCP wheels yet.
  - python>=3.10,<3.15
  - pip
  - pip:
{main}
      # The results viewer only; nothing that generates a result needs these.
{viewer}
"""


def render_requirements_txt() -> str:
    main = "\n".join(_pins(("main", "cad"), markers=True))
    viewer = "\n".join(_pins(("viewer",), markers=True))
    return f"""{_GENERATED}#
# The harness env for a venv, uv or system Python (3.10-3.14):
#
#   python3 -m venv .venv && .venv/bin/pip install -e .     # reads this file
#   uv venv && uv pip install -r requirements-main.txt
#   .venv/bin/python -m dycraft.install cad
#   .venv/bin/python -m dycraft.install check --deep
#
# Blender is separate (environment-blender.yml, or a blender.org build named in
# .blender_binary). With conda, environment-main.yml is the same set.
{main}
# The results viewer only; nothing that generates a result needs these.
{viewer}
"""


ENV_FILES = {"environment-main.yml": render_environment_yml,
             "requirements-main.txt": render_requirements_txt}

# ===========================================================================
# results, checks, and the runner
# ===========================================================================

PASS, FAIL, DEGRADED, WARN, UNAVAILABLE, SKIP, INFO = (
    "pass", "fail", "degraded", "warning", "unavailable", "skipped", "info")
SYMBOL = {PASS: "✔", FAIL: "✖", DEGRADED: "◐", WARN: "⚠", UNAVAILABLE: "◌", SKIP: "–",
          INFO: "·"}
SEVERITY = {PASS: 0, INFO: 0, WARN: 1, SKIP: 2, UNAVAILABLE: 3, DEGRADED: 4, FAIL: 5}

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_TINT = {PASS: "32", FAIL: "31", DEGRADED: "35", WARN: "33", UNAVAILABLE: "36", SKIP: "90",
         INFO: "90"}


def tint(text: str, status: str) -> str:
    return f"\033[{_TINT[status]}m{text}\033[0m" if COLOR else text


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if COLOR else text


@dataclass
class Result:
    status: str
    summary: str
    detail: list = field(default_factory=list)
    fix: list = field(default_factory=list)
    lost: str = ""
    data: dict = field(default_factory=dict)


def ok(summary, **kw):
    return Result(PASS, summary, **kw)


def fail(summary, **kw):
    return Result(FAIL, summary, **kw)


def degraded(summary, **kw):
    return Result(DEGRADED, summary, **kw)


def warn(summary, **kw):
    """An optional feature is missing or misbehaving, or a hazard to avoid: worth
    fixing, but no result is lost or made less trustworthy by it."""
    return Result(WARN, summary, **kw)


def unavailable(summary, **kw):
    return Result(UNAVAILABLE, summary, **kw)


def info(summary, **kw):
    return Result(INFO, summary, **kw)


# Capabilities: what the readiness table at the end is organized around.
CAP_DYCRAFT = "dycraft animation arm"
CAP_BARE = "bare animation arm"
CAP_STAGING = "bare-arm staging"
CAP_VIEWER = "results viewer"
CAP_GPU = "GPU rendering"
CAP_FLUID = "fluid simulation"
CAP_IK = "IK solving (cuRobo)"
CAP_MOCAP = "motion capture"
OPTIONAL_CAPS = (CAP_VIEWER, CAP_GPU, CAP_FLUID, CAP_IK, CAP_MOCAP)


@dataclass
class Check:
    id: str
    title: str
    section: str
    fn: object
    tier: str = "static"         # static | deep | e2e | backend
    needs: tuple = ()
    caps: tuple = ()


CHECKS: list[Check] = []


def check(id, title, section, *, tier="static", needs=(), caps=()):
    def register(fn):
        CHECKS.append(Check(id, title, section, fn, tier, tuple(needs), tuple(caps)))
        return fn
    return register


class Ctx:
    def __init__(self, args):
        self.args = args
        self.tmp = Path(tempfile.mkdtemp(prefix="dycraft_envtest_"))
        #: Scratch kept elsewhere than `tmp`, removed with it.
        self.cleanup: list[Path] = []
        self.results: dict[str, Result] = {}
        self.checks: dict[str, Check] = {}
        self.facts: dict = {}
        self.interactive = (not args.non_interactive and sys.stdin.isatty()
                            and sys.stdout.isatty())
        self.pkgroot: Path | None = None

    def child_env(self, **extra) -> dict:
        """The environment a DyCraft child process gets: this copy first on
        PYTHONPATH, so a child imports the code under check and not another copy."""
        env = dict(os.environ)
        if self.pkgroot:
            env["PYTHONPATH"] = os.pathsep.join(
                [str(self.pkgroot), *filter(None, [env.get("PYTHONPATH")])])
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def passed(self, check_id: str) -> bool:
        r = self.results.get(check_id)
        return r is not None and r.status in (PASS, INFO, WARN, DEGRADED)


def run_checks(ctx: Ctx, checks: list[Check]) -> None:
    section = None
    for c in checks:
        ctx.checks[c.id] = c
        if c.section != section:
            section = c.section
            print(f"\n{bold(section)}")
        blocked = [n for n in c.needs if n in ctx.results and not ctx.passed(n)]
        if blocked:
            names = ", ".join(ctx.checks[n].title for n in blocked)
            result = Result(SKIP, f"needs: {names}")
        else:
            if sys.stdout.isatty():
                print(f"  … {c.title}", end="\r", flush=True)
            started = time.monotonic()
            try:
                result = c.fn(ctx)
            except KeyboardInterrupt:
                raise
            except ModuleNotFoundError as exc:
                # The DyCraft code this check asks could not load; say which package, not "bug".
                top = (exc.name or "").split(".")[0]
                req = next((r for r in REQUIREMENTS if r.module == top), None)
                result = (Result(SKIP, f"needs {req.dist}: the code this check asks imports it")
                          if req else
                          fail(f"the code this check asks does not import: {exc}",
                               detail=traceback.format_exc().rstrip().splitlines()[-4:]))
            except Exception:
                result = fail("the check itself crashed — a bug in environment_test.py",
                              detail=traceback.format_exc().rstrip().splitlines()[-8:])
            result.data.setdefault("seconds", round(time.monotonic() - started, 2))
        ctx.results[c.id] = result
        print_result(c, result)


#: The checks a launch depends on, run by `python -m dycraft.cli` (`launch_preflight`)
#: before it stages anything or starts a rewriter, so a broken install is refused
#: with this file's own message and fix before any agent spends usage. Every launch
#: needs the `None` entry; each layer adds its own, and each backend family its
#: harness check. Only static checks: they take a few seconds together.
LAUNCH_CHECKS = {
    None: ("host.python", "repo.import", "repo.shadow"),
    "object": ("cad.interpreter", "render.stills"),
    "scene": ("cad.interpreter", "render.stills"),
    # DyCraft's own sandbox (bwrap, user namespaces) confines the animation layers' Blender workers only.
    "animation": ("cad.interpreter", "render.stills", "blender.animation", "blender.bpy", "tools.ffmpeg",
                  "sandbox.bwrap", "sandbox.userns"),
    "plain": (),
}
LAUNCH_HARNESS = {"claude": "harness.claude", "codex": "harness.codex"}

#: A degraded result that still refuses a launch: Claude Code's sandbox cannot
#: start. (A codex whose sandbox does not work is the codex backend's own
#: preflight to refuse.) With `--allow-unsandboxed`, no result about a sandbox
#: that cannot start (`data["sandbox"]`: the CLI's, or Blender's where it has
#: neither bubblewrap nor Landlock) refuses: the launch goes on without it, and
#: says so.
LAUNCH_REFUSING_DEGRADED = {"claude": ("harness.claude",)}


def launch_preflight(layer: str, family: str, progress=None) -> list:
    """Run the checks a `layer` launch on a `family` backend depends on, quietly; return
    `[(Check, Result), ...]` in check order. `progress(title)` is called as each starts."""
    wanted = [*LAUNCH_CHECKS[None], *LAUNCH_CHECKS.get(layer, ())]
    if family in LAUNCH_HARNESS:
        wanted.append(LAUNCH_HARNESS[family])
    by_id = {c.id: c for c in CHECKS}
    ids, stack = set(), list(wanted)
    while stack:                                   # and everything they need, so `needs` gates as in a full run
        cid = stack.pop()
        if cid in by_id and cid not in ids:
            ids.add(cid)
            stack.extend(by_id[cid].needs)
    ctx = Ctx(parse_args(["--non-interactive"]))
    out = []
    try:
        for c in [c for c in CHECKS if c.id in ids]:
            ctx.checks[c.id] = c
            blocked = [n for n in c.needs if n in ctx.results and not ctx.passed(n)]
            if blocked:
                result = Result(SKIP, "needs: " + ", ".join(ctx.checks[n].title for n in blocked))
            else:
                if progress:
                    progress(c.title)
                try:
                    result = c.fn(ctx)
                except KeyboardInterrupt:
                    raise
                except Exception:                  # noqa: BLE001
                    result = fail("the check itself crashed — a bug in environment_test.py",
                                  detail=traceback.format_exc().rstrip().splitlines()[-4:])
            ctx.results[c.id] = result
            out.append((c, result))
    finally:
        for path in [ctx.tmp, *ctx.cleanup]:
            shutil.rmtree(path, ignore_errors=True)
    return out


def launch_refusals(results: list, family: str) -> list:
    """The `(Check, Result)` pairs of `launch_preflight` that refuse the launch."""
    from dycraft.cli.backends import allow_unsandboxed
    degraded_too = LAUNCH_REFUSING_DEGRADED.get(family, ())
    opted_out = allow_unsandboxed()
    return [(c, r) for c, r in results
            if (r.status in (FAIL, UNAVAILABLE) or (r.status == DEGRADED and c.id in degraded_too
                                                    and not r.data.get("login")))
            and not (opted_out and r.data.get("sandbox"))]


def print_result(c: Check, r: Result) -> None:
    head = f"  {tint(SYMBOL[r.status], r.status)} {c.title:<30} {r.summary}"
    print(("\033[2K" if sys.stdout.isatty() else "") + head)
    if r.status in (FAIL, DEGRADED, WARN, UNAVAILABLE) and r.lost:
        print(_wrap(f"lost: {r.lost}", "      ", "            "))
    for line in r.detail:
        print(_wrap(str(line), "      ", "        "))
    if r.status in (FAIL, DEGRADED, WARN, UNAVAILABLE):
        for line in r.fix:
            print(_wrap(f"→ {line}", "      ", "        "))


def _wrap(text: str, first: str, rest: str) -> str:
    width = max(60, shutil.get_terminal_size((110, 20)).columns - 2)
    out = []
    for i, para in enumerate(str(text).splitlines() or [""]):
        out.append(textwrap.fill(para, width, initial_indent=first if i == 0 else rest,
                                 subsequent_indent=rest, break_on_hyphens=False,
                                 break_long_words=False) or first)
    return "\n".join(out)


# ===========================================================================
# processes
# ===========================================================================


@dataclass
class Ran:
    rc: int | None
    out: str
    err: str
    timed_out: bool = False
    error: str = ""
    seconds: float = 0.0
    watched: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return (self.out or "") + "\n" + (self.err or "")

    def tail(self, n: int = 12) -> list[str]:
        lines = [l for l in self.text.splitlines() if l.strip()]
        return lines[-n:]


def run(argv, *, timeout=120, env=None, cwd=None, watch=None) -> Ran:
    """Run `argv` to completion or `timeout`; never raises.

    Started in its own session so a timeout takes the whole process tree (a CLI
    that forks a worker, a Blender that forks a coal child) and not just the
    parent. `watch`, if given, is called every 0.2 s with the Popen while the
    process runs, and its last non-None return is kept in `Ran.watched`.
    """
    started = time.monotonic()
    argv = [str(a) for a in argv]
    if _oom is not None:
        argv = _oom.tiered(argv, _oom.WORKER)
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                env=env, cwd=cwd, text=True, errors="replace",
                                start_new_session=True)
    except OSError as exc:
        return Ran(None, "", "", error=f"{type(exc).__name__}: {exc}")
    watched: dict = {}
    if watch is not None:
        def poll():
            while proc.poll() is None:
                try:
                    got = watch(proc)
                except Exception:           # noqa: BLE001 — a watcher must never kill the run
                    got = None
                if got:
                    watched.update(got)
                time.sleep(0.2)
        threading.Thread(target=poll, daemon=True).start()
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate()
    return Ran(proc.returncode, out, err, timed_out=timed_out,
               seconds=round(time.monotonic() - started, 1), watched=watched)


def marked_json(ran: Ran, marker: str = "ENVTEST_JSON "):
    """The payload of the last `marker` line a child printed, or None."""
    for line in reversed((ran.out or "").splitlines()):
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
            except ValueError:
                return None
    return None


def trailing_json(text: str):
    """The JSON document a child printed last, single-line or indented."""
    lines = (text or "").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("{"):
            try:
                return json.loads("\n".join(lines[i:]))
            except ValueError:
                continue
    return None


def which(name: str) -> str | None:
    return shutil.which(name)


def version_of(binary, *flags, timeout=30) -> str:
    ran = run([binary, *(flags or ("--version",))], timeout=timeout)
    lines = [l.strip() for l in ran.text.splitlines() if l.strip()]
    return lines[0] if lines else "?"


# Failure texts a CLI or provider prints, most specific first. Shared by every
# live probe so a usage limit reads as a usage limit and not as "no answer".
_FAILURE_PATTERNS = (
    ("usage limit", r"usage.?limit|rate.?limit|limit (reached|exceeded)|quota|credits? "
                    r"(depleted|exhausted)|workspace_member_credits_depleted|too many requests|"
                    r"\b429\b|try again (at|in)|resets? (at|in)"),
    ("not logged in", r"not logged in|please (log ?in|login|run .*login)|/login|unauthori[sz]ed|"
                      r"\b401\b|invalid (api )?key|authenticat|expired (token|session)"),
    ("model unavailable", r"model[^\n]{0,60}(not found|does not exist|not available|unsupported|"
                          r"not supported|unknown|invalid|no access)|unknown model|model_not_found"),
    ("sandbox unavailable", r"sandbox[^\n]{0,80}(unavailable|not available|failed|required|"
                            r"error)|bubblewrap|\bbwrap\b|socat|landlock|seccomp"),
    ("untrusted directory", r"not inside a trusted directory|trust (this|the) (folder|directory)|"
                            r"skip-git-repo-check"),
    ("network", r"could not resolve|connection (refused|reset|error)|getaddrinfo|ENOTFOUND|"
                r"ECONNREFUSED|network is unreachable"),
)


def classify(text: str) -> tuple[str, str]:
    # A launcher echoes its own argv, whose settings JSON names the sandbox
    # ("failIfUnavailable"); matching that would label every failure a sandbox one.
    text = "\n".join(line for line in (text or "").splitlines() if "argv=" not in line)
    for label, pattern in _FAILURE_PATTERNS:
        m = re.search(pattern, text or "", re.I)
        if m:
            line = next((l.strip() for l in text.splitlines() if m.group(0) in l), m.group(0))
            return label, line[:300]
    return "", ""


# ===========================================================================
# images with known contents (stdlib only)
# ===========================================================================

_DIGITS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
}


def write_png(path: Path, width: int, height: int, pixel) -> Path:
    """An 8-bit RGB PNG whose pixel (x, y) is `pixel(x, y)`."""
    rows = []
    for y in range(height):
        row = bytearray(b"\x00")
        for x in range(width):
            row += bytes(pixel(x, y))
        rows.append(bytes(row))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
                     + chunk(b"IEND", b""))
    return path


def digits_png(text: str, path: Path, scale: int = 14, pad: int = 2) -> Path:
    """`text` (digits) drawn large, black on white: legible to any vision model."""
    cols = len(text) * 6 - 1 + 2 * pad
    grid = [[0] * cols for _ in range(7 + 2 * pad)]
    for i, ch in enumerate(text):
        for r, bits in enumerate(_DIGITS[ch]):
            for c, bit in enumerate(bits):
                if bit == "1":
                    grid[pad + r][pad + i * 6 + c] = 1
    return write_png(path, cols * scale, len(grid) * scale,
                     lambda x, y: (20, 20, 20) if grid[y // scale][x // scale] else (255, 255, 255))


def random_digits(n: int = 6) -> str:
    return str(random.randint(1, 9)) + "".join(str(random.randint(0, 9)) for _ in range(n - 1))


def arithmetic() -> tuple[str, str]:
    """A question whose answer appears nowhere in the prompt, so an echo cannot pass."""
    a, b = random.randint(1000, 4999), random.randint(1000, 4999)
    return (f"What is {a} + {b}? Reply with only the resulting number, digits only.", str(a + b))


# ===========================================================================
# this interpreter, and how to install into it
# ===========================================================================


def describe_python() -> dict:
    prefix, base = Path(sys.prefix), Path(sys.base_prefix)
    cfg = prefix / "pyvenv.cfg"
    d = {"executable": sys.executable, "version": platform.python_version(),
         "prefix": str(prefix), "base_prefix": str(base), "kind": "system", "manager": ""}
    if (prefix / "conda-meta").is_dir():
        d["kind"] = "conda"
        d["manager"] = ("pixi" if "/.pixi/" in str(prefix) else
                        "micromamba" if "micromamba" in str(prefix) else "conda/mamba")
    elif prefix != base or cfg.is_file():
        d["kind"] = "venv"
        text = cfg.read_text(errors="replace") if cfg.is_file() else ""
        d["manager"] = ("uv" if re.search(r"(?m)^uv\s*=", text) else
                        "virtualenv" if re.search(r"(?m)^virtualenv\s*=", text) else
                        "poetry" if "pypoetry" in str(prefix) else
                        "pipenv" if "/.local/share/virtualenvs/" in str(prefix) else
                        "hatch" if "/hatch/" in str(prefix) else "venv")
        d["copies"] = not Path(sys.executable).is_symlink()
        d["base_interpreter"] = str(Path(sys.executable).resolve())
    elif "/.pyenv/versions/" in str(prefix):
        d["kind"] = "pyenv"
    elif str(prefix).startswith("/nix/store"):
        d["kind"] = "nix"
    elif str(prefix).startswith(("/opt/homebrew", "/home/linuxbrew")):
        d["kind"] = "homebrew"
    import site
    import sysconfig
    d["user_site"] = bool(site.ENABLE_USER_SITE and site.getusersitepackages() in sys.path)
    d["externally_managed"] = (d["kind"] == "system" and
                               (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").is_file())
    import importlib.util
    d["pip"] = importlib.util.find_spec("pip") is not None
    return d


def install_cmd(ctx: Ctx, *specs: str) -> str:
    """How to install `specs` into *this* interpreter, whatever manages it."""
    d = ctx.facts.get("python") or describe_python()
    quoted = " ".join(f"'{s}'" for s in specs)
    if d.get("externally_managed"):
        return (f"this system Python is externally managed (PEP 668): make a venv first — "
                f"python3 -m venv ~/.venvs/dycraft && ~/.venvs/dycraft/bin/pip install -r "
                f"{REPO / 'requirements-main.txt'}")
    if not d.get("pip") and which("uv"):
        return f"uv pip install --python {sys.executable} {quoted}"
    if not d.get("pip"):
        return f"{sys.executable} -m ensurepip && {sys.executable} -m pip install {quoted}"
    return f"{sys.executable} -m pip install {quoted}"


def install_all_cmd(ctx: Ctx) -> str | None:
    """How to install every package this checkout declares into *this* interpreter, in one command; None for an
    installed package, which has no checkout to install from."""
    if INSTALLED:
        return None
    d = ctx.facts.get("python") or describe_python()
    if d.get("externally_managed"):
        return install_cmd(ctx)
    if not d.get("pip") and which("uv"):
        return f"uv pip install --python {sys.executable} -e {REPO}"
    if not d.get("pip"):
        return f"{sys.executable} -m ensurepip && {sys.executable} -m pip install -e {REPO}"
    return f"{sys.executable} -m pip install -e {REPO}"


# ===========================================================================
# finding interpreters and Blenders, and the pointer-file dialog
# ===========================================================================

_BPY_CODE = ("import json, sys\nimport bpy\ntry:\n    import numpy\n    np = numpy.__version__\n"
             "except Exception:\n    np = None\nprint('ENVTEST_JSON ' + json.dumps({'version': "
             "bpy.app.version_string, 'python': sys.version.split()[0], 'numpy': np}))\n")

_CAD_CODE = r'''
import json, sys
import importlib.metadata as md
out = {"python": sys.version.split()[0]}
try:
    out["version"] = md.version("build123d")
    import build123d, OCP, pyexpat
    for dist in ("cadquery-ocp-novtk", "cadquery-ocp"):
        try:
            out["ocp"] = md.version(dist); break
        except md.PackageNotFoundError:
            pass
    out["ok"] = True
except BaseException as exc:
    out["ok"] = False
    out["error"] = f"{type(exc).__name__}: {exc}"[:600]
print("ENVTEST_JSON " + json.dumps(out))
'''


def probe_bpy_python(python: str) -> dict:
    ran = run([python, "-c", _BPY_CODE], timeout=180, env={**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1"})
    got = marked_json(ran)
    if got:
        return {"ok": True, **got}
    return {"ok": False, "error": (ran.tail(1) or [ran.error or "no output"])[0]}


def probe_blender_binary(binary: str) -> dict:
    ran = run([binary, "--background", "--factory-startup", "--python-exit-code", "1",
               "--python-expr", _BPY_CODE], timeout=240)
    got = marked_json(ran)
    if got:
        return {"ok": True, **got}
    return {"ok": False, "error": (ran.tail(1) or [ran.error or "no output"])[0]}


def probe_cad_python(python: str) -> dict:
    env = dict(os.environ)
    lib = Path(python).resolve().parent.parent / "lib"
    if lib.is_dir():   # what `_cad_env.b123d_child_env` does for every CAD child
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(lib), env.get("LD_LIBRARY_PATH")]))
    ran = run([python, "-c", _CAD_CODE], timeout=180, env=env)
    return marked_json(ran) or {"ok": False, "error": (ran.tail(1) or [ran.error or "no output"])[0]}


def blender_packaging_problem(binary: str) -> str:
    real = str(Path(binary).resolve())
    if "/snap/" in binary or Path(real).name == "snap":
        return "a snap package, whose confinement cannot run inside bubblewrap"
    if "flatpak" in binary or "flatpak" in real:
        return "a flatpak, which cannot run inside bubblewrap"
    return ""


def python_candidates() -> list[str]:
    """Interpreters worth asking, from every layout this might run on."""
    home = Path.home()
    seen: dict[str, None] = {}
    keys: set = set()

    def add(p):
        if not p:
            return
        p = Path(str(p)).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            venv = (p.parent.parent / "pyvenv.cfg").is_file()
            key = str(p) if venv else os.path.realpath(p)
            if key not in keys:
                keys.add(key)
                seen.setdefault(str(p), None)

    add(sys.executable)
    for pointer in (".b123d_python", ".blender_python"):
        add(read_pointer(REPO / pointer))
    conda_exe = os.environ.get("CONDA_EXE")
    roots = [os.environ.get("CONDA_PREFIX"), os.environ.get("MAMBA_ROOT_PREFIX"),
             Path(conda_exe).parents[1] if conda_exe else None,
             home / "miniconda3", home / "anaconda3", home / ".miniconda3", home / "miniforge3",
             home / "mambaforge", home / "micromamba", home / ".conda", "/opt/conda"]
    for root in filter(None, roots):
        root = Path(root)
        add(root / "bin" / "python")
        for env in sorted((root / "envs").glob("*")) if (root / "envs").is_dir() else ():
            add(env / "bin" / "python")
    envs_txt = home / ".conda" / "environments.txt"
    if envs_txt.is_file():
        for line in envs_txt.read_text(errors="replace").splitlines():
            add(Path(line.strip()) / "bin" / "python")
    for base in (home, REPO.parent, REPO.parent.parent):
        for env in sorted((base / "conda_envs").glob("*")) if (base / "conda_envs").is_dir() else ():
            add(env / "bin" / "python")
    for venv in (REPO / ".venv", REPO / "venv"):
        add(venv / "bin" / "python")
    for parent in (home / ".venvs", home / "venvs", home / ".virtualenvs",
                   home / ".local" / "share" / "virtualenvs", home / ".pyenv" / "versions",
                   home / ".local" / "share" / "uv" / "python"):
        for env in sorted(parent.glob("*")) if parent.is_dir() else ():
            add(env / "bin" / "python")
            add(env / "bin" / "python3")
    for name in ("python3.10", "python3.11", "python3.12", "python3", "python"):
        add(which(name))
    return list(seen)


def blender_candidates() -> list[str]:
    home = Path.home()
    found: dict[str, None] = {}
    for p in [which("blender"), "/snap/bin/blender",
              *Path("/opt").glob("blender*/blender"), *Path("/opt").glob("*/blender"),
              *Path("/usr/local").glob("blender*/blender"), *home.glob("blender*/blender"),
              *home.glob("blender/*/blender"), *home.glob("Applications/blender*/blender"),
              home / ".local" / "bin" / "blender"]:
        if p and Path(p).is_file() and os.access(p, os.X_OK):
            found.setdefault(str(Path(p).resolve()), None)
    return list(found)


def read_pointer(path: Path) -> str | None:
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        return None
    return None


def _scan(paths, probe) -> list[tuple[str, dict]]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(zip(paths, pool.map(probe, paths)))


_POINTERS = {
    # name: (consumer, comment lines allowed)
    ".b123d_python": ("agent/cad/_cad_env.py", True),
    ".blender_python": ("render/_blender_env.py (bpy module: bare arms, staging, stills)", True),
    # `_blender_env.blender_binary` reads the whole file as the path.
    ".blender_binary": ("render/_blender_env.py (the animation layer's Blender)", False),
}


def write_pointer(ctx: Ctx, name: str, value: str) -> bool:
    path = state_path(name)
    consumer, commented = _POINTERS[name]
    if path.exists():
        print(f"      {name} currently contains:")
        for line in path.read_text(errors="replace").splitlines()[:6]:
            print(f"        | {line}")
        if not ask(f"      Replace it with {value}? [y/N] ", default="n").startswith("y"):
            return False
    header = (f"# written by environment_test.py on {date.today().isoformat()}; "
              f"read by {consumer}\n") if commented else ""
    path.write_text(header + value + "\n")
    print(f"      wrote {path}")
    return True


def ask(prompt: str, default: str = "") -> str:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return default
    return answer or default


def pointer_dialog(ctx: Ctx, *, want: str, searched: list[str],
                   kinds: tuple) -> bool:
    """Find candidates for `kinds` ("bpy", "binary", "cad"), let the user pick one,
    and write the pointer file it belongs in. Returns whether one was written."""
    if not ctx.interactive:
        return False
    print(f"\n      {bold(want)}")
    for line in searched:
        print(_wrap(f"searched: {line}", "        ", "          "))
    print("        scanning this machine for candidates …", flush=True)
    rows = []   # (path, pointer, verdict_ok, label)
    pythons = python_candidates() if ("bpy" in kinds or "cad" in kinds) else []
    if "binary" in kinds:
        for path, got in _scan(blender_candidates(), probe_blender_binary):
            problem = blender_packaging_problem(path)
            good = got.get("ok") and not problem
            label = (f"Blender {got.get('version')} binary" if got.get("ok") else got.get("error", ""))
            rows.append((path, ".blender_binary", good, label + (f" — {problem}" if problem else "")))
    if "bpy" in kinds:
        for path, got in _scan(pythons, probe_bpy_python):
            label = (f"bpy {got.get('version')} (Python {got.get('python')}, NumPy {got.get('numpy')})"
                     if got.get("ok") else "no bpy")
            rows.append((path, ".blender_python", bool(got.get("ok")), label))
    if "cad" in kinds:
        for path, got in _scan(pythons, probe_cad_python):
            good = got.get("ok") and got.get("version") == "0.11.1"
            label = (f"build123d {got.get('version')}" if got.get("version") else "no build123d")
            if got.get("version") and not got.get("ok"):
                label += f" (broken: {got.get('error', '')[:80]})"
            rows.append((path, ".b123d_python", bool(good), label))
    usable = [r for r in rows if r[2]]
    for n, (path, pointer, _, label) in enumerate(usable, 1):
        print(f"        {n}) ✔ {path}\n             {label}  → {pointer}")
    rejected = [r for r in rows if not r[2]]
    if rejected:
        print(_wrap(f"({len(rejected)} other candidate(s) rejected, e.g. "
                    + "; ".join(f"{p}: {l}" for p, _, _, l in rejected[:3]) + ")",
                    "        ", "          "))
    pointer_for_path = kinds[0] if len(kinds) == 1 else None
    while True:
        choices = f"1-{len(usable)} to use, " if usable else ""
        answer = ask(f"      {choices}p to enter a path, s to skip: ", default="s")
        if answer == "s":
            return False
        if answer.isdigit() and 1 <= int(answer) <= len(usable):
            path, pointer, _, _ = usable[int(answer) - 1]
            return write_pointer(ctx, pointer, path)
        if answer == "p":
            path = input("      path: ").strip()
            if not path:
                continue
            path = str(Path(path).expanduser())
            kind = pointer_for_path or ("binary" if Path(path).name.startswith("blender") else
                                        "bpy" if "bpy" in kinds else "cad")
            probe = {"binary": probe_blender_binary, "bpy": probe_bpy_python,
                     "cad": probe_cad_python}[kind]
            got = probe(path)
            good = got.get("ok") and (kind != "cad" or got.get("version") == "0.11.1")
            if not good:
                print(f"      ✖ {path} is not usable: {got.get('error') or got.get('version')}")
                continue
            pointer = {"binary": ".blender_binary", "bpy": ".blender_python",
                       "cad": ".b123d_python"}[kind]
            return write_pointer(ctx, pointer, path)


def with_dialog(ctx: Ctx, result: Result, recheck, **dialog) -> Result:
    """`result`, or — when it failed, a person is present and a pointer file got
    written — the check run again against what they chose."""
    if result.status == FAIL and pointer_dialog(ctx, **dialog):
        return recheck(ctx)
    if result.status == FAIL and not ctx.interactive:
        result.fix.append("or run `python -m dycraft.install check` in a terminal: it asks which interpreter to "
                          "use and writes the pointer file for you")
    return result


# ===========================================================================
# static checks
# ===========================================================================

S_HOST, S_REPO, S_PKG, S_CAD, S_BLENDER = ("Host", "Checkout", "Python packages",
                                           "CAD interpreter (build123d)", "Blender")
S_SANDBOX, S_RENDER, S_HARNESS = "Sandbox", "Rendering and media", "Agent harnesses"
S_BARE, S_OPTIONAL = "Bare-agent comparison (benchmark/)", "Optional capabilities"


@check("host.platform", "platform", S_HOST, caps=(CAP_DYCRAFT, CAP_BARE))
def check_platform(ctx):
    if not sys.platform.startswith("linux"):
        return fail(f"{platform.system()} {platform.release()} — DyCraft runs on Linux only",
                    lost="Blender is sandboxed with bubblewrap and agents are found through "
                         "/proc; neither exists here",
                    fix=["use a Linux host, VM, or WSL2"])
    name = "Linux"
    try:
        text = Path("/etc/os-release").read_text()
        name = re.search(r'(?m)^PRETTY_NAME="?([^"\n]+)', text).group(1)
    except (OSError, AttributeError):
        pass
    return ok(f"{name}, kernel {platform.release()}, {platform.machine()}")


@check("host.python", "python", S_HOST, caps=(CAP_DYCRAFT,))
def check_python(ctx):
    v = sys.version_info
    if v < (3, 10):
        return fail(f"Python {platform.python_version()} — DyCraft needs 3.10 or newer",
                    fix=["create an env from `environment-main.yml` or `requirements-main.txt`"])
    note = "" if v[:2] == (3, 10) else "  (the validated harness env is 3.10)"
    return ok(f"Python {platform.python_version()}{note}")


@check("host.env", "environment", S_HOST)
def check_env_kind(ctx):
    d = ctx.facts["python"] = describe_python()
    kind = d["kind"] + (f" ({d['manager']})" if d["manager"] else "")
    detail = [f"interpreter: {d['executable']}"]
    if d["kind"] == "venv":
        detail.append(f"base interpreter: {d['base_interpreter']} "
                      f"({'copied' if d['copies'] else 'symlinked'})")
    if d["user_site"]:
        detail.append("user site-packages are on sys.path")
    if d["externally_managed"]:
        detail.append("externally managed (PEP 668): install packages into a venv, not here")
    if not d["pip"]:
        detail.append("no pip in this environment" + (" (uv is available)" if which("uv") else ""))
    detail.append(f"packages are installed with: {install_cmd(ctx, '<package>')}")
    return info(f"{kind}: {d['prefix']}", detail=detail)


@check("repo.import", "imports this checkout", S_REPO, caps=(CAP_DYCRAFT, CAP_STAGING))
def check_import(ctx):
    root = ctx.tmp / "pkgroot"
    root.mkdir(exist_ok=True)
    link = root / "dycraft"
    if not link.exists():
        link.symlink_to(REPO)
    sys.path.insert(0, str(root))
    ctx.pkgroot = root
    try:
        import dycraft   # noqa: F401
    except Exception as exc:
        return fail(f"`import dycraft` raises {type(exc).__name__}: {exc}",
                    detail=traceback.format_exc().rstrip().splitlines()[-6:],
                    lost="nothing in the harness can run; every check that asks the code is skipped")
    where = Path(sys.modules["dycraft"].__file__).resolve().parent
    if where != REPO:
        return fail(f"`import dycraft` resolved to {where}, not this checkout",
                    lost="every check below would describe a different checkout")
    return ok(f"DyCraft imports from {REPO}")


_WHERE = "import dycraft, json; print('ENVTEST_JSON ' + json.dumps(dycraft.__file__))"


#: What `install` leaves in DyCraft's state directory: one of these says a directory is one.
_STATE_MARKERS = (".blender_python", ".blender_binary", ".b123d_python", ".tools_bin", ".gl_prefix", "envs", "runs",
                  "assets")


@check("repo.state", "state directory", S_REPO, needs=("repo.import",), caps=(CAP_DYCRAFT, CAP_STAGING))
def check_state_dir(ctx):
    """Where pointer files, environments, assets and runs are read and written (`_home.home`), and why there: an
    installed package uses the current directory, so a launch from elsewhere finds none of what `install` set up."""
    from dycraft import _home
    root = _home.home()
    if os.environ.get(_home.ENV_VAR):
        return ok(f"{root} (DYCRAFT_HOME)")
    if not _home.installed():
        return ok(f"{root} (this checkout)")
    if any((root / name).exists() for name in _STATE_MARKERS):
        return ok(f"{root} (the current directory, as for an installed package)")
    return warn(f"{root}: the current directory, which holds no DyCraft state (no pointer files, envs or runs)",
                lost="the interpreters, tools and runs `python -m dycraft.install` set up elsewhere are not found "
                     "from here, and new ones would be made here",
                fix=["run DyCraft from the directory you installed it from",
                     "or set DYCRAFT_HOME to that directory, for example in your shell's profile"])


@check("repo.shadow", "no other checkout shadows it", S_REPO, needs=("repo.import",),
       caps=(CAP_DYCRAFT,))
def check_shadow(ctx):
    """What `import dycraft` actually resolves to from the places a run is started.

    Measured, not inferred: Python puts the current directory ahead of every
    PYTHONPATH entry and ahead of an editable install, so nothing beats a
    directory named `dycraft` beside the process's cwd.
    """
    problems, fix = [], []
    sibling = REPO.parent / "dycraft"
    if sibling.is_dir() and sibling.resolve() != REPO:
        ran = run([sys.executable, "-c", _WHERE], cwd=REPO.parent, env=ctx.child_env(), timeout=60)
        got = marked_json(ran)
        if got and Path(got).resolve().parent != REPO:
            last = run(["git", "-C", sibling, "log", "-1", "--format=%h, %ad", "--date=short"],
                       timeout=20).out.strip() if which("git") else ""
            problems.append(f"started from {REPO.parent}, `import dycraft` loads "
                            f"{Path(got).resolve().parent}{f' (last commit {last})' if last else ''}"
                            " — even with this checkout's pkgroot on PYTHONPATH, because Python "
                            "searches the current directory first")
            fix += [f"launch from inside {REPO}, never from {REPO.parent}",
                    f"or rename the old checkout: mv {sibling} {sibling}-old"]
    # The PYTHONPATH entry this very copy came from, when it did: it then shadows the installed checkout, and the
    # fix is to drop the entry, never to install this copy.
    ours = None
    for entry in filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        cand = Path(entry) / "dycraft"
        if entry != str(ctx.pkgroot) and cand.exists() and cand.resolve() == REPO:
            ours = entry
        if entry != str(ctx.pkgroot) and cand.exists() and cand.resolve() != REPO:
            problems.append(f"PYTHONPATH entry {entry} provides a different `dycraft` ({cand.resolve()})")
            fix.append(f"drop {entry} from PYTHONPATH")
    try:
        import importlib.metadata as md
        direct = json.loads(md.distribution("dycraft").read_text("direct_url.json") or "{}")
        url = direct.get("url", "")
        # Only an editable install maps the name to a directory; a regular install's URL names the wheel or sdist
        # it was built from, which is not a checkout to compare with.
        if direct.get("dir_info", {}).get("editable") and url.startswith("file://") \
                and Path(url[7:]).resolve() != REPO:
            if ours:
                problems.append(f"this copy ({REPO}) comes from PYTHONPATH entry {ours}, ahead of the installed "
                                f"checkout {url[7:]}")
                fix.append(f"remove {ours} from PYTHONPATH (it puts another dycraft ahead of this checkout), or "
                           "unset PYTHONPATH")
            else:
                problems.append(f"the editable install maps `dycraft` to {url[7:]}")
                fix.append(f"{sys.executable} -m pip install -e {REPO}")
    except Exception:   # noqa: BLE001 — not installed is the normal case
        pass
    if problems:
        return warn("another `dycraft` can be imported instead of this checkout", detail=problems,
                    lost="a launch from the wrong directory silently runs a different checkout's "
                         "code; launches from inside this checkout are unaffected", fix=fix)
    return ok("nothing else on the path provides `dycraft`")


@check("repo.git", "git worktrees", S_REPO, caps=(CAP_DYCRAFT, CAP_BARE))
def check_git(ctx):
    if INSTALLED:
        return info("an installed package: runs record no source commit")
    git = which("git")
    lost = ("runs cannot record which commit, and which local changes, they ran (the daemon notes both "
            "on `daemon_start`)")
    if not git:
        return warn("git is not installed", lost=lost,
                    fix=["`sudo apt-get install git` (Fedora and RHEL: `sudo dnf install git`)"])
    ran = run([git, "-C", REPO, "rev-parse", "--short", "HEAD"], timeout=30)
    if ran.rc != 0:
        return warn("this directory is not a git checkout", lost=lost,
                        fix=["clone the repository instead of copying it"])
    return ok(f"{version_of(git)}; HEAD {ran.out.strip()}")


@check("repo.env_files", "environment files in sync", S_REPO)
def check_env_files(ctx):
    if INSTALLED:
        return info("an installed package: the environment specs ship with it")
    stale = []
    for name, render in ENV_FILES.items():
        path = REPO / name
        if not path.is_file() or path.read_text() != render():
            stale.append(name)
    if stale:
        return warn(f"{', '.join(stale)} missing or out of date with REQUIREMENTS",
                        lost="a machine built from them would not match what this checker expects",
                        fix=[f"{sys.executable} {Path(__file__).name} --write-env"])
    return ok("environment-main.yml and requirements-main.txt match REQUIREMENTS")


_DISABLING = {
    "DYCRAFT_BLENDER_PYTHON": "the bpy-module Blender (bare arms, staging, the blender stills backend)",
    "DYCRAFT_BLENDER_BINARY": "the Blender binary — the animation layer falls back to the bpy wheel "
                              "(no fluid)",
    "DYCRAFT_B123D_PYTHON": "the CAD interpreter — object, scene, and animation runs refuse to start",
    "DYCRAFT_IK_PYTHON": "h.ik",
}
_NOTABLE = ("DYCRAFT_COLLIDE_BACKEND", "DYCRAFT_RENDER_BACKEND",
            "DYCRAFT_BLENDER_DEVICE", "DYCRAFT_BLENDER_ENGINE", "DYCRAFT_COAL_PYTHON",
            "DYCRAFT_CLAUDE_EXECUTABLE", "DYCRAFT_CODEX_EXECUTABLE", "CODEX_HOME",
            "CLAUDE_CONFIG_DIR")


@check("repo.overrides", "environment overrides", S_REPO)
def check_overrides(ctx):
    off = [n for n in _DISABLING if n in os.environ and not os.environ[n].strip()]
    notes = [f"{n}={os.environ[n]}" for n in _NOTABLE if os.environ.get(n)]
    if off:
        return degraded("this shell switches subsystems off: " + ", ".join(off),
                        detail=[f"{n} is set but empty → disables {_DISABLING[n]}" for n in off] + notes,
                        lost="a run launched from this shell silently loses them",
                        fix=["unset " + " ".join(off) + "   before launching a run"])
    return info(f"{len(notes)} override(s) active" if notes else "none", detail=notes)


_PKG_PROBE = r'''
import importlib, importlib.metadata as md, json, sys
for mod, dist in zip(sys.argv[1::2], sys.argv[2::2]):
    rec = {}
    try:
        rec["version"] = md.version(dist)
    except Exception:
        rec["version"] = None
    if mod != "-":
        try:
            m = importlib.import_module(mod)
            rec["ok"] = True
            rec["file"] = getattr(m, "__file__", None)
        except BaseException as exc:
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"[:700]
    print("ENVTEST_PKG " + json.dumps([mod, dist, rec]), flush=True)
'''
_EXTRA_DISTS = ("python-fcl", "hpp-fcl", "cadquery-ocp", "cadquery-ocp-novtk", "bpy")


def package_facts(ctx) -> dict:
    """Every requirement imported in one child, one line per module as it goes,
    so a module that kills the interpreter is named rather than guessed."""
    if "pkgs" in ctx.facts:
        return ctx.facts["pkgs"]
    todo = [(r.module, r.dist) for r in REQUIREMENTS] + [("-", d) for d in _EXTRA_DISTS]
    facts: dict = {}
    while todo:
        argv = [sys.executable, "-c", _PKG_PROBE]
        for mod, dist in todo:
            argv += [mod, dist]
        ran = run(argv, timeout=600)
        for line in (ran.out or "").splitlines():
            if line.startswith("ENVTEST_PKG "):
                mod, dist, rec = json.loads(line[12:])
                facts[dist] = rec
        todo = [(m, d) for m, d in todo if d not in facts]
        if todo:
            mod, dist = todo.pop(0)
            facts[dist] = {"ok": False, "version": None,
                           "error": f"importing {mod} killed the interpreter "
                                    f"(exit {ran.rc}{', timed out' if ran.timed_out else ''})"}
    ctx.facts["pkgs"] = facts
    return facts


def _marker_holds(marker: str) -> bool:
    m = re.fullmatch(r'python_version\s*(<|>=)\s*"(\d+)\.(\d+)"', marker.strip())
    if not m:
        return True
    have, want = sys.version_info[:2], (int(m.group(2)), int(m.group(3)))
    return have < want if m.group(1) == "<" else have >= want


def _violates(rule: str, version: str) -> bool:
    def parts(v):
        return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
    if rule.startswith("=="):
        return version != rule[2:]
    if rule.startswith(">="):
        return parts(version) < parts(rule[2:])
    return False


def check_package(req: Req, ctx) -> Result:
    if req.marker and not _marker_holds(req.marker):
        return info(f"not needed on Python {platform.python_version()} ({req.marker})")
    rec = package_facts(ctx).get(req.dist, {})
    version, err = rec.get("version"), rec.get("error", "")
    status = FAIL if req.group in ("main", "cad") else UNAVAILABLE
    detail = [f"needed for: {req.why}"]
    fix = [install_cmd(ctx, f"{req.dist}{req.rule}")]
    if not rec.get("ok"):
        if "undefined symbol: XML_" in err:
            detail.append("a conda-forge pyexpat meeting PyPI OCP's system libexpat; prefix the "
                          "env's own lib/ to LD_LIBRARY_PATH (the CAD child does) — see "
                          "environment-b123d.yml")
        if req.dist == "coal":
            facts = package_facts(ctx)
            if facts.get("python-fcl", {}).get("version") or facts.get("hpp-fcl", {}).get("version"):
                detail.append("python-fcl / hpp-fcl are installed, but this path needs coal's "
                              "security_margin and signed-separation queries")
        lib = missing_library(err) if version else None
        if lib:
            return Result(status, f"{req.dist} {version} is installed, but this machine lacks {lib[0]}, a system "
                                  "library it loads", detail=detail, fix=lib[1], lost=req.lost)
        what = (f"{req.dist} {version} is installed but does not import: {err}" if version
                else f"{req.dist} is not installed")
        return Result(status, what, detail=detail, fix=fix, lost=req.lost)
    if req.rule and version and _violates(req.rule, version):
        return Result(status, f"{req.dist} {version}, but the code requires {req.rule}",
                      detail=detail, fix=fix, lost=req.lost)
    note = "" if version == req.pin else f"  (tested {req.pin})"
    return ok(f"{version}{note}")


def _register_packages():
    for req in REQUIREMENTS:
        if req.group == "cad":
            continue      # checked against the CAD interpreter, below
        caps = (CAP_VIEWER,) if req.group == "viewer" else (
            (CAP_DYCRAFT, CAP_STAGING) if req.dist in ("numpy", "scipy", "coal") else (CAP_DYCRAFT,))
        CHECKS.append(Check(f"pkg.{req.dist}", req.dist, S_PKG,
                            (lambda r: (lambda ctx: check_package(r, ctx)))(req), caps=caps))


_register_packages()


@check("pkg.conflicts", "known package conflicts", S_PKG, caps=(CAP_DYCRAFT,))
def check_conflicts(ctx):
    facts = package_facts(ctx)
    pypi_ocp = [d for d in ("cadquery-ocp", "cadquery-ocp-novtk") if facts.get(d, {}).get("version")]
    conda_ocp = sorted(p.name for p in (Path(sys.prefix) / "conda-meta").glob("ocp-*.json"))
    if pypi_ocp and conda_ocp:
        return fail(f"two packages provide OCP: PyPI {', '.join(pypi_ocp)} and conda {conda_ocp[0]}",
                    lost="build123d imports whichever wins, and they are different OCCT builds",
                    fix=["conda remove ocp   # build123d 0.11.1 wants the PyPI cadquery-ocp-novtk"])
    if facts.get("bpy", {}).get("version"):
        return degraded(f"bpy {facts['bpy']['version']} is installed in the harness env",
                        lost="bpy vendors NumPy 1.x while coal needs NumPy 2; one of them breaks",
                        fix=["keep bpy in its own interpreter (.blender_python)"])
    return ok("none")


@check("cad.interpreter", "build123d interpreter", S_CAD, needs=("repo.import",),
       caps=(CAP_DYCRAFT,))
def check_cad(ctx):
    from dycraft.agent.cad import _cad_env as ce
    from dycraft.agent.cad._b123d_version import REQUIRED_BUILD123D
    env_value = os.environ.get("DYCRAFT_B123D_PYTHON")
    fix = [f"single env: {install_cmd(ctx, f'build123d=={REQUIRED_BUILD123D}')}, then "
           f"`{sys.executable} -m dycraft.install cad`",
           "or a separate env: `conda env create -p <prefix>/dycraft-b123d-0.11.1 -f "
           "environment-b123d.yml`, recorded in `.b123d_python`"]
    lost = "object, scene, and animation runs refuse to start (LayerSession raises)"
    if env_value is not None and not env_value.strip():
        return fail("DYCRAFT_B123D_PYTHON is set but empty: CAD is deliberately switched off",
                    lost=lost, fix=["`unset DYCRAFT_B123D_PYTHON`"])
    interp = ce._resolve_b123d_python()
    if env_value and not interp:
        return fail(f"DYCRAFT_B123D_PYTHON names {env_value}, which is not an interpreter",
                    detail=["the variable beats every pointer file, so fix or unset it"],
                    lost=lost, fix=["`unset DYCRAFT_B123D_PYTHON`, or point it at a Python with "
                                    f"build123d {REQUIRED_BUILD123D}"])
    pointer = ce._pointer_file_python()
    searched = [f"DYCRAFT_B123D_PYTHON ({env_value or 'unset'})",
                f".b123d_python ({pointer or 'absent'})",
                "conventional prefixes: " + ", ".join(ce._conventional_pythons()[:3]) + " …",
                f"this interpreter, if it carries build123d {REQUIRED_BUILD123D}"]
    if not interp:
        return with_dialog(ctx, fail("none found", detail=searched,
                                     lost=lost, fix=fix),
                           check_cad, want="Choose the interpreter that carries build123d 0.11.1",
                           searched=searched, kinds=("cad",))
    source = ("DYCRAFT_B123D_PYTHON" if env_value else
              ".b123d_python" if pointer and ce._usable(pointer) == interp else
              "a conventional prefix" if interp in ce._conventional_pythons() else
              "this interpreter")
    got = probe_cad_python(interp)
    lib = missing_library(got.get("error") or "") if not got.get("ok") else None
    if lib:
        return fail(f"{interp} ({source}) cannot import build123d: this machine lacks {lib[0]}, a system library "
                    "it loads", lost=lost, fix=lib[1])
    if not got.get("ok"):
        return with_dialog(ctx, fail(f"{interp} ({source}) cannot import build123d: "
                                     f"{got.get('error')}", lost=lost, fix=fix),
                           check_cad, want="Choose a working build123d 0.11.1 interpreter",
                           searched=searched, kinds=("cad",))
    if got.get("version") != REQUIRED_BUILD123D:
        return with_dialog(ctx, fail(f"{interp} ({source}) has build123d {got.get('version')}; "
                                     f"DyCraft requires exactly {REQUIRED_BUILD123D}",
                                     lost="every CAD child refuses to run "
                                          "(build123d_version_mismatch)", fix=fix),
                           check_cad, want="Choose the interpreter with build123d 0.11.1",
                           searched=searched, kinds=("cad",))
    ctx.facts["cad_python"] = interp
    if pointer and not env_value and not ce._usable(pointer):
        return warn(f"build123d {got['version']} — {interp} [{source}], but .b123d_python names {pointer}, which is "
                    "not an interpreter", fix=["`python -m dycraft.install cad` records the interpreter again"])
    return ok(f"build123d {got['version']} (OCP {got.get('ocp', '?')}, Python "
              f"{got.get('python')}) — {interp}  [{source}]")


def _blender_sources(ctx):
    from dycraft.render import _blender_env as be
    binary, bpy = be.blender_binary(), be.blender_python()
    b_env = os.environ.get(be._BINARY_ENV_VAR)
    b_ptr = read_pointer(be._BINARY_POINTER_FILE) if be._BINARY_POINTER_FILE.is_file() else None
    b_src = ("DYCRAFT_BLENDER_BINARY" if b_env else
             ".blender_binary" if b_ptr and binary and str(Path(b_ptr).expanduser().resolve()) == binary
             else "PATH (`blender`)" if binary and which("blender") and
             str(Path(which("blender")).resolve()) == binary else "a conventional install root")
    p_env = os.environ.get("DYCRAFT_BLENDER_PYTHON")
    p_ptr = be._pointer_file_python()
    p_src = ("DYCRAFT_BLENDER_PYTHON" if p_env else ".blender_python" if p_ptr and bpy and
             be._usable(p_ptr) == bpy else "a conventional prefix")
    return binary, b_src, bpy, p_src


@check("blender.animation", "the animation layer's Blender", S_BLENDER, needs=("repo.import",),
       caps=(CAP_DYCRAFT,))
def check_blender_animation(ctx):
    binary, b_src, bpy, p_src = _blender_sources(ctx)
    ctx.facts.update(blender_binary=binary, bpy_python=bpy)
    named = os.environ.get("DYCRAFT_BLENDER_BINARY")
    if named and not binary:
        # `blender_binary()` answers None for a variable it cannot use, and the
        # animation layer then runs the bpy wheel without saying so. Only fluid
        # is lost -- unless there is no wheel either, and then nothing animates.
        return (warn if bpy else fail)(
            f"DYCRAFT_BLENDER_BINARY names {named}, which is not an executable Blender",
            lost=("the animation layer silently falls back to the bpy wheel, which cannot bake "
                  "fluid") if bpy else "there is no Blender at all for the animation layer",
            fix=["fix or unset DYCRAFT_BLENDER_BINARY (set it to the empty string to choose the "
                 "bpy wheel on purpose)"])
    searched = ["DYCRAFT_BLENDER_BINARY, .blender_binary, ~/blender/*, `blender` on PATH",
                "DYCRAFT_BLENDER_PYTHON, .blender_python, ~/conda_envs/dycraft-blender …"]
    if binary:
        problem = blender_packaging_problem(binary)
        if problem:
            return fail(f"{binary} [{b_src}] is {problem}",
                        lost="every Blender worker fails to start inside the sandbox",
                        fix=["install a blender.org 4.5 LTS tarball and record it in `.blender_binary`"])
        got = probe_blender_binary(binary)
        lib = missing_library(got.get("error") or "") if not got.get("ok") else None
        if lib:
            return fail(f"{binary} [{b_src}] does not start: this machine lacks {lib[0]}, a system library it loads",
                        lost="--layer animation cannot make its first edit", fix=lib[1])
        if not got.get("ok"):
            return fail(f"{binary} [{b_src}] does not start: {got.get('error')}",
                        lost="--layer animation cannot make its first edit")
        ctx.facts["blender_version"] = got["version"]
        detail, status = [], PASS
        if not got["version"].startswith("4.5"):
            status = DEGRADED
            detail.append("validated on 4.5 LTS; render engine names and the Python API move "
                          "between releases")
        if b_src.startswith("PATH") and bpy:
            detail.append(f"found on PATH, so it takes precedence over .blender_python for the "
                          f"animation layer; pin it with .blender_binary to make that explicit")
        return Result(status, f"Blender {got['version']} binary (fluid-capable) — {binary}  "
                              f"[{b_src}]", detail=detail,
                      lost="results may differ from the validated Blender" if status == DEGRADED else "")
    if bpy:
        got = probe_bpy_python(bpy)
        lib = missing_library(got.get("error") or "") if not got.get("ok") else None
        if lib:
            return fail(f"{bpy} [{p_src}] has bpy, but cannot import it: this machine lacks {lib[0]}, a system "
                        "library it loads", lost="--layer animation cannot make its first edit", fix=lib[1])
        if not got.get("ok"):
            return fail(f"{bpy} [{p_src}] cannot import bpy: {got.get('error')}",
                        lost="--layer animation cannot make its first edit")
        ctx.facts["blender_version"] = got["version"]
        return ok(f"bpy {got['version']} module — {bpy}  [{p_src}]",
                  detail=["no Blender binary: fluid bakes are unavailable (see Optional)"])
    return with_dialog(ctx, fail("none found", detail=searched,
                                 lost="--layer animation cannot make its first edit",
                                 fix=["`python -m dycraft.install blender` (`./setup_blender.sh` in a clone): "
                                      "a bpy 4.5 env and the official 4.5 LTS build"]),
                       check_blender_animation, want="Choose a Blender for the animation layer",
                       searched=searched, kinds=("binary", "bpy"))


def _release_line(version: str) -> tuple[int, ...] | str:
    """A Blender version's release line, `(major, minor)`: "4.5.13 LTS" -> (4, 5)."""
    m = re.match(r"\s*(\d+)\.(\d+)", version or "")
    return (int(m.group(1)), int(m.group(2))) if m else version


@check("blender.bpy", "bpy module interpreter", S_BLENDER, needs=("repo.import",),
       caps=(CAP_BARE, CAP_STAGING))
def check_bpy(ctx):
    from dycraft.render import _blender_env as be
    _, _, bpy, p_src = _blender_sources(ctx)
    named = os.environ.get("DYCRAFT_BLENDER_PYTHON")
    if named and not bpy:
        return fail(f"DYCRAFT_BLENDER_PYTHON names {named}, which is not an interpreter",
                    detail=["the variable beats .blender_python, so fix or unset it"],
                    lost="bare arms and staging have no Blender",
                    fix=["`unset DYCRAFT_BLENDER_PYTHON`, or point it at a Python that imports bpy"])
    lost = ("the bare-agent comparison stages every bare arm's .blend through it (benchmark.stage) "
            "and runs the bare arm on it; a Blender binary "
            "alone serves neither")
    fix = ["`python -m dycraft.install blender`",
           "or write the path of a Python 3.11 that imports bpy 4.5, on one line, into "
           f"`{be._PYTHON_POINTER_FILE}`"]
    if not bpy:
        return with_dialog(ctx, fail("none found", lost=lost, fix=fix),
                           check_bpy, want="Choose the interpreter that imports bpy",
                           searched=["DYCRAFT_BLENDER_PYTHON, .blender_python, "
                                     "~/conda_envs/dycraft-blender"], kinds=("bpy",))
    got = probe_bpy_python(bpy)
    lib = missing_library(got.get("error") or "") if not got.get("ok") else None
    if lib:
        return fail(f"{bpy} [{p_src}] has bpy, but cannot import it: this machine lacks {lib[0]}, a system library "
                    "it loads", lost=lost, fix=lib[1])
    if not got.get("ok"):
        return with_dialog(ctx, fail(f"{bpy} [{p_src}] cannot import bpy: {got.get('error')}",
                                     lost=lost, fix=fix),
                           check_bpy, want="Choose the interpreter that imports bpy",
                           searched=[f".blender_python → {bpy}"], kinds=("bpy",))
    ctx.facts["bpy_version"] = got["version"]
    detail, status = [], PASS
    if got.get("numpy") and int(got["numpy"].split(".")[0]) >= 2:
        status = DEGRADED
        detail.append("environment-blender.yml pins NumPy < 2 beside bpy 4.5; this is untested")
    animation = ctx.facts.get("blender_version")
    if animation and _release_line(animation) != _release_line(got["version"]):
        status = DEGRADED
        detail.append(f"the animation layer runs Blender {animation} but bare arms get bpy "
                      f"{got['version']}: the two arms are not on the same Blender")
    elif animation and animation != got["version"]:
        # What `install blender` sets up: the newest bpy of the 4.5 line beside the pinned
        # official binary. Patch releases of one LTS line keep its API and behavior.
        detail.append(f"bare arms get bpy {got['version']} beside the animation layer's Blender "
                      f"{animation}: the same release line, as `install blender` sets them up")
    return Result(status, f"bpy {got['version']} (Python {got['python']}, NumPy {got['numpy']}) — "
                          f"{bpy}  [{p_src}]", detail=detail,
                  lost="the bare/dycraft comparison is not like-for-like" if status == DEGRADED else "")


#: Said wherever Blender can be run only without a sandbox.
_BLENDER_OPT_IN_FIX = ("without sudo: `--allow-unsandboxed` runs Blender without a sandbox, so the agent's Blender code "
                       "can read and write anywhere you can")


def _landlock_abi() -> int:
    """This kernel's Landlock ABI version, 0 where it has none, by the worker's own launcher."""
    try:
        from dycraft.animation.blender import runtime
        return runtime.landlock_abi()
    except Exception:                              # noqa: BLE001 - no answer is no Landlock
        return 0


@check("sandbox.bwrap", "bubblewrap", S_SANDBOX, needs=("repo.import",), caps=(CAP_DYCRAFT,))
def check_bwrap(ctx):
    bwrap = which("bwrap") or str(Path(sys.executable).parent / "bwrap")
    if not Path(bwrap).is_file():
        abi = _landlock_abi()
        if abi:
            return info(f"not installed: the Blender worker runs confined by Landlock (ABI {abi}) instead, with this "
                        "machine's paths (user namespaces, below)",
                        fix=["for its own filesystem as well: `python -m dycraft.install tools`, or your "
                             "distribution's `sudo apt-get install bubblewrap socat`"])
        return fail("bwrap not found, and this kernel has no Landlock to confine Blender in its place",
                    lost="the animation layers refuse to run Blender unconfined, so nothing animates",
                    fix=["`python -m dycraft.install tools`",
                         "or your distribution's: `sudo apt-get install bubblewrap socat` (`dnf install bubblewrap socat`)",
                         _BLENDER_OPT_IN_FIX], data={"sandbox": True})
    ctx.facts["bwrap"] = bwrap
    return ok(f"{version_of(bwrap)} — {bwrap}")


@check("sandbox.userns", "user namespaces", S_SANDBOX, needs=("sandbox.bwrap", "repo.import"),
       caps=(CAP_DYCRAFT,))
def check_userns(ctx):
    """Whether the Blender worker gets a sandbox with a filesystem of its own (namespaces), Landlock with this
    machine's paths in its place (`runtime.sandbox_mode`), or, where neither works, none, with --allow-unsandboxed."""
    from dycraft.animation.blender import runtime
    forced = (os.environ.get(runtime.SANDBOX_ENV) or "auto").strip().lower()
    if forced == "landlock":
        abi = _landlock_abi()
        return warn(f"{runtime.SANDBOX_ENV}=landlock: the Blender worker runs confined by Landlock (ABI {abi}) with "
                    "this machine's paths") if abi else \
            fail(f"{runtime.SANDBOX_ENV}=landlock, but this kernel has no Landlock",
                 lost="every Blender worker fails to start", fix=[f"unset {runtime.SANDBOX_ENV}"])
    prefix = runtime._userns_prefix(which("unshare"))
    if prefix:
        return ok("`unshare -Urn` makes them")
    bwrap = ctx.facts.get("bwrap")
    ran = run([bwrap, "--unshare-user", "--unshare-net", "--ro-bind", "/", "/", "/bin/true"],
              timeout=30) if bwrap else None
    if ran is not None and ran.rc == 0:
        return ok("bwrap makes them (`unshare -Urn` is refused here, which is expected on "
                  "Ubuntu 24.04's AppArmor policy)")
    abi = _landlock_abi()
    if abi:
        detail = (ran.tail(2) if ran is not None else ["bubblewrap is not installed"]) + [
            "the worker sees this machine's paths, not /workspace, /tools, /output and a /tmp of its own: authored "
            "code is told the project's real path, and our mocap helpers take /references/mocap/... as before"]
        if abi < 6:
            detail.append(f"this kernel's Landlock (ABI {abi}) does not stop a worker signalling your other "
                          "processes or reaching their abstract Unix sockets; Linux 6.12 (ABI 6) does")
        from dycraft import _userns
        steps = _userns.fixes() or ["the system's bubblewrap: `sudo apt-get install bubblewrap socat`"]
        return warn(f"bubblewrap cannot make them here, so the Blender worker runs confined by Landlock (ABI {abi}) "
                    "with this machine's paths", detail=detail,
                    fix=[f"for a filesystem of its own as well, with root: {steps[0]}", *steps[1:]])
    if ran is None:
        return fail("no bubblewrap, and this kernel has no Landlock", lost="every Blender worker fails to start",
                    fix=["`python -m dycraft.install tools`", _BLENDER_OPT_IN_FIX], data={"sandbox": True})
    if os.environ.get("DYCRAFT_ALLOW_UNSANDBOXED") == "1" or os.environ.get("DYCRAFT_ALLOW_UNSANDBOXED_CODEX") == "1":
        return fail("running Blender without a sandbox (--allow-unsandboxed): neither bubblewrap nor Landlock can "
                    "confine it here, so the agent's Blender code can read and write anywhere you can",
                    detail=ran.tail(2), data={"sandbox": True})
    restrict = ""
    try:
        restrict = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").read_text().strip()
    except OSError:
        pass
    # With root, what works on this system (`_userns.fixes`): the package where /usr/bin/bwrap is missing, a profile
    # where none grants it user namespaces (a minimal image ships none), and lifting the restriction.
    from dycraft import _userns
    if restrict == "1" and ctx.facts["bwrap"] != "/usr/bin/bwrap":
        return fail(f"this bwrap ({ctx.facts['bwrap']}) cannot create a user namespace: Ubuntu's AppArmor restriction "
                    "allows that only to a bwrap with a profile, and this kernel has no Landlock",
                    detail=ran.tail(2), lost="every Blender worker fails to start",
                    fix=[*_userns.fixes(), _BLENDER_OPT_IN_FIX], data={"sandbox": True})
    if restrict == "1":
        return fail("/usr/bin/bwrap cannot create a user namespace: Ubuntu's AppArmor restriction stops it here, and "
                    "this kernel has no Landlock",
                    detail=ran.tail(2), lost="every Blender worker fails to start",
                    fix=[*_userns.fixes(), _BLENDER_OPT_IN_FIX], data={"sandbox": True})
    fix = ["`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` (Ubuntu 24.04), or use "
           "`/usr/bin/bwrap`, which ships with an AppArmor profile that permits it"]
    if Path("/proc/sys/kernel/unprivileged_userns_clone").is_file():
        fix.append("`sudo sysctl -w kernel.unprivileged_userns_clone=1` (Debian)")
    return fail("neither unshare nor bwrap can create a user namespace, and this kernel has no Landlock",
                detail=ran.tail(2) + ([f"kernel.apparmor_restrict_unprivileged_userns={restrict}"]
                                      if restrict else []),
                lost="every Blender worker fails to start", fix=[*fix, _BLENDER_OPT_IN_FIX], data={"sandbox": True})


def _in_worker_sandbox(ctx, name: str, program, timeout: int = 180) -> Ran:
    """Run `program(cmd)`'s argv in the sandbox `runtime.command` builds for a worker (same namespaces, binds and
    closed environment) in place of the worker's own program, from `/workspace`."""
    from dycraft.animation.blender import runtime
    root = ctx.tmp / name
    (root / "ws").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(exist_ok=True)
    (root / "job.json").write_text("{}")
    cmd = runtime.command(root / "ws", root / "out", root / "job.json")
    return run(runtime.with_program(cmd, program(cmd)), timeout=timeout, env=runtime.worker_env())


def _setenv_of(cmd: list, name: str) -> str | None:
    """A variable the sandbox sets for its program, in either form `runtime.command` builds."""
    from dycraft.animation.blender import runtime
    spec = runtime._spec_of(cmd)
    if spec is not None:
        return spec.get("env", {}).get(name)
    return next((cmd[i + 2] for i, a in enumerate(cmd[:-2]) if a == "--setenv" and cmd[i + 1] == name), None)


def _first_error(ran: Ran) -> str:
    """The line that says why a sandboxed interpreter did not start: the first fatal one, else the last."""
    lines = ran.tail(40)
    return next((l for l in lines if l.startswith(("Fatal Python error", "ModuleNotFoundError", "ImportError",
                                                    "bwrap:"))), lines[-1] if lines else ran.error or "no output")


@check("sandbox.coal_view", "collision interpreter in the sandbox", S_SANDBOX,
       needs=("repo.import", "pkg.coal", "sandbox.userns", "blender.animation"), caps=(CAP_DYCRAFT,))
def check_coal_view(ctx):
    """Started inside the worker's sandbox as `CoalClient` starts it (`-I -B`), so a layout whose binds look right
    but which cannot start there is caught here, not first in a run."""
    from dycraft._interpreters import sandbox_view
    from dycraft.animation.blender import runtime
    lost = "collision checks fail inside the sandbox; the animation layer cannot verify penetration"
    python = os.environ.get("DYCRAFT_COAL_PYTHON") or sys.executable
    try:
        view = sandbox_view(python, runtime.COAL_IMPORTS)
    except RuntimeError as exc:
        return fail(str(exc), lost=lost)
    binds = [f"bind {s}" + (f" → {d}" if s != d else "") for s, d in view.binds]
    # `/tools` as an argument of its own, which a sandbox with this machine's paths maps (`runtime._host_command`).
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import coal_worker; coal_worker.extend_path(); "
            + "; ".join(f"import {m}" for m in runtime.COAL_IMPORTS) + "; print('COAL_STARTED')")
    try:
        ran = _in_worker_sandbox(ctx, "coal_view", lambda cmd: [_setenv_of(cmd, "DYCRAFT_COAL_PYTHON") or view.launch,
                                                                 "-I", "-B", "-c", code, "/tools"], timeout=120)
    except RuntimeError as exc:
        return fail(f"the worker's sandbox cannot be built: {exc}", detail=binds, lost=lost)
    if ran.rc != 0 or "COAL_STARTED" not in ran.out:
        return fail(f"{view.launch} does not start inside the sandbox: {_first_error(ran)}",
                    detail=ran.tail(6) + binds, lost=lost)
    extra = f"; re-adds {len(view.sitedirs)} import root(s) an isolated start misses" if view.sitedirs else ""
    return ok(f"{view.launch}: starts and imports coal inside the sandbox, {len(view.binds)} bind(s){extra}",
              detail=binds)


@check("sandbox.bpy_view", "bpy interpreter in the sandbox", S_SANDBOX,
       needs=("repo.import", "blender.animation", "sandbox.userns"), caps=(CAP_DYCRAFT,))
def check_bpy_view(ctx):
    if ctx.facts.get("blender_binary"):
        return info("not used: the animation layer runs the Blender binary")
    from dycraft._interpreters import sandbox_view
    from dycraft.animation.blender import runtime
    lost = "Blender workers cannot start inside the sandbox"
    try:
        view = sandbox_view(ctx.facts["bpy_python"], runtime.BPY_IMPORTS)
    except RuntimeError as exc:
        return fail(str(exc), lost=lost)
    code = "import bpy, numpy; print('BPY_STARTED', bpy.app.version_string)"
    try:
        # The worker's own program starts with this interpreter, its PYTHONPATH already set in the sandbox.
        ran = _in_worker_sandbox(ctx, "bpy_view", lambda cmd: [view.launch, "-B", "-c", code])
    except RuntimeError as exc:
        return fail(f"the worker's sandbox cannot be built: {exc}", lost=lost)
    if ran.rc != 0 or "BPY_STARTED" not in ran.out:
        return fail(f"{view.launch} does not start inside the sandbox: {_first_error(ran)}", detail=ran.tail(6),
                    lost=lost)
    return ok(f"{view.launch}: starts and imports bpy inside the sandbox, {len(view.binds)} bind(s)")


#: The system packages for the GL and X11 libraries, with Mesa's drivers, that Blender, build123d and open3d
#: load. Without root, `install tools` installs them instead.
GL_APT = ("`sudo apt-get install libgl1 libglx0 libopengl0 libegl1 libgl1-mesa-dri libice6 libsm6 libxfixes3 libxi6 "
          "libxrender1 libxt6t64 libvulkan1 mesa-vulkan-drivers`")
GL_DNF = ("`sudo dnf install mesa-libGL mesa-libEGL mesa-dri-drivers libglvnd-opengl libICE libSM libXfixes libXi "
          "libXrender libXt vulkan-loader mesa-vulkan-drivers`")
GL_SYSTEM_FIXES = (f"or the system's, with root: {GL_APT} (Ubuntu 22.04 and Debian 12: `libxt6` for `libxt6t64`)",
                   f"or, on Fedora and RHEL: {GL_DNF}")


#: System libraries an import needs and a minimal server or container lacks: soname → (apt package, dnf package).
#: The GL and X11 ones are what `install tools` installs without root (`install.GL_LIBRARIES`).
SYSTEM_LIBRARIES = {
    "libGL.so.1": ("libgl1", "mesa-libGL"), "libEGL.so.1": ("libegl1", "mesa-libEGL"),
    "libGLX.so.0": ("libglx0", "libglvnd-glx"), "libOpenGL.so.0": ("libopengl0", "libglvnd-opengl"),
    "libX11.so.6": ("libx11-6", "libX11"), "libXext.so.6": ("libxext6", "libXext"),
    "libXrender.so.1": ("libxrender1", "libXrender"), "libXi.so.6": ("libxi6", "libXi"),
    "libXfixes.so.3": ("libxfixes3", "libXfixes"), "libSM.so.6": ("libsm6", "libSM"),
    "libICE.so.6": ("libice6", "libICE"), "libXt.so.6": ("libxt6t64", "libXt"),
    "libxkbcommon.so.0": ("libxkbcommon0", "libxkbcommon"), "libgomp.so.1": ("libgomp1", "libgomp"),
}
#: The ones `install tools` provides.
_GL_SONAMES = {"libGL.so.1", "libEGL.so.1", "libGLX.so.0", "libOpenGL.so.0", "libX11.so.6", "libXext.so.6",
               "libXrender.so.1", "libXi.so.6", "libXfixes.so.3", "libSM.so.6", "libICE.so.6", "libXt.so.6"}


def missing_library(text: str) -> tuple[str, list[str]] | None:
    """`(soname, fixes)` when `text` is the loader saying a shared library is missing ("libGL.so.1: cannot open shared
    object file"): a system library this machine lacks, not a package to reinstall. None otherwise."""
    m = re.search(r"(lib[\w.+-]*?\.so[\w.]*): cannot open shared object file", text or "")
    if not m:
        return None
    soname = m.group(1)
    if soname in _GL_SONAMES:
        return soname, ["without sudo: `python -m dycraft.install tools`", *GL_SYSTEM_FIXES]
    packages = SYSTEM_LIBRARIES.get(soname)
    if packages:
        return soname, [f"the system's, with root: `sudo apt-get install {packages[0]}` "
                        f"(Fedora and RHEL: `sudo dnf install {packages[1]}`)"]
    return soname, [f"install the system package that provides {soname} (`apt-file search {soname}` names it)"]


@check("render.stills", "stills renderer", S_RENDER, needs=("repo.import",), caps=(CAP_DYCRAFT,))
def check_stills(ctx):
    from dycraft.render import default_backend
    backend = default_backend()
    if backend == "open3d":
        rec = package_facts(ctx).get("open3d", {})
        if not rec.get("ok"):
            return fail("open3d (the default backend) does not import; the open3d check above says why",
                        lost="snapshots and storyboards' stills fail")
        import ctypes
        # open3d 0.20's offscreen renderer draws through Vulkan, earlier ones through EGL. Loaded as the renderer
        # loads it, so the loader path (`install tools`' GL libraries) counts.
        version = str(rec.get("version") or "")
        vulkan = _release_line(version) >= (0, 20) if isinstance(_release_line(version), tuple) else True
        library, what = ("libvulkan.so.1", "Vulkan") if vulkan else ("libEGL.so.1", "EGL")
        try:
            ctypes.CDLL(library)
        except OSError:
            # A refusal, not a warning: an agent that cannot see its renders builds blind, and delivers no views.
            return fail(f"open3d {version}, but no {library} was found",
                        lost=f"open3d's offscreen renderer needs {what} (Mesa's software driver is enough): "
                             "snapshots, storyboards' stills and an object's views fail",
                        fix=["without sudo: `python -m dycraft.install tools`", *GL_SYSTEM_FIXES])
        return ok(f"open3d {version} (offscreen, {what})  — proven by --deep")
    if backend == "blender":
        # Resolved here, not read from the Blender checks: a launch's checks for an object or scene run do not
        # include them, and a run that renders with Blender was then refused with its interpreter in place.
        if not ctx.facts.get("bpy_python"):
            ctx.facts["bpy_python"] = _blender_sources(ctx)[2]
        return ok("blender (DYCRAFT_RENDER_BACKEND=blender)") if ctx.facts.get("bpy_python") else \
            fail("DYCRAFT_RENDER_BACKEND=blender but there is no bpy interpreter",
                 lost="every still fails with `rendering unavailable`",
                 fix=["`python -m dycraft.install blender`", "or render with open3d: `--render-backend open3d`"])
    return fail(f"unknown DYCRAFT_RENDER_BACKEND={backend!r}")


@check("render.gpu", "GPU", S_RENDER, needs=("repo.import",), caps=(CAP_GPU,))
def check_gpu(ctx):
    from dycraft.animation.blender import runtime
    nodes = sorted(str(p) for p in Path("/dev").glob("nvidia[0-9]*"))
    if runtime.nvidia_usable():
        names = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=20).out \
            if which("nvidia-smi") else ""
        return ok("NVIDIA: " + (", ".join(filter(None, names.strip().splitlines())) or ", ".join(nodes)))
    lost = ("EEVEE draws with Mesa's software rasteriser (~9 s a frame at 64 samples) and Cycles "
            "on the CPU; everything works, much slower")
    if nodes:
        return unavailable("GPU device nodes are visible but cannot be opened", lost=lost,
                           detail=["a slurm job without --gres=gpu sees the nodes and is refused "
                                   "at open()"])
    return unavailable("no usable GPU", lost=lost)


@check("tools.ffmpeg", "ffmpeg / ffprobe", S_RENDER, caps=(CAP_DYCRAFT,))
def check_ffmpeg(ctx):
    ffmpeg, ffprobe = which("ffmpeg"), which("ffprobe")
    lost = "final_render cannot encode or verify the H.264 film, so the DyCraft arm cannot finish"
    if not ffmpeg or not ffprobe:
        return fail("missing: " + ", ".join(n for n, p in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
                                            if not p), lost=lost,
                    fix=["`python -m dycraft.install tools`", "or your distribution's: `sudo apt-get install ffmpeg`"])
    encoders = run([ffmpeg, "-hide_banner", "-encoders"], timeout=30).out
    if "libx264" not in encoders:
        return fail("ffmpeg has no libx264 encoder", lost=lost,
                    fix=["install an ffmpeg built with libx264 (the distro or conda-forge build)"])
    return ok(f"{version_of(ffmpeg, '-version')} with libx264")


@check("tools.choom", "OOM tiers", S_RENDER)
def check_choom(ctx):
    if which("choom"):
        return ok("choom (util-linux) prices every worker for the OOM killer")
    return info("no choom: OOM tiers are applied by a shell fallback instead (works)")


def _claude_accounts() -> list[str]:
    out = []
    for d in [Path.home() / ".claude"] + [Path.home() / f".claude{n}" for n in range(1, 10)]:
        if d.is_dir():
            out.append(f"{d.name}{'' if (d / '.credentials.json').is_file() else ' (no credentials file)'}")
    return out


def _codex_accounts() -> list[str]:
    out = []
    for d in [Path.home() / ".codex"] + [Path.home() / f".codex{n}" for n in range(1, 10)]:
        if d.is_dir():
            out.append(f"{d.name}{'' if (d / 'auth.json').is_file() else ' (no auth.json)'}")
    return out


#: The paper's runs used Claude Code 2.1.316–2.1.319 and Codex CLI 0.153.4. This is a note for developers, not a
#: requirement: DyCraft is expected to work with newer releases, and nothing here checks the installed version. If a
#: launch fails after a CLI update, these are the releases to compare against.
@check("harness.claude", "claude", S_HARNESS, needs=("repo.import",))
def check_claude(ctx):
    from dycraft.cli.backends import ClaudeBackend
    exe, pinned = ClaudeBackend._resolve_executable()
    path = which(exe)
    if not path:
        return unavailable(f"`{exe}` is not on PATH",
                           lost="--backend claude* and bare claude arms cannot run",
                           fix=["install Claude Code: `curl -fsSL https://claude.ai/install.sh | bash` "
                                "(https://code.claude.com/docs/en/setup)", "then log in: `claude auth login`"])
    ctx.facts["claude"] = path
    detail = [f"accounts: {', '.join(_claude_accounts()) or 'none'}"]
    if pinned:
        detail.append(f"plain --backend claude authenticates as {pinned} (DYCRAFT_CLAUDE_EXECUTABLE)")
    version = version_of(path)
    from dycraft.cli.backends import claude_sandbox_problem
    problem = claude_sandbox_problem()
    if problem:
        missing = problem.endswith("not on PATH")
        return degraded(f"{version} — its Bash sandbox cannot start here, so Claude Code runs only with "
                        f"--allow-unsandboxed (unconfined): {problem}", detail=detail,
                        lost="DyCraft refuses to start Claude Code agents here without --allow-unsandboxed, and "
                             "bare claude arms cannot run (failIfUnavailable)",
                        fix=(["`python -m dycraft.install tools`",
                              "or your distribution's: `sudo apt-get install bubblewrap socat`"] if missing else
                             _claude_userns_fixes())
                            + ["without sudo: `--allow-unsandboxed` runs Claude Code without its sandbox, so its "
                               "commands can read and write anywhere you can"],
                        data={"sandbox": True})
    # The account a plain `--backend claude` uses; a `claudeN` launch checks its own (`ClaudeBackend.preflight`), so
    # this does not refuse one (`data["login"]`, `launch_refusals`).
    from dycraft.cli.backends.claude import claude_login_fix, claude_login_problem
    account = pinned or os.environ.get("CLAUDE_CONFIG_DIR")
    where = str(ClaudeBackend._config_dir()).replace(str(Path.home()), "~", 1)
    detail.insert(0, f"--backend claude uses the account in {where} "
                     f"({'CLAUDE_CONFIG_DIR' if account else 'the default'}); claudeN uses ~/.claudeN")
    login = claude_login_problem(exe, account)
    if login:
        return degraded(f"{version} — but the account in {where} cannot start an agent: {login}", detail=detail,
                        lost="--backend claude refuses to start here until it is logged in", data={"login": True},
                        fix=[claude_login_fix(account)])
    return ok(f"{version} — {path}", detail=detail)


@check("harness.codex", "codex", S_HARNESS, needs=("repo.import",))
def check_codex(ctx):
    from dycraft.cli.backends import CodexBackend
    binary, home = CodexBackend._resolve_codex_cli()
    if not binary or not which(binary):
        return unavailable("not on PATH, and DYCRAFT_CODEX_EXECUTABLE does not name one",
                           lost="--backend codex* and bare codex arms cannot run",
                           fix=["install the Codex CLI: `curl -fsSL https://chatgpt.com/codex/install.sh | sh` "
                                "(https://learn.chatgpt.com/docs/codex/cli)", "then log in: `codex login`"])
    ctx.facts["codex"] = binary
    detail = [f"accounts: {', '.join(_codex_accounts()) or 'none'}"]
    version = version_of(binary)
    # The account a plain `--backend codex` uses: DyCraft pins it (`CodexBackend.HOME`), so an ambient CODEX_HOME
    # elsewhere is not it. A `codexN` launch checks its own (`codex_account_problem`), so this does not refuse one.
    shown = home.replace(str(Path.home()), "~", 1)
    detail.insert(0, f"--backend codex uses the account in {shown}; codexN uses ~/.codexN")
    ambient = os.environ.get("CODEX_HOME")
    if ambient and Path(ambient).expanduser().resolve() != Path(home).resolve():
        detail.insert(1, f"CODEX_HOME={ambient} is not used: DyCraft runs --backend codex on {shown}")
    from dycraft.cli.backends.codex import codex_account_problem
    account = codex_account_problem(home)
    if account:
        _, message, fix = account
        return degraded(f"{version} — but the account in {shown} cannot start an agent: {message}", detail=detail,
                        lost="--backend codex refuses to start here until it is logged in", data={"login": True},
                        fix=list(fix))
    problem, tmpdir, inherited, wrapper = _codex_sandbox_problem(binary, home)
    if problem:
        return degraded(f"{version} — its sandbox does not work here, so Codex runs only with --allow-unsandboxed "
                        f"(unconfined): {problem}", detail=detail,
                        lost="DyCraft refuses to start Codex agents here without --allow-unsandboxed",
                        fix=list(_codex_sandbox_fixes()), data={"sandbox": True})
    from dycraft.cli.backends.codex import RUN_TMPDIR
    own = bool(tmpdir) and Path(tmpdir).name == RUN_TMPDIR
    where = "inside its run directory" if own else f"in {tmpdir}"
    # How the capability probes start Codex (`HarnessProbe.launch`), as a launch here would.
    ctx.facts["codex_launch"] = {"wrapper": list(wrapper), "tmpdir": RUN_TMPDIR if own else tmpdir}
    from dycraft.cli.backends.codex import CODEX_NAMESPACE_ENV
    if wrapper and str(inherited or "").startswith(CODEX_NAMESPACE_ENV):
        return warn(f"{version} — {inherited}: each launch runs Codex in a namespace of its own with a fresh /tmp, "
                    f"and TMPDIR {where}", detail=detail)
    if wrapper:
        return warn(f"{version} — its sandbox cannot start with this machine's /tmp, so each launch runs Codex in a "
                    f"namespace of its own with a fresh /tmp, and TMPDIR {where}",
                    detail=detail + [f"with the inherited /tmp and TMPDIR: {inherited}"])
    if tmpdir:
        return warn(f"{version} — its sandbox cannot use TMPDIR={os.environ.get('TMPDIR') or '/tmp'}, so each "
                    f"launch runs Codex with TMPDIR {where}", detail=detail + [f"with the "
                    f"inherited TMPDIR: {inherited}"])
    return ok(f"{version} — {binary}", detail=detail)


def _under_tmp(path: Path) -> bool:
    """Whether `path` is under this machine's /tmp, which Codex's own namespace replaces (`codex_namespace_wrapper`)."""
    try:
        return Path(path).resolve().is_relative_to(Path("/tmp").resolve())
    except OSError:
        return True


def _codex_sandbox_fixes() -> list[str]:
    """The fixes a refused Codex launch records too (`cli/backends/codex.py`)."""
    from dycraft.cli.backends.codex import codex_sandbox_fixes
    return codex_sandbox_fixes()


def _claude_userns_fixes() -> list[str]:
    """With root, what lets Claude Code's bubblewrap make its namespaces on this system."""
    from dycraft.cli.backends.claude import claude_userns_fixes
    return claude_userns_fixes()


def _codex_sandbox_problem(binary: str, home: str) -> tuple[str | None, str | None, str | None, list[str]]:
    """The launch's own sandbox probe, with its retries in other TMPDIRs and with a /tmp of its own
    (`cli/backends/codex.py::codex_sandbox_probe_or_tmpdir`), in a scratch directory."""
    import tempfile
    from dycraft.cli.backends.codex import codex_sandbox_probe_or_tmpdir
    scratch = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "dycraft"
    scratch.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="envcheck_codex_", dir=scratch))
    try:
        return codex_sandbox_probe_or_tmpdir(binary, home, run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@check("harness.any", "at least one harness", S_HARNESS, caps=(CAP_DYCRAFT, CAP_BARE))
def check_any_harness(ctx):
    have = [n for n in ("claude", "codex") if ctx.facts.get(n)]
    if not have:
        return fail("neither claude nor codex is installed",
                    lost="no agent can run: both arms are CLI sessions")
    return ok(", ".join(have))


@check("bare.code", "bare-agent comparison present", S_BARE, needs=("repo.import",),
       caps=(CAP_STAGING, CAP_BARE))
def check_bare_code(ctx):
    """The bare-agent comparison ships as `dycraft.benchmark` (`python -m dycraft.benchmark.bare`). Its
    other needs are checked where they live: the bpy interpreter (blender.bpy), user namespaces and
    bwrap (sandbox), ffmpeg (media) and an agent CLI (harnesses)."""
    import importlib.util
    # find_spec on a submodule imports its parent, and raises if the parent is missing.
    missing = (["dycraft.benchmark"] if importlib.util.find_spec("dycraft.benchmark") is None else
               [m for m in ("dycraft.benchmark.bare", "dycraft.benchmark.stage")
                if importlib.util.find_spec(m) is None])
    if missing:
        return unavailable("this copy of dycraft has no " + ", ".join(missing),
                           lost="the bare-agent comparison (python -m dycraft.benchmark.bare) cannot run")
    from dycraft.benchmark import bare
    g1 = Path(getattr(bare, "G1_SOURCE", None) or state_path("assets") / "robots" / "g1_description")
    fetch = getattr(bare, "G1_FETCH_COMMAND", None)
    detail = [] if g1.is_dir() else [
        f"briefs B01–B08 also need the G1 robot description at {g1}"
        + (f": {fetch}" if fetch else " (it is not in git)")]
    return ok("dycraft.benchmark present", detail=detail)


#: Run under the IK interpreter (`install ik` runs it too): a kernel on the GPU, because an import and
#: `torch.cuda.is_available()` both pass with a torch build that has no kernels for this GPU.
IK_PROBE = r'''
import json
out = {}
try:
    import torch
    out.update(torch=torch.__version__, cuda=torch.cuda.is_available(), arch=torch.cuda.get_arch_list())
    if out["cuda"]:
        out["gpu"] = torch.cuda.get_device_name(0)
        out["cap"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
        try:
            x = torch.ones(64, 64, device="cuda")
            out["kernel"] = float((x @ x).sum()) == 64.0 ** 3
        except Exception as e:
            out["kernel"] = False
            out["error"] = (str(e).splitlines() or [type(e).__name__])[0]
    import curobo
    out["curobo"] = getattr(curobo, "__version__", "?")
except Exception as e:
    out.setdefault("error", f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}")
print("ENVTEST_JSON " + json.dumps(out))
'''

#: The known answer, run through `h.ik`'s own path (`animation.ik.ik`, so whatever the IK child sets up runs too):
#: cuRobo's bundled Franka posed by FK at its default configuration, and IK solved back to that pose.
IK_SOLVE = """
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose
solver = InverseKinematics(InverseKinematicsCfg.create(robot="franka.yml", num_seeds=16))
link = solver.tool_frames[0]
want = solver.compute_kinematics(solver.default_joint_state).tool_poses[link]
goal = Pose(position=want.position.reshape(1, 3).clone(), quaternion=want.quaternion.reshape(1, 4).clone())
res = solver.solve_pose(GoalToolPose.from_poses({link: goal}, num_goalset=1))
result["solved"] = bool(res.success.reshape(-1)[0])
result["error_mm"] = float(res.position_error.reshape(-1)[0]) * 1000
"""


def solve_ik(python=None) -> dict:
    """`IK_SOLVE` through `h.ik`'s path, under `python` (default: the configured IK interpreter). Returns
    `{"solved", "error_mm"}`, or `{"error": message}`."""
    import importlib
    import tempfile
    mod = importlib.import_module("dycraft.animation.ik.ik")
    saved = os.environ.get("DYCRAFT_IK_PYTHON")
    if python:
        os.environ["DYCRAFT_IK_PYTHON"] = str(python)
    try:
        with tempfile.TemporaryDirectory(prefix="dycraft_ik_check_") as d:
            out = mod.ik(IK_SOLVE, ik_dir=d, require_scene=False, timeout_s=900)
    finally:
        if python:
            if saved is None:
                os.environ.pop("DYCRAFT_IK_PYTHON", None)
            else:
                os.environ["DYCRAFT_IK_PYTHON"] = saved
    if not out.get("ok"):
        msg = (out.get("message") or out.get("error_kind") or "?").strip().splitlines()
        return {"error": msg[-1] if msg else "?"}
    return out.get("result") or {"error": "the solve returned no result"}


_IK_LOST = "h.ik returns error_kind='unavailable'; the agent solves reaches and footsteps itself"
_IK_FIX = ("python -m dycraft.install ik   (./setup_ik.sh in a clone; needs an NVIDIA GPU, and picks the PyTorch "
           "build for it)")


#: The IK row on a machine with no NVIDIA GPU: nothing to fix, since cuRobo cannot run there.
IK_NO_GPU = ("unavailable here: no NVIDIA GPU (cuRobo has no CPU path); optional — the agent solves kinematics "
             "itself")


def _ik_module():
    import importlib
    # By module path: the package re-exports a *function* named `ik`.
    return importlib.import_module("dycraft.animation.ik.ik")


def _ik_python():
    return _ik_module()._resolve_ik_python()


def _ik_kernel_failure(got):
    """Why the GPU half of the probe failed, or None."""
    if not got.get("cuda"):
        return (f"torch {got['torch']} sees no CUDA device" if got.get("torch") else
                f"cannot import torch: {got.get('error', '?')}")
    if not got.get("kernel"):
        return (f"torch {got['torch']} cannot run a kernel on {got.get('gpu')} ({got.get('cap')}); this build has "
                f"kernels for {', '.join(got.get('arch', [])) or 'nothing'}: {got.get('error', '?')}")
    return None


@check("optional.ik", "IK (cuRobo)", S_OPTIONAL, needs=("repo.import",), caps=(CAP_IK,))
def check_ik(ctx):
    py = _ik_python()
    visible, why = _ik_module().gpu_visible()
    if not visible:
        if not py:
            return unavailable(IK_NO_GPU, lost=_IK_LOST)
        return unavailable(f"installed, but no NVIDIA GPU is visible ({why}); runs leave h.ik out until one is",
                           lost=_IK_LOST, detail=[f"interpreter: {py}"])
    if not py:
        return unavailable("not configured (DYCRAFT_IK_PYTHON, .ik_python)", lost=_IK_LOST, fix=[_IK_FIX])
    got = marked_json(run([py, "-c", IK_PROBE], timeout=240)) or {}
    bad = _ik_kernel_failure(got)
    if bad:
        return unavailable(bad, lost=_IK_LOST, fix=[_IK_FIX],
                           detail=["cuRobo has no CPU path, and a torch without kernels for the GPU fails every "
                                   "call"])
    if "curobo" not in got:
        return unavailable(f"{py} cannot import curobo: {got.get('error', '?')}", lost=_IK_LOST, fix=[_IK_FIX])
    return ok(f"torch {got['torch']} runs a kernel on {got['gpu']} ({got['cap']}); --deep solves IK — {py}")


@check("optional.fluid", "fluid simulation", S_OPTIONAL, needs=("blender.animation",),
       caps=(CAP_FLUID,))
def check_fluid(ctx):
    if ctx.facts.get("blender_binary"):
        return ok("official Blender binary (Mantaflow works)  — baked by --deep")
    return unavailable("the animation layer runs the bpy wheel, whose Mantaflow bindings abort "
                       "every fluid bake",
                       lost="liquid and gas effects cannot be baked",
                       fix=["install a blender.org 4.5 LTS build and record it in .blender_binary"])


def _files_by_extension(directory: Path) -> str:
    """`directory`'s non-hidden files at any depth, counted by extension, most common first ("2,548 .bvh, 2 .txt").
    Directory links are followed, each real directory once, as `runtime.mocap_libraries` follows them."""
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for here, dirs, files in os.walk(directory, followlinks=True):
        real = os.path.realpath(here)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if not name.startswith("."):
                ext = Path(name).suffix.lower() or "other"
                counts[ext] = counts.get(ext, 0) + 1
    return ", ".join(f"{n:,} {ext}" for ext, n in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


@check("optional.mocap", "motion capture", S_OPTIONAL, needs=("repo.import",), caps=(CAP_MOCAP,))
def check_mocap(ctx):
    """Report-only: the motion-capture libraries under `assets/mocap/`, each with its files by extension.

    The libraries come from `runtime.mocap_libraries`, which also decides whether agents are shown mocap, so this
    report and what the agents see agree. Absent data is an optional capability, never a failure."""
    from dycraft.animation.blender.runtime import mocap_libraries
    lib = _home.state_path("assets", "mocap").resolve()
    libraries = mocap_libraries()
    if not libraries:
        return unavailable(f"no library in {lib}",
                           lost="agents animate without captured reference motion, and the "
                                "mocap_reference skill is not offered",
                           fix=["python -m dycraft.install mocap fetches the CMU library (--dog adds the MANN "
                                f"dog captures), or put any capture library in its own folder under {lib}"])
    found = "; ".join(f"{name} {_files_by_extension(path)}" for name, path in libraries.items())
    if "mann_dog" in libraries and not (libraries["mann_dog"] / "index.txt").is_file():
        index = libraries["mann_dog"] / "index.txt"
        return warn(f"{found}, but {index} is missing; install mocap --dog rewrites it, with the captures' notes",
                    lost="the dog helpers cannot find the labeled spans of the captures",
                    fix=["python -m dycraft.install mocap --dog"])
    return ok(found)


@check("optional.library", "asset library", S_OPTIONAL, needs=("repo.import",))
def check_library(ctx):
    """Report-only: the installed asset library (`python -m dycraft.install library`), read from its catalog's
    `library_version` and `assets` (one entry per character). Absent, it is an optional part, never a failure.

    Agents are told about the library only through its `INDEX.md`, found by `asset_library.installed_index`, the
    same test that decides whether the animation prompt shows its asset-library section; so a library linked
    without one is a warning, and this report and what the agents see agree."""
    root = _library_root()
    broken = next((p for p in (root, root.parent) if p.is_symlink() and not p.exists()), None)
    if broken is not None:
        return unavailable(f"{broken} is a broken link to {os.readlink(broken)}",
                           lost="animation agents cannot cast characters from the library",
                           fix=[f"restore what it points at, or remove the link (`rm {broken}`) and run "
                                "`python -m dycraft.install library`"])
    try:
        catalog = json.loads((root / "catalog.json").read_text())
    except (OSError, ValueError):
        catalog = None
    if _installed_index(root) is None:
        if catalog is None:
            return unavailable("not installed", detail=[f"looked in {root}"],
                               lost="animation agents cannot cast characters from the library",
                               fix=["python -m dycraft.install library"])
        return warn("the library is linked but has no INDEX.md, so agents are not told about it",
                    detail=[str(root)],
                    lost="the animation prompt leaves out the asset library, so agents build the characters a "
                         "brief needs",
                    fix=["python -m dycraft.install library"])
    if catalog is None:                        # a library built from your own imports, indexed by asset_library
        try:
            count = len(json.loads((root / "index.json").read_text()).get("assets") or [])
        except (OSError, ValueError):
            count = 0
        return ok(f"your own library, {count} character{'s' * (count != 1)}", detail=[str(root)])
    count = len(catalog.get("assets") or [])
    return ok(f"{catalog.get('library_version') or 'unknown version'}, {count} character{'s' * (count != 1)}",
              detail=[str(root)])


def _installed_index(root: Path):
    """The library's INDEX.md as the prompt's test finds it, or None. `root` is `_library_root()`, which is the
    prompt's own `library_root()` whenever a library exists. A failed import is not caught: it would otherwise read
    as a library without an index."""
    from dycraft.agent.assets.asset_library import installed_index
    return installed_index(root)


def _library_root() -> Path:
    """Where the library is installed: the library module's own answer, else `<home>/assets/imported`."""
    try:
        from dycraft.agent.assets.asset_library import library_root
        root = library_root()                  # None when there is no library
    except (ImportError, AttributeError):
        root = None
    return Path(root) if root else _home.state_path("assets", "imported")


@check("optional.viewer", "results viewer", S_OPTIONAL, caps=(CAP_VIEWER,))
def check_viewer(ctx):
    facts = package_facts(ctx)
    missing = [r.dist for r in REQUIREMENTS if r.group == "viewer" and not facts.get(r.dist, {}).get("ok")]
    if missing:
        return unavailable("missing " + ", ".join(missing),
                           lost="results can be generated but not browsed in the viewer",
                           fix=[install_cmd(ctx, *_pins(("viewer",), markers=False))])
    return ok("python -m dycraft.viewer can start")


# ===========================================================================
# --deep: every subsystem through its real code path, against a known answer
# ===========================================================================

S_DEEP = "Deep: subsystems through their real code paths"

_COAL_CODE = r'''
import json, numpy as np, coal
a = coal.CollisionObject(coal.Box(1.0, 1.0, 1.0), coal.Transform3s())
b = coal.CollisionObject(coal.Box(1.0, 1.0, 1.0), coal.Transform3s(np.eye(3), np.array([3.0, 0.0, 0.0])))
res = coal.DistanceResult()
d = coal.distance(a, b, coal.DistanceRequest(), res)
print("ENVTEST_JSON " + json.dumps({"distance": float(d)}))
'''


@check("deep.coal", "coal distance query", S_DEEP, tier="deep", needs=("pkg.coal",),
       caps=(CAP_DYCRAFT,))
def deep_coal(ctx):
    ran = run([sys.executable, "-c", _COAL_CODE], timeout=120)
    got = marked_json(ran)
    if not got:
        return fail("coal could not answer a distance query", detail=ran.tail(4),
                    lost="collision checks raise")
    if abs(got["distance"] - 2.0) > 1e-6:
        return fail(f"two unit boxes 3 m apart measured {got['distance']:.6f} m, not 2",
                    lost="collision verdicts are wrong")
    return ok("two unit boxes 3 m apart: 2.000000 m")


@check("deep.ik", "cuRobo IK round trip", S_DEEP, tier="deep", needs=("optional.ik",), caps=(CAP_IK,))
def deep_ik(ctx):
    got = solve_ik()
    if not got.get("solved"):
        return unavailable(f"cuRobo did not solve its own Franka back to an FK pose: "
                           f"{got.get('error') or 'error %.3f mm' % got.get('error_mm', float('nan'))}",
                           lost=_IK_LOST, fix=[_IK_FIX])
    return ok(f"Franka, FK pose solved back to {got['error_mm']:.3f} mm through h.ik's path")


_CAD_BUILD = r'''
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
report = {}
SRC = """
red = Material(color=(220, 40, 40))
obj = Object("envtest_block")
shape = Box(0.2, 0.12, 0.08) - Pos(0, 0, 0.02) * Cylinder(0.03, 0.1)
body = obj.add_part("body", shape, material=red)
obj.declare_root(body)
ir = obj.finalize()
"""
from dycraft.agent.cad.object_worker import run_object_source
out = run_object_source(SRC, run_dir=root / "cad")
report["cad_ok"] = bool(out.get("ok"))
report["cad_error"] = str(out.get("build_error") or out.get("sandbox_violation") or "")[:800]
if report["cad_ok"]:
    ir = out["ir"]
    report["parts"] = [p.get("name") for p in ir.get("parts", [])]
    try:
        from dycraft.render import RenderRequest, render_request
        res = render_request(RenderRequest(layer="object", ir=ir, catalog=None, target="all",
                                           out_dir=root / "renders", step=0,
                                           asset_base_dir=root / "cad"))
        from PIL import Image
        import numpy as np
        stats = []
        for label, path in res.ordered_items():
            px = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3).astype(int)
            red = (px[:, 0] > 140) & (px[:, 1] < 0.6 * px[:, 0]) & (px[:, 2] < 0.6 * px[:, 0])
            stats.append({"view": label, "red": round(float(red.mean()), 4), "path": str(path)})
        report["render_ok"] = True
        report["views"] = stats
    except Exception as exc:
        report["render_ok"] = False
        report["render_error"] = f"{type(exc).__name__}: {exc}"[:800]
print("ENVTEST_JSON " + json.dumps(report))
'''


@check("deep.cad", "CAD build + stills render", S_DEEP, tier="deep",
       needs=("cad.interpreter", "render.stills", "pkg.pillow"), caps=(CAP_DYCRAFT,))
def deep_cad(ctx):
    root = ctx.tmp / "deep_cad"
    ran = run([sys.executable, "-c", _CAD_BUILD, root], timeout=900, env=ctx.child_env())
    got = marked_json(ran)
    if not got:
        return fail("the CAD/render probe died", detail=ran.tail(6))
    if not got["cad_ok"]:
        return fail("a known-good object.py does not build in the CAD worker",
                    detail=[got["cad_error"]], lost="no object can be authored")
    if not got.get("render_ok"):
        return fail(f"the part built, but the stills renderer failed: {got.get('render_error')}",
                    lost="snapshots and storyboard stills fail")
    views = got["views"]
    blank = [v["view"] for v in views if v["red"] < 0.002]
    if len(blank) == len(views):
        return fail(f"all {len(views)} views rendered, but none shows the red part",
                    detail=[f"e.g. {views[0]['path']}"],
                    lost="renders come back blank: snapshots show nothing")
    detail = [f"views without the part: {', '.join(blank)}"] if blank else []
    best = max(views, key=lambda v: v["red"])
    return ok(f"built 1 part; {len(views)} views, the part fills up to {best['red']:.0%} "
              f"of {best['view']}", detail=detail)


_BLENDER_PROBE = r'''
import json, os, sys, time, traceback
sys.path.insert(0, "/tools")
report = {}
def step(name, fn):
    t = time.time()
    try:
        report[name] = {"ok": True, "value": fn()}
    except BaseException as exc:
        report[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1500],
                        "trace": traceback.format_exc()[-3000:]}
    report[name]["s"] = round(time.time() - t, 2)
import bpy
step("bpy", lambda: bpy.app.version_string)
step("numpy", lambda: __import__("numpy").__version__)
step("worker", lambda: (__import__("worker"), "imported")[1])
def coal_child():
    from collision import CoalClient
    client = CoalClient()
    try:
        return client.version
    finally:
        client.close()
step("coal", coal_child)
def build():
    import mathutils
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"
    mat = bpy.data.materials.new("envtest_red")
    mat.diffuse_color = (0.8, 0.02, 0.02, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.8, 0.02, 0.02, 1.0)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0.25))
    cube = bpy.context.active_object
    cube.data.materials.append(mat)
    cube.keyframe_insert("location", frame=1)
    cube.location.x = 0.5
    cube.keyframe_insert("location", frame=10)
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
    bpy.context.active_object.data.energy = 3.0
    bpy.ops.object.camera_add(location=(1.8, -1.8, 1.4))
    cam = bpy.context.active_object
    cam.rotation_euler = (mathutils.Vector((0.25, 0, 0.25)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    scene.world = bpy.data.worlds.new("envtest_world")
    scene.world.color = (0.05, 0.05, 0.05)
    scene.frame_start, scene.frame_end = 1, 10
    xs = []
    for frame in (1, 10):
        scene.frame_set(frame)
        xs.append(round(cube.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.translation.x, 4))
    return {"x_frame_1": xs[0], "x_frame_10": xs[1]}
step("evaluate", build)
def rigid():
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, -2))
    ground = bpy.context.active_object
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(-1.5, 0, 1.0))
    ball = bpy.context.active_object
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    scene.rigidbody_world.point_cache.frame_start = 1
    scene.rigidbody_world.point_cache.frame_end = 30
    for frame in range(1, 31):
        scene.frame_set(frame)
    z = ball.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.translation.z
    bpy.data.objects.remove(ball)
    bpy.data.objects.remove(ground)
    scene.frame_set(1)
    return {"z_frame_30": round(z, 3)}
step("rigid_body", rigid)
def render(engine):
    import numpy as np
    scene = bpy.context.scene
    scene.frame_set(1)
    scene.render.engine = engine
    scene.render.resolution_x = scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    if engine == "BLENDER_WORKBENCH":
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
    else:
        scene.eevee.taa_render_samples = 4
    path = f"{os.environ.get('DYCRAFT_BLENDER_OUTPUT_ROOT') or '/output'}/envtest_{engine.lower()}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)
    red = (px[:, 0] > 0.3) & (px[:, 1] < 0.5 * px[:, 0]) & (px[:, 2] < 0.5 * px[:, 0])
    return {"red": round(float(red.mean()), 4)}
engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
report["eevee_engine"] = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
step("render_workbench", lambda: render("BLENDER_WORKBENCH"))
step("render_eevee", lambda: render(report["eevee_engine"]))
def cycles_devices():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    found = {}
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        found[kind] = sorted({d.name for d in prefs.devices if d.type == kind})
    return found
step("cycles_devices", cycles_devices)
print("ENVTEST_JSON " + json.dumps(report), flush=True)
'''

_FLUID_PROBE = r'''
import json, os, time, traceback
report = {}
t = time.time()
try:
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 4
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.5))
    domain = bpy.context.active_object
    mod = domain.modifiers.new("fluid", "FLUID")
    mod.fluid_type = "DOMAIN"
    ds = mod.domain_settings
    ds.domain_type = "LIQUID"
    ds.resolution_max = 12
    ds.cache_type = "ALL"
    ds.cache_directory = "/output/fluid_cache"
    ds.cache_frame_start, ds.cache_frame_end = 1, 4
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 0.6))
    flow = bpy.context.active_object
    fm = flow.modifiers.new("fluid", "FLUID")
    fm.fluid_type = "FLOW"
    fm.flow_settings.flow_type = "LIQUID"
    fm.flow_settings.flow_behavior = "GEOMETRY"
    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)
    with bpy.context.temp_override(object=domain, active_object=domain):
        result = bpy.ops.fluid.bake_all()
    files = [f for _, _, fs in os.walk("/output/fluid_cache") for f in fs]
    report = {"ok": "FINISHED" in result and bool(files), "result": sorted(result), "files": len(files)}
except BaseException as exc:
    report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
report["s"] = round(time.time() - t, 2)
print("ENVTEST_JSON " + json.dumps(report), flush=True)
'''


def sandboxed_blender(ctx, code: str, name: str, timeout: int = 900) -> tuple:
    """Run `code` in the exact sandbox `runtime.command` builds for a worker —
    same namespaces, binds, closed environment, and Blender — with only the final
    program swapped for `code`."""
    from dycraft.animation.blender import runtime
    root = ctx.tmp / name
    (root / "ws").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(exist_ok=True)
    (root / "job.json").write_text("{}")
    cmd = runtime.command(root / "ws", root / "out", root / "job.json")
    tail = runtime.program_of(cmd)
    exe, binary = tail[0], "--background" in tail
    program = ([exe, "--background", "--factory-startup", "--python-use-system-env",
                "--python-exit-code", "1", "--python-expr", code] if binary
               else [exe, "-B", "-c", code])
    ran = run(runtime.with_program(cmd, program), timeout=timeout, env=runtime.worker_env())
    return marked_json(ran), ran, binary


@check("deep.blender", "Blender in the sandbox", S_DEEP, tier="deep",
       needs=("blender.animation", "sandbox.userns", "sandbox.coal_view"), caps=(CAP_DYCRAFT,))
def deep_blender(ctx):
    got, ran, binary = sandboxed_blender(ctx, _BLENDER_PROBE, "deep_blender")
    if not got:
        return fail("Blender did not complete a probe inside the sandbox", detail=ran.tail(8),
                    lost="every Blender worker job fails")
    ctx.facts["blender_probe"] = got
    step = {k: v for k, v in got.items() if isinstance(v, dict)}
    fatal = [(k, v) for k, v in step.items() if k in ("bpy", "numpy", "worker", "coal", "evaluate")
             and not v["ok"]]
    if fatal:
        return fail("inside the sandbox: " + "; ".join(f"{k}: {v['error']}" for k, v in fatal),
                    detail=fatal[0][1].get("trace", "").strip().splitlines()[-8:],
                    lost={"coal": "collision checks fail inside every worker"}.get(fatal[0][0],
                          "Blender worker jobs fail"))
    ev = step["evaluate"]["value"]
    if abs(ev["x_frame_10"] - 0.5) > 1e-3:
        return fail(f"a keyframed cube is at x={ev['x_frame_10']} on frame 10, not 0.5",
                    lost="evaluated animation does not match what was authored")
    lines, status, lost = [], PASS, ""
    rigid = step["rigid_body"]
    if not rigid["ok"] or abs(rigid["value"]["z_frame_30"] - (-1.9)) > 0.05:
        status, lost = DEGRADED, "rigid-body simulation does not settle as expected"
        lines.append(f"rigid body: {rigid.get('error') or rigid['value']}")
    wb, ee = step["render_workbench"], step["render_eevee"]
    if not wb["ok"] or wb["value"]["red"] < 0.005:
        return fail(f"a Workbench render does not show the red cube: {wb.get('error') or wb['value']}",
                    lost="snapshots, storyboards, and diagnostic frames come back blank")
    if not ee["ok"] or ee["value"]["red"] < 0.005:
        status, lost = DEGRADED, f"{got['eevee_engine']} renders fail or come back blank"
        lines.append(f"EEVEE: {ee.get('error') or ee['value']}")
    devices = step["cycles_devices"].get("value") or {}
    gpus = sorted({n for names in devices.values() for n in names})
    lines.append(f"Cycles devices inside the sandbox: {', '.join(gpus) or 'CPU only'}")
    return Result(status, f"{'binary' if binary else 'bpy'} {step['bpy']['value']}: worker imports, "
                          f"coal {step['coal']['value']} answers, keyframes evaluate, rigid body "
                          f"lands, Workbench + EEVEE draw the cube", detail=lines, lost=lost)


@check("deep.fluid", "fluid bake in the sandbox", S_DEEP, tier="deep",
       needs=("optional.fluid", "deep.blender"), caps=(CAP_FLUID,))
def deep_fluid(ctx):
    got, ran, _ = sandboxed_blender(ctx, _FLUID_PROBE, "deep_fluid", timeout=600)
    if not got or not got.get("ok"):
        return unavailable(f"a 12³ liquid bake did not finish: "
                           f"{(got or {}).get('error') or (got or {}).get('result') or ran.tail(1)}",
                           lost="liquid and gas effects cannot be baked")
    return ok(f"a 12³ liquid domain baked {got['files']} cache files in {got['s']} s")


@check("deep.ffmpeg", "H.264 encode + probe", S_DEEP, tier="deep", needs=("tools.ffmpeg",),
       caps=(CAP_DYCRAFT,))
def deep_ffmpeg(ctx):
    root = ctx.tmp / "deep_ffmpeg"
    for i in range(3):
        write_png(root / "frames" / f"{i:06d}.png", 64, 64,
                  lambda x, y, i=i: (40 + 60 * i, 80, 200 - 50 * i))
    clip = root / "clip.mp4"
    # The command `animation/delivery.py` runs, and the probe it checks with.
    enc = run(["ffmpeg", "-y", "-v", "error", "-framerate", "24", "-i", root / "frames/%06d.png",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], timeout=120)
    if enc.rc != 0:
        return fail("ffmpeg could not encode three PNG frames to H.264", detail=enc.tail(3),
                    lost="final_render cannot produce the film")
    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=codec_name,pix_fmt,nb_read_frames", "-of", "json", clip],
                timeout=60)
    try:
        stream = json.loads(probe.out)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return fail("ffprobe could not read the encoded clip", detail=probe.tail(3))
    if (stream.get("codec_name"), stream.get("pix_fmt"), stream.get("nb_read_frames")) != \
            ("h264", "yuv420p", "3"):
        return fail(f"encoded clip reads back as {stream}", lost="delivery's own check would refuse it")
    return ok("3 frames → h264/yuv420p → 3 frames read back")


# ===========================================================================
# --e2e: the pipeline end to end, with a script where the model would be
# ===========================================================================

S_E2E = "End to end (no model)"

_STUB_AGENT = r'''#!{python}
"""Stands in for the claude CLI: finds the task's token and submits it."""
import os, re, subprocess, sys
text = " ".join(sys.argv[1:])
m = re.search(r"ENVTEST-[0-9a-f]{{12}}", text)
answer = m.group(0) if m else "NO-TOKEN-IN-ARGV"
py = os.environ.get("DYCRAFT_PYTHON", sys.executable)
done = subprocess.run([py, "-c", f"import harness as h; h.submit_result(text={{answer!r}})"],
                      capture_output=True, text=True)
with open({log!r}, "a") as fh:
    fh.write(f"cwd={{os.getcwd()}}\nargv0={{sys.argv[0]}} add_dir={{'--add-dir' in sys.argv}}\n"
             f"submit rc={{done.returncode}}\n{{done.stdout[-400:]}}\n{{done.stderr[-800:]}}\n")
'''

_ONEOFF = r'''
import json, sys
from pathlib import Path
from dycraft.agent.oneoff import run_oneoff
root, backend, model, effort, prompt, timeout = sys.argv[1:7]
res = run_oneoff(prompt=prompt, backend=backend, run_dir=Path(root), model=model or None,
                 reasoning_effort=effort or None, timeout_s=float(timeout))
print("ENVTEST_JSON " + json.dumps({k: res.get(k) for k in (
    "ok", "text", "n_op_calls", "outcome", "timed_out", "returncode", "wall_time_s",
    "stderr_tail", "sub_run")}))
'''


@check("e2e.launcher", "launcher → daemon → harness", S_E2E, tier="e2e", needs=("repo.import",),
       caps=(CAP_DYCRAFT,))
def e2e_launcher(ctx):
    root = ctx.tmp / "e2e_launcher"
    (root / "bin").mkdir(parents=True, exist_ok=True)
    stub, log = root / "bin" / "claude", root / "stub.log"
    stub.write_text(_STUB_AGENT.format(python=sys.executable, log=str(log)))
    stub.chmod(0o755)
    token = f"ENVTEST-{random.getrandbits(48):012x}"
    env = ctx.child_env(DYCRAFT_CLAUDE_EXECUTABLE=stub)
    env.pop("CLAUDE_CONFIG_DIR", None)
    ran = run([sys.executable, "-c", _ONEOFF, root, "claude", "", "",
               f"Reply with exactly this token: {token}", "240"], timeout=420, env=env)
    got = marked_json(ran)
    stub_log = log.read_text(errors="replace").splitlines()[-6:] if log.is_file() else ["the stub never ran"]
    if not got:
        return fail("run_oneoff did not report", detail=ran.tail(6) + stub_log)
    if got.get("text") != token or not got.get("n_op_calls"):
        return fail(f"a scripted agent's submit_result did not come back "
                    f"(outcome={got.get('outcome')}, ops={got.get('n_op_calls')})",
                    detail=[(got.get("stderr_tail") or "")[-500:]] + stub_log,
                    lost="the DyCraft launch path (cli.py → daemon → harness.py) is broken "
                         "before any model is involved")
    return ok(f"python -m dycraft.cli --layer plain, a scripted agent submitted its answer: "
              f"{got['n_op_calls']} op call(s) in {got.get('wall_time_s')} s")


_E2E_ANIMATION = r'''
import json, sys, time
from pathlib import Path
from dycraft.tests._animation_fixtures import ANIMATE, STAGE, call, plan, service, source, state_reviews
from dycraft.animation.blender.bundle import validate_manifest
root = Path(sys.argv[1])
steps = []
def step(name, fn):
    t = time.time()
    r = fn()
    good = bool(r.get("ok")) if isinstance(r, dict) else bool(r)
    steps.append({"step": name, "ok": good, "s": round(time.time() - t, 1),
                  "error": None if good else str((r or {}).get("error") or r)[:900]})
    if not good:
        raise SystemExit
cheap = {"engine": "WORKBENCH", "width": 128, "height": 96, "threads": 2}
svc = service(root)
try:
    (svc.design / "scene.py").write_text(source())
    step("plan", lambda: call(svc, "plan", plan=plan()))
    step("build (CAD child → Blender import)", lambda: call(svc, "build", timeout=1200))
    step("stage (Blender edit)", lambda: call(svc, "blender", code=STAGE, stage="storyboard", timeout=900))
    step("storyboard (renders)", lambda: call(svc, "storyboard", render_settings=cheap, timeout=900))
    step("accept storyboard", lambda: call(svc, "accept_storyboard",
                                           reviews=state_reviews("environment_test fixture")))
    step("animate (Blender edit)", lambda: call(svc, "blender", code=ANIMATE, timeout=900))
    # The clean path checks the frames before the film is drawn: a film over unchecked frames
    # cannot be committed without exceptions.
    step("compile (verifiers)", lambda: call(svc, "compile", timeout=1200))
    step("final render (film)", lambda: call(svc, "final_render", timeout=1200,
        render_settings={"engine": "WORKBENCH", "width": 64, "height": 64, "threads": 2}))
    # finish() needs a verdict for every plan criterion, judged on the finished film; this fixture's
    # film is scripted, so each criterion is passed with that stated as the evidence.
    verdicts = {c["id"]: {"verdict": "pass", "evidence": "environment_test: scripted fixture film"}
                for c in plan()["criteria"]}
    step("finish (verifiers → bundle)", lambda: call(svc, "finish", criteria=verdicts, timeout=1200))
    manifest = json.loads((root / "ir.json").read_text())
    validate_manifest(root, manifest)
    steps.append({"step": "manifest", "ok": True, "n_frames": manifest.get("n_frames"),
                  "film": manifest.get("film")})
except SystemExit:
    pass
except BaseException as exc:
    steps.append({"step": "exception", "ok": False, "error": f"{type(exc).__name__}: {exc}"[:900]})
finally:
    svc.jobs.close()
    print("ENVTEST_JSON " + json.dumps(steps), flush=True)
'''


@check("e2e.animation", "animation pipeline", S_E2E, tier="e2e",
       needs=("cad.interpreter", "blender.animation", "sandbox.userns", "sandbox.coal_view",
              "tools.ffmpeg"), caps=(CAP_DYCRAFT,))
def e2e_animation(ctx):
    root = ctx.tmp / "e2e_animation"
    root.mkdir(parents=True, exist_ok=True)
    ran = run([sys.executable, "-c", _E2E_ANIMATION, root], timeout=3600, env=ctx.child_env())
    steps = marked_json(ran)
    if not steps:
        return fail("the pipeline script died", detail=ran.tail(8))
    lines = [f"{'✔' if s['ok'] else '✖'} {s['step']}" + (f" ({s['s']} s)" if "s" in s else "")
             for s in steps]
    bad = next((s for s in steps if not s["ok"]), None)
    if bad:
        return fail(f"stopped at `{bad['step']}`: {bad['error']}", detail=lines,
                    lost="the DyCraft animation arm cannot deliver a result on this machine")
    ctx.facts["e2e_blend"] = root / "bundle" / "animation.blend"
    last = steps[-1]
    total = sum(s.get("s", 0) for s in steps)
    return ok(f"plan → CAD → Blender → storyboard → compile → film → finish: {last.get('n_frames')} frames, "
              f"committed bundle validates ({total:.0f} s)", detail=lines)


_STAGE = r"""
import json, sys
from pathlib import Path
from dycraft.benchmark.stage import stage
record = stage(Path(sys.argv[1]), Path(sys.argv[2]), scene_ref="environment_test",
               render_missing_film=True)
print("ENVTEST_JSON " + json.dumps({"evaluation": record.get("evaluation"), "film": record.get("film"),
                                    "staged": (Path(sys.argv[2]) / "ir.json").is_file()}, default=str))
"""


@check("e2e.bare_staging", "bare-arm staging", S_E2E, tier="e2e",
       needs=("e2e.animation", "bare.code", "blender.bpy"), caps=(CAP_STAGING,))
def e2e_bare_staging(ctx):
    """The committed .blend, staged as a bare arm's would be, with a film rendered for it (benchmark.stage)."""
    blend = ctx.facts["e2e_blend"]
    out = ctx.tmp / "e2e_bare_stage"
    ran = run([sys.executable, "-c", _STAGE, blend, out], timeout=1800, env=ctx.child_env())
    got = marked_json(ran)
    if not got or not got.get("staged"):
        return fail("benchmark.stage could not stage a committed .blend", detail=ran.tail(6),
                    lost="bare arms' .blend files cannot be shown in the viewer")
    film = got.get("film") or {}
    if film.get("source") != "harness":
        return degraded("staged, but no film could be rendered for it",
                        detail=[str(film.get("error"))[:600]],
                        lost="a bare arm that delivers no film of its own gets none")
    frames = (got.get("evaluation") or {}).get("n_frames")
    return ok(f"the committed .blend, staged as a bare arm's: {frames} frames evaluated, "
              f"{film.get('frames')} rendered at {(film.get('settings') or {}).get('width')}x"
              f"{(film.get('settings') or {}).get('height')}")


# ===========================================================================
# harnesses: built-in specs for claude and codex, JSON specs for new ones
# ===========================================================================

_SPEC_FIELDS = {
    "name": "a label",
    "binary": "executable name or path",
    "config_dir_env": "env var that selects the account (e.g. NEWCLI_HOME), or null",
    "config_dir": "the account directory the probes authenticate as, or null",
    "credentials": "file inside config_dir that holds a login, or null",
    "model_args": 'e.g. ["--model", "{model}"]',
    "effort_args": 'e.g. ["--effort", "{effort}"], or null',
    "system_args": 'e.g. ["--system", "{system}"], or null when there is no system channel',
    "oneshot": 'headless one-turn argv; answer on stdout: ["{binary}", "run", "{model_args}", "{prompt}"]',
    "agentic": 'headless tool-using turn confined to {cwd}; must carry "--add-dir", "{cwd}"-style '
               'argv the daemon can match',
    "read_deny": "true if the harness can deny reads of paths (bare-arm blindness)",
    "transcript_glob": 'e.g. "{config_dir}/sessions/**/*.jsonl", or null',
    "cmdline_marker": "substring of the agent process's argv (the daemon's watchdog)",
    "usage_limit_patterns": "regexes that identify a usage-limit message",
}


@dataclass
class HarnessSpec:
    name: str
    family: str
    binary: str | None
    oneshot: list
    agentic: list
    config_dir_env: str | None = None
    config_dir: str | None = None
    pin_config_dir: bool = False
    credentials: str | None = None
    model_args: list | None = None
    effort_args: list | None = None
    system_args: list | None = None
    settings: str | None = None          # "claude": write the bare-arm sandbox settings
    read_deny: bool = False
    transcript_glob: str | None = None
    cmdline_marker: str | None = None
    usage_limit_patterns: list = field(default_factory=list)
    backend: str | None = None           # a cli/backends/ backend name, when there is one


def builtin_spec(name: str) -> HarnessSpec:
    from dycraft.cli.backends import BACKENDS, ClaudeBackend, CodexBackend
    cls = BACKENDS[name]
    if issubclass(cls, ClaudeBackend):
        exe, pinned = cls._resolve_executable()
        return HarnessSpec(
            name=name, family="claude", backend=name, binary=which(exe),
            config_dir_env="CLAUDE_CONFIG_DIR", config_dir=str(cls._config_dir()),
            pin_config_dir=bool(pinned or os.environ.get("CLAUDE_CONFIG_DIR")),
            credentials=".credentials.json",
            model_args=["--model", "{model}"], effort_args=["--effort", "{effort}"],
            system_args=["--append-system-prompt", "{system}"],
            # How the paper's bare claude arms were invoked.
            oneshot=["{binary}", "--print", "--permission-mode", "bypassPermissions",
                     "--settings", "{settings}", "{model_args}", "{effort_args}", "{system_args}",
                     "--", "{prompt}"],
            # How DyCraft launches it (`claude_agent_settings`, `dontAsk`), so the confinement and
            # read-deny probes describe DyCraft's agents.
            agentic=["{binary}", "--print", "--add-dir", "{cwd}", "--permission-mode",
                     "dontAsk", "--settings", "{settings}", "--strict-mcp-config",
                     "{model_args}", "{effort_args}", "--", "{prompt}"],
            settings="claude", read_deny=True,
            transcript_glob="{config_dir}/projects/**/*.jsonl", cmdline_marker="claude")
    if issubclass(cls, CodexBackend):
        binary, home = cls._resolve_codex_cli()
        return HarnessSpec(
            name=name, family="codex", backend=name, binary=binary and which(binary),
            config_dir_env="CODEX_HOME", config_dir=home, pin_config_dir=True,
            credentials="auth.json", model_args=["-m", "{model}"],
            effort_args=["-c", "model_reasoning_effort={effort}"], system_args=None,
            # How the paper's bare codex arms were invoked.
            oneshot=["{binary}", "exec", "--skip-git-repo-check", "-C", "{cwd}", "-s",
                     "workspace-write", "-c", "check_for_update_on_startup=false",
                     "{model_args}", "{effort_args}", "{prompt}"],
            # With DyCraft's sandbox and approval flags (`CodexBackend.execute`), so the confinement
            # probe describes DyCraft's agents.
            agentic=["{binary}", "exec", "--skip-git-repo-check", "-C", "{cwd}", "--add-dir",
                     "{cwd}", "--sandbox", "workspace-write", "-c", 'approval_policy="never"',
                     "-c", "sandbox_workspace_write.network_access=true",
                     "-c", "check_for_update_on_startup=false",
                     "{model_args}", "{effort_args}", "{prompt}"],
            transcript_glob="{config_dir}/sessions/**/rollout-*.jsonl", cmdline_marker="codex")
    raise ValueError(f"{name}: not a CLI backend")


def load_spec(path: str) -> HarnessSpec:
    raw = json.loads(Path(path).read_text())
    unknown = set(raw) - set(_SPEC_FIELDS) - {"binary_env"}
    if unknown:
        raise SystemExit(f"{path}: unknown field(s) {sorted(unknown)}; known: {sorted(_SPEC_FIELDS)}")
    for need in ("name", "binary", "oneshot", "agentic"):
        if need not in raw:
            raise SystemExit(f"{path}: `{need}` is required ({_SPEC_FIELDS[need]})")
    binary = os.environ.get(raw.get("binary_env") or "", "") or raw["binary"]
    config_dir = raw.get("config_dir")
    return HarnessSpec(
        name=raw["name"], family="custom", binary=which(str(Path(binary).expanduser())),
        oneshot=raw["oneshot"], agentic=raw["agentic"],
        config_dir_env=raw.get("config_dir_env"),
        config_dir=str(Path(config_dir).expanduser()) if config_dir else None,
        pin_config_dir=bool(raw.get("config_dir_env") and config_dir),
        credentials=raw.get("credentials"), model_args=raw.get("model_args"),
        effort_args=raw.get("effort_args"), system_args=raw.get("system_args"),
        read_deny=bool(raw.get("read_deny")),
        transcript_glob=raw.get("transcript_glob"), cmdline_marker=raw.get("cmdline_marker"),
        usage_limit_patterns=list(raw.get("usage_limit_patterns") or []))


_GROUP_GATE = {"model_args": "model", "effort_args": "effort", "system_args": "system"}


def expand(spec: HarnessSpec, template: list, values: dict) -> list:
    out = []
    for item in template:
        m = re.fullmatch(r"\{(\w+)\}", item)
        if m and m.group(1) in _GROUP_GATE:
            group = getattr(spec, m.group(1))
            if group and values.get(_GROUP_GATE[m.group(1)]):
                out += [str(g).format(**values) for g in group]
            continue
        out.append(str(item).format(**values))
    return out


def bare_settings(cwd: Path, deny: list = ()) -> dict:
    """The sandbox settings the paper's bare claude arms ran with."""
    from dycraft.cli.backends import claude_sandbox_socket_fallback
    return {"sandbox": claude_sandbox_socket_fallback({
                "enabled": True, "autoAllowBashIfSandboxed": True, "failIfUnavailable": True,
                "filesystem": {"allowWrite": [str(cwd)]},
                "network": {"allowedDomains": []}}),
            "permissions": {"allow": ["Bash(*)", "Read", f"Edit(//{cwd}/**)"],
                            "deny": [f"Read(//{d}/**)" for d in deny]}}


class HarnessProbe:
    """Everything the live probes of one harness share: its spec, the model, a
    scratch area, and the texts every run printed (for the usage-limit row)."""

    def __init__(self, ctx: Ctx, spec: HarnessSpec):
        self.ctx, self.spec = ctx, spec
        self.model = ctx.args.model or ""
        self.effort = ctx.args.effort or ""
        name = f"harness_{re.sub(r'[^A-Za-z0-9_]', '_', spec.name)}"
        how = ctx.facts.get("codex_launch") if spec.family == "codex" else None
        if how and how.get("wrapper") and _under_tmp(ctx.tmp):
            # Codex runs in a namespace with a fresh /tmp here (`check_codex`), which would hide a scratch area under
            # this machine's /tmp, the probe's run directory included; a launch's run directory is not there.
            scratch = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "dycraft"
            scratch.mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix=f"envtest_{name}_", dir=scratch))
            ctx.cleanup.append(self.root)
        else:
            self.root = ctx.tmp / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.outputs: list[str] = []

    def env(self, **extra) -> dict:
        env = dict(os.environ)
        if self.spec.pin_config_dir and self.spec.config_dir_env and self.spec.config_dir:
            env[self.spec.config_dir_env] = self.spec.config_dir
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def launch(self, template, *, prompt, cwd, system="", deny=(), timeout=300,
               env=None, watch=None) -> Ran:
        cwd.mkdir(parents=True, exist_ok=True)
        values = {"binary": self.spec.binary, "model": self.model, "effort": self.effort,
                  "prompt": prompt, "system": system, "cwd": str(cwd),
                  "config_dir": self.spec.config_dir or "", "settings": ""}
        if self.spec.settings == "claude":
            settings = cwd.parent / f"{cwd.name}.settings.json"
            if template is self.spec.agentic:
                from dycraft.cli.backends import claude_agent_settings
                body = claude_agent_settings(str(cwd), deny=[f"Read(//{d}/**)" for d in deny])
            else:
                body = bare_settings(cwd, deny)
            settings.write_text(json.dumps(body))
            values["settings"] = str(settings)
        cmd, env = expand(self.spec, template, values), env or self.env()
        how = self.ctx.facts.get("codex_launch") if self.spec.family == "codex" else None
        if how and template is self.spec.agentic:
            # As `CodexBackend` starts it here: in a namespace of its own and with another TMPDIR, when its
            # sandbox needs them (`check_codex`).
            from dycraft.cli.backends.codex import RUN_TMPDIR
            cmd = [*how["wrapper"], *cmd]
            if how["tmpdir"]:
                tmp = cwd / RUN_TMPDIR if how["tmpdir"] == RUN_TMPDIR else Path(how["tmpdir"])
                tmp.mkdir(parents=True, exist_ok=True)
                env = {**env, "TMPDIR": str(tmp)}
        ran = run(cmd, timeout=timeout, cwd=cwd, env=env, watch=watch)
        self.outputs.append(ran.text)
        return ran

    def failure(self, ran: Ran, what: str, lost: str) -> Result:
        label, line = classify(ran.text)
        if ran.timed_out:
            label, line = label or "hung", line or "no answer before the timeout"
        if ran.error:
            label, line = "did not start", ran.error
        return fail(f"{what}: {label or 'no correct answer'}" + (f" — {line}" if line else ""),
                    detail=ran.tail(4), lost=lost)


def register_backend_checks(ctx: Ctx, spec: HarnessSpec) -> list[Check]:
    """The live checks for one harness, in the order they run."""
    probe = HarnessProbe(ctx, spec)
    sec = f"Harness: {spec.name}" + (f" ({ctx.args.model})" if ctx.args.model else "")
    base = f"backend.{spec.name}"
    cap = f"harness {spec.name}"
    checks = []

    def add(suffix, title, fn, needs=(), caps=(cap,)):
        checks.append(Check(f"{base}.{suffix}", title, sec, fn, "backend",
                            tuple(f"{base}.{n}" for n in needs), caps))

    def cli(ctx):
        if not spec.binary:
            return fail(f"the {spec.name} CLI is not installed or not on PATH",
                        lost=f"nothing can run on {spec.name}")
        detail = []
        if spec.config_dir:
            creds = Path(spec.config_dir) / (spec.credentials or "")
            if not Path(spec.config_dir).is_dir():
                return fail(f"account directory {spec.config_dir} does not exist",
                            lost="the CLI starts logged out",
                            fix=[f"{spec.config_dir_env}={spec.config_dir} {spec.binary} "
                                 "(then log in)"])
            if spec.credentials and not creds.is_file():
                detail.append(f"no {spec.credentials} in {spec.config_dir}: probably not logged in")
        model = probe.model
        if not model and spec.backend:
            from dycraft.cli.backends import BACKENDS
            model = BACKENDS[spec.backend].resolve_agent_model(None) or ""
            detail.append(f"no --model: the CLI's configured model is {model or 'its built-in default'}")
        version = run([spec.binary, "--version"], timeout=30, env=probe.env()).tail(1)
        return ok(f"{(version or ['?'])[0]}; account {spec.config_dir or 'the CLI default'}",
                  detail=detail)

    def oneshot(ctx):
        prompt, answer = arithmetic()
        ran = probe.launch(spec.oneshot, prompt=prompt, cwd=probe.root / "oneshot")
        if answer in (ran.out or ""):
            return ok(f"answered correctly in {ran.seconds} s", data={"seconds": ran.seconds})
        return probe.failure(ran, "no correct answer", f"a bare {spec.name} arm cannot run")

    def dycraft_launch(ctx):
        prompt, answer = arithmetic()
        root = probe.root / "dycraft_launch"
        root.mkdir(exist_ok=True)
        ran = run([sys.executable, "-c", _ONEOFF, root, spec.backend, probe.model, probe.effort,
                   prompt, "600"], timeout=900, env=ctx.child_env())
        got = marked_json(ran) or {}
        probe.outputs.append(str(got.get("stderr_tail", "")) + ran.text)
        text = str(got.get("text") or "")
        if answer in text and got.get("n_op_calls"):
            return ok(f"answered through submit_result ({got['n_op_calls']} op call(s)) in "
                      f"{got.get('wall_time_s')} s")
        if answer in text:
            return degraded("answered, but without calling the harness",
                            lost="the submit_result path — how every DyCraft layer returns work — "
                                 "was not exercised")
        label, line = classify(str(got.get("stderr_tail", "")) + ran.text)
        return fail(f"no correct answer (outcome={got.get('outcome')}"
                    f"{', ' + label if label else ''})" + (f" — {line}" if line else ""),
                    detail=[str(got.get("stderr_tail", ""))[-600:]] + ran.tail(3),
                    lost=f"a DyCraft arm cannot run on {spec.name}")

    add("cli", "CLI and account", cli)
    add("oneshot", "one-shot round trip (bare arm)", oneshot, needs=("cli",), caps=(cap, CAP_BARE))
    if spec.backend:
        add("dycraft", "through the dycraft launcher", dycraft_launch, needs=("cli",),
            caps=(cap, CAP_DYCRAFT))
    if ctx.args.capabilities:
        capability_checks(probe, add)
    return checks


def capability_checks(probe: HarnessProbe, add) -> None:
    """The harness checklist: what `cli/backends/` relies on a CLI to do."""
    spec = probe.spec
    shared: dict = {}

    def isolation(ctx):
        if not spec.config_dir_env:
            return warn("no account variable declared",
                        lost="several accounts cannot be driven side by side")
        empty = probe.root / "empty_account"
        empty.mkdir(exist_ok=True)
        prompt, answer = arithmetic()
        ran = probe.launch(spec.oneshot, prompt=prompt, cwd=probe.root / "isolation", timeout=90,
                           env=probe.env(**{spec.config_dir_env: empty}))
        if answer in (ran.out or ""):
            leaks = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
                     if os.environ.get(k)]
            return warn(f"it answered with an empty {spec.config_dir_env}",
                        detail=[f"credentials come from elsewhere: {', '.join(leaks)} is set"]
                        if leaks else [],
                        lost=f"{spec.config_dir_env} does not select the account: runs bill "
                             "whatever credential leaks in")
        return ok(f"an empty {spec.config_dir_env} is refused, so it really selects the account")

    def system_prompt(ctx):
        code = random_digits()
        filler = "This line is padding for a size test and carries no instructions.\n" * 900
        if spec.system_args:
            ran = probe.launch(spec.oneshot, cwd=probe.root / "system",
                               system=f"The access code is {code}.\n{filler}",
                               prompt="Reply with the access code from your instructions, digits only.")
            if code in (ran.out or ""):
                return ok(f"a {len(filler) // 1024} KB system prompt reaches the model")
            return probe.failure(ran, "the system prompt did not arrive",
                                 "dycraft's layer instructions would not reach the agent")
        ran = probe.launch(spec.oneshot, cwd=probe.root / "system",
                           prompt=f"{filler}\nThe access code is {code}. Reply with the access "
                                  f"code, digits only.")
        if code in (ran.out or ""):
            return ok(f"no system channel: dycraft concatenates it into the first message, and a "
                      f"{len(filler) // 1024} KB single-argument prompt works (Linux caps one "
                      f"argument at 128 KiB)")
        return probe.failure(ran, "a large single-argument prompt did not arrive",
                             "dycraft's concatenated instructions would not reach the agent")

    def shell(ctx):
        a, b = random.randint(100, 999), random.randint(100, 999)
        outside = Path.home() / ".cache" / "dycraft" / f"envtest_outside_{random.getrandbits(32):08x}"
        outside.mkdir(parents=True, exist_ok=True)
        work = probe.root / "agentic" / "work"

        def watch(proc):
            from dycraft.agent.server import _find_agent_pid
            pid = _find_agent_pid(work, spec.cmdline_marker)
            return {"pid": pid} if pid else None

        started = time.time()
        try:
            ran = probe.launch(spec.agentic, cwd=work, timeout=600,
                               watch=watch if spec.cmdline_marker else None,
                               prompt=f"Use your shell tool to run these two commands, then reply "
                                      f"DONE.\n1) python3 -c \"print({a}*{b})\" > product.txt\n"
                                      f"2) echo leaked > {outside}/leak.txt")
            leaked = (outside / "leak.txt").exists()
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        product = (work / "product.txt").read_text().strip() if (work / "product.txt").is_file() else ""
        # `{a}*{b}` is in the prompt, so any transcript that records the turn carries it;
        # the product never appears anywhere but product.txt.
        shared.update(ran=ran, started=started, marker=f"{a}*{b}", pid=ran.watched.get("pid"),
                      leaked=leaked, outside=outside)
        if ran.timed_out:
            return fail("the tool-using turn hung (an approval prompt nobody can answer?)",
                        detail=ran.tail(3), lost="headless agents stall")
        if product != str(a * b):
            return probe.failure(ran, f"the shell tool did not run (product.txt={product or 'absent'})",
                                 "the agent cannot drive the harness, which is reached through a shell")
        return ok("ran shell commands in a headless turn, with no approval prompt")

    def confinement(ctx):
        if shared.get("leaked"):
            allowed = None
            if spec.family == "codex" and spec.config_dir:
                from dycraft.cli.backends.codex import codex_allowed_approval_policies
                allowed = codex_allowed_approval_policies(spec.config_dir)
            if allowed is not None and "never" not in allowed:
                # The sandbox refused the write; the agent then asked to run it outside, and Codex's reviewer said yes.
                return warn(f"it wrote outside its directory ({shared['outside']}), through Codex's automatic "
                            f"reviewer: this account's organization allows only {', '.join(allowed) or 'on-request'} "
                            "approvals, so the reviewer can approve a command outside the sandbox",
                            lost="on this account, write confinement rests on Codex's reviewer: an agent can modify "
                                 "the repository or another run when the reviewer approves",
                            fix=["an account whose organization allows `never`, or `--backend claude`"])
            return warn(f"it wrote outside its directory ({shared['outside']})",
                        lost="write confinement is not enforced: an agent can modify the repository "
                             "or another run")
        return ok("a write outside its directory was refused")

    def read_deny(ctx):
        if not spec.read_deny:
            return warn("no read-deny mechanism",
                               lost="a bare arm's blindness to the repository rests on its prompt alone")
        secret_dir = probe.root / "agentic" / "secret"
        secret_dir.mkdir(parents=True, exist_ok=True)
        secret = random_digits()
        (secret_dir / "secret.txt").write_text(secret + "\n")
        ran = probe.launch(spec.agentic, cwd=probe.root / "agentic" / "deny", deny=[secret_dir],
                           timeout=400,
                           prompt=f"What number is written in the file {secret_dir}/secret.txt? "
                                  f"Reply with just the number, or UNREADABLE if you cannot read it.")
        if secret in (ran.out or ""):
            return warn("a denied path was read anyway",
                        lost="bare arms can read what their sandbox settings deny them — the repository's "
                             "animation/ and verify/ code included — so they are not blind")
        if ran.timed_out:
            return warn("the read-deny turn hung", detail=ran.tail(3))
        return ok("a path under a Read deny rule stayed unreadable, by every tool the agent tried")

    def image(ctx):
        code = random_digits()
        work = probe.root / "agentic" / "image"
        digits_png(code, work / "code.png")
        ran = probe.launch(spec.agentic, cwd=work, timeout=400,
                           prompt="Look at the image file code.png in the current directory. It "
                                  "shows a six-digit number. Reply with just those six digits.")
        if code in (ran.out or ""):
            return ok(f"read {code} from a PNG on disk")
        return probe.failure(ran, "it did not read the image correctly",
                             "the agent cannot look at its own snapshots and renders")

    def transcript(ctx):
        if not spec.transcript_glob:
            return warn("no transcript location declared",
                               lost="agent_transcript.jsonl, session stats, and the viewer's "
                                    "transcript panel stay empty")
        pattern = spec.transcript_glob.format(config_dir=spec.config_dir or "")
        root = Path(pattern.split("*")[0]).parent
        rel = pattern[len(str(root)) + 1:]
        fresh = [p for p in root.glob(rel) if p.is_file() and p.stat().st_mtime >= shared["started"] - 1]
        hit = [p for p in fresh if shared["marker"] in p.read_text(errors="replace")]
        if hit:
            return ok(f"written to {hit[0]}")
        return warn(f"no transcript of the tool-using turn under {root}",
                    lost="agent_transcript.jsonl, session stats, and the viewer's transcript panel "
                         "stay empty for this harness")

    def process(ctx):
        if not spec.cmdline_marker:
            return fail("no argv marker declared",
                        lost="the daemon's watchdog cannot follow the agent process, which every "
                             "DyCraft launch needs (`Backend.agent_cmdline_marker`)")
        if shared.get("pid"):
            return ok(f"the daemon's `_find_agent_pid` found it (pid {shared['pid']})")
        return fail("the daemon's `_find_agent_pid` never matched the agent's argv",
                    lost="the watchdog cannot tell a working agent from a dead one")

    def limits(ctx):
        text = "\n".join(probe.outputs)
        patterns = list(spec.usage_limit_patterns) + [_FAILURE_PATTERNS[0][1]]
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if m:
                return warn(f"a usage-limit message appeared: {m.group(0)}",
                                lost="probes above may have failed for quota, not capability")
        return info("no usage-limit message in any probe; resume logic recognizes: "
                    + ", ".join(spec.usage_limit_patterns or ["the built-in patterns"]))

    add("cap.isolation", "account isolation", isolation, needs=("oneshot",))
    add("cap.system", "system prompt channel", system_prompt, needs=("oneshot",))
    add("cap.shell", "shell tool, headless", shell, needs=("oneshot",))
    add("cap.confinement", "write confinement", confinement, needs=("cap.shell",),
        caps=(f"harness {spec.name}", CAP_BARE, CAP_DYCRAFT))
    add("cap.read_deny", "read-deny (bare blindness)", read_deny, needs=("cap.shell",),
        caps=(f"harness {spec.name}", CAP_BARE))
    add("cap.image", "agent reads an image", image, needs=("cap.shell",))
    add("cap.transcript", "transcript on disk", transcript, needs=("cap.shell",))
    add("cap.process", "process found by argv", process, needs=("cap.shell",))
    add("cap.limits", "usage-limit messages", limits)


# ===========================================================================
# the report
# ===========================================================================


def readiness(ctx: Ctx) -> dict:
    caps: dict[str, list] = {}
    for cid, result in ctx.results.items():
        for cap in ctx.checks[cid].caps:
            caps.setdefault(cap, []).append(result.status)
    out = {}
    for cap, statuses in caps.items():
        worst = max(statuses, key=lambda s: SEVERITY[s])
        warnings = statuses.count(WARN)
        out[cap] = {FAIL: "not ready", DEGRADED: "degraded", UNAVAILABLE: "unavailable",
                    SKIP: "not verified"}.get(
            worst, f"ready, {warnings} warning{'s' * (warnings != 1)}" if warnings else "ready")
    return out


def next_steps(ctx) -> list[tuple[str, str]]:
    """`(check title, fix)` for each thing to do next, in order: what fails, then what runs degraded, then an agent
    CLI that is not installed. Optional parts (ik, mocap, the library) are not listed: the report shows them, and a
    reader should not take them for steps. One line per fix."""
    steps: list[tuple[str, str]] = []
    for status in (FAIL, DEGRADED, UNAVAILABLE):
        for cid, r in ctx.results.items():
            if r.status != status or not r.fix:
                continue
            fix = r.fix[0]
            if status == UNAVAILABLE and ctx.checks[cid].section != S_HARNESS:
                continue
            if fix not in (f for _, f in steps):
                steps.append((ctx.checks[cid].title, fix))
    # Several missing packages are one step, the checkout's own install, not one command each.
    packages = [i for i, (title, fix) in enumerate(steps)
                if any(cid.startswith("pkg.") and ctx.checks[cid].title == title for cid in ctx.results)
                and fix.startswith(install_cmd(ctx).split("'")[0])]
    every = install_all_cmd(ctx) if len(packages) > 1 else None
    if every:
        first = packages[0]
        steps = [step for i, step in enumerate(steps) if i not in packages]
        steps.insert(first, ("Python packages", every))
    return steps


def report(ctx: Ctx, started: float) -> int:
    counts = {s: 0 for s in SYMBOL}
    for r in ctx.results.values():
        counts[r.status] += 1
    print(f"\n{bold('Summary')}  " + " · ".join(
        f"{tint(str(counts[s]), s)} {s}" for s in (PASS, WARN, DEGRADED, UNAVAILABLE, FAIL, SKIP)
        if counts[s]) + f"   ({time.monotonic() - started:.0f} s)")
    blocks = ((DEGRADED, "DEGRADED: RUNS COMPLETE BUT LOSE CHECKING OR QUALITY"),
              (UNAVAILABLE, "NOT AVAILABLE ON THIS MACHINE"),
              (WARN, "WARNINGS: OPTIONAL FEATURES AND HAZARDS"))
    for status, heading in blocks:
        flagged = [(ctx.checks[i], r) for i, r in ctx.results.items() if r.status == status]
        if not flagged:
            continue
        print(f"\n{bold(heading)}")
        for c, r in flagged:
            print(f"  {tint(SYMBOL[r.status], r.status)} {c.title}: {r.summary}")
            if r.lost:
                print(_wrap(f"lost: {r.lost}", "      ", "            "))
            for f in r.fix[:1]:
                print(_wrap(f"→ {f}", "      ", "        "))
    failed = [(ctx.checks[i], r) for i, r in ctx.results.items() if r.status == FAIL]
    if failed:
        print(f"\n{bold('FAILED')}")
        for c, r in failed:
            print(f"  {tint(SYMBOL[FAIL], FAIL)} {c.title}: {r.summary}")
    table = readiness(ctx)
    print(f"\n{bold('Readiness')}")
    order = [CAP_DYCRAFT, CAP_BARE, CAP_STAGING, *OPTIONAL_CAPS]
    for cap in order + sorted(set(table) - set(order)):
        if cap in table:
            state = table[cap]
            status = (WARN if state.startswith("ready,") else
                      {"ready": PASS, "degraded": DEGRADED, "not ready": FAIL,
                       "unavailable": UNAVAILABLE}.get(state, SKIP))
            print(f"  {tint(SYMBOL[status], status)} {cap:<30} {state}")
    steps = next_steps(ctx)
    if steps:
        print(f"\n{bold('Next:')}")
        for n, (title, fix) in enumerate(steps, 1):
            print(_wrap(f"{n}. {fix}   ({title})", "  ", "     "))
    tiers = [t for t, on in (("deep", ctx.args.deep), ("e2e", ctx.args.e2e)) if on]
    if not tiers:
        print("\n  Static checks only. `--deep` proves each subsystem; `--e2e` runs the pipeline.")
    code = 1 if counts[FAIL] else (2 if ctx.args.strict and counts[DEGRADED] else 0)
    if ctx.args.json:
        payload = {"repo": str(REPO), "python": sys.executable, "exit_code": code,
                   "readiness": table,
                   "results": [{"id": i, "title": ctx.checks[i].title,
                                "section": ctx.checks[i].section, "status": r.status,
                                "summary": r.summary, "detail": r.detail, "fix": r.fix,
                                "lost": r.lost, "data": r.data}
                               for i, r in ctx.results.items()]}
        Path(ctx.args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n  JSON report: {ctx.args.json}")
    return code


# ===========================================================================
# main
# ===========================================================================


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep", action="store_true",
                    help="exercise every subsystem through its real code path (minutes)")
    ap.add_argument("--e2e", action="store_true",
                    help="also run the animation pipeline end to end with no model (implies --deep)")
    ap.add_argument("--backend", action="append", default=[],
                    help="live-probe a dycraft backend (claude, claudeN, codex, codexN); repeatable "
                         "or comma-separated. Spends model requests on that account")
    ap.add_argument("--harness-spec", action="append", default=[],
                    help="live-probe a coding CLI described by a JSON spec (see _SPEC_FIELDS)")
    ap.add_argument("--model", help="model for the live probes (default: the CLI's configured one)")
    ap.add_argument("--effort", help="reasoning effort for the live probes")
    ap.add_argument("--capabilities", action="store_true",
                    help="run the full harness capability checklist on each probed harness")
    ap.add_argument("--python", help="check this interpreter instead (re-execs under it)")
    ap.add_argument("--non-interactive", action="store_true",
                    help="never prompt (the default when stdin is not a terminal)")
    ap.add_argument("--strict", action="store_true", help="exit 2 when anything is degraded")
    ap.add_argument("--json", metavar="FILE", help="also write the report as JSON")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    ap.add_argument("--write-env", action="store_true",
                    help="write environment-main.yml and requirements-main.txt from REQUIREMENTS")
    args = ap.parse_args(argv)
    args.deep = args.deep or args.e2e
    args.backend = [b for chunk in args.backend for b in chunk.split(",") if b]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    _home.tools_on_path()                 # what `install tools` set up, found where a launch finds it
    if args.python and Path(args.python).resolve() != Path(sys.executable).resolve():
        rest = [a for a in (argv if argv is not None else sys.argv[1:])]
        i = rest.index("--python") if "--python" in rest else None
        if i is not None:
            del rest[i:i + 2]
        else:
            rest = [a for a in rest if not a.startswith("--python=")]
        os.execv(args.python, [args.python, str(Path(__file__).resolve()), *rest])
    if args.write_env:
        for name, render in ENV_FILES.items():
            (REPO / name).write_text(render())
            print(f"wrote {REPO / name}")
        return 0

    ctx = Ctx(args)
    started = time.monotonic()
    tiers = {"static"} | ({"deep"} if args.deep else set()) | ({"e2e"} if args.e2e else set())
    print(bold("DyCraft environment check") + f" — {REPO}")
    print(f"interpreter {sys.executable} (Python {platform.python_version()}); "
          f"tiers: {', '.join(t for t in ('static', 'deep', 'e2e') if t in tiers)}"
          + (f"; harnesses: {', '.join(args.backend + args.harness_spec)}"
             if args.backend or args.harness_spec else ""))
    try:
        run_checks(ctx, [c for c in CHECKS if c.tier in tiers])
        if args.backend or args.harness_spec:
            harness_checks = []
            for name in args.backend:
                try:
                    from dycraft.cli.backends import BACKENDS
                except Exception as exc:   # noqa: BLE001
                    print(f"\n  ✖ cannot load cli/backends: {exc}")
                    break
                if name not in BACKENDS:
                    print(f"\n  ✖ unknown backend {name!r}; known: "
                          + ", ".join(sorted(BACKENDS)))
                    continue
                harness_checks += register_backend_checks(ctx, builtin_spec(name))
            for path in args.harness_spec:
                harness_checks += register_backend_checks(ctx, load_spec(path))
            run_checks(ctx, harness_checks)
    except KeyboardInterrupt:
        print("\n\ninterrupted — reporting what ran")
    code = report(ctx, started)
    keep = args.keep or any(r.status == FAIL for i, r in ctx.results.items()
                            if ctx.checks[i].tier != "static")
    if keep:
        print(f"\n  scratch kept for inspection: {ctx.tmp}")
    else:
        for path in [ctx.tmp, *ctx.cleanup]:
            shutil.rmtree(path, ignore_errors=True)
    return code


if __name__ == "__main__":
    # Before anything loads GL: the libraries `install tools` added are found only by a process started with them.
    try:
        from dycraft._home import reexec_for_loader
        reexec_for_loader()
    except ImportError:
        pass
    sys.exit(main())
