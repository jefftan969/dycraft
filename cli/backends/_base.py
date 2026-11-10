"""The interface every agent CLI backend implements, the launch plan it runs, and
the shell environment both CLIs' agents get.
"""
from __future__ import annotations

import os
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


#: Set to "1" (by `--allow-unsandboxed`) to start an agent whose CLI's own sandbox (Codex's, or Claude Code's Bash
#: sandbox) cannot start on this machine, without that sandbox. An environment variable, so every child the harness
#: daemon dispatches inherits the user's decision. `DYCRAFT_ALLOW_UNSANDBOXED_CODEX`, its name when it covered Codex
#: only, means the same.
ALLOW_UNSANDBOXED_ENV = "DYCRAFT_ALLOW_UNSANDBOXED"
ALLOW_UNSANDBOXED_ENV_ALIASES = ("DYCRAFT_ALLOW_UNSANDBOXED_CODEX",)


def allow_unsandboxed(env=None) -> bool:
    """Whether the user passed `--allow-unsandboxed` (or set its environment variable)."""
    env = os.environ if env is None else env
    return any(env.get(name) == "1" for name in (ALLOW_UNSANDBOXED_ENV, *ALLOW_UNSANDBOXED_ENV_ALIASES))


def _fmt_cmd_for_log(cmd: list[str], *, max_arg_len: int = 200) -> str:
    """shlex.join with overlong args truncated and embedded newlines escaped.

    Logs the exact argv handed to the agent CLI without dumping a multi-kB
    system prompt or footer into stderr; tests parse this line to check that
    `--model` and the effort flag propagate. Newlines are escaped so the line
    stays one line for line-based readers of the captured stderr.
    """
    parts: list[str] = []
    for a in cmd:
        s = str(a)
        if len(s) > max_arg_len:
            s = s[:max_arg_len] + f"…[+{len(s) - max_arg_len}B]"
        s = s.replace("\n", "\\n").replace("\r", "\\r")
        parts.append(s)
    return shlex.join(parts)


@dataclass
class LaunchPlan:
    """Backend-agnostic launch context, populated by cli.py."""

    layer: str                          # a key of `agent.LAYERS`
    user_prompt: str
    run_dir: Path
    runs_dir: Path
    run_id: str
    model: str | None
    env: dict[str, str]
    system_prompt_text: str             # the layer's prompt file (`compose_system_prompt`)
    footer_text: str                    # composed user-task footer
    # Reasoning effort forwarded to the agent CLI. Claude takes it as
    # `--effort <value>`; codex takes it as `-c model_reasoning_effort=<value>`.
    # `None` means "let the agent CLI use its own default" (no flag emitted).
    reasoning_effort: str | None = None
    #: Hand an existing conversation back instead of starting one: the claude
    #: or codex session id the run ran as (`--resume`, `cli/resume.py`).
    #: `footer_text` is then the message appended to that conversation, not a
    #: fresh task.
    resume_session: str | None = None


class Backend(ABC):
    """Interface for the agent CLIs."""

    name: str
    # Substring that appears in the agent process's argv-cmdline once
    # `execute` has started it. Used by `agent/server.py::_find_agent_pid` to retarget
    # the harness daemon's watchdog from the launcher pid to the real agent
    # pid (the agent CLI's process can internally fork/detach a worker, so
    # the launcher pid disappears post-exec while the agent keeps running at
    # a different host pid). The marker only has to be specific enough to
    # disambiguate from non-agent ancestors — usually the binary basename
    # ("claude", "codex") is enough since discovery additionally requires an
    # exact `--add-dir <run_dir>` argv-pair match.
    agent_cmdline_marker: str = ""
    #: How long a harness call waits for its job before returning it unfinished
    #: (`_harness_core.WAIT_S`, exported as `DYCRAFT_HARNESS_WAIT_S`), for a CLI
    #: whose shell kills a command sooner than the harness default. None keeps
    #: the default.
    harness_wait_s: str | None = None

    def prepare_shared_env(self, env: dict[str, str]) -> None:
        """Backend-specific edits to the env shared by the agent AND daemon.

        Called by cli.py on the `build_env_dict` result before the harness
        daemon is spawned, so anything a backend resolves for itself (a
        credential dir) is also visible to the daemon, and so to every sub-run
        it starts. Default: no-op.
        """

    #: `meta.json` keys `preflight` recorded about this launch.
    launch_meta: dict = {}

    #: Whether `preflight` needs the run's folder (Codex probes its sandbox in it). Otherwise a launch runs it before
    #: making one, with `run_dir=None`, so a refused launch leaves no folder behind.
    preflight_needs_run_dir = False

    def preflight(self, run_dir: Path | None, layer: str | None = None) -> int | None:
        """Checks before anything is started for this launch: an exit status to refuse with, or None to go on.
        `run_dir` is None before the run's folder exists, and a refusal is then only printed. Default: none."""
        return None

    def credential_meta(self) -> dict[str, str]:
        """`meta.json` keys naming the account/home this run authenticated as.

        Recorded because the external CLI's own transcript lives under that
        home, and a reader (the viewer, a later analysis script) runs in a
        different shell than the run did. Also disambiguates the case where
        one run dir was launched twice under different accounts — first-match
        would otherwise stage the wrong session. See
        `agent/transcript_capture.py::find_claude_transcript`.
        """
        return {}

    @classmethod
    def resolve_agent_model(cls, model: str | None) -> str | None:
        """The model the agent will run under: `--model` verbatim, else the
        agent CLI's own configured default where it can be read, else None
        ("not knowable without a live call"). `install check` reports it.
        """
        return model

    @abstractmethod
    def execute(self, plan: LaunchPlan) -> int:
        ...


#: The numbered account slots both CLI families expose: `claude1`..`claude9`
#: select `~/.claude1`..`~/.claude9`, and `codex1`..`codex9` select
#: `~/.codex1`..`~/.codex9`. Single digits because that is the range the shell
#: aliases cover, and *one* range shared by both families so an account opened
#: on one side never needs a code change on the other.
#:
#: Every slot is registered whether or not its directory exists. Gating on
#: `Path.is_dir()` would make the `--backend` choices — and so `--help`, which
#: a script may parse to decide whether a revision supports a backend — depend
#: on which machine the launcher happens to run on. A slot with no account
#: behind it fails at the CLI's own auth check instead, which says what is
#: actually wrong ("run /login") rather than argparse's "invalid choice".
ACCOUNT_SLOTS: tuple[int, ...] = tuple(range(1, 10))


def agent_shell_env(env: dict, run_dir) -> dict:
    """Adjust the agent CLI's environment for the shells it runs tools in.

    Applied to the agent CLI only -- both backends call it on their own copy
    of the launch env -- never to the harness daemon, which keeps the
    environment `cli.build_env_dict` gave it. Three changes:

    * the run root goes first on `PYTHONPATH`, so `import harness` works from
      a helper script in a subdirectory (`python animation_src/x.py` puts
      `animation_src/`, not the run root, on `sys.path`). Agent-side only:
      on the daemon's path a stray `json.py` in the run dir would shadow its
      imports.
    * conda base auto-activation is switched off. Tool shells run as
      `bash -lc`, and a login shell's `.bashrc` can re-activate conda's base
      env over the launcher's, so plain `python` would no longer be the
      harness interpreter. `CONDA_AUTO_ACTIVATE_BASE` is the conda <25 name,
      `CONDA_AUTO_ACTIVATE` the newer one.
    * a directory holding only `python` and `python3`, both `DYCRAFT_PYTHON`,
      goes first on `PATH`: plain `python` then means the harness interpreter
      even when the launcher was started by absolute path with no env active.
      Only those two names, so no other tool the caller put on PATH (codex,
      node, bwrap, ...) is shadowed.
    """
    run_root = str(Path(run_dir).resolve())
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if run_root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([run_root, *parts])
    env.setdefault("CONDA_AUTO_ACTIVATE_BASE", "false")
    env.setdefault("CONDA_AUTO_ACTIVATE", "false")
    shim = _python_shim_dir(env.get("DYCRAFT_PYTHON") or "")
    if shim:
        path = [p for p in env.get("PATH", "").split(os.pathsep) if p and p != shim]
        env["PATH"] = os.pathsep.join([shim, *path])
    return env


def _python_shim_dir(interpreter: str) -> str | None:
    """`<cache>/dycraft/pybin/<hash>/` with `python`/`python3` -> `interpreter`.

    Keyed on the resolved interpreter, so launches with different envs never
    share one. Built with a rename, so concurrent launches cannot observe a
    half-made directory. None, and PATH untouched, if it cannot be built.
    """
    try:
        target = Path(interpreter).expanduser().resolve()
        if not interpreter or not target.is_file():
            return None
        import hashlib
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        root = cache / "dycraft" / "pybin"
        final = root / hashlib.md5(str(target).encode()).hexdigest()[:12]
        if all((final / n).resolve() == target for n in ("python", "python3")):
            return str(final)
        root.mkdir(parents=True, exist_ok=True)
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix=".pybin-", dir=root))
        for name in ("python", "python3"):
            (tmp / name).symlink_to(target)
        try:
            os.rename(tmp, final)
        except OSError:                     # another launch won the rename
            import shutil as _shutil
            _shutil.rmtree(tmp, ignore_errors=True)
        return str(final) if (final / "python").resolve() == target else None
    except Exception:                       # noqa: BLE001 - best-effort
        return None
