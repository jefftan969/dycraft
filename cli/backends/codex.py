"""`CodexBackend`: runs the `codex` CLI on a run directory: its sandbox probe and
refusal, the trust entries it writes into `config.toml`, and the `codexN`
account selectors.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

try:  # tomllib is stdlib from 3.11; the pinned interpreter here is 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - one of the two is present
    import tomli as tomllib  # same API, same TOMLDecodeError-under-ValueError

from . import _process
from ._base import ALLOW_UNSANDBOXED_ENV_ALIASES, Backend, LaunchPlan, _fmt_cmd_for_log, agent_shell_env, \
    allow_unsandboxed


class _SignalsDeferred:
    """Hold SIGTERM and SIGINT for the duration of a short critical section, then deliver any that arrived.

    Python runs signal handlers in the main thread, so swapping the handlers there defers them whichever thread
    the kernel delivers to. Elsewhere (no handlers can be set) it does nothing; the edits it guards are atomic anyway.
    """

    _SIGNALS = ("SIGTERM", "SIGINT")

    def __enter__(self):
        import signal
        import threading
        self.saved, self.pending = {}, []
        if threading.current_thread() is not threading.main_thread():
            return self
        for name in self._SIGNALS:
            num = getattr(signal, name)
            self.saved[num] = signal.signal(num, lambda signum, _frame: self.pending.append(signum))
        return self

    def __exit__(self, *_exc) -> None:
        import signal
        for num, handler in self.saved.items():
            signal.signal(num, handler)
        for num in self.pending:
            os.kill(os.getpid(), num)


def _edit_codex_config(cfg: Path, change) -> None:
    """Apply `change(text) -> text` to Codex's `config.toml` so that no reader or writer ever sees a partial file.

    Concurrent DyCraft launches serialize on a sidecar lock (`config.toml.lock`, never the file itself, which is
    replaced). The new text is written to a temporary file in the same directory, flushed to disk and renamed over
    the original, so a kill, a full disk or a Codex reading without the lock finds either the old file or the new
    one. A symlinked `config.toml` is edited at its target. SIGTERM and SIGINT wait for the rename.
    """
    import fcntl
    import tempfile
    cfg = cfg.resolve() if cfg.is_symlink() else cfg
    cfg.parent.mkdir(parents=True, exist_ok=True)
    with _SignalsDeferred(), open(cfg.parent / (cfg.name + ".lock"), "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                text = cfg.read_text(encoding="utf-8")
            except FileNotFoundError:
                text = ""
            new_text = change(text)
            if new_text == text:
                return
            mode = cfg.stat().st_mode & 0o7777 if cfg.exists() else 0o600
            fd, tmp = tempfile.mkstemp(prefix=f".{cfg.name}.", suffix=".tmp", dir=cfg.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(new_text)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.chmod(tmp, mode)
                os.replace(tmp, cfg)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def mark_codex_untrusted(codex_home, run_dir, label: str = "codex"):
    """Record `run_dir` as untrusted in `$CODEX_HOME/config.toml` for one launch; return the callable that undoes it.

    A headless Codex in a directory with no trust decision waits forever on its "Do you trust the contents of this
    directory?" screen. An `untrusted` entry answers that screen as a `trusted` one would, and keeps Codex from
    loading project-local config, hooks and exec policies from the run directory, which the agent, and for a child's
    directory its parent agent, can write: in a trusted directory a planted `.codex/config.toml` starts its MCP
    servers outside the sandbox. An entry for the directory itself takes precedence over a trusted checkout above
    it. A `-c` override does not reach Codex's trust table, so the entry goes in the file, keyed by the resolved
    path Codex matches against.

    An existing entry for the directory is set to `untrusted` and stays so; an entry this call added is removed by
    the returned callable. Every edit is atomic (`_edit_codex_config`). Best-effort: on failure the launch goes
    ahead, and Codex's trust screen, if it appears, is caught by the PTY watch.
    """
    key = str(Path(run_dir).resolve())
    header = f'[projects."{key}"]'
    block = f'\n{header}\ntrust_level = "untrusted"\n'
    cfg = Path(codex_home).expanduser() / "config.toml"
    added = False

    def record(text: str) -> str:
        nonlocal added
        lines = text.splitlines(keepends=True)
        at = next((i for i, line in enumerate(lines) if line.strip() == header), None)
        if at is None:
            added = True
            return text + block
        for i in range(at + 1, len(lines)):
            if lines[i].lstrip().startswith("["):
                break
            if lines[i].split("=", 1)[0].strip() == "trust_level":
                lines[i] = 'trust_level = "untrusted"\n'
                return "".join(lines)
        lines.insert(at + 1, 'trust_level = "untrusted"\n')
        return "".join(lines)

    try:
        _edit_codex_config(cfg, record)
    except Exception as exc:  # noqa: BLE001 - see docstring
        print(f"[{label}] could not record {key} as untrusted in {cfg} (continuing): {exc}",
              file=sys.stderr)
        return lambda: None

    def undo() -> None:
        if not added:
            return
        try:
            _edit_codex_config(cfg, lambda text: text.replace(block, "", 1))
        except Exception as exc:  # noqa: BLE001 - a leftover untrusted entry is inert
            print(f"[{label}] could not remove the untrusted entry for {key} from {cfg}: {exc}",
                  file=sys.stderr)
    return undo


#: `ALLOW_UNSANDBOXED_ENV`'s other name (`_base.allow_unsandboxed`).
ALLOW_UNSANDBOXED_CODEX_ENV = ALLOW_UNSANDBOXED_ENV_ALIASES[0]


#: How long the sandbox probe may take; it normally takes a fraction of a second.
CODEX_SANDBOX_PROBE_TIMEOUT_S = 60


_PROBE_SCRIPT = ('echo ok > "$1" || exit 3; '
                 'if echo x > "$2" 2>/dev/null; then echo LEAKED; else echo CONFINED; fi')


def _probe_outside_parent(run_dir: Path, env: dict) -> Path | None:
    """A directory for the probe's outside write: not under the run directory, `/tmp` or `$TMPDIR`, which are
    the sandbox's writable roots."""
    roots = [run_dir, Path("/tmp").resolve()]
    if env.get("TMPDIR"):
        roots.append(Path(env["TMPDIR"]).resolve())
    for candidate in (Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "dycraft",
                      Path.home() / ".cache" / "dycraft", run_dir.parent):
        resolved = candidate.resolve()
        if not any(resolved == root or root in resolved.parents for root in roots):
            return candidate
    return None


def codex_sandbox_probe(binary: str, codex_home: str, run_dir: Path, env: dict | None = None,
                        timeout: float = CODEX_SANDBOX_PROBE_TIMEOUT_S, wrapper: list[str] | tuple = ()) -> str | None:
    """None when Codex's `workspace-write` sandbox confines writes to `run_dir` here; otherwise what went wrong.

    Runs one shell command under `codex sandbox` (the same bubblewrap sandbox the agent's commands get, and no
    model call): it writes a file in `run_dir`, then tries one in a directory outside every writable root.
    """
    import subprocess
    import tempfile
    run_dir = Path(run_dir).resolve()
    probe_env = {**(env if env is not None else os.environ), "CODEX_HOME": str(codex_home)}
    parent = _probe_outside_parent(run_dir, probe_env)
    if parent is None:
        return ("the probe found no directory outside the sandbox's writable roots (the run directory, /tmp, "
                "$TMPDIR) to test a write in")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        outside = Path(tempfile.mkdtemp(prefix=".sandbox_probe_outside-", dir=parent))
    except OSError as exc:
        return f"the probe could not make a scratch directory in {parent}: {exc}"
    inside = run_dir / f".sandbox_probe_{outside.name}"
    cmd = [*wrapper, binary, "sandbox", "-P", ":workspace", "-C", str(run_dir),
           "-c", "sandbox_workspace_write.network_access=true",
           "--", "sh", "-c", _PROBE_SCRIPT, "sh", str(inside), str(outside / "leak")]
    from ... import _oom
    try:
        ran = subprocess.run(_oom.tiered(cmd, _oom.WORKER), env=probe_env, cwd=str(run_dir),
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"the probe timed out after {timeout:g} s"
    except OSError as exc:
        return f"the probe could not run: {exc}"
    finally:
        leaked = (outside / "leak").exists()
        inside_ok = inside.is_file()
        inside.unlink(missing_ok=True)
        shutil.rmtree(outside, ignore_errors=True)
    if leaked or "LEAKED" in ran.stdout:
        return "a write outside the run directory succeeded"
    if ran.returncode == 0 and inside_ok and "CONFINED" in ran.stdout:
        return None
    lines = [line.strip() for line in ran.stderr.splitlines() if line.strip()]
    last = next((line for line in reversed(lines) if not line.startswith("note: run with")), "")
    return f"the sandbox failed to start: {last or f'exit status {ran.returncode}'}"


#: The run's own temporary directory, the first one tried for a Codex whose sandbox cannot use the TMPDIR (or /tmp) it
#: inherited: a cluster job's private one, such as Slurm's /scr/<job>/, where bwrap cannot make the mount point it
#: needs. It is inside the run directory, which is already writable in the sandbox, so it opens nothing new.
RUN_TMPDIR = ".tmp"


def _fallback_tmpdirs(run_dir: Path, env: dict) -> list[Path]:
    """Where to try Codex's TMPDIR, in order, when the inherited one fails: the run's own, this node's /tmp, then the
    user's cache. Each is only a scratch directory; the inherited one is skipped if it is among them."""
    cache = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "dycraft" / "tmp"
    candidates = [Path(run_dir).resolve() / RUN_TMPDIR, Path("/tmp") / f"dycraft-{os.getuid()}", cache]
    inherited = env.get("TMPDIR")
    inherited = Path(inherited).resolve() if inherited else None
    return [c for c in candidates if inherited is None or c.resolve() != inherited]


#: `always` runs Codex in its own namespace (`codex_namespace_wrapper`) even where it does not need one, `never` never;
#: unset or `auto`, only where nothing else lets its sandbox start. For diagnosis.
CODEX_NAMESPACE_ENV = "DYCRAFT_CODEX_NAMESPACE"

#: The size of the fresh /tmp `codex_namespace_wrapper` gives Codex. Its commands' large temporary files go to TMPDIR,
#: which is set to the run's own (on disk) whenever the wrapper is used.
CODEX_TMPFS_SIZE = "1g"


def codex_namespace_wrapper() -> list[str]:
    """The argv prefix that runs Codex in a user and mount namespace of its own, with a fresh tmpfs over /tmp; empty
    where `unshare` is not installed.

    For a node whose /tmp is a bind mount of a directory the user cannot reach, such as a Slurm job's per-job /tmp
    under a root-owned /scr/<job>/. Codex hides its app-server socket directory, /tmp/codex-daemon-<uid>, from the
    sandbox with a mask placed at a path built from the source of the mount it sits in, and bwrap cannot make that
    path, so the sandbox never starts, whatever TMPDIR or Codex setting is used. A tmpfs has no source path, so the
    mask lands inside the new /tmp, where it can. Codex's commands then see uid 0, which is the user outside, and
    a /tmp of their own.
    """
    unshare = shutil.which("unshare")
    if not unshare:
        return []
    return [unshare, "-r", "--mount", "sh", "-c",
            f'mount -t tmpfs -o size={CODEX_TMPFS_SIZE},mode=1777 tmpfs /tmp && exec "$@"', "sh"]


def codex_sandbox_probe_or_tmpdir(binary: str, codex_home: str, run_dir: Path, env: dict | None = None,
                                  ) -> tuple[str | None, str | None, str | None, list[str]]:
    """`(problem, tmpdir, inherited_problem, wrapper)`: how Codex's sandbox can confine its commands here.

    First as inherited; when that fails, with each of `_fallback_tmpdirs` as TMPDIR; and when those fail too, in a
    namespace of its own with a fresh /tmp (`codex_namespace_wrapper`) and the run's own TMPDIR. Each is accepted
    only when its probe confines: a write inside the run lands and one outside every writable root is refused.
    `tmpdir` and `wrapper` are what to start Codex with (None and [] as inherited), and `inherited_problem` is why
    the inherited setup did not work. A write that leaked is never retried. When nothing works, `problem` names each
    setup tried and what it reported."""
    env = dict(os.environ if env is None else env)
    namespace = (env.get(CODEX_NAMESPACE_ENV) or "auto").strip().lower()
    if namespace == "always":
        wrapper = codex_namespace_wrapper()
        tmpdir = Path(run_dir).resolve() / RUN_TMPDIR
        tmpdir.mkdir(parents=True, exist_ok=True)
        problem = codex_sandbox_probe(binary, codex_home, run_dir, {**env, "TMPDIR": str(tmpdir)}, wrapper=wrapper) \
            if wrapper else f"{CODEX_NAMESPACE_ENV}=always, but `unshare` is not installed"
        return (problem, None, None, []) if problem else (None, str(tmpdir), f"{CODEX_NAMESPACE_ENV}=always", wrapper)
    first = codex_sandbox_probe(binary, codex_home, run_dir, env)
    if first is None or first.startswith("a write outside"):
        return first, None, None, []
    tried = [(f"TMPDIR={env.get('TMPDIR') or '/tmp'}", first)]
    for tmpdir in _fallback_tmpdirs(run_dir, env):
        try:
            tmpdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            tried.append((f"TMPDIR={tmpdir}", f"could not make it: {exc}"))
            continue
        problem = codex_sandbox_probe(binary, codex_home, run_dir, {**env, "TMPDIR": str(tmpdir)})
        if problem is None:
            return None, str(tmpdir), first, []
        if problem.startswith("a write outside"):
            return problem, None, None, []
        tried.append((f"TMPDIR={tmpdir}", problem))
    wrapper = codex_namespace_wrapper() if namespace != "never" else []
    if wrapper:
        tmpdir = Path(run_dir).resolve() / RUN_TMPDIR
        tmpdir.mkdir(parents=True, exist_ok=True)
        problem = codex_sandbox_probe(binary, codex_home, run_dir, {**env, "TMPDIR": str(tmpdir)}, wrapper=wrapper)
        if problem is None:
            return None, str(tmpdir), first, wrapper
        if problem.startswith("a write outside"):
            return problem, None, None, []
        tried.append((f"a /tmp of its own (unshare -r --mount, TMPDIR={tmpdir})", problem))
    return ("it failed in every setup tried: " + "; ".join(f"{where}: {why}" for where, why in tried),
            None, None, [])


def _first_failure(problem: str) -> str:
    """The first setup's failure in `codex_sandbox_probe_or_tmpdir`'s every-setup report, for a one-line reason (the
    run's README); the report itself stays whole in `meta.json`'s `codex_sandbox.probe`."""
    prefix = "it failed in every setup tried: "
    if not problem.startswith(prefix):
        return problem
    first = problem[len(prefix):].split("; ", 1)[0]
    return first.split(": ", 1)[1] if ": " in first else first


def codex_account_problem(codex_home: str, env: dict | None = None) -> tuple[str, str, list[str]] | None:
    """Why this Codex account cannot start a session, as `(reason, message, fix lines)`, or None.

    Checked before the sandbox probe, which runs Codex too and would otherwise report each of these as a sandbox
    that does not start, with fixes for bubblewrap. Reads `config.toml`, and only tests that `auth.json` exists.
    The login test applies to Codex's own provider keeping its login in a file, the default; an API key in the
    environment, another credential store or another provider is left to Codex.
    """
    env = os.environ if env is None else env
    home = Path(codex_home).expanduser()
    login = f"`CODEX_HOME={home} codex login`"
    if not home.is_dir():
        return ("codex_home_missing", f"Codex's account home {home} does not exist, so this account has no login",
                [login, "or pick another `--backend`"])
    config: dict = {}
    path = home / "config.toml"
    if path.is_file():
        try:
            config = tomllib.loads(path.read_text())
        except (OSError, ValueError) as exc:
            return ("codex_config_invalid", f"Codex cannot read its config {path}: {exc}",
                    [f"fix `{path}`, or move it aside to start Codex with its defaults"])
    own_login = (str(config.get("model_provider") or "openai") == "openai"
                 and str(config.get("cli_auth_credentials_store") or "file") == "file"
                 and not any(env.get(key) for key in ("OPENAI_API_KEY", "CODEX_API_KEY")))
    if own_login and not (home / "auth.json").is_file():
        return ("codex_not_logged_in", f"Codex is not logged in on {home}: it holds no auth.json", [login])
    return None


def codex_allowed_approval_policies(codex_home: str) -> list[str] | None:
    """The approval policies this account's managed requirements allow, or None when nothing restricts them (or
    nothing can be read).

    An organization's workspace can restrict them (`allowed_approval_policies`); Codex then replaces a policy it
    does not allow, `never` included, with an allowed one, without saying so. Read from the system requirements
    file and the cached cloud bundle Codex keeps in `$CODEX_HOME`, for reporting only: the policy a session
    actually ran under is in its rollout (`_CodexRolloutWatch.approval_policy`).
    """
    texts: list[str] = []
    try:
        texts.append(Path("/etc/codex/requirements.toml").read_text())
    except OSError:
        pass
    try:
        bundle = json.loads((Path(codex_home) / "cloud-config-bundle-cache.json").read_text())
        managed = bundle["signed_payload"]["bundle"]["requirements_toml"].get("enterprise_managed") or []
        texts += [entry.get("contents") or "" for entry in managed]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    allowed = None
    for text in texts:
        try:
            found = tomllib.loads(text).get("allowed_approval_policies")
        except ValueError:
            continue
        if isinstance(found, list):
            found = [str(p) for p in found]
            allowed = found if allowed is None else [p for p in allowed if p in found]
    return allowed


#: What to do about a Codex whose sandbox does not work, past what this system's AppArmor policy needs
#: (`codex_sandbox_fixes`): `install check` lists them, and a refused launch records them for its run's README.
CODEX_SANDBOX_FIXES = ("a full or unwritable `/tmp` or `$TMPDIR`: free space, or set `TMPDIR` to a writable directory",
                       "without sudo: `--allow-unsandboxed` runs Codex without its sandbox, so its commands can read "
                       "and write anywhere you can",
                       "see `docs/install.md#sandbox-fails-to-start`")


def _userns_steps() -> list[str]:
    """With root, what lets bubblewrap make user namespaces on this system (`_userns.fixes`)."""
    from ... import _userns
    return _userns.fixes("`sudo apt-get install bubblewrap`") or [
        "Ubuntu 24.04: the system's `bubblewrap` package, `sudo apt-get install bubblewrap`"]


def codex_sandbox_fixes() -> list[str]:
    """`CODEX_SANDBOX_FIXES`, after what this system's AppArmor policy needs."""
    return [*_userns_steps(), *CODEX_SANDBOX_FIXES]


def codex_sandbox_refusal(problem: str, binary: str, codex_home: str, version: str) -> str:
    """The message a refused Codex launch prints."""
    steps = "".join(f"  - With root: {step[0].upper()}{step[1:]}.\n" for step in _userns_steps())
    return (
        "FATAL: Codex's sandbox does not work here, so DyCraft will not start a Codex agent without write "
        "confinement.\n"
        f"  probe:  codex sandbox -P :workspace ({version}, CODEX_HOME={codex_home})\n"
        f"  result: {problem}\n"
        "Fixes (docs/install.md#sandbox-fails-to-start):\n"
        + steps +
        "  - A full or unwritable /tmp or $TMPDIR: free space, or point TMPDIR at a writable directory. (DyCraft\n"
        "    already retries a TMPDIR the sandbox cannot use in the run directory, this node's /tmp and\n"
        "    ~/.cache/dycraft/tmp, then with a /tmp of Codex's own; the result above names each setup tried.)\n"
        "  - Or launch with --backend claude where Claude Code's sandbox works (python -m dycraft.install check).\n"
        "Or, without sudo, pass --allow-unsandboxed to run Codex without its sandbox: its commands can then read\n"
        "and write anywhere you can, once Codex's automatic reviewer approves them.")


#: Flags every codex this framework launches carries, TUI or `exec`. codex's
#: startup screens wait for a keypress, and most of what this
#: framework starts has nobody at the keyboard. Two of them a flag can switch
#: off: the update notice, and the "switch to a cheaper model for lower credit
#: usage?" prompt codex shows near an account's usage limit, mid-run, which held
#: an unattended paper run until someone answered it. The trust screen needs
#: `mark_codex_untrusted` (a `-c` override does not reach codex's trust table).
#: Anything that spawns `codex` directly rather than through `CodexBackend`
#: should splice this in too.
CODEX_HEADLESS_ARGS: tuple[str, ...] = ("-c", "check_for_update_on_startup=false",
                                        "-c", "notice.hide_rate_limit_model_nudge=true")


class CodexBackend(Backend):
    """Spawns the `codex` CLI with the system prompt and footer concatenated
    into one positional argument (codex has no `--append-system-prompt`). Same
    daemon socket as ClaudeBackend.

    Confinement fails closed, as Claude's does (`failIfUnavailable`). Before
    each launch `preflight` checks that Codex's sandbox starts and confines
    writes here (`codex_sandbox_probe_or_tmpdir`) and refuses otherwise, unless
    the user opted in with `--allow-unsandboxed` (`allow_unsandboxed`, which
    the daemon's children inherit). The agent then runs with
    `--sandbox workspace-write --ask-for-approval never`: writes only in the
    run directory (its working root) and Codex's temporary directories, and no
    path out of the sandbox, so a command the sandbox refuses fails, even if
    the sandbox breaks mid-run. Network access is on, so tool shells reach the
    harness daemon.

    PTY: codex refuses to start without a terminal on its standard streams.
    When any of them is not a tty, the CLI runs on a PTY driven by
    `_spawn_with_pty_term_shim`, which also answers the terminal probes codex
    blocks on. `DYCRAFT_CODEX_NO_PTY_WRAP=1` skips the wrap, for a caller that
    already provides a terminal.
    """

    name = "codex"
    agent_cmdline_marker = "codex"
    HOME = str(Path("~/.codex").expanduser())
    #: What `preflight` found: None when the sandbox confines, else the probe's result line.
    _sandbox_problem: str | None = None
    #: The TMPDIR Codex runs with when its sandbox cannot use the inherited one (`codex_sandbox_probe_or_tmpdir`).
    _tmpdir: str | None = None
    #: The argv prefix Codex runs under when its sandbox needs a /tmp of its own (`codex_namespace_wrapper`).
    _wrapper: tuple = ()
    _preflight_done = False
    preflight_needs_run_dir = True
    #: Whether the launch goes through Codex's automatic reviewer (`--approve-for-me`) rather than `never`.
    _uses_reviewer = False

    # NOTE: deliberately *not* named `_resolve_executable`. `cli.py`'s
    # meta.json writer duck-types that name to record the *claude* account
    # (`getattr(backend, "_resolve_executable", None)` →
    # `extra_meta["claude_config_dir"]`), so a same-named codex method would
    # file this backend's home under a claude key and
    # `agent/transcript_capture.py` would read it back as a claude account.
    @classmethod
    def _resolve_codex_cli(cls) -> tuple[str | None, str]:
        """Return `(binary | None, CODEX_HOME)` for this backend.

        `None` binary means "not found on PATH"; `execute` turns that into a
        fatal launch error. Shared with `prepare_shared_env` so the daemon and
        the agent use the same account.
        """
        binary = os.environ.get("DYCRAFT_CODEX_EXECUTABLE") or shutil.which("codex")
        return binary, str(Path(cls.HOME).expanduser())

    @classmethod
    def resolve_agent_model(cls, model: str | None) -> str | None:
        """`--model`, else the model `$CODEX_HOME/config.toml` selects.

        With no `--model` the launcher emits no `-m`, so the agent runs
        config.toml's `model`. Returns None when config.toml names no model
        (the codex CLI's own built-in default then applies).
        """
        if model:
            return model
        _, home = cls._resolve_codex_cli()
        try:
            with open(Path(home) / "config.toml", "rb") as fh:
                config = tomllib.load(fh)
        except (OSError, ValueError):
            # ValueError covers tomllib.TOMLDecodeError. Never block a launch
            # over a malformed user config.
            return None
        found = config.get("model") if isinstance(config, dict) else None
        return str(found) if found else None

    def prepare_shared_env(self, env: dict[str, str]) -> None:
        """Pin the account for the daemon as well as the agent.

        `execute`'s private env copy is set after the daemon has been spawned,
        so CODEX_HOME has to be on the env the daemon inherits for every
        sub-run it starts to use the agent's account.
        """
        _, home = self._resolve_codex_cli()
        env["CODEX_HOME"] = home

    def credential_meta(self) -> dict[str, str]:
        """Always recorded: this backend *pins* CODEX_HOME rather than
        inheriting it, so the account is always a DyCraft choice and always
        knowable. Codex files its session rollouts under
        `$CODEX_HOME/sessions/`, so transcript capture needs this."""
        _, home = self._resolve_codex_cli()
        return {"codex_home": home}

    def preflight(self, run_dir: Path, layer: str | None = None) -> int | None:
        """Refuse (exit status 2) where this Codex account cannot start a session (`codex_account_problem`), or
        where Codex's sandbox does not confine, unless the user opted in.

        Sets `launch_meta["codex_sandbox"]` either way, and records a refusal in `meta.json` (`launch_failure`) so
        a parent sees why its child never started. `execute` calls this itself when no caller did.
        """
        bin_path, codex_home = self._resolve_codex_cli()
        if not bin_path:
            return None                     # `execute` reports the missing CLI
        account = None if os.environ.get("DYCRAFT_SKIP_PREFLIGHT") == "1" else codex_account_problem(codex_home)
        if account is not None:
            from ...agent.agent_session import record_launch_failure
            reason, message, fix = account
            print(f"FATAL: {message}.\n" + "".join(f"  fix: {line}\n" for line in fix)
                  + "  Nothing was started. DYCRAFT_SKIP_PREFLIGHT=1 launches without this check.",
                  file=sys.stderr, flush=True)
            record_launch_failure(Path(run_dir), reason=reason, detail=message, message=f"FATAL: {message}.",
                                  fix=fix)
            return 2
        self._sandbox_problem, self._tmpdir, inherited, wrapper = codex_sandbox_probe_or_tmpdir(
            bin_path, codex_home, run_dir)
        self._wrapper = tuple(wrapper)
        if self._wrapper and inherited and inherited.startswith(CODEX_NAMESPACE_ENV):
            print(f"[cli] {inherited}: Codex runs in a namespace of its own with a fresh /tmp, and "
                  f"TMPDIR={self._tmpdir}", file=sys.stderr, flush=True)
        elif self._wrapper:
            print(f"[cli] Codex's sandbox cannot start with this machine's /tmp ({inherited}); Codex runs in a "
                  f"namespace of its own with a fresh /tmp, and TMPDIR={self._tmpdir}", file=sys.stderr, flush=True)
        elif self._tmpdir:
            print(f"[cli] Codex's sandbox cannot use TMPDIR={os.environ.get('TMPDIR') or '/tmp'} ({inherited}); "
                  f"Codex runs with TMPDIR={self._tmpdir} instead", file=sys.stderr, flush=True)
        self._preflight_done = True
        self._uses_reviewer = self._sandbox_problem is not None
        if self._sandbox_problem is None:
            record = {"probe": "ok", "mode": "workspace-write", "approval": "never"}
            if self._tmpdir:
                record["tmpdir"] = self._tmpdir
            if self._wrapper:
                record["namespace"] = f"its own /tmp (tmpfs, {CODEX_TMPFS_SIZE})"
            allowed = codex_allowed_approval_policies(codex_home)
            if allowed is not None and "never" not in allowed:
                record["approval"] = f"{allowed[0] if allowed else 'on-request'} (organization policy)"
                print(f"WARNING: this Codex account's organization does not allow `--ask-for-approval never` "
                      f"(allowed: {', '.join(allowed) or 'none listed'}). Codex runs with "
                      f"{allowed[0] if allowed else 'on-request'} approvals instead: its automatic reviewer can "
                      "approve a command outside the sandbox.", file=sys.stderr, flush=True)
            if allow_unsandboxed():
                print("[cli] --allow-unsandboxed: Codex's sandbox works here, so it stays on", file=sys.stderr,
                      flush=True)
            self.launch_meta = {"codex_sandbox": record}
            return None
        if not allow_unsandboxed():
            import subprocess
            from ... import _oom
            from ...agent.agent_session import record_launch_failure
            try:
                version = subprocess.run(_oom.tiered([bin_path, "--version"], _oom.WORKER),
                                         capture_output=True, text=True,
                                         timeout=30).stdout.strip() or "version unknown"
            except (OSError, subprocess.SubprocessError):
                version = "version unknown"
            refusal = codex_sandbox_refusal(self._sandbox_problem, bin_path, codex_home, version)
            print(refusal, file=sys.stderr, flush=True)
            # The refusal's opening sentence, which the terminal shows wrapped, as one line.
            opening = []
            for line in refusal.splitlines():
                if line.startswith(" "):
                    break
                opening.append(line.strip())
            record_launch_failure(Path(run_dir), reason="codex_sandbox_refused", detail=self._sandbox_problem,
                                  message=" ".join(opening) or None, fix=codex_sandbox_fixes())
            return 2
        print(f"WARNING: running Codex without a sandbox (--allow-unsandboxed): its sandbox does not work here "
              f"({self._sandbox_problem}). The agent's commands can read and write anywhere you can, once Codex's "
              "automatic reviewer approves them.", file=sys.stderr, flush=True)
        self.launch_meta = {"codex_sandbox": {"probe": self._sandbox_problem,
                                              "mode": "unsandboxed (--allow-unsandboxed)",
                                              "approval": "on-request (automatic reviewer)"},
                            "unsandboxed": {"agent": "Codex", "reason": _first_failure(self._sandbox_problem)}}
        return None

    def execute(self, plan: LaunchPlan) -> int:
        bin_path, codex_home = self._resolve_codex_cli()
        if not bin_path:
            print("FATAL: codex CLI not found. Set DYCRAFT_CODEX_EXECUTABLE "
                  "or ensure `codex` is on PATH.", file=sys.stderr)
            return 2
        if not self._preflight_done:
            refusal = self.preflight(plan.run_dir, plan.layer)
            if refusal is not None:
                return refusal
        env = agent_shell_env(dict(plan.env), plan.run_dir)
        # The user's own account home, so its login, model choice and
        # acknowledged startup notices carry over; a fresh home would show
        # screens nobody can answer. Confinement does not depend on it, but a
        # sandbox-widening key in that `config.toml` (extra `writable_roots`)
        # applies to DyCraft's agents too (docs/architecture.md).
        env["CODEX_HOME"] = codex_home
        if self._tmpdir:
            env["TMPDIR"] = self._tmpdir          # the inherited one does not work in the sandbox (`preflight`)
        # The working root is the run directory, which is what `workspace-write`
        # lets the agent write. `--add-dir <run_dir>` is the argv pair the
        # daemon's watchdog finds the agent's pid by.
        cmd = [bin_path, "-C", str(plan.run_dir),
               "--add-dir", str(plan.run_dir)]
        # `--ask-for-approval never` leaves no path out of the sandbox: a
        # refused command fails and the model is told. `--approve-for-me`
        # (Codex's automatic reviewer, which can approve a command outside the
        # sandbox) only where the user opted in on a host whose sandbox does
        # not work.
        # Where an organization does not allow `never`, Codex runs `on-request`
        # instead (`codex_allowed_approval_policies`); the automatic reviewer,
        # pinned here, then answers what a person would, so a headless run never
        # waits on an approval prompt.
        if self._uses_reviewer:
            cmd += ["--approve-for-me"]
        else:
            cmd += ["--sandbox", "workspace-write", "--ask-for-approval", "never",
                    "-c", 'approvals_reviewer="auto_review"']
        cmd += ["-c", "sandbox_workspace_write.network_access=true"]
        # The "Update available! ... Press enter to continue" screen is another
        # startup prompt nobody can answer in a headless run; an account that
        # has not dismissed the newest release shows it on every launch.
        cmd += list(CODEX_HEADLESS_ARGS)
        if plan.model:
            cmd += ["-m", plan.model]
        if plan.reasoning_effort:
            # Codex has no top-level `--effort` flag; reasoning effort is
            # set via the generic `-c key=value` override against
            # `model_reasoning_effort`. Forwarded verbatim — we never
            # silently rewrite, so an unsupported value fails at codex
            # startup with a clear error rather than masking the request.
            #
            # Do NOT assume this backend caps out below claude's. Each
            # model publishes its own ladder in
            # `$CODEX_HOME/models_cache.json::supported_reasoning_levels`,
            # and current entries go up to `xhigh`, `max` and `ultra` — the
            # same range as claude's `--effort`, sometimes wider; nothing
            # here enforces a ceiling either way.
            cmd += ["-c", f"model_reasoning_effort={plan.reasoning_effort}"]
        if plan.resume_session:
            # `codex resume [OPTIONS] SESSION_ID PROMPT`, with a cold start's options: the session already holds the
            # system prompt and the task, and goes on with this message.
            cmd = [cmd[0], "resume", *cmd[1:], plan.resume_session, plan.footer_text]
        else:
            cmd.append(plan.system_prompt_text + "\n\n" + plan.footer_text)
        # Last, so the forms above still find the binary at cmd[0]: the namespace the preflight's probe confined in.
        cmd = [*self._wrapper, *cmd]

        # Auto-PTY-wrap when any standard stream is non-tty
        # (subprocess.run(capture_output=True), CI runners, nested
        # agent-driven runs). Skipping the wrap when stdin/stdout/stderr
        # are all ttys avoids the extra fork on the normal terminal path.
        wrap_for_pty = (
            not os.environ.get("DYCRAFT_CODEX_NO_PTY_WRAP")
            and not _process._stdio_is_tty()
        )

        print(f"[codex_{plan.layer}] run_dir={plan.run_dir}", file=sys.stderr)
        record = self.launch_meta.get("codex_sandbox", {})
        print(f"[codex_{plan.layer}] launching codex "
              f"(workspace=$RUN_DIR, writable=$RUN_DIR, sandbox={record.get('mode')}, "
              f"approval={record.get('approval')}, "
              f"model={plan.model}, effort={plan.reasoning_effort}"
              f"{', pty-wrapped' if wrap_for_pty else ''})", file=sys.stderr)
        # Same audit-trail rationale as ClaudeBackend — tests parse the
        # argv line out of captured stderr to verify model + reasoning
        # reach the agent CLI faithfully.
        print(f"[codex_{plan.layer}] "
              f"argv={_fmt_cmd_for_log(cmd)}",
              file=sys.stderr, flush=True)

        # Answers Codex's trust screen for this directory without trusting it
        # (`mark_codex_untrusted`), for this launch only.
        undo_untrusted = mark_codex_untrusted(codex_home, plan.run_dir, label=f"codex_{plan.layer}")
        from ... import _oom
        oom_before = _oom.oom_kills()
        try:
            if wrap_for_pty:
                status = _process._spawn_with_pty_term_shim(cmd, env)
            else:
                status = _process.run_agent_cli(cmd, env)
        finally:
            undo_untrusted()
        return _process.report_agent_kill(status, plan.run_dir, oom_before)


def _codex_account_backend(slot: int) -> type[CodexBackend]:
    """Build `CodexNBackend`: `--backend codexN` == `--backend codex` with
    `CODEX_HOME=~/.codexN`.

    Inherits `name = "codex"`, so name-based conditioning still sees "codex" —
    the same contract the `claudeN` selectors keep. Unlike those, no executable
    is pinned: `CodexBackend` resolves the binary from
    `DYCRAFT_CODEX_EXECUTABLE` or PATH independently of the account, so the
    only thing an account selector has to change is `HOME`, which reaches both
    `CODEX_HOME` and the `config.toml` this backend writes trust entries into.

    Generated, reachable only through `BACKENDS`, and `__module__`-corrected
    for the reasons `_claude_account_backend` gives. A slot sets only its
    docstring and `HOME`.
    """
    return type(f"Codex{slot}Backend", (CodexBackend,), {
        "__module__": __name__,
        "__doc__": (f"`--backend codex{slot}` == `--backend codex` with "
                    f"`CODEX_HOME=~/.codex{slot}`. Inherits "
                    f'`name = "codex"`; only the account is pinned.'),
        "HOME": str(Path(f"~/.codex{slot}").expanduser()),
    })
