"""`python -m dycraft.cli --resume RUN [--prompt TEXT|FILE] [--max-hours H]`: carry a stopped run on in its own agent
session, or revise a delivered one.

The agent CLI resumes the conversation the run ran as (`claude --resume <id>`, `codex resume <id>`), with the run's
layer, model, effort, and render backend from `meta.json`, on the account it last ran on, or on another account of the
same CLI (`--backend codex2`): its session is then copied into that account's home first, so runs can be chained
across accounts as each reaches its usage limit. In order:

  1. read the run, and refuse (one FATAL line, exit 2, nothing changed) a run that is still live, never started, has
     lost its session file, was moved (claude finds a session by its path), delivered with no --prompt, used its time
     with no --max-hours, or was ended by its own agent with no --prompt;
  2. the launch's own checks: the environment preflight and the backend's;
  3. set the stopped run's `.terminate` aside; a delivered run with --prompt has its delivery moved to
     `prior_deliveries/<n>/` (`run_layout.reopen_delivery`); `meta.json` records the resume;
  4. restage the harness client and skills from this checkout (the new daemon's ops), keep the staged system prompt;
     the rewriters, prior work, and preloaded assets are not run again;
  5. a time limit of the run's own allowance less the agent time it used (`agent_session.agent_time`), or
     `--max-hours`; the advisory budget likewise;
  6. a new daemon (its steps go on from the trace) and watchers, and the CLI with the resume message.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .._layers import canonical_layer
from ..agent.agent_session import agent_time

#: What each stop is called in the message the session resumes with ("This session was interrupted (WHY)").
_WHY = {
    "claude_rate_limit": "your account reached its usage limit",
    "claude_authentication_failed": "your account's login was refused",
    "claude_billing_error": "your account could not be billed",
    "oom_kill": "the machine ran out of memory and your CLI was stopped",
    "sigkill": "your CLI was stopped",
    "parent_gone": "the launcher running you was stopped",
    "codex_lost_stream": "your model could not be reached",
    "codex_reconnecting": "your model could not be reached",
    "claude_exited_before_answering": "your CLI exited before it answered",
    "codex_exited_before_answering": "your CLI exited before it answered",
}

ONE_LINE_FLAGS = ("--resume takes the layer, model, effort and render backend from the run; only --prompt, "
                  "--max-hours, --backend (another account of the same CLI) and --allow-unsandboxed go with it.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dycraft.cli --resume",
        description="Carry a stopped run on in its own agent session, or revise a delivered one with --prompt.")
    parser.add_argument("--resume", required=True, metavar="RUN_DIR",
                        help="The run: its folder under runs/, or the folder inside it that holds meta.json.")
    parser.add_argument("--prompt", default=None,
                        help="What the agent is told on resuming: text, or a file, read as a launch's --prompt is "
                             "(a text file is the message; any other file is staged in inputs/ for the agent to "
                             "open). Required to revise a delivered run, or one its agent ended itself.")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Hours the resumed session may run (0: no limit). Default: the run's own allowance less "
                             "the agent time it used.")
    parser.add_argument("--backend", default=None, metavar="BACKEND",
                        help="Resume on another account of the same CLI (claude, claude1..claude9; codex, "
                             "codex1..codex9), for one whose usage limit was reached. Default: the account the run "
                             "last ran on.")
    parser.add_argument("--allow-unsandboxed", "--allow-unsandboxed-codex", dest="allow_unsandboxed",
                        action="store_true",
                        help="As for a launch: resume without the CLI's sandbox where it cannot start here. Not "
                             "carried over from the launch.")
    return parser


def _refuse(message: str, fix: list[str] | None = None) -> int:
    print(f"FATAL: {message}", file=sys.stderr)
    for line in fix or ():
        print(f"  fix: {line}", file=sys.stderr)
    sys.stderr.flush()
    return 2


def _layout():
    """`(internal_dir, reader_dir, reopen_delivery)`: where a run's records are, the folder a reader opens, and how a
    delivered run is reopened (`run_layout`)."""
    from ..run_layout import internal_dir, reader_dir, reopen_delivery
    return internal_dir, reader_dir, reopen_delivery


# ---------------------------------------------------------------------------
# The session the run ran as
# ---------------------------------------------------------------------------

def _answered_claude(path: Path) -> bool:
    try:
        with path.open(errors="replace") as fh:
            for line in fh:
                if '"assistant"' in line and '"isApiErrorMessage"' not in line:
                    try:
                        if json.loads(line).get("type") == "assistant":
                            return True
                    except ValueError:
                        continue
    except OSError:
        pass
    return False


def _claude_session(run_dir: Path, meta: dict) -> dict:
    """`{"id", "path", "account", "cwd"}` of the claude session the run ran as; `path` None when none is left."""
    from ..agent.transcript_capture import project_key
    account = Path(meta.get("claude_config_dir") or os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude").expanduser()
    project = account / "projects" / project_key(run_dir)
    sid, cwd = meta.get("claude_session_id"), None
    copy = run_dir / "agent_transcript.jsonl"           # the daemon's copy, for runs that predate claude_session_id
    if copy.is_file():
        with copy.open(errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                sid, cwd = sid or rec.get("sessionId"), cwd or rec.get("cwd")
                if sid and cwd:
                    break
    if not sid and project.is_dir():
        answered = sorted((p for p in project.glob("*.jsonl") if _answered_claude(p)), key=lambda p: p.stat().st_mtime)
        sid = answered[-1].stem if answered else None
    path = project / f"{sid}.jsonl" if sid else None
    return {"id": sid, "path": path if path and path.is_file() else None, "account": account, "cwd": cwd,
            "answered": bool(path and path.is_file() and _answered_claude(path))}


def _codex_session(run_dir: Path, meta: dict) -> dict:
    from ..agent.transcript_capture import _rollout_session_id, find_codex_transcript
    from .backends._process import _model_answered
    home = Path(meta.get("codex_home") or "~/.codex").expanduser()
    sid, path = meta.get("codex_session_id"), None
    if sid:
        path = next(iter(sorted((home / "sessions").glob(f"**/rollout-*{sid}.jsonl"))), None)
    else:
        path = find_codex_transcript(run_dir)
        sid = _rollout_session_id(path) if path else None
    answered = False
    if path is not None:
        with path.open(errors="replace") as fh:
            for line in fh:
                try:
                    if _model_answered(json.loads(line)):
                        answered = True
                        break
                except ValueError:
                    continue
    return {"id": sid, "path": path, "account": home, "cwd": None, "answered": answered}


def _backend_for(kind: str, account: Path):
    """The backend whose account is `account`, the one the session lives in."""
    from .backends import BACKENDS
    from .backends.claude import CLAUDE_CONFIG_DIR_ALIASES
    want = account.resolve()
    if kind == "claude":
        for name, alias in CLAUDE_CONFIG_DIR_ALIASES.items():
            if Path(alias).expanduser().resolve() == want and name in BACKENDS:
                return BACKENDS[name]()
        os.environ["CLAUDE_CONFIG_DIR"] = str(account)  # plain `claude` follows the ambient account
        return BACKENDS["claude"]()
    for name, cls in BACKENDS.items():
        if name.startswith("codex") and Path(cls.HOME).expanduser().resolve() == want:
            return cls()
    return None


def _account_of(backend) -> Path:
    """The home a backend's CLI keeps its login and sessions in."""
    if backend.name.startswith("codex"):
        return Path(backend.HOME).expanduser()
    return type(backend)._config_dir().expanduser()


def _carry_session(kind: str, run_dir: Path, session: dict, account: Path) -> Path:
    """Copy the session into `account`'s home and return its path there. A copy, not a move: the account it ran on
    keeps its record. A copy already there, from an earlier hop of a chain, is replaced by the newer session."""
    import shutil
    source: Path = session["path"]
    if kind == "claude":
        from ..agent.transcript_capture import project_key
        target = account / "projects" / project_key(run_dir) / source.name
        sidecar = source.with_suffix("")                     # `<id>/`: the session's sub-agent and tool records
        if sidecar.is_dir():
            shutil.copytree(sidecar, target.with_suffix(""), dirs_exist_ok=True)
    else:
        try:
            target = account / source.relative_to(session["account"])
        except ValueError:                                   # found outside its home: keep codex's own layout
            target = account / "sessions" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _live(run_dir: Path, kind: str, session: dict) -> str | None:
    """What still drives or serves this run, or None."""
    import fcntl
    from ..agent.server import OWNER_LOCK_NAME, _find_agent_pid, read_owner_info
    lock = run_dir / OWNER_LOCK_NAME
    if lock.exists():
        with lock.open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                return f"a harness daemon holds it (pid {read_owner_info(run_dir).get('pid')})"
    pid = _find_agent_pid(run_dir, kind)
    if pid:
        return f"pid {pid} is running its agent CLI"
    if kind == "codex" and session["id"]:
        held = session["account"] / "thread-writer-locks" / f"{session['id']}.lock"
        if held.exists():
            with held.open("a") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    return f"a codex process holds its session {session['id']}"
    return None


# ---------------------------------------------------------------------------
# The message the session resumes with
# ---------------------------------------------------------------------------

def why_it_stopped(meta: dict, ended_itself: bool) -> str:
    """The reason in the resume message, from what `meta.json` recorded about the stop."""
    if (meta.get("time_limit") or {}).get("reached"):
        return "the run's time limit was reached"
    if ended_itself:
        return "you ended the session yourself without delivering"
    stop = meta.get("agent_stop") or {}
    reason, detail = str(stop.get("reason") or ""), str(stop.get("detail") or "")
    if reason == "codex_fatal_error":
        return ("your account reached its usage limit" if detail.startswith("usage_limit_exceeded")
                else "your model could not be reached")
    if reason.startswith("claude_api_error"):
        return "your model could not be reached"
    return _WHY.get(reason, "for a reason that was not recorded")


def resume_message(*, why: str, prompt: str | None, delivered_note: str | None) -> str:
    """The message the session resumes with (agent-facing)."""
    if delivered_note:
        return f"{delivered_note}\n\n{prompt}"
    said = (f"This session was interrupted ({why}) and has been resumed: the harness daemon was restarted and "
            "answers again, and your files are as you left them.")
    if prompt:
        return f"{said} The operator adds:\n\n{prompt}"
    return f"{said} Continue the task from where you left off."


def _read_prompt(text: str | None):
    """`(prompt, brief, error)`: `--prompt` read as a launch reads it (`prompt_files`), a non-text file left to stage."""
    if text is None:
        return None, None, None
    from . import prompt_files
    read, brief, note, error = prompt_files.read(text)
    if error:
        return None, None, error
    if read is not None and not read.strip():
        return None, None, "--prompt is empty"
    if note:
        print(note, file=sys.stderr)
    return read, brief, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    from . import cli as C
    args, extra = build_parser().parse_known_args(argv)
    if extra:
        return _refuse(f"{ONE_LINE_FLAGS} Got: {' '.join(extra)}")
    if args.max_hours is not None and args.max_hours < 0:
        return _refuse("--max-hours must be 0 (no limit) or a positive number of hours")
    if args.allow_unsandboxed:
        from .backends import ALLOW_UNSANDBOXED_ENV
        os.environ[ALLOW_UNSANDBOXED_ENV] = "1"
    internal_dir, reader_dir, reopen_delivery = _layout()
    given = Path(args.resume).expanduser().resolve()
    run_dir = internal_dir(given)
    shown = reader_dir(run_dir)
    try:
        meta = json.loads((run_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return _refuse(f"{given} is not a run directory: it has no readable meta.json.")
    prompt, brief, bad_prompt = _read_prompt(args.prompt)
    if bad_prompt:
        return _refuse(bad_prompt)
    prompted = prompt is not None or brief is not None

    runtime = str(meta.get("runtime") or "")
    kind = "codex" if runtime.startswith("codex") else "claude" if runtime.startswith("claude") else ""
    layer = canonical_layer(meta.get("layer"))
    if not kind or not layer:
        return _refuse(f"{shown}'s meta.json names no claude or codex backend and layer "
                       f"(runtime {runtime!r}, layer {meta.get('layer')!r}).")
    session = _claude_session(run_dir, meta) if kind == "claude" else _codex_session(run_dir, meta)

    # 1. What cannot be resumed, before anything changes.
    live = _live(run_dir, kind, session)
    if live:
        return _refuse(f"{shown} is still running: {live}. Stop it first; --resume starts its own daemon and CLI.")
    if kind == "claude" and session["cwd"] and Path(session["cwd"]).resolve() != run_dir:
        return _refuse(f"{shown} was started at {session['cwd']}, and claude looks its session up by that path; "
                       "move it back to resume it.")
    if not session["id"] or (session["path"] is not None and not session["answered"]):
        failure = str((meta.get("launch_failure") or {}).get("message") or "no session was recorded")
        failure = failure[len("FATAL: "):] if failure.startswith("FATAL: ") else failure
        return _refuse(f"nothing to resume in {shown}: its agent never started ({failure}). Launch it again with "
                       "`python -m dycraft.cli ...`.")
    if session["path"] is None:
        where = "CLAUDE_CONFIG_DIR" if kind == "claude" else "CODEX_HOME"
        return _refuse(f"session {session['id']}, which this run used, is no longer in {session['account']}, so it "
                       f"cannot be resumed. Was {where} cleaned or changed? A new launch with the same prompt starts "
                       "over.")
    delivered = meta.get("outcome") in ("committed", "delivered_with_exceptions")
    if delivered and not prompted:
        return _refuse(f"{shown} delivered its film (outcome {meta.get('outcome')}). To revise it, resume with "
                       "--prompt \"<what to change>\".")
    terminate = run_dir / ".terminate"
    ended_itself = terminate.exists() and not delivered
    if ended_itself and not prompted:
        stamp = time.strftime("%H:%M %a %d %b", time.localtime(terminate.stat().st_mtime))
        return _refuse(f"the agent ended this run itself without delivering (at {stamp}). To continue it anyway, "
                       "resume with --prompt \"<what to do>\".")
    used = agent_time(run_dir)
    limit = meta.get("time_limit") or {}
    allowance = limit.get("hours")
    if args.max_hours is not None:
        hours = args.max_hours or None
    elif allowance:
        left = float(allowance) * 3600 - used["seconds"]
        if limit.get("reached") or left < 60:
            return _refuse(f"this run used its {float(allowance):g} h limit; pass --max-hours H to continue it for H "
                           "more hours")
        hours = left / 3600
    else:
        hours = None
    if args.backend:
        from .backends import BACKENDS
        family = "codex" if args.backend.startswith("codex") else "claude" if args.backend.startswith("claude") else ""
        if args.backend not in BACKENDS or family != kind:
            return _refuse(f"--backend {args.backend} is not a {kind} account, and this run's session is a {kind} "
                           f"session; resume it on {kind} or {kind}1..{kind}9.")
        backend = BACKENDS[args.backend]()
    else:
        backend = _backend_for(kind, session["account"])
        if backend is None:
            return _refuse(f"the session lives in {session['account']}, which no codex backend uses (codex, "
                           "codex1..codex9 use ~/.codex, ~/.codexN).")
    account = _account_of(backend)
    moved_from = session["account"] if account.resolve() != Path(session["account"]).resolve() else None

    # 2. The launch's own checks, leaving meta.json as it was when they refuse.
    from ..render import DEFAULT_RENDER_BACKEND, RENDER_BACKEND_ENV
    os.environ[RENDER_BACKEND_ENV] = meta.get("render_backend") or DEFAULT_RENDER_BACKEND
    before = (run_dir / "meta.json").read_bytes()
    runtime = args.backend or runtime
    checks = argparse.Namespace(layer=layer, backend=runtime)
    refusal = C._environment_preflight(checks, backend, run_dir) or backend.preflight(run_dir, layer)
    if refusal is not None:
        (run_dir / "meta.json").write_bytes(before)
        return refusal

    # 3. The session in the account it resumes on, and the run dir made ready.
    if moved_from is not None:
        session["path"] = _carry_session(kind, run_dir, session, account)
        print(f"[cli] copied session {session['id']} from {moved_from} to {account}", file=sys.stderr)
    why = why_it_stopped(meta, ended_itself)
    stamp = time.strftime("%y%m%d_%H%M%S")
    print(f"[cli] resuming {shown}: {runtime}, {layer}, session {session['id']}, last stopped: {why}",
          file=sys.stderr, flush=True)
    if terminate.exists():
        terminate.rename(run_dir / f".terminate.{stamp}")
    reopened = None
    if delivered:
        try:
            reopened = reopen_delivery(run_dir)
        except RuntimeError as exc:
            return _refuse(f"could not move {shown}'s delivery aside: {exc}. Nothing was changed.")
        if reopened:
            print(f"[cli] moved the previous delivery to {run_dir / reopened['prior']}/", file=sys.stderr)
    entry = {"at": C._iso(time.time()), "from_stop": (meta.get("agent_stop") or {}).get("reason"),
             "previous_outcome": meta.get("outcome"), "session": session["id"], "backend": runtime,
             "account": str(account), "prior_delivery": reopened["prior"] if reopened else None}
    if moved_from is not None:
        entry["from_account"] = str(moved_from)
    # What the backend's preflight found about its sandbox for this session (`codex_sandbox`, `unsandboxed`), and how
    # this machine confines Blender.
    entry.update(backend.launch_meta)
    if layer == "animation":
        entry["blender_sandbox"] = C.blender_sandbox_meta()
    if brief is not None:
        from . import prompt_files
        staged = prompt_files.stage(brief, run_dir)
        prompt = brief.task(staged)
        entry.update(prompt_file=str(brief.path), staged=staged)
    elif args.prompt is not None and Path(args.prompt).expanduser().is_file():
        entry["prompt_file"] = str(Path(args.prompt).expanduser().resolve())
    elif prompt is not None:
        entry["prompt"] = prompt
    deadline = time.time() + hours * 3600 if hours else None
    advisory = float(meta.get("advisory_hours", 1.0) or 0)
    advisory = max(1 / 60, advisory - used["seconds"] / 3600) if advisory > 0 else 0.0

    def change(record: dict) -> None:
        record.update(outcome=None, finished_at=None, agent_time=used, runtime=runtime)
        record["claude_config_dir" if kind == "claude" else "codex_home"] = str(account)
        record.setdefault("resumes", []).append(entry)
        for key in ("unsandboxed", "blender_sandbox"):
            if key in entry:
                record[key] = entry[key]
        record.pop("agent_stop", None)
        if deadline is not None:
            record["time_limit"] = {"hours": round(used["seconds"] / 3600 + hours, 3), "deadline": C._iso(deadline),
                                    "reached": False}
        else:
            record.pop("time_limit", None)
    C._merge_meta(run_dir, change)
    if deadline is not None:
        print(f"[cli] time limit: {hours:.2g} h from now ({used['seconds'] / 3600:.2g} h used by the agent so far)",
              file=sys.stderr)

    # 4-6. Restage, start the daemon and watchers, and resume the session.
    from ..agent import LAYERS, stage_harness_client
    from ..agent.agent_runner import SYSTEM_PROMPT_FILE, TASK_FILE, TASK_FILE_ENV
    from .backends import LaunchPlan
    stage_harness_client(run_dir, layer)
    system_prompt = (run_dir / SYSTEM_PROMPT_FILE).read_text() if (run_dir / SYSTEM_PROMPT_FILE).is_file() else ""
    message = resume_message(why=why, prompt=prompt, delivered_note=reopened["note"] if reopened else None)
    message += C.time_limit_note(deadline, layer, since="this resume")
    env = C.build_env_dict(meta.get("prompt") or "", run_dir=run_dir, runs_dir=shown.parent, run_id=shown.name,
                           layer=layer, runtime=runtime, advisory_hours=advisory,
                           render_backend=os.environ[RENDER_BACKEND_ENV])
    if (run_dir / TASK_FILE).is_file():
        env[TASK_FILE_ENV] = str(run_dir / TASK_FILE)
    env[C.DEADLINE_ENV] = repr(deadline) if deadline is not None else "inf"
    if backend.harness_wait_s:
        env.setdefault("DYCRAFT_HARNESS_WAIT_S", backend.harness_wait_s)
    if backend.agent_cmdline_marker:
        env["DYCRAFT_AGENT_CMDLINE_MARKER"] = backend.agent_cmdline_marker
    backend.prepare_shared_env(env)
    label = f"{runtime}_{layer}"
    daemon_pid = C.start_harness_daemon(run_dir, env, daemon_module=LAYERS[layer].server_module, log_label=label)
    C.start_self_kill_watcher(run_dir, os.getpid(), log_label=label)
    C.install_self_kill_exit(run_dir, log_label=label)
    from ..run_layout import publish
    publish(run_dir)                # the README says it is running again; a delivery set aside leaves the top level
    plan = LaunchPlan(layer=layer, user_prompt=meta.get("prompt") or "", run_dir=run_dir, runs_dir=shown.parent,
                      run_id=shown.name, model=meta.get("model"), env=env, system_prompt_text=system_prompt,
                      footer_text=message, reasoning_effort=meta.get("reasoning_effort"),
                      resume_session=session["id"])
    try:
        return C.run_agent(backend, plan, deadline, daemon_pid=daemon_pid)
    finally:
        publish(run_dir)


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main(sys.argv[1:]))
