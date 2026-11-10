"""Capture the external agent CLI's session transcript into the run bundle.

The `claude`/`codex` backends run an external CLI, so the LLM loop — and
with it the per-turn token usage and the conversation record — lives in that
CLI's own session file, not in `runs/<run_id>/`. The two CLIs store it
differently, so finding it is two lookups behind one entry point:

    $CLAUDE_CONFIG_DIR/projects/<munged-run-dir>/<uuid>.jsonl

where `<munged-run-dir>` is the run dir's absolute path with `/`, `_` and `.`
all rewritten to `-` (the CLI's own project-key scheme; the agent is launched
with the run dir as its cwd, which is what gets munged). For `codex` it is

    $CODEX_HOME/sessions/<YYYY>/<MM>/<DD>/rollout-<local-ts>-<uuid>.jsonl

which is keyed by date rather than by project, so the run→session join is on
`session_meta.payload.cwd` *inside* the file instead of on the path. Both are
found through `find_any_external_transcript`, which dispatches on the run's
recorded `runtime`.

Copying it to `run_dir/agent_transcript.jsonl` keeps a run bundle
self-contained, which is the existing invariant for every other artifact. The
copy runs from the daemon's shutdown path (`agent/server.py`), i.e. after the
agent process is gone, so the transcript is complete — a copy taken when the
agent calls `finish` would be truncated.

A codex rollout is staged with its inline images stripped (see
`_strip_inline_images`); a claude transcript is copied byte for byte.

Best-effort throughout: a missing transcript (a run with no agent CLI, such as
a test's) is normal and must never fail a run.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .usage import summarize_rollouts

TRANSCRIPT_NAME = "agent_transcript.jsonl"

# Config dirs to guess at when neither the run bundle nor the environment says
# which account the run used — the viewer runs in a different shell than the
# run did. Only ever consulted as a last resort: substituting one account's
# session for another's is how a stale transcript gets staged. An entry may be
# a glob: `~/.claude*` is every account dir of the usual shape (`~/.claude`,
# `~/.claude2`, `~/.claude-work`, ...). Order does not decide anything — the
# guess picks the most recent session — so a glob loses nothing.
_FALLBACK_CONFIG_DIRS = ("~/.claude*",)

# meta.json key written by cli.py for the external-CLI claude backends.
_META_CONFIG_DIR_KEY = "claude_config_dir"

# meta.json key written by cli.py for the codex backends: `$CODEX_HOME`, which
# is the account (`~/.codex`, or `~/.codex1`..`~/.codex9` for a numbered
# selector) whose `sessions/` tree holds the rollout.
_META_CODEX_HOME_KEY = "codex_home"

# Guess list for a bundle that predates `_META_CODEX_HOME_KEY` and whose
# `runtime` does not name the account either. Same last-resort role — and same
# hazard — as `_FALLBACK_CONFIG_DIRS`.
_FALLBACK_CODEX_HOMES = ("~/.codex1", "~/.codex", "~/.codex2")

# `codex1` (or the older `codex1_cli`) -> `~/.codex1`, `codex` -> `~/.codex`.
# `runtime` is the backend selector, and the codex selectors differ only in
# which `CODEX_HOME` they point at (`cli/backends/codex.py::_codex_account_backend`),
# so the runtime string recovers the account for a bundle that did not record
# `codex_home`.
_CODEX_RUNTIME_RE = re.compile(r"\A(codex\d*)(?:_cli)?\Z")


def project_key(run_dir: Path) -> str:
    """The CLI's project-directory name for a run dir."""
    return re.sub(r"[/_.]", "-", str(Path(run_dir).resolve()))


def _configured_config_dir() -> Path | None:
    """The account this process is explicitly pointed at, if any."""
    configured = os.getenv("CLAUDE_CONFIG_DIR")
    return Path(configured).expanduser() if configured else None


def read_meta(run_dir: Path) -> dict:
    """The run's `meta.json`, or `{}` when there isn't a readable one."""
    try:
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _recorded_config_dir(run_dir: Path) -> Path | None:
    """The account the run itself recorded, from its own meta.json.

    A property of the run, so it outranks `$CLAUDE_CONFIG_DIR` — the reader's
    shell (viewer, a later analysis script) may well point somewhere else.
    """
    recorded = read_meta(run_dir).get(_META_CONFIG_DIR_KEY)
    if not recorded or not str(recorded).strip():
        return None
    return Path(str(recorded)).expanduser()


def _fallback_config_dirs() -> list[Path]:
    """The guess list, globs expanded to the directories they match, minus
    whatever `$CLAUDE_CONFIG_DIR` already covers."""
    configured = _configured_config_dir()
    out: list[Path] = []
    for raw in _FALLBACK_CONFIG_DIRS:
        path = Path(raw).expanduser()
        if any(ch in path.name for ch in "*?["):
            matches = sorted(p for p in path.parent.glob(path.name) if p.is_dir())
        else:
            matches = [path]
        for match in matches:
            if match == configured or match in out:
                continue
            out.append(match)
    return out


def _candidate_config_dirs() -> list[Path]:
    """Every config dir worth searching, `$CLAUDE_CONFIG_DIR` first."""
    configured = _configured_config_dir()
    return ([configured] if configured else []) + _fallback_config_dirs()


def _normalized(name: str) -> str:
    """Collapse every non-alphanumeric run to a single '-'.

    Guards against the CLI changing exactly which punctuation it rewrites: the
    primary lookup is the exact `project_key`, and this is the fallback match.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "-", name).strip("-").lower()


def _transcripts_under(config_dir: Path, key: str, key_norm: str) -> list[Path]:
    """Every `*.jsonl` this config dir holds for one run-dir project key."""
    projects = config_dir / "projects"
    if not projects.is_dir():
        return []
    exact = projects / key
    candidates = [exact] if exact.is_dir() else [
        d for d in projects.iterdir()
        if d.is_dir() and _normalized(d.name) == key_norm
    ]
    return [f for project in candidates
            for f in project.glob("*.jsonl") if f.is_file()]


def find_claude_transcript(run_dir: Path) -> Path | None:
    """Newest `*.jsonl` in the claude CLI's project dir for `run_dir`, or None.

    When the account is *known* — recorded in the run's meta.json, else named by
    `$CLAUDE_CONFIG_DIR` — only that account is searched. Finding nothing there
    means the session file is not written yet (the CLI can still be flushing it
    as the daemon shuts down), and "not yet" must not silently resolve to some
    other account's leftover for the same run dir.

    That substitution is a real failure, not a hypothetical: relaunching a
    failed run into the same run dir under a second account leaves two sessions
    keyed to that dir, and first-account-wins staged the *stale* one — a 13 KB
    `Not logged in` session from an unauthenticated attempt displaced the 964 KB
    real transcript, so the viewer rendered the run as a single synthetic turn
    with no tool calls and no tokens.

    Only when the account is unknown (an old bundle read from an unrelated
    shell) do we guess across `_FALLBACK_CONFIG_DIRS`, and then by recency
    rather than list order — the run that produced the bundle wrote last.
    """
    key = project_key(run_dir)
    key_norm = _normalized(key)

    known = _recorded_config_dir(run_dir) or _configured_config_dir()
    if known is not None:
        files = _transcripts_under(known, key, key_norm)
        return max(files, key=lambda f: f.stat().st_mtime) if files else None

    files = [f for config_dir in _fallback_config_dirs()
             for f in _transcripts_under(config_dir, key, key_norm)]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------
#
# Codex files its session under a *date* tree rather than a project tree:
#
#     $CODEX_HOME/sessions/<YYYY>/<MM>/<DD>/rollout-<local-ts>-<uuid>.jsonl
#
# so there is no path-munging lookup to do. The run→session join key lives
# inside the file instead: the first record is `session_meta`, whose
# `payload.cwd` is the directory codex was launched in — and `CodexBackend`
# launches with `-C <run_dir>`, so that value is the run dir. Matching on it
# is exact, which means codex needs none of the fuzzy `_normalized` fallback
# the claude path carries.

# The codex session this run drove. Recorded on first successful discovery so
# rediscovery survives a renamed run directory — see `remember_codex_session`.
_META_CODEX_SESSION_KEY = "codex_session_id"
_META_CLAUDE_SESSION_KEY = "claude_session_id"
# Recursive rather than the literal `sessions/<Y>/<M>/<D>/` depth, so a change
# to codex's sharding does not silently return "no transcript" — which would
# look exactly like an `api`-backend run rather than like a broken lookup.
_ROLLOUT_GLOB = "sessions/**/rollout-*.jsonl"


def _codex_homes(run_dir: Path) -> list[Path]:
    """Which `$CODEX_HOME`s to search, most authoritative first.

    Same precedence as the claude side: what the run recorded outranks the
    reader's environment, because the viewer runs in a different shell than
    the run did, which in turn outranks guessing. `CodexBackend` always pins
    CODEX_HOME, so the recorded value is present for every codex run launched
    after that landed.

    **A list, and usually a one-element one.** A recorded or runtime-named
    account is searched *alone*: codex's rollouts are keyed by date across
    every project, so a wrong account is a much larger haystack of other runs'
    sessions to accidentally match in, not just one stale file. Only a bundle
    that names no account at all falls back to the guess list, and then the
    order is the hazard it accepts.

    The runtime route is what keeps pre-`codex_home` bundles resolvable —
    `runtime: codex1_cli` recovers `~/.codex1`. Recorded runs from before that
    field exist, which is why it is not dropped as dead code.
    """
    meta = read_meta(run_dir)
    recorded = meta.get(_META_CODEX_HOME_KEY)
    if recorded and str(recorded).strip():
        return [Path(str(recorded)).expanduser()]
    match = _CODEX_RUNTIME_RE.search(str(meta.get("runtime") or ""))
    if match:
        return [Path(f"~/.{match.group(1)}").expanduser()]
    configured = os.getenv("CODEX_HOME")
    if configured:
        return [Path(configured).expanduser()]
    return [Path(raw).expanduser() for raw in _FALLBACK_CODEX_HOMES]


def _rollout_head(path: Path) -> dict:
    """`session_meta.payload` from a rollout's first record, or `{}`.

    Reads one line, not the file: a rollout runs to megabytes and we may be
    scanning a whole day's worth to find one run's session.
    """
    try:
        import json
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
        record = json.loads(first)
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict) or record.get("type") != "session_meta":
        return {}
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _rollout_cwd(path: Path) -> str | None:
    cwd = _rollout_head(path).get("cwd")
    return str(cwd) if cwd else None


def _is_driving_session(path: Path) -> bool:
    """Whether this rollout is the run's own session, not a sub-thread of it.

    A run also spawns reviewer sub-threads that share its cwd
    (`approvals_reviewer` in codex's `config.toml`). Those carry a
    `parent_thread_id`; the driving session does not. Staging a sub-thread
    reports a fraction of the run, and because the scan below is newest-first
    a sub-thread is the *likelier* match — it is written after the session
    that spawned it.
    """
    return not _rollout_head(path).get("parent_thread_id")


def _rollout_session_id(path: Path) -> str | None:
    sid = _rollout_head(path).get("session_id") or _rollout_head(path).get("id")
    return str(sid) if sid else None


def _recorded_codex_session_id(run_dir: Path) -> str | None:
    """The codex session this run recorded for itself, if any."""
    sid = read_meta(run_dir).get(_META_CODEX_SESSION_KEY)
    return str(sid) if sid and str(sid).strip() else None


def remember_claude_session(run_dir: Path, session_id: str) -> None:
    """Persist the claude session id into `meta.json` (`claude_session_id`), best-effort, for `--resume`: the
    project dir can hold several sessions (a launch whose CLI was started again), and this names the one that ran."""
    _remember(run_dir, _META_CLAUDE_SESSION_KEY, session_id)


def remember_codex_session(run_dir: Path, session_id: str) -> None:
    """Persist the codex session id into `meta.json`, best-effort.

    This is what makes rediscovery survive a renamed run directory. The cwd
    join is otherwise the only link between a run and its rollout, so renaming
    the directory would permanently orphan the session. A recorded id is a property of the run and
    needs no directory-name heuristics: no fuzzy matching, no prefix rules,
    just an identifier that was true when the run happened.

    Merge-read-write rather than a dedicated file so it travels with the
    bundle. Never raises: losing the id costs a fallback, not a run.
    """
    _remember(run_dir, _META_CODEX_SESSION_KEY, session_id)


def _remember(run_dir: Path, key: str, session_id: str) -> None:
    try:
        import json
        path = Path(run_dir) / "meta.json"
        meta = json.loads(path.read_text())
        if not isinstance(meta, dict):
            return
        if meta.get(key) == session_id:
            return
        meta[key] = session_id
        tmp = path.with_name(f".meta.json.{os.getpid()}.session.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, path)
    except (OSError, ValueError):
        return


def find_codex_transcript(run_dir: Path) -> Path | None:
    """The codex rollout for `run_dir`, or None.

    Two joins, in order of authority:

    1. **The recorded `session_id`** (`meta.json`'s `codex_session_id`). An
       identifier, so it keeps working after the run directory is renamed and
       needs no name heuristics.
    2. **`session_meta.cwd == run_dir`**, the only link available before the id
       has been recorded. On a hit the id is persisted, so the *next* lookup
       takes route 1 and the run becomes rename-proof from then on.

    Newest-wins within route 2. A run dir relaunched after a failure leaves a
    short dead rollout keyed to the same cwd, and route 1 handles that case
    (the id pins the session outright), so route 2 needs no size heuristic.

    Both routes run per account, in `_codex_homes` order, so a bundle that
    named its account never has another account's `sessions/` searched.
    """
    run_dir = Path(run_dir)
    try:
        target = str(run_dir.resolve())
    except OSError:
        target = str(run_dir)
    recorded = _recorded_codex_session_id(run_dir)

    for home in _codex_homes(run_dir):
        if not (home / "sessions").is_dir():
            continue
        if recorded:
            for path in home.glob(_ROLLOUT_GLOB):
                # The id is in the filename (`rollout-<ts>-<uuid>.jsonl`), so
                # this needs no file reads at all in the common case; the
                # header check only guards against a name that happens to
                # embed the substring.
                if recorded in path.name and _rollout_session_id(path) == recorded:
                    return path
            # Recorded but absent: fall through rather than give up. The
            # session file may have been pruned while the bundle still carries
            # the id.
        # Walk newest-first and stop at the first match. It opens one file
        # instead of every rollout in the tree — this directory accumulates one
        # file per codex session ever run on the machine, while the one we want
        # is almost always the most recent.
        candidates = []
        for path in home.glob(_ROLLOUT_GLOB):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
        for _, path in sorted(candidates, reverse=True):
            if _rollout_cwd(path) == target and _is_driving_session(path):
                session_id = _rollout_session_id(path)
                if session_id:
                    remember_codex_session(run_dir, session_id)
                return path
    return None


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def _runtime(run_dir: Path) -> str:
    """`meta.json`'s `runtime` field (the backend selector: "claude", "codex1", ...)."""
    return str(read_meta(run_dir).get("runtime") or "")


def is_codex_run(run_dir: Path) -> bool:
    """Whether this bundle's LLM loop was owned by the codex CLI.

    Deliberately not `_runtime(...).startswith("codex")`. A run that recorded
    its `$CODEX_HOME` is a codex run whatever its `runtime` says or fails to
    say, and the runtime itself is the backend selector — so the test has to be
    the same `_CODEX_RUNTIME_RE` the account lookup uses, or the two can
    disagree about one bundle. Both fields are load-bearing; see
    `_codex_homes`, which reads exactly these two to pick a `sessions/` tree.
    """
    meta = read_meta(run_dir)
    return bool(meta.get(_META_CODEX_HOME_KEY)) or bool(
        _CODEX_RUNTIME_RE.search(str(meta.get("runtime") or "")))


def find_any_external_transcript(run_dir: Path) -> Path | None:
    """The external transcript for `run_dir`, whichever CLI produced it.

    Dispatches on `meta.json`'s `runtime` rather than trying both: the two
    schemes cannot collide (one is a project-keyed dir, the other a dated
    rollout), but searching the wrong one is wasted I/O over a whole
    `sessions/` tree. An unknown runtime tries claude first and codex second,
    which is what an old bundle written before `runtime` existed needs.
    """
    run_dir = Path(run_dir)
    if is_codex_run(run_dir):
        return find_codex_transcript(run_dir)
    if _runtime(run_dir).startswith("claude"):
        return find_claude_transcript(run_dir)
    return find_claude_transcript(run_dir) or find_codex_transcript(run_dir)


def resolve_transcript(run_dir: Path) -> Path | None:
    """The transcript for `run_dir`: the staged copy if present, else external.

    This is the read path used by the viewer. It prefers the in-bundle copy so
    a run stays viewable after the CLI's config dir is cleaned or the bundle is
    moved to another machine.
    """
    run_dir = Path(run_dir)
    staged = run_dir / TRANSCRIPT_NAME
    if staged.is_file() and staged.stat().st_size > 0:
        return staged
    return find_any_external_transcript(run_dir)


# A base64 data URI inside a JSON string. Base64 has no character that JSON
# escapes, so the string cannot contain a backslash and `"` reliably ends it.
# The length floor keeps the pattern off the short `data:` URIs that appear in
# prose (an agent quoting a one-line SVG placeholder, say).
_INLINE_IMAGE_RE = re.compile(
    rb'"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]{512,}"')


def _strip_inline_images(data: bytes) -> tuple[bytes, int]:
    """Replace inline base64 images with a marker. Returns `(bytes, count)`.

    Every image codex sends the model is a snapshot the harness rendered, and
    the PNG is already in the bundle under `snapshots/` — the rollout's own
    `ImageView` items record the paths. A second base64 copy would be most of
    the file's size.

    Byte-substituted rather than parsed and re-serialized, so every other byte
    of the record — key order, spacing, unicode escaping — survives the copy
    exactly as codex wrote it.
    """
    count = 0

    def replace(match: "re.Match[bytes]") -> bytes:
        nonlocal count
        count += 1
        # -2 for the quotes the pattern includes.
        return (b'"<inline image stripped by dycraft: %d bytes of base64. '
                b'The render itself is in this bundle under snapshots/>"'
                % (len(match.group(0)) - 2))

    return _INLINE_IMAGE_RE.sub(replace, data), count


def capture(run_dir: Path) -> str:
    """Stage the external transcript into `run_dir`. Returns a status string.

    ``"copied"`` on success, ``"unavailable"`` when there is no external
    transcript, ``"failed"`` on an I/O error.

    Raises nothing of its own, but it is **not** proof against a shutdown
    signal: the file copied here runs to megabytes and this is called from
    `AgentSession.finalize_meta` on the way out, so a `SystemExit` raised in a
    signal handler can land inside `read_bytes` — as one did, taking
    `meta.json`'s outcome with it. That is handled by the caller, which holds
    the exception until `meta.json` is written; see the comment there.

    A claude transcript is copied verbatim. A codex rollout has its inline
    images stripped first (`_strip_inline_images`) — nothing unrecoverable, and
    the difference between a 1 MB bundle artifact and a 50 MB one.
    """
    try:
        run_dir = Path(run_dir)
        source = find_any_external_transcript(run_dir)
        if source is None:
            return "unavailable"
        target = run_dir / TRANSCRIPT_NAME
        data = source.read_bytes()
        if not data:
            return "unavailable"
        if is_codex_run(run_dir):
            data, _ = _strip_inline_images(data)
        # Write-then-replace: a reader must never see a half-copied transcript.
        tmp = target.with_suffix(".jsonl.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        # Record the session id here too, not only on a discovery miss. A run
        # that staged successfully never calls `find_codex_transcript` again
        # (`resolve_transcript` prefers the in-bundle copy), so without this it
        # would be the one shape with no recorded id — and if the staged copy
        # were later pruned from a since-renamed directory there would be
        # nothing left to re-find it by.
        session_id = _rollout_session_id(source)
        if session_id:
            remember_codex_session(run_dir, session_id)
        return "copied"
    except OSError:
        return "failed"


# ---------------------------------------------------------------------------
# The approval reviewer
# ---------------------------------------------------------------------------

def _reviewer_rollouts(run_dir: Path, driving_id: str | None) -> list[Path]:
    """Rollouts that are approval-reviewer sub-threads of this run.

    Keyed on `parent_thread_id`, which `_is_driving_session` already relies on:
    a reviewer sub-thread carries the driving session's id there and the
    driving session carries nothing. Falling back to cwd when the run recorded
    no session id keeps pre-`codex_session` bundles resolvable, at the cost of
    matching a reviewer of a *different* run in the same directory -- which is
    why the id is preferred when it exists.

    A sub-agent the run spawned has a `parent_thread_id` too; only
    `thread_source == "guardian_review"` is the reviewer.
    """
    found: list[Path] = []
    target = str(run_dir.resolve())
    for home in _codex_homes(run_dir):
        if not home.is_dir():
            continue
        for path in home.glob(_ROLLOUT_GLOB):
            head = _rollout_head(path)
            parent = head.get("parent_thread_id")
            if not parent or head.get("thread_source") != "guardian_review":
                continue
            if driving_id:
                if str(parent) == driving_id:
                    found.append(path)
            elif str(head.get("cwd") or "") == target:
                found.append(path)
    return found


def approval_reviewer_usage(run_dir: Path) -> dict | None:
    """What the approval reviewer cost this run, or `None` if there was none.

    **Why this is recorded at all.** Every escalated command is adjudicated by
    a second model session that returns `{"risk_level", "user_authorization",
    "outcome", "rationale"}` and holds no tools. It is invisible in
    `meta.json`, yet it is not small: it re-reads the agent's transcript for
    each verdict, so its cost scales with transcript length times escalation
    count, superlinearly in run length. The transport's FIFO rung
    (`_harness_core.py`) is what stops those escalations being generated; this
    is how a run's cost shows whether it did.

    Never raises: losing the accounting must not cost a run its `meta.json`.
    """
    import json

    try:
        run_dir = Path(run_dir)
        paths = _reviewer_rollouts(run_dir, _recorded_codex_session_id(run_dir))
        if not paths:
            return None
        totals = {"input_tokens": 0, "cached_input_tokens": 0,
                  "output_tokens": 0, "total_tokens": 0}
        outcomes: dict[str, int] = {}
        verdicts = 0
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    payload = record.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if record.get("type") == "token_usage_record":
                        usage = payload.get("usage")
                        if isinstance(usage, dict):
                            for key in totals:
                                totals[key] += int(usage.get(key) or 0)
                    elif (record.get("type") == "response_item"
                            and payload.get("type") == "message"
                            and payload.get("role") == "assistant"):
                        text = "".join(
                            part.get("text", "")
                            for part in (payload.get("content") or [])
                            if isinstance(part, dict))
                        outcome = _verdict_outcome(text)
                        if outcome is not None:
                            verdicts += 1
                            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        totals["uncached_input_tokens"] = (totals["input_tokens"]
                                           - totals["cached_input_tokens"])
        # The reviewer's *own* id: its `session_meta` carries the parent's under
        # `session_id`, so preferring that would name the agent's session twice.
        return {"sessions": [str(_rollout_head(p).get("id")
                                 or _rollout_session_id(p) or p.stem)
                             for p in paths],
                "verdicts": verdicts, "outcomes": outcomes, **totals}
    except (OSError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Sub-agents a codex run spawned
# ---------------------------------------------------------------------------

def _subagent_rollouts(run_dir: Path, driving_id: str) -> list[Path]:
    """Rollouts of every thread the run's own session spawned, at any depth.

    A codex sub-agent is a session of its own, with its own rollout, that names
    its spawner as `parent_thread_id`; one it spawns in turn names it. Its
    rollout records only its own responses (a fork's history is not re-counted
    there), so the threads add without double counting. The driving session's
    approval reviewer is a sub-thread too, but `approval_reviewer_usage`
    already accounts for it, so it is left out here.
    """
    children: dict[str, list[tuple[Path, str, str]]] = {}
    for home in _codex_homes(run_dir):
        if not home.is_dir():
            continue
        for path in home.glob(_ROLLOUT_GLOB):
            head = _rollout_head(path)
            parent = head.get("parent_thread_id")
            if parent:
                children.setdefault(str(parent), []).append(
                    (path, str(head.get("id") or ""),
                     str(head.get("thread_source") or "")))
    found: list[Path] = []
    seen = {driving_id}
    frontier = [driving_id]
    while frontier:
        thread = frontier.pop()
        for path, child, source in children.get(thread, ()):
            if thread == driving_id and source == "guardian_review":
                continue
            if path in found:
                continue
            found.append(path)
            if child and child not in seen:
                seen.add(child)
                frontier.append(child)
    return sorted(found)


def subagent_usage(run_dir: Path) -> dict | None:
    """What the sub-agents a codex run spawned cost, or `None` if it spawned
    none (or is not a codex run).

    The run's own transcript is one thread. A codex agent can spawn
    sub-agents, each with a rollout of its own that `capture` does not stage
    and `usage.summarize` therefore never reads, and they can spend about as
    many tokens as the run's own session. This finds them by walking `parent_thread_id`
    down from the run's session, and sums them the way the run's own
    transcript is summed.

    Never raises: losing the accounting must not cost a run its `meta.json`.
    """
    try:
        run_dir = Path(run_dir)
        driving = (_recorded_codex_session_id(run_dir)
                   or _rollout_head(run_dir / TRANSCRIPT_NAME).get("id"))
        if not driving:
            return None
        paths = _subagent_rollouts(run_dir, str(driving))
        if not paths:
            return None
        block = summarize_rollouts(paths)
        if block:
            block["sessions"] = [str(_rollout_head(p).get("id") or p.stem)
                                 for p in paths]
        return block
    except (OSError, ValueError, TypeError):
        return None


def _verdict_outcome(text: str) -> str | None:
    """The `outcome` of one adjudication, or None if this is not a verdict.

    The reviewer answers with a bare JSON object, but it is a model and the
    object arrives fenced often enough that a strict `json.loads` of the whole
    message misses verdicts. A narrow regex over the one field we are counting
    is both more robust and cheaper than parsing every assistant message.
    """
    import re

    match = re.search(r'"outcome"\s*:\s*"([A-Za-z_]+)"', text)
    return match.group(1) if match else None
