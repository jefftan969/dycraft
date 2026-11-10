"""One object / scene / animation agent run, laid out as one labeled band.

The inside of a single run: every harness op, every tool call the harness never sees, and the two LLM latencies
that dominate the clock. Everything lands on one timeline in nine labeled categories, with the legend below it.
One row puts any two hues side by side; the palette in `web/src/index.css` clears the all-pairs color-vision floors
(worst normal-vision dE 17.4, worst min(protan, deutan) dE 10.4, against 15 / 8), so category order is semantic,
and every segment also carries a text label, with color reinforcing it.

The categories follow where a run's time goes. LLM generation is over half of a run's wall-clock, so it is a
category, not background. A `review` runs a sub-agent that grades the work against the brief, so it is its own
Judge category rather than a Look.

Duration provenance, in priority order:

1. `trace.jsonl`'s `op_result.elapsed_ms` for a harness op, which is authoritative, paired to its `op_call`
   through a per-step FIFO because `step` repeats across daemon sessions in a resumed run.
2. Transcript timestamp deltas for everything else (`Read`, `Edit`, `Skill`, plain shell), which include the
   socket round trip and can overlap when the CLI runs calls in parallel; `notes` says so.
3. Model generation runs from the previous record to the turn's first tool call, so it abuts its own tool calls
   rather than containing them.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .._layers import canonical_layer
from . import _trace
from .trajectory import build as build_trajectory

# ---------------------------------------------------------------------------
# lanes
# ---------------------------------------------------------------------------

#: Semantic order — the arc of a run, with the two model categories first
#: because they are the two biggest. Unlike the previous faceted revision this
#: order is NOT load-bearing: the palette clears the all-pairs CVD floors, so
#: no pair depends on never being adjacent. It is only the legend's order.
CATEGORY_ORDER = ("model", "judge", "author", "build", "inspect", "look",
                  "solve", "deliver", "aside")

CATEGORY_LABEL = {
    "model": "Model",
    "judge": "Review",
    "author": "Author",
    "build": "Build",
    "inspect": "Inspect",
    "look": "Look",
    "solve": "Solve",
    "deliver": "Deliver",
    "aside": "Aside",
}

CATEGORY_BLURB = {
    "model": "the agent's own LLM generating a turn",
    "judge": "an independent review the agent asked for",
    "author": "the source changed",
    "build": "the IR was rebuilt and checked",
    "inspect": "measured or re-read; nothing changed",
    "look": "the agent chose to look at pixels",
    "solve": "external compute produced numbers to author from",
    "deliver": "terminal artifacts, and the caveats on them",
    "aside": "housekeeping outside the graded loop",
}

#: Which category every action key belongs to. Exhaustive over what the corpus
#: produces; anything unrecognized becomes `aside.unknown`, which is asserted
#: empty by the test suite so that a new CLI tool shows up as a test failure
#: rather than silently vanishing into the last category.
ACTION_CATEGORY = {
    # --- the two model categories ---------------------------------------
    "model.turn": "model",
    # `review` dispatches a sub-agent that grades the scene against the brief
    # and returns per-claim verdicts. It is Judge and not Look even though the
    # agent asked for it: what the row measures is a *grading pass*.
    "op:review": "judge",
    # --- author ---------------------------------------------------------
    "author.source": "author",
    "author.aux": "author",
    "author.summary": "author",
    "author.experiment": "author",
    "op:write_file": "author",
    "op:plan": "author",
    "op:author_solids": "author",   # retired; still in older bundles
    "op:author_mesh": "author",     # retired
    # --- build ----------------------------------------------------------
    "op:compile": "build",
    "op:build": "build",
    "op:finish": "build",
    # --- inspect --------------------------------------------------------
    "op:probe": "inspect",
    # The animation layer's diagnostic measurement, and its `probe`
    # counterpart. `animation_cli.md:79` calls it an "optional fragment
    # check" that reports "advisory MotionContinuity diagnostics" and notes
    # that "a passing fragment is diagnostic; full acceptance still checks
    # the entire clip" — so it measures and changes nothing, which is this
    # lane, while acceptance stays with `finish` in `build`.
    "op:check_motion": "inspect",
    "op:read_file": "inspect",
    "op:dispatch_status": "inspect",
    # The two halves of the background-job polling surface: ask what a job
    # is doing, and read what it returned. Neither changes anything, so both
    # are Inspect beside `dispatch_status`.
    "op:job_status": "inspect",
    "op:job_result": "inspect",
    "op:contribution_status": "inspect",
    "inspect.probe": "inspect",
    "inspect.read_source": "inspect",
    "inspect.read_reference": "inspect",
    "inspect.report": "inspect",
    "inspect.search": "inspect",
    # --- look -----------------------------------------------------------
    "op:view": "look",
    "op:snapshot": "look",
    "op:storyboard": "look",
    "op:accept_storyboard": "look",
    "op:motion_image": "look",
    # `render` draws any frame selection through the native cuts and writes
    # outside the workspace, so looking never moves the revision. It is the
    # cheap draft-film op, and it is Look for the same reason `snapshot` is -- it guarantees nothing
    # and satisfies no part of delivery. `inspect` answers "where is
    # everything at these moments" as numbers rather than an image, which is
    # Inspect beside `probe`. Both postdate this table; the corpus tripwire
    # is what surfaced them.
    "op:render": "look",
    "op:inspect": "inspect",
    "look.read_image": "look",
    # --- solve ----------------------------------------------------------
    "op:ik": "solve",
    "op:dispatch_object": "solve",
    "op:dispatch_scene": "solve",
    "op:dispatch_motion": "solve",
    "op:cancel_contribution": "solve",
    "op:await_any": "solve",
    "op:cancel_dispatch": "solve",
    # `job_cancel` stops a job that is running, so it sits with the other
    # cancels rather than beside `job_status`/`job_result` in Inspect:
    # those read what a job is doing, this one changes it.
    "op:job_cancel": "solve",
    # `blender` runs the agent's own bpy snippet in the Blender interpreter and
    # hands back whatever it printed. That is external compute producing
    # numbers to author from — the Solve blurb exactly — not a render.
    "op:blender": "solve",
    # The same action taken outside the harness. An agent that runs
    # `blender -b project.blend -P script.py` is doing exactly what `op:blender`
    # does -- its own bpy code in the Blender interpreter, handing back numbers.
    # Giving it the same lane is what makes "how much of this run went through
    # the harness" a question the timeline can answer: an agent can drift from
    # harness ops to its own Blender invocations as a run goes on.
    "solve.local_blender": "solve",
    # codex's native sub-agent tools. Not harness ops, so they arrive through
    # `classify_tool` rather than the trace, but they mean the same thing as
    # `ask_subagent`: work handed to another process.
    "solve.subagent": "solve",
    # --- deliver --------------------------------------------------------
    "op:final_render": "deliver",
    "op:record_deviation": "deliver",
    "op:submit_result": "deliver",
    "op:submit_object": "deliver",
    "deliver.terminate": "deliver",
    "deliver.commit": "deliver",
    # --- aside ----------------------------------------------------------
    # Switching renderer is configuration, not work on the scene. It earns a
    # row because the round trip is not free — five scenes asked for Blender,
    # were refused, and switched back (audit C2).
    "op:set_render_backend": "aside",
    "aside.wait": "aside",
    "aside.oob_python": "aside",
    "aside.plumbing": "aside",
    "aside.tool_search": "aside",
    "aside.unknown": "aside",
}


#: Actions with no duration of their own. `record_deviation` is 1236 calls and
#: 0 s across the corpus — it is bookkeeping, so it renders as a tick and must
#: never be given a bar (a 0-width bar is invisible and unclickable).
ZERO_DURATION_ACTIONS = frozenset({
    "op:record_deviation", "deliver.terminate", "deliver.commit",
})

#: Which layer can fill which category, so a run's legend shows no category it
#: could never use. Every layer has `model`; `judge` is a review the author
#: asks for, which `plain` never has.
#: `plain` is the one-off layer: a single `submit_result` and its own CLI tools.
LAYER_CATEGORIES = {
    "object": ("model", "judge", "author", "build", "inspect", "look",
               "deliver", "aside"),
    "scene": ("model", "judge", "author", "build", "inspect", "look",
              "deliver", "aside"),
    "animation": ("model", "judge", "author", "build", "inspect", "look",
                  "solve", "deliver", "aside"),
    "plain": ("model", "author", "inspect", "deliver", "aside"),
}


# ---------------------------------------------------------------------------
# bash / path classification
# ---------------------------------------------------------------------------

_PY_EXE_RE = re.compile(r"(?:^|/)(?:python[\d.]*|pytest)$")

# An out-of-band python one-liner is one of two very different things. Reading
# `compile_reports/latest.json` is the agent inspecting its own grade; importing
# build123d to check a fillet before editing `object.py` is authoring. Lumping
# them misfiled 14% of measured runtime.
_RUN_ARTIFACT_RE = re.compile(
    r"compile_reports|\bir\.json|\bmeta\.json|latest\.json|verifier")
_GEOMETRY_RE = re.compile(
    r"\b(?:from|import)\s+(?:build123d|dycraft|numpy|trimesh|scipy)\b"
    r"|\bBox\(|\bTorus\(|\bCylinder\(|\bbounding_box\b")
#: Modules an artifact *reader* needs and nothing more. An import outside this
#: set means the script is computing something, not printing a report — which
#: is how `import scratch_clearance` (the agent's own geometry helper, 361 s of
#: one run) is told apart from `import json; print(open('ir.json'))`.
_READER_IMPORT_RE = re.compile(
    r"\b(?:from|import)\s+(?!json\b|sys\b|os\b|pathlib\b|pprint\b|re\b"
    r"|textwrap\b|collections\b|itertools\b|math\b|argparse\b|glob\b)"
    r"([A-Za-z_][A-Za-z0-9_]*)")
#: A `.py` argument means the agent is running a file it wrote itself.
_PY_SCRIPT_ARG_RE = re.compile(r"\S+\.py\b")

#: Writing a file from the shell rather than through `Write`/`Edit`. Agents do
#: this constantly for helper scripts — `cat > ik_probe.py <<'PYEOF'` is how
#: every IK experiment in the animation corpus starts — and reading it as a
#: `cat` would file authoring under Inspect.
#: `(?<![0-9&])` so a file-descriptor redirect is not a write target: `cat
#: object.py 2>/dev/null` was reading the `2>` as "writes to /dev/null" and
#: filing 218 corpus reads under Author. `/dev/null` is excluded outright — it
#: is never an authored file.
_SHELL_WRITE_RE = re.compile(
    r"(?:^|[\s|;&])(?:cat|tee)\b[^<>|;&]*(?<![0-9&])>>?\s*"
    r"(?!/dev/null)([^\s|;&<>]+)")
#: `tee FILE` writes without a redirect at all.
_TEE_ARG_RE = re.compile(r"(?:^|[\s|;&])tee\b(?:\s+-\S+)*\s+(?!/dev/null)([^\s|;&<>]+)")
_SED_INPLACE_RE = re.compile(
    r"\bsed\s+-i\S*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+([^\s|;&<>]+)")

#: `h._call("ik", {...})` — the raw transport, used to pass a longer timeout
#: than the stub allows. It is really a call to the named op, and `_detect_op`
#: can only see `_call`.
RAW_CALL_RE = re.compile(r"_call\(\s*[\"']([a-z_]+)[\"']")

_INSPECT_CMDS = frozenset({
    "grep", "find", "ls", "sed", "cat", "head", "tail", "rg", "awk", "wc",
    "diff", "file", "sort", "uniq", "cut", "jq", "stat", "tree", "readlink",
    "basename", "dirname", "nl", "column", "less", "zcat",
    # Reads container metadata -- dimensions, frame count, duration -- and
    # writes nothing. `ffmpeg` is not here: it encodes, so it lands wherever
    # its output path puts it.
    "ffprobe",
    # Digesting files is asking a question about them, and the recorded use is
    # exactly that: an arm ran `md5sum` across `native/storyboard_views` to
    # find which panels had actually changed between storyboard calls -- the
    # question `storyboard()`'s own `retained` answers. Reads, writes nothing.
    "md5sum", "sha1sum", "sha256sum", "cksum",
})
_PLUMBING_CMDS = frozenset({
    "mkdir", "env", "ps", "pgrep", "touch", "echo", "tr", "rm", "cp", "mv",
    "chmod", "export", "which", "kill", "printf", "true", "false", "df", "du",
    "nohup", "set", "unset", "date", "hostname", "id", "pwd", "sync", "ln",
    "tar", "unzip", "curl", "wget", "git", "npm", "node", "pip", "conda",
    "nproc", "free", "top", "nvidia-smi", "xargs", "test",
    # Only reached by `base64 -d`. The *encode* direction is Look and is
    # matched earlier in `classify_bash` — see the comment there. Decoding is
    # transport, and it needs a home here or it lands in `aside.unknown`,
    # which `test_corpus_classifies_completely` asserts is empty.
    "base64",
})
_WAIT_CMDS = frozenset({"sleep", "wait"})
# Words that carry no intent: the `cd $RUN_DIR && …` preamble every harness
# call and most explorations open with, plus shell keywords and operators.
_SHELL_NOISE = frozenset({
    "cd", "&&", "||", "|", ";", "\\", "{", "}", "(", ")", "then", "do", "done",
    "fi", "else", "elif", "if", "for", "while", "until", "in", "time",
    "timeout", "eval", "exec", "source", ".", "!", "sudo", "bash", "sh", "zsh",
})
_TASK_OUTPUT_RE = re.compile(r"/tasks/\w+\.output")

_SOURCE_FILES = frozenset({"object.py", "scene.py", "animation.py"})
_RUN_ARTIFACTS = frozenset({
    "ir.json", "manifest.json", "latest.json", "meta.json", "harness.py",
    "system_prompt.md", "result.json", "scene.json",
})
_IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
_NUMBERED_JSON_RE = re.compile(r"^\d+(_\w+)?\.json$")


def _command_words(command: str) -> tuple[list[str], dict[str, str]]:
    """Meaningful words in a shell command, plus its `VAR=value` assignments.

    A bare `$P` head is resolved against an assignment in the same command
    (`P=/path/to/python; $P -c …`), because that idiom is common enough in the
    corpus to matter and an unresolved `$P` classifies as nothing.
    """
    words: list[str] = []
    assigns: dict[str, str] = {}
    for raw in re.split(r"[\s\n|;&()]+", command):
        token = raw.strip().strip("\"'")
        if not token:
            continue
        assignment = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", token)
        if assignment:
            assigns[assignment.group(1)] = assignment.group(2)
            continue
        words.append(token)
    resolved = []
    for token in words:
        var = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$", token)
        if not var:
            resolved.append(token)
            continue
        name = var.group(1)
        if name in assigns:
            resolved.append(assigns[name])
            continue
        # Exported by the launcher rather than assigned in the command, so
        # there is nothing here to resolve against. The *name* is the
        # evidence: `"$DYCRAFT_PYTHON" - <<'PY'` is a Python heredoc, and
        # leaving it as the bare word `DYCRAFT_PYTHON` would classify a whole
        # idiom as `aside.unknown`.
        resolved.append("python" if name.upper().endswith("PYTHON") else name)
    return resolved, assigns


def classify_bash(command: str) -> str:
    """The action key for a shell command that is *not* a harness call.

    Ordered by specificity, not by frequency: `.terminate` wins over
    everything because `test -s context_summary.md && touch .terminate` is
    otherwise just a `test`, and it is the agent declaring itself done.
    """
    words, assigns = _command_words(command)
    names = [os.path.basename(w) for w in words]
    haystack = command + " " + " ".join(assigns.values())

    if ".terminate" in command:
        return "deliver.terminate"
    if (any(_PY_EXE_RE.search(n) or "PYTHON" in n.upper() for n in names)
            or re.search(r"\bpython[\d.]*\s+-c|\bpytest\b", command)):
        # Runs its own file, or imports something a report reader would not:
        # an experiment. `harness.py` is excluded because `-c 'import harness'`
        # is the harness call itself, handled by the caller.
        scripts = [s for s in _PY_SCRIPT_ARG_RE.findall(command)
                   if os.path.basename(s) != "harness.py"]
        if (_GEOMETRY_RE.search(command) or scripts
                or _READER_IMPORT_RE.search(command)):
            return "author.experiment"
        if _RUN_ARTIFACT_RE.search(command):
            return "inspect.report"
        return "aside.oob_python"

    # Beside the python branch, and before the write branch, for the same
    # reason it gives: `blender -b scene.blend -P probe.py > diag.log` redirects
    # to a log but is compute, not the authoring of a log. Gated on a Blender
    # flag so a path that merely ends in `blender` cannot trip it.
    if "blender" in names and re.search(r"(?:^|\s)-[bP]\b|--background\b|--python\b", command):
        return "solve.local_blender"

    # After the python branch on purpose: `$PY -u scratch.py > scratch.log`
    # redirects to a log but is an experiment, not the authoring of a log.
    write_target = (_SHELL_WRITE_RE.search(command)
                    or _TEE_ARG_RE.search(command)
                    or _SED_INPLACE_RE.search(command))
    if write_target:
        return classify_file("write", write_target.group(1))

    if any(n in _WAIT_CMDS for n in names) or _TASK_OUTPUT_RE.search(haystack):
        return "aside.wait"

    # `base64 -w0 <render>` is how a review sub-agent gets pixels into a VLM
    # prompt: there is no image tool on that path, so it encodes the PNG on
    # stdout and pastes it. That is the agent choosing to look at its own
    # render, so it belongs in Look with `Read <image>` rather than in Aside
    # with the plumbing — 35 events in a recorded corpus, every one of them a
    # render or a crop of one.
    #
    # Not gated on an image extension, because the codex `exec` transcript
    # records the JavaScript wrapper's unexpanded form (`base64 -w0 '${p}'`,
    # and in two cases no argument at all), so most of these carry no readable
    # path. Decoding is the one direction that is not looking.
    if "base64" in names and not re.search(r"\bbase64\b.*\s-{1,2}(?:d|decode)\b",
                                           command):
        return "look.read_image"

    real = [n for n in names if n not in _SHELL_NOISE and not n.startswith("-")]
    for name in real:
        if name in _INSPECT_CMDS:
            return "inspect.search"
    for name in real:
        if name in _PLUMBING_CMDS:
            return "aside.plumbing"
    # A command of pure noise (`cd $RUN_DIR && :`) did something, just nothing
    # nameable. Plumbing rather than unknown, which is reserved for tools.
    return "aside.plumbing" if not real else "aside.unknown"


def classify_file(tool: str, path: Any) -> str:
    """The action key for a file-touching tool call."""
    base = os.path.basename(str(path or ""))
    ext = os.path.splitext(base)[1].lower()
    if tool == "read":
        if ext in _IMAGE_EXT:
            return "look.read_image"
        if base.endswith(".output"):
            return "aside.wait"
        if (base in _SOURCE_FILES or base in _RUN_ARTIFACTS
                or _NUMBERED_JSON_RE.match(base)):
            return "inspect.read_source"
        return "inspect.read_reference"
    if base == "context_summary.md":
        return "author.summary"
    if base in _SOURCE_FILES:
        return "author.source"
    return "author.aux"


def classify_tool(name: str | None, tool_input: Any) -> str:
    """The action key for one agent tool call.

    Harness ops are *not* routed here — they come off `trace.jsonl` with an
    authoritative duration. This is only the tools the harness never sees.
    """
    args = tool_input if isinstance(tool_input, dict) else {}
    if not name:
        return "aside.unknown"
    if name == "Bash":
        return classify_bash(args.get("command") or "")
    if name == "exec":
        # codex's one general-purpose tool. `viewer/trajectory` splits an `exec`
        # call into one entry per shell it actually spawned and puts that
        # shell's argv under `command`, so it classifies exactly like `Bash`.
        # A call that spawned none carries codex's JavaScript snippet instead,
        # which `classify_bash` cannot read — those are the tool-list and
        # file-read queries, and `aside.plumbing` is the honest lane for them.
        command = args.get("command")
        return classify_bash(command) if command else "aside.plumbing"
    if name == "Read":
        return classify_file("read", args.get("file_path") or args.get("path"))
    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return classify_file("write", args.get("file_path") or args.get("path"))
    if name == "Skill":
        skill = str(args.get("skill") or args.get("command") or "")
        # `save_context` writes context_summary.md; the reference skills are
        # read-only documentation.
        return ("author.summary" if "save_context" in skill
                else "inspect.read_reference")
    if name in ("TaskOutput", "Monitor", "Task", "Agent", "BashOutput",
                "KillShell"):
        return "aside.wait"
    # Asking the operator a question and waiting for the answer. Named
    # separately from the group above only because it is the tool this test's
    # tripwire actually caught: it did not exist when the viewer was written,
    # surfaced as `aside.unknown` on a live animation run, and the docstring
    # of `test_corpus_classifies_completely` asks for exactly this response
    # rather than a silent reclassification. Wall time here is a human
    # deciding, which is outside the graded loop by definition.
    if name in ("AskUserQuestion", "request_user_input",
                "request_user_input_async"):
        return "aside.wait"
    # codex's own tool names are lowercase and arrive here unchanged. `sleep`
    # and `wait` are its two yield primitives — 54 events in a recorded corpus,
    # and they are waiting, exactly as the equivalent `Bash sleep` is.
    if name in ("sleep", "wait"):
        return "aside.wait"
    # `spawn_agent` starts a codex sub-agent and `followup_task` sends it more
    # work. Both are work handed to another process, like `ask_subagent`.
    if name in ("spawn_agent", "followup_task"):
        return "solve.subagent"
    if name == "ToolSearch":
        return "aside.tool_search"
    if name in ("Glob", "Grep", "WebFetch", "WebSearch"):
        return "inspect.search"
    if name == "TodoWrite":
        return "aside.plumbing"
    return "aside.unknown"


# ---------------------------------------------------------------------------
# trace reading
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict]:
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _epoch(value: Any) -> float | None:
    """Seconds since the epoch, from an ISO-8601 string or a number."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


#: Lifecycle events worth an axis mark, and whether each reads as a failure.
#: Value is (label, is_failure).
LIFECYCLE_MARKS = {
    "daemon_start": ("daemon start", False),
    "daemon_shutdown": ("daemon shutdown", False),
    "catalog_loaded": ("catalog loaded", False),
    "scene_loaded": ("scene loaded", False),
    "commit": ("commit", False),
    "cad_worker_unavailable": ("no CAD interpreter", True),
}


def _read_trace(run_dir: Path) -> dict:
    """Op spans, compile detail, and lifecycle marks from `trace.jsonl`.

    `op_call` and `op_result` pair on `step`, which is nearly unique — `tick()`
    issues one step per op, so the concurrent ops the daemon's thread pool
    allows cannot collide *within one session*. They do collide **across**
    sessions: a resumed run starts a second daemon and the counter restarts
    from 1, so a resumed run can have two different `op_result`s for one
    step. So the map is
    step -> FIFO queue, and a step is consumed in arrival order rather than
    overwritten, which would have silently dropped the first compile's start.
    """
    calls: dict[Any, list[dict]] = {}
    spans: list[dict] = []
    marks: list[dict] = []
    compiles: dict[Any, list[dict]] = {}

    for rec in _trace.events(run_dir / "trace.jsonl", _TRACE_EVENTS, args=_LABEL_ARGS):
        ev = rec.get("ev")
        ts = _epoch(rec.get("ts"))
        if ev == "op_call":
            calls.setdefault(rec.get("step"), []).append({
                "op": rec.get("op"), "step": rec.get("step"),
                "start": ts, "args": rec.get("args"),
            })
        elif ev == "op_result":
            queue = calls.get(rec.get("step")) or []
            call = queue.pop(0) if queue else None
            elapsed = rec.get("elapsed_ms")
            elapsed = float(elapsed) if isinstance(elapsed, (int, float)) else None
            # Prefer the op's own elapsed_ms over (end - start): the two agree,
            # but elapsed_ms is measured around the handler while the two
            # timestamps are written on either side of it.
            start = (call or {}).get("start")
            if start is None and ts is not None and elapsed is not None:
                start = ts - elapsed / 1000.0
            spans.append({
                "op": rec.get("op"),
                "step": rec.get("step"),
                "start": start,
                "end": ts,
                "elapsed_ms": elapsed,
                "ok": rec.get("ok"),
                "cache_hit": rec.get("cache_hit"),
                "args": (call or {}).get("args"),
            })
        elif ev in ("compile_attempt", "finish_attempt"):
            compiles.setdefault(rec.get("step"), []).append(
                dict(rec, _at=ts))
        elif ev in LIFECYCLE_MARKS:
            label, failure = LIFECYCLE_MARKS[ev]
            detail = {k: v for k, v in rec.items()
                      if k not in ("ev", "ts", "step")}
            marks.append({"kind": ev, "at": _iso(ts), "label": label,
                          "failure": failure, "step": rec.get("step"),
                          "detail": detail or None})

    # An op whose result was never written: the daemon was killed mid-call.
    # Keep it — it is the last thing the run did, which is exactly what you
    # want to see — but with no end, so the layout draws it as unknown.
    for queue in calls.values():
        for call in queue:
            spans.append({
                "op": call["op"], "step": call["step"], "start": call["start"],
                "end": None, "elapsed_ms": None, "ok": None,
                "cache_hit": None, "args": call.get("args"),
            })

    spans.sort(key=lambda s: (s["start"] is None, s["start"] or 0))
    return {"spans": spans, "marks": marks, "compiles": compiles}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _short(ms: float) -> str:
    """A duration at the precision a label can carry."""
    if ms < 1000:
        return f"{round(ms)}ms"
    if ms < 90_000:
        return f"{ms / 1000:.0f}s"
    if ms < 5_400_000:
        return f"{ms / 60_000:.0f}m"
    return f"{ms / 3_600_000:.1f}h"


def model_phases(began_at: Any, records: list[dict]) -> list[dict]:
    """A turn's model time, split into `think` / `text` / `write` stretches.

    Measured, not apportioned: each stretch runs from the previous record to
    the record that closed the block. `write` is the model composing a tool
    call's input — on an object run that is 31% of all model time (29.7 h), and
    for a `Write object.py` turn it is the authoring itself. Bundling it with
    thinking loses the single largest thing the timeline could tell you about
    where a run's effort went.

    Empty on the `api` path, which writes one record per response and so has no
    internal boundaries to read.
    """
    start = _epoch(began_at)
    if start is None or not records:
        return []
    phases: list[dict] = []
    cursor = start
    for record in records:
        at = _epoch(record.get("at"))
        if at is None or at < cursor:
            continue
        kinds = record.get("kinds") or []
        # A record carrying several blocks cannot be split further, so it is
        # named by the most specific thing in it: composing a tool call is the
        # claim worth making, thinking next.
        kind = ("write" if any(k in ("tool_use", "tool_call") for k in kinds)
                else "think" if any(k in ("thinking", "redacted_thinking")
                                    for k in kinds)
                else "text" if "text" in kinds
                else None)
        if kind is not None and at > cursor:
            phases.append({"kind": kind,
                           "offset_ms": round((cursor - start) * 1000.0, 1),
                           "duration_ms": round((at - cursor) * 1000.0, 1)})
        cursor = max(cursor, at)
    return phases


def _attribute_turns(events: list[dict], turns: list[dict]) -> None:
    """Give every trace-sourced event the turn number that issued it, in place.

    Without it a trace row would only look nested under the model turn above it, and the second half of a long
    transcript would read as scattered harness work beside the conversation. The harness cannot stamp the number: it is
    a socket server, and the agent is a separate CLI process that never says which turn it is on. So the join is made
    here from the one fact both sides share: a harness op runs inside the shell command that called it, a tool call
    with a start and an end.

    Containment, not nearest-neighbor, because nearest-neighbor would attribute every row to something and make a
    guess indistinguishable from a join; a row no call contains keeps `turn: None`. The few rows outside every call are
    ops that outlived the call that issued them (a Codex cell's window closes when Codex writes the shell's output
    record, and the op can run on for minutes), and attributing those by proximity would be silently wrong for a
    genuine background dispatch.

    The tool-call window is preferred over the turn window because it is tighter and calls in one turn can run
    concurrently. The turn window is the fallback for a transcript whose calls carry no end, which matters on the
    Claude path only: the Codex path records no `ended_at` per turn.
    """
    calls: list[tuple[float, float, int]] = []
    spans: list[tuple[float, float, int]] = []
    for turn in turns:
        index = turn.get("index")
        if index is None:
            continue
        for call in (turn.get("tool_calls") or ()):
            start = _epoch(call.get("started_at"))
            end = _epoch(call.get("finished_at"))
            if start is None:
                continue
            calls.append((start, end if end is not None else start, index))
        lo = _epoch(turn.get("began_at")) or _epoch(turn.get("ts"))
        hi = _epoch(turn.get("ended_at"))
        if lo is not None:
            spans.append((lo, hi if hi is not None else lo, index))
    if not calls and not spans:
        return

    calls.sort()
    spans.sort()

    def owner(at: float) -> int | None:
        # A call whose end is unknown was stored as a zero-width window, so it
        # can only claim an op that started in the same instant. Widest
        # containing window wins ties, since a call that ran longer is the one
        # that was still open.
        best: tuple[float, int] | None = None
        for table in (calls, spans):
            for start, end, index in table:
                if start <= at <= end:
                    width = end - start
                    if best is None or width > best[0]:
                        best = (width, index)
            if best is not None:
                return best[1]
        return None

    for event in events:
        if event.get("source") != "trace" or event.get("turn") is not None:
            continue
        at = _epoch(event.get("started_at"))
        if at is None:
            continue
        event["turn"] = owner(at)


#: The ops whose labels read their arguments (`_op_label`); every other op's arguments are left unread.
#: The argument fields `_op_label` reads, per op: all the timeline keeps of an op's arguments.
_LABEL_ARGS = {"probe": ("expression",), "view": ("target",), "compile": ("freeze",),
               "final_render": ("failed",), "record_deviation": ("severity",)}
_TRACE_EVENTS = frozenset({"op_call", "op_result", "compile_attempt", "finish_attempt", *LIFECYCLE_MARKS})
_trace.register(_TRACE_EVENTS, _LABEL_ARGS)


def _op_label(op: str | None, args: Any) -> str:
    """A short label for an op event — the one detail worth reading inline."""
    if op == "probe" and isinstance(args, dict):
        expression = str(args.get("expression") or "").strip()
        head = re.match(r"[A-Za-z_][A-Za-z0-9_.]*", expression)
        return f"probe {head.group(0)}" if head else "probe"
    if op == "view" and isinstance(args, dict):
        return f"view {args.get('target') or 'all'}"
    if op == "compile" and isinstance(args, dict) and args.get("freeze"):
        return "compile (freeze)"
    if op == "final_render" and isinstance(args, dict) and args.get("failed"):
        return "final_render (failed path)"
    if op == "record_deviation" and isinstance(args, dict):
        severity = args.get("severity")
        return f"deviation ({severity})" if severity else "deviation"
    return op or "op"


def _status(*, ok: Any, has_end: bool, run_finished: bool) -> str:
    if not has_end:
        # A bar running to `now` is honest only while the run itself is going.
        # Once it has finished, nobody recorded a result and the duration is
        # unknown; drawing it to `now` would grow every time the page opened.
        return "unknown" if run_finished else "running"
    if ok is None:
        return "ok"
    return "ok" if ok else "failed"


def _attempt_for(attempts: list[dict] | None, span: dict) -> dict | None:
    """The `compile_attempt` emitted inside this span.

    Matched by containment, not by position: `step` repeats across daemon
    sessions in a resumed run, so a step can hold several attempts and taking
    the first would describe the wrong compile. The attempt is traced from
    inside the op, so its timestamp always falls in the op's own window.
    """
    if not attempts:
        return None
    if len(attempts) == 1:
        return attempts[0]
    start, end = span.get("start"), span.get("end")
    if start is not None and end is not None:
        for attempt in attempts:
            at = attempt.get("_at")
            if at is not None and start <= at <= end:
                return attempt
    return attempts[0]


def _compile_detail(rec: dict | None) -> dict | None:
    if not rec:
        return None
    verifiers = rec.get("verifier_summary") or []
    failed = [v.get("name") for v in verifiers if v.get("verdict") == "fail"]
    return {
        "attempt": rec.get("attempt"),
        "is_finish": rec.get("is_finish"),
        "build_error": rec.get("build_error"),
        "n_verifiers": len(verifiers),
        "n_failed": len(failed),
        "failed_verifiers": failed,
        "verifiers": verifiers,
    }


def build(run_dir: Path, *, now: float | None = None) -> dict:
    """The lane timeline for one agent run.

    `now` is injectable so the running/unknown split is testable without
    freezing the clock.
    """
    run_dir = Path(run_dir)
    now = now if now is not None else datetime.now(timezone.utc).timestamp()

    meta: dict = {}
    try:
        with open(run_dir / "meta.json", "r", errors="replace") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        meta = {}
    layer = canonical_layer(meta.get("layer")) or "unknown"
    outcome = meta.get("outcome")
    run_finished = bool(meta.get("finished_at")) or bool(outcome)

    trace = _read_trace(run_dir)
    notes: list[str] = []
    # Steps the transcript shows called but the harness log has no result for, by name in order of first call: one
    # note for them all, not one each.
    unrecorded: list[str] = []

    events: list[dict] = []
    seen_steps: set[Any] = set()

    def emit(**kw) -> None:
        kw.setdefault("step", None)
        kw.setdefault("turn", None)
        kw.setdefault("cache_hit", None)
        kw.setdefault("zero_duration", False)
        kw.setdefault("detail", {})
        events.append(kw)

    # --- harness ops, from the trace ------------------------------------
    for span in trace["spans"]:
        action = f"op:{span['op']}"
        category = ACTION_CATEGORY.get(action)
        if category is None:
            category = "aside"
            action = "aside.unknown"
            notes.append(
                f"{span['op']} is not a step this viewer knows, so it is filed under Aside.")
        seen_steps.add(span["step"])
        elapsed = span["elapsed_ms"]
        zero = action in ZERO_DURATION_ACTIONS
        start_ts = span["start"]
        status = _status(ok=span["ok"], has_end=span["end"] is not None,
                         run_finished=run_finished)
        label = _op_label(span["op"], span["args"])
        # The op's arguments stay out of `detail`: the label already summarizes
        # them, and one probe or edit can carry megabytes of code or data, which
        # on a long run makes the response too large for a browser to parse.
        detail = {
            "op": span["op"],
            "compile": _compile_detail(_attempt_for(
                trace["compiles"].get(span["step"]), span)),
        }

        if zero or start_ts is None or elapsed is None:
            emit(id=f"op-{span['step']}-{span['op']}", category=category,
                 action=action, label=label, step=span["step"],
                 started_at=_iso(start_ts), finished_at=None,
                 duration_ms=None, status=status,
                 cache_hit=span["cache_hit"], zero_duration=zero,
                 source="trace", detail=detail)
            continue

        emit(id=f"op-{span['step']}-{span['op']}",
             category=category, action=action, label=label,
             step=span["step"], started_at=_iso(start_ts),
             finished_at=_iso(start_ts + elapsed / 1000.0),
             duration_ms=round(elapsed, 1), status=status,
             cache_hit=span["cache_hit"], source="trace", detail=detail)

    # --- everything the harness never saw, from the transcript ----------
    trajectory = build_trajectory(run_dir)
    turns = trajectory.get("turns") or []
    for turn in turns:
        tool_calls = turn.get("tool_calls") or []
        # The model's window ends where its first tool call goes out, so the
        # two abut instead of the turn appearing to contain its own calls.
        first_call = min(
            (c["started_at"] for c in tool_calls if c.get("started_at")),
            default=None)
        model_end = first_call or turn.get("ended_at") or turn.get("ts")
        # From when the model was unblocked, not from its first emitted block:
        # `ts` is the first block's record, so starting there drops the
        # prefill and the bulk of the generation (1555 s of an object run
        # became 196 s).
        model_start = turn.get("began_at") or turn.get("ts")
        # Clamp rather than trust: transcript timestamps are not guaranteed
        # monotonic across interleaved writers, and a segment whose end
        # precedes its start breaks the layout outright.
        gen_ms = None
        a, b = _epoch(model_start), _epoch(model_end)
        if a is not None and b is not None:
            if b < a:
                model_start = model_end
                a = b
            gen_ms = round((b - a) * 1000.0, 1)
        if gen_ms is None:
            fallback = turn.get("generation_ms")
            gen_ms = (max(0.0, float(fallback))
                      if isinstance(fallback, (int, float)) else None)
        phases = [p for p in model_phases(model_start, turn.get("records"))
                  if p["offset_ms"] + p["duration_ms"] <= (gen_ms or 0) + 1]
        by_kind: dict[str, float] = {}
        for phase in phases:
            by_kind[phase["kind"]] = (by_kind.get(phase["kind"], 0.0)
                                      + phase["duration_ms"])
        # The label leads with the split, because that is the part of a run
        # nothing else in the bundle records. "thinking" covers text too: it is
        # 2% of model time and not worth a third word.
        parts = []
        if by_kind.get("think") or by_kind.get("text"):
            parts.append(_short(by_kind.get("think", 0.0)
                                + by_kind.get("text", 0.0)) + " thinking")
        if by_kind.get("write"):
            parts.append(_short(by_kind["write"]) + " writing")
        if not parts:
            parts.append(f"{len(tool_calls)} tool call"
                         f"{'' if len(tool_calls) == 1 else 's'}"
                         if tool_calls else "no output")
        emit(id=f"model-{turn['index']}", category="model", action="model.turn",
             label=f"turn {turn['index']} · " + " + ".join(parts),
             turn=turn["index"], started_at=model_start, finished_at=model_end,
             duration_ms=gen_ms, status="ok", source="transcript",
             detail={"model": turn.get("model"), "effort": turn.get("effort"),
                     "stop_reason": turn.get("stop_reason"),
                     "cost_usd": turn.get("cost_usd"),
                     "thought": bool(turn.get("thought_summary")),
                     "text": bool(turn.get("text")),
                     "n_tool_calls": len(tool_calls),
                     "phases": phases,
                     "think_ms": round(by_kind.get("think", 0.0), 1),
                     "text_ms": round(by_kind.get("text", 0.0), 1),
                     "write_ms": round(by_kind.get("write", 0.0), 1)})

        for index, call in enumerate(tool_calls):
            if (call.get("dycraft_step") is not None
                    and call["dycraft_step"] in seen_steps):
                continue
            op = call.get("dycraft_op")
            if op == "_call":
                args = call.get("input")
                command = args.get("command") if isinstance(args, dict) else ""
                inner = RAW_CALL_RE.search(str(command or ""))
                op = inner.group(1) if inner else None
            if op:
                action = f"op:{op}"
                if action not in ACTION_CATEGORY:
                    action = "aside.unknown"
                if op not in unrecorded:
                    unrecorded.append(op)
            else:
                action = classify_tool(call.get("name"), call.get("input"))
            zero = action in ZERO_DURATION_ACTIONS
            emit(id=f"turn-{turn['index']}-{index}",
                 category=ACTION_CATEGORY.get(action, "aside"), action=action,
                 label=_tool_label(call, action), step=call.get("dycraft_step"),
                 turn=turn["index"], started_at=call.get("started_at"),
                 finished_at=None if zero else call.get("finished_at"),
                 duration_ms=None if zero else call.get("duration_ms"),
                 status=("failed" if call.get("is_error")
                         else _status(ok=True,
                                      has_end=call.get("finished_at") is not None,
                                      run_finished=run_finished)),
                 zero_duration=zero, source="transcript",
                 detail={"tool": call.get("name"), "dycraft_op": call.get("dycraft_op")})

    _attribute_turns(events, turns)

    events.sort(key=lambda e: (e["started_at"] is None, e["started_at"] or "",
                               e["id"]))
    for order, event in enumerate(events):
        event["order"] = order

    # --- categories actually offered ------------------------------------
    allowed = LAYER_CATEGORIES.get(layer)
    filled = {e["category"] for e in events}
    keys = [k for k in CATEGORY_ORDER
            if (k in allowed if allowed else True) or k in filled]
    totals = _totals(events)
    categories = [{"key": k, "label": CATEGORY_LABEL[k],
                   "blurb": CATEGORY_BLURB[k],
                   "n_events": totals["categories"][k]["n"],
                   "ms": totals["categories"][k]["ms"]}
                  for k in keys]

    if unrecorded:
        notes.append(_unrecorded_note(unrecorded))
    notes.extend(_notes(trajectory, events))

    started, finished = _span_of(meta, events)
    return {
        "run_id": meta.get("run_id"),
        "layer": layer,
        "outcome": outcome,
        "started_at": started,
        "finished_at": finished,
        "run_finished": run_finished,
        "now": _iso(now),
        "categories": categories,
        "events": events,
        "marks": trace["marks"],
        "totals": totals,
        # Deduped, order preserved: several events can raise the same note.
        "notes": list(dict.fromkeys(notes)),
    }


def _tool_label(call: dict, action: str) -> str:
    """A label for a non-harness tool call, from the part that identifies it."""
    name = call.get("name") or "tool"
    args = call.get("input") if isinstance(call.get("input"), dict) else {}
    if name in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit"):
        base = os.path.basename(str(args.get("file_path")
                                    or args.get("path") or ""))
        return f"{name} {base}" if base else name
    if name == "Skill":
        return f"Skill {args.get('skill') or args.get('command') or ''}".strip()
    if name in ("Bash", "exec"):
        command = str(args.get("command") or "").strip().split("\n")[0]
        if action == "deliver.terminate":
            return "declare done (.terminate)"
        return command[:56] or name
    return name


def _span_of(meta: dict, events: list[dict]) -> tuple[str | None, str | None]:
    """The run's own window, widened to hold anything recorded outside it.

    `meta.json` is the authority, but it is written at launch and finalize, and
    a bundle can be missing either — so the observed stamps are a floor.
    """
    stamps = [e["started_at"] for e in events if e["started_at"]]
    stamps += [e["finished_at"] for e in events if e["finished_at"]]
    lo = ([meta["started_at"]] if meta.get("started_at") else []) + stamps
    hi = ([meta["finished_at"]] if meta.get("finished_at") else []) + stamps
    return (min(lo) if lo else None), (max(hi) if hi else None)


def _totals(events: list[dict]) -> dict:
    """Per-category wall-clock and count, plus two denominators.

    `action_ms` excludes `model` and `judge`, the LLM time (the agent's own
    model and the review sub-agent's grading), because that is the only
    denominator on which the agent's own actions are comparable: at whole-run
    scale LLM time is most of a run and everything else is a sliver. `all_ms`
    is the sum of every segment.
    """
    per_ms: dict[str, float] = {}
    per_n: dict[str, int] = {}
    for event in events:
        key = event["category"]
        per_n[key] = per_n.get(key, 0) + 1
        duration = event.get("duration_ms")
        if isinstance(duration, (int, float)):
            per_ms[key] = per_ms.get(key, 0.0) + duration
    all_ms = sum(per_ms.values())
    action_ms = sum(v for k, v in per_ms.items() if k not in ("model", "judge"))
    return {
        "categories": {k: {"n": per_n.get(k, 0),
                           "ms": round(per_ms.get(k, 0.0), 1)}
                       for k in CATEGORY_ORDER},
        "action_ms": round(action_ms, 1),
        "model_ms": round(per_ms.get("model", 0.0), 1),
        "judge_ms": round(per_ms.get("judge", 0.0), 1),
        "model_n": per_n.get("model", 0),
        "judge_n": per_n.get("judge", 0),
        "all_ms": round(all_ms, 1),
        "n_events": len(events),
    }


#: The harness tool timeout. A cluster of durations at exactly this value is a
#: censored distribution, not a real one, and the panel should say so.
_TOOL_TIMEOUT_MS = 600_000


def _unrecorded_note(ops: list[str]) -> str:
    """One note for every kind of step the transcript shows called but the harness log has no result for."""
    if len(ops) == 1:
        return (f"1 kind of step ({ops[0]}) has no recorded result, so its times come from the agent's transcript and "
                "whether it worked is unknown.")
    names = (f"{ops[0]} and {ops[1]}" if len(ops) == 2 else f"{ops[0]}, {ops[1]}, and {ops[2]}" if len(ops) == 3
             else f"{', '.join(ops[:3])}, and {len(ops) - 3} more")
    return (f"{len(ops)} kinds of step ({names}) have no recorded result, so their times come from the agent's "
            "transcript and whether they worked is unknown.")


def _notes(trajectory: dict, events: list[dict]) -> list[str]:
    notes: list[str] = []
    if not trajectory.get("turns"):
        notes.append(
            trajectory.get("reasoning_note")
            or "No agent transcript in this bundle, so only harness ops "
               "appear — Model, Author, Inspect, and Aside will look emptier "
               "than the run really was.")
    elif not trajectory.get("reasoning_available"):
        notes.append(trajectory.get("reasoning_note") or "")
    if any(e["action"] == "aside.unknown" for e in events):
        notes.append(
            "Some actions could not be classified and are filed under Aside; "
            "they are labeled unknown in the tooltip.")
    unknown = sum(1 for e in events if e["status"] == "unknown")
    if unknown:
        notes.append(
            "1 step never recorded an end before the run finished, so how long it took is unknown; it is drawn as a "
            "thin tick." if unknown == 1 else
            f"{unknown} steps never recorded an end before the run finished, so how long they took is unknown; they "
            "are drawn as thin ticks.")
    censored = sum(1 for e in events
                   if isinstance(e.get("duration_ms"), (int, float))
                   and e["duration_ms"] >= _TOOL_TIMEOUT_MS - 5_000)
    if censored:
        limit = f"{_TOOL_TIMEOUT_MS // 60_000}-minute tool time limit"
        notes.append(
            f"1 call hit the {limit}, so it took at least that long." if censored == 1 else
            f"{censored} calls hit the {limit}, so they took at least that long.")
    if any(e["source"] == "transcript" and e["duration_ms"] is not None
           for e in events):
        notes.append(
            "Times for the agent's own tools are read from the transcript, so they include some overhead, and calls "
            "made together can overlap.")
    return [n for n in notes if n]
