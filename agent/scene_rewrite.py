"""The per-brief scene entry, generated inside the run by `dycraft.cli`.

**Where this runs.** `cli.py` calls `rewrite()` before the daemon starts, for a
fresh `animation` authoring run that was not handed a `--scene-brief`
file, and writes the result into the run as `scene.txt`. The prompt is given
`POINTER`, one sentence naming that file. The cost -- one sub-agent call of a
few minutes, reading two reference files from its own directory -- is paid out
of that run's own budget.

**What it supplies, and why the animation contract cannot.** The contract is one
document used across every brief, so it can require *extent* -- build the whole
place, not only the arrangements the action uses -- but it cannot supply what is
specific to one. An author working the contract alone builds fewer objects and
**nothing outside the room shell**: the other activities the place supports,
whatever lies past each boundary, and the objects earlier use has left are
missing. Those are per-brief by nature; `agent/prompts/scene_rewriter.md` is
pointed at exactly them, and at one more thing of the same kind: how each
object is made, since most authored parts are otherwise bare primitives
although the contract already says a one-primitive part reads as one. The
prompt walks the objects through the same default-then-missing step as the
place. How the performance is carried out is
left to `motion_rewrite.py`, which runs next and reads this entry.

**A known consequence.** Only runs that go through `dycraft.cli` get an entry.
In a paired experiment the bare arms do not, so the two halves read different
tasks; that is a deliberate choice, not an oversight. Each DyCraft arm also
generates its own, so arms in one experiment read different places. Use
`--scene-brief FILE` to hand every run the same frozen entry when a comparison
needs one, or `--no-scene-brief` to run without.

**When it fails.** `rewrite()` raises, and `cli.py --on-rewrite-failure`
decides: `ask` (the default) asks y/N on a terminal and stops without one,
`abort` stops before the agent starts, and `continue` proceeds without an
entry, the `--no-scene-brief` path. See `cli._rewrite_failed`.
"""
from __future__ import annotations

import datetime
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REWRITER_PROMPT = REPO / "agent" / "prompts" / "scene_rewriter.md"
CONTRACT = REPO / "agent" / "prompts" / "animation.md"
#: The SDK reference the animation author is given, as `source_reference.md`
#: (`animation/session.py`). The rewriters write for that author, so they read
#: the same file rather than the scene agent's prompt.
SOURCE_REFERENCE = REPO / "agent" / "prompts" / "animation_reference.md"
#: The reference's asset-library section. The rewriters get the library from
#: `library_block`, so the section is dropped, as the author's copy drops it
#: when the run has no library.
_ASSET_LIBRARY_SECTION = re.compile(r"(?ms)^## Asset library\n.*?(?=^## )")


def source_reference() -> str:
    """`animation_reference.md` without its asset-library section."""
    return _ASSET_LIBRARY_SECTION.sub("", SOURCE_REFERENCE.read_text())

#: An explicit override, and nothing else -- a bare CLI model id such as
#: `gpt-6-astra`, which is what `--model` wants; not a provider scheme.
#: The rewriter otherwise runs on the model the agent itself runs on, on the
#: same account, because `cli.py` hands it that run's own `--backend`,
#: `--model` and `--reasoning-effort`.
#:
#: There is deliberately no hardcoded fallback. A default nobody chose is worse
#: than no default: one the backend cannot route kills the run at startup,
#: before its daemon exists, while an unsupplied model is a readable error at
#: the call site.
MODEL_OVERRIDE = os.environ.get("DYCRAFT_SCENE_REWRITER_MODEL", "")

#: How long the sub-agent gets. A rewrite takes several minutes at medium
#: effort, so this is roughly 2x headroom; on expiry `run_oneoff` SIGTERMs the
#: sub-CLI. It is wall clock spent inside the arm's own budget, which is the
#: argument against making it generous: a wedge at 30 minutes is a fifth of a
#: 2.5h run.
#:
#: At xhigh effort a rewrite can take longer than the bound. It is deliberately
#: NOT raised for that, since it is real budget; instead the sub-agent is told
#: what the bound is (`deadline_note`, so it can submit something rather than
#: be killed holding everything), and a miss can be survived (`cli.py
#: --on-rewrite-failure continue` proceeds without the entry).
DEFAULT_TIMEOUT_S = 900.0

#: What `deadline_note` holds back from the bound for the submission itself,
#: as a fraction and a floor. The entry is 1,600-2,300 words, so writing and
#: submitting it is one turn; what the reserve actually buys is the room to
#: notice the deadline and stop. A quarter of 900 s is 225 s, a little under
#: the longest single reasoning turn a rewriter takes (about five minutes), so
#: the floor matters for short bounds and the fraction for long ones.
SUBMIT_RESERVE_FRACTION = 0.25
SUBMIT_RESERVE_MIN_S = 120.0

#: Appended to the task text, because the rewriter prompt is written for a
#: chat and ends by asking for "the entry and nothing else", while the plain
#: layer discards chat replies and reads only `submit_result`. Both are
#: right about their own channel and the sub-agent sees both, so this says
#: which one carries the answer. Kept here rather than in
#: `scene_rewriter.md` so that file stays a spec of what to write and
#: survives a change of transport.
SUBMIT_INSTRUCTION = (
    "\n\n---\n\n"
    "Deliver the entry by calling `submit_result` with the entry itself as "
    "its `text`, exactly once, and nothing else in the call. The entry is "
    "the whole of the answer: no title, no preamble, no commentary. A chat "
    "reply is not read by anyone and does not end this task. If you draft "
    "the entry in a file, draft it in the directory you are working in, "
    "never under `/tmp` or any other shared path: other runs on this machine "
    "are writing their own entries at the same time."
)
# The last sentence is load-bearing. Codex's workspace-write sandbox leaves
# `/tmp` writable (`cli/backends/codex.py` keeps that default on purpose), and
# rewriters tend to draft at a fixed name such as `/tmp/motion_entry.txt`; with
# several running at once, one can silently submit another brief's entry as its
# own. Batches launch their arms together, so this is the production case, not
# a test artifact.


#: The sentence the brief carries instead of the entry. Both arm kinds find the
#: file in the directory they work in, so one wording serves both.
POINTER = (
    "The place this happens in is described in `scene.txt`, in the same "
    "directory you are working in. Read it before planning: it specifies the "
    "setting, its contents and what surrounds it, and it is part of this task."
)

#: Written into the run as this name; the contract names it at §Workspace.
STAGED_NAME = "scene.txt"

#: The sub-agent's own directory under `<run>/sub_runs/`, as
#: `motion_rewrite.SUB_RUN` is. Named once so `rewrite()`, `cli.py`'s warning
#: and anything auditing a failed rewrite all point at the same place.
SUB_RUN = "scene_rewrite"


def deadline_note(timeout_s: float, *, started: float | None = None) -> str:
    """The wall-clock bound, as the sub-agent needs to hear it. Both rewriters.

    Appended after `SUBMIT_INSTRUCTION` so it is the last thing read, because
    it is the operational constraint on everything above it. Kept here rather
    than in the two `*_rewriter.md` specs for the reason
    `SUBMIT_INSTRUCTION` is: the bound belongs to the transport, not to what
    an entry should say, and the specs survive a change of either.

    WHAT THIS IS FOR. `run_oneoff` SIGTERMs the sub-CLI at `timeout_s`. A
    sub-agent that does not know the bound exists can spend it all reading and
    reasoning, be killed mid-turn without calling `submit_result`, and leave
    the run nothing from fifteen minutes of its own budget.

    The note gives three things, and the third is the one that changes the
    outcome: the size of the bound, an absolute clock time to compare `date`
    against, and permission to submit something short. An agent that knows it
    will be killed still has to be told that a shorter delivered entry beats a
    better undelivered one, because the task text above asks for a thorough
    entry and nothing else contradicts it.

    `started` defaults to now, which is a few seconds before the sub-agent
    actually starts -- the prompt is written, then the CLI launches. That
    error is in the safe direction and is not corrected for.
    """
    started = time.time() if started is None else started
    reserve = max(SUBMIT_RESERVE_MIN_S, timeout_s * SUBMIT_RESERVE_FRACTION)
    soft = max(0.0, timeout_s - reserve)

    def clock(offset: float) -> str:
        return datetime.datetime.fromtimestamp(started + offset).strftime("%H:%M:%S")

    return (
        "\n\n---\n\n"
        f"**You have {timeout_s / 60:.0f} minutes of wall clock for this "
        f"task, and it is being spent out of the animation run's own budget.** "
        f"The clock started at {clock(0)}. At {clock(timeout_s)} this "
        f"sub-agent is killed where it stands, and a killed sub-agent delivers "
        f"nothing: no entry reaches the run, and the work done up to that "
        f"point is discarded in full. `date` is how you check where you are.\n\n"
        f"So treat {clock(soft)} as the deadline for calling `submit_result`, "
        f"which leaves room for the call itself. Reading the brief and the two "
        f"references is the smaller part of this budget, not the larger one. "
        f"If {clock(soft)} arrives while you are still resolving something, "
        f"stop resolving it and submit the entry as it stands, leaving that "
        f"point unstated rather than unfinished: a shorter entry that is "
        f"delivered is worth more to the author than a better one that is not."
    )


def asset_summary(preload_specs) -> dict:
    """What the arm will be handed, as the rewriter needs to see it.

    The dimensions and joint ranges of a preloaded actor are load-bearing for
    the entry: a rewriter that reads them can derive a blocking conflict — a
    two-finger parallel gripper cannot part a shell one-handed — from the
    finger joints alone.
    """
    if not preload_specs:
        return {}
    from .assets.preload_asset import _describe, _load_and_verify, parse_spec
    out = {}
    for spec in preload_specs:
        name, bundle = parse_spec(spec)
        try:
            out[name] = _describe(_load_and_verify(bundle, name))
        except Exception as exc:                      # staging must not die here
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def library_block() -> str:
    """The asset library the author picks its characters and robots from.

    The author retrieves its actors from the library by what the brief
    describes. The rewriter sees the same list so
    the entry it writes can rely on the size and kind of what exists (a
    German shepherd is 0.75 m tall, not a generic dog) and does not ask the
    author to build a person, an animal or a robot the library already has.
    One line per asset, a few KB: inlined, well inside the argv cap below."""
    from .assets.asset_library import library_listing
    listing = library_listing()
    if not listing:
        return ""
    return ("\n\n# The asset library the author picks its characters and robots from\n\n"
            "The author looks here first for people, animals and robots, and builds one "
            "only when none fits. Plan around what is here, at these sizes.\n\n" + listing)


#: The two references, under the names they are staged as in the sub-agent's
#: own working directory. Order is the order `build_prompt` names them in.
REFERENCES = (
    ("animation.md", CONTRACT.read_text,
     "the animation contract the author works to"),
    ("source_reference.md", source_reference,
     "the SDK reference the author builds with"),
)


def stage_references(workspace: Path) -> list[str]:
    """Copy both references into `workspace`; return the names written.

    `workspace` is the sub-agent's own directory -- `run_oneoff` launches the
    sub-CLI with `--runs-dir <run>/sub_runs --run-id scene_rewrite`, so
    `<run>/sub_runs/scene_rewrite/` is what codex gets as its workspace and the
    only place a relative path in the prompt resolves against. Copies rather
    than pointers into the checkout for two reasons: the arm may run in a
    per-run git worktree, so the repo is not at a path this prompt could name;
    and the sandbox makes the run directory writable and everything outside it
    not necessarily readable.

    `run_oneoff` only ever `mkdir(exist_ok=True)`s this directory, so staging
    into it before the call is safe and nothing removes these afterwards --
    which also means the exact references a rewrite read stay next to the
    rewrite that read them.
    """
    from .agent_runner import render_prompt
    workspace.mkdir(parents=True, exist_ok=True)
    written = []
    for name, read, _ in REFERENCES:
        # As the author's system prompt has it (`compose_system_prompt`).
        (workspace / name).write_text(render_prompt(read()))
        written.append(name)
    return written


def build_prompt(brief: str, assets: dict | None = None) -> str:
    """The one-shot task text: instructions, the brief, then the references BY NAME.

    Both references are whole and required reading -- the rewriter is asked not
    to write what the author already produces unprompted, and it cannot honor
    that without reading what the contract already demands, while the SDK
    reference is what keeps it from requiring geometry the SDK cannot express.
    They arrive as two files in the sub-agent's own directory
    (`stage_references`) rather than pasted in.

    WHY NOT INLINE. The prompt reaches `codex` as a single positional argument
    (`cli/backends/codex.py`, `cmd.append(system + footer)`; `codex [OPTIONS]
    [PROMPT]` has no file or stdin form for the TUI), and Linux caps one argv
    element at MAX_ARG_STRLEN = 32 pages = 131,072 bytes. With both references
    inlined the text sits at that cap, and over it for a brief with a preloaded
    asset: execve returns E2BIG and the run never starts. Named, the prompt is
    a small fraction of the cap. The cost is that the sub-agent has to actually
    open them, so the section below is written as an instruction rather than a
    reference list, and it is the last thing in the prompt.
    """
    import json
    parts = [REWRITER_PROMPT.read_text(),
             "\n\n# The brief\n\n" + brief.strip()]
    if assets:
        parts.append("\n\n# Preloaded assets the author will be handed\n\n```json\n"
                     + json.dumps(assets, indent=1) + "\n```")
    parts.append(library_block())
    parts.append(
        "\n\n# Read these two files before you write anything\n\n"
        "They are in the directory you are working in, and they are part of "
        "this task rather than background: the entry you write must not "
        "restate what the first already demands of every brief, and must not "
        "ask for geometry the second cannot express. Read both in full "
        "first.\n\n"
        + "".join(f"* `{name}` — {what}\n" for name, _, what in REFERENCES))
    return "".join(parts)


def rewrite(brief: str, *, preload_specs=(), backend: str,
            run_dir: Path | str, model: str | None = None,
            reasoning_effort: str | None = None,
            timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """Return the scene entry for `brief`. Raises on failure; the caller decides.

    Runs as a one-off CLI sub-agent on the subscription the arm is already
    using -- `run_oneoff` rides the same launch path the object and scene
    dispatches do (daemon, harness, PTY-wrapped CLI, sandbox, staged home,
    no-progress watchdog) and bounds it with an explicit wall clock, so a
    wedged CLI cannot hang the run before its daemon exists.
    """
    from .oneoff import run_oneoff
    from .oneoff import SUB_RUNS_DIR
    run_dir = Path(run_dir)
    # The references are staged into the sub-agent's own directory and named
    # by `build_prompt` rather than pasted into it; see that docstring for
    # the 131,072-byte argv cap this works around. Staging has to happen
    # first: the prompt promises the files are there.
    stage_references(run_dir / SUB_RUNS_DIR / SUB_RUN)
    # `deadline_note` last, after the submit instruction: it is the constraint
    # on everything above it, and it is what makes `timeout_s` a bound the
    # sub-agent can work to rather than one that merely happens to it.
    text_in = (build_prompt(brief, asset_summary(preload_specs))
               + SUBMIT_INSTRUCTION + deadline_note(timeout_s))
    # Handed to the sub-CLI as a file rather than as argv: it costs nothing
    # (`cli.py` resolves a `--prompt` ending in `.md` by reading it) and it
    # leaves the exact request on disk next to the run that made it.
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = run_dir / "scene_rewrite_prompt.md"
    prompt_file.write_text(text_in)
    res = run_oneoff(
        prompt=str(prompt_file),
        backend=backend,
        run_dir=run_dir,
        sub_run_name=SUB_RUN,
        model=MODEL_OVERRIDE or model,
        reasoning_effort=reasoning_effort,
        timeout_s=timeout_s,
        heartbeat="scene rewrite",
    )
    text = (res.get("text") or "").strip()
    if not res.get("ok") or not text:
        # `run_oneoff` never raises; it reports. Turn that into the exception
        # the caller already handles, carrying the fields that say which of
        # the failures this was.
        raise RuntimeError(
            f"the scene rewriter did not return an entry "
            f"(outcome={res.get('outcome')!r}, timed_out={res.get('timed_out')}, "
            f"op_calls={res.get('n_op_calls')}, rc={res.get('returncode')}, "
            f"{res.get('wall_time_s', 0):.0f}s, sub_run={res.get('sub_run')!r})"
            + (f": {res['fatal']}" if res.get("fatal")
               else f": {res['stderr_tail'][-300:]}" if res.get("stderr_tail") else ""))
    return text
