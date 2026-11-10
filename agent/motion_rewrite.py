"""The per-brief motion entry, generated inside the run by `dycraft.cli`.

**Where this runs.** Straight after `scene_rewrite.rewrite()`, under the same
conditions -- a fresh `animation` authoring run -- unless the run was
handed `--motion-brief FILE` or `--no-motion-brief`. It reads the scene entry
the step before it wrote, so the performance it describes happens in that
place, and writes the result into the run as `motion.txt`. The prompt is given
`POINTER`, one sentence naming that file. Measured on three briefs at
medium effort: 106-139 s and 1,600-2,300 words per brief, paid out of the
run's own budget like the scene entry.

**What it supplies, and why the animation contract cannot.** The contract's
stage 1 already asks for an action series with preconditions, results and
changes of control, and authors write one -- then commit to it within
minutes, with states that sit at the ends of the brief's verbs. What a run of
the contract otherwise discovers only after building is per-brief: the
technique these bodies use to reach each outcome, the states between the
brief's verbs where what holds or supports a participant changes, and the
spatial relationships each of those states needs.
`agent/prompts/motion_rewriter.md` is pointed at exactly those, after
naming the default performance it displaces.

**Scene entry absent.** With `--no-scene-brief` there is no `scene.txt`; the
motion entry is still written, from the brief alone, and the prompt is told
so rather than handed an empty section. A scene rewrite that *failed* reaches
this same path, because `cli.py` passes the entry only if the file exists.

**When it fails.** `rewrite()` raises, and `cli.py --on-rewrite-failure`
decides, as for the scene entry: `ask` (the default; it stops when there is no
terminal), `abort`, or `continue` without a motion entry. See
`cli._rewrite_failed`, and `DEFAULT_TIMEOUT_S` below for the run that made
stopping unconditionally untenable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .scene_rewrite import (REFERENCES, SUBMIT_INSTRUCTION, asset_summary, library_block,
                            deadline_note, stage_references)

REPO = Path(__file__).resolve().parents[1]
REWRITER_PROMPT = REPO / "agent" / "prompts" / "motion_rewriter.md"

#: As `scene_rewrite.MODEL_OVERRIDE`: a bare CLI model id, and nothing else.
#: Otherwise the rewriter runs on the model, account and effort of the run
#: that asked for it, which `cli.py` passes through.
MODEL_OVERRIDE = os.environ.get("DYCRAFT_MOTION_REWRITER_MODEL", "")

#: A motion rewrite takes about two minutes at medium effort and can run past
#: the bound at xhigh. Kept equal to the scene rewriter's bound rather than
#: tightened, since both are spent before the daemon exists. `deadline_note`
#: tells the sub-agent the bound, and on a miss `cli._rewrite_failed` asks,
#: stops or continues without the entry (`--on-rewrite-failure`). See
#: `scene_rewrite.DEFAULT_TIMEOUT_S` for why the number is not simply raised.
DEFAULT_TIMEOUT_S = 900.0

#: The sentence the brief carries instead of the entry. Deliberately does not
#: mention `scene.txt`: a run with `--no-scene-brief` still gets this one.
POINTER = (
    "How the performance is physically carried out is described in "
    "`motion.txt`, in the same directory you are working in. Read it before "
    "planning: it sets out the technique, the states between the brief's "
    "actions and the spatial relationships they need, and it is part of this "
    "task."
)

#: Written into the run as this name; the contract names it at §Workspace.
STAGED_NAME = "motion.txt"

#: The sub-agent's own directory under `<run>/sub_runs/`.
SUB_RUN = "motion_rewrite"

#: What the scene section says when there is no scene entry.
NO_SCENE = ("No scene entry was written for this brief. The author builds the "
            "place from the brief alone, so state what the performance needs "
            "of it as requirements on the place.")


#: Seconds of film to plan per hour of `--max-hours`, and the most to plan for any limit. From the
#: benchmark's runtime calibration (`benchmark/sizing.py`): a film this long fits its limit with room
#: to render, inspect and finish.
FILM_SECONDS_PER_HOUR = 9
FILM_SECONDS_CAP = 150


def time_limit_block(hours: float | None) -> str:
    """The sizing section for a run launched with `--max-hours H` (H > 0), or "" without one.

    It caps and does not fill: a brief implying more than fits is cut to its point, and a short brief
    is not padded."""
    if not hours or hours <= 0:
        return ""
    seconds = max(1, round(min(FILM_SECONDS_CAP, FILM_SECONDS_PER_HOUR * hours)))
    return ("\n\n# The time limit\n\n"
            f"This run stops after {hours:g} hours, and everything — planning, building, animating, "
            "rendering and checking — has to fit. Size the performance you describe to it: time every "
            "state at the pace the `humanoid_motion` skill requires, and keep the total to about "
            f"{seconds} seconds of film or less. If the brief implies more, keep what it exists to show "
            "and its complication, and leave out setup and teardown. If it implies less, describe what "
            "it implies and no more.")


def build_prompt(brief: str, scene: str | None, assets: dict | None = None,
                 time_limit_hours: float | None = None) -> str:
    """The one-shot task text: instructions, brief, assets, scene entry, references.

    The same shape as `scene_rewrite.build_prompt` with one section added: the
    scene entry, inlined because it is per-brief and a few KB, where the two
    references are staged as files because they are ~120 KB together and the
    prompt reaches `codex` as one argv element (see that docstring).
    """
    parts = [REWRITER_PROMPT.read_text(),
             "\n\n# The brief\n\n" + brief.strip(),
             time_limit_block(time_limit_hours)]
    if assets:
        parts.append("\n\n# Preloaded assets the author will be handed\n\n```json\n"
                     + json.dumps(assets, indent=1) + "\n```")
    parts.append(library_block())
    parts.append("\n\n# The scene entry\n\n"
                 + (scene.strip() if scene and scene.strip() else NO_SCENE))
    parts.append(
        "\n\n# Read these two files before you write anything\n\n"
        "They are in the directory you are working in, and they are part of "
        "this task rather than background: the entry you write must not "
        "restate what the first already demands of every brief, and must not "
        "require motion or articulation the second cannot express. Read both "
        "in full first.\n\n"
        + "".join(f"* `{name}` — {what}\n" for name, _, what in REFERENCES))
    return "".join(parts)


def rewrite(brief: str, *, scene: str | None, preload_specs=(), backend: str,
            run_dir: Path | str, model: str | None = None,
            reasoning_effort: str | None = None,
            timeout_s: float = DEFAULT_TIMEOUT_S,
            time_limit_hours: float | None = None) -> str:
    """Return the motion entry for `brief` in `scene`. Raises on failure.

    `time_limit_hours` is this launch's `--max-hours` when one was given: the
    entry is then sized to fit it (`time_limit_block`).

    The same launch as `scene_rewrite.rewrite`: a one-off CLI sub-agent via
    `run_oneoff`, references staged into its own workspace first, the request
    left on disk as `motion_rewrite_prompt.md` beside the run.
    """
    from .oneoff import SUB_RUNS_DIR, run_oneoff
    run_dir = Path(run_dir)
    stage_references(run_dir / SUB_RUNS_DIR / SUB_RUN)
    text_in = (build_prompt(brief, scene, asset_summary(preload_specs), time_limit_hours)
               + SUBMIT_INSTRUCTION + deadline_note(timeout_s))
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = run_dir / "motion_rewrite_prompt.md"
    prompt_file.write_text(text_in)
    res = run_oneoff(
        prompt=str(prompt_file),
        backend=backend,
        run_dir=run_dir,
        sub_run_name=SUB_RUN,
        model=MODEL_OVERRIDE or model,
        reasoning_effort=reasoning_effort,
        timeout_s=timeout_s,
        heartbeat="motion rewrite",
    )
    text = (res.get("text") or "").strip()
    if not res.get("ok") or not text:
        raise RuntimeError(
            f"the motion rewriter did not return an entry "
            f"(outcome={res.get('outcome')!r}, timed_out={res.get('timed_out')}, "
            f"op_calls={res.get('n_op_calls')}, rc={res.get('returncode')}, "
            f"{res.get('wall_time_s', 0):.0f}s, sub_run={res.get('sub_run')!r})"
            + (f": {res['fatal']}" if res.get("fatal")
               else f": {res['stderr_tail'][-300:]}" if res.get("stderr_tail") else ""))
    return text
