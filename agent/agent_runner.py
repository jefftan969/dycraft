"""Run-directory and prompt helpers shared by the launcher and the daemons.

  * `local_run_id`      — `YYMMDD_HHMMSS_<8 hex>` time-sorted run IDs.
  * `ensure_run_dir`    — mkdir + return `(run_id, run_dir)`.
  * `compose_system_prompt(layer)` — the layer's prompt file, with its
    optional blocks resolved.
  * `compose_footer(...)` — builds the user-task footer appended to the
    system prompt.
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ---------------------------------------------------------------------------
# Run-dir primitives
# ---------------------------------------------------------------------------


def local_run_id() -> str:
    """`YYMMDD_HHMMSS_<8 hex chars>` in local time."""
    return datetime.now().strftime("%y%m%d_%H%M%S") + "_" + secrets.token_hex(4)


def ensure_run_dir(runs_dir: Path | str, run_id: str | None, *,
                   reader: bool = False) -> tuple[str, Path]:
    """Create the run's folder and return `(run_id, run_dir)`, `run_dir` being
    where its records go and the agent works. Generates a run_id if none was
    given.

    With `reader`, the run is laid out for a reader: `run_dir` is
    `<runs_dir>/<run_id>/src`, and the top level is left for the deliverables
    (`run_layout`). Otherwise it is `<runs_dir>/<run_id>`. Nothing else is
    made here: a layer makes its own folders when it first writes to them, so
    a launch refused before it starts leaves none.
    """
    runs_dir = Path(runs_dir)
    run_id = run_id or local_run_id()
    if reader:
        from ..run_layout import new_run
        return run_id, new_run(runs_dir, run_id)
    run_dir = (runs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------


def _has_mocap_library() -> bool:
    from ..animation.blender.runtime import mocap_library
    return mocap_library() is not None


def _has_ik() -> bool:
    from ..animation.ik.ik import is_available
    return is_available()


def _has_asset_library() -> bool:
    """An installed asset library with its index: what the prompt points the agent at, as `references/assets/INDEX.md`
    (`python -m dycraft.install library`)."""
    from .assets.asset_library import installed_index
    return installed_index() is not None


#: Prompt and Skill material about something only some checkouts have, and the
#: test for it. In a prompt it is a block of whole lines between `<!-- name -->`
#: and `<!-- /name -->` (`render_prompt`), optionally with an `<!-- else -->`
#: line before the lines that replace it when the test fails; a Skill is named
#: by its directory (`OPTIONAL_SKILLS`, read by `stage_skills`). The agent never
#: sees the markers, and sees the material only while the test holds.
OPTIONAL = {"mocap": _has_mocap_library, "ik": _has_ik, "assets": _has_asset_library}
OPTIONAL_SKILLS = {"mocap_reference": "mocap"}

_OPTIONAL_BLOCK = re.compile(r"^<!-- (\w+) -->\n(.*?)^<!-- /\1 -->\n(\n?)", re.M | re.S)
_ELSE = re.compile(r"^<!-- else -->\n", re.M)


def _resolve_block(match: re.Match) -> str:
    branches = _ELSE.split(match.group(2), maxsplit=1)
    held = OPTIONAL[match.group(1)]()
    body = branches[0] if held else (branches[1] if len(branches) > 1 else "")
    return body + match.group(3) if body else ""


def render_prompt(text: str) -> str:
    """`text` with each optional block resolved to the lines its test picks,
    without its markers, or dropped together with the blank line after it when
    it picks none."""
    return _OPTIONAL_BLOCK.sub(_resolve_block, text)


#: Each layer's system prompt file under `agent/prompts/`.
ROLE_PROMPTS = {"object": "object.md", "scene": "scene.md",
                "animation": "animation.md", "plain": "plain.md"}


def compose_system_prompt(layer: str) -> str:
    """The layer's system prompt: its prompt file with optional blocks resolved."""
    try:
        name = ROLE_PROMPTS[layer]
    except KeyError:
        raise ValueError(f"layer must be one of {sorted(ROLE_PROMPTS)}, got {layer!r}") from None
    return render_prompt((_PROMPTS_DIR / name).read_text())


#: Snapshot of the composed system prompt, written into every run bundle.
SYSTEM_PROMPT_FILE = "system_prompt.md"


def stage_system_prompt(run_dir: Path, text: str) -> Path:
    """Snapshot the composed system prompt into the run bundle.

    The prompt files under `agent/prompts/` are edited far more often than
    they are committed, so neither a git SHA nor the file paths identify what
    a given run actually ran against — a run whose behaviour depended on an
    uncommitted prompt section is otherwise unreconstructable. And the prompt
    is first-hand evidence for "why did the agent do that", which the trace
    and the verifier verdicts are not.

    Same reasoning, and the same shape, as two existing precedents: the
    transport core is copied verbatim into each run dir as `harness.py` "so a
    run dir stays replay/debug-safe", and the object layer stages each part's
    build123d recipe at `assets/src/<digest>.py`. Store the artifact, not a
    pointer to it.

    **Call this from every surface.** Run bundles are required to be
    identical in shape across backends; a snapshot written only on the CLI
    path would break that.
    """
    dst = run_dir / SYSTEM_PROMPT_FILE
    dst.write_text(text)
    return dst


#: Where a task too long to pass inline is written, in the run directory.
TASK_FILE = "task.md"
#: Set when the task is in `TASK_FILE`; `$DYCRAFT_USER_PROMPT` is then the
#: pointer the agent was given. See `user_prompt_from_env`.
TASK_FILE_ENV = "DYCRAFT_USER_PROMPT_FILE"
#: How much of one argument the agent CLI's first message may take. Linux
#: refuses to start a program any one of whose arguments or environment strings
#: is over 128 KiB (`MAX_ARG_STRLEN`), with E2BIG and no output from the
#: program. Codex receives the system prompt and the footer as one argument,
#: which makes it the binding case; the margin is for what the CLIs add.
INLINE_TASK_LIMIT = 120 * 1024


def stage_long_task(run_dir: Path, task: str, system_prompt: str,
                    footer) -> tuple[str, Path | None]:
    """The task as the agent CLI should receive it, and where it was written.

    `footer(task)` composes the first message for a given task. When the system
    prompt and that message fit in one argument the task is passed unchanged
    and nothing is written. Otherwise it goes to `TASK_FILE` and the agent is
    handed a pointer to it. Briefs and dispatches are far below the limit; a
    review's prompt, which embeds the plan and its evidence, is not always.
    """
    combined = system_prompt + "\n\n" + footer(task)
    if len(combined.encode("utf-8")) <= INLINE_TASK_LIMIT:
        return task, None
    path = Path(run_dir) / TASK_FILE
    path.write_text(task, encoding="utf-8")
    size = len(task.encode("utf-8"))
    return (f"Your task is in `{path}`. It is {size // 1024} KiB, too long to pass here. "
            "Read all of it before you start.", path)


def user_prompt_from_env(environ=None) -> str:
    """The run's task, for a harness daemon: `TASK_FILE` when the launcher
    wrote one, else `$DYCRAFT_USER_PROMPT`."""
    environ = os.environ if environ is None else environ
    path = environ.get(TASK_FILE_ENV)
    if path:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError:
            pass
    return environ.get("DYCRAFT_USER_PROMPT", "")


def _task_footer(intro: str, prompt: str, run_dir: Path, closing: str,
                 extra: tuple = ()) -> str:
    """The first user message: the brief and the facts that depend on this run.

    The system prompt owns the workflow, the harness and how to finish, so this
    restates none of them. Every line of the brief is quoted, so the notes a
    launcher appends to it read as brief rather than as harness text.
    """
    quoted = [f"> {line}" if line.strip() else ">" for line in prompt.strip().splitlines()]
    return "\n".join([
        "", "---", "", "## User task (this session)", "", intro, "", *quoted, "",
        f"- Run directory: `$DYCRAFT_RUN_DIR={run_dir}`; `cd` here for every command. The harness "
        "client, `harness.py`, is already there.",
        *extra,
        "- Skills: `$DYCRAFT_RUN_DIR/.claude/skills/` and `$DYCRAFT_RUN_DIR/.agents/skills/`. Read "
        "`<skill>/SKILL.md` there and resolve its supporting files relative to that directory.",
        "",
        closing,
    ])


def compose_footer(
    layer: str, prompt: str, run_dir: Path,
    *, contribution_role: str | None = None,
) -> str:
    """Compose the user-task footer sent as the initial user message
    (Claude) or appended to the system prompt (Codex)."""
    if layer == "plain":
        # Minimal footer: just the task. The system prompt already states the
        # one rule (return via submit_result) and how to invoke it.
        return "\n".join([
            "", "---", "", "## Task", "",
            f"> {prompt}",
            "",
            "Deliver the answer by running the `submit_result` command now — "
            "do not stop at writing it in a reply.",
        ])
    if layer == "animation":
        intro = ("Author the motion contribution described below." if contribution_role == "motion"
                 else "Design, stage, animate, and deliver the complete film described below.")
        return _task_footer(intro, prompt, run_dir, f"After `finish()` succeeds, print `DONE {run_dir}/ir.json`.")
    sentinel = f"Stop once the sentinel `DONE {run_dir}/ir.json` has been printed."
    if layer == "object":
        return _task_footer(
            "Build the object described below.", prompt, run_dir,
            "Begin by planning the object (§Planning the object): its parts, its operating states, "
            "and the interfaces that constrain them. " + sentinel)
    if layer == "scene":
        return _task_footer(
            "Build the scene described below — the objects in it as well as their arrangement.",
            prompt, run_dir,
            "Begin by planning the scene (§Planning the scene): what it contains, how it is laid out "
            "and used, which objects recur as families, and which are worth delegating. Dispatch "
            "those early. " + sentinel,
            extra=("- `$DYCRAFT_RUN_DIR/objects/`: where delegated object runs are written. Empty until "
                   "you dispatch one; their sources are yours to read and edit.",))
    raise ValueError(f"unknown layer {layer!r}")


__all__ = [
    "local_run_id",
    "ensure_run_dir",
    "compose_system_prompt",
    "compose_footer",
    "stage_system_prompt",
    "stage_long_task",
    "user_prompt_from_env",
    "TASK_FILE",
    "TASK_FILE_ENV",
    "INLINE_TASK_LIMIT",
    "SYSTEM_PROMPT_FILE",
]
