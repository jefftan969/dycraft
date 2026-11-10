"""The README at the top of a run folder: what was asked, how it ended, what is here, and what to do next.

Rendered by `run_layout.publish` from the run's own records (`meta.json`, `ir.json`), never re-derived, so it says
exactly what the launcher and the harness recorded. It is rewritten at launch, at a commit and when the launcher
exits, so it always describes the run's current state.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from ._layers import canonical_layer

_KIND = {"animation": "animation run", "object": "object run", "scene": "scene run", "plain": "run"}
#: What each layer delivers, as the README names it.
_DELIVERS = {"animation": "a film", "object": "an object", "scene": "a scene", "plain": "an answer"}
#: Frame ranges and storyboard states listed in full before the rest are counted.
_RANGES_SHOWN = 5
_STATES_SHOWN = 8


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _when(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _backend(meta: dict) -> str:
    runtime = str(meta.get("runtime") or "")
    return "Claude Code" if runtime.startswith("claude") else "Codex" if runtime.startswith("codex") else runtime


def _agent(meta: dict) -> str:
    text = ", ".join(p for p in (_backend(meta), meta.get("model")) if p)
    if meta.get("reasoning_effort"):
        text += f" (reasoning effort {meta['reasoning_effort']})"
    return text


def _span(meta: dict) -> str:
    start, end = _when(meta.get("started_at")), _when(meta.get("finished_at"))
    if start is None:
        return ""
    text = start.strftime("%Y-%m-%d %H:%M")
    if end is not None:
        text += f"–{end.strftime('%H:%M') if end.date() == start.date() else end.strftime('%Y-%m-%d %H:%M')}"
        text += f" UTC ({(end - start).total_seconds() / 3600:.1f} h)"
    else:
        text += " UTC"
    return text


def _title(meta: dict) -> str:
    """The brief's first sentence, without its closing period; the brief's file name for a brief given as a file."""
    source = meta.get("prompt_source")
    if isinstance(source, dict) and source.get("path"):
        return f"Brief: {Path(source['path']).name}"
    for line in str(meta.get("prompt") or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            sentence = re.split(r"(?<=[.!?])\s+", line, maxsplit=1)[0].rstrip(".")
            return sentence if len(sentence) <= 120 else sentence[:117].rstrip() + "…"
    return "DyCraft Run"


def _cell(text) -> str:
    return " ".join(str(text or "").split()).replace("|", "\\|")


def _and(words: list[str]) -> str:
    """"a", "a and b", "a, b, and c": the serial comma, as every user-facing list here has it."""
    if len(words) < 3:
        return " and ".join(words)
    return ", ".join(words[:-1]) + ", and " + words[-1]


def _ranges(frames) -> str:
    """`[[9, 323], [328, 2221]]` as "9–323 and 328–2221", the long tail counted."""
    spans = [(int(a), int(b)) for a, b in frames or () if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    words = [f"{a}" if a == b else f"{a}–{b}" for a, b in spans]
    if not words:
        return "no frames"
    if len(words) > _RANGES_SHOWN:
        rest = spans[_RANGES_SHOWN:]
        n = sum(b - a + 1 for a, b in rest)
        return _and(words[:_RANGES_SHOWN] + [f"{len(rest)} more ranges ({n} frames)"])
    return _and(words)


def _unmet_line(item) -> str | None:
    """One unmet item in a sentence, or None for a criterion, which the criteria table already shows."""
    if not isinstance(item, dict):
        return _cell(item)
    kind = item.get("kind")
    if kind == "criterion":
        return None
    if kind == "check":
        return f"`{item.get('verifier')}` fails on frames {_ranges(item.get('frames'))}."
    if kind == "storyboard":
        names = [f"`{s}`" for s in item.get("states") or ()]
        if len(names) > _STATES_SHOWN:
            names = names[:_STATES_SHOWN] + [f"{len(names) - _STATES_SHOWN} more"]
        return f"Storyboard states not accepted: {_and(names) or 'some'}."
    if kind == "unchecked":
        return f"Frames no check covered: {_ranges(item.get('frames'))}."
    if kind == "design":
        repairs = "; ".join(_cell(r) for r in item.get("repairs") or ())
        return f"`{item.get('verifier')}` asks for a design repair" + (f": {repairs.rstrip('.')}." if repairs else ".")
    rest = {k: v for k, v in item.items() if k != "kind"}
    return f"{kind}: {_cell(json.dumps(rest, sort_keys=True))}"


def _film_line(ir: dict, *, exceptions: bool) -> str:
    render = ir.get("render") or {}
    facts = []
    if ir.get("duration") is not None:
        facts.append(f"{float(ir['duration']):.1f} s")
    if ir.get("n_frames") and ir.get("fps"):
        facts.append(f"{ir['n_frames']} frames at {float(ir['fps']):g} fps")
    if render.get("width") and render.get("height"):
        scale = (render.get("resolution_percentage") or 100) / 100
        facts.append(f"{round(render['width'] * scale)}×{round(render['height'] * scale)}")
    version = str(ir.get("blender_version") or "").split(" ")[0]
    blender = f"the Blender {'.'.join(version.split('.')[:2])} project" if version else "the Blender project"
    film = f"`film.mp4` ({', '.join(facts)})" if facts else "`film.mp4`"
    return (f"The film was delivered{' with exceptions' if exceptions else ''}. It is {film}, and `animation.blend` "
            f"is {blender} that renders it.")


def _stopped_by(meta: dict, stop: dict) -> str:
    """Why the agent stopped, in the words recorded for it: the agent CLI's own, or DyCraft's when it stopped it."""
    reason, detail = str(stop.get("reason") or ""), str(stop.get("detail") or "").strip()
    if reason.startswith(("claude_", "codex_")) and detail:
        return f"{_backend(meta) or 'The agent CLI'} stopped with \"{detail.rstrip('.')}\"."
    return f"The agent was stopped: {detail.rstrip('.') or reason}."


def _status(internal: Path, meta: dict, ir: dict, layer: str, here: list) -> list[str]:
    """The outcome paragraph."""
    failure = meta.get("launch_failure")
    if isinstance(failure, dict):
        why = str(failure.get("message") or failure.get("detail") or failure.get("reason") or "").strip()
        why = re.sub(r"^FATAL:\s*", "", (why.splitlines() or [""])[0]).rstrip(".")
        fixes = [str(f) for f in failure.get("fix") or ()]
        lines = [f"The run did not start: {why}." + (" To fix it:" if fixes else "")]
        if fixes:
            lines += [""] + [f"- {fix[:1].upper() + fix[1:]}" for fix in fixes]
        return lines
    if layer == "animation" and "film.mp4" in here:
        return [_film_line(ir, exceptions=ir.get("outcome") == "delivered_with_exceptions" or bool(ir.get("unmet")))]
    if layer in ("object", "scene") and f"{layer}.urdf" in here:
        views = len(list((internal.parent / "views").glob("*.png"))) if "views" in here else 0
        it = f"`{layer}.urdf`" + (", with the meshes it names in `meshes/`" if "meshes" in here else "")
        return [f"The {layer} was delivered. It is {it}" + (f", and `views/` holds {views} renders of it." if views
                                                            else ".")]
    if layer == "plain" and "answer.md" in here:
        return ["The answer is in `answer.md`."]
    delivers = _DELIVERS.get(layer, "anything")
    limit = meta.get("time_limit") or {}
    stop = meta.get("agent_stop")
    if limit.get("reached"):
        lines = [f"The run stopped at its {float(limit.get('hours') or 0):g}-hour time limit without delivering "
                 f"{delivers}."]
    elif isinstance(stop, dict):
        lines = [f"The run stopped without delivering {delivers}. {_stopped_by(meta, stop)}"]
    elif not meta.get("finished_at") and not meta.get("outcome"):
        return ["The run is still going. " + {
            "animation": "The film and its Blender project appear here when the agent delivers them.",
            "object": "The object's URDF, meshes, and views appear here when the agent delivers them.",
            "scene": "The scene's URDF, meshes, and views appear here when the agent delivers them.",
        }.get(layer, "Its answer appears here when the agent submits it.")]
    else:
        lines = [f"The run ended without delivering {delivers} (outcome: {meta.get('outcome') or 'none recorded'})."]
    if layer == "animation":
        film = _last_film(internal)
        if film:
            lines[0] += f" The last film the agent rendered, `{film}`, was not checked or delivered."
    return lines


def _last_film(internal: Path) -> str | None:
    """The newest film a not-delivered animation rendered, unchecked and uncommitted."""
    films = sorted((internal / "native" / "jobs").glob("*/delivery/film/clip.mp4"), key=lambda p: p.stat().st_mtime)
    return films[-1].relative_to(internal.parent).as_posix() if films else None


def render(internal, here) -> str:
    """The README for the run whose records are in `internal`; `here` is what its top level holds."""
    internal = Path(internal)
    top = internal.parent
    meta = _read_json(internal / "meta.json") or {}
    ir = _read_json(internal / "ir.json") or {}
    layer = canonical_layer(meta.get("layer")) or "plain"
    refused = isinstance(meta.get("launch_failure"), dict)
    out = [f"# {_title(meta)}", ""]
    byline = " · ".join(p for p in (f"DyCraft {_KIND.get(layer, 'run')} `{top.name}`", _agent(meta), _span(meta)) if p)
    out += [byline, ""]
    unsandboxed = meta.get("unsandboxed")
    if isinstance(unsandboxed, dict):
        reason = str(unsandboxed.get("reason") or "").strip()
        reason = reason if len(reason) <= 300 else reason[:299] + "…"
        out += [f"> **Ran without a sandbox** (`--allow-unsandboxed`): {unsandboxed.get('agent') or 'the agent'}'s "
                "sandbox could not start on this machine" + (f" ({reason})" if reason else "") + ", so the agent's "
                "commands could read and write anywhere the user who launched it could.", ""]
    blender = meta.get("blender_sandbox")
    if isinstance(blender, dict) and blender.get("mode") == "unconfined":
        reason = str(blender.get("reason") or "").strip()
        reason = reason if len(reason) <= 300 else reason[:299] + "…"
        out += ["> **Blender ran without a sandbox** (`--allow-unsandboxed`): neither bubblewrap nor Landlock could "
                "confine it on this machine" + (f" ({reason})" if reason else "") + ", so the agent's Blender code "
                "could read and write anywhere the user who launched it could.", ""]
    out += _status(internal, meta, ir, layer, here)
    out.append("")

    delivered = any(name in here for name in ("film.mp4", "views", f"{layer}.urdf", "answer.md"))
    if layer == "animation" and delivered:
        criteria = ir.get("criteria") or {}
        if criteria:
            out += ["## Criteria", "", "| Criterion | Verdict | Evidence |", "|---|---|---|"]
            for name, entry in criteria.items():
                entry = entry if isinstance(entry, dict) else {}
                out.append(f"| {_cell(name)} | {_cell(entry.get('verdict'))} | {_cell(entry.get('evidence'))} |")
            out.append("")
        unmet = ir.get("unmet") or []
        lines = [line for line in map(_unmet_line, unmet) if line]
        if lines or not unmet:
            out += ["## Unmet", ""]
            out += [f"- {line}" for line in lines] or ["Nothing: every frame, state, and check is clean."]
            out.append("")
        if "notes" in here:
            notes = sorted(p.relative_to(top).as_posix() for p in (top / "notes").rglob("*") if p.is_file())
            if notes:
                out += ["## The Agent's Notes", ""] + [f"- [{n}]({n})" for n in notes] + [""]

    if not refused:
        out += ["## Next", ""]
        running = not (meta.get("finished_at") or meta.get("outcome") or meta.get("agent_stop") or delivered)
        out.append(f"- {'Follow' if running else 'Open'} it in the viewer: "
                   f"`python -m dycraft.viewer --runs-dir {top.parent}`")
        if delivered and layer in ("animation", "object", "scene"):
            out.append(f"- Export it to GLB: `python -m dycraft.export glb {top}`")
        if layer == "animation" and delivered:
            out.append(f"- Revise it: `python -m dycraft.cli --resume {top} --prompt \"what to change\"`. A revision "
                       "starts from the delivered project, not from edits made to `animation.blend` by hand.")
        elif layer in ("object", "scene") and delivered:
            out.append(f"- Use it in a scene or a film: `--preload-asset NAME={top}`")
        if not delivered and not running and not (meta.get("claude_session_id") or meta.get("codex_session_id")):
            # Its agent never opened a session (stopped at its start), so there is nothing for --resume to carry on.
            out.append("- Its agent never started a session, so there is nothing to resume: fix what stopped it, "
                       "above, then run the same command again.")
        elif not delivered and not running:
            stop = meta.get("agent_stop") or {}
            if (meta.get("time_limit") or {}).get("reached"):
                out.append(f"- Continue it with more time: `python -m dycraft.cli --resume {top} --max-hours 4`")
            elif "limit" in f"{stop.get('reason', '')} {stop.get('detail', '')}".lower():
                out.append(f"- Continue it once the limit resets: `python -m dycraft.cli --resume {top}`")
            else:
                out.append(f"- Continue it: `python -m dycraft.cli --resume {top}`")
        out.append("")

    brief = str(meta.get("prompt") or "").strip()
    if brief:
        out += ["## Brief", ""] + [f"> {line}" if line.strip() else ">" for line in brief.splitlines()] + [""]
    if refused:
        footer = "DyCraft's records and logs are in `src/`."
    elif layer == "animation" and delivered:
        footer = ("The agent's working files, DyCraft's records and logs, and the delivered bundle that the viewer and "
                  "the scorer read are in `src/`.")
    else:
        footer = "The agent's working files and DyCraft's records and logs are in `src/`."
    out += ["---", footer]
    return "\n".join(out) + "\n"


__all__ = ["render"]
