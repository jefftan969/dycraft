"""Private contributors. The parent run alone decides how to integrate results."""
import shutil
from pathlib import Path

from ._process import run_cancellable
from .blender.bundle import skip_sandbox_masks, tree_hash, write_json
from .blender.runtime import Jobs
from .context import read_json


def running(service):
    jobs = getattr(service, "contributions", None)
    if jobs is None:
        return False
    return any(jobs.status(ident)["status"] in ("queued", "running") for ident in service.contribution_ids)


def dispatch(service, *, role, name, brief, clip=None, max_steps=None):
    """Start a private contributor. `max_steps` is accepted and ignored.

    The child CLI enforces no step limit, so the client does not send it; a
    `harness.py` staged by an older client still does, and a resumed run keeps
    the client it was staged with.
    """
    if role not in ("object", "scene", "motion"):
        raise ValueError("contribution role must be object, scene, or motion")
    if clip is not None and role != "motion":
        raise ValueError("only a motion contribution owns a clip")
    if clip is not None and (not isinstance(clip, str) or not clip.strip()):
        raise ValueError("clip must name an NLA strip in your project")
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("name must contain only letters, digits, underscores, and hyphens")
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("supply an independently actionable contribution brief")
    if clip is not None:
        known = project_clips(service)
        if clip not in known:
            raise ValueError(
                f"clip {clip!r} is not an NLA strip of your project, which has "
                + (", ".join(sorted(known)) if known else "no NLA strips")
                + ". A contributor holds itself to the storyboard states inside the clip it "
                "owns, so it could not find any of them.")
    with service.lock:
        # Two refusals with two messages: one is worth retrying in a moment and
        # the other never is, and an agent must be able to tell which it has.
        if service.ctx.committed:
            raise ValueError("the project is committed; contributions cannot start after completion")
        if service.edit_pending:
            raise ValueError("an edit is still running; dispatch again once it has finished, "
                             "since a contributor starts from a copy of the project")
        # A canceled contributor is not counted: its process is still being
        # stopped, which takes a moment, and a dispatch refused in that moment
        # was refused for a slot that was already free.
        live = [i for i in service.contribution_ids
                if service.contributions.status(i)["status"] in ("running", "queued")
                and not service.contributions.cancel_requested(i)]
        if len(live) >= 3:
            raise ValueError("three contributions are already running")
        base = service.root / "contributions" / name
        if base.exists():
            raise ValueError("contribution name already exists; use a new name to preserve earlier work")
        tree_hash(service.design)
        base.mkdir(parents=True)
        version = service.version()
        shutil.copytree(service.work, base / "input_project")
        shutil.copytree(service.design, base / "input_sources", ignore=skip_sandbox_masks)
        if role == "motion":
            service.build_of(base / "input_project")
        # Written into the child's run dir rather than passed as a CLI flag:
        # the launcher already stages `brief.md` here, the child reads its own
        # root at startup, and keeping it out of the CLI keeps one fewer
        # surface in step. A child that owns a clip is held to the storyboard
        # states inside it, and to any state not yet drawn (whose clip is
        # unknown). Dispatch itself has no storyboard precondition: most
        # recorded motion dispatches were probes, made before any acceptance.
        write_json(base / "contribution.json", {"role": role, "clip": clip})
        prompt = base / "brief.md"
        prompt.write_text(contribution_brief(service, role, brief, clip))
        from ..agent.layers._object_dispatch import _child_argv
        argv = _child_argv(service.ctx, name=name, prompt_path=prompt,
                           sub_runs=base.parent,
                           advisory_hours=child_hours(service.wall_budget()))
        argv[argv.index("--layer") + 1] = {"object": "object", "scene": "scene", "motion": "animation"}[role]
        if role == "scene":
            shutil.copytree(base / "input_sources", base, dirs_exist_ok=True)
        elif role == "motion":
            argv += ["--prior-animation", str(base / "input_project"), "--contribution-role", "motion"]
        def execute(directory, cancel):
            with (base / "child.log").open("w") as log:
                result = run_cancellable(argv, stdout=log, stderr=log, cancel=cancel)
            manifest = read_json(base / "ir.json")
            # `ok` is that the child returned, however it ended: requiring a
            # committed `ir.json`, which only `finish()` writes, would report a
            # probe child that never called `finish()` as failed while its
            # files sit in `run_dir`. `finished`, `committed`, and `ended_by`
            # say how it ended.
            ending = how_it_ended(base, result.returncode, committed=manifest is not None)
            return {"ok": True, "returncode": result.returncode, **ending,
                    "committed": manifest is not None, "role": role,
                    "run_dir": str(base), "ir": str(base / "ir.json") if manifest else None,
                    "source": str(base / ("object.py" if role == "object" else "scene.py" if role == "scene" else "design/scene.py")),
                    "blend": str(base / "native/work/animation.blend") if role == "motion" else None,
                    "base_version": version, "integration": "explicitly copy/adapt sources or Blender data after inspection"}
        record = service.contributions.submit("dispatch_" + role, execute, step=service.session.current_op_step(), version=version)
        service.contribution_ids.append(record["job_id"])
        return {**record, "run_dir": str(base), "role": role}


def child_hours(budget):
    """The `--advisory-hours` a contributor is launched with: what is left of
    the parent run's own budget, at least a minute, or 0 (no budget) when the
    parent has none (`wall_budget` is None), so a child does not fall back to
    the CLI's default."""
    if budget is None:
        return 0
    return max(60, budget["remaining_s"]) / 3600.0


#: Exit statuses that name why a contributor's CLI stopped
#: (`cli/backends.py`, `cli/cli.py::self_kill_exit_code`).
_ENDINGS = {0: "exited", 86: "cli_fatal_error", 87: "blocked_at_startup", 137: "killed", -9: "killed",
            143: "terminated", -15: "terminated"}


def how_it_ended(base, returncode, *, committed):
    """`finished`, `ended_by` and, when it applies, `finish_refused` or
    `cli_error`, for a contributor that returned.

    `finished` is not "`finish_result.json` exists", which `finish()` also
    writes when it refuses, and why a child stopped is otherwise only in
    `child.log`: a parent would read a child that hit its usage limit as a
    crash, or one that never started as finished.
    """
    record = read_json(base / "finish_result.json")
    reached = _reached_harness(base / "trace.jsonl")
    out = {"reached_harness": reached}
    if record is not None:
        out["finished"] = bool(record.get("ok"))
        if not out["finished"]:
            out["ended_by"] = "finish_refused"
            out["finish_refused"] = record.get("error") or record.get("reason") or "finish() refused"
            return out
        out["ended_by"] = "finished"
        return out
    # Object and scene contributors have no `finish()`; their commit is it.
    out["finished"] = committed
    if committed:
        out["ended_by"] = "finished"
    elif not reached and returncode not in (86, 87):
        out["ended_by"] = "never_reached_harness"
    else:
        out["ended_by"] = _ENDINGS.get(returncode, f"exit {returncode}")
    if returncode == 86:
        out["cli_error"] = _cli_fatal_reason(base / "child.log")
    return out


def _reached_harness(trace):
    try:
        with open(trace, "rb") as fh:
            return any(b'"op_call"' in line for line in fh)
    except OSError:
        return False


def _cli_fatal_reason(log):
    """The fatal error the CLI watchdog quoted, such as a usage limit."""
    import re
    try:
        text = Path(log).read_text(errors="replace")[-20000:]
    except OSError:
        return None
    found = re.findall(r"recorded a fatal error with no activity for \S+: (.+)", text)
    return found[-1].strip() if found else None


def project_clips(service) -> dict:
    """The live project's NLA clips, by name.

    Checked at dispatch rather than by the child: a clip-owning contributor
    scopes its storyboard gates to the states inside its clip, so a name the
    project does not have gated it on no drawn state at all, and the parent
    would only have learned that from the child's result. One `place_only`
    evaluation answers it: it reads the timeline and steps or draws no frame.
    """
    import tempfile
    import threading
    scratch = service.native / "dispatch_checks"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clips-", dir=scratch) as tmp:
        info = service.worker(service.work, Path(tmp),
                              {"op": "evaluate", "render_states": False,
                               "place_only": True, "states": []},
                              threading.Event())
    return dict(info.get("clips") or {})


def contribution_brief(service, role, brief, clip=None):
    """The child's task: its own work first, the parent's brief as context only.

    The parent's brief carries its time budget, its film length, and pointers
    to `scene.txt` and `motion.txt` "in the same directory", none of which is
    true of the child, so it is quoted last and labeled.
    """
    where = {
        "motion": ("Your run starts from a copy of that agent's project; work on it with the "
                   "harness as usual."),
        "scene": ("Your editable scene source is `scene.py` in your run directory, a copy of that "
                  "agent's design."),
        "object": "Author the object in `object.py` in your run directory.",
    }[role]
    lines = [f"Contribute {role} work to an animated film, in this private run. The agent that "
             "dispatched you reviews what you return and integrates it into its own project; do "
             "not edit its files.", "", f"Contribution: {brief}", ""]
    if clip:
        lines += [f"You own the clip {clip!r}.", ""]
    lines += [where, "`input_project/` and `input_sources/` hold copies of that agent's project "
              "and design for reference.", "",
              "When you finish, summarize what you changed, the interfaces it depends on, and "
              "anything unresolved.", "",
              "That agent's whole task, for context. Its time budget and film length are its "
              f"own, and its `scene.txt` and `motion.txt` are in `{service.root}`:", ""]
    lines += [f"> {line}" if line.strip() else ">" for line in service.ctx.user_prompt.strip().splitlines()]
    return "\n".join(lines) + "\n"


def register(session, service):
    service.contributions = Jobs(service.native / "contribution_jobs", session.trace)
    service.contribution_ids = []
    for role in ("object", "scene", "motion"):
        def handler(_role=role, **kwargs):
            if _role != "motion":
                kwargs.pop("clip", None)
            return dispatch(service, role=_role, **kwargs)
        session.op("dispatch_" + role)(handler)
    session.register_handler("contribution_status", service.contributions.status)
    session.register_handler("cancel_contribution", service.contributions.cancel)
    def close(timeout_s=30):
        service.contributions.close(timeout_s=timeout_s)
        service.jobs.close(timeout_s=timeout_s)
    session.drain_on_shutdown = close
