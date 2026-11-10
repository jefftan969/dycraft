"""Prompt and tool-contract guards for the animation layer."""

import json
import re
from pathlib import Path

import pytest

_PROMPTS = Path(__file__).resolve().parent.parent / "agent" / "prompts"
_CLI = _PROMPTS / "animation.md"


def test_only_the_animation_layer_offers_ik(tmp_path):
    """The animation service registers `ik` on the film agent's session; the
    object and scene layers author static geometry, which a kinematics solver
    has nothing to say about, and their session registers no such op."""
    from dycraft.agent.layers.film import FilmSession
    from dycraft.agent.layers.layer import object_layer_config, scene_layer_config
    from dycraft.agent.layers.session import LayerSession, SessionConfig

    session = FilmSession(tmp_path / "animation")
    try:
        assert "ik" in session.handlers
    finally:
        session.blender_animation.jobs.close()
    for name, config in (("object", object_layer_config), ("scene", scene_layer_config)):
        layer_session = LayerSession(SessionConfig(run_dir=tmp_path / name, layer=config()))
        assert "ik" not in layer_session.handlers, name


def test_the_offline_curobo_mirror_is_present_and_indexed():
    """The run sandbox has no network (`allowedDomains = ["localhost"]`), so a
    URL in the prompt is not a pointer to anything. The V2 docs are mirrored
    into the repo instead, and the prompt has to name the mirror rather than
    the site.

    Pinned because the failure is silent in the worst way: the agent concludes
    a capability is missing and hand-rolls it, for example straight-line
    motion, which `plan_grasp` does natively.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    mirror = root / "animation" / "ik" / "curobo_docs"
    index = mirror / "README.md"
    assert index.is_file(), \
        "animation/ik/curobo_docs/README.md is the entry point"
    pages = {p.stem for p in mirror.glob("*.md")}
    # The two an h.ik script cannot work without: the planner surface and the
    # types that carry goals and constraints.
    for need in ("api_motion_planner", "api_types", "guide_motion_planning",
                 "guide_inverse_kinematics"):
        assert need in pages, f"{need}.md missing from the mirror"
    idx = index.read_text()
    for name in sorted(pages - {"README"}):
        assert name in idx, f"{name}.md is not listed in the mirror's README"
    planner = (mirror / "api_motion_planner.md").read_text()
    assert "plan_grasp" in planner and "update_tool_pose_criteria" in planner, \
        "the mirrored planner page must carry the constrained-planning surface"
    text = " ".join(_CLI.read_text().split())
    assert "references/ik/curobo_docs" in text, \
        "the prompt must name the offline mirror's staged copy"


def test_curobo_md_keeps_the_costly_ik_lessons():
    """CUROBO.md teaches the IK mistakes that cost a run the most.

    Hand-rolling straight-line IK that `linear_motion` does natively; leaving
    segments unplanned, so they penetrate; building a solver per waypoint, which
    times out; writing a joint track from a raw radian `q`; and subtracting a
    base translation instead of inverting the SE3. Each is checked here so it
    stays taught.
    """
    text = " ".join((_PROMPTS.parent.parent / "animation" / "ik"
                     / "CUROBO.md").read_text().split())
    # The DyCraft seam — neither of these is a cuRobo concern, so this file is
    # the only place they can live.
    assert "world_to_base" in text and "622" in text
    assert "to_authoring_joint_track" in text
    # Cost, and the one mistake that dominates it.
    assert "Build the solver once, outside the loop" in text
    assert "`timeout` returns your `stdout`" in text
    # The two results a reader will otherwise trust and should not.
    assert "res.success" in text and "both directions" in text.lower()
    assert "Not a proof of infeasibility" in text
    # Constrained planning exists; not knowing that is what starts hand-rolling.
    assert "plan_grasp" in text and "linear_motion" in text
    # Naming an API that does *not* exist here invites it, so the doc teaches
    # the live name and carries no table of retired cuRobo ones. A *pattern*
    # generalization (`.create` on `KinematicsCfg`) is prevented only by a
    # correct example.
    assert "ToolPoseCriteria" in text
    for gone in ("PoseCostMetric", "MotionGenConfig", "plan_single",
                 "load_from_robot_config"):
        assert gone not in text, f"{gone} does not exist here; naming it invites it"
    # And the two dycraft_ik entry points every script starts from.
    assert "robot_config" in text and "scene_config" in text


def test_the_animation_prompt_sends_ik_users_to_the_curobo_docs(tmp_path):
    """The prompt presents `h.ik` as cuRobo on a GPU with collision-aware
    whole-chain planning and names the staged copies of its docs and examples,
    a G1 example among them; without a route to them an agent writes its own
    IK and runs it on a CPU beside an idle GPU. The copies are staged into the
    run because a path into the DyCraft checkout is one a run cannot resolve."""
    from dycraft.agent import stage_ik_references
    text = (_PROMPTS / "animation.md").read_text()
    assert "at up to 16 frames" not in text
    assert "cuRobo on a GPU" in text
    assert "collision-aware whole-chain planning" in text
    staged = stage_ik_references(tmp_path)
    for doc in ('references/ik/README.md', 'references/ik/CUROBO.md',
                'references/ik/curobo_docs/README.md', 'references/ik/examples/g1_sequence.py'):
        assert doc in text, doc
        assert (tmp_path / doc).is_file(), doc
    assert staged == tmp_path / "references" / "ik"
    assert not list(staged.rglob("__pycache__"))

def test_the_animation_ops_match_its_harness_stubs(monkeypatch):
    """Every op the animation daemon registers is named by its prompt.

    `test_object_prompt_contract` holds stub-against-handler for this layer;
    this holds stub-against-prompt. An op the daemon registers and the client
    exposes but the prompt never names is one an agent will not call.

    Combined spellings count: the prompt writes
    `h.dispatch_object/scene/motion(...)` and
    `h.job_status/job_result/job_cancel(job_id)` rather than six rows.
    """
    import ast
    import tempfile
    from dycraft.agent import stage_harness_client
    from dycraft.agent.agent_runner import compose_system_prompt

    # With IK, the prompt that names every op; without it, `h.ik` is left out
    # on purpose (`test_ik_availability`).
    import importlib
    ik_mod = importlib.import_module("dycraft.animation.ik.ik")
    monkeypatch.setattr(ik_mod, "is_available", lambda: True)
    run = Path(tempfile.mkdtemp())
    stage_harness_client(run, "animation")
    stubs = {n.name for n in ast.parse((run / "harness.py").read_text()).body
             if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    text = compose_system_prompt("animation")
    # `h.a/b/c(` documents a, b, and c, and the shorthand elides a shared
    # prefix: `h.dispatch_object/scene/motion(` names all three dispatches.
    named = set(re.findall(r"h\.(\w+)\(", text))
    for group in re.findall(r"h\.((?:\w+/)+\w+)\(", text):
        parts = group.split("/")
        prefix = parts[0].rsplit("_", 1)[0] + "_" if "_" in parts[0] else ""
        named.update(parts)
        named.update(prefix + part for part in parts)
    assert not stubs - named, f"undocumented in animation.md: {sorted(stubs - named)}"

    # And the surface really is the one this session built.
    assert {"render", "inspect", "storyboard", "check_motion"} <= stubs
    assert "record_deviation" not in stubs
def test_no_staged_layer_defines_a_name_twice():
    """Each layer's `harness.py` defines every top-level name once.

    The client is concatenated from fragments, so a name defined in two of
    them silently resolves to the later one, and the shadowed definition is
    dead surface whose docstring can document a parameter the daemon rejects.
    """
    import ast
    import collections
    import tempfile

    from dycraft.agent import LAYERS, stage_harness_client

    for layer in sorted(LAYERS):
        with tempfile.TemporaryDirectory() as tmp:
            stage_harness_client(Path(tmp), layer)
            tree = ast.parse((Path(tmp) / "harness.py").read_text())
        counts = collections.Counter(
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        assert not {n for n, c in counts.items() if c > 1}, (layer, counts)


def _animation_client(tmp_path, exchange=None):
    """The staged animation client, with `_call` recording or `_exchange` faked."""
    from dycraft.agent import LAYERS, _HARNESS_CORE_SRC

    spec = LAYERS["animation"]
    namespace = {"__file__": str(tmp_path / "harness.py")}
    exec(_HARNESS_CORE_SRC.read_text().split("# Operations:")[0], namespace)
    exec(spec.tools_src.read_text(), namespace)
    calls = []
    if exchange is None:
        namespace["_call"] = lambda op, args, **kw: calls.append((op, args)) or (op, args)
    else:
        namespace["_exchange"] = lambda method, args, timeout: (
            calls.append(method) or exchange(method, args))
        namespace["POLL_S"] = 0.01
    return namespace, calls


def test_the_animation_stubs_send_what_the_daemon_reads(tmp_path):
    """The wrapper arguments reach the wire; the retired ones do not."""
    import inspect

    h, _ = _animation_client(tmp_path)
    criteria = {"support": {"verdict": "pass", "evidence": "stays on the floor"}}
    assert h["finish"](criteria=criteria, exceptions=True) == (
        "finish", {"batch": None, "fresh": False, "criteria": criteria, "exceptions": True})
    sent = h["blender"]("pass", span=("a", "b"))[1]
    assert sent["span"] == ("a", "b") and sent["shot"] is None and sent["stage"] == "animation"
    assert h["blender"]("pass", shot="s1")[1]["shot"] == "s1"
    for op in ("compile", "check_motion"):
        assert "freeze" not in inspect.signature(h[op]).parameters
        assert h[op](workers=6)[1]["workers"] == 6
    for op in ("dispatch_object", "dispatch_scene", "dispatch_motion"):
        assert "max_steps" not in inspect.signature(h[op]).parameters
        assert "max_steps" not in h[op]("n", "b")[1]
        # The client's header said every project operation takes `batch=` and
        # `fresh=`, and these raised TypeError when an agent passed them. They
        # return at once and take neither; the header names them as the exception.
        assert not {"batch", "fresh"} & set(inspect.signature(h[op]).parameters)
    from dycraft.agent import LAYERS
    text = LAYERS["animation"].tools_src.read_text()
    header = text[text.index("# The animation client."):].split("def plan(")[0]
    assert "(`dispatch_*`, `contribution_status`, `cancel_contribution`) do not" in header
    # This layer's forms: no `out_dir`, a `frames_from`.
    assert {"camera_name", "render_settings"} <= set(inspect.signature(h["snapshot"]).parameters)
    assert "out_dir" not in inspect.signature(h["snapshot"]).parameters
    assert "frames_from" in inspect.signature(h["final_render"]).parameters


def test_job_result_waits_for_a_job_that_finishes_meanwhile(tmp_path):
    states = iter(["running", "running", "completed"])
    h, calls = _animation_client(tmp_path, lambda method, args: {
        "ok": True, "result": {"job_id": args["job_id"], "status": next(states)}})
    assert h["job_result"]("j1", wait=30)["status"] == "completed"
    assert calls == ["job_result"] * 3


def test_job_result_returns_the_pending_record_when_the_wait_runs_out(tmp_path):
    h, calls = _animation_client(tmp_path, lambda method, args: {
        "ok": True, "result": {"job_id": args["job_id"], "status": "running"}})
    assert h["job_result"]("j1", wait=0.1)["status"] == "running"
    assert len(calls) > 1
    calls.clear()
    # No wait is one read.
    assert h["job_result"]("j1")["status"] == "running"
    assert calls == ["job_result"]


def test_the_rendering_table_matches_the_render_defaults():
    """The Rendering table in animation.md is the one place an agent reads the
    render defaults, so each row must match what the code applies: the
    diagnostic and review settings, the final-render recipe, and the plan's
    delivery floor."""
    from dycraft.animation.blender.render_config import (
        DIAGNOSTIC_RENDER_SETTINGS, REVIEW_RENDER_SETTINGS, final_render_settings)
    from dycraft.animation.planning import DEFAULT_MIN_PIXELS

    text = _CLI.read_text()
    table = text[text.index("\n## Rendering\n"):].split("\n\n")[1]
    rows = dict(re.findall(r"^\| (.+?) \| (.+?) \|$", table, re.M))

    def row(op):
        return next(default for use, default in rows.items() if f"`{op}`" in use)

    def size(settings):
        return f"{settings['width']}×{settings['height']}"

    diagnostic, review = DIAGNOSTIC_RENDER_SETTINGS, REVIEW_RENDER_SETTINGS
    assert row("render") == f"{diagnostic['engine'].title()}, {size(diagnostic)}"
    assert final_render_settings(dict(review))["denoise"]
    assert row("storyboard") == (
        f"{review['engine'].title()}, {size(review)}, {review['samples']} denoised samples")

    final = final_render_settings()
    assert final["denoise"]
    delivery = row("final_render")
    assert delivery.startswith(f"{final['engine'].title()}, {final['samples']} denoised samples")
    floor = re.search(r"\((\d+)×(\d+) pixels by default\)", delivery)
    assert floor and int(floor[1]) * int(floor[2]) == DEFAULT_MIN_PIXELS
