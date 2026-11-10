"""The half of `animation/ik/dycraft_ik.py` that needs no GPU: frames, units, defaults.

cuRobo is imported inside the functions that need it, precisely so this file
can run under the ordinary interpreter. These are the tests that pin the traps
that fail *silently* — a wrong frame or a wrong unit puts the arm somewhere
plausible and every downstream verifier still passes.

The adapter is staged into the export beside `_rotation.py`, so it is imported
the way the child does: by path, with the staged rotation module on `sys.path`.
"""

from __future__ import annotations

import importlib
import math
import shutil
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _load_adapter(tmp_path: Path):
    """Stage `dycraft_ik.py` + `_rotation.py` and import them as the child does."""
    d = tmp_path / "ik"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(_REPO / "animation" / "ik" / "dycraft_ik.py", d / "dycraft_ik.py")
    shutil.copy(_REPO / "sdk" / "_rotation.py", d / "_rotation.py")
    sys.path.insert(0, str(d))
    for name in ("dycraft_ik", "_rotation"):
        sys.modules.pop(name, None)
    try:
        return importlib.import_module("dycraft_ik"), d
    finally:
        sys.path.pop(0)


def _meta(*, base_translation=(0.0, -0.32, 0.752),
          base_quat=(1.0, 0.0, 0.0, 0.0)) -> dict:
    """A `scene_meta.json` shaped like the real one, with a hinged instance.

    A revolute joint at a non-zero default (35 deg) and a prismatic one
    (0.1 m), so the deg->rad conversion and the "prismatic is already meters" case are both live. Plus
    a joint nobody thinks of as part of the arm, which is the one that gets
    left out of a hand-written joint list.
    """
    return {
        "scene_ref": "deadbeef",
        "static": [{"instance": "arm", "object_ref": "obj1",
                    "world_pose": {"translation": list(base_translation),
                                   "quaternion_wxyz": list(base_quat)},
                    "aabb": [[0, 0, 0], [1, 1, 1]]}],
        "free_bodies": [],
        "joints": {"arm": {
            "mount": {"type": "fixed", "limits": None, "default": None,
                      "axis": None, "unit": None},
            "hinge": {"type": "revolute", "limits": (0.0, 90.0),
                      "default": 35.0, "axis": (0, 0, 1), "unit": "deg"},
            "tray": {"type": "prismatic", "limits": (0.0, 0.25),
                     "default": 0.1, "axis": (1, 0, 0), "unit": "m"},
            "cam_tilt": {"type": "revolute", "limits": (-30.0, 30.0),
                         "default": 0.0, "axis": (0, 1, 0), "unit": "deg"},
        }},
    }


# ---------------------------------------------------------------------------
# Frames  [trap 1]
# ---------------------------------------------------------------------------

def test_a_world_target_round_trips_through_the_base_frame(tmp_path):
    """And the placement is **rotated**, which is the whole point.

    Arms are usually placed axis-aligned, so a `world_to_base` written as a
    translation subtraction passes the case anyone checks by hand and then
    mislocates the first rotated instance without saying anything.
    """
    ik, _ = _load_adapter(tmp_path)
    # 37 deg about +Z, and a base that is nowhere near the origin.
    import _rotation as _rot
    q = _rot.quat_from_axis_angle_rad((0.0, 0.0, 1.0), math.radians(37.0))
    meta = _meta(base_translation=(0.41, -0.32, 0.752), base_quat=q)

    want_p = (0.13, 0.22, 1.05)
    want_q = _rot.quat_from_axis_angle_rad((0.0, 1.0, 0.0), math.radians(-21.0))
    bp, bq = ik.world_to_base("arm", want_p, want_q, meta=meta)
    got_p, got_q = ik.base_to_world("arm", bp, bq, meta=meta)

    assert got_p == pytest.approx(want_p, abs=1e-12)
    assert _rot.quat_angle_deg(got_q, want_q) < 1e-9
    # And the base-frame answer is genuinely different from the world one —
    # otherwise the round-trip would pass on a function that does nothing.
    assert bp != pytest.approx(want_p, abs=1e-6)


def test_translation_subtraction_would_not_survive_a_rotated_base(tmp_path):
    """Names the wrong implementation so the next reader knows what is guarded."""
    ik, _ = _load_adapter(tmp_path)
    import _rotation as _rot
    q = _rot.quat_from_axis_angle_rad((0.0, 0.0, 1.0), math.radians(90.0))
    meta = _meta(base_translation=(1.0, 0.0, 0.0), base_quat=q)
    got = ik.world_to_base("arm", (2.0, 0.0, 0.0), meta=meta)
    naive = (2.0 - 1.0, 0.0, 0.0)
    # Rz(90) maps the world +X offset onto the base's -Y.
    assert got == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    assert got != pytest.approx(naive, abs=1e-6)


def test_base_pose_is_the_pose_the_export_recorded(tmp_path):
    ik, _ = _load_adapter(tmp_path)
    meta = _meta()
    t, q = ik.base_pose("arm", meta=meta)
    rec = meta["static"][0]["world_pose"]
    assert list(t) == rec["translation"]
    assert list(q) == rec["quaternion_wxyz"]


def test_an_unknown_instance_names_the_ones_it_knows(tmp_path):
    ik, _ = _load_adapter(tmp_path)
    with pytest.raises(KeyError, match="arm"):
        ik.base_pose("nosuch", meta=_meta())


# ---------------------------------------------------------------------------
# Joint names  [trap 4]
# ---------------------------------------------------------------------------

def test_every_driveable_joint_is_in_the_order_and_fixed_ones_are_not(tmp_path):
    """cuRobo's loader raises on a joint missing from `cspace.joint_names`, and
    `RobotBuilder.build()` leaves that list empty — measured, it does not fill
    it in from the URDF. The joint most easily forgotten is the one nobody
    thinks of as part of the arm; here that is `cam_tilt`.
    """
    ik, _ = _load_adapter(tmp_path)
    names = ik.joint_names("arm", meta=_meta())
    assert names == ["cam_tilt", "hinge", "tray"]
    assert "mount" not in names, "a fixed joint is not part of a configuration"


# ---------------------------------------------------------------------------
# Units and the rest pose  [trap 2]
# ---------------------------------------------------------------------------

def test_the_default_configuration_is_the_committed_one_in_internal_units(tmp_path):
    """URDF q=0 is not the rest pose, and this is the vector that fixes it.

    A revolute default must reach cuRobo in radians; a prismatic one must NOT
    be converted, because both units are already meters.
    """
    ik, _ = _load_adapter(tmp_path)
    meta = _meta()
    q = ik.default_q("arm", meta=meta)
    names = ik.joint_names("arm", meta=meta)
    by = dict(zip(names, q))
    assert by["hinge"] == pytest.approx(math.radians(35.0), abs=1e-12)
    assert by["tray"] == pytest.approx(0.1, abs=1e-12)
    assert by["cam_tilt"] == pytest.approx(0.0, abs=1e-12)


def test_surface_and_internal_round_trip(tmp_path):
    ik, _ = _load_adapter(tmp_path)
    meta = _meta()
    surface = {"hinge": 62.5, "tray": 0.2, "cam_tilt": -12.0}
    q = ik.q_from_surface("arm", surface, meta=meta)
    assert ik.q_to_surface("arm", q, meta=meta) == pytest.approx(surface)


def test_an_omitted_joint_takes_its_default_not_zero(tmp_path):
    """"Leave that one alone" means where the scene put it, which for 17% of
    the corpus is not zero."""
    ik, _ = _load_adapter(tmp_path)
    meta = _meta()
    q = ik.q_from_surface("arm", {"cam_tilt": 5.0}, meta=meta)
    got = ik.q_to_surface("arm", q, meta=meta)
    assert got["hinge"] == pytest.approx(35.0)
    assert got["tray"] == pytest.approx(0.1)
    assert got["cam_tilt"] == pytest.approx(5.0)


def test_limits_come_back_in_internal_units(tmp_path):
    ik, _ = _load_adapter(tmp_path)
    lim = ik.joint_limits_internal("arm", meta=_meta())
    assert lim["hinge"] == pytest.approx((0.0, math.radians(90.0)))
    assert lim["tray"] == pytest.approx((0.0, 0.25))


def test_a_wrong_length_q_says_so_instead_of_silently_zipping_short(tmp_path):
    ik, _ = _load_adapter(tmp_path)
    with pytest.raises(ValueError, match="driveable"):
        ik.q_to_surface("arm", [0.0, 0.0], meta=_meta())


def test_an_unknown_joint_name_is_rejected(tmp_path):
    ik, _ = _load_adapter(tmp_path)
    with pytest.raises(KeyError, match="nosuch"):
        ik.q_from_surface("arm", {"nosuch": 1.0}, meta=_meta())


def test_a_track_is_per_joint_lists_ready_for_keys(tmp_path):
    ik, _ = _load_adapter(tmp_path)
    meta = _meta()
    qs = [ik.q_from_surface("arm", {"hinge": v}, meta=meta) for v in (0.0, 45.0, 90.0)]
    track = ik.to_authoring_joint_track("arm", qs, meta=meta)
    assert track["hinge"] == pytest.approx([0.0, 45.0, 90.0])
    assert track["tray"] == pytest.approx([0.1, 0.1, 0.1]), "held at its default"


# ---------------------------------------------------------------------------
# Judging  [trap 3]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Staging and ik_meta — the parent's half, still no GPU
# ---------------------------------------------------------------------------

def test_staging_puts_the_adapter_and_the_sdks_own_rotation_in_the_world(tmp_path):
    """`_rotation.py` byte-identical to `sdk/`'s, for the reason the sim export
    copies it too: a second implementation of the surface/internal conversion
    living in the IK env is a factor-of-57 divergence nothing could detect."""
    from dycraft.animation.ik.ik import stage_helpers
    d = tmp_path / "ik"
    stage_helpers(d)
    assert (d / "dycraft_ik.py").is_file()
    assert (d / "_rotation.py").read_text() == (_REPO / "sdk" / "_rotation.py").read_text()


def _tiny_scene():
    """Two boxes on a table-ish parent, one of them articulated."""
    from dycraft.sdk import ir as _ir  # noqa: F401  (import shape check only)
    box = {"id": "objA", "parts": [
        {"name": "body", "primitive": "box",
         "params": {"aabb_min": [-0.1, -0.1, 0.0], "aabb_max": [0.1, 0.1, 0.2]}}],
        "joints": []}
    return box


def test_a_timeout_returns_what_the_script_printed_before_it_was_killed(
        tmp_path, monkeypatch):
    """A timed-out IK script returns what it printed before it was killed.

    Two halves have to hold together and each fails silently without the
    other: `-u` on the child, so a killed process has not buffered its output
    away, and reading `TimeoutExpired.stdout` in the parent. Without `-u` this
    returns `None`; without the parent half it never arrives.

    The timeout comes from a measured start-up of the same child, so the script
    has printed before it is killed even on a loaded machine; a fixed two
    seconds can expire before the child has started.
    """
    import json as _json
    import time
    from dycraft.animation.ik.ik import ik
    d = tmp_path / "ik"
    d.mkdir()
    (d / "scene_meta.json").write_text(_json.dumps({"instances": {}}))
    monkeypatch.setenv("DYCRAFT_IK_PYTHON", sys.executable)
    import importlib
    ik_mod = importlib.import_module("dycraft.animation.ik.ik")
    # This interpreter stands in for cuRobo's: a GPU is taken as visible, and
    # the setup's no-GPU backstop (its own test is in test_ik_setup) left out.
    monkeypatch.setattr(ik_mod, "gpu_visible", lambda: (True, ""))
    (tmp_path / "no_setup.py").write_text("")
    monkeypatch.setattr(ik_mod, "_SETUP_PATH", tmp_path / "no_setup.py")
    started = time.perf_counter()
    assert ik("print('ready')", ik_dir=d, timeout_s=120.0)["ok"]
    timeout = max(2.0, round(2 * (time.perf_counter() - started) + 1.0, 1))
    prov = tmp_path / "ik_runs" / "runaway"
    out = ik("print('built the solver'); import time; time.sleep(600)",
             ik_dir=d, provenance_dir=prov, timeout_s=timeout)
    assert out["error_kind"] == "timeout"
    assert "built the solver" in out["stdout"]
    assert out["timeout_s"] == timeout
    assert "built the solver" in (prov / "stdout.txt").read_text()
    # The advice names a cause the script can fix (a solver built inside a
    # loop), not a knob it cannot reach.
    assert "loop" in out["message"]


def test_the_joint_order_comes_from_the_export_not_the_alphabet(tmp_path):
    """The joint order comes from the export, because a sorted order fails silently.

    A `default_q` in alphabetical order, handed to a solver that reads URDF
    order, passes every self-consistency check (the FK round trip still closes)
    and puts the hand far from where it was asked for, with nothing raised.

    So the order is read from `ik_meta.json` beside the export by default, and
    the alphabetical fallback applies only where there is no export at all.
    """
    import json as _json
    ik, d = _load_adapter(tmp_path)
    meta = _meta()
    # No ik_meta.json yet: the fallback is alphabetical.
    assert ik.joint_names("arm", meta=meta, ik_dir=str(d)) == ["cam_tilt", "hinge", "tray"]
    # With one, the export's order wins — and it is deliberately NOT sorted.
    (d / "ik_meta.json").write_text(_json.dumps({
        "scene_ref": "x",
        "instances": {"arm": {"active_joints": ["hinge", "tray", "cam_tilt"]}},
        "obstacles": []}))
    assert ik.joint_names("arm", meta=meta, ik_dir=str(d)) == ["hinge", "tray", "cam_tilt"]
    # …and every derived vector follows it without the caller saying so.
    q = ik.default_q("arm", meta=meta, ik_dir=str(d))
    assert q == pytest.approx([math.radians(35.0), 0.1, 0.0])
    assert list(ik.q_to_surface("arm", q, meta=meta, ik_dir=str(d))) == \
        ["hinge", "tray", "cam_tilt"]


# ---------------------------------------------------------------------------
# Expensive — needs the IK interpreter and a GPU
# ---------------------------------------------------------------------------

def _ik_available():
    from dycraft.animation.ik import is_available
    return is_available()


@pytest.mark.expensive
def test_the_curobo_reference_still_describes_the_installed_package(tmp_path):
    """`animation/ik/CUROBO.md`'s claims, re-read off the build. Fix the doc, not this.

    Three of them are *negative* claims — things that do NOT work — and those
    are the ones worth a test, because they decay silently into a doc that
    confidently warns about a bug someone has since fixed.
    """
    from dycraft.animation.ik import ik as run_ik, is_available
    if not is_available():
        pytest.skip("no IK interpreter configured")
    d = tmp_path / "ik"
    d.mkdir()
    (d / "scene_meta.json").write_text("{}")
    code = """
import inspect
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsState
from curobo.robot_builder import RobotBuilder
from curobo.sphere_fit import fit_spheres_to_mesh, SphereFitType
from curobo.types import GoalToolPose, Pose, JointState
result["create"] = str(inspect.signature(InverseKinematicsCfg.create))
result["fit"] = str(inspect.signature(RobotBuilder.fit_collision_spheres))
result["matrix"] = str(inspect.signature(RobotBuilder.compute_collision_matrix))
result["build_returns"] = str(RobotBuilder.build.__annotations__.get("return", "?"))
result["ik_has_load_from_config"] = hasattr(InverseKinematics, "load_from_config")
result["state_fields"] = [a for a in dir(KinematicsState) if not a.startswith("_")]
result["sphere_fit_has_seed"] = "seed" in inspect.signature(fit_spheres_to_mesh).parameters
"""
    out = run_ik(code, ik_dir=d)
    assert out["ok"], out
    r = out["result"]
    # §2 — the entry point, and the flag we refuse to trust
    for kw in ("num_seeds", "self_collision_check", "position_tolerance",
               "orientation_tolerance", "scene_model"):
        assert kw in r["create"], f"CUROBO.md §2 names {kw}"
    assert r["ik_has_load_from_config"] is False, "CUROBO.md §1 says there is none"
    # §3 — the builder knobs the adapter relies on
    assert "sphere_density" in r["fit"] and "compute_metrics" in r["fit"]
    assert "use_collision_mesh" in r["fit"]
    assert "custom_ignore" in r["matrix"] and "batch_size" in r["matrix"]
    assert "KinematicsLoaderCfg" in str(r["build_returns"]), \
        "CUROBO.md §3: build() returns an object, which is why save() cannot round-trip it"
    # §3 — the fit still has no seed, which is why the cache is a correctness
    # mechanism rather than an optimization
    assert r["sphere_fit_has_seed"] is False
    # §4 — the pose comes off tool_poses; there is no ee_position on this build
    assert "tool_poses" in r["state_fields"]
    assert "ee_position" not in r["state_fields"]


@pytest.mark.expensive
def test_constrained_planning_is_still_here_and_still_renamed(tmp_path):
    """`animation/ik/CUROBO.md` §5b-i..iii: constrained planning is here, renamed.

    Searching this build for `PoseCostMetric`, the name other cuRobo releases
    use, finds nothing, which reads as "this build cannot constrain a straight
    line". It can: `ToolPoseCriteria`. Without it, straight segments get
    hand-rolled out of thousands of single-goal IK solves.

    If a future bump brings back `PoseCostMetric`, or moves these, the doc has to
    change with it — that is what this test is for.
    """
    from dycraft.animation.ik import ik as run_ik, is_available
    if not is_available():
        pytest.skip("no IK interpreter configured")
    d = tmp_path / "ik"
    d.mkdir()
    (d / "scene_meta.json").write_text("{}")
    out = run_ik("""
import dataclasses, inspect
import curobo.types as T
from curobo.motion_planner import MotionPlanner, GraspPlanResult
result["has_tool_pose_criteria"] = hasattr(T, "ToolPoseCriteria")
result["has_pose_cost_metric"] = hasattr(T, "PoseCostMetric")
result["criteria_ctors"] = [m for m in dir(T.ToolPoseCriteria)
                            if not m.startswith("_")]
result["linear_motion"] = str(inspect.signature(T.ToolPoseCriteria.linear_motion))
result["plan_grasp"] = str(inspect.signature(MotionPlanner.plan_grasp))
result["update_criteria"] = str(inspect.signature(
    MotionPlanner.update_tool_pose_criteria))
result["grasp_result_fields"] = [f.name for f in
                                 dataclasses.fields(GraspPlanResult)]
""", ik_dir=d)
    assert out["ok"], out
    r = out["result"]
    # §5b-i — the rename, in both directions. The negative half matters most:
    # absence of `PoseCostMetric` is not absence of the feature.
    assert r["has_tool_pose_criteria"] is True
    assert r["has_pose_cost_metric"] is False, \
        "if this build regains PoseCostMetric, CUROBO.md §5b-i must say so"
    for ctor in ("linear_motion", "track_position", "track_orientation",
                 "track_position_and_orientation", "disabled"):
        assert ctor in r["criteria_ctors"], f"§5b-i lists {ctor}"
    assert "axis" in r["linear_motion"]
    # It is planner state, not a plan_pose argument — the sticky-criteria trap.
    assert "tool_pose_criteria" in r["update_criteria"]
    # §5b-ii — approach/grasp/lift in one call, over a goalset.
    for kw in ("grasp_poses", "grasp_approach_axis", "grasp_approach_offset",
               "grasp_lift_axis", "plan_approach_to_grasp",
               "disable_collision_links"):
        assert kw in r["plan_grasp"], f"§5b-ii names {kw}"
    # §5b-ii/iii — the per-phase result fields the retiming reads.
    for f in ("approach_interpolated_trajectory", "grasp_interpolated_trajectory",
              "lift_interpolated_trajectory", "goalset_index",
              "approach_interpolated_last_tstep"):
        assert f in r["grasp_result_fields"], f"§5b-ii names {f}"


@pytest.mark.expensive
def test_the_result_object_still_forgets_its_tolerances(tmp_path):
    """CUROBO.md's claim that the result object drops its tolerances. If this
    ever starts passing the tolerances through, `raw_success` becomes usable and the docs should say so.
    """
    from dycraft.animation.ik import ik as run_ik, is_available
    if not is_available():
        pytest.skip("no IK interpreter configured")
    d = tmp_path / "ik"
    d.mkdir()
    (d / "scene_meta.json").write_text("{}")
    out = run_ik("""
import dataclasses
from curobo.inverse_kinematics import InverseKinematicsResult as R
result["fields"] = [f.name for f in dataclasses.fields(R)] if dataclasses.is_dataclass(R) \
    else [a for a in dir(R) if not a.startswith("_")]
""", ik_dir=d)
    assert out["ok"], out
    for f in ("success", "position_error", "rotation_error", "position_tolerance"):
        assert f in out["result"]["fields"], f"CUROBO.md §2 reads {f} off the result"
