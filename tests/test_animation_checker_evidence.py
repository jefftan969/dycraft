"""What the animation checker computes and says.

Each block names the recorded failure it pins. The pure helpers run in the
test interpreter; `deepest_vertex_m` and the streamed `verify()` case need
the Blender interpreter (`mathutils`, `bpy`) and run there as a child.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dycraft.animation.blender.verify import (
    ROOM_SURFACE_TAGS, allowed, escaped_faces, inherited_allowances, joint_findings,
    out_of_room, pairs_by_depth, penetration_pairs, replaced_surfaces, resolve_declarations,
    resolve_room_surfaces, room_containment, validate_declarations)


def _slab(lo, hi):
    return (np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))


#: A 4 x 4 x 3 m room, 0.1 m shell, interior [-2, 2] x [-2, 2] x [0, 3].
FLOOR = _slab([-2.1, -2.1, -0.1], [2.1, 2.1, 0.0])
CEILING = _slab([-2.1, -2.1, 3.0], [2.1, 2.1, 3.1])
WALLS = {"wall_xmin": _slab([-2.1, -2.1, 0], [-2.0, 2.1, 3]),
         "wall_xmax": _slab([2.0, -2.1, 0], [2.1, 2.1, 3]),
         "wall_ymin": _slab([-2.1, -2.1, 0], [2.1, -2.0, 3]),
         "wall_ymax": _slab([-2.1, 2.0, 0], [2.1, 2.1, 3])}


# --- room surfaces resolve without a ceiling ------------------------------

def test_a_lone_floor_slab_is_the_floor_not_the_ceiling():
    """`open_air` builds one surface. The naive union-midpoint test tagged it
    `ceiling`, because a lone box's center is the union's midpoint."""
    assert resolve_room_surfaces({"room_0": FLOOR}) == {"room_0": "floor"}


def test_an_open_top_room_resolves_its_floor_and_four_walls():
    boxes = {"room_0": FLOOR, **{f"room_{i + 1}": box for i, box in enumerate(WALLS.values())}}
    tags = resolve_room_surfaces(boxes)
    assert sorted(tags.values()) == sorted(["floor", *WALLS])
    assert {tags[f"room_{i + 1}"] for i in range(4)} == set(WALLS)
    assert "ceiling" not in tags.values()


def test_a_capped_room_still_resolves_against_its_declared_box():
    boxes = {"f": FLOOR, "c": CEILING, **WALLS}
    tags = resolve_room_surfaces(boxes, room=[[-2, -2, 0], [2, 2, 3]])
    assert tags == {"f": "floor", "c": "ceiling", **{name: name for name in WALLS}}


def test_a_lone_wall_is_not_guessed():
    assert resolve_room_surfaces({"w": WALLS["wall_xmax"]}) == {}


def test_containment_checks_only_the_faces_that_were_built():
    walls = {name: box for name, box in WALLS.items()}
    boxes = {"floor": FLOOR, **walls}
    tags = {"floor": "floor", **{name: name for name in walls}}
    containment = room_containment(boxes, tags)
    lo, hi, faces = containment
    assert sorted(faces.values()) == sorted(["floor", *WALLS])
    assert np.allclose(lo, [-2, -2, 0]) and np.allclose(hi[:2], [2, 2])
    # Taller than the walls is not "outside" a room with no ceiling ...
    assert escaped_faces(np.array([0, 0, 0]), np.array([.5, .5, 9]), containment) == []
    # ... but leaving through a wall still is.
    assert escaped_faces(np.array([1.8, 0, 0]), np.array([2.4, .5, 1]),
                         containment) == ["wall_xmax"]


def test_out_of_room_measures_only_the_checked_faces():
    escapes = [{"object": "tall", "frame": 3, "aabb": [[1.8, 0, 0], [2.3, .5, 9]]}]
    faces = {(0, -1): "wall_xmin", (0, +1): "wall_xmax", (2, -1): "floor"}
    row, = out_of_room(escapes, ([-2, -2, 0], [2, 2, 3.1]), faces)
    # 0.3 m through the wall, not 5.9 m above a ceiling that does not exist.
    assert row["outside_by_m"] == pytest.approx(.3)
    row, = out_of_room(escapes, ([-2, -2, 0], [2, 2, 3.1]))
    assert row["outside_by_m"] == pytest.approx(5.9)


def _bodies(**boxes):
    """`bodies` rows as the frame loop builds them: name, verts, tris, lo, hi."""
    return [(name, None, None, *box) for name, box in boxes.items()]


def test_a_replaced_floor_is_still_room_floor():
    """`replace_room_surface('floor', floor)` removes the shell floor, and
    `room.floor` -- the documented ground of `supported` and `planted` --
    must still resolve, or those events are never checked. It names the
    replacement's part that spans the most of the floor."""
    bodies = _bodies(**{"floor/skirting": _slab([-2, -2, 0], [2, -1.98, .1]),
                        "floor/slab": FLOOR, "crate/body": _slab([0, 0, 0], [1, 1, 1]),
                        "room.wall_xmin": WALLS["wall_xmin"]})
    lookup = {b[0]: i for i, b in enumerate(bodies)}
    aliases = replaced_surfaces(bodies, {"floor": "floor", "wall_xmax": "panel"}, lookup)
    assert aliases == {"room.floor": 1}
    # A surface the shell still has keeps its own body.
    assert replaced_surfaces(bodies, {"wall_xmin": "crate"}, lookup) == {}


def test_a_replacing_wall_resolves_across_its_own_plane():
    bodies = _bodies(**{"panel/frame": _slab([2.0, -2, 0], [2.3, 2, .2]),
                        "panel/board": _slab([2.0, -2, 0], [2.05, 2, 3])})
    lookup = {b[0]: i for i, b in enumerate(bodies)}
    assert replaced_surfaces(bodies, {"wall_xmax": "panel"}, lookup) == {"room.wall_xmax": 1}


# --- declarations are collected, not raised, and never written ------------

INFO = {"fps": 24.0, "n_frames": 48, "frame_start": 0, "clips": {}}
BASELINE = {"identities": {"x/p": {}, "y/q": {}}, "room_objects": ["room_0"]}


def test_one_bad_record_is_reported_and_the_rest_still_resolve():
    declarations = {"units": "frames", "collision_allowances": [
        {"a": "x/p", "b": "y/q", "reason": "seat", "window": [4, 8]},
        {"a": "x/p", "b": "typo/q", "reason": "seat"},
        {"a": "x/p", "b": "y/q"},
        {"a": "x/p", "b": "y/q", "reason": "late", "window": [40, 90]},
    ], "events": [
        {"name": "rests", "predicate": "touch", "a": "x/p", "b": "room.floor",
         "window": [0, 47]},
        {"name": "bogus", "predicate": "hovers", "a": "x/p", "b": "y/q", "window": [0, 1]},
        {"name": "no_joint", "predicate": "joint_at", "value": 0.0, "window": [0, 1]},
        "not an event",
    ]}
    snapshot = json.dumps(declarations, sort_keys=True)
    index, events, errors = resolve_declarations(declarations, BASELINE, INFO)
    assert json.dumps(declarations, sort_keys=True) == snapshot
    assert allowed(index, "y/q", "x/p", 6) and not allowed(index, "x/p", "y/q", 9)
    assert [event["name"] for event, _ in events] == ["rests"]
    kinds = [(row["kind"], row["index"]) for row in errors]
    assert kinds == [("collision_allowance", 1), ("collision_allowance", 2),
                     ("collision_allowance", 3), ("event", 1), ("event", 2), ("event", 3)]
    assert "unknown 'typo/q'" in errors[0]["reason"]
    assert "hovers" in errors[3]["reason"] and errors[3]["name"] == "bogus"
    assert "joint" in errors[4]["reason"]
    assert validate_declarations(declarations, BASELINE, INFO) == errors


def test_a_supported_event_naming_no_instance_is_a_declaration_error():
    """`supported` reads its instance's parts by prefix; a name that matches none left the margin unmeasured, which
    counted as holding, so the blocking EventsSatisfied passed on a typo. It is now rejected like any bad event."""
    declarations = {"units": "frames", "events": [
        {"name": "sits", "predicate": "supported", "a": "chiar", "window": [0, 10]},
        {"name": "stands", "predicate": "supported", "a": "x", "window": [0, 10]}]}
    _, events, errors = resolve_declarations(declarations, BASELINE, INFO)
    assert [event["name"] for event, _ in events] == ["stands"]
    assert errors == [{"kind": "event", "index": 0, "name": "sits", "a": "chiar", "window": [0, 10],
                       "reason": "a 'chiar' names no instance of this scene"}]
    # Instances also come from the evaluated scene's objects, where the design lists no identities.
    scene_only = {**INFO, "collision_objects": ["chair/seat", "chair/leg"]}
    _, events, errors = resolve_declarations(declarations, {**BASELINE, "identities": {}}, scene_only)
    assert [event["name"] for event, _ in events] == [] and [e["name"] for e in errors] == ["sits", "stands"]
    declarations["events"][0]["a"] = "chair"
    _, events, errors = resolve_declarations(declarations, {**BASELINE, "identities": {}}, scene_only)
    assert [event["name"] for event, _ in events] == ["sits"]
    # With no names known at all, nothing is rejected for want of them.
    _, events, _ = resolve_declarations(declarations, {**BASELINE, "identities": {}}, INFO)
    assert [event["name"] for event, _ in events] == ["sits", "stands"]


def test_room_surfaces_are_legal_allowance_operands():
    """`room.floor` resolved as an event operand and validated nowhere as an
    allowance; `room_0` was the reverse. It now validates in both."""
    declarations = {"collision_allowances": [
        {"a": "x/p", "b": "room.floor", "reason": "rests"}]}
    _, _, errors = resolve_declarations(declarations, BASELINE, INFO)
    assert errors == []
    _, _, errors = resolve_declarations(declarations, {**BASELINE, "room_objects": []}, INFO)
    assert len(errors) == 1


def test_only_an_unreadable_blob_still_raises():
    with pytest.raises(ValueError):
        resolve_declarations([], BASELINE, INFO)
    with pytest.raises(ValueError):
        resolve_declarations({"units": "fortnights"}, BASELINE, INFO)
    assert validate_declarations({"units": "fortnights"}, BASELINE, INFO)[0]["kind"] == "declarations"


# --- every JointLimits finding carries its repair -------------------------

def test_each_joint_finding_ships_a_repair_and_the_limit_case_quotes_numbers():
    rows = ([{"frame": f, "object": "link/dog__back_paw_L",
              "reason": "original mechanism frame changed"} for f in range(3)]
            + [{"frame": 5, "joint": "arm__elbow", "q": 2.4, "limits": [0.0, 2.0],
                "reason": "joint limit or motion axis changed"}]
            + [{"frame": 6, "joint": "arm__wrist", "reason": "nonfinite or invalid evaluated joint q"}])
    findings = {finding["reason"]: finding for finding in joint_findings(rows)}
    assert "CopyTransforms" in findings["original mechanism frame changed"]["repair"]
    limit = findings["joint limit or motion axis changed"]["repair"]
    assert "arm__elbow" in limit and "2.4" in limit and "[0.0, 2.0]" in limit
    assert "key_joint" in findings["nonfinite or invalid evaluated joint q"]["repair"]


# --- the shift is named for what it is, and depth ranks separately -------

def _interval(a, b, start, end, shift, depth):
    return {"a": a, "b": b, "kind": "surface_crossing", "start_frame": start,
            "end_frame": end, "representative_frame": start,
            "directional_separation_m": shift, "separation_direction": [0, 0, 1],
            "max_depth_m": depth}


def test_pairs_keep_duration_order_and_depth_gets_its_own_list():
    pairs = penetration_pairs([_interval("g1/shoulder", "g1/torso", 0, 199, .11, .004),
                               _interval("g1/hand", "tray/rim", 10, 12, .02, .06)])
    assert [pair["a"] for pair in pairs] == ["g1/shoulder", "g1/hand"]
    shoulder = pairs[0]
    assert shoulder["worst_separating_shift_m"] == shoulder["spans"][0]["separating_shift_m"] == .11
    assert "worst_penetration_m" not in shoulder and "penetration_m" not in shoulder["spans"][0]
    assert shoulder["max_depth_m"] == .004
    assert [row["a"] for row in pairs_by_depth(pairs)] == ["g1/hand", "g1/shoulder"]


def test_a_pair_without_a_depth_is_listed_as_unresolved_never_dropped_or_zeroed():
    """An unmeasured depth is listed, never dropped: it can be a deep overlap."""
    pairs = penetration_pairs([_interval("a/x", "b/y", 0, 3, .01, None),
                               _interval("c/x", "d/y", 0, 1, .02, .01)])
    assert "max_depth_m" not in next(p for p in pairs if p["a"] == "a/x")
    assert pairs_by_depth(pairs) == [
        {"a": "c/x", "b": "d/y", "max_depth_m": .01},
        {"a": "a/x", "b": "b/y", "max_depth_m": None, "unresolved": True}]


def test_a_span_reports_the_frame_its_depth_was_measured_on():
    """Depth is the deepest over several frames now, so it says which one."""
    interval = {**_interval("a/x", "b/y", 800, 808, .16, .0778), "max_depth_frame": 806}
    span, = penetration_pairs([interval])[0]["spans"]
    assert span["max_depth_m"] == .0778 and span["max_depth_frame"] == 806
    assert span["representative_frame"] == 800


# --- inherited exemptions are grouped and measured -------------------------

def test_a_complete_part_graph_is_named_and_an_adjacency_list_is_not():
    catalog = {"service": {"parts": [{"name": n} for n in "abcd"]},
               "robot": {"parts": [{"name": n} for n in "abcdefgh"]}}
    evidence = ([{"a": x, "b": y, "instance": "service_0", "object_ref": "service"}
                 for i, x in enumerate("abcd") for y in "abcd"[i + 1:]]
                + [{"a": x, "b": y, "instance": f"robot_{k}", "object_ref": "robot"}
                   for k in range(2) for x, y in zip("abcdefg", "bcdefgh")])
    out = inherited_allowances(evidence, catalog)
    by_ref = {row["object_ref"]: row for row in out["groups"]}
    assert by_ref["service"]["fraction_of_all_pairs"] == 1.0
    assert by_ref["service"]["exempts_every_pair"] is True
    assert by_ref["robot"]["pairs_exempted"] == 7 and by_ref["robot"]["possible_pairs"] == 28
    assert by_ref["robot"]["instances"] == ["robot_0", "robot_1"]
    assert "exempts_every_pair" not in by_ref["robot"]
    assert out["objects_exempting_every_pair"] == ["service"]


# --- Blender interpreter: depth and the streamed end-to-end check ----------

def _run_in_blender(tmp_path, pkgroot, script):
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip("bpy interpreter unavailable")
    source = tmp_path / "case.py"
    source.write_text("import sys\nsys.path.insert(0, " + repr(str(pkgroot)) + ")\n" + script)
    env = dict(os.environ, DYCRAFT_COAL_PYTHON=os.environ.get("DYCRAFT_COAL_PYTHON")
               or sys.executable)
    result = subprocess.run([interpreter, str(source)], capture_output=True, text=True,
                            timeout=120, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


_CUBE = r'''
import bpy  # makes `mathutils` importable
import numpy as np
v = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
t = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
'''


def test_depth_is_the_deepest_vertex_inside_the_other_mesh(tmp_path, pkgroot):
    _run_in_blender(tmp_path, pkgroot, _CUBE + r'''
from dycraft.animation.blender.collision import Body, GeometryCache
from dycraft.animation.blender.verify import deepest_vertex_m
a = Body(v, t)
# A corner pushed 0.1 into the cube along x: depth 0.1, whatever the shift.
assert abs(deepest_vertex_m(a, Body(v + [.9, .3, .3], t)) - .1) < 1e-6  # mathutils BVH is float32
# Containment: the small cube spans [.3, .5]^3, so its deepest vertex, the
# corner at (.5, .5, .5), is 0.5 from the big cube's skin.
assert abs(deepest_vertex_m(a, Body(v * .2 + .3, t)) - .5) < 1e-6  # mathutils BVH is float32
# Placed bodies measure the same as world ones.
cache = GeometryCache()
m = np.eye(4); m[:3, 3] = [.9, .3, .3]
placed = cache.placed(v, t, m)
assert abs(deepest_vertex_m(cache.placed(v, t, np.eye(4)), placed) - .1) < 1e-6  # mathutils BVH is float32
# Open sheets have no inside, so no depth.
sheet = Body(np.array([[-1, -1, .5], [2, -1, .5], [0, 2, .5]]), np.array([[0, 1, 2]]))
cross = Body(np.array([[.5, -1, -1], [.5, 2, -1], [.5, 0, 2]]), np.array([[0, 1, 2]]))
assert deepest_vertex_m(sheet, cross) is None
''')


def test_open_air_floor_resolves_and_bad_declarations_do_not_abort(tmp_path, pkgroot):
    """End to end: an `open_air` room (floor only, no room box), an event on
    `room.floor`, one malformed allowance, and one malformed event. The check
    returns verdicts rather than a traceback, and the event resolves."""
    out = _run_in_blender(tmp_path, pkgroot, _CUBE + r'''
import json
from dycraft.animation.blender.verify import verify
floor_v = v * [4, 4, .1] + [-2, -2, -.1]
record = {"frame": 0, "transforms": {}, "objects": [
    {"name": "box", "identity": "x/p", "matrix": np.eye(4).tolist()},
    {"name": "room_0", "room": True, "matrix": np.eye(4).tolist()}]}
declarations = {"units": "frames",
    "collision_allowances": [{"a": "x/p", "b": "nope/q", "reason": "typo"}],
    "events": [{"name": "rests", "predicate": "touch", "a": "x/p", "b": "room.floor",
                "window": [0, 0]},
               {"name": "bad", "predicate": "touch", "a": "x/p", "window": [0, 0]}]}
info = {"fps": 1, "n_frames": 1, "scene_ref": "s", "declarations": declarations}
before = json.dumps(declarations, sort_keys=True)
reports = verify(".", {"identities": {"x/p": {"link": "none"}}, "joints": {},
                       "room_objects": ["room_0"]},
                 {"id": "s", "instances": []}, {}, info=info,
                 records=iter([(record, {"box": (v, t), "room_0": (floor_v, t)})]))
assert json.dumps(declarations, sort_keys=True) == before
print(json.dumps({r["name"]: r for r in reports}))
''')
    reports = json.loads(out.strip().splitlines()[-1])
    events = {row["name"]: row for row in reports["EventsSatisfied"]["evidence"]["events"]}
    assert events["rests"]["status"] == "satisfied"
    assert events["bad"]["status"] == "invalid" and events["bad"]["ok"] is False
    assert reports["EventsSatisfied"]["verdict"] == "fail"
    penetration = reports["NoPenetration"]
    assert penetration["verdict"] == "pass"
    assert penetration["evidence"]["invalid_allowances"][0]["index"] == 0
    assert penetration["suggested_repairs"]
    assert penetration["evidence"]["containment"] == {
        "checked_surfaces": [], "unchecked": list(ROOM_SURFACE_TAGS)}


# --- one rigid motion is one row, however many parts it carried ------------

def _rigid_sequence(fps=30):
    """A head that snaps a quarter turn at frame 5, carrying two attached parts,
    and an unattached cup that teleports on the same frame."""
    from scipy.spatial.transform import Rotation
    from dycraft.animation.blender.continuity import MotionContinuity
    check = MotionContinuity(fps)
    tetra = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1]])
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    offsets = {"head": np.eye(4), "brows": np.eye(4), "cap_brim": np.eye(4)}
    offsets["brows"][:3, 3] = [.05, .1, .12]
    offsets["cap_brim"][:3, 3] = [-.02, .08, .2]
    sizes = {"head": 3.0, "brows": .5, "cap_brim": .8}
    for frame in range(11):
        head = np.eye(4)
        head[:3, :3] = Rotation.from_euler("z", 90 if frame >= 5 else 0,
                                           degrees=True).as_matrix()
        head[:3, 3] = [1., 2., 1.5]
        objects, geometry = [], {}
        for name, offset in offsets.items():
            matrix = head @ offset
            objects.append({"name": name, "matrix": matrix.tolist()})
            local = tetra * sizes[name]
            geometry[name] = (local @ matrix[:3, :3].T + matrix[:3, 3], tris)
        cup = np.eye(4)
        cup[0, 3] = 3. if frame >= 5 else 0.
        objects.append({"name": "cup", "matrix": cup.tolist()})
        geometry["cup"] = (tetra + cup[:3, 3], tris)
        check.add(frame, {"objects": objects}, geometry)
    return check.report()


def test_parts_carried_by_one_rigid_motion_collapse_into_its_row():
    report = _rigid_sequence()
    rows = report["evidence"]["spikes"]
    rotation = [row for row in rows if row["channel"] == "rotation"]
    assert len(rotation) == 1
    assert rotation[0]["object"] == "head"
    assert rotation[0]["also"] == ["brows", "cap_brim"]
    # The cup moved on the same frame but not with the head: its own row.
    cup = [row for row in rows if row["object"] == "cup"]
    assert len(cup) == 1 and "also" not in cup[0]
    assert report["evidence"]["rigid_duplicates"] >= 2
    assert report["verdict"] == "uncertain"
    # The headline counts the rows an author reads; the raw total is separate.
    assert report["evidence"]["warning_count"] == len(rows)
    assert report["evidence"]["occurrences_total"] == sum(row["occurrences"] for row in rows)


def test_liquids_name_parts_instances_or_native_objects_and_bad_ones_are_reported():
    """Declared liquids are exempt from penetration checks. A liquid that names nothing is reported where it is
    written, like a bad allowance, and only makes the check stricter."""
    from dycraft.animation.blender.verify import resolve_liquids
    native = {**BASELINE, "allow_native_collision_allowances": True}
    info = {**INFO, "collision_objects": ["Water / tank"]}
    liquids, errors = resolve_liquids(
        {"liquids": ["x/p", "y", {"object": "Water / tank", "reason": "the tank's water"},
                     "z/r", 7]}, native, info)
    assert liquids == {"x/p", "y", "Water / tank"}
    assert [(e["index"], e["kind"]) for e in errors] == [(3, "liquid"), (4, "liquid")]
    assert "names no part" in errors[0]["reason"]
    # Native names need the design's permission, as allowances do.
    assert resolve_liquids({"liquids": ["Water / tank"]}, BASELINE, info)[0] == set()
    # Written-time validation sees them.
    assert [e["kind"] for e in validate_declarations({"liquids": ["z/r"]}, BASELINE, INFO)] == ["liquid"]
    assert resolve_liquids({"liquids": "x/p"}, BASELINE, INFO)[1][0]["kind"] == "liquid"
