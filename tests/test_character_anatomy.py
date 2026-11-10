"""AnatomicalJointLimits on an imported person's whole body, and how its rig bends.

The whole-body measures check shoulders, hips, forearm rotation, wrist
direction and the head's turn, and read elbows and knees so that a rolled arm
is not "bent backwards". These tests pin the integration and the meaning of
each finding on a synthetic rig with a real hierarchy.

Pure Python: no Blender.
"""
import math

import numpy as np
import pytest

from dycraft.agent.assets.import_gltf import skeleton_anatomy
from dycraft.animation.blender.continuity import quaternion_matrices
from dycraft.animation.blender.verify import (bone_anatomy, bone_anatomy_rows, body_findings, body_measures,
                                              current_anatomy)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rotation(axis, degrees):
    a = unit(axis)
    half = math.radians(degrees) / 2
    return quaternion_matrices([[math.cos(half), *(math.sin(half) * a)]])[0]


def frame_along(direction, x_axis):
    """Columns (X, Y, Z) with Y along the bone, X as given (made orthogonal)."""
    y = unit(direction)
    x = unit(np.asarray(x_axis, float) - y * (y @ x_axis))
    return np.column_stack([x, y, np.cross(x, y)])


# The figure faces +Y, z up, its left at -X; T-pose, palms down (-Z).
def figure(family):
    """`(bones, names)`: a 1.7 m figure with a trunk, arms, legs, and four three-bone fingers.

    `family` picks both the bone names and the finger bones' local axes: a
    Mixamo finger curls toward the palm about its +X, a Biped's about its -Z.
    """
    def name(role, side=None):
        S = {"L": "Left", "R": "Right"}.get(side, "")
        if family == "mixamo":
            table = {"pelvis": "Hips", "spine": "Spine", "chest": "Spine2", "neck": "Neck",
                     "head": "Head", "upper_arm": S + "Arm", "forearm": S + "ForeArm",
                     "hand": S + "Hand", "thigh": S + "UpLeg", "shin": S + "Leg", "foot": S + "Foot"}
            if role in table:
                return "mixamorig:" + table[role]
            digit, k = role.split("_")
            return f"mixamorig:{S}Hand{digit.title()}{k}"
        sd = f"{side} " if side else ""
        table = {"pelvis": "Pelvis", "spine": "Spine", "chest": "Spine2", "neck": "Neck",
                 "head": "Head", "upper_arm": sd + "UpperArm", "forearm": sd + "Forearm",
                 "hand": sd + "Hand", "thigh": sd + "Thigh", "shin": sd + "Calf", "foot": sd + "Foot"}
        if role in table:
            return "Bip01 " + table[role]
        digit, k = role.split("_")
        d = ("thumb", "index", "middle", "ring", "pinky").index(digit)
        return f"Bip01 {sd}Finger{d}{'' if k == '1' else int(k) - 1}"

    bones = []
    def add(role, side, head, tail, parent, x_axis=(1, 0, 0)):
        bones.append({"name": name(role, side), "parent": parent and name(*parent),
                      "head": list(map(float, head)), "tail": list(map(float, tail)),
                      "deform": True,
                      "rotation": frame_along(np.subtract(tail, head), x_axis).tolist()})
    # A bone along +Z or -Z keeps +X as its X axis; one along x keeps +Z... as given.
    add("pelvis", None, (0, 0, 0.92), (0, 0, 1.02), None)
    add("spine", None, (0, 0, 1.02), (0, 0, 1.20), ("pelvis",))
    add("chest", None, (0, 0, 1.20), (0, 0, 1.42), ("spine",))
    add("neck", None, (0, 0, 1.45), (0, 0, 1.55), ("chest",))
    add("head", None, (0, 0, 1.55), (0, 0, 1.70), ("neck",))
    for side, s in (("L", -1.0), ("R", 1.0)):
        add("upper_arm", side, (0.18 * s, 0, 1.42), (0.46 * s, 0, 1.42), ("chest",), (0, 0, 1))
        add("forearm", side, (0.46 * s, 0, 1.42), (0.72 * s, 0, 1.42), ("upper_arm", side), (0, 0, 1))
        add("hand", side, (0.72 * s, 0, 1.42), (0.80 * s, 0, 1.42), ("forearm", side), (0, 0, 1))
        # Curl toward the palm (-Z) is a turn about d x (-Z) = -Y on the left, +Y on the right.
        curl = np.array([0.0, -1.0, 0.0]) * (1 if side == "L" else -1)
        for digit, y in (("index", 0.03), ("middle", 0.01), ("ring", -0.01), ("pinky", -0.03)):
            x = 0.80
            for k in (1, 2, 3):
                head, tail = (x * s, y, 1.42), ((x + 0.03) * s, y, 1.42)
                x_axis = curl if family == "mixamo" else np.cross(curl, unit((s, 0, 0)))
                parent = ("hand", side) if k == 1 else (f"{digit}_{k - 1}", side)
                add(f"{digit}_{k}", side, head, tail, parent, x_axis)
                x += 0.03
        add("thigh", side, (0.10 * s, 0, 0.92), (0.10 * s, 0, 0.50), ("pelvis",))
        add("shin", side, (0.10 * s, 0, 0.50), (0.10 * s, 0, 0.08), ("thigh", side))
        add("foot", side, (0.10 * s, 0, 0.08), (0.10 * s, 0.16, 0.0), ("shin", side))
        bones.append({"name": name("foot", side) + "_toe", "parent": name("foot", side),
                      "head": [0.10 * s, 0.16, 0.0], "tail": [0.10 * s, 0.22, 0.0], "deform": True,
                      "rotation": frame_along((0, 1, 0), (1, 0, 0)).tolist()})
    return bones, name


def pose(bones, local):
    """Armature-space rotations after turning bones by `local` {name: R in armature axes at rest}.

    Each turn is about armature axes through the bone's joint, applied to the
    bone and carried by its descendants, as rotating it in Blender does.
    """
    rest = {b["name"]: np.asarray(b["rotation"]) for b in bones}
    parent = {b["name"]: b["parent"] for b in bones}
    posed = {}
    for b in bones:  # parents precede children in `figure`
        n = b["name"]
        carried = (posed[parent[n]] @ rest[parent[n]].T) if parent[n] else np.eye(3)
        posed[n] = carried @ local.get(n, np.eye(3)) @ rest[n]
    return posed


def findings_at(bones, anatomy, local):
    posed = pose(bones, local)
    def q_of(m):
        w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
        if w < 1e-6:  # a half turn: take x, y, or z as the large component
            i = int(np.argmax(np.diag(m)))
            v = np.zeros(3)
            v[i] = math.sqrt(max(0.0, 1 + 2 * m[i, i] - np.trace(m))) / 2
            for j in range(3):
                if j != i:
                    v[j] = (m[i, j] + m[j, i]) / (4 * v[i])
            return [0.0, *v]
        return [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                (m[1, 0] - m[0, 1]) / (4 * w)]
    names = [b["name"] for b in bones]
    rigs = {"rig/c/body": {"bones": names}}
    return bone_anatomy(0, {"bone_rotations": {"rig/c/body": [q_of(posed[n]) for n in names]}},
                        rigs, {"rig/c/body": anatomy})


def found(rows):
    return {(r["joint"], r["measure"], r["reason"]) for r in rows}


@pytest.mark.parametrize("family", ["mixamo", "biped"])
def test_the_whole_body_is_declared_with_palms_and_bend_axes(family):
    bones, name = figure(family)
    anatomy = skeleton_anatomy(bones)
    assert anatomy["skeleton"] == family and anatomy["version"] == 2
    assert set(anatomy["body"]) == {f"{j}_{s}" for j in ("shoulder", "hip", "wrist", "ankle") for s in "LR"} | {
        "head", "spine"}
    for side in "LR":
        assert np.allclose(anatomy["hands"][side]["palm_normal"], (0, 0, -1), atol=1e-6)
        for digit in ("index", "middle", "ring", "pinky"):
            assert {f"{digit}_knuckle_{side}", f"{digit}_pip_{side}", f"{digit}_dip_{side}"} <= set(anatomy["bend_axes"])


@pytest.mark.parametrize("family,axis", [("mixamo", (1, 0, 0)), ("biped", (0, 0, -1))])
def test_a_finger_turned_about_its_bend_axis_curls_toward_the_palm(family, axis):
    """The same curl is +X on a Mixamo finger and -Z on a Biped's."""
    bones, name = figure(family)
    anatomy = skeleton_anatomy(bones)
    assert np.allclose(anatomy["bend_axes"]["index_pip_L"], axis, atol=1e-6)
    child = name("index_2", "L")
    rest = {b["name"]: np.asarray(b["rotation"]) for b in bones}
    seg = rest[child][:, 1]
    curled = rest[child] @ rotation(axis, 60) @ np.array([0.0, 1.0, 0.0])
    palm = np.array([0.0, 0.0, -1.0])
    assert (curled - seg) @ palm > 0.5


def test_the_rest_pose_and_working_poses_pass():
    bones, name = figure("mixamo")
    anatomy = skeleton_anatomy(bones)
    assert findings_at(bones, anatomy, {}) == []
    work = {name("upper_arm", "L"): rotation((0, 0, 1), -70),    # the arm swung forward
            name("thigh", "R"): rotation((1, 0, 0), 100),       # a squatting hip
            name("foot", "R"): rotation((1, 0, 0), 30),         # the ankle bent toward the shin
            name("chest", None): rotation((1, 0, 0), -35)}      # leaning forward
    assert findings_at(bones, anatomy, work) == []
    assert findings_at(bones, anatomy, {name("foot", "L"): rotation((1, 0, 0), -70)}) == []  # kneeling on the instep


def test_broken_poses_are_reported():
    bones, name = figure("biped")
    anatomy = skeleton_anatomy(bones)
    rest = {b["name"]: np.asarray(b["rotation"]) for b in bones}
    hand = name("hand", "R")
    cases = {
        # The left arm points -X and its elbow folds toward +Y, so a turn about +Z bends it backwards:
        # nearly straight, that reads from the hinge's flexion; further, as a roll near 180 degrees.
        ("elbow_L", "flexion", "bent backwards"): {name("forearm", "L"): rotation((0, 0, 1), 20)},
        ("shoulder_L", "roll", "rolled outward past its range"): {name("forearm", "L"): rotation((0, 0, 1), 50)},
        ("ankle_L", "flexion", "bent toward the shin past its range"): {name("foot", "L"): rotation((1, 0, 0), 70)},
        # The right palm faces -Z; the hand bent 100 degrees toward +Z is bent back.
        ("wrist_R", "flexion", "bent back past its range"): {hand: rotation((0, 1, 0), -100)},
        ("head", "turn", "turned past its range"): {name("head", None): rotation((0, 0, 1), 150)},
    }
    for expected, local in cases.items():
        assert expected in found(findings_at(bones, anatomy, local)), (expected, found(findings_at(bones, anatomy, local)))


def test_a_rolled_arm_bent_naturally_is_not_bent_backwards():
    """A hinge read in the upper bone's frame alone calls a turned upper arm "off its hinge plane"; with the
    elbow folding the way it bends -- the arm out to the side, turned so the forearm folds up into a goal-post --
    neither the roll nor where the bend sits round the hinge reads it as broken. Turned the other way, the
    same shape bends the elbow against its fold, which renders as a pinched, crumpled elbow (see
    `test_a_limb_bent_backwards_against_its_hinge_is_reported`)."""
    bones, name = figure("mixamo")
    anatomy = skeleton_anatomy(bones)
    up, fore = name("upper_arm", "L"), name("forearm", "L")
    turn = rotation((-1, 0, 0), -90)                    # the left upper arm turned so its elbow folds up (+Z)
    raise_ = _turn_to(np.array([-1.0, 0.0, 0.0]), turn.T @ np.array([0.0, 0.0, 1.0]))   # the forearm straight up
    rows = findings_at(bones, anatomy, {up: turn, fore: raise_})
    assert not [r for r in rows if r["joint"] in ("elbow_L", "shoulder_L")], rows


def test_rows_group_with_open_ended_limits():
    rows = [{"frame": f, "rig": "rig/c/body", "joint": "elbow_L", "measure": "bend", "reason": "over-flexed",
             "value_deg": 160.0 + f, "limits_deg": [None, 155]} for f in (3, 4, 5)]
    out, = bone_anatomy_rows(rows, 24)
    assert out["frames"] == [[3, 5]] and out["worst_deg"] == 165.0


def test_a_rig_imported_before_the_whole_body_check_is_derived_from_its_recorded_rest():
    """A content-addressed IR keeps the five-joint block it froze; the check derives the rest."""
    bones, name = figure("biped")
    fresh = skeleton_anatomy(bones)
    frozen = {"skeleton": "biped", "policy": "auto", "joints": fresh["joints"]}
    names = [b["name"] for b in bones]
    rigs = {"rig/c/body": {"bones": names,
                           "parents": [names.index(b["parent"]) if b["parent"] else -1 for b in bones],
                           "rest": [{k: b[k] for k in ("head", "tail", "deform", "rotation")} for b in bones]}}
    derived = current_anatomy({"rig/c/body": frozen}, rigs)["rig/c/body"]
    assert derived["version"] == fresh["version"] and derived["policy"] == "auto"
    assert set(derived["body"]) == set(fresh["body"]) and "shoulder_L" in derived["body"]
    assert np.allclose(derived["bend_axes"]["index_pip_L"], fresh["bend_axes"]["index_pip_L"])
    hand = name("hand", "R")
    assert ("wrist_R", "flexion", "bent back past its range") in found(
        findings_at(bones, derived, {hand: rotation((0, 1, 0), -100)}))
    # Current anatomy, or an evaluation without rest frames, is kept as stored.
    assert current_anatomy({"rig/c/body": fresh}, rigs)["rig/c/body"] is fresh
    assert current_anatomy({"rig/c/body": frozen}, {"rig/c/body": {"bones": names}})["rig/c/body"] is frozen


def _rolled(bones, seed):
    """Every bone's rest frame turned at random about its own line: the measures must not care."""
    rng = np.random.default_rng(seed)
    out = []
    for b in bones:
        a = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(a), math.sin(a)
        out.append({**b, "rotation": (np.asarray(b["rotation"]) @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])).tolist()})
    return out


def _limb_cases(name):
    """Right-side motions about armature axes (the figure faces +Y, its right at +X, palms down, thumbs
    forward) and the value or sign each measure's documented convention gives them."""
    fa, up, ha, th, sh, ft = (name(r, "R") for r in ("forearm", "upper_arm", "hand", "thigh", "shin", "foot"))
    bent = rotation((0, 0, 1), 90)                      # the forearm folded forward
    return [
        ({ha: rotation((0, 1, 0), 40)}, "wrist_flex_R", 40),          # toward the palm
        ({ha: rotation((0, 1, 0), -40)}, "wrist_flex_R", -40),        # toward the back of the hand
        ({ha: rotation((0, 0, 1), 20)}, "wrist_tilt_R", 20),          # toward the thumb
        ({fa: rotation((1, 0, 0), 60)}, "forearm_rot_R", 60),         # supinated, elbow straight
        ({fa: rotation((1, 0, 0), -60)}, "forearm_rot_R", -60),       # pronated
        ({fa: rotation((0, 1, 0), 40) @ bent}, "forearm_rot_R", 40),  # supinated, elbow bent
        ({fa: rotation((0, 0, 1), -40)}, "forearm_rot_R", None),      # bent backwards: no forearm reading
        ({fa: bent}, "elbow_bend_R", 90),
        ({up: rotation((1, 0, 0), 45), fa: bent}, "shoulder_roll_R", "+"),   # turned out: forearm up
        ({up: rotation((1, 0, 0), -45), fa: bent}, "shoulder_roll_R", "-"),
        ({sh: rotation((1, 0, 0), -90)}, "knee_bend_R", 90),
        ({th: rotation((0, 0, 1), -30), sh: rotation((1, 0, 0), -90)}, "hip_rot_R", "+"),   # turned out
        ({th: rotation((0, 0, 1), 30), sh: rotation((1, 0, 0), -90)}, "hip_rot_R", "-"),
        ({ft: rotation((1, 0, 0), 20)}, "ankle_flex_R", 20),          # toes up
        ({ft: rotation((1, 0, 0), -30)}, "ankle_flex_R", -30),
    ]


def _reads(value, want):
    if want is None:
        return value is None
    return value is not None and (value > 1 if want == "+" else value < -1 if want == "-" else abs(value - want) < 2)


@pytest.mark.parametrize("family", ["mixamo", "biped"])
@pytest.mark.parametrize("roll", [False, True])
def test_each_measure_reads_its_documented_sign_on_either_side(family, roll):
    """Each limb measure gives a known motion its documented sign, and the mirror image on the left side
    the same value; the head's and spine's read nod down, right ear down, turn and twist left as +."""
    bones, name = figure(family)
    bones = _rolled(bones, 7) if roll else bones
    anatomy = skeleton_anatomy(bones)
    M = np.diag([-1.0, 1.0, 1.0])
    for local, key, want in _limb_cases(name):
        right = body_measures(anatomy, pose(bones, local)).get(key)
        mirrored = {b.replace("Right", "Left").replace(" R ", " L "): M @ R @ M for b, R in local.items()}
        left = body_measures(anatomy, pose(bones, mirrored)).get(key[:-1] + "L")
        assert _reads(right, want) and _reads(left, want if want is None else right), (key, want, right, left)
    head, chest = name("head", None), name("chest", None)
    for local, key in (({head: rotation((1, 0, 0), -20)}, "head_nod"), ({head: rotation((0, 1, 0), 20)}, "head_tilt"),
                       ({head: rotation((0, 0, 1), 20)}, "head_axial"), ({chest: rotation((0, 0, 1), 20)}, "spine_axial")):
        assert _reads(body_measures(anatomy, pose(bones, local)).get(key), 20), key


def _turn_to(start, target):
    """The rotation taking unit `start` to unit `target` (neither opposite)."""
    axis = np.cross(start, target)
    return rotation(axis, math.degrees(math.acos(float(np.clip(np.dot(start, target), -1, 1)))))


@pytest.mark.parametrize("family", ["mixamo", "biped"])
@pytest.mark.parametrize("side", ["L", "R"])
def test_a_limb_bent_backwards_against_its_hinge_is_reported(family, side):
    """The arm raised forward with the forearm hanging down is natural when the upper arm is turned so its elbow
    folds downward, and an elbow bent 90 degrees backwards when it is turned the other way. Both read a shoulder
    roll of -90 (the roll adds the upper arm's turn to where the bend sits and wraps), so only where the bend
    sits round the hinge tells them apart. Likewise a knee bent forward instead of back."""
    bones, name = figure(family)
    anatomy = skeleton_anatomy(bones)
    s = 1.0 if side == "R" else -1.0
    M = np.diag([s, 1.0, 1.0])                          # the left side as the mirror image of the right
    up, fore, thigh, shin = (name(r, side) for r in ("upper_arm", "forearm", "thigh", "shin"))
    for roll, broken in ((90, True), (-90, False)):
        Q = M @ rotation((0, 1, 0), roll) @ rotation((0, 0, 1), 90) @ M        # aim forward, then turn the arm
        hang = _turn_to(np.array([s, 0.0, 0.0]), Q.T @ np.array([0.0, 0.0, -1.0]))
        rows = findings_at(bones, anatomy, {up: Q, fore: hang})
        assert ((f"elbow_{side}", "bend direction", "bent backwards") in found(rows)) == broken, (roll, rows)
        if not broken:
            assert not [r for r in rows if r["joint"] in (f"elbow_{side}", f"shoulder_{side}")], rows
    assert (f"knee_{side}", "bend direction", "bent backwards") in found(
        findings_at(bones, anatomy, {shin: rotation((1, 0, 0), 60)}))       # the shin swung forward
    assert not findings_at(bones, anatomy, {thigh: rotation((1, 0, 0), 60), shin: rotation((1, 0, 0), -90)})


def test_roles_written_by_hand_replace_the_names():
    """A rig whose bone names no family maps gets the whole-body anatomy from roles given by hand."""
    bones, name = figure("mixamo")
    rename = {b["name"]: f"bone{k}" for k, b in enumerate(bones)}
    plain = [{**b, "name": rename[b["name"]], "parent": rename.get(b["parent"])} for b in bones]
    assert skeleton_anatomy(plain)["skeleton"] == "unknown"
    roles = {(r, s): rename[name(r, s)] for r in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot",
                                                    "index_1", "middle_1", "pinky_1") for s in "LR"}
    roles.update({(r, None): rename[name(r, None)] for r in ("neck", "head")})
    given = skeleton_anatomy(plain, roles=roles)
    named = skeleton_anatomy(bones)
    assert given["skeleton"] == "given" and set(given["body"]) == set(named["body"])
    assert np.allclose(given["hands"]["L"]["palm_normal"], named["hands"]["L"]["palm_normal"])
