"""MotionContinuity's `bone` channel and AnatomicalJointLimits on imported rigs.

A skinned character is one body with no IR joints, so without these a forearm
flipped by 175 degrees in a single frame, or an elbow bent backwards, passes
every check: the body channels see the flip as a centimeter of skin RMS, and
AnatomicalJointLimits has nothing to read.

Pure Python: synthetic rigs, no Blender.
"""
import math

import numpy as np
import pytest

from dycraft.agent.assets.import_gltf import (
    skeleton_anatomy, skeleton_roles)
from dycraft.animation.blender.chunking import merge_rows
from dycraft.animation.blender.continuity import (
    MotionContinuity, joint_rotations, quaternion_matrices)
from dycraft.animation.blender.evaluation import check_replay_frame
from dycraft.animation.blender.verify import (
    ROWS_FORMAT, anatomical_measure, bone_anatomy, bone_anatomy_rows, rig_anatomy_table)


def quat(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    half = math.radians(degrees) / 2
    return [math.cos(half), *(math.sin(half) * axis)]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]


# --- the bone channel -------------------------------------------------------

RIG = {"rig/a/body": {"bones": ["root", "arm"], "parents": [-1, 0]}}


def run_channel(arm_angles, root_angles=None, fps=24):
    check = MotionContinuity(fps)
    for frame, angle in enumerate(arm_angles):
        root = quat((0, 0, 1), root_angles[frame] if root_angles else 0.0)
        arm = qmul(root, quat((1, 0, 0), angle))
        check.add_bones(frame, {"bone_rotations": {"rig/a/body": [root, arm]}}, RIG)
    return check.report()


def test_a_single_frame_limb_flip_is_one_spike_on_that_bone():
    angles = [2.0 * i for i in range(40)]
    angles[20] += 170.0
    report = run_channel(angles)
    assert report["verdict"] == "uncertain"
    rows = report["evidence"]["spikes"]
    assert {r["object"] for r in rows} == {"rig/a/body:arm"}
    assert rows[0]["channel"] == "bone" and rows[0]["units"] == "rad/s"
    assert {20, 21} & {f for lo, hi in rows[0]["frames"] for f in range(lo, hi + 1)}


def test_smooth_fast_motion_and_oscillation_are_not_reported():
    """600 deg/s is below the floor; a 20 degree wag reverses every frame and is
    exactly what the jitter rule would have flagged on a tail."""
    sweep = run_channel([20.0 * i for i in range(40)])
    wag = run_channel([20.0 if i % 2 else -20.0 for i in range(40)])
    assert sweep["verdict"] == "pass" and wag["verdict"] == "pass"


def test_a_joint_is_measured_relative_to_its_parent():
    """The whole rig turning (the root) is not the arm's own motion."""
    root = [3.0 * i for i in range(40)]
    root[20] += 120.0
    report = run_channel([0.0] * 40, root_angles=root)
    objects = {r["object"] for r in report["evidence"].get("spikes", [])}
    assert "rig/a/body:arm" not in objects and "rig/a/body:root" in objects


def test_joint_rotations_undo_the_parent():
    root, arm = quat((0, 0, 1), 90), qmul(quat((0, 0, 1), 90), quat((1, 0, 0), 30))
    local = joint_rotations([root, arm], [-1, 0])
    assert np.allclose(local[1], quaternion_matrices([quat((1, 0, 0), 30)])[0], atol=1e-9)


def test_a_rig_disagreeing_with_its_table_is_skipped():
    check = MotionContinuity(24)
    for frame in range(5):
        check.add_bones(frame, {"bone_rotations": {"rig/a/body": [quat((1, 0, 0), 0)]}}, RIG)
    assert check.report()["evidence"]["transitions_checked"] == 0


# --- skeleton families --------------------------------------------------------

def t_pose(names):
    """A standing 1.7 m figure in T-pose, bone axes = armature axes.

    `names` maps a canonical role (and side) to the bone name the family uses.
    Bones point along their own +Y, as Blender's do, so each bone's rest
    rotation is the rotation taking +Y to its direction.
    """
    def rot_to(direction):
        y = np.array([0.0, 1.0, 0.0])
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        v = np.cross(y, d)
        c = float(y @ d)
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + k + k @ k * (1 / (1 + c))
    joints = {}
    for side, s in (("L", 1.0), ("R", -1.0)):
        # +X is the figure's left (viewer-relative axis convention).
        joints.update({
            ("upper_arm", side): ((0.18 * s, 0, 1.42), (0.46 * s, 0, 1.42)),
            ("forearm", side): ((0.46 * s, 0, 1.42), (0.72 * s, 0, 1.42)),
            ("hand", side): ((0.72 * s, 0, 1.42), (0.90 * s, 0, 1.42)),
            ("thigh", side): ((0.10 * s, 0, 0.92), (0.10 * s, 0, 0.50)),
            ("shin", side): ((0.10 * s, 0, 0.50), (0.10 * s, 0, 0.08)),
            ("foot", side): ((0.10 * s, 0, 0.08), (0.10 * s, 0.16, 0.0)),
        })
    joints[("neck", None)] = ((0, 0, 1.45), (0, 0, 1.55))
    joints[("head", None)] = ((0, 0, 1.55), (0, 0, 1.70))
    bones = []
    for key, (head, tail) in joints.items():
        bones.append({"name": names(*key), "parent": None, "head": list(head),
                      "tail": list(tail),
                      "rotation": rot_to(np.subtract(tail, head)).tolist()})
    return bones


BIPED = lambda role, side: "Bip01 " + (f"{side} " if side else "") + {
    "upper_arm": "UpperArm", "forearm": "Forearm", "hand": "Hand", "thigh": "Thigh",
    "shin": "Calf", "foot": "Foot", "neck": "Neck", "head": "Head"}[role]
MIXAMO = lambda role, side: "mixamorig:" + ({"L": "Left", "R": "Right"}.get(side, "")) + {
    "upper_arm": "Arm", "forearm": "ForeArm", "hand": "Hand", "thigh": "UpLeg",
    "shin": "Leg", "foot": "Foot", "neck": "Neck", "head": "Head"}[role] + "_12"
UNREAL = lambda role, side: {
    "upper_arm": "upperarm", "forearm": "lowerarm", "hand": "hand", "thigh": "upperleg",
    "shin": "lowerleg", "foot": "foot", "neck": "neck", "head": "head"}[role] + (
    f"_{side.lower()}" if side else "") + "_024"


_LIMB = ("upper_arm", "forearm", "hand", "thigh", "shin", "foot")


def _family(words, sided, middle=None):
    """A namer from a family's six limb words, how it attaches a side, and its neck and head names."""
    table = dict(zip(_LIMB, words))
    return lambda role, side: sided(table[role], side) if side else middle[role]


VALVE = _family(("UpperArm", "Forearm", "Hand", "Thigh", "Calf", "Foot"),
                lambda w, s: f"ValveBiped.Bip01_{s}_{w}",
                {"neck": "ValveBiped.Bip01_Neck1", "head": "ValveBiped.Bip01_Head1"})
BLENDER = _family(("upper_arm", "forearm", "hand", "thigh", "shin", "foot"),
                  lambda w, s: f"{w}.{s}", {"neck": "neck", "head": "head"})
BLENDER_CAPS = _family(("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot"),
                       lambda w, s: f"{w}.{s}", {"neck": "Neck", "head": "Head"})
RIGIFY = _family(("upper_arm", "forearm", "hand", "thigh", "shin", "foot"),
                 lambda w, s: f"DEF-{w}.{s}", {"neck": "DEF-spine.004", "head": "DEF-spine.006"})
CHARACTER_CREATOR = _family(("Upperarm", "Forearm", "Hand", "Thigh", "Calf", "Foot"),
                            lambda w, s: f"CC_Base_{s}_{w}", {"neck": "CC_Base_NeckTwist01", "head": "CC_Base_Head"})
SOURCE2 = _family(("upperArm", "lowerArm", "hand", "hip", "knee", "foot"),
                  lambda w, s: f"bip_{w}_{s}", {"neck": "bip_neck", "head": "bip_head"})
ADVANCED_SKELETON = _family(("Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle"),
                            lambda w, s: f"{w}_{s}", {"neck": "Neck_M", "head": "Head_M"})
MMD = _family(("arm", "elbow", "wrist", "leg", "knee", "ankle"),
              lambda w, s: f"{'Left' if s == 'L' else 'Right'} {w}", {"neck": "Neck", "head": "Head"})
BN = _family(("Arm_01", "Arm_02", "Arm_03", "Leg_01", "Leg_02", "Leg_03"),
             lambda w, s: f"{s}_{w}_Bn", {"neck": "Neck_Bn", "head": "Head_Bn"})
JNT = _family(("Shoulder", "Elbow", "Wrist", "UpperLeg", "Knee", "Ankle"),
              lambda w, s: f"{s}_{w}_jnt", {"neck": "Neck_jnt", "head": "Head_jnt"})
UNITY = _family(("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot"),
                lambda w, s: f"{'Left' if s == 'L' else 'Right'}{w}", {"neck": "Neck", "head": "Head"})
VROID = _family(("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot"),
                lambda w, s: f"J_Bip_{s}_{w}", {"neck": "J_Bip_C_Neck", "head": "J_Bip_C_Head"})


@pytest.mark.parametrize("family,names", [
    ("biped", BIPED), ("mixamo", MIXAMO), ("unreal", UNREAL), ("biped", VALVE), ("blender", BLENDER),
    ("blender", BLENDER_CAPS), ("rigify", RIGIFY), ("character_creator", CHARACTER_CREATOR),
    ("source2", SOURCE2), ("advanced_skeleton", ADVANCED_SKELETON), ("mmd", MMD), ("bn", BN), ("jnt", JNT),
    ("unity", UNITY), ("unity", VROID)])
def test_each_family_is_recognised_and_gets_all_nine_joints(family, names):
    bones = t_pose(names)
    found, roles = skeleton_roles(bones)
    assert found == family and len(roles) == 14
    anatomy = skeleton_anatomy(bones)
    assert anatomy["skeleton"] == family
    assert sorted(j["name"] for j in anatomy["joints"]) == sorted(
        ["elbow_L", "elbow_R", "knee_L", "knee_R", "wrist_L", "wrist_R",
         "ankle_L", "ankle_R", "head"])
    assert all(abs(j["rest_flexion_deg"]) < 1e-6 for j in anatomy["joints"] if j["kind"] == "hinge")


def test_a_biped_named_bird_gets_no_human_limits_unless_asked():
    """The library's crow is a 3ds Max Biped. Same names, a bird's proportions."""
    bones = t_pose(BIPED)
    for b in bones:  # squash the legs: hips at a fifth of the height
        for end in ("head", "tail"):
            if b["name"].split()[-1] in ("Thigh", "Calf", "Foot"):
                b[end][2] *= 0.25
    auto = skeleton_anatomy(bones)
    assert auto["skeleton"] == "biped" and auto["joints"] == []
    assert "standing human" in auto["reason"]
    assert len(skeleton_anatomy(bones, "human")["joints"]) == 9
    assert skeleton_anatomy(bones, "none")["joints"] == []


def test_a_four_legged_figure_on_a_humanoid_skeleton_gets_no_human_limits():
    """An ibex auto-rigged with Mixamo's skeleton: head high, hind legs half its height, so the head, hip, and
    leg proportions pass; but its front legs hang from shoulders level with its hips."""
    bones = t_pose(MIXAMO)
    hip = next(b for b in bones if b["name"] == MIXAMO("thigh", "L"))["head"][2]
    for b in bones:
        if any(b["name"] == MIXAMO(r, s) for r in ("upper_arm", "forearm", "hand") for s in "LR"):
            for end in ("head", "tail"):
                b[end][2] = hip
    auto = skeleton_anatomy(bones)
    assert auto["skeleton"] == "mixamo" and auto["joints"] == []
    assert "torso does not stand" in auto["reason"]


def test_rigify_maps_its_deform_bones_and_takes_the_first_segment():
    """A generated Rigify rig splits each limb into `DEF-upper_arm.L` and `.L.001` and carries ORG-, MCH-, and
    control bones of the same words; only the deform bones count, and the unsuffixed segment is the bone."""
    bones = t_pose(RIGIFY)
    extra = []
    for b in bones:
        if b["name"].startswith("DEF-") and b["name"][-2:] in (".L", ".R"):
            word = b["name"][4:]
            for name in (b["name"] + ".001", "ORG-" + word, "MCH-" + word + "_parent", word.replace(".", "_fk.")):
                extra.append(dict(b, name=name))
    family, roles = skeleton_roles(bones + extra)
    assert family == "rigify" and len(roles) == 14
    assert roles[("upper_arm", "L")] == "DEF-upper_arm.L" and roles[("head", None)] == "DEF-spine.006"


def test_a_joint_named_rig_keeps_only_the_chains_its_parents_form():
    """AdvancedSkeleton names the upper arm `Shoulder_L`, but `Shoulder_L` is the clavicle in rigs that name
    their arms otherwise; a mapping holds only if the elbow hangs from it and the wrist from the elbow."""
    bones = t_pose(ADVANCED_SKELETON)
    parent = {"Elbow_L": "Shoulder_L", "Wrist_L": "Elbow_L", "Knee_L": "Hip_L", "Ankle_L": "Knee_L",
              "Elbow_R": "UpperArm_R", "Wrist_R": "Elbow_R", "Knee_R": "Hip_R", "Ankle_R": "Knee_R"}
    for b in bones:
        b["parent"] = parent.get(b["name"], "Root")
    family, roles = skeleton_roles(bones)
    assert family == "advanced_skeleton"
    assert ("forearm", "R") not in roles and ("hand", "R") not in roles, "R elbow hangs from another bone"
    assert roles[("forearm", "L")] == "Elbow_L" and roles[("foot", "R")] == "Ankle_R"


def test_a_biped_with_two_neck_bones_keeps_its_first():
    """3ds Max Bipeds often carry `Bip01 Neck` and `Bip01 Neck1`; only Valve's rigs name the neck `Neck1`."""
    bones = t_pose(BIPED)
    bones.append(dict(next(b for b in bones if b["name"] == "Bip01 Neck"), name="Bip01 Neck1"))
    assert skeleton_roles(bones)[1][("neck", None)] == "Bip01 Neck"


def test_an_unknown_skeleton_declares_nothing():
    bones = [{"name": n, "parent": None, "head": [0, 0, 0], "tail": [0, 1, 0],
              "rotation": np.eye(3).tolist()} for n in ("Spine_1", "BackLeg.L_26", "Tail_3")]
    anatomy = skeleton_anatomy(bones)
    assert anatomy["skeleton"] == "unknown" and anatomy["joints"] == []


# --- measuring a pose ----------------------------------------------------------

def elbow_case():
    bones = t_pose(BIPED)
    anatomy = skeleton_anatomy(bones)
    joint = next(j for j in anatomy["joints"] if j["name"] == "elbow_L")
    rest = {b["name"]: np.asarray(b["rotation"]) for b in bones}
    return anatomy, joint, rest


def bend(rest, axis, degrees):
    """The forearm's armature-space rotation after turning it about `axis` (armature)."""
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    r = quaternion_matrices([quat(a, degrees)])[0]
    return r @ rest["Bip01 L Forearm"]


def test_elbow_flexion_is_positive_forward_and_negative_backward():
    _, joint, rest = elbow_case()
    upper = rest["Bip01 L UpperArm"]
    # Left arm along +X; folding the forearm toward the front (+Y) is a turn about +Z.
    forward = anatomical_measure(joint, upper, bend(rest, (0, 0, 1), 90))
    backward = anatomical_measure(joint, upper, bend(rest, (0, 0, 1), -40))
    sideways = anatomical_measure(joint, upper, bend(rest, (0, 1, 0), 50))
    assert forward["flexion"] == pytest.approx(90, abs=1e-6) and abs(forward["off_plane"]) < 1e-6
    assert backward["flexion"] == pytest.approx(-40, abs=1e-6)
    assert abs(sideways["off_plane"]) == pytest.approx(50, abs=1e-6)


def test_the_measure_follows_the_parent_bone():
    """Turning the whole arm (upper arm and forearm together) changes nothing."""
    _, joint, rest = elbow_case()
    turn = quaternion_matrices([quat((1, 1, 0), 70)])[0]
    bent = bend(rest, (0, 0, 1), 60)
    a = anatomical_measure(joint, rest["Bip01 L UpperArm"], bent)
    b = anatomical_measure(joint, turn @ rest["Bip01 L UpperArm"], turn @ bent)
    assert a["flexion"] == pytest.approx(b["flexion"], abs=1e-6)


def test_backward_bend_is_reported_once_per_joint_with_its_frames():
    anatomy, _, rest = elbow_case()
    names = [b for b in rest]
    rigs = {"rig/t/body": {"bones": names, "parents": [-1] * len(names)}}
    measured = []
    def q_of(m):
        w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
        return [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]
    for frame in range(10):
        pose = dict(rest)
        if 3 <= frame <= 5:
            pose["Bip01 L Forearm"] = bend(rest, (0, 0, 1), -50)
        rows = [q_of(pose[n]) for n in names]
        measured += bone_anatomy(frame, {"bone_rotations": {"rig/t/body": rows}}, rigs,
                                 {"rig/t/body": anatomy})
    assert {row["frame"] for row in measured} == {3, 4, 5}
    rows = bone_anatomy_rows(measured, 24)
    assert [(r["joint"], r["measure"], r["reason"]) for r in rows] == [("elbow_L", "flexion", "bent backwards")]
    assert rows[0]["frames"] == [[3, 5]] and rows[0]["worst_deg"] == pytest.approx(-50, abs=1e-3)


def test_the_table_names_each_placed_characters_armature():
    catalog = {"ref": {"native_rig": {"part": "body", "anatomy": {"joints": [{"name": "knee_L"}]}}},
               "plain": {"native_rig": {"part": "body", "anatomy": {"joints": []}}}}
    scene = {"instances": [{"name": "trainer", "object_ref": "ref"},
                           {"name": "statue", "object_ref": "plain"}]}
    assert list(rig_anatomy_table(scene, catalog)) == ["rig/trainer/body"]


# --- chunked checks and replay ---------------------------------------------------

def test_bone_findings_from_two_chunks_merge_into_one_advisory_row():
    """Per-frame rows concatenate through the real chunk merge, and grouping
    the merged rows gives what one pass over all the frames gives."""
    def chunk(owned, frames, values):
        rows = [{"frame": f, "rig": "rig/t/body", "joint": "elbow_R", "measure": "flexion",
                 "reason": "bent backwards", "value_deg": v, "limits_deg": [-10.0, 160.0]}
                for f, v in zip(frames, values)]
        return {"format": ROWS_FORMAT, "frames": list(range(*owned)), "context": {"fps": 24},
                "stats": {}, "rows": {"AnatomicalJointLimits": rows}}
    first = chunk((0, 100), range(90, 100), [-40.0 if f == 95 else -20.0 for f in range(90, 100)])
    second = chunk((100, 200), range(100, 121), [-120.0 if f == 110 else -30.0 for f in range(100, 121)])
    merged = merge_rows([second, first])["rows"]["AnatomicalJointLimits"]
    rows = bone_anatomy_rows(merged, 24)
    assert len(rows) == 1 and rows[0]["frames"] == [[90, 120]]
    assert rows[0]["frame_count"] == 31 and rows[0]["worst_deg"] == -120.0
    assert rows[0]["worst_frame"] == 110
    whole = first["rows"]["AnatomicalJointLimits"] + second["rows"]["AnatomicalJointLimits"]
    assert bone_anatomy_rows(whole, 24) == rows


def test_replay_against_a_cache_without_bones_ignores_them(tmp_path):
    from dycraft.animation.blender.evaluation import frame_signature, write_record
    base = {"frame": 0, "objects": [], "transforms": {"a": np.eye(4).tolist()},
            "joint_positions": {}}
    base["signature"] = frame_signature(base)
    write_record(tmp_path, base)
    fresh = dict(base, bone_rotations={"rig/t/body": [[1.0, 0.0, 0.0, 0.0]]})
    check_replay_frame(tmp_path, 0, fresh, {})


def test_a_neck_and_head_alone_are_not_a_skeleton_family():
    """A labrador rig names `Neck1_14` and `Head_1`, and nothing else a family knows."""
    bones = [{"name": n, "parent": None, "head": [0, 0, 0.5], "tail": [0, 0.1, 0.5],
              "rotation": np.eye(3).tolist()}
             for n in ("Neck1_14", "Head_1", "FrontUpperLeg.L_17", "BackLeg.L_26")]
    assert skeleton_roles(bones) == (None, {})
    assert skeleton_anatomy(bones)["skeleton"] == "unknown"
