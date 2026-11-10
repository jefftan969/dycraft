"""The dog-capture -> rigify-dog helper agents import inside Blender (`/tools/dog_retarget.py`).

Clip edits act on captured joint positions and are tested with stubbed bpy; the
solve is tested against real Blender on the German shepherd, where the gitignored
library and rig exist. Two failures the solve tests pin: front paws held off the
floor by a lift meant for the rump or the tail, and a sit whose hind legs splay
sideways.
"""
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dycraft.animation.blender.runtime import REPO
from dycraft.render._blender_env import blender_python

LIBRARY = REPO / "assets" / "mocap" / "mann_dog"
RIG = REPO / "assets" / "imported" / "sf_dog_gsd_puppyclub_rig"
NOTES = REPO / "animation" / "mocap" / "libraries" / "mann_dog"


def _module(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setenv("DYCRAFT_DOG_MOCAP", str(LIBRARY))
    spec = importlib.util.spec_from_file_location("dog_retarget_under_test",
                                                  REPO / "animation/mocap/dog_retarget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _needs_library():
    if not (LIBRARY / "data" / "D1_008_KAN01_002.bvh").is_file():
        pytest.skip("needs the dog capture library (assets/mocap/mann_dog)")


def _armature(parents):
    """A stub armature from `{bone: parent}` (a list of names: no parents)."""
    if not isinstance(parents, dict):
        parents = dict.fromkeys(parents)
    bones = {n: SimpleNamespace(name=n, parent=None) for n in parents}
    for n, p in parents.items():
        bones[n].parent = bones.get(p)
    return SimpleNamespace(name="rig", data=SimpleNamespace(bones=list(bones.values())))


def _hierarchy(text):
    """`{child: parent}` from "child<parent child<parent ..." (a parent of '-' is none)."""
    pairs = (item.split("<") for item in text.split())
    return {c: (None if p == "-" else p) for c, p in pairs}


#: The library German shepherd's skeleton (`sf_dog_gsd_puppyclub_rig`, its ir.json `native_rig.bones`), glTF suffixes kept.
SHEPHERD = _hierarchy("""
    GLTF_created_0_rootJoint<- Root_46<GLTF_created_0_rootJoint spine.004_45<Root_46 spine.003_5<spine.004_45
    spine.002_4<spine.003_5 spine.001_3<spine.002_4 spine_2<spine.001_3 spine.005_34<spine.004_45
    spine.006_33<spine.005_34 spine.007_32<spine.006_33 spine.008_31<spine.007_32 spine.009_18<spine.008_31
    spine.010_17<spine.009_18 Head_16<spine.010_17 Ear_R_7<Head_16 EarTip_R.001_6<Ear_R_7 Ear_L_9<Head_16
    EarTip_L.001_8<Ear_L_9 Jaw_11<Head_16 Eye_R_12<Head_16 Eye_L_13<Head_16 shoulder.L_23<spine.008_31
    front_thigh.L_22<shoulder.L_23 front_shin.L_21<front_thigh.L_22 front_foot.L_20<front_shin.L_21
    front_toe.L_19<front_foot.L_20 shoulder.R_28<spine.008_31 front_thigh.R_27<shoulder.R_28
    front_shin.R_26<front_thigh.R_27 front_foot.R_25<front_shin.R_26 front_toe.R_24<front_foot.R_25
    breast.L_29<spine.008_31 breast.R_30<spine.008_31 pelvis.L_35<spine.004_45 pelvis.R_36<spine.004_45
    thigh.L_40<spine.004_45 shin.L_39<thigh.L_40 foot.L_38<shin.L_39 toe.L_37<foot.L_38 thigh.R_44<spine.004_45
    shin.R_43<thigh.R_44 foot.R_42<shin.R_43 toe.R_41<foot.R_42""")

#: The library horse (`horse_realistic_chestnut_horse_with_wh_86d47bdc`): rigify-dog bone names, but the hips are
#: spine.005, spine.004 starts the tail, the thorax is two bones, and the hind hip bones are named shoulder.L/R.
HORSE = _hierarchy("""
    Root<- spine.005<Root shoulder.R<spine.005 thigh.R<shoulder.R shin.R<thigh.R foot.R<shin.R toe.R<foot.R
    spine.004<spine.005 spine.003<spine.004 spine.002<spine.003 spine.001<spine.002 spine<spine.001
    spine_end<spine spine.006<spine.005 spine.007<spine.006 front_shoulder.R<spine.007
    front_thigh.R<front_shoulder.R front_shin.R<front_thigh.R front_foot.R<front_shin.R front_toe.R<front_foot.R
    spine.008<spine.007 spine.009<spine.008 spine.010<spine.009 scull<spine.010 ear.R<scull ear.R_end<ear.R
    ear.L<scull ear.L_end<ear.L front_shoulder.L<spine.007 front_thigh.L<front_shoulder.L
    front_shin.L<front_thigh.L front_foot.L<front_shin.L front_toe.L<front_foot.L shoulder.L<spine.005
    thigh.L<shoulder.L shin.L<thigh.L foot.L<shin.L toe.L<foot.L""")

#: Blender 4.5's Rigify wolf metarig (`rigify/metarigs/Animals/wolf.py`), body and legs: the rigify dog's numbering,
#: with the thighs hanging from spine.005, the lumbar bone under the pelvis spine.004.
WOLF = _hierarchy("""
    spine.004<- spine.003<spine.004 spine.002<spine.003 spine.001<spine.002 spine<spine.001 spine.005<spine.004
    spine.006<spine.005 spine.007<spine.006 spine.008<spine.007 spine.009<spine.008 spine.010<spine.009
    spine.011<spine.010 pelvis.L<spine.005 pelvis.R<spine.005 thigh.L<spine.005 thigh.R<spine.005 shin.L<thigh.L
    shin.R<thigh.R foot.L<shin.L foot.R<shin.R toe.L<foot.L toe.R<foot.R shoulder.L<spine.008 shoulder.R<spine.008
    breast.L<spine.008 breast.R<spine.008 front_thigh.L<shoulder.L front_thigh.R<shoulder.R
    front_shin.L<front_thigh.L front_shin.R<front_thigh.R front_foot.L<front_shin.L front_foot.R<front_shin.R
    front_toe.L<front_foot.L front_toe.R<front_foot.R""")

#: The shepherd without its import suffixes or its head and face bones.
RIGIFY = {n.rsplit("_", 1)[0] if n[-1].isdigit() and "_" in n else n: p for n, p in SHEPHERD.items()
          if not n.startswith(("Ear", "Eye", "Jaw", "Head", "GLTF", "Root", "breast", "pelvis."))}
RIGIFY = {n: (p.rsplit("_", 1)[0] if p and p[-1].isdigit() else p) for n, p in RIGIFY.items()}
RIGIFY["spine.004"] = None


def test_a_rigify_dog_is_recognised_with_or_without_import_suffixes(monkeypatch):
    dr = _module(monkeypatch)
    plain = dr.profile(_armature(RIGIFY))
    suffix = {n: f"{n}_{i}" for i, n in enumerate(RIGIFY)}
    suffixed = dr.profile(_armature({suffix[n]: suffix.get(p) for n, p in RIGIFY.items()}))
    assert plain["legs"]["HL"]["bones"] == ("thigh.L", "shin.L", "foot.L")
    assert suffixed["pelvis"].startswith("spine.004_") and suffixed["legs"]["FR"]["ctrl"].startswith("front_toe.R_")
    assert "spine.011" not in plain["chain"], "a bone the rig lacks is dropped, not invented"


def test_the_library_shepherd_is_mapped_by_name(monkeypatch):
    dr = _module(monkeypatch)
    prof = dr.profile(_armature(SHEPHERD))
    assert prof["pelvis"] == "spine.004_45"
    assert prof["hips"] == ("thigh.L_40", "thigh.R_44") and prof["shoulders"] == ("front_thigh.L_22", "front_thigh.R_27")
    assert set(prof["chain"]) == {"spine.005_34", "spine.006_33", "spine.007_32", "spine.008_31", "spine.009_18",
                                  "spine.010_17", "Head_16", "shoulder.L_23", "shoulder.R_28", "spine.003_5",
                                  "spine.002_4", "spine.001_3", "spine_2"}


def test_the_rigify_wolf_metarig_is_accepted_with_its_thighs_under_the_lumbar_bone(monkeypatch):
    """Blender's own Rigify dog skeleton hangs the thighs from spine.005, directly under the pelvis spine.004; it is a
    rigify dog and must be mapped as one."""
    dr = _module(monkeypatch)
    prof = dr.profile(_armature(WOLF))
    assert prof["pelvis"] == "spine.004" and prof["hips"] == ("thigh.L", "thigh.R")
    assert {"spine.005", "spine.008", "shoulder.L", "spine.003"} <= set(prof["chain"])


def test_a_quadruped_whose_spine_is_numbered_otherwise_is_refused(monkeypatch):
    """By name alone the horse's tail root would be driven as its pelvis (the tail stretched meters behind the
    body) and its hind hip bones as front shoulders; the refusal is a ValueError, which callers fall back on."""
    dr = _module(monkeypatch)
    with pytest.raises(ValueError, match="hind legs hang from spine.005, not from spine.004"):
        dr.profile(_armature(HORSE))


def _misplaced():
    front_on_pelvis = dict(RIGIFY, **{"shoulder.L": "spine.004", "shoulder.R": "spine.004"})
    shoulder_elsewhere = dict(RIGIFY, **{"front_thigh.L": "spine.008"})
    tail_carries_leg = dict(RIGIFY, **{"thigh.L": "spine.002"})
    hind_on_nothing = dict(RIGIFY, **{"thigh.L": None, "thigh.R": None})
    one_hind_leg = {n: p for n, p in RIGIFY.items() if not n.endswith(".R") or not n.startswith(("thigh", "shin",
                                                                                                "foot", "toe"))}
    return {"front legs on the pelvis": front_on_pelvis, "a shoulder that carries no front leg": shoulder_elsewhere,
            "a tail bone that carries a hind leg": tail_carries_leg, "hind legs parented to nothing": hind_on_nothing,
            "no parents at all": list(RIGIFY), "one hind leg": one_hind_leg,
            "a front paw without its toe": {n: p for n, p in RIGIFY.items() if n != "front_toe.R"}}


@pytest.mark.parametrize("case", sorted(_misplaced()))
def test_a_misplaced_role_is_refused_with_a_value_error_only(monkeypatch, case):
    """Every refusal is a ValueError, the one exception callers catch to fall back to keyed poses; any other
    would stop them."""
    dr = _module(monkeypatch)
    with pytest.raises(Exception) as err:
        dr.profile(_armature(_misplaced()[case]))
    assert type(err.value) is ValueError, f"{case}: {type(err.value).__name__}: {err.value}"


def test_a_rig_without_dog_legs_is_refused_by_name(monkeypatch):
    dr = _module(monkeypatch)
    with pytest.raises(ValueError, match="has no rigify dog bone names"):
        dr.profile(_armature(["Hips", "LeftUpLeg", "RightUpLeg", "Spine"]))


def _quadruped(fwd=1.0, hind=None, front=None, front_z=0.0, helpers=False, legs_below_hips=False):
    """A four-legged skeleton as rest bones (name, parent, head, tail), Z up, the head toward +Y (`fwd=1`) or -Y,
    with its own names: a pelvis root carrying the tail, the spine to the chest, the neck, and a head with a jaw
    and ears; hind legs thigh/shin/foot/toe; front legs blade/humerus/radius/metacarpus/toe. `hind` and `front`
    replace a leg's joints as [(y, z), ...] from the top down, `front_z` lifts the front paws, `helpers` adds
    a stacked twin at each knee and a twist joint partway down each shin, and `legs_below_hips` hangs the hind
    legs and the spine from a bone under the skinned pelvis, the tail staying on the pelvis (as on a 3ds Max
    Biped, whose thighs hang from its spine). Returns (bones, roles), the roles naming the bones each leg should
    map to, by side."""
    y = lambda v: fwd * v
    bones, roles = [], {}

    def add(name, parent, head, tail=None):
        head = (head[0], y(head[1]), head[2])
        bones.append(dict(name=name, parent=parent, head=head, tail=tail or (head[0], head[1], head[2] + 0.02)))
        return name

    add("u0", None, (0, -0.30, 0.55))
    add("u1", "u0", (0, -0.35, 0.55)), add("u2", "u1", (0, -0.50, 0.50)), add("u3", "u2", (0, -0.65, 0.45))
    trunk = add("uS", "u0", (0, -0.25, 0.55)) if legs_below_hips else "u0"
    add("u4", trunk, (0, -0.15, 0.56)), add("u5", "u4", (0, 0.05, 0.57)), add("u6", "u5", (0, 0.20, 0.58))
    add("u7", "u6", (0, 0.30, 0.65)), add("u8", "u7", (0, 0.40, 0.75))
    add("u9", "u8", (0, 0.50, 0.70)), add("u10", "u8", (-0.03, 0.38, 0.85)), add("u11", "u8", (0.03, 0.38, 0.85))
    hind = hind or [(-0.30, 0.50), (-0.25, 0.33), (-0.33, 0.15), (-0.30, 0.0)]
    front = front or [(0.18, 0.55), (0.22, 0.40), (0.17, 0.28), (0.20, 0.06), (0.22, 0.0)]
    # The animal's own left is up x forward: -X when it faces +Y.
    for side, x in (("L", -0.1 * fwd), ("R", 0.1 * fwd)):
        names, parent = [], trunk
        for i, (yy, zz) in enumerate(hind):
            parent = add(f"h{side}{i}", parent, (x, yy, zz))
            names.append(parent)
            if helpers and i == 1:
                parent = add(f"h{side}{i}twin", parent, (x, yy, zz))
                parent = add(f"h{side}{i}twist", parent, (x, yy + 0.4 * (hind[2][0] - yy), zz + 0.4 * (hind[2][1] - zz)))
        roles["H" + side] = tuple(names[:4])
        names, parent = [], "u6"
        for i, (yy, zz) in enumerate(front):
            parent = add(f"f{side}{i}", parent, (x, yy, zz + (front_z if i == len(front) - 1 or i >= 3 else 0)))
            names.append(parent)
        roles["F" + side] = tuple(names[1:5])
        roles["blade" + side] = names[0]
    return bones, roles


def _bone_objects(bones):
    """Blender-like bones (name, parent, head_local, tail_local, use_deform) for `profile` on a stub armature."""
    objs = {b["name"]: SimpleNamespace(name=b["name"], parent=None, head_local=b["head"], tail_local=b["tail"],
                                       use_deform=True) for b in bones}
    for b in bones:
        objs[b["name"]].parent = objs.get(b["parent"])
    return SimpleNamespace(name="rig", data=SimpleNamespace(bones=list(objs.values())))


def _legs(q):
    return {k: (*v["bones"], v["ctrl"]) for k, v in q["legs"].items()}


@pytest.mark.parametrize("fwd", [1.0, -1.0])
def test_a_quadruped_without_the_rigify_names_is_mapped_by_its_structure(monkeypatch, fwd):
    """Names carry nothing here: the legs, their sides (the animal's own left, whichever way it faces), the
    shoulder blades, the trunk, neck, head, and tail all come from where the joints are and how they hang."""
    dr = _module(monkeypatch)
    bones, roles = _quadruped(fwd=fwd)
    q = dr._structural_profile(bones)
    assert _legs(q) == {k: roles[k] for k in ("FL", "FR", "HL", "HR")}
    assert q["pelvis"] == "u0" and q["hips"] == ("hL0", "hR0") and q["shoulders"] == ("fL1", "fR1")
    assert q["chain"]["fL0"][0] == "LeftShoulder" and q["chain"]["fR0"][0] == "RightShoulder"
    assert q["absolute"] == ("u1", "u2", "u3")
    assert q["chain"]["u7"][:2] == ("Neck", "Head") and q["chain"]["u8"][:2] == ("Head", "Head_end")
    dr._check_roles(_bone_objects(bones), q)


def test_the_bone_that_moves_the_body_carries_every_skinned_bone(monkeypatch):
    """Where the legs hang from a bone below the skinned pelvis and tail, the body is moved from the pelvis
    above them: moving it from the legs' common bone would leave the rump and tail standing while the dog walks
    off. The tail is found on the pelvis, behind the hind legs."""
    dr = _module(monkeypatch)
    bones, roles = _quadruped(legs_below_hips=True)
    q = dr._structural_profile(bones)
    assert q["pelvis"] == "u0" and _legs(q)["HL"] == roles["HL"]
    assert q["chain"]["uS"][0] == "Hips" and q["absolute"] == ("u1", "u2", "u3")
    dr._check_roles(_bone_objects(bones), q)


def test_a_leg_whose_first_bone_starts_on_the_midline_is_read_from_its_hip(monkeypatch):
    """Some rigs start both legs of a pair at one point on the midline and step out to the hips; the body's
    left-right axis runs between the hips, so the leg is read from the joint off the midline, and a rig whose
    hips still meet at one point is refused rather than rolled."""
    dr = _module(monkeypatch)
    bones, roles = _quadruped()
    by = {b["name"]: b for b in bones}
    for side, x in (("L", -0.1), ("R", 0.1)):                    # a helper from the pelvis's center to each hip
        by[f"h{side}0"]["parent"] = f"h{side}c"
        bones.append(dict(name=f"h{side}c", parent="u0", head=(0, -0.30, 0.55), tail=(x, -0.30, 0.50)))
    assert _legs(dr._structural_profile(bones))["HL"] == roles["HL"]
    for b in bones:
        if b["name"] in ("hL0", "hR0"):
            b["head"] = (0.0, b["head"][1], b["head"][2])             # now the hips themselves meet
    with pytest.raises(ValueError, match="sit at one point|fewer than three segments"):
        dr._structural_profile(bones)


def test_profile_keeps_the_rigify_names_and_reads_other_rigs_by_structure(monkeypatch):
    """The name path runs first, unchanged, so a rigify dog's profile is what it was; a rig without those names
    is read by structure; `roles=` takes a map as given, after the same checks."""
    dr = _module(monkeypatch)
    assert dr.profile(_armature(SHEPHERD))["pelvis"] == "spine.004_45"
    bones, roles = _quadruped()
    q = dr.profile(_bone_objects(bones))
    assert _legs(q)["HL"] == roles["HL"]
    assert dr.profile(_bone_objects(bones), roles=q) == q


def test_helper_bones_stacked_on_a_joint_or_partway_along_a_bone_are_passed_over(monkeypatch):
    dr = _module(monkeypatch)
    bones, roles = _quadruped(helpers=True)
    assert _legs(dr._structural_profile(bones))["HL"] == roles["HL"]


@pytest.mark.parametrize("hind, paw", [
    ([(-0.30, 0.50), (-0.22, 0.27), (-0.30, 0.02), (-0.20, 0.0)], 3),               # a bear: the ankle is down too
    ([(-0.30, 0.50), (-0.22, 0.35), (-0.30, 0.20), (-0.29, 0.06), (-0.27, 0.0)], 3),  # a horse: the hoof below
])
def test_the_paw_is_the_contact_joint_with_three_segments_above_it(monkeypatch, hind, paw):
    """A bear's ankle lies on the floor beside its ball, and a horse's hoof is below its fetlock: the paw is the
    uppermost joint near the floor with thigh, shin, and foot above it, so the ball, and the fetlock."""
    dr = _module(monkeypatch)
    bones, _ = _quadruped(hind=hind)
    assert _legs(dr._structural_profile(bones))["HL"] == tuple(f"hL{i}" for i in range(paw + 1))


def test_a_rig_not_standing_on_four_legs_in_its_rest_pose_is_refused(monkeypatch):
    dr = _module(monkeypatch)
    bones, _ = _quadruped(front_z=0.3)
    with pytest.raises(ValueError, match="standing on four legs"):
        dr._structural_profile(bones)


def _swing(bones, names, pivot, deg):
    """Turn joints `names` about the head of bone `pivot`, in the body's side-on plane (about X): a pose, so every
    bone keeps its length."""
    import math
    by = {b["name"]: b for b in bones}
    py, pz = by[pivot]["head"][1], by[pivot]["head"][2]
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    turn = lambda p: (p[0], py + c * (p[1] - py) - s_ * (p[2] - pz), pz + s_ * (p[1] - py) + c * (p[2] - pz))
    for n in names:
        by[n]["head"], by[n]["tail"] = turn(by[n]["head"]), turn(by[n]["tail"])


def test_a_rest_pose_caught_mid_stride_pairs_each_leg_with_its_twin(monkeypatch):
    """One forepaw well ahead of the other, as an animal sculpted walking stands: by position it is nearer a hind
    paw's mirror image than its twin's, so the legs are paired by how deep they sit in the skeleton."""
    dr = _module(monkeypatch)
    bones, roles = _quadruped()
    _swing(bones, ["fL2", "fL3", "fL4"], "fL1", 40)           # the left foreleg reaching forward
    _swing(bones, ["hR1", "hR2", "hR3"], "hR0", -35)          # the right hind leg pushing back
    q = dr._structural_profile(bones)
    assert _legs(q) == {k: roles[k] for k in ("FL", "FR", "HL", "HR")}


def test_a_lifted_paw_pairs_with_its_twin_on_the_floor(monkeypatch):
    """A forepaw raised off the floor (a pointer's pose, an elephant stepping) still has its twin standing: the
    lifted leg is found across the midline from it, as deep in the skeleton and low on the body."""
    dr = _module(monkeypatch)
    bones, roles = _quadruped()
    _swing(bones, ["fR3", "fR4"], "fR2", 70)                   # the right forepaw folded up off the floor
    q = dr._structural_profile(bones)
    assert _legs(q)["FR"] == roles["FR"] and _legs(q)["FL"] == roles["FL"]


def test_a_nub_dropped_to_the_floor_from_the_head_is_not_a_paw(monkeypatch):
    """An exporter's end bone left on the floor far below the head it hangs from (a 3ds Max Biped head nub) is
    passed over, so it neither sets the floor nor stands in for a paw."""
    dr = _module(monkeypatch)
    bones, roles = _quadruped()
    bones.append(dict(name="headNub", parent="u8", head=(0.0, 0.05, -0.05), tail=(0.0, 0.05, -0.03)))
    q = dr._structural_profile(bones)
    assert _legs(q) == {k: roles[k] for k in ("FL", "FR", "HL", "HR")}


def test_a_body_resting_sitting_with_its_back_raised_is_refused(monkeypatch):
    """Four paws on the floor and the shoulders far above the hips is a cat sculpted sitting: the captures would drive
    a body that stays upright through the walk, so the rig is refused and says why."""
    dr = _module(monkeypatch)
    bones, _ = _quadruped()
    for b in bones:
        if b["name"] in ("u5", "u6", "u7", "u8", "u9", "u10", "u11", "fL0", "fL1", "fR0", "fR1"):
            b["head"] = (b["head"][0], b["head"][1], b["head"][2] + 0.5)
            b["tail"] = (b["tail"][0], b["tail"][1], b["tail"][2] + 0.5)
    with pytest.raises(ValueError, match="rests sitting or rearing"):
        dr._structural_profile(bones)


def test_left_and_right_legs_that_are_not_mirror_images_are_refused(monkeypatch):
    """A rig whose left and right legs would map to different joints is refused rather than half wrong."""
    dr = _module(monkeypatch)
    bones, _ = _quadruped()
    for b in bones:
        if b["name"] == "hL1":
            b["head"] = (b["head"][0], b["head"][1], 0.42)
    with pytest.raises(ValueError, match="not mirror images"):
        dr._structural_profile(bones)


def test_the_hind_legs_may_hang_from_a_lumbar_chain_under_the_pelvis(monkeypatch):
    """Rigs rooted mid-back carry the hind legs on lumbar bones below the root; a chain of them is accepted, and
    a hind leg hanging anywhere else is still refused."""
    dr = _module(monkeypatch)
    q = dict(pelvis="p", hips=("tL", "tR"), shoulders=("aL", "aR"), absolute=(),
             chain={"l1": ("Hips", "Spine1", "pel"), "l2": ("Hips", "Spine1", "pel"), "c": ("Spine1", "Neck", "tho")},
             legs={})
    arm = _armature({"p": None, "l1": "p", "l2": "l1", "tL": "l2", "tR": "l2", "c": "p", "aL": "c", "aR": "c"})
    dr._check_roles(arm, q)
    arm = _armature({"p": None, "l1": "p", "x": "l1", "tL": "x", "tR": "x", "c": "p", "aL": "c", "aR": "c"})
    with pytest.raises(ValueError, match="hind legs hang from x"):
        dr._check_roles(arm, q)


def test_inspect_reads_the_capture_and_its_index_spans(monkeypatch):
    _needs_library()
    dr = _module(monkeypatch)
    info = dr.inspect_bvh("D1_008_KAN01_002")
    assert info["fps"] == 60 and 14 < info["seconds"] < 16
    assert any("sit" in s for s in info["spans"]), info["spans"]


def test_a_clip_stands_on_the_floor_and_edits_keep_its_length_honest(monkeypatch):
    _needs_library()
    dr = _module(monkeypatch)
    c = dr.clip("D1_009_KAN01_001", 3.0, 7.0)
    paws = c.P[:, [c.J[k] for k in dr.PAWS], 2].min(1)
    assert abs(np.percentile(paws, 3)) < 0.01, "the capture's floor is z = 0"
    assert abs(c.seconds - 4.0) < 0.05
    assert abs(c.speed(2.0).seconds - 2.0) < 0.05
    assert abs(c.cycle(seconds=9.0).seconds - 9.0) < 0.05
    assert c.smooth(2).P.shape == c.P.shape
    sit = dr.clip("D1_008_KAN01_002", 0.3, 6.0).in_place()
    hip = sit._hips()[:, :2]
    assert np.abs(hip - hip[0]).max() < 1e-9, "in_place pins the hips"
    assert abs(sit._heading()[-1] - sit._heading()[0]) < 1e-6
    st = c.stride(0.8)
    assert abs(st.info()["root_travel_m"] - 0.8 * c.info()["root_travel_m"]) < 0.05


def test_a_splice_continues_where_and_how_the_first_clip_ends(monkeypatch):
    _needs_library()
    dr = _module(monkeypatch)
    a = dr.clip("D1_008_KAN01_002", 0.0, 6.0)
    b = dr.clip("D1_009_KAN01_001", 3.0, 6.0)
    s = a.then(b, blend_s=0.3)
    hips = s._hips()[:, :2]
    jumps = np.linalg.norm(np.diff(hips, axis=0), axis=1)
    assert jumps.max() < 0.08, "no jump in position across the join"
    head = np.degrees(np.abs(np.diff(s._heading())))
    assert head.max() < 8, "no jump in heading across the join"
    assert abs(s.seconds - (a.seconds + b.seconds - 0.3)) < 0.05


def test_follow_lays_the_clip_along_the_path_and_faces_along_it(monkeypatch):
    _needs_library()
    dr = _module(monkeypatch)
    c = dr.clip("D1_009_KAN01_001", 3.0, 7.0)              # a trot, ~5 m
    f = c.follow([(0, 0), (0, 3), (4, 3)])
    hips = f._hips()[:, :2]
    assert f.placed and np.allclose(hips[0], [0, 0], atol=0.05)
    # 3 m up the first leg, then along the second: past the corner, heading +x
    assert hips[-1][1] > 2.5 and hips[-1][0] > 1.0, hips[-1]
    yaw = np.degrees(f._heading()[-1]) % 360
    assert yaw < 30 or yaw > 330, yaw


def test_only_the_pinned_leg_solve_remains(monkeypatch):
    """Only legs="ik" (each planted paw pinned) is accepted, since unpinned paws slide and
    float; asking for another solve fails with a ValueError before any Blender work."""
    dr = _module(monkeypatch)
    for legs in ("angles", "ik_trunk"):
        with pytest.raises(ValueError, match="only 'ik' is supported"):
            dr.retarget(None, None, legs=legs)


def test_the_dog_notes_teach_the_helper_and_ship_its_index():
    notes = (NOTES / "NOTES.md").read_text()
    assert "import dog_retarget as dr" in notes and "dr.retarget(" in notes
    index = (NOTES / "index.txt").read_text().splitlines()
    assert len(index) > 1000 and index[0].startswith("D1_")
    for prompt in ("animation.md",):
        assert "dog_retarget" in (REPO / "agent/prompts" / prompt).read_text()


def test_the_dog_joint_map_names_every_bone_and_segment_the_helper_maps(monkeypatch):
    """JOINT_MAPS.md is the dog helper's bone map as a table; it fails if PROFILE gains a bone the table lacks."""
    dr = _module(monkeypatch)
    table = (NOTES / "JOINT_MAPS.md").read_text()
    p = dr.PROFILE
    bones = {p["pelvis"], *p["chain"], *(b for leg in p["legs"].values() for b in (*leg["bones"], leg["ctrl"]))}
    segments = {j for start, end, _ in p["chain"].values() for j in (start, end)} | \
               {j for leg in p["legs"].values() for j in leg["cap"]}
    for name in bones | segments:
        assert f"`{name}`" in table, name



_BAKE = r'''
import bpy, json, sys, numpy as np
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
c = dr.clip("D1_008_KAN01_002", 0.3, 6.0).in_place()   # walk, then sit, where it stands
rep = dr.retarget(arm, c, name="sit")
assigned_after = arm.animation_data.action.name if arm.animation_data.action else None
act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
if act.slots: arm.animation_data.action_slot = act.slots[0]
prof = dr.profile(arm); low = dr._lowest(arm, rep["frames"], dr._skin_masks(arm, prof))
f = int(4.0 * 24)                                    # seated
bpy.context.scene.frame_set(f)
P = lambda n: np.array(arm.matrix_world @ arm.pose.bones[n].head)
hips = P(prof["hips"][0]) - P(prof["hips"][1])
hl, hr = P(prof["legs"]["HL"]["bones"][1]), P(prof["legs"]["HR"]["bones"][1])
stretch = 0.0
for g in range(1, rep["frames"] + 1):
    bpy.context.scene.frame_set(g)
    for v in prof["legs"].values():
        stretch = max(stretch, (arm.pose.bones[v["bones"][2]].tail - arm.pose.bones[v["ctrl"]].head).length)
bpy.context.scene.frame_set(f)
rep.update(assigned_after=assigned_after, stretch_m=stretch,
           paw_p50={k: float(np.median(low[k])) for k in prof["legs"]},
           knee_spread_m=float(np.linalg.norm((hl - hr)[:2])), hip_width_m=float(np.linalg.norm(hips[:2])))
print("REPORT " + json.dumps(rep))
'''


@pytest.mark.expensive
def test_a_sit_keeps_all_four_paws_down_and_the_hind_legs_tucked(tmp_path):
    py = blender_python()
    rigs = list((RIG / "assets/rigs").glob("*.blend"))
    if py is None or not rigs:
        pytest.skip("needs the dycraft-blender interpreter and the German shepherd rig")
    _needs_library()
    script = tmp_path / "bake.py"
    script.write_text(_BAKE)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), "x"],
                         capture_output=True, text=True, timeout=1800,
                         env={**__import__("os").environ, "DYCRAFT_DOG_MOCAP": str(LIBRARY)})
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["assigned_after"] is None, "the rig's own action is restored"
    for leg, z in rep["paw_p50"].items():
        assert -0.01 < z < 0.02, f"{leg} paw off the floor: {rep['paw_p50']}"
    assert rep["below_ground_max_m"] < 0.06, rep
    # a splayed sit throws the stifles out sideways, far wider than the hips
    assert rep["knee_spread_m"] < 2.5 * rep["hip_width_m"], rep
    # paws reach their targets with every bone on its parent: no stretched leg
    assert rep["stretch_m"] < 1e-4, rep["stretch_m"]
    assert max(rep["paw_shortfall_m"].values()) < 0.01, rep["paw_shortfall_m"]


def test_follow_does_not_spin_a_slow_sniffing_dog(monkeypatch):
    """At sniffing speed the travel direction is sway and noise, and lining it up with the
    path would spin the dog in place; slow frames follow the body's heading."""
    _needs_library()
    dr = _module(monkeypatch)
    f = dr.clip("D1_055_KAN01_001", 176, 191).follow([(0, 0), (0, 2), (1, 3)])
    hd = np.degrees(f._heading())
    assert hd.max() - hd.min() < 30, "a dog standing and sniffing should not turn round"
    assert np.abs(np.diff(hd)).max() < 3


_LIE = r'''
import bpy, json, sys
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
rep = dr.retarget(arm, dr.clip("D1_013_KAN01_001", 28, 40), name="lie")   # stand, lie down, get up
print("REPORT " + json.dumps(rep))
'''


@pytest.mark.expensive
def test_lying_down_keeps_the_chest_out_of_the_floor_without_launching_the_rump(tmp_path):
    """A chest on the floor must not lift the whole dog while the planted front paws pull it
    back down (the lift would pile up with the rump in the air)."""
    py = blender_python()
    rigs = list((RIG / "assets/rigs").glob("*.blend"))
    if py is None or not rigs:
        pytest.skip("needs the dycraft-blender interpreter and the German shepherd rig")
    _needs_library()
    script = tmp_path / "lie.py"
    script.write_text(_LIE)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), "x"],
                         capture_output=True, text=True, timeout=1800,
                         env={**__import__("os").environ, "DYCRAFT_DOG_MOCAP": str(LIBRARY)})
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["lift_max_m"] < 0.2, rep
    assert rep["below_ground_max_m"] < 0.05, rep
    assert max(rep["paw_shortfall_m"].values()) < 0.06, rep["paw_shortfall_m"]


_FLIPS = r'''
import bpy, json, math, sys
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
out = {}
for tag, a, b, c in (("trot", "D1_009_KAN01_001", 3, 9), ("lie", "D1_013_KAN01_001", 28, 40)):
    bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); s = bpy.context.scene; s.render.fps = 24
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); prof = dr.profile(arm)
    rep = dr.retarget(arm, dr.clip(a, b, c), name=tag, ground=False)
    act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
    if act.slots: arm.animation_data.action_slot = act.slots[0]
    worst = {}; prev = {}
    for f in range(1, rep["frames"] + 1):
        s.frame_set(f)
        for v in prof["legs"].values():
            for n in v["bones"][:2]:
                q = (arm.matrix_world @ arm.pose.bones[n].matrix).to_quaternion()
                if n in prev:
                    worst[n] = max(worst.get(n, 0.0), math.degrees(2 * math.acos(min(1.0, abs(prev[n].dot(q))))))
                prev[n] = q
    out[tag] = worst
print("REPORT " + json.dumps(out))
'''


@pytest.mark.expensive
def test_thighs_and_shins_turn_continuously(tmp_path):
    """Knees must not swing round a near-straight or fully folded leg: a trot's thigh and
    shin turn well under 80 deg per frame at 24 fps."""
    py = blender_python()
    rigs = list((RIG / "assets/rigs").glob("*.blend"))
    if py is None or not rigs:
        pytest.skip("needs the dycraft-blender interpreter and the German shepherd rig")
    _needs_library()
    script = tmp_path / "flips.py"
    script.write_text(_FLIPS)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), "x"],
                         capture_output=True, text=True, timeout=1800,
                         env={**__import__("os").environ, "DYCRAFT_DOG_MOCAP": str(LIBRARY)})
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    rep = json.loads(line[7:])
    assert set(rep) == {"trot", "lie"} and all(len(worst) == 8 for worst in rep.values()), rep   # thigh and shin x 4
    for clip, worst in rep.items():
        for bone, deg in worst.items():
            assert deg < 60, f"{clip}: {bone} turned {deg:.0f} deg in one frame"


_STALE = r'''
import bpy, json, math, sys, mathutils
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
holder = bpy.data.objects.new("holder", None); bpy.context.scene.collection.objects.link(holder)
arm.parent = holder
holder.matrix_world = mathutils.Matrix.Translation((2.0, 1.0, 0.3)) @ mathutils.Matrix.Rotation(math.pi / 2, 4, "Z")
bpy.context.view_layer.update()
holder.matrix_world = mathutils.Matrix.Identity(4)      # moved back, scene not re-evaluated
rep = dr.retarget(arm, dr.clip("D1_008_KAN01_001", 6.8, 10.3).smooth(2.0), name="walk")
print("REPORT " + json.dumps(rep))
'''


@pytest.mark.expensive
def test_a_dog_moved_just_before_the_retarget_is_retargeted_where_it_now_stands(tmp_path):
    """A dog whose parent is moved just before the retarget is retargeted where it now
    stands: a stale world matrix would put the floor off and the paws out of reach,
    flipping the knees."""
    py = blender_python()
    rigs = list((RIG / "assets/rigs").glob("*.blend"))
    if py is None or not rigs:
        pytest.skip("needs the dycraft-blender interpreter and the German shepherd rig")
    _needs_library()
    script = tmp_path / "stale.py"
    script.write_text(_STALE)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), "x"],
                         capture_output=True, text=True, timeout=1800,
                         env={**__import__("os").environ, "DYCRAFT_DOG_MOCAP": str(LIBRARY)})
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    rep = json.loads(line[7:])
    # The pinned solve never stretches a leg, so a target out of reach reports a shortfall;
    # a stale world matrix would leave the paws tens of centimeters short.
    assert max(rep["paw_shortfall_m"].values()) < 0.02, rep["paw_shortfall_m"]
    assert rep["lift_max_m"] < 0.01, rep


_FEET = r'''
import bpy, json, math, sys
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); s = bpy.context.scene; s.render.fps = 24
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); prof = dr.profile(arm)
rep = dr.retarget(arm, dr.clip("D1_ex06_KAN02_001", 23.5, 27.0).smooth(2.0).in_place(), name="sniff")
act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
if act.slots: arm.animation_data.action_slot = act.slots[0]
worst = {}; prev = {}
for f in range(1, rep["frames"] + 1):
    s.frame_set(f)
    for v in prof["legs"].values():
        n = v["bones"][2]; q = (arm.matrix_world @ arm.pose.bones[n].matrix).to_quaternion()
        if n in prev:
            worst[n] = max(worst.get(n, 0.0), math.degrees(2 * math.acos(min(1.0, abs(prev[n].dot(q))))))
        prev[n] = q
print("REPORT " + json.dumps(worst))
'''


@pytest.mark.expensive
def test_a_paw_out_of_reach_does_not_snap_its_metatarsus(tmp_path):
    """Out of reach, the wrist goes to the point of the reach circle nearest the capture's
    own: putting it behind the paw whatever the capture had would turn a sniffing dog's front
    foot far further in one frame than the capture's metacarpus turns (at most 26 deg)."""
    py = blender_python()
    rigs = list((RIG / "assets/rigs").glob("*.blend"))
    if py is None or not rigs:
        pytest.skip("needs the dycraft-blender interpreter and the German shepherd rig")
    _needs_library()
    script = tmp_path / "feet.py"
    script.write_text(_FEET)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), "x"],
                         capture_output=True, text=True, timeout=1800,
                         env={**__import__("os").environ, "DYCRAFT_DOG_MOCAP": str(LIBRARY)})
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    rep = json.loads(line[7:])
    assert len(rep) == 4, rep                                          # one metatarsus per leg
    for bone, deg in rep.items():
        assert deg < 45, f"{bone} turned {deg:.0f} deg in one frame"


def _run_blender(tmp_path, code, name):
    py = blender_python()
    rigs = list((RIG / "assets/rigs").glob("*.blend"))
    if py is None or not rigs:
        pytest.skip("needs the dycraft-blender interpreter and the German shepherd rig")
    _needs_library()
    script = tmp_path / name
    script.write_text(code)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), "x"],
                         capture_output=True, text=True, timeout=1800,
                         env={**__import__("os").environ, "DYCRAFT_DOG_MOCAP": str(LIBRARY)})
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    return json.loads(line[7:])


_LIFTOFF = r'''
import bpy, json, math, sys
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); s = bpy.context.scene; s.render.fps = 24
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); prof = dr.profile(arm)
rep = dr.retarget(arm, dr.clip("D1_004_KAN01_001", 10.5, 16.8), name="walk")
act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
if act.slots: arm.animation_data.action_slot = act.slots[0]
out = {}; prev = {}
for f in range(1, rep["frames"] + 1):
    s.frame_set(f)
    for k, v in prof["legs"].items():
        toe = (arm.matrix_world @ arm.pose.bones[v["bones"][-1]].head).z
        q = (arm.matrix_world @ arm.pose.bones[v["bones"][0]].matrix).to_quaternion()
        if k in prev:
            dz = abs(toe - prev[k][0]); da = math.degrees(2 * math.acos(min(1.0, abs(prev[k][1].dot(q)))))
            o = out.setdefault(k, [0.0, 0.0]); o[0] = max(o[0], dz); o[1] = max(o[1], da)
        prev[k] = (toe, q)
print("REPORT " + json.dumps(out))
'''


@pytest.mark.expensive
def test_a_paw_lifts_off_and_lands_without_a_jump(tmp_path):
    """A captured planted paw can map centimeters above this rig's floor, so the swing height
    is taken above the paw's own lift-off and landing, not above the floor: the paw rises from
    and returns to its locked footing without a jump, and the foreleg turns smoothly."""
    out = _run_blender(tmp_path, _LIFTOFF, "liftoff.py")
    for leg, (dz, deg) in out.items():
        assert deg < 30, f"{leg} upper leg turned {deg:.0f} deg in one frame"
    for leg in ("FL", "FR"):                    # a quick swing moves ~4 cm in a frame
        assert out[leg][0] < 0.06, f"{leg} paw moved {out[leg][0] * 100:.1f} cm vertically in one frame"


_FORELEGS = r'''
import bpy, json, sys, numpy as np
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
out = {}
for tag, name, a, b in (("hold", "D1_005_KAN01_001", 2, 12), ("sit_to_lie", "D1_013_KAN01_001", 12.9, 16.3)):
    bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); s = bpy.context.scene; s.render.fps = 24
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); prof = dr.profile(arm)
    rep = dr.retarget(arm, dr.clip(name, a, b).smooth(2.0).in_place(), name=tag)
    act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
    if act.slots: arm.animation_data.action_slot = act.slots[0]
    L = prof["legs"]["FL"]; rows = []
    for f in range(1, rep["frames"] + 1):
        s.frame_set(f); z = lambda n: (arm.matrix_world @ arm.pose.bones[n].head).z
        rows.append((z(L["bones"][1]), z(L["bones"][2]), z(L["bones"][0])))
    rows = np.array(rows); low = rows[:, 2] < np.percentile(rows[:, 2], 25)
    out[tag] = dict(elbow=float(np.median(rows[low, 0])), wrist=float(np.median(rows[low, 1])),
                    short=max(rep["paw_shortfall_m"][k] for k in ("FL", "FR")))
print("REPORT " + json.dumps(out))
'''


@pytest.mark.expensive
def test_a_lying_dog_rests_its_forearms_on_the_floor(tmp_path):
    """A lying dog's forearm lies along the floor with the elbow bent beneath it, not the
    foreleg straight on its toes: the wrist floor follows the capture, the chest does not
    sink, and a capture leg longer than the rig's keeps its fraction of the rig's length."""
    rep = _run_blender(tmp_path, _FORELEGS, "forelegs.py")
    hold = rep["hold"]
    assert hold["elbow"] < 0.06 and hold["elbow"] - hold["wrist"] < 0.012, hold
    for tag, r in rep.items():
        # Folding from a sit to lying, a forepaw's target passes out of the leg's reach for a
        # few frames; the pinned solve keeps the leg at its length and reports the gap.
        assert r["short"] < 0.075, (tag, r)


_STAIRS = r'''
import bpy, json, sys, math, numpy as np
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
def box(name, x0, x1, y0, y1, z1):
    me = bpy.data.meshes.new(name)
    v = [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    me.from_pydata(v, [], [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)])
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o); return o
out = {}
for tag, path in (("up", [(0.0, -1.6), (0.0, 2.6)]), ("down", [(0.0, 2.4), (0.0, -1.8)])):
    bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); s = bpy.context.scene; s.render.fps = 24
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); prof = dr.profile(arm)
    fl = dr.floor_from_objects([box("s1", -1, 1, 0.5, 0.9, 0.14), box("s2", -1, 1, 0.9, 1.3, 0.28),
                                box("porch", -1, 1, 1.3, 3.5, 0.28)], arm, cell=0.02)
    c = dr.clip("D1_008_KAN01_001", 6.8, 10.3).smooth(2.0).cycle(seconds=6).follow(path)
    rep = dr.retarget(arm, c, name=tag, floor=fl)
    act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
    if act.slots: arm.animation_data.action_slot = act.slots[0]
    prev = {}; worst = 0.0; toe = []; top = []
    for f in range(1, rep["frames"] + 1):
        s.frame_set(f)
        for v in prof["legs"].values():
            for b in v["bones"]:
                q = (arm.matrix_world @ arm.pose.bones[b].matrix).to_quaternion()
                if b in prev: worst = max(worst, math.degrees(2 * math.acos(min(1.0, abs(prev[b].dot(q))))))
                prev[b] = q
            t = arm.matrix_world @ arm.pose.bones[v["ctrl"]].head
            toe.append(t.z - float(fl(np.array([t.x]), np.array([t.y]))[0]))
        top.append((arm.matrix_world @ arm.pose.bones[prof["pelvis"]].head).z)
    out[tag] = dict(worst=worst, toe_min=min(toe), below=rep["below_ground_max_m"],
                    short=max(rep["paw_shortfall_m"].values()), rise=max(top) - min(top))
print("REPORT " + json.dumps(out))
'''


@pytest.mark.expensive
def test_a_dog_walks_up_and_down_steps(tmp_path):
    """`floor` lets a retarget climb: paws stand on each tread, swings clear the edges, the
    trunk rises with the footing and no skin goes into the steps."""
    rep = _run_blender(tmp_path, _STAIRS, "stairs.py")
    assert set(rep) == {"up", "down"}, rep
    for tag, r in rep.items():
        assert r["toe_min"] > 0.02, (tag, r)           # no toe joint below the tread it is over
        assert r["below"] < 0.015, (tag, r)
        # Going down, a forepaw reaching for the lower tread can stop short of it: the leg
        # is never stretched (a known limit the skill states).
        assert r["short"] < 0.09, (tag, r)
        assert r["worst"] < 90, (tag, r)
        assert r["rise"] > 0.2, (tag, r)                # the hips climbed the 0.28 m


_HEAD = r'''
import bpy, json, sys
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
out = {}
for tag, kw in (("none", {}), ("down", {"head_pitch_deg": 30})):
    bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); s = bpy.context.scene; s.render.fps = 24
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    rep = dr.retarget(arm, dr.clip("D1_009_KAN01_001", 3, 6).smooth(2.0), name=tag, **kw)
    act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
    if act.slots: arm.animation_data.action_slot = act.slots[0]
    hb = [b for b in arm.pose.bones if b.name.startswith("Head")][0]
    s.frame_set(rep["frames"] - 2); out[tag] = (arm.matrix_world @ hb.tail).z
print("REPORT " + json.dumps(out))
'''


@pytest.mark.expensive
def test_head_pitch_lowers_the_head(tmp_path):
    """The captured dog often looks up at its handler; `head_pitch_deg` bends the neck down."""
    rep = _run_blender(tmp_path, _HEAD, "head.py")
    assert rep["none"] - rep["down"] > 0.03, rep


_SNIFF = r'''
import bpy, json, math, sys, numpy as np
sys.path.insert(0, sys.argv[-3]); import dog_retarget as dr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2]); s = bpy.context.scene; s.render.fps = 24
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); prof = dr.profile(arm)
neck = [b.name for b in arm.data.bones if prof["chain"].get(b.name, ("",))[:2] == ("Neck", "Head")]
head = next(b.name for b in arm.data.bones if prof["chain"].get(b.name, ("",))[:2] == ("Head", "Head_end"))
pitch = lambda v: math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
out = {}
for tag, clip, a, b in (("sniff", "D1_055_KAN01_001", 181.9, 187.2), ("walk", "D1_008_KAN01_001", 6.8, 10.3)):
    c = dr.clip(clip, a, b).in_place()
    rep = dr.retarget(arm, c, name=tag)
    act = bpy.data.actions[rep["action"]]; arm.animation_data.action = act
    if act.slots: arm.animation_data.action_slot = act.slots[0]
    gap = []
    for f in range(1, rep["frames"] + 1, 4):
        s.frame_set(f)
        W = arm.matrix_world
        k = min(int(round((f - 1) / 24 * c.fps)), len(c.P) - 1)
        gap.append(pitch(W @ arm.pose.bones[head].head - W @ arm.pose.bones[neck[0]].head)
                   - pitch(c.P[k, c.J["Head"]] - c.P[k, c.J["Neck"]]))
    out[tag] = {"neck_above_capture_deg": float(np.median(gap)), "below_ground_max_m": rep.get("below_ground_max_m")}
    arm.animation_data.action = None
print("REPORT " + json.dumps(out))
'''


@pytest.mark.expensive
def test_a_sniffing_dog_reaches_down_with_the_capture_and_a_walking_one_keeps_its_carriage(tmp_path):
    """D1_055's sniff holds the nose near the floor. The solve carries the neck by the capture's
    change from its neutral stand, which keeps the shepherd's rest carriage, 25-30 degrees above
    the capture's; below level the neck turns to the capture's own direction. So the sniff's neck
    follows the capture, and the walk keeps the rig's bearing."""
    rep = _run_blender(tmp_path, _SNIFF, "sniff.py")
    assert abs(rep["sniff"]["neck_above_capture_deg"]) < 8, rep
    assert rep["walk"]["neck_above_capture_deg"] > 15, rep
    assert rep["sniff"]["below_ground_max_m"] < 0.02, rep
