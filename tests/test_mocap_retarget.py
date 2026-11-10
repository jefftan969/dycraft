"""The CMU -> humanoid retarget helper agents import inside Blender (`/tools/mocap_retarget.py`).

The helper solves once the five traps the skill lists (rest poses, bone axes, facing,
proportions, ground contact). Mapping is tested against stubbed bpy; the bake itself
against real Blender on one rig per skeleton family, where the library and rigs exist.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dycraft.animation.blender.runtime import REPO
from dycraft.render._blender_env import blender_python

LIBRARY = REPO / "assets" / "mocap" / "cmu" / "data"
WALK = LIBRARY / "008" / "08_01.bvh"
RIGS = {"biped": "rb_male_adult_03_rig", "mixamo": "soldier_rig", "unreal": "sf_rp_eric_rig"}


def _module(monkeypatch):
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Matrix=None, Quaternion=None, Vector=None))
    spec = importlib.util.spec_from_file_location(
        "retarget_under_test", REPO / "animation/mocap/mocap_retarget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _armature(names):
    return SimpleNamespace(name="rig", data=SimpleNamespace(
        bones=[SimpleNamespace(name=n) for n in names]))


def test_names_normalise_across_exporters(monkeypatch):
    mr = _module(monkeypatch)
    assert mr.normalise("mixamorig:LeftForeArm") == "leftforearm"
    assert mr.normalise("upperarm_l_024") == "upperarm_l"
    assert mr.normalise("Bip01 L Thigh") == "bip01 l thigh"


@pytest.mark.parametrize("names, legs, hips", [
    (["Bip01 Pelvis", "Bip01 Spine", "Bip01 L Thigh", "Bip01 L Calf", "Bip01 L Foot",
      "Bip01 R Thigh", "Bip01 R Calf", "Bip01 R Foot"], ("Bip01 L Thigh", "Bip01 R Foot"),
     "Bip01 Pelvis"),
    (["mixamorig:Hips", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot",
      "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot"],
     ("mixamorig:LeftUpLeg", "mixamorig:RightFoot"), "mixamorig:Hips"),
    # Renderpeople-style Unreal: `hip`, `upperleg_*`, and the exporter's index suffixes.
    (["_rootJoint", "root_01", "hip_02", "upperleg_l_060", "lowerleg_l_061", "foot_l_062",
      "upperleg_r_065", "lowerleg_r_066", "foot_r_067"], ("upperleg_l_060", "foot_r_067"), "hip_02"),
])
def test_each_skeleton_family_maps_its_legs(monkeypatch, names, legs, hips):
    mapping = _module(monkeypatch).bone_map(_armature(names))
    assert mapping["LeftUpLeg"] == legs[0] and mapping["RightFoot"] == legs[1]
    assert mapping["Hips"] == hips


def test_a_quadruped_is_refused_by_name_not_mangled(monkeypatch):
    dog = ["Back_38", "FrontUpperLeg.L_17", "BackUpperLeg.L_25", "Neck1_14"]
    with pytest.raises(ValueError, match="quadrupeds"):
        _module(monkeypatch).bone_map(_armature(dog))


def test_limb_segments_are_measured_between_joints_not_along_bone_tails(monkeypatch):
    """A rig imported from Maya or a game engine leaves tails along a local axis; the segment must still run
    from the joint to the next mapped joint, and a bone with no next joint points at its only child."""
    import numpy as np
    mr = _module(monkeypatch)
    v = lambda *x: np.array(x, float)
    child = SimpleNamespace(name="thumb2", head_local=v(0.1, 0.0, 1.0), tail_local=v(0.2, 0.0, 1.0), children=[])
    bones = {"thigh": SimpleNamespace(name="thigh", head_local=v(0, 0, 1), tail_local=v(0.4, 0, 1)),    # tail 90 deg off
             "calf": SimpleNamespace(name="calf", head_local=v(0, 0, 0.5), tail_local=v(0, 0, 1)),      # tail reversed
             "foot": SimpleNamespace(name="foot", head_local=v(0, 0, 0.05), tail_local=v(0, 0.1, 0)),
             "thumb1": SimpleNamespace(name="thumb1", head_local=v(0, 0, 1), tail_local=v(0, 0.3, 1), children=[child]),
             "thumb2": child}
    mapping = {"LeftUpLeg": "thigh", "LeftLeg": "calf", "LeftFoot": "foot"}
    assert np.allclose(mr._limb_vector(bones, mapping, "LeftUpLeg"), v(0, 0, -0.5))
    assert np.allclose(mr._limb_vector(bones, mapping, "LeftLeg"), v(0, 0, -0.45))
    assert np.allclose(mr._tip(bones, "thumb1"), child.head_local)
    helper = SimpleNamespace(name="helper", head_local=v(0, 0, 0.05), tail_local=v(0, 0, 0.1), children=[])
    ankle = SimpleNamespace(name="ankle", head_local=v(0, 0, 0.05), tail_local=v(0, 0.12, 0), children=[helper])
    assert np.allclose(mr._tip({"ankle": ankle}, "ankle"), ankle.tail_local), "a child on the bone's head gives no direction"
    q = SimpleNamespace(w=np.cos(0.3), x=0.0, y=np.sin(0.3), z=0.0)                  # 0.6 rad about +Y
    assert np.isclose(mr._twist_angle(q, SimpleNamespace(x=0, y=1, z=0)), 0.6)
    assert np.isclose(mr._twist_angle(q, SimpleNamespace(x=0, y=-1, z=0)), -0.6)    # a reversed bone axis


_OFF_AXIS = r"""
import json, sys, bpy
from mathutils import Matrix
sys.path.insert(0, sys.argv[-4])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-3], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
m = mr.bone_map(arm)
hand_refs = {h: mr._hand_refs(arm, h) for h in mr.HANDS}
variant, hands = sys.argv[-1].split(':')
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
eb = arm.data.edit_bones
for b in eb:
    b.use_connect = False
for j in ("LeftForeArm", "RightForeArm"):        # a twist child, so the forearm has more than one child
    f = eb[m[j]]; t = eb.new(f.name + "_twist"); t.parent = f
    t.head = f.head.lerp(f.tail, 0.5); t.tail = t.head + (f.tail - f.head) * 0.25
if variant == 'off':
    # What a Maya or game export does, heads kept: tails turned 90 degrees about a perpendicular,
    # forearms included, and the calves reversed.
    for j in ("LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm"):
        b = eb[m[j]]; seg = b.tail - b.head
        b.tail = b.head + Matrix.Rotation(1.5708, 3, seg.orthogonal().normalized()) @ seg
    for j in ("LeftLeg", "RightLeg"):
        b = eb[m[j]]; b.tail = b.head - (b.tail - b.head)
    for h in mr.HANDS:
        b = eb[hand_refs[h][1]]; seg = b.tail - b.head
        b.tail = b.head + Matrix.Rotation(1.5708, 3, seg.orthogonal().normalized()) @ seg
bpy.ops.object.mode_set(mode='OBJECT')
rep = mr.retarget(arm, sys.argv[-2], start_s=1.0, end_s=4.0, name='walk', hands=hands == 'hands')
arm.animation_data.action = bpy.data.actions[rep['action']]
names = sorted(set(m.values()) | {n for refs in hand_refs.values() for n in refs if n})
heads = []
for f in range(1, rep['frames'] + 1, 4):
    bpy.context.scene.frame_set(f)
    heads.append({n: list(arm.matrix_world @ arm.pose.bones[n].head) for n in names})
print('REPORT ' + json.dumps({'heads': heads, 'warnings': rep['warnings']}))
"""


@pytest.mark.expensive
def test_a_rig_with_off_axis_tails_moves_its_joints_like_the_same_rig_with_tails_on_the_limb(tmp_path):
    """Retargeting aligns limbs by joint positions, so turning every limb and thumb tail off its limb (heads
    kept) must not move any joint, the palm's finger bases included, with the hands driven or not; the report
    names the off-axis bones. A forearm tail turned 90 degrees also pins the twist axis, which a reversed one
    cannot (its bone-space Y still lies along the forearm)."""
    py = blender_python()
    rigs = list((REPO / "assets/imported" / RIGS["biped"] / "assets/rigs").glob("*.blend"))
    if py is None or not WALK.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "off_axis.py"
    script.write_text(_OFF_AXIS)
    runs = {}
    for variant in ("on:hands", "off:hands", "on:nohands", "off:nohands"):
        out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(WALK), variant],
                             capture_output=True, text=True, timeout=900)
        line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
        assert line, out.stdout[-2000:] + out.stderr[-2000:]
        runs[variant] = json.loads(line[7:])
    import numpy as np
    for hands in ("hands", "nohands"):
        on, off = runs[f"on:{hands}"], runs[f"off:{hands}"]
        worst = max(float(np.linalg.norm(np.subtract(a[n], b[n]))) for a, b in zip(on["heads"], off["heads"]) for n in a)
        assert worst < 0.01, f"{hands}: a joint moved {worst * 1000:.1f} mm"
        assert not any("tails point away" in w for w in on["warnings"]), on["warnings"]
        warned = next((w for w in off["warnings"] if "tails point away" in w), "")
        assert all(b in warned for b in ("Forearm", "Calf", "Thigh")), (hands, off["warnings"])


_BODY = r"""
import json, sys, bpy
sys.path.insert(0, sys.argv[-1])
import mocap_retarget as mr
bpy.ops.wm.read_factory_settings(use_empty=True)
arm = bpy.data.objects.new("rig", bpy.data.armatures.new("rig"))
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.data.edit_bones
def bone(name, head, tail, parent=None, deform=True):
    b = eb.new(name); b.head, b.tail = head, tail; b.parent = eb[parent] if parent else None; b.use_deform = deform
bone("root", (0, 0, 0), (0, 0.2, 0), deform=False)          # an exporter's root, no skin
bone("prop_root", (1, 0, 0), (1, 0.2, 0), deform=True)      # a second root: a prop, not this body
bone("pelvis", (0, 0, 1.0), (0, 0, 1.1), "root")            # a Biped's pelvis, skinned at the waist
bone("spine", (0, 0, 1.1), (0, 0, 1.3), "pelvis")           # the thighs hang from here: the hips' role
for side, x in (("L", 0.1), ("R", -0.1)):
    bone("thigh." + side, (x, 0, 1.05), (x, 0, 0.55), "spine")
    bone("shin." + side, (x, 0, 0.55), (x, 0, 0.1), "thigh." + side)
bpy.ops.object.mode_set(mode="OBJECT")
me = bpy.data.meshes.new("skin")
me.from_pydata([(0, 0, 1.0), (0.1, 0, 0.8), (-0.1, 0, 0.8), (0, 0, 1.2), (1, 0, 0.1)], [], [])
skin = bpy.data.objects.new("skin", me); bpy.context.scene.collection.objects.link(skin)
skin.modifiers.new("Armature", "ARMATURE").object = arm
def weigh(group, vert, w=1.0):
    g = skin.vertex_groups.get(group) or skin.vertex_groups.new(name=group); g.add([vert], w, "REPLACE")
weigh("thigh.L", 1); weigh("thigh.R", 2); weigh("spine", 3); weigh("prop_root", 4)
out = {"no_skin_above": mr._body_bone(arm, "spine")}
weigh("pelvis", 0)
out["pelvis_skinned"] = mr._body_bone(arm, "spine")
out["hips_already"] = mr._body_bone(arm, "pelvis")
arm.data.bones["root"].use_deform = True; weigh("root", 0, 0.4)
out["stray_vertex_on_root"] = mr._body_bone(arm, "spine")
print("BODY " + json.dumps(out))
"""


def test_the_body_moves_from_the_bone_carrying_the_skin_not_from_a_hips_below_it(tmp_path):
    """On a Biped-style rig the thighs hang from a spine bone, so the hips' role lands there while the waist is skinned
    to the pelvis above it: moving the hips left the waist where the clip started, stretched across the floor. The
    body bone is the lowest bone carrying every skinned bone of the hips' own hierarchy (a prop under another root
    does not count), and the hips themselves when nothing above them is skinned."""
    py = blender_python()
    if py is None:
        pytest.skip("needs the dycraft-blender interpreter")
    script = tmp_path / "body.py"
    script.write_text(_BODY)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap")], capture_output=True, text=True,
                         timeout=300)
    line = next((l for l in out.stdout.splitlines() if l.startswith("BODY ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    assert json.loads(line[5:]) == {"no_skin_above": "spine", "pelvis_skinned": "pelvis", "hips_already": "pelvis",
                                    "stray_vertex_on_root": "root"}


_BAKE = r"""
import json, sys, bpy
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
rep = mr.retarget(arm, sys.argv[-1], start_s=1.0, end_s=6.0, name='walk')
rep['assigned_after'] = arm.animation_data.action.name if arm.animation_data.action else None
rep['bones'] = len(rep['bones'])
print('REPORT ' + json.dumps(rep))
"""


@pytest.mark.expensive
@pytest.mark.parametrize("family", sorted(RIGS))
def test_a_walk_retargets_grounded_and_planted(tmp_path, family):
    py = blender_python()
    rigs = list((REPO / "assets/imported" / RIGS[family] / "assets/rigs").glob("*.blend"))
    if py is None or not WALK.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "bake.py"
    script.write_text(_BAKE)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(WALK)],
                         capture_output=True, text=True, timeout=900)
    line = next((l for l in out.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["frames"] > 20 and rep["bones"] >= 16
    # The rig's own animation is untouched; the result is a separate action.
    assert rep["assigned_after"] is None and rep["action"] == "walk"
    # Planted soles on the floor, and no sole through it; the size of the
    # correction that got them there is the raw retarget's hover, not a quality.
    assert rep["ground_offset_m"]["max_abs"] < 0.15
    assert rep["stance_sole_gap_mm"]["p90"] < 10 and rep["sole_below_ground_mm"] < 5, rep
    assert rep["warnings"] == [], rep["warnings"]
    # Slip is the contact point's (the lowest sole point, followed as itself), not
    # the ankle's, which moves on purpose as the heel rises; the p90 is set by the
    # lock's ease in and out at the ends of each stance (mid-stance is ~0).
    slip = rep["stance_slip_p90_mps"]
    assert slip["after"] < 0.12 and slip["after"] < 0.5 * slip["before"]
    assert 1.0 < rep["root_travel_m"] < 3.5


@pytest.mark.expensive
def test_authored_code_in_the_sandbox_imports_the_helper_and_reads_the_library(tmp_path):
    """The path an agent actually takes: `h.blender` code, `/tools` on PYTHONPATH,
    the library at `/references/mocap`."""
    if blender_python() is None or not WALK.exists():
        pytest.skip("needs the dycraft-blender interpreter and the CMU library")
    from dycraft.tests._animation_fixtures import call, plan, service, source
    svc = service(tmp_path)
    try:
        (svc.design / "scene.py").write_text(source())
        assert call(svc, "plan", plan=plan())["ok"]
        assert call(svc, "build")["ok"]
        r = call(svc, "blender", stage="storyboard", code=(
            "import mocap_retarget as mr\n"
            "result = mr.inspect_bvh('/references/mocap/cmu/data/008/08_01.bvh')\n"))
        assert r["ok"], r
        info = r["result"]
        assert info["frames"] > 20 and info["root_displacement_m"] > 1
    finally:
        svc.jobs.close()


_EDITS = r"""
import json, sys, bpy
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
out = {}
walk = mr.retarget(arm, sys.argv[-1], start_s=1.0, end_s=7.0, name='walk')
out['cycle'] = mr.cycle(arm, walk['action'], seconds=8)
out['path'] = mr.follow_path(arm, out['cycle']['action'], [(0, 0), (0, 4), (3, 7)])
out['adjust'] = mr.adjust(arm, walk['action'], speed=0.9, stride=1.1)
out['fast'] = mr.adjust(arm, walk['action'], speed=1.5)
out['splice'] = mr.splice(arm, walk['action'], out['cycle']['action'])
out['assigned_after'] = arm.animation_data.action.name if arm.animation_data.action else None
print('REPORT ' + json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'bones'} if isinstance(v, dict) else v)
                              for k, v in out.items()}))
"""


@pytest.mark.expensive
def test_the_standard_edits_keep_feet_planted_and_warn_past_their_range(tmp_path):
    py = blender_python()
    clip = LIBRARY / "081" / "81_03.bvh"                  # ~9 s of steady walking
    rigs = list((REPO / "assets/imported" / RIGS["biped"] / "assets/rigs").glob("*.blend"))
    if py is None or not clip.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "edits.py"
    script.write_text(_EDITS)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(clip)],
                         capture_output=True, text=True, timeout=1500)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["assigned_after"] is None
    assert rep["cycle"]["frames"] == 192 and abs(rep["cycle"]["cycle_turn_deg"]) < 10
    for key in ("cycle", "path", "adjust", "fast", "splice"):
        # `cycle` keeps each stance's sole angle by taking a leg's seam residual in its
        # swing; a foot planted across the seam then moves ~1 cm on that one frame.
        assert rep[key]["stance_slip_p90_mps"]["after"] < (0.16 if key in ("cycle", "path", "splice") else 0.12), (key, rep[key])
        assert rep[key]["sole_below_ground_mm"] < 5, (key, rep[key])
    assert any("lengthen it with cycle" in w for w in rep["path"]["warnings"])
    assert rep["adjust"]["warnings"] == [] and rep["adjust"]["frames"] > rep["fast"]["frames"]
    assert any("past +-20%" in w for w in rep["fast"]["warnings"])
    assert rep["splice"]["blend_frames"] == 6 and -180 <= rep["splice"]["turned_deg"] <= 180


_SEAM = r"""
import json, math, sys, bpy
sys.path.insert(0, sys.argv[-5])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-4], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
w = mr.retarget(arm, sys.argv[-3], start_s=float(sys.argv[-2]), end_s=float(sys.argv[-1]), name='w')
c = mr.cycle(arm, 'w', cycles=3, name='c')
L, A = c['cycle_frames'], c['cycle_start'] - 1
root = w['bones']['Hips']
src, out = mr._roots(arm, bpy.data.actions['w'], root), mr._roots(arm, bpy.data.actions['c'], root)
ang = lambda r, i: math.degrees(r[i].to_quaternion().rotation_difference(r[i + 1].to_quaternion()).angle)
print('REPORT ' + json.dumps({'L': L, 'orig': ang(src, A + L - 1), 'seams': [ang(out, k * L - 1) for k in (1, 2)]}))
"""


@pytest.mark.expensive
@pytest.mark.parametrize("clip, start, end", [("002/02_03", 0.05, 1.40),   # jog
                                              ("016/16_36", 0.0, 3.0),     # jog: one cycle, on the right foot
                                              ("081/81_03", 1.0, 7.0)])    # walk
def test_a_cycle_seam_moves_like_the_clip_does_there(tmp_path, clip, start, end):
    """A cycle's seam moves the way the clip moves there: the root's orientation and height
    must not snap back at each seam, and a jog's noisy contact must not read as a 2-frame cycle."""
    py = blender_python()
    bvh = LIBRARY / clip.split("/")[0] / (clip.split("/")[1] + ".bvh")
    rigs = list((REPO / "assets/imported" / "rb_sports_male_01_rig" / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "seam.py"
    script.write_text(_SEAM)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh),
                          str(start), str(end)], capture_output=True, text=True, timeout=900)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["L"] >= 10
    for seam in rep["seams"]:
        assert abs(seam - rep["orig"]) < 1.0, rep


_SMOOTH = r"""
import json, sys, bpy
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
w = mr.retarget(arm, sys.argv[-1], name='w')
r = mr.smooth(arm, 'w', sigma_frames=1.0)
r['assigned_after'] = arm.animation_data.action.name if arm.animation_data.action else None
print('REPORT ' + json.dumps({k: v for k, v in r.items() if k != 'bones'}))
"""


@pytest.mark.expensive
def test_smoothing_removes_flicker_and_keeps_planted_feet(tmp_path):
    py = blender_python()
    bvh = LIBRARY / "005" / "05_02.bvh"                     # dance: fast arms, the worst flicker
    rigs = list((REPO / "assets/imported" / RIGS["biped"] / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "smooth.py"
    script.write_text(_SMOOTH)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh)],
                         capture_output=True, text=True, timeout=900)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    j = rep["jitter_p95_deg"]
    assert j["after"] < 0.6 * j["before"], j
    assert rep["stance_slip_p90_mps"]["after"] < 0.08
    assert rep["assigned_after"] is None


_JOG = r"""
import json, sys, bpy
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
body = max((o for o in bpy.context.scene.objects if o.type == 'MESH'
            and any(m.type == 'ARMATURE' and m.object is arm for m in o.modifiers)),
           key=lambda o: len(o.data.vertices))
out = {}
out['jog'] = mr.retarget(arm, sys.argv[-1], name='jog')
out['cycle'] = mr.cycle(arm, 'jog', seconds=4, name='cyc')
# A garment bound to the armature that does not deform (left at rest) must not
# pin the "lowest point" at the rest height and switch grounding off.
g = body.copy(); g.data = body.data.copy(); bpy.context.scene.collection.objects.link(g)
for m in g.modifiers:
    if m.type == 'ARMATURE':
        m.show_viewport = False
out['with_static_garment'] = mr.retarget(arm, sys.argv[-1], name='jog2')
print('REPORT ' + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'bones'} for k, v in out.items()}))
"""


@pytest.mark.expensive
def test_a_jog_on_an_uneven_capture_floor_plants_both_feet_and_cycles_evenly(tmp_path):
    """CMU 16_56: its right foot stands 5-6 cm higher in its first stance than in its
    second, so judged against the clip-wide lowest point that foot never plants (no
    lock, no sole calibration) and `cycle` cuts the wrong stride length. Each stance is
    judged against its own neighborhood, so both feet plant and the cycle is even; a
    static garment on the armature must not lift the ground."""
    py = blender_python()
    bvh = LIBRARY / "016" / "16_56.bvh"
    rigs = list((REPO / "assets/imported" / RIGS["biped"] / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "jog.py"
    script.write_text(_JOG)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh)],
                         capture_output=True, text=True, timeout=1500)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    jog, cyc, garm = rep["jog"], rep["cycle"], rep["with_static_garment"]
    assert all(c >= 2 for c in jog["sole_calibration_frames"].values()), jog
    assert min(jog["planted_fraction"]) > 0.2, jog
    assert jog["stance_sole_gap_mm"]["p90"] < 10 and jog["sole_below_ground_mm"] < 5, jog
    a, b = cyc["step_frames"]
    assert abs(a - b) <= 2 and 14 <= cyc["cycle_frames"] <= 20, cyc
    assert not any("uneven steps" in w for w in cyc["warnings"]), cyc
    assert abs(garm["ground_offset_m"]["median"] - jog["ground_offset_m"]["median"]) < 0.002, (garm, jog)
    assert garm["stance_sole_gap_mm"]["p90"] < 10, garm


_SOLE = r"""
import json, math, sys, bpy
from mathutils import Vector
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
rep = mr.retarget(arm, sys.argv[-1], name='jog')
meta = mr._meta(bpy.data.actions['jog']); m = meta['bones']; n = mr._length(bpy.data.actions['jog'])
T0 = arm.matrix_world.copy()
rest = {j: (T0 @ arm.data.bones[m[j]].matrix_local).to_3x3() for j in ('LeftFoot', 'RightFoot')}
lr = (T0 @ arm.data.bones[m['LeftUpLeg']].head_local) - (T0 @ arm.data.bones[m['RightUpLeg']].head_local)
fwd0 = Vector((lr.y, -lr.x, 0)).normalized()
planted = mr._contacts(arm, bpy.data.actions['jog'], meta)
roll = {j: [] for j in rest}                  # + = sole turned to face the other foot
with mr._Using(arm, bpy.data.actions['jog']):
    for f in range(n):
        mr._go(f + 1)
        for i, j in enumerate(rest):
            if planted[f, i]:
                R = (T0 @ arm.pose.bones[m[j]].matrix).to_3x3() @ rest[j].inverted()
                up = R @ Vector((0, 0, 1)); fw = R @ fwd0; fw.z = 0; fw.normalize()
                out = Vector((0, 0, 1)).cross(fw) * (1 if j == 'LeftFoot' else -1)
                roll[j].append(math.degrees(math.asin(max(-1.0, min(1.0, up.dot(out))))))
print('REPORT ' + json.dumps({'calib': rep['sole_calibration_frames'], 'warnings': rep['warnings'],
                              'roll': {j: sorted(v)[len(v) // 2] for j, v in roll.items()}}))
"""


@pytest.mark.expensive
def test_both_soles_land_flat_on_a_short_jog(tmp_path):
    """CMU 16_36: its left foot has only 4 flat-stance frames, and still gets its own
    sole calibration (a rest-pose fallback rolls the sole ~30 degrees inward)."""
    py = blender_python()
    bvh = LIBRARY / "016" / "16_36.bvh"
    rigs = list((REPO / "assets/imported" / "rb_sports_male_01_rig" / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "sole.py"
    script.write_text(_SOLE)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh)],
                         capture_output=True, text=True, timeout=900)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    assert all(c >= 2 for c in rep["calib"].values()) and rep["warnings"] == [], rep
    # median roll of the planted sole about the foot's length: flat, and both feet alike
    roll = rep["roll"]
    assert all(abs(r) < 8 for r in roll.values()) and abs(roll["LeftFoot"] - roll["RightFoot"]) < 8, rep


_CYCLE_STANCE = r"""
import json, math, sys, bpy
from mathutils import Vector
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
mr.retarget(arm, sys.argv[-1], name='jog')
mr.cycle(arm, 'jog', seconds=6, name='cyc')
T0 = arm.matrix_world.copy()
out = {}
for name in ('jog', 'cyc'):
    act = bpy.data.actions[name]; meta = mr._meta(act); m = meta['bones']; n = mr._length(act)
    rest = {j: (T0 @ arm.data.bones[m[j]].matrix_local).to_3x3() for j in ('LeftFoot', 'RightFoot')}
    lr = (T0 @ arm.data.bones[m['LeftUpLeg']].head_local) - (T0 @ arm.data.bones[m['RightUpLeg']].head_local)
    fwd0 = Vector((lr.y, -lr.x, 0)).normalized()
    planted = mr._contacts(arm, act, meta)
    pitch = {j: [] for j in rest}               # + = toe down
    with mr._Using(arm, act):
        for f in range(n):
            mr._go(f + 1)
            for i, j in enumerate(rest):
                if planted[f, i]:
                    R = (T0 @ arm.pose.bones[m[j]].matrix).to_3x3() @ rest[j].inverted()
                    up = R @ Vector((0, 0, 1)); fw = R @ fwd0; fw.z = 0; fw.normalize()
                    pitch[j].append(math.degrees(math.asin(max(-1.0, min(1.0, up.dot(fw))))))
    out[name] = {j: sorted(v)[len(v) // 2] for j, v in pitch.items()}
print('REPORT ' + json.dumps(out))
"""


@pytest.mark.expensive
def test_a_cycle_leaves_each_stance_as_the_clip_had_it(tmp_path):
    """CMU 16_56: a leg takes the cycle's seam residual only in the swing running into the
    seam, so every stance keeps the clip's sole angle (spread over the whole cycle, the
    right foot's toe-off/swing mismatch tips its mid-cycle stance ~25 degrees toe-up)."""
    py = blender_python()
    bvh = LIBRARY / "016" / "16_56.bvh"
    rigs = list((REPO / "assets/imported" / "rb_sports_male_01_rig" / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "cyc.py"
    script.write_text(_CYCLE_STANCE)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh)],
                         capture_output=True, text=True, timeout=900)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    for foot in ("LeftFoot", "RightFoot"):
        assert abs(rep["cyc"][foot] - rep["jog"][foot]) < 6, rep
    assert abs(rep["cyc"]["LeftFoot"] - rep["cyc"]["RightFoot"]) < 8, rep


_CLAVICLE = r"""
import json, math, sys, bpy
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
rep = mr.retarget(arm, sys.argv[-1], name='walk')
m = rep['bones']; M = arm.matrix_world
def rel(chest, clav, posed):
    c = (arm.pose.bones if posed else arm.data.bones)
    mc = M @ (c[chest].matrix if posed else c[chest].matrix_local)
    mv = M @ (c[clav].matrix if posed else c[clav].matrix_local)
    return mc.to_quaternion().inverted() @ mv.to_quaternion()
off = []
chest = m.get('Spine1') or m['Spine']
with mr._Using(arm, bpy.data.actions['walk']):
    for f in range(1, mr._length(bpy.data.actions['walk']) + 1, 4):
        mr._go(f)
        for j in ('LeftShoulder', 'RightShoulder'):
            if j in m:
                off.append(math.degrees(rel(chest, m[j], True).rotation_difference(rel(chest, m[j], False)).angle))
print('REPORT ' + json.dumps({'max_deg': max(off) if off else None}))
"""


@pytest.mark.expensive
def test_clavicles_stay_where_the_rig_holds_them(tmp_path):
    """Aligning the clavicle by bone direction would copy CMU's, which slopes up from the
    neck base, onto a Biped's nearly level one: a constant shrug of ~30 degrees. CMU never
    moves the clavicle relative to the chest, so the target's must not move either."""
    py = blender_python()
    rigs = list((REPO / "assets/imported" / "rb_sports_male_01_rig" / "assets/rigs").glob("*.blend"))
    if py is None or not WALK.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "clav.py"
    script.write_text(_CLAVICLE)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(WALK)],
                         capture_output=True, text=True, timeout=900)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["max_deg"] is not None and rep["max_deg"] < 3, rep


_LEAN = r"""
import json, math, sys, bpy
import numpy as np
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
jog = mr.retarget(arm, sys.argv[-1], name='jog')
mr.cycle(arm, 'jog', seconds=9, name='jogc')
R = 2.2                                  # a garden path's corner radius
arc = lambda cx, cy, a0: [(cx + R * math.cos(a), cy + R * math.sin(a)) for a in np.linspace(a0, a0 + math.pi / 2, 12)[1:]]
pts = [(0, 0), (3, 0)] + arc(3, R, -math.pi / 2) + [(3 + R, R + 3)] + arc(3, R + 3, 0) + [(0, 2 * R + 3)]
M, m = arm.matrix_world, jog['bones']
def track(act):
    arm.animation_data.action = bpy.data.actions[act]
    n = mr._length(bpy.data.actions[act]); P, U = [], []
    for f in range(1, n + 1):
        bpy.context.scene.frame_set(f)
        hip, head = ((M @ arm.pose.bones[m[b]].matrix).translation for b in ('Hips', 'Head'))
        P.append(list(hip)); U.append(list(head - hip))
    P, U = np.array(P), np.array(U)
    v = np.gradient(P[:, :2], axis=0); fwd = v / np.linalg.norm(v, axis=1, keepdims=True)
    side = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
    acc = P[2:] - 2 * P[1:-1] + P[:-2]
    tilt = np.degrees(np.arctan2(np.sum(U[:, :2] * side, 1), U[:, 2]))
    return {'side_acc_p95_mm': float(np.percentile(np.abs(np.sum(acc[:, :2] * side[1:-1], 1)), 95) * 1000),
            'tilt_max': float(tilt.max())}
out = {}
for lean in (0.0, 1.0):
    r = mr.follow_path(arm, 'jogc', pts, name=f'p{lean:g}', lean=lean)
    out[f'lean{lean:g}'] = {**track(f'p{lean:g}'), 'lean_deg': r['lean_deg'],
                            'slip': r['stance_slip_p90_mps']['after'], 'below': r['sole_below_ground_mm']}
print('REPORT ' + json.dumps(out))
"""


@pytest.mark.expensive
def test_follow_path_leans_into_turns_without_pumping_each_step(tmp_path):
    """A lean from the pelvis's per-frame speed, which rises and falls inside every stride,
    swings several degrees each step and shakes the upper body sideways. follow_path's lean
    uses stride-averaged speed and curvature, and keeps the feet out of the tilt so a
    push-off's rise is not sideways slip."""
    py = blender_python()
    bvh = LIBRARY / "016" / "16_36.bvh"
    rigs = list((REPO / "assets/imported" / RIGS["biped"] / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "lean.py"
    script.write_text(_LEAN)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh)],
                         capture_output=True, text=True, timeout=1500)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    flat, lean = rep["lean0"], rep["lean1"]
    assert flat["lean_deg"]["max"] == 0
    # ~2.5 m/s round a 2.2 m radius wants ~16 degrees; it builds in, it does not jump.
    assert 10 <= lean["lean_deg"]["max"] <= 15 and lean["lean_deg"]["max_frame_step"] < 1.5, rep
    assert lean["tilt_max"] > flat["tilt_max"] + 8, rep
    assert lean["side_acc_p95_mm"] < 1.5 * flat["side_acc_p95_mm"], rep
    assert lean["slip"] <= flat["slip"] + 0.02 and lean["below"] < 5, rep


_KNEEL = r"""
import json, math, sys, bpy
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
sc = bpy.context.scene
sc.render.fps = 24
arm = next(o for o in sc.objects if o.type == 'ARMATURE')
rep = mr.retarget(arm, sys.argv[-1], end_s=8.0, name='kneel')
m, pb = rep['bones'], arm.pose.bones
arm.animation_data_create()
arm.animation_data.action = bpy.data.actions['kneel']
off = {'Left': [], 'Right': []}                 # degrees the foot turns out of its leg's plane
for f in range(1, rep['frames'] + 1):
    sc.frame_set(f)
    for side, v in off.items():
        end = pb[m[side + 'ToeBase']].head if side + 'ToeBase' in m else pb[m[side + 'Foot']].tail
        h, k, a, t = (arm.matrix_world @ p for p in (pb[m[side + 'UpLeg']].head, pb[m[side + 'Leg']].head,
                                                      pb[m[side + 'Foot']].head, end))
        n = (k - h).cross(a - k).normalized()
        v.append(math.degrees(math.asin(min(1.0, abs((t - a).normalized().dot(n))))))
print('REPORT ' + json.dumps({'warnings': rep['warnings'], 'median': {s: sorted(v)[len(v) // 2] for s, v in off.items()},
                              'max': {s: max(v) for s, v in off.items()}}))
"""


@pytest.mark.expensive
@pytest.mark.parametrize("rig", ["rb_business_male_01_rig", "sf_chimp_sfm_rig"])
def test_a_foot_kneeling_on_its_toes_keeps_to_its_leg(tmp_path, rig):
    """CMU 23_03 kneels for about three seconds with the rear (right) foot on its toes, low and
    still, so its frames pass the planted test; they are too steep to calibrate a flat sole.
    Calibrated on them, the right foot turns up to 70 degrees out of its leg's plane through the
    whole clip, where the actor's stays within 2-30."""
    py = blender_python()
    bvh = LIBRARY / "023" / "23_03.bvh"
    rigs = list((REPO / "assets/imported" / rig / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "kneel.py"
    script.write_text(_KNEEL)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh)],
                         capture_output=True, text=True, timeout=900)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["warnings"] == [], rep
    # Like the actor's feet: a median of about 8 degrees, and no frame far past the actor's worst.
    assert rep["median"]["Right"] < 15 and rep["max"]["Right"] < 40, rep
    assert rep["median"]["Left"] < 15, rep



_LONG_SHOES = r"""
import json, math, sys, bpy
import numpy as np
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
sc = bpy.context.scene
sc.render.fps = 24
arm = next(o for o in sc.objects if o.type == 'ARMATURE')
m = mr.bone_map(arm)
rep = mr.retarget(arm, sys.argv[-1], name='jog')
cy = mr.cycle(arm, rep['action'], seconds=3.0)


def knees(name):
    act = bpy.data.actions[name]
    arm.animation_data_create()
    arm.animation_data.action = act
    out = []
    for f in range(1, int(act.frame_range[1]) + 1):
        sc.frame_set(f)
        W, pb = arm.matrix_world, arm.pose.bones
        out.append([math.degrees((W @ pb[m[s + 'Leg']].head - W @ pb[m[s + 'UpLeg']].head)
                                 .angle(W @ pb[m[s + 'Foot']].head - W @ pb[m[s + 'Leg']].head))
                    for s in ('Left', 'Right')])
    return np.array(out)


def foot_turns(name):
    act = bpy.data.actions[name]
    arm.animation_data.action = act
    q = []
    for f in range(1, int(act.frame_range[1]) + 1):
        sc.frame_set(f)
        q.append([(arm.matrix_world @ arm.pose.bones[m[s + 'Foot']].matrix).to_quaternion() for s in ('Left', 'Right')])
    return np.array([[math.degrees(q[f][i].rotation_difference(q[f - 1][i]).angle) for i in (0, 1)]
                     for f in range(1, len(q))])


free = mr.retarget(arm, sys.argv[-1], name='free', ground=True, foot_lock=False)
raw, loop, unlocked = knees(rep['action']), knees(cy['action']), knees(free['action'])
print('REPORT ' + json.dumps({'steps': cy['step_frames'], 'warnings': cy['warnings'],
                              'raw_median': np.median(raw, axis=0).tolist(),
                              'unlocked_median': np.median(unlocked, axis=0).tolist(),
                              'cycle_median': np.median(loop, axis=0).tolist(),
                              'cycle_min': loop.min(axis=0).tolist(),
                              'turn_beyond_unlocked': (foot_turns(rep['action'])
                                                       - foot_turns(free['action'])).max(axis=0).tolist(),
                              'below_ground_mm': rep['sole_below_ground_mm']}))
"""


@pytest.fixture(scope="module")
def long_shoes(tmp_path_factory):
    """The clown's 35_22 jog through `_LONG_SHOES`, run once for the two tests that read it."""
    py = blender_python()
    bvh = LIBRARY / "035" / "35_22.bvh"
    rigs = list((REPO / "assets/imported" / "clown_circus_classic_circus_clown_adult_male__45aab083"
                 / "assets/rigs").glob("*.blend"))
    if py is None or not bvh.exists() or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported clown")
    script = tmp_path_factory.mktemp("long_shoes") / "long_shoes.py"
    script.write_text(_LONG_SHOES)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]), str(bvh)],
                         capture_output=True, text=True, timeout=900)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    return json.loads(line[7:])


@pytest.mark.expensive
def test_a_cycle_on_long_shoes_repeats_the_capture_s_stride(long_shoes):
    """A long shoe's toe, lifted to the floor where it dips in mid-swing, sits on the floor
    there and moves slowly, so contacts read off the character's soles find a landing in the
    swing. A cycle then repeats a stride that starts there, with that foot pinned mid-swing,
    and the leg stays bent through every repeat. The edits keep only the contacts the source
    skeleton has, which the retarget locked with: CMU 35_22's stride is 9 frames and 9."""
    rep = long_shoes
    assert rep["steps"] and abs(rep["steps"][0] - rep["steps"][1]) <= 2, rep
    assert not any("uneven" in w for w in rep["warnings"]), rep
    # Each leg bends in the cycle as it does in the retarget it repeats, and straightens.
    for side in (0, 1):
        assert abs(rep["cycle_median"][side] - rep["raw_median"][side]) < 8, rep
        assert rep["cycle_min"][side] < 35, rep


@pytest.mark.expensive
def test_long_shoes_are_pitched_clear_of_the_floor_in_a_swing_not_lifted_by_the_knee(long_shoes):
    """A long shoe pitched toe-down at push-off, or swung through low, dips 10-18 cm below the
    floor where the actor's toes clear it, and lifting it that far by the ankle keeps the knee
    bent through every swing. The foot is pitched toe-up instead, so the knees bend about as
    they do with no lock, along one curve per swing that turns no faster than
    SWING_PITCH_RATE_DEG_S, so no frame turns the foot much further than the capture does."""
    rep = long_shoes
    for side in (0, 1):
        assert rep["raw_median"][side] - rep["unlocked_median"][side] < 8, rep
        assert rep["turn_beyond_unlocked"][side] < 15, rep
    assert rep["below_ground_mm"] < 5, rep


def test_only_a_foot_reaching_far_past_its_ankle_is_long(monkeypatch):
    """The swing pitch is for oversized feet and shoes only: a human-proportioned foot, about a
    quarter to a third of a leg ahead of its ankle, is lifted as it always was, however deep a
    dance or a crawl takes it. A sole centered beside the ankle names no direction to pitch in."""
    mr = _module(monkeypatch)
    hip, knee, ankle = [0.0, 0.0, 0.9], [0.0, 0.0, 0.48], [0.0, 0.0, 0.08]   # an 0.82 m leg

    def sole(reach):
        return [[x, y, 0.0] for x in (-0.06, reach) for y in (-0.04, 0.04)]
    assert mr._long_foot(sole(0.26), ankle, knee, hip) is None              # a person's: 0.32 of a leg
    ahead = mr._long_foot(sole(0.55), ankle, knee, hip)                      # a clown's shoe: 0.67
    assert ahead is not None and abs(ahead[0] - 1.0) < 1e-6
    assert mr._long_foot([[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]], ankle, knee, hip) is None


_PEOPLE_UNPITCHED = r"""
import json, sys, bpy
import numpy as np
sys.path.insert(0, sys.argv[-3])
import mocap_retarget as mr
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2], load_ui=False)
bpy.context.scene.render.fps = 24
arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
seen = {'axes': [], 'pitch': 0.0}
axes, plan = mr._pitch_axes, mr._swing_pitch
def spy_axes(*a):
    out = axes(*a)
    seen['axes'].append([u is not None for u in out])
    return out
def spy_plan(*a):
    p = plan(*a)
    seen['pitch'] = max(seen['pitch'], float(np.abs(p).max()))
    return p
mr._pitch_axes, mr._swing_pitch = spy_axes, spy_plan
for clip in sys.argv[-1].split(','):
    mr.retarget(arm, clip, end_s=8.0, name='p')
print('REPORT ' + json.dumps(seen))
"""


@pytest.mark.expensive
def test_a_person_s_feet_are_never_pitched_in_a_walk_a_dance_or_a_crawl(tmp_path):
    """A person's swing foot dips up to about 11 cm below the floor in a salsa or a crawl; it is
    lifted, as before, and never pitched, since the pitch is only for oversized feet and shoes."""
    py = blender_python()
    clips = [LIBRARY / c for c in ("008/08_01.bvh", "060/60_01.bvh", "111/111_03.bvh")]
    rigs = list((REPO / "assets/imported" / RIGS["biped"] / "assets/rigs").glob("*.blend"))
    if py is None or not all(c.exists() for c in clips) or not rigs:
        pytest.skip("needs the dycraft-blender interpreter, the CMU library and the imported rig")
    script = tmp_path / "people.py"
    script.write_text(_PEOPLE_UNPITCHED)
    res = subprocess.run([str(py), str(script), str(REPO / "animation/mocap"), str(rigs[0]),
                          ",".join(str(c) for c in clips)], capture_output=True, text=True, timeout=1800)
    line = next((l for l in res.stdout.splitlines() if l.startswith("REPORT ")), None)
    assert line, res.stdout[-2000:] + res.stderr[-2000:]
    rep = json.loads(line[7:])
    assert rep["axes"] and not any(any(a) for a in rep["axes"]), rep
    assert rep["pitch"] == 0.0, rep
