"""Retarget CMU motion capture onto a humanoid character, and make the usual edits.

Runs inside the Blender sandbox: the `mocap_reference` skill tells an author to
`import mocap_retarget` from code passed to `h.blender` / `h.probe`, and
`runtime.command` binds this file read-only at `/tools/mocap_retarget.py`.

    import mocap_retarget as mr
    arm = bpy.data.objects['rig/trainer/body']
    walk = mr.retarget(arm, '/references/mocap/cmu/data/008/08_01.bvh', start_s=0.5, end_s=4.0)
    loop = mr.cycle(arm, walk['action'], seconds=12)          # a longer steady walk
    route = mr.follow_path(arm, loop['action'], [(0, 0), (2, 1), (4, 1)])
    slow = mr.adjust(arm, route['action'], speed=0.9)
    both = mr.splice(arm, slow['action'], stop['action'])     # e.g. walk into a stop clip
    clean = mr.smooth(arm, both['action'], sigma_frames=1.0)  # remove marker/retarget flicker

Every operation writes a *new* Action keyed on frames 1..N and leaves the
armature's own action and NLA as they were; placing, layering, and editing the
result is the author's job. These are conveniences, not the only way: an author
may edit the keys directly, layer corrections on NLA tracks, or write their own
retarget. The operations are the standard ones (cycle extraction, path
warping, time warping, stride warping, phase-aligned blending) and they hold up
for *small* changes. Each one reports what it did and warns past the range where
the result stops looking natural -- the better fix there is a different clip.

A retarget written from scratch trips over the same five traps -- rest poses, bone
axes, facing, proportions, ground contact (the skill lists them) -- so this module
solves them once.

Retarget method, per mapped bone:

    A(t)     = R_src(t) R_src_rest^-1      world rotation the source bone underwent
    C        = rotation taking the target's rest segment to the source's
    R_tgt(t) = A(t) C R_tgt_rest

so each limb points where the actor's limb pointed, whatever the two rest poses
(CMU T-pose against an A-pose character). Torso, neck, and head take A alone:
their bone *axes* disagree across rigs (an FBX Biped head bone points at the jaw),
and aligning them tips heads back by ~90 degrees. Hip translation is scaled by
the ratio of leg lengths. Then the corrections every edit here reuses:

* ground -- on frames where a foot is planted, the root is moved along world up
  so that foot's sole sits at the rest pose's sole height (a raw retarget
  hovers several centimeters); between contacts the correction is interpolated, so a run
  keeps its flight phase. The sole is measured on the skin's foot vertices
  carried rigidly by the foot bones, so clothes, hidden meshes, or a garment that
  does not deform cannot move the ground;
* foot lock -- while a foot is planted (low for its neighborhood and slow) the
  root is shifted to cancel that foot's drift, and two-bone leg IK pins it where
  it landed and sets its sole on the ground, with a short ease in and out.

Unmapped bones (fingers, face, twist) keep their rest pose.
"""
from __future__ import annotations

import json
import math
import os
import re

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

#: CMU joint -> candidate bone names after `normalise`, lower-case, per family.
CANDIDATES = {
    "Hips": ["bip01 pelvis", "hips", "pelvis", "hip"],
    "LowerBack": ["bip01 spine", "spine", "spine_01"],
    "Spine": ["bip01 spine1", "spine1", "spine_02"],
    "Spine1": ["bip01 spine2", "spine2", "spine_03"],
    "Neck": ["bip01 neck", "neck", "neck_01"],
    "Head": ["bip01 head", "head"],
    "LeftShoulder": ["bip01 l clavicle", "leftshoulder", "clavicle_l", "shoulder_l"],
    "LeftArm": ["bip01 l upperarm", "leftarm", "upperarm_l"],
    "LeftForeArm": ["bip01 l forearm", "leftforearm", "lowerarm_l"],
    "LeftHand": ["bip01 l hand", "lefthand", "hand_l"],
    "RightShoulder": ["bip01 r clavicle", "rightshoulder", "clavicle_r", "shoulder_r"],
    "RightArm": ["bip01 r upperarm", "rightarm", "upperarm_r"],
    "RightForeArm": ["bip01 r forearm", "rightforearm", "lowerarm_r"],
    "RightHand": ["bip01 r hand", "righthand", "hand_r"],
    "LeftUpLeg": ["bip01 l thigh", "leftupleg", "thigh_l", "upperleg_l"],
    "LeftLeg": ["bip01 l calf", "leftleg", "calf_l", "lowerleg_l"],
    "LeftFoot": ["bip01 l foot", "leftfoot", "foot_l"],
    "RightUpLeg": ["bip01 r thigh", "rightupleg", "thigh_r", "upperleg_r"],
    "RightLeg": ["bip01 r calf", "rightleg", "calf_r", "lowerleg_r"],
    "RightFoot": ["bip01 r foot", "rightfoot", "foot_r"],
}
#: Joints whose world rotation change is copied as is (C = identity).
AXIAL = {"Hips", "LowerBack", "Spine", "Spine1", "Neck", "Head"}
#: Clavicles are copied like the axial bones -- each moves from its own rest by what
#: the source's moves from its rest -- not aligned by bone direction. CMU's clavicle
#: rises from the neck base to the shoulder joint while a Biped's is nearly level,
#: so matching directions would hold every retargeted clavicle ~30 degrees up, a
#: constant shrug. CMU never moves the clavicle relative to the chest, so the
#: target's stays at its rest.
CLAVICLES = {"LeftShoulder", "RightShoulder"}
#: Without these the result is not a body; anything else may be missing.
REQUIRED = ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot")
#: Where an edit stops looking like the clip it came from (measured guidance, not
#: a hard limit): beyond these the report warns and suggests another clip.
SPEED_RANGE, STRIDE_RANGE, TURN_RATE_DEG_S = 0.2, 0.15, 90.0
#: Pronation/supination: how far the forearm may turn the palm from the upper arm.
FOREARM_TWIST_DEG = 90.0
#: follow_path's lean into a turn: tan(lean) = v^2 * curvature / g, with the speed
#: and curvature smoothed over LEAN_SIGMA_S. A runner's pelvis speeds up and slows
#: down inside every stride, so a lean from the per-frame speed would swing several
#: degrees each step and throw the upper body sideways. A Gaussian of 0.3 s leaves
#: ~3 % of a 1.4 Hz stride wobble.
LEAN_SIGMA_S, LEAN_MAX_DEG, GRAVITY = 0.3, 15.0, 9.81
META = "dycraft_mocap"
#: The contact mask an action was locked with, one character per frame (bit 0 the left
#: foot, bit 1 the right), so an edit keeps only the contacts the capture has (`_contacts`).
#: On feet far from human proportions the soles alone mislead: the lock lifts a long
#: shoe's dipping toe to the floor in mid-swing, where it then reads as a landing, and
#: `cycle` repeats a stride that starts there.
CONTACTS = "dycraft_mocap_contacts"


def normalise(name):
    """`mixamorig:LeftArm` -> `leftarm`, `upperarm_l_024` -> `upperarm_l`."""
    return re.sub(r"[_.]\d+$", "", name.split(":")[-1]).lower()


def bone_map(armature):
    """`{cmu_joint: bone_name}` for this armature; raises if the legs are missing."""
    by_norm = {}
    for b in armature.data.bones:
        by_norm.setdefault(normalise(b.name), b.name)
    found = {}
    for joint, names in CANDIDATES.items():
        for n in names:
            if n in by_norm:
                found[joint] = by_norm[n]
                break
    missing = [j for j in REQUIRED if j not in found]
    if missing:
        raise ValueError(
            f"{armature.name}: no bone for {missing}. This helper knows 3ds Max Biped, "
            "Mixamo/ReadyPlayerMe and Unreal-style names (see references/mocap/cmu/JOINT_MAPS.md); "
            "for another humanoid rig pass mapping={cmu_joint: bone_name}. It does not "
            "apply to quadrupeds or birds.")
    return found


# --------------------------------------------------------------------- BVH input

#: Where the capture libraries appear to Blender code, as the skill names them (`runtime.MOCAP_MOUNT`).
LIBRARY_MOUNT = "/references/mocap"


def _library_path(path):
    """`path`, with the documented `/references/mocap/…` taken to the library's real root where the worker sees host
    paths (`DYCRAFT_MOCAP_ROOT`, `runtime.sandbox_mode`): the same name works in either mode."""
    path, root = str(path), os.environ.get("DYCRAFT_MOCAP_ROOT")
    if root and (path == LIBRARY_MOUNT or path.startswith(LIBRARY_MOUNT + "/")):
        return root + path[len(LIBRARY_MOUNT):]
    return path


def _import_bvh(path):
    path = _library_path(path)
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=path, global_scale=1.0, frame_start=1, use_fps_scale=True,
                            update_scene_fps=False, update_scene_duration=False,
                            rotate_mode="NATIVE", axis_forward="-Z", axis_up="Y")
    return next(o for o in bpy.data.objects if o not in before and o.type == "ARMATURE")


def _remove(src):
    act = src.animation_data.action if src.animation_data else None
    data = src.data
    bpy.data.objects.remove(src, do_unlink=True)
    if data.users == 0:
        bpy.data.armatures.remove(data)
    if act is not None and act.users == 0:
        bpy.data.actions.remove(act)


def inspect_bvh(path):
    """Length and root path of a clip, without touching the scene.

    Frame 1 of the cgspeed CMU conversion is a T-pose and is excluded. Distances
    use the usual 0.056 m per unit, so they are a guide to how far the actor travels.
    """
    scene = bpy.context.scene
    src, keep = _import_bvh(path), scene.frame_current
    try:
        f0, f1 = (int(v) for v in src.animation_data.action.frame_range)
        f0 += 1
        fps = scene.render.fps / scene.render.fps_base
        hips = src.pose.bones["Hips"]
        pts = []
        for f in range(f0, f1 + 1, max(1, (f1 - f0) // 60)):
            scene.frame_set(f)
            pts.append((src.matrix_world @ hips.head).copy())
        path_len = sum((b - a).to_2d().length for a, b in zip(pts, pts[1:])) * 0.056
        return {"frames": f1 - f0 + 1, "fps": fps, "seconds": round((f1 - f0 + 1) / fps, 3),
                "root_travel_m": round(path_len, 2),
                "root_displacement_m": round((pts[-1] - pts[0]).to_2d().length * 0.056, 2)}
    finally:
        scene.frame_set(keep)
        _remove(src)


# ----------------------------------------------------------------- shared plumbing

class _Using:
    """Evaluate `action` alone on `armature` for the duration; restore after."""

    def __init__(self, armature, action):
        self.arm, self.action = armature, action

    def __enter__(self):
        ad = self.arm.animation_data_create()
        self.saved = (ad.action, getattr(ad, "action_slot", None), ad.use_nla,
                      bpy.context.scene.frame_current)
        ad.use_nla = False
        ad.action = self.action
        if self.action is not None and getattr(self.action, "slots", None) and hasattr(ad, "action_slot"):
            if ad.action_slot is None and len(self.action.slots):
                ad.action_slot = self.action.slots[0]
        return self

    def __exit__(self, *exc):
        ad = self.arm.animation_data
        ad.action = self.saved[0]
        if self.saved[1] is not None and hasattr(ad, "action_slot"):
            try:
                ad.action_slot = self.saved[1]
            except Exception:
                pass
        ad.use_nla = self.saved[2]
        bpy.context.scene.frame_set(self.saved[3])
        return False


def _go(frame, sub=0.0):
    bpy.context.scene.frame_set(int(frame), subframe=float(sub))
    bpy.context.view_layer.update()


def _meta(action):
    try:
        return json.loads(action[META])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"action {action.name!r} was not made by mocap_retarget; "
                         "these edits need its bone map (or edit the keys yourself)")


def _store_contacts(action, planted):
    """Record `planted` (frames x 2) on `action` (`CONTACTS`)."""
    action[CONTACTS] = "".join(str(int(bool(l)) + 2 * int(bool(r))) for l, r in planted)


def _stored_contacts(action):
    """The mask `_store_contacts` recorded (frames x 2), or None when there is none or
    its length no longer matches the action's."""
    s = action.get(CONTACTS)
    if not isinstance(s, str) or len(s) != _length(action) or not set(s) <= set("0123"):
        return None
    v = np.array([int(c) for c in s])
    return np.stack([(v & 1) > 0, (v & 2) > 0], axis=1)


def _resolve(action):
    return bpy.data.actions[action] if isinstance(action, str) else action


def _length(action):
    return int(round(action.frame_range[1]))


def _skin_meshes(armature):
    return [o for o in bpy.context.scene.objects if o.type == "MESH"
            and any(m.type == "ARMATURE" and m.object is armature for m in o.modifiers)]


def _skinned_bones(armature):
    """The deforming bones a skinned mesh actually weights a vertex to."""
    bones = armature.data.bones
    out = set()
    for o in _skin_meshes(armature):
        names = {g.index: g.name for g in o.vertex_groups}
        for v in o.data.vertices:
            for g in v.groups:
                name = names.get(g.group)
                if g.weight > 0 and name in bones and bones[name].use_deform:
                    out.add(name)
    return out


def _body_bone(armature, hips):
    """The bone that carries the whole skin: the lowest common ancestor of every skinned bone in the hips' root
    hierarchy. Most rigs weight nothing above the hips, so it is `hips`. Some do: a rig whose thighs hang from the
    spine has the hips' role on a spine bone and the waist skinned to a bone above it, and an exporter may weight a
    stray vertex or two to the root. Moving the hips alone would leave that skin where the clip started, stretched
    across the floor behind the walk."""
    bones = armature.data.bones
    def chain(name):
        out, b = [], bones[name]
        while b is not None:
            out.append(b.name)
            b = b.parent
        return out
    top = chain(hips)[-1]
    common = None
    for name in _skinned_bones(armature):
        c = chain(name)
        if c[-1] != top:                       # under another root: a prop or a control, not this body
            continue
        common = c if common is None else [a for a in c if a in common]
    if not common or common[0] == hips or common[0] not in chain(hips)[1:]:
        return hips
    return common[0]


def _body(meta):
    """The bone every edit moves the body by: the hips, or the skinned bone above them (`_body_bone`)."""
    return meta.get("body") or meta["bones"]["Hips"]


def _keyed(meta):
    """Every bone a clip keys: the mapped joints, and the body bone when it is not one of them."""
    return list(dict.fromkeys([*meta["bones"].values(), *([meta["body"]] if meta.get("body") else [])]))


def _sample(armature, action, bones, times):
    """Basis (location, rotation) of `bones` at each of `times` (fractional frames)."""
    out = []
    with _Using(armature, action):
        for t in times:
            _go(math.floor(t), t - math.floor(t))
            out.append({b: (armature.pose.bones[b].location.copy(),
                            armature.pose.bones[b].rotation_quaternion.copy()) for b in bones})
    return out


def _new_action(armature, name, meta, frames, root_world=None):
    """Key `frames` (list of {bone: (loc, quat)}) into a new action; optionally
    override the root bone's world matrix per frame."""
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    action[META] = json.dumps(meta)
    root = _body(meta)
    T0 = armature.matrix_world.copy()
    with _Using(armature, action):
        for pb in armature.pose.bones:
            pb.rotation_mode = "QUATERNION"
        for k, pose in enumerate(frames):
            f = k + 1
            _go(f)
            for b, (loc, q) in pose.items():
                armature.pose.bones[b].location = loc
                armature.pose.bones[b].rotation_quaternion = q
            if root_world is not None:
                bpy.context.view_layer.update()
                armature.pose.bones[root].matrix = T0.inverted() @ root_world[k]
            for b in pose:
                armature.pose.bones[b].keyframe_insert("rotation_quaternion", frame=f)
            armature.pose.bones[root].keyframe_insert("location", frame=f)
    return action


def _two_bone(H, K, A, T):
    """Knee and ankle for hip `H` reaching `T`, keeping the bend plane of `H, K, A`."""
    l1, l2 = (K - H).length, (A - K).length
    d_vec = T - H
    d = min(max(d_vec.length, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    u = d_vec.normalized()
    pole = (K - H) - u * (K - H).dot(u)
    if pole.length < 1e-6:
        pole = (A - H).cross(u).cross(u)
    pole.normalize()
    along = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    return H + u * along + pole * math.sqrt(max(0.0, l1 * l1 - along * along)), H + u * d


def _aim(armature, T0, bone, head, tail_from, tail_to):
    """Rotate `bone` in world space so the segment it carries points at `tail_to`."""
    pb = armature.pose.bones[bone]
    world = T0 @ pb.matrix
    q = (tail_from - head).rotation_difference(tail_to - head)
    new = Matrix.Translation(world.translation) @ q.to_matrix().to_4x4() @ Matrix.Translation(-world.translation) @ world
    pb.matrix = T0.inverted() @ new


def _feet(armature, T0, feet, n):
    out = []
    for f in range(1, n + 1):
        _go(f)
        out.append([list(T0 @ armature.pose.bones[b].head) for b in feet])
    return np.array(out)                                             # (n, 2, 3)


#: A foot's height is judged against its lowest point within this many seconds
#: either side (less than a running stride, so one stance's floor is not judged
#: by the next one's), not against the whole clip. CMU's floor is not level:
#: 16_56's right foot stands 5-6 cm higher in its first stance than in its
#: second, so "within 4 cm of the clip's lowest" never planted that foot (0.6 s
#: still reached the second stance). Foot lock then left it sliding, `cycle` cut a
#: cycle a stride and a half long, and the sole calibration fell back to a pose
#: 23 degrees toe-down.
CONTACT_WINDOW_S = 0.35


def _local_low(z, w):
    """Rolling minimum of `z` (frames along axis 0) over `w` frames either side."""
    return np.stack([z[max(0, i - w):i + w + 1].min(axis=0) for i in range(len(z))])


def _stance(track, fps, height, speed, z=None, ground=None):
    """Planted mask (frames x feet) and horizontal speed. Height is `z` when given
    (the sole), else the track's own; speed is always the track's (the ankle,
    which does not jump when the sole's lowest point rolls from heel to toe).
    Height is measured from `ground` when the floor is known (a grounded clip),
    else from the lowest point nearby."""
    z = track[:, :, 2] if z is None else z
    v = np.zeros_like(track[:, :, 2])
    v[1:] = np.linalg.norm(np.diff(track[:, :, :2], axis=0), axis=2) * fps
    ref = ground if ground is not None else _local_low(z, max(1, int(round(CONTACT_WINDOW_S * fps))))
    return (z - ref < height) & (v < speed), v


def _foot_bones(armature, foot):
    """The foot bone and everything below it (toes)."""
    b = armature.data.bones[foot]
    return [b.name] + [c.name for c in b.children_recursive]


def _sole(armature, meta):
    """Per foot, [(bone, points in that bone's rest space)] on the sole.

    Taken from every skinned mesh's undeformed vertices weighted at least 0.5 to
    the foot or a toe bone, keeping those within 2 cm of the foot's lowest at rest,
    and carried rigidly by their bone from then on. The evaluated meshes are not
    used: anything bound to the armature (clothes added later, a hidden mesh) would
    set the ground. With no skin, the foot and toe bones' heads and tails stand in."""
    m = meta["bones"]
    T0 = armature.matrix_world
    out = []
    for joint in ("LeftFoot", "RightFoot"):
        names = _foot_bones(armature, m[joint])
        rest = {n: armature.data.bones[n].matrix_local for n in names}
        pts = []                                         # (bone, armature-space point)
        for o in _skin_meshes(armature):
            idx = {o.vertex_groups[n].index: n for n in names if n in o.vertex_groups}
            if not idx:
                continue
            to_arm = T0.inverted() @ o.matrix_world
            for v in o.data.vertices:
                g = max(((e.weight, idx[e.group]) for e in v.groups if e.group in idx), default=None)
                if g and g[0] >= 0.5:
                    pts.append((g[1], to_arm @ v.co))
        if not pts:
            for n in names:
                b = armature.data.bones[n]
                pts += [(n, b.head_local.copy()), (n, b.tail_local.copy())]
        low = min((T0 @ p).z for _, p in pts)
        keep = [(n, p) for n, p in pts if (T0 @ p).z - low < 0.02] or pts
        out.append([(n, np.array([list(rest[n].inverted() @ p) for b_, p in keep if b_ == n]))
                    for n in names if any(b_ == n for b_, _ in keep)])
    return out


def _sole_points(armature, T0, sole, n=None, rest=False):
    """World positions of every sole point: per foot an array (frames, points, 3),
    or (points, 3) at rest."""
    def now():
        feet = []
        for parts in sole:
            chunks = []
            for bone, P in parts:
                M = np.array(T0 @ (armature.data.bones[bone].matrix_local if rest
                                   else armature.pose.bones[bone].matrix))
                chunks.append(P @ M[:3, :3].T + M[:3, 3])
            feet.append(np.concatenate(chunks))
        return feet
    if rest:
        return now()
    rows = []
    for f in range(1, n + 1):
        _go(f)
        rows.append(now())
    return [np.stack([r[i] for r in rows]) for i in range(len(sole))]


def _sole_heights(armature, T0, sole, n=None, rest=False):
    """World height of each foot's lowest sole point: (frames, 2), or (2,) at rest."""
    pts = _sole_points(armature, T0, sole, n, rest)
    return np.array([p[:, 2].min() for p in pts]) if rest else \
        np.stack([p[:, :, 2].min(axis=1) for p in pts], axis=1)


def _bridged(planted, gap=2):
    """`planted` with gaps of up to `gap` frames inside one foot's contact filled in.

    A foot cannot lift off and land again within two frames. A contact point that
    rolls fast for a frame (CMU 02_03's right foot: 0.96 m/s on one mid-stance frame)
    split one stance in two: the lock let go for that frame, and the stride report
    read a 1-frame contact and 7/11-frame steps in a clip whose steps are 9/9."""
    out = planted.copy()
    for i in range(out.shape[1]):
        col = out[:, i]
        f = 0
        while f < len(col):
            if col[f]:
                f += 1
                continue
            g = f
            while g < len(col) and not col[g]:
                g += 1
            if 0 < f and g < len(col) and g - f <= gap:
                col[f:g] = True
            f = g
    return out


def _floor_contacts(points, ground, fps, height=0.04, speed=0.5):
    """Planted mask (frames x 2) once the floor is known: the sole within
    `height` of it and its contact point (the lowest sole point, followed as
    itself) slower than `speed`. The ankle's speed is not used: it moves through
    every running stance as the foot rolls, so a 2-3 frame jog contact would fail it."""
    n = len(points[0])
    out = np.zeros((n, 2), dtype=bool)
    for i, P in enumerate(points):
        z = P[:, :, 2].min(axis=1)
        v = np.zeros(n)
        for f in range(1, n):
            k = int(P[f, :, 2].argmin())
            v[f] = float(np.linalg.norm(P[f, k, :2] - P[f - 1, k, :2])) * fps
        if n > 1:
            v[0] = v[1]
        out[:, i] = (z - ground < height) & (v < speed)
    return _bridged(out)


def _contact_slip(points, planted, fps):
    """p90 horizontal speed (m/s) of the contact point in mid-stance: the lowest
    sole point, followed as itself from one frame to the next, on frames planted
    along with both neighbors. The ankle moves on purpose as the heel rises, so
    its speed is not slip."""
    s = []
    for i, P in enumerate(points):
        for f in range(1, len(P) - 1):
            if planted[f - 1, i] and planted[f, i] and planted[f + 1, i]:
                k = int(P[f, :, 2].argmin())
                s.append(float(np.linalg.norm(P[f, k, :2] - P[f - 1, k, :2])) * fps)
    return round(float(np.percentile(s, 90)), 3) if s else None


#: Frames the lock eases in before a landing and out after a lift-off. Outside
#: the stance, not inside it: a jog's contact lasts 2-3 frames, and a 3-frame
#: ease inside it never reached full weight, so the lock barely acted.
LOCK_EASE = 2

#: A long foot -- one reaching more than `LONG_FOOT_REACH` leg lengths ahead of its ankle at rest,
#: as an oversized shoe does -- off a full stance, whose sole dips more than `SWING_PITCH_FROM`
#: below the floor, is pitched toe-up about its ankle until it dips no more than `SWING_PITCH_TO`,
#: and only what is left is lifted. Lifting is what keeps a sole out of the floor; but a long
#: shoe pitched toe-down at push-off, or swung through low, reaches 10-18 cm below the floor in
#: every swing of a walk or jog, and lifting it that far keeps the knee bent throughout. A
#: human-proportioned foot (about a quarter to a third of a leg) is always lifted, however deep
#: a dance or a crawl takes it.
LONG_FOOT_REACH = 0.5
SWING_PITCH_FROM, SWING_PITCH_TO = 0.06, 0.01
#: The furthest that pitch goes, and how fast it may turn: a pitch chosen frame by frame
#: jumps between frames, so each swing gets one curve that turns no faster than this.
SWING_PITCH_MAX_DEG, SWING_PITCH_RATE_DEG_S = 50.0, 300.0


def _long_foot(sole_rest, ankle, knee, hip):
    """The horizontal direction (unit, 3) a long foot points at rest, or None for a foot whose sole
    reaches no more than `LONG_FOOT_REACH` leg lengths (hip-knee-ankle) ahead of the ankle, or whose
    sole's center sits too close beside the ankle to say which way it points."""
    P, A = np.asarray(sole_rest, float), np.asarray(ankle, float)
    ahead = P.mean(axis=0) - A
    ahead[2] = 0.0
    if np.linalg.norm(ahead) < 0.02:
        return None
    ahead /= np.linalg.norm(ahead)
    leg = np.linalg.norm(np.asarray(knee) - np.asarray(hip)) + np.linalg.norm(A - np.asarray(knee))
    reach = float(((P - A) @ ahead).max())
    return ahead if reach > LONG_FOOT_REACH * leg else None


def _pitch_axes(armature, T0, sole, legs):
    """Per leg (thigh, calf, foot), the axis a toe-up pitch of a long foot (`_long_foot`) turns
    about, in the foot bone's own frame, or None for a foot that is not long: across the foot at
    rest, signed so a positive turn raises the toes. Fixed to the foot, it turns with it and never
    changes sign between frames, as an axis rebuilt each frame from the pose can when the foot is
    steeply pitched."""
    out = []
    for parts, (thigh, calf, foot) in zip(_sole_points(armature, T0, sole, rest=True), legs):
        H, K, R = (T0 @ armature.data.bones[b].matrix_local for b in (thigh, calf, foot))
        ahead = _long_foot(parts, R.translation, K.translation, H.translation)
        if ahead is None:
            out.append(None)
            continue
        u = np.cross(ahead, [0.0, 0.0, 1.0])
        u /= np.linalg.norm(u)
        if np.cross(u, ahead)[2] < 0:
            u = -u
        out.append(np.array(R.to_3x3().inverted()) @ u)
    return out


def _swing_pitch(points, ankles, axes, ground, weight, fps):
    """Per frame and foot, the toe-up pitch (radians) of a foot off a full stance, about
    `axes` (frames, 2, 3; zero where a foot has none).

    A frame whose sole dips more than `SWING_PITCH_FROM` needs the least pitch, up to
    `SWING_PITCH_MAX_DEG`, that brings it within `SWING_PITCH_TO` of the floor. The pitch over
    a swing is the smallest curve that meets every frame's need and turns no faster than
    `SWING_PITCH_RATE_DEG_S`, starting from none at the stance on either side. So it starts
    turning before the dip that needs it, and a frame it cannot serve in time is lifted."""
    n = len(points[0])
    th = np.radians(np.arange(0.0, SWING_PITCH_MAX_DEG + 0.5, 1.0))
    rate = math.radians(SWING_PITCH_RATE_DEG_S) / fps
    plan = np.zeros((n, 2))
    for i in range(2):
        need = np.zeros(n)
        for f in range(n):
            u = axes[f, i]
            if weight[f, i] >= 1.0 or not u.any():
                continue
            P, A = points[i][f], ankles[f, i]
            V = P - A
            if ground - P[:, 2].min() <= SWING_PITCH_FROM:
                continue
            low = (A[2] + np.outer(np.cos(th), V[:, 2]) + np.outer(np.sin(th), np.cross(u, V)[:, 2])
                   + np.outer(1.0 - np.cos(th), u[2] * (V @ u))).min(axis=1)
            ok = np.flatnonzero(ground - low <= SWING_PITCH_TO)
            need[f] = th[ok[0]] if len(ok) else th[int(np.argmax(low))]
        held = weight[:, i] >= 1.0
        f = 0
        while f < n:
            if held[f]:
                f += 1
                continue
            g = f
            while g + 1 < n and not held[g + 1]:
                g += 1
            idx = np.arange(f, g + 1)
            for h in idx:
                cone = float((need[f:g + 1] - rate * np.abs(idx - h)).max())
                # None at a stance: rise from the one before, fall to the one after.
                edge = min(h - f + 1 if f > 0 else n, g - h + 1 if g < n - 1 else n)
                plan[h, i] = max(0.0, min(cone, rate * edge))
            f = g + 1
    return plan


def _contact_drift(points, planted):
    """Per foot and frame, the horizontal move (frames, 2, 2) that holds the
    sole's contact point where it touched down, and the lock's weight (frames, 2):
    1 through the stance, easing over `LOCK_EASE` frames either side of it.

    The contact point is the lowest sole point, and it rolls during a stance --
    heel at landing, ball and toes at push-off -- so it is anchored where it
    first touches and re-anchored, continuously, whenever another point becomes
    the lowest. Pinning the ankle instead would hold it still while the heel
    rises, dragging a toe-off backwards."""
    n = planted.shape[0]
    drift = np.zeros((n, 2, 2))
    weight = planted.astype(float)
    for i, P in enumerate(points):
        f = 0
        while f < n:
            if not planted[f, i]:
                f += 1
                continue
            g = f
            while g + 1 < n and planted[g + 1, i]:
                g += 1
            cur = int(P[f, :, 2].argmin())
            anchor = P[f, cur, :2].copy()
            for h in range(f, g + 1):
                low = int(P[h, :, 2].argmin())
                if low != cur and P[h, low, 2] < P[h, cur, 2] - 0.003:
                    # The new lowest point takes over where it is now, corrected
                    # by the drift already applied, so the foot does not jump.
                    prev = drift[h - 1, i] if h > f else np.zeros(2)
                    anchor = P[h, low, :2] + prev
                    cur = low
                drift[h, i] = anchor - P[h, cur, :2]
            for k in range(1, LOCK_EASE + 1):
                w = 1.0 - k / (LOCK_EASE + 1)
                for h, src in ((f - k, f), (g + k, g)):
                    if 0 <= h < n and not planted[h, i] and w > weight[h, i]:
                        weight[h, i], drift[h, i] = w, drift[src, i] * w
            f = g + 1
    return drift, weight


def _landings(planted, bridge=2):
    """Frame indices where each foot comes down (start of a planted run).

    Planted runs separated by `bridge` frames or fewer are one contact: a jogging
    foot's short, noisy contact otherwise reads as several landings a frame or two
    apart (on CMU 16_36, a "2-frame gait cycle").
    """
    out = []
    for i in range(planted.shape[1]):
        col = planted[:, i].copy()
        f = 0
        while f < len(col):
            if col[f]:
                f += 1
                continue
            g = f
            while g < len(col) and not col[g]:
                g += 1
            if 0 < f and g < len(col) and g - f <= bridge:
                col[f:g] = True
            f = g
        out.append([f for f in range(len(col)) if col[f] and (f == 0 or not col[f - 1])])
    return out


def _lock(armature, action, meta, *, shift_root=True, height=0.04, speed=0.5, planted=None):
    """Pin planted feet in place on `action` (edits its keys), and set a planted
    foot's sole on the ground (`meta["ground_z"]`, from `retarget`). Returns
    slip stats and how far planted soles sit from the ground afterwards. A sole
    that would pass below the ground on any frame is lifted to it. `planted`
    (frames x 2) overrides contact detection, as `retarget` does with the
    contacts it reads off the source skeleton."""
    m = meta["bones"]
    fps, root = meta["fps"], _body(meta)
    legs = [(m["LeftUpLeg"], m["LeftLeg"], m["LeftFoot"]), (m["RightUpLeg"], m["RightLeg"], m["RightFoot"])]
    feet = [legs[0][2], legs[1][2]]
    n = _length(action)
    T0 = armature.matrix_world.copy()
    sole = _sole(armature, meta)
    ground = meta.get("ground_z")
    with _Using(armature, action):
        rootpb = armature.pose.bones[root]
        before = _feet(armature, T0, feet, n)
        points_before = _sole_points(armature, T0, sole, n)
        if planted is None:
            planted = (_floor_contacts(points_before, ground, fps, height, speed) if ground is not None
                       else _stance(before, fps, height, speed, z=_sole_heights(armature, T0, sole, n))[0])
        _store_contacts(action, planted)
        if shift_root:
            contact = _sole_points(armature, T0, sole, n)
            offset = np.zeros((n, 2))
            for f in range(1, n):
                # The contact point's own drift (lowest sole point), not the ankle's.
                step = np.array([P[f, P[f, :, 2].argmin(), :2] - P[f - 1, P[f, :, 2].argmin(), :2]
                                 for P in contact])
                down = planted[f] & planted[f - 1]
                offset[f] = offset[f - 1] - (step[down].mean(axis=0) if down.any() else 0.0)
            kernel = np.ones(5) / 5
            offset = np.stack([np.convolve(np.pad(offset[:, i], 2, mode="edge"), kernel, mode="valid")
                               for i in range(2)], axis=1)
            for f in range(1, n + 1):
                _go(f)
                world = T0 @ rootpb.matrix
                world.translation += Vector((float(offset[f - 1, 0]), float(offset[f - 1, 1]), 0.0))
                rootpb.matrix = T0.inverted() @ world
                rootpb.keyframe_insert("location", frame=f)
        points = _sole_points(armature, T0, sole, n)
        soles = np.stack([P[:, :, 2].min(axis=1) for P in points], axis=1) if ground is not None else None
        drift, weight = _contact_drift(points, planted)
        pbs = armature.pose.bones
        pitch = np.zeros((n, 2))
        local = _pitch_axes(armature, T0, sole, legs)
        if soles is not None and any(u is not None for u in local):
            ankles, axes = np.zeros((n, 2, 3)), np.zeros((n, 2, 3))
            for f in range(n):
                _go(f + 1)
                for i, foot in enumerate(feet):
                    W = T0 @ pbs[foot].matrix
                    ankles[f, i] = W.translation
                    if local[i] is not None:
                        u = np.array(W.to_3x3()) @ local[i]
                        axes[f, i] = u / np.linalg.norm(u)
            pitch = _swing_pitch(points, ankles, axes, ground, weight, fps)
        for f in range(n):
            below = soles is not None and (soles[f] < ground - 0.002).any()
            if not weight[f].any() and not below and not pitch[f].any():
                continue
            _go(f + 1)
            for i, (thigh, calf, foot) in enumerate(legs):
                w = weight[f, i]
                if w <= 0 and not (soles is not None and soles[f, i] < ground - 0.002) and not pitch[f, i]:
                    continue
                foot_rot = (T0 @ pbs[foot].matrix).to_3x3().copy()
                H, K, A = ((T0 @ pbs[b].matrix).translation.copy() for b in (thigh, calf, foot))
                lift = 0.0
                if soles is not None:
                    gap = ground - soles[f, i]                   # > 0: below the floor
                    if pitch[f, i]:
                        # The pitched sole is what is lifted: what the pitch leaves of a dip, or a
                        # heel it lowers into the floor.
                        turn = Matrix.Rotation(float(pitch[f, i]), 3, Vector(tuple(axes[f, i])))
                        foot_rot = turn @ foot_rot
                        a = np.array(A)
                        P = a + (points[i][f] - a) @ np.array(turn).T
                        gap = ground - float(P[:, 2].min())
                    lift = max(gap * w, gap) if gap > 0 else gap * w
                goal = Vector((A.x + float(drift[f, i, 0]), A.y + float(drift[f, i, 1]), A.z + lift))
                K2, A2 = _two_bone(H, K, A, goal)
                _aim(armature, T0, thigh, H, K, K2)
                bpy.context.view_layer.update()
                _aim(armature, T0, calf, (T0 @ pbs[calf].matrix).translation.copy(),
                     (T0 @ pbs[foot].matrix).translation.copy(), A2)
                bpy.context.view_layer.update()
                mtx = foot_rot.to_4x4()
                mtx.translation = (T0 @ pbs[foot].matrix).translation
                pbs[foot].matrix = T0.inverted() @ mtx
                bpy.context.view_layer.update()
                for b in (thigh, calf, foot):
                    pbs[b].keyframe_insert("rotation_quaternion", frame=f + 1)
        out = {"stance_slip_p90_mps": {"before": _contact_slip(points_before, planted, fps),
                                       "after": _contact_slip(_sole_points(armature, T0, sole, n),
                                                              planted, fps)},
               "planted_fraction": [round(float(planted[:, i].mean()), 3) for i in range(2)]}
        if ground is not None and planted.any():
            final = _sole_heights(armature, T0, sole, n)
            out["sole_below_ground_mm"] = round(float(max(0.0, ground - final.min())) * 1000, 1)
            gap = np.abs(final - ground)[planted]
            out["stance_sole_gap_mm"] = {"median": round(float(np.median(gap)) * 1000, 1),
                                         "p90": round(float(np.percentile(gap, 90)) * 1000, 1)}
    return out


def _contacts(armature, action, meta, height=0.04, speed=0.5):
    """Planted mask (frames x 2) of an existing action, read before an edit.

    An edit keeps the input's contact timing, so it carries this mask over
    instead of detecting contacts on its own output: bending a path or scaling a
    stride is exactly what makes a planted foot slide, and a sliding foot would
    fail the speed test and be left unlocked. The soles set the contacts' timing;
    a contact counts only where the mask the input was locked with (`CONTACTS`,
    back to the source skeleton's) has one within `CONTACT_MATCH_S`."""
    m = meta["bones"]
    n = _length(action)
    T0 = armature.matrix_world.copy()
    sole = _sole(armature, meta)
    with _Using(armature, action):
        if meta.get("ground_z") is not None:
            found = _floor_contacts(_sole_points(armature, T0, sole, n), meta["ground_z"], meta["fps"],
                                    height, speed)
        else:
            tr = _feet(armature, T0, [m["LeftFoot"], m["RightFoot"]], n)
            found = _stance(tr, meta["fps"], height, speed, z=_sole_heights(armature, T0, sole, n))[0]
    stored = _stored_contacts(action)
    return found if stored is None else _confirmed(found, stored, meta["fps"])


#: How far (seconds) a contact read off the soles may lie from one in the locked mask and
#: still be the same contact: a sole's floor test and the source skeleton's ankle-and-toe
#: test place a landing a frame or two apart at 24 fps. A run that matches is kept whole,
#: since the soles set the contact's timing; the mask decides only whether it exists.
CONTACT_MATCH_S = 0.08


def _confirmed(found, locked, fps):
    """`found` (frames x 2) without the contact runs that no run of `locked` comes
    within `CONTACT_MATCH_S` of."""
    w = max(1, int(round(CONTACT_MATCH_S * fps)))
    out = found.copy()
    n = len(found)
    for i in range(found.shape[1]):
        near = np.zeros(n, dtype=bool)
        for f in np.flatnonzero(locked[:, i]):
            near[max(0, f - w):f + w + 1] = True
        f = 0
        while f < n:
            if not found[f, i]:
                f += 1
                continue
            g = f
            while g + 1 < n and found[g + 1, i]:
                g += 1
            if not near[f:g + 1].any():
                out[f:g + 1, i] = False
            f = g + 1
    return out


def _roots(armature, action, root):
    T0 = armature.matrix_world.copy()
    n = _length(action)
    with _Using(armature, action):
        out = []
        for f in range(1, n + 1):
            _go(f)
            out.append((T0 @ armature.pose.bones[root].matrix).copy())
    return out


def _facing(armature, action, meta, frames):
    """Heading (radians, about world up) the body faces: the hip line's normal."""
    m = meta["bones"]
    T0 = armature.matrix_world.copy()
    out = []
    with _Using(armature, action):
        for f in frames:
            _go(f)
            lr = (T0 @ armature.pose.bones[m["LeftUpLeg"]].head) - (T0 @ armature.pose.bones[m["RightUpLeg"]].head)
            out.append(math.atan2(lr.y, lr.x) - math.pi / 2)     # forward = left turned -90 deg about up
    return out


def _report(action, meta, **extra):
    n = _length(action)
    return {"action": action.name, "frames": n, "fps": meta["fps"],
            "seconds": round(n / meta["fps"], 3), **extra}


# ---------------------------------------------------------------------- retarget

#: CMU's hand joint has zero length in the BVH (its two children, finger base and
#: thumb, both start at it), so Blender gives the bone an arbitrary tail and its
#: "direction" is meaningless; aligning segments by it bent every wrist ~90
#: degrees outward. Hands are aligned by a frame built from the palm instead:
#: hand -> middle-finger base (source: `*FingerBase`) and hand -> thumb tip.
HANDS = ("LeftHand", "RightHand")
#: Feet are aligned by full frame with world up as the roll reference.
FEET = ("LeftFoot", "RightFoot")
#: Arm segments aligned by full frame, keyed to the hand whose thumb sets their roll.
ARM_SEGMENTS = {"LeftArm": "LeftHand", "LeftForeArm": "LeftHand",
                "RightArm": "RightHand", "RightForeArm": "RightHand"}
HAND_REFS = {
    "LeftHand": (("LeftFingerBase", "LThumb"),
                 ["bip01 l finger2", "lefthandmiddle1", "middle_01_l"],
                 ["bip01 l finger0", "lefthandthumb1", "thumb_01_l"]),
    "RightHand": (("RightFingerBase", "RThumb"),
                  ["bip01 r finger2", "righthandmiddle1", "middle_01_r"],
                  ["bip01 r finger0", "righthandthumb1", "thumb_01_r"]),
}


#: The joint each limb segment ends at: a segment is measured from its joint to this one's head.
NEXT_JOINT = {"LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToeBase",
              "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand",
              "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToeBase",
              "RightArm": "RightForeArm", "RightForeArm": "RightHand"}
#: A mapped limb bone whose own tail is further than this from its segment is reported.
TAIL_OFF_AXIS_DEG = 20.0


def _tip(bones, name):
    """Rest point a bone points at: its only child's head (unless that sits on the bone's own head), else its tail."""
    b = bones[name]
    kids = list(getattr(b, "children", ()))
    if len(kids) == 1 and math.dist(kids[0].head_local, b.head_local) > 1e-6:
        return kids[0].head_local
    return b.tail_local


def _limb_vector(bones, mapping, joint):
    """Rest vector of `joint`'s limb segment (armature space), from joint positions.

    It runs from the bone's head to the next mapped joint's head (`NEXT_JOINT`), else to the bone's
    tip. A bone's own tail is not used where a joint is known: rigs imported from Maya or game
    engines leave tails along a local axis, as much as 90-180 degrees from the limb. A leaf bone with
    no next joint (a toe) has nothing to measure to and keeps its own tail.
    `mapping` must still hold the hands when they are not driven: the hand's head is the wrist."""
    b = bones[mapping[joint]]
    nxt = NEXT_JOINT.get(joint)
    end = bones[mapping[nxt]].head_local if nxt in mapping else _tip(bones, mapping[joint])
    return end - b.head_local


def _twist_angle(q, axis):
    """The angle of `q`'s twist about the unit `axis` (swing-twist decomposition)."""
    return 2.0 * math.atan2(q.x * axis.x + q.y * axis.y + q.z * axis.z, q.w)


def _tails_off_axis(bones, mapping):
    """`{bone: degrees}` for mapped limb bones whose own tail is more than `TAIL_OFF_AXIS_DEG` from their segment."""
    out = {}
    for joint, nxt in NEXT_JOINT.items():
        if joint in mapping and nxt in mapping:
            b = bones[mapping[joint]]
            own, seg = b.tail_local - b.head_local, _limb_vector(bones, mapping, joint)
            deg = math.degrees(own.angle(seg, 0.0))
            if deg > TAIL_OFF_AXIS_DEG:
                out[b.name] = round(deg)
    return out


def _frame(primary, secondary):
    x = primary.normalized()
    z = x.cross(secondary).normalized()
    return Matrix((x, z.cross(x), z)).transposed()


def _hand_refs(armature, joint):
    """Target (middle-finger base bone, thumb bone) for a hand, or None."""
    by_norm = {}
    for b in armature.data.bones:
        by_norm.setdefault(normalise(b.name), b.name)
    _, mids, thumbs = HAND_REFS[joint]
    mid = next((by_norm[n] for n in mids if n in by_norm), None)
    thumb = next((by_norm[n] for n in thumbs if n in by_norm), None)
    return (mid, thumb) if mid and thumb else None


#: Flat-stance frames a foot needs for its own sole calibration. A jog can give one
#: foot only a few (CMU 16_36's left foot has 4), and a foot below the minimum falls
#: back to the rest pose, which is off by tens of degrees whenever the BVH's foot
#: roll is; even one or two real flat contacts beat that guess.
MIN_SOLE_FRAMES = 2

#: A planted frame calibrates the sole only while the source's ankle-to-toe-base line
#: pitches below horizontal by no more than this past the foot's flattest planted
#: frames (their 10th percentile). Where that line lies on a flat foot depends on the
#: actor (10-30 degrees), and a push-off or a jog's forefoot contact adds up to about 25.
#: A foot kneeling on its toes is low and still for seconds, so it passes the height and
#: speed tests, but pitches it 50-80 degrees; calibrated on those frames, the foot turns
#: out of its leg's plane for the whole clip.
SOLE_PITCH_SPREAD_DEG = 25.0


def _sole_offsets(src, src_world, W, Rz, f0, f1, fps, feet):
    """Per foot, the rotation from the source foot bone's frame to a flat-sole frame
    (toes forward, sole on the floor), averaged over the frames it is planted.

    Planted: ankle within 4 cm and toe base within 3 cm of their lowest nearby
    (`CONTACT_WINDOW_S` either side; the capture floor is not level), the
    slower of the two moving under 0.5 m/s, and the foot pitched within
    `SOLE_PITCH_SPREAD_DEG` of its flattest such frames. CMU units are ~0.056 m.
    None for a foot with fewer than `MIN_SOLE_FRAMES` such frames (jumps,
    sitting), which then falls back to the rest pose.
    Returns the offsets, per foot how many frames calibrated it, and the
    source's contacts (frames x 2, left then right): ankle or toe base within
    3 cm of its lowest nearby and slower than 0.5 m/s. Read on the source
    because its units are clean and each stance is judged against its own
    neighborhood; a heel strike and a forefoot strike both count.
    """
    unit, up = 0.056, Vector((0.0, 0.0, 1.0))
    track = {j: [] for j in feet}
    for frame in range(f0, f1 + 1):
        sw = {nm: W @ Rz @ m for nm, m in src_world(frame).items()}
        for j in feet:
            toe = j.replace("Foot", "ToeBase")
            track[j].append((sw[j].translation.copy(), sw[toe].translation.copy(), sw[j].to_3x3()))
    out, counts = {}, {}
    w = max(1, int(round(CONTACT_WINDOW_S * fps)))
    contacts = []
    for j in feet:
        rows = track[j]
        az, tz = np.array([r[0].z for r in rows]), np.array([r[1].z for r in rows])
        sp = np.zeros(len(rows))
        for i in range(1, len(rows)):
            sp[i] = min((rows[i][0] - rows[i - 1][0]).to_2d().length,
                        (rows[i][1] - rows[i - 1][1]).to_2d().length) * unit * fps
        sp[0] = sp[1] if len(rows) > 1 else 0.0
        low = np.minimum(az - _local_low(az, w), tz - _local_low(tz, w)) * unit
        contacts.append((low < 0.03) & (sp < 0.5))
    if contacts:
        contacts = list(_bridged(np.stack(contacts, axis=1)).T)
    for j, rows in track.items():
        low_a = _local_low(np.array([r[0].z for r in rows]), w)
        low_t = _local_low(np.array([r[1].z for r in rows]), w)
        planted = []                                     # (frame, pitch of ankle -> toe base)
        for i in range(1, len(rows)):
            a, t, R = rows[i]
            speed = min((rows[i][0] - rows[i - 1][0]).to_2d().length,
                        (rows[i][1] - rows[i - 1][1]).to_2d().length) * unit * fps
            if a.z - low_a[i] > 0.04 / unit or t.z - low_t[i] > 0.03 / unit or speed > 0.5:
                continue
            planted.append((i, math.degrees(math.atan2(a.z - t.z, (t - a).to_2d().length))))
        steepest = (np.percentile([p for _, p in planted], 10) + SOLE_PITCH_SPREAD_DEG) if planted else 0.0
        acc, count, ref = Quaternion((0, 0, 0, 0)), 0, None
        for i, pitch in planted:
            a, t, R = rows[i]
            if pitch > steepest:
                continue
            fwd = t - a
            fwd.z = 0.0
            if fwd.length < 1e-6:
                continue
            q = (R.transposed() @ _frame(fwd, up)).to_quaternion()
            ref = ref or q
            if ref.dot(q) < 0:
                q.negate()
            acc = Quaternion((acc.w + q.w, acc.x + q.x, acc.y + q.y, acc.z + q.z))
            count += 1
        out[j] = acc.normalized().to_matrix() if count >= MIN_SOLE_FRAMES else None
        counts[j] = count
    return out, counts, np.stack(contacts, axis=1) if contacts else None


def retarget(armature, bvh_path, *, start_s=None, end_s=None, name=None, mapping=None,
             heading_deg=0.0, ground=True, foot_lock=True, contact_height=0.04,
             contact_speed=0.5, hands=True):
    """Bake a CMU clip (or the part between `start_s` and `end_s`) onto `armature`.

    Returns a report; the motion itself is a new Action keyed at the scene's fps on
    frames 1..N, left *unassigned* (the armature's own action and NLA are
    restored). The actor starts facing the character's own front; `heading_deg`
    turns the whole performance about world up from there. Root translation is
    scaled by leg length. `ground` keeps the skin's lowest point at its rest
    height; `foot_lock` pins planted feet (within `contact_height` m of the
    foot's lowest height in the clip and slower than `contact_speed` m/s).
    Hands are aligned by the palm (see `HANDS`); `hands=False`, or a rig with no
    middle-finger and thumb bones, keeps them at rest relative to the forearm.
    Limbs are aligned by joint positions (`_limb_vector`), so a rig whose bone
    tails lie along a local axis retargets like one whose tails meet the next joint.
    """
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    joints = dict(mapping or bone_map(armature))      # every mapped joint: segments are measured between them
    mapping = dict(joints)
    for joint in HANDS:                     # no palm to align by: leave the hand at rest
        if joint in mapping and (not hands or _hand_refs(armature, joint) is None):
            mapping.pop(joint)
    meta = {"bones": mapping, "fps": fps, "source": bvh_path}
    T0 = armature.matrix_world.copy()
    sole = _sole(armature, meta)
    if ground:                       # the floor is where the rest pose stands: its lowest sole point
        meta["ground_z"] = float(_sole_heights(armature, T0, sole, rest=True).min())
    tgt_rest = {b.name: T0 @ b.matrix_local for b in armature.data.bones}

    src = _import_bvh(bvh_path)
    try:
        f0, f1 = (int(v) for v in src.animation_data.action.frame_range)
        f0 += 1                                                      # cgspeed frame 1 is a T-pose
        if start_s is not None:
            f0 = max(f0, 1 + int(round(start_s * fps)))
        if end_s is not None:
            f1 = min(f1, 1 + int(round(end_s * fps)))
        if f1 <= f0:
            raise ValueError(f"empty range: frames {f0}..{f1} of {bvh_path}")

        def src_world(frame):
            scene.frame_set(frame)
            return {pb.name: src.matrix_world @ pb.matrix for pb in src.pose.bones}

        src_rest = {b.name: src.matrix_world @ b.matrix_local for b in src.data.bones}

        def leg(rest, hip, knee, ankle):
            return ((rest[knee].translation - rest[hip].translation).length
                    + (rest[ankle].translation - rest[knee].translation).length)

        scale = (leg(tgt_rest, mapping["LeftUpLeg"], mapping["LeftLeg"], mapping["LeftFoot"])
                 / leg(src_rest, "LeftUpLeg", "LeftLeg", "LeftFoot"))

        def yaw_of(v):
            return math.atan2(v.y, v.x)

        tgt_lr = tgt_rest[mapping["LeftUpLeg"]].translation - tgt_rest[mapping["RightUpLeg"]].translation
        src_lr = src_rest["LeftUpLeg"].translation - src_rest["RightUpLeg"].translation
        Rz = Matrix.Rotation(yaw_of(tgt_lr) - yaw_of(src_lr), 4, "Z")
        first = {n: Rz @ m for n, m in src_world(f0).items()}
        lr0 = first["LeftUpLeg"].translation - first["RightUpLeg"].translation
        W = Matrix.Rotation(yaw_of(tgt_lr) - yaw_of(lr0) + math.radians(heading_deg), 4, "Z")
        hips0 = first["Hips"].translation.copy()

        def direction(bones, bone, mw):
            b = bones[bone]
            return (mw.to_3x3() @ (b.tail_local - b.head_local)).normalized()

        src_rest_rot = {n: (Rz @ m).to_3x3() for n, m in src_rest.items()}
        foot_offset, calibrated, contacts = _sole_offsets(src, src_world, W, Rz, f0, f1, fps,
                                                          [j for j in FEET if j in mapping])
        corr = {}
        for joint, bone in mapping.items():
            if joint in AXIAL or joint in CLAVICLES:
                corr[joint] = Matrix.Identity(3)
            elif joint in ARM_SEGMENTS and ARM_SEGMENTS[joint] in mapping:
                # Upper arm and forearm: align the whole rest frame, not just the
                # segment direction. The shortest rotation from an A-pose arm to a
                # T-pose arm leaves the roll about the arm wherever it falls, and the
                # palm would then need ~75 degrees of standing forearm twist on a plain
                # walk. Roll is referenced to the rest-pose thumb, the same on both
                # skeletons (palms down in a T-pose, toward the thigh in an A-pose).
                hand = ARM_SEGMENTS[joint]
                (s_mid, s_thumb), _, _ = HAND_REFS[hand]
                mid, thumb = _hand_refs(armature, hand)
                sb, tb, M = src.data.bones, armature.data.bones, src.matrix_world.to_3x3()
                RzM, T3 = Rz.to_3x3() @ M, T0.to_3x3()
                Fs = _frame(RzM @ (sb[joint].tail_local - sb[joint].head_local),
                            RzM @ (sb[s_thumb].tail_local - sb[hand].head_local))
                Ft = _frame(T3 @ _limb_vector(tb, joints, joint),
                            T3 @ (_tip(tb, thumb) - tb[mapping[hand]].head_local))
                corr[joint] = Fs @ Ft.transposed()
            elif joint in FEET:
                # Feet are aligned by the sole, calibrated from the clip itself. The
                # BVH skeleton fixes joint positions but not the foot's roll about
                # its own length, so neither bone direction (it pitches planted soles
                # ~15-20 degrees toe-down) nor "the zero pose is flat" (it rolls them
                # ~15 degrees onto their outer edges, soles facing each other)
                # holds. What does hold: a planted foot's sole is flat on the floor.
                # So the source foot's flat-sole frame is measured on its stance
                # frames, and the character's flat rest foot is matched to it.
                up = Vector((0.0, 0.0, 1.0))
                full = T0.to_3x3() @ _limb_vector(armature.data.bones, joints, joint)
                f_tgt = full.copy()
                f_tgt.z = 0.0
                forward = Vector((tgt_lr.y, -tgt_lr.x, 0.0))      # forward = left turned -90 deg
                # A foot rests pointing roughly forward; a segment that does not (mostly
                # vertical, or a tail left along a sideways or backward local axis) is
                # replaced by the body's forward.
                if f_tgt.length <= 0.3 * full.length or f_tgt.angle(forward, 0.0) > math.radians(60):
                    f_tgt = forward
                O = foot_offset.get(joint)
                if O is not None:                                  # R_tgt = R_src O F_t^T R_tgt_rest
                    corr[joint] = src_rest_rot[joint] @ O @ _frame(f_tgt, up).transposed()
                else:                                              # no stance in the clip: rest pose
                    sb = src.data.bones
                    toe = joint.replace("Foot", "ToeBase")
                    f_src = Rz.to_3x3() @ src.matrix_world.to_3x3() @ (sb[toe].head_local - sb[joint].head_local)
                    f_src.z = 0.0
                    corr[joint] = _frame(f_src, up) @ _frame(f_tgt, up).transposed()
            elif joint in HANDS:
                (s_mid, s_thumb), _, _ = HAND_REFS[joint]
                mid, thumb = _hand_refs(armature, joint)
                sb, tb, M = src.data.bones, armature.data.bones, src.matrix_world.to_3x3()
                Fs = _frame(Rz.to_3x3() @ M @ (sb[s_mid].tail_local - sb[joint].head_local),
                            Rz.to_3x3() @ M @ (sb[s_thumb].tail_local - sb[joint].head_local))
                T3 = T0.to_3x3()
                Ft = _frame(T3 @ (tb[mid].head_local - tb[bone].head_local),
                            T3 @ (_tip(tb, thumb) - tb[bone].head_local))
                corr[joint] = Ft                          # target rest palm frame; see the bake loop
            else:
                d_src = Rz.to_3x3() @ direction(src.data.bones, joint, src.matrix_world)
                d_tgt = (T0.to_3x3() @ _limb_vector(armature.data.bones, joints, joint)).normalized()
                corr[joint] = d_tgt.rotation_difference(d_src).to_matrix()
        order = [b.name for b in armature.data.bones if b.name in mapping.values()]
        joint_of = {v: k for k, v in mapping.items()}
        root = mapping["Hips"]
        root_rest = tgt_rest[root].translation.copy()
        # The skin may hang from a bone above the hips (`_body_bone`): it then moves as the hips do, rigidly, so the
        # turn pivots at the hips, and every later edit moves the body by it.
        body = _body_bone(armature, root)
        if body != root:
            meta["body"] = body

        action = bpy.data.actions.new(name or f"mocap_{bvh_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
        action.use_fake_user = True
        action[META] = json.dumps(meta)
        with _Using(armature, action):
            for pb in armature.pose.bones:
                pb.rotation_mode = "QUATERNION"
            twist_state = {}
            for k, frame in enumerate(range(f0, f1 + 1)):
                sw = {nm: W @ Rz @ m for nm, m in src_world(frame).items()}
                out = 1 + k
                for bone in order:
                    joint = joint_of[bone]
                    if joint in HANDS:
                        # The palm as the actor's joints actually place it this frame:
                        # CMU keeps part of the wrist in the finger-base joint, so the
                        # hand joint's own rotation alone misses it.
                        (s_mid, s_thumb), _, _ = HAND_REFS[joint]
                        WR = (W @ Rz).to_3x3()
                        h = src.matrix_world @ src.pose.bones[joint].head
                        # Roll from the hand joint's own rotation carrying the rest
                        # thumb vector, not the posed thumb tip: CMU's thumb joint
                        # has its own channels, and its flexing read as wrist roll.
                        hb = src.data.bones
                        thumb_rest = hb[s_thumb].tail_local - hb[joint].head_local
                        A_h = sw[joint].to_3x3() @ src_rest_rot[joint].inverted()
                        Fs = _frame(WR @ ((src.matrix_world @ src.pose.bones[s_mid].tail) - h),
                                    A_h @ Rz.to_3x3() @ src.matrix_world.to_3x3() @ thumb_rest)
                        R = Fs @ corr[joint].transposed() @ tgt_rest[bone].to_3x3()
                        # A wrist cannot twist: turning the palm over is the forearm
                        # rotating about its own axis. Move the hand's twist about the
                        # forearm (its elbow-to-wrist axis) out of the wrist and into the forearm,
                        # leaving the hand's world orientation unchanged. Without this
                        # the whole roll mismatch (30-90 degrees) lands on the wrist
                        # and the skin there wrings like a candy wrapper.
                        fore = mapping["LeftForeArm" if joint == "LeftHand" else "RightForeArm"] \
                            if ("LeftForeArm" if joint == "LeftHand" else "RightForeArm") in mapping else None
                        if fore is not None:
                            bpy.context.view_layer.update()
                            fpb = armature.pose.bones[fore]
                            Fw = T0 @ fpb.matrix
                            Rf = Fw.to_3x3()
                            # The forearm's elbow-to-wrist axis in its own rest frame: bone-space Y
                            # on a rig whose tails point at the next joint, elsewhere where they do not.
                            fore_joint = "LeftForeArm" if joint == "LeftHand" else "RightForeArm"
                            axis = (tgt_rest[fore].to_3x3().transposed() @ T0.to_3x3()
                                    @ _limb_vector(armature.data.bones, joints, fore_joint)).normalized()
                            L0 = tgt_rest[fore].to_3x3().transposed() @ tgt_rest[bone].to_3x3()
                            D = (Rf.transposed() @ R @ L0.transposed()).to_quaternion()
                            if D.w < 0:
                                D.negate()
                            theta = _twist_angle(D, axis)
                            # Continuous over time: the extracted angle lives in
                            # (-180, 180], and an unwrapped jump there would spin
                            # the forearm a full half turn between two frames.
                            prev = twist_state.get(joint)
                            if prev is not None:
                                theta += 2 * math.pi * round((prev - theta) / (2 * math.pi))
                            twist_state[joint] = theta
                            # The forearm turns the palm about +-90 degrees from the
                            # upper arm and no further; what is left stays at the wrist
                            # rather than wringing the elbow round.
                            upper = mapping.get("LeftArm" if joint == "LeftHand" else "RightArm")
                            if upper is not None:
                                Ru = (T0 @ armature.pose.bones[upper].matrix).to_3x3()
                                U0 = tgt_rest[upper].to_3x3().transposed() @ tgt_rest[fore].to_3x3()
                                E = (U0.transposed() @ Ru.transposed() @ Rf).to_quaternion()
                                if E.w < 0:
                                    E.negate()
                                tau0 = _twist_angle(E, axis)
                                lim = math.radians(FOREARM_TWIST_DEG)
                                theta = min(max(tau0 + theta, -lim), lim) - tau0
                            twist = Quaternion(axis, theta)
                            Fn = (Rf @ twist.to_matrix()).to_4x4()
                            Fn.translation = Fw.translation
                            fpb.matrix = T0.inverted() @ Fn
                    else:
                        R = sw[joint].to_3x3() @ src_rest_rot[joint].inverted() @ corr[joint] @ tgt_rest[bone].to_3x3()
                    pb = armature.pose.bones[bone]
                    if bone == root:
                        head = root_rest + (sw["Hips"].translation - W @ hips0) * scale
                    else:
                        bpy.context.view_layer.update()
                        head = (T0 @ pb.matrix).translation
                    world = R.to_4x4()
                    world.translation = head
                    if bone == root and body != root:
                        armature.pose.bones[body].matrix = T0.inverted() @ (world @ tgt_rest[root].inverted()
                                                                            @ tgt_rest[body])
                        bpy.context.view_layer.update()
                    pb.matrix = T0.inverted() @ world
                for bone in order:
                    armature.pose.bones[bone].keyframe_insert("rotation_quaternion", frame=out)
                armature.pose.bones[root].keyframe_insert("location", frame=out)
                if body != root:
                    armature.pose.bones[body].keyframe_insert("rotation_quaternion", frame=out)
                    armature.pose.bones[body].keyframe_insert("location", frame=out)
    finally:
        _remove(src)

    action[META] = json.dumps(meta)
    n = f1 - f0 + 1
    extra = {"source_frames": [f0, f1], "leg_scale": round(scale, 5), "bones": mapping,
             "sole_calibration_frames": {mapping[j]: c for j, c in calibrated.items()}}
    warnings = [f"{mapping[j]}: no flat stance in the clip ({c} frames), so its sole angle is the "
                "rest pose's; check that foot's pitch and roll on contact"
                for j, c in calibrated.items() if c < MIN_SOLE_FRAMES]
    off_axis = _tails_off_axis(armature.data.bones, joints)
    if off_axis:
        warnings.append(f"limb bones whose tails point away from the next joint ({off_axis}, degrees): the retarget "
                        "aligns them by joint positions, but a rotation keyed on them by hand turns about their "
                        "own axes, not along the limb")
    if ground:
        with _Using(armature, action):
            rootpb = armature.pose.bones[body]
            soles = _sole_heights(armature, T0, sole, n)
            planted = contacts if contacts is not None else \
                _stance(_feet(armature, T0, [mapping["LeftFoot"], mapping["RightFoot"]], n),
                        fps, contact_height, contact_speed, z=soles)[0]
            # Correct on the frames a foot is down, by the lowest planted sole;
            # interpolate between them, so flight stays flight (pulling the lowest
            # point to the floor on every frame dragged a runner down mid-air).
            gap = np.where(planted, soles - meta["ground_z"], np.inf).min(axis=1)
            down = np.isfinite(gap)
            if down.any():
                lows = np.interp(np.arange(n), np.flatnonzero(down), gap[down])
            else:
                lows = soles.min(axis=1) - meta["ground_z"]
            lows = np.convolve(np.pad(lows, 1, mode="edge"), np.ones(3) / 3, mode="valid")
            for f, dz in zip(range(1, n + 1), lows):
                _go(f)
                world = T0 @ rootpb.matrix
                world.translation.z -= float(dz)
                rootpb.matrix = T0.inverted() @ world
                rootpb.keyframe_insert("location", frame=f)
        extra["ground_offset_m"] = {"median": round(float(np.median(lows)), 4),
                                    "max_abs": round(float(np.abs(lows).max()), 4)}
    if foot_lock:
        extra.update(_lock(armature, action, meta, height=contact_height, speed=contact_speed,
                           planted=contacts))
    elif contacts is not None:
        _store_contacts(action, contacts)
    roots = _roots(armature, action, body)
    extra["root_travel_m"] = round((roots[-1].translation - roots[0].translation).to_2d().length, 3)
    if body != root:
        extra["body_bone"] = body
    warnings += _contact_warnings(extra, n / fps)
    return _report(action, meta, warnings=warnings, **extra)


def _contact_warnings(stats, seconds):
    """What an author should look at, from `_lock`'s stats."""
    out = []
    frac = stats.get("planted_fraction") or []
    if seconds > 1.0 and len(frac) == 2 and min(frac) < 0.08 <= max(frac):
        side = "left" if frac[0] < frac[1] else "right"
        out.append(f"the {side} foot is planted on only {min(frac):.0%} of frames, so it is barely "
                   "locked and may slide; check its contacts")
    gap = stats.get("stance_sole_gap_mm")
    if gap and gap["p90"] > 15:
        out.append(f"planted soles sit up to {gap['p90']:.0f} mm off the ground (p90); "
                   "check contact in a side view")
    return out


# ------------------------------------------------------------------------ edits

def cycle(armature, action, *, seconds=None, cycles=None, foot="left", stride=None, name=None):
    """Find one full gait cycle in a steady walk/run and repeat it, root travel accumulated.

    The cycle's length is the gait period, measured by autocorrelating both feet's
    sole heights; it starts at a landing of `foot`. The small mismatch between its
    ends -- every bone's pose, and the root's orientation and height -- is spread
    across the cycle so the seam does not pop. The report gives the two steps'
    lengths in frames (`step_frames`, `foot` landing to the other foot's, then back)
    and warns when they differ by more than ~a quarter: a limp or a skip repeats. Spreading the root's
    orientation also straightens a cycle that turns a little, so repeats do not
    curve (`cycle_turn_deg` reports how much was removed).
    Give `seconds` or `cycles` (default 2). Works for steady, straight
    locomotion; a clip that turns or changes speed makes a cycle that drifts, and
    the report says how far the cycle turned.

    Whatever one stride does is repeated every cycle, so the report's `symmetry`
    describes the stride it took and lists every full stride in the clip: each
    foot's contact length and the two step intervals, in frames, and `uneven` =
    |contact L - R| + |step L->R - R->L| (0 is a perfectly even stride). A real
    capture's strides differ: CMU 16_56's first stride holds the left foot down 7
    frames and the right 5, its next one 4 and 5. Pass `stride=<index>` (into
    `symmetry.strides`) to take another one; the default is the first from `foot`. A
    stride with `uneven` over 2 is warned about, naming the most even one.
    """
    action = _resolve(action)
    meta = _meta(action)
    m = meta["bones"]
    fps, root = meta["fps"], _body(meta)
    n = _length(action)
    T0 = armature.matrix_world.copy()
    side = 0 if foot == "left" else 1
    with _Using(armature, action):
        soles = _sole_heights(armature, T0, _sole(armature, meta), n)
    planted = _contacts(armature, action, meta)
    # One stride: two landings of a foot 0.4-1.6 s apart. With contacts read
    # against the floor these are reliable; before, a jog's missed contact made a
    # cycle a stride and a half long (CMU 16_56: 18 frames where the stride is 17,
    # starting mid-stance), which then repeated a 4-frame/15-frame skip. When no
    # foot lands twice, the period comes from the feet's height over time.
    all_lands = _landings(planted)
    lo, hi = max(2, int(round(0.4 * fps))), int(round(1.6 * fps))
    a = b = None
    strides = _strides(planted, all_lands, lo, hi)
    if stride is not None:
        if not 0 <= stride < len(strides):
            raise ValueError(f"stride={stride}: this clip has {len(strides)} full strides (0..{len(strides) - 1}); "
                             "the report's symmetry.strides lists them")
        pick = strides[stride]
    else:                                         # the first from `foot`, else the first from the other
        pick = next((st for st in strides if st["foot"] == foot), strides[0] if strides else None)
    if pick:
        a, b = pick["start"] - 1, pick["start"] - 1 + pick["frames"]
        side = 0 if pick["foot"] == "left" else 1
    if a is None:                                 # no stride with one landing of the other foot inside
        for s_ in (side, 1 - side):
            lands = [f for f in all_lands[s_] if f > 0]
            pairs = [(x, y) for x, y in zip(lands, lands[1:]) if lo <= y - x <= hi]
            if pairs:
                (a, b), side = pairs[0], s_
                break
    if a is None:
        P = _period(soles, fps)
        for s_ in (side, 1 - side):
            fits = [f for f in all_lands[s_] if f > 0 and P and f + P <= n - 1]
            if fits:
                a, b, side = fits[0], fits[0] + P, s_
                break
    if a is None:
        raise ValueError(f"{action.name}: no full gait cycle (two landings of a foot, or a clear "
                         "period after one); retarget a longer stretch of steady locomotion")
    L = b - a
    other = [f for f in all_lands[1 - side] if a < f < b]
    steps = [other[0] - a, b - other[0]] if other else None
    chosen = next((st for st in strides if st["start"] == a + 1), None)
    bones = _keyed(meta)
    poses = _sample(armature, action, bones, [a + 1 + i for i in range(L + 1)])
    roots = _roots(armature, action, root)
    turn = math.degrees(_facing(armature, action, meta, [b + 1])[0] - _facing(armature, action, meta, [a + 1])[0])
    turn = (turn + 180) % 360 - 180
    D = (roots[b].translation - roots[a].translation)
    D.z = 0.0
    total = int(round(seconds * fps)) if seconds else (cycles or 2) * L
    # Residual between the cycle's two ends, removed progressively over the cycle.
    err = {bn: poses[L][bn][1] @ poses[0][bn][1].inverted() for bn in bones}
    # The root's own world orientation and height differ between the cycle's ends
    # too. Left alone they snap back at every seam -- a visible pop each step cycle.
    # Spread them over the cycle like every other bone's residual.
    root_err = roots[b].to_quaternion() @ roots[a].to_quaternion().inverted()
    if root_err.w < 0:
        root_err.negate()
    dz = roots[b].translation.z - roots[a].translation.z
    # Where in the cycle each bone takes up its seam residual. A leg -- thigh, calf, foot,
    # toes -- takes it in the swing that runs into the seam, after its last contact in
    # the cycle, so every stance keeps the clip's own sole angle. Spread linearly over the
    # whole cycle, a toe-off/swing mismatch would tip the mid-cycle stance toe-up (~25
    # degrees on CMU 16_56), and moving only the foot is not enough, because the calf
    # carries most of it. A leg planted at the seam (or never
    # lifting) takes it in its last swing before the seam instead. Other bones: linearly.
    # The cost: the root's residual stays linear, so a foot planted across the seam can
    # move ~1 cm on that one frame (81_03 walk), which the lock does not fully absorb.
    side_of = {}
    for side, hip in ((0, m["LeftUpLeg"]), (1, m["RightUpLeg"])):
        for name_ in _foot_bones(armature, hip):
            side_of[name_] = side
    ramps = []
    for side in (0, 1):
        down = [bool(planted[a + i, side]) for i in range(L)]
        end = L
        while end > 0 and down[end - 1]:          # planted through the seam: use the swing before it
            end -= 1
        start = end
        while start > 0 and not down[start - 1]:
            start -= 1
        if end - start < 2:                       # no usable swing: fall back to linear
            ramps.append([i / L for i in range(L + 1)])
            continue
        w = []
        for i in range(L + 1):
            x = 0.0 if i <= start else 1.0 if i >= end else (i - start) / (end - start)
            w.append(x * x * (3 - 2 * x))
        ramps.append(w)
    frames, root_world = [], []
    for k in range(total):
        i, rep = k % L, k // L
        p = i / L
        pose = {}
        for bn in bones:
            loc, q = poses[i][bn]
            pb_ = ramps[side_of[bn]][i] if bn in side_of else p
            corr = Quaternion().slerp(err[bn].inverted(), pb_)
            pose[bn] = (loc + (poses[0][bn][0] - poses[L][bn][0]) * pb_, corr @ q)
        frames.append(pose)
        src = roots[a + i]
        rot = (Quaternion().slerp(root_err.inverted(), p) @ src.to_quaternion()).to_matrix().to_4x4()
        rot.translation = src.translation + D * rep - Vector((0.0, 0.0, dz * p))
        root_world.append(rot)
    new = _new_action(armature, name or f"{action.name}_cycle", meta, frames, root_world)
    stats = _lock(armature, new, meta, planted=np.stack([planted[a + k % L] for k in range(total)]))
    warnings = ([f"the cycle turned {turn:.0f} degrees and was straightened; "
                 "for a curving walk use follow_path or a turning clip"] if abs(turn) > 10 else [])
    if steps is None:
        warnings.append("the other foot never lands inside the cycle; check it is a full stride")
    elif abs(steps[0] - steps[1]) > 0.25 * L:
        warnings.append(f"uneven steps in the cycle ({steps[0]} and {steps[1]} frames): the limp or "
                        "skip repeats every cycle; try another stretch of the clip or another clip")
    if chosen and chosen["uneven"] > 2:
        better = min(strides, key=lambda st: st["uneven"])
        hint = (f"stride={strides.index(better)} is the clip's most even ({better['uneven']})"
                if better["uneven"] < chosen["uneven"] else "no stride in this clip is more even; try another clip")
        warnings.append(f"this stride is uneven ({chosen['uneven']}: contacts {chosen['contact_frames']}, "
                        f"steps {chosen['step_frames']} frames), and every cycle repeats it; {hint}")
    warnings += _contact_warnings(stats, total / fps)
    return _report(new, meta, cycle_frames=L, cycle_start=a + 1, cycle_stride_m=round(D.length, 3),
                   cycle_turn_deg=round(turn, 1), step_frames=steps,
                   symmetry={"stride": chosen, "strides": strides}, warnings=warnings, **stats)



def _strides(planted, lands, lo, hi):
    """Every full stride (two landings of one foot `lo`..`hi` frames apart, with one
    landing of the other foot between) and how even it is. Frames are 1-based, as
    the action is keyed; the last contact before a landing ends where it ends."""
    def contact(i, f):
        g = f
        while g + 1 < len(planted) and planted[g + 1, i]:
            g += 1
        return g - f + 1
    out = []
    for side, label in ((0, "left"), (1, "right")):
        ls = [f for f in lands[side] if f > 0]
        for x, y in zip(ls, ls[1:]):
            if not lo <= y - x <= hi:
                continue
            between = [o for o in lands[1 - side] if x < o < y]
            if len(between) != 1:
                continue
            o = between[0]
            c_self, c_other = contact(side, x), contact(1 - side, o)
            contact_lr = [c_self, c_other] if side == 0 else [c_other, c_self]
            steps_lr = [o - x, y - o] if side == 0 else [y - o, o - x]
            out.append({"foot": label, "start": x + 1, "frames": y - x, "contact_frames": contact_lr,
                        "step_frames": steps_lr,
                        "uneven": abs(contact_lr[0] - contact_lr[1]) + abs(steps_lr[0] - steps_lr[1])})
    return out


def _period(heights, fps):
    """Gait period in frames from the feet's sole heights (frames x 2), or None.

    The first lag between 0.35 and 1.6 s whose mean autocorrelation over both
    feet comes within 10 % of the best; the first, so twice the period does not
    win on a long clip."""
    n = len(heights)
    lo, hi = max(2, int(round(0.35 * fps))), min(int(round(1.6 * fps)), n - max(3, int(0.3 * fps)))
    if hi <= lo:
        return None
    x = heights - heights.mean(axis=0)
    score = {}
    for lag in range(lo, hi + 1):
        a, b = x[:-lag], x[lag:]
        den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
        score[lag] = float(np.mean((a * b).sum(axis=0) / np.maximum(den, 1e-12)))
    best = max(score.values())
    if best < 0.5:
        return None
    return next((lag for lag in sorted(score) if score[lag] >= 0.9 * best and
                 score[lag] >= score.get(lag - 1, -1) and score[lag] >= score.get(lag + 1, -1)),
                max(score, key=score.get))


def follow_path(armature, action, points, *, name=None, smooth_m=0.4, lean=1.0):
    """Bend a clip's root path onto `points` [(x, y), ...] in world coordinates.

    The clip's progress along its own direction of travel is laid along the path
    (corners rounded over `smooth_m`), its sideways sway is kept, and the body is
    turned with the path and leans into its turns. Feet are re-pinned afterwards.
    Frames beyond the path's end are dropped; a path longer than the clip is not
    stretched -- lengthen the clip with `cycle` first. Gentle curves look right;
    turns faster than about 90 degrees per second are reported, and want a turning
    clip instead.

    The lean is the one that balances the turn, tan = v^2 * curvature / g, times
    `lean` (0 turns it off), capped at LEAN_MAX_DEG. Speed and curvature are
    averaged over a stride (LEAN_SIGMA_S) so the lean builds into a corner and holds
    instead of pumping with each step. The body tips about the ground under the
    pelvis and the feet stay flat.
    """
    action = _resolve(action)
    meta = _meta(action)
    root, fps = _body(meta), meta["fps"]
    roots = _roots(armature, action, root)
    n = len(roots)
    start = roots[0].translation.to_2d()
    travel = roots[-1].translation.to_2d() - start
    if travel.length < 0.2:
        raise ValueError(f"{action.name} travels {travel.length:.2f} m; follow_path needs locomotion")
    h0 = travel.normalized()
    side = Vector((-h0.y, h0.x))
    prog = [(r.translation.to_2d() - start).dot(h0) for r in roots]
    sway = [(r.translation.to_2d() - start).dot(side) for r in roots]

    pts = [Vector(p[:2]) for p in points]
    dense = [pts[0]]
    for p, q in zip(pts, pts[1:]):
        steps = max(1, int((q - p).length / 0.05))
        dense += [p.lerp(q, j / steps) for j in range(1, steps + 1)]
    arr = np.array([list(v) for v in dense])
    win = max(1, int(smooth_m / 0.05))
    if win > 1 and len(arr) > win:
        pad = np.pad(arr, ((win, win), (0, 0)), mode="edge")
        kern = np.ones(2 * win + 1) / (2 * win + 1)
        arr = np.stack([np.convolve(pad[:, i], kern, mode="valid") for i in range(2)], axis=1)
        arr[0], arr[-1] = list(pts[0]), list(pts[-1])
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    s_at = np.concatenate([[0.0], np.cumsum(seg)])
    path_len = float(s_at[-1])

    def at(s):
        s = min(max(s, 0.0), path_len)
        j = int(np.searchsorted(s_at, s, side="right") - 1)
        j = min(j, len(arr) - 2)
        t = (s - s_at[j]) / max(seg[j], 1e-9)
        pos = Vector(arr[j]) .lerp(Vector(arr[j + 1]), t)
        tan = Vector(arr[j + 1]) - Vector(arr[j])
        return pos, math.atan2(tan.y, tan.x)

    keep = [k for k in range(n) if prog[k] <= path_len + 1e-6] or [0]
    h0_yaw = math.atan2(h0.y, h0.x)
    root_world, yaws, line = [], [], []
    for k in keep:
        pos, yaw = at(prog[k])
        nrm = Vector((-math.sin(yaw), math.cos(yaw)))
        target = pos + nrm * sway[k]
        R = Matrix.Rotation(yaw - h0_yaw, 4, "Z")
        w = R @ roots[k]
        w.translation = Vector((target.x, target.y, roots[k].translation.z))
        root_world.append(w)
        yaws.append(yaw)
        line.append(pos)
    leans = _turn_lean(prog, keep, at, fps, lean)
    ground = meta.get("ground_z")
    ground = armature.matrix_world.translation.z if ground is None else ground
    tilts = []
    for j, (w, yaw) in enumerate(zip(root_world, yaws)):
        fwd = Vector((math.cos(yaw), math.sin(yaw), 0.0))
        tilt = Matrix.Rotation(-leans[j], 4, fwd)          # a left turn tips the top to the left
        # About the path on the ground, not under the swaying pelvis: a pivot that
        # moves sideways each step moves the planted feet up and down with it.
        piv = Vector((line[j].x, line[j].y, ground))
        root_world[j] = Matrix.Translation(piv) @ tilt @ Matrix.Translation(-piv) @ w
        tilts.append((tilt.to_3x3(), piv))
    bones = _keyed(meta)
    poses = _sample(armature, action, bones, [k + 1 for k in keep])
    planted = _contacts(armature, action, meta)
    new = _new_action(armature, name or f"{action.name}_path", meta, poses, root_world)
    if any(abs(a) > 1e-6 for a in leans):
        _untilt_feet(armature, new, meta, tilts)
    stats = _lock(armature, new, meta, shift_root=False, planted=planted[keep])
    rate = (max(abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi)) for a, b in zip(yaws, yaws[1:])) * fps
            if len(yaws) > 1 else 0.0)
    warnings = []
    if prog[-1] < path_len - 0.05:
        warnings.append(f"the clip covers {prog[-1]:.2f} m of a {path_len:.2f} m path; "
                        "lengthen it with cycle() or shorten the path")
    if len(keep) < n:
        warnings.append(f"dropped {n - len(keep)} frames past the path's end")
    if rate > TURN_RATE_DEG_S:
        warnings.append(f"turns up to {rate:.0f} deg/s; a turning clip will read better than bending a straight walk")
    if any(abs(math.degrees(a)) >= LEAN_MAX_DEG - 1e-6 for a in leans):
        warnings.append(f"the lean reached its {LEAN_MAX_DEG:.0f} deg cap: the turn is tight for this speed")
    return _report(new, meta, path_length_m=round(path_len, 3),
                   covered_m=round(min(prog[keep[-1]], path_len), 3),
                   max_turn_rate_deg_s=round(rate, 1),
                   lean_deg={"max": round(max((abs(math.degrees(a)) for a in leans), default=0.0), 1),
                             "max_frame_step": round(max((abs(math.degrees(b - a)) for a, b in zip(leans, leans[1:])),
                                                         default=0.0), 2)},
                   warnings=warnings, **stats)


def _gauss_1d(x, sigma):
    """Gaussian smoothing of a 1-D series (edges held), sigma in samples."""
    x = np.asarray(x, dtype=float)
    if sigma <= 0 or len(x) < 2:
        return x
    r = max(1, int(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(x, r, mode="edge"), w / w.sum(), mode="valid")


def _turn_lean(prog, keep, at, fps, scale):
    """Per kept frame, the lean (radians, + = to the left) that balances the turn:
    stride-averaged speed along the path and the path's curvature there, both
    smoothed over LEAN_SIGMA_S."""
    if not scale or len(keep) < 3:
        return [0.0] * len(keep)
    sigma = LEAN_SIGMA_S * fps
    p = np.array([prog[k] for k in keep])
    speed = _gauss_1d(np.gradient(p) * fps, sigma)
    d = 0.1
    curv = []
    for s in p:
        a, b = at(s - d)[1], at(s + d)[1]
        curv.append(((b - a + math.pi) % (2 * math.pi) - math.pi) / (2 * d))
    curv = _gauss_1d(curv, sigma)
    cap = math.radians(LEAN_MAX_DEG)
    return [max(-cap, min(cap, math.atan(scale * v * v * c / GRAVITY))) for v, c in zip(speed, curv)]


def _untilt_feet(armature, action, meta, tilts):
    """Lean the body, not the feet: put each foot back where and how it was before
    its frame's tilt (legs re-solved with two-bone IK), so soles stay flat and a
    foot's rise at push-off stays a rise. Tipping the feet with the body turned
    ~a quarter of that rise sideways at 15 degrees, which read as stance slip."""
    T0 = armature.matrix_world.copy()
    m = meta["bones"]
    legs = [(m["LeftUpLeg"], m["LeftLeg"], m["LeftFoot"]), (m["RightUpLeg"], m["RightLeg"], m["RightFoot"])]
    with _Using(armature, action):
        pbs = armature.pose.bones
        for f, (tilt, piv) in enumerate(tilts):
            _go(f + 1)
            back = tilt.inverted()
            for thigh, calf, foot in legs:
                M = T0 @ pbs[foot].matrix
                rot = back @ M.to_3x3()
                H, K, A = ((T0 @ pbs[b].matrix).translation.copy() for b in (thigh, calf, foot))
                goal = piv + back @ (A - piv)
                K2, A2 = _two_bone(H, K, A, goal)
                _aim(armature, T0, thigh, H, K, K2)
                bpy.context.view_layer.update()
                _aim(armature, T0, calf, (T0 @ pbs[calf].matrix).translation.copy(),
                     (T0 @ pbs[foot].matrix).translation.copy(), A2)
                bpy.context.view_layer.update()
                R = rot.to_4x4()
                R.translation = (T0 @ pbs[foot].matrix).translation
                pbs[foot].matrix = T0.inverted() @ R
                bpy.context.view_layer.update()
                for b in (thigh, calf, foot):
                    pbs[b].keyframe_insert("rotation_quaternion", frame=f + 1)


def adjust(armature, action, *, speed=1.0, stride=1.0, name=None):
    """Retime (`speed` > 1 is faster) and/or scale step length (`stride`) a little.

    Speed resamples time; stride scales the root's horizontal travel and re-pins
    the feet with leg IK at the new spacing. Both are standard small edits: past
    about +-20 % speed or +-15 % stride a clip at the wanted pace reads better, and
    the report says so.
    """
    action = _resolve(action)
    meta = _meta(action)
    root = _body(meta)
    n = _length(action)
    bones = _keyed(meta)
    count = max(2, int(round((n - 1) / speed)) + 1)
    times = [min(1 + k * speed, n) for k in range(count)]
    poses = _sample(armature, action, bones, times)
    T0 = armature.matrix_world.copy()
    root_world = None
    if abs(stride - 1.0) > 1e-6:
        with _Using(armature, action):
            first = None
            root_world = []
            for t in times:
                _go(math.floor(t), t - math.floor(t))
                w = (T0 @ armature.pose.bones[root].matrix).copy()
                first = first or w.translation.copy()
                d = w.translation - first
                w.translation = first + Vector((d.x * stride, d.y * stride, d.z))
                root_world.append(w)
    planted = _contacts(armature, action, meta)
    new = _new_action(armature, name or f"{action.name}_adj", meta, poses, root_world)
    stats = _lock(armature, new, meta, shift_root=abs(stride - 1.0) <= 1e-6,
                  planted=np.stack([planted[min(n - 1, int(round(t)) - 1)] for t in times]))
    warnings = []
    if abs(speed - 1) > SPEED_RANGE:
        warnings.append(f"speed x{speed:g} is past +-{SPEED_RANGE:.0%}; a clip at that pace will look more natural")
    if abs(stride - 1) > STRIDE_RANGE:
        warnings.append(f"stride x{stride:g} is past +-{STRIDE_RANGE:.0%}; legs will over-reach or crouch")
    return _report(new, meta, speed=speed, stride=stride, warnings=warnings, **stats)


def splice(armature, first, second, *, blend_s=0.25, name=None):
    """Join two retargeted clips at matching gait phase, with a short crossfade.

    The cut is the last landing of the left foot in `first` that leaves room for
    the blend; `second` enters at its first left-foot landing (both clips' start
    and end when either has none). `second` is moved and turned so its root and
    facing continue from `first` at the cut.
    """
    first, second = _resolve(first), _resolve(second)
    meta = _meta(first)
    m, fps = meta["bones"], meta["fps"]
    root = _body(meta)
    B = max(2, int(round(blend_s * fps)))
    n1, n2 = _length(first), _length(second)
    T0 = armature.matrix_world.copy()

    def lands(action, n):
        return _landings(_contacts(armature, action, _meta(action)))[0]

    l1 = [f for f in lands(first, n1) if f <= n1 - 1 - B]
    l2 = [f for f in lands(second, n2) if f <= n2 - 1 - B]
    a = l1[-1] if l1 else n1 - 1 - B
    b = l2[0] if l2 else 0
    roots1, roots2 = _roots(armature, first, root), _roots(armature, second, root)
    yaw1 = _facing(armature, first, meta, [a + 1])[0]
    yaw2 = _facing(armature, second, meta, [b + 1])[0]
    dyaw = (yaw1 - yaw2 + math.pi) % (2 * math.pi) - math.pi
    R = Matrix.Rotation(dyaw, 4, "Z")
    shift = roots1[a].translation - (R @ roots2[b]).translation
    shift.z = 0.0

    def moved(w):
        out = R @ w
        out.translation += shift
        return out

    bones = _keyed(meta)
    p1 = _sample(armature, first, bones, [f + 1 for f in range(min(n1, a + B))])
    p2 = _sample(armature, second, bones, [f + 1 for f in range(b, n2)])
    frames, roots = [], []
    for k in range(a):
        frames.append(p1[k])
        roots.append(roots1[k])
    for k in range(n2 - b):
        if k < B:
            w = (k + 1) / (B + 1)
            frames.append({bn: (p1[a + k][bn][0].lerp(p2[k][bn][0], w),
                                p1[a + k][bn][1].slerp(p2[k][bn][1], w)) for bn in bones})
            r1, r2 = roots1[a + k], moved(roots2[b + k])
            r = Matrix.Translation(r1.translation.lerp(r2.translation, w)) @ \
                r1.to_quaternion().slerp(r2.to_quaternion(), w).to_matrix().to_4x4()
            roots.append(r)
        else:
            frames.append(p2[k])
            roots.append(moved(roots2[b + k]))
    c1, c2 = _contacts(armature, first, meta), _contacts(armature, second, _meta(second))
    planted = np.stack([c1[k] for k in range(a)] +
                       [(c1[a + k] if k < B and (k + 1) / (B + 1) < 0.5 else c2[b + k]) for k in range(n2 - b)])
    new = _new_action(armature, name or f"{first.name}+{second.name}", meta, frames, roots)
    stats = _lock(armature, new, meta, planted=planted)
    return _report(new, meta, cut_first=a + 1, enter_second=b + 1, blend_frames=B,
                   turned_deg=round(math.degrees(dyaw), 1),
                   warnings=([] if (l1 and l2) else
                             ["no matching foot landing in one clip; joined at its ends instead"]), **stats)


def _jitter(frames, bones):
    """p95 over bones and frames of each rotation's departure from the midpoint of
    its neighbors (degrees): what one frame adds that the frames around it do not."""
    out = []
    for k in range(1, len(frames) - 1):
        for b in bones:
            q0, q1, q2 = frames[k - 1][b][1], frames[k][b][1], frames[k + 1][b][1]
            mid = q0.slerp(q2, 0.5)
            out.append(math.degrees(mid.rotation_difference(q1).angle))
    return round(float(np.percentile(out, 95)), 3) if out else 0.0


def smooth(armature, action, *, sigma_frames=1.0, root_sigma_frames=None, bones=None,
           name=None, repin=True):
    """Low-pass the motion: a Gaussian over time on every keyed rotation and the root.

    Optical mocap carries marker noise, and the per-frame wrist, foot, and IK
    corrections add frame-to-frame flicker, most visible in hands, feet, head, and
    the root. `sigma_frames` ~1 (a 3-5 frame window at 24 fps) removes flicker
    without softening real motion; 2+ starts to round off fast accents (a stamp, a
    clap). `bones` limits it to some bones; `root_sigma_frames` sets the root's
    translation separately (default: same). Quaternions are kept in one hemisphere
    and renormalized, so a smoothed rotation never takes the long way round.
    Smoothing moves planted feet a little, so `repin` re-pins them afterwards.
    Returns a new action and the jitter (p95 degrees, see `_jitter`) before and after.
    """
    action = _resolve(action)
    meta = _meta(action)
    root = _body(meta)
    n = _length(action)
    keyed = _keyed(meta)
    which = [b for b in keyed if bones is None or b in bones]
    frames = _sample(armature, action, keyed, [k + 1 for k in range(n)])

    def kernel(sigma):
        r = max(1, int(math.ceil(3 * sigma)))
        w = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sigma, 1e-6)) ** 2)
        return w / w.sum(), r

    def filt(series, sigma):
        if sigma <= 0:
            return series
        w, r = kernel(sigma)
        pad = np.pad(series, ((r, r), (0, 0)), mode="edge")
        return np.stack([np.convolve(pad[:, i], w, mode="valid") for i in range(series.shape[1])], axis=1)

    out = [dict(f) for f in frames]
    for b in which:
        qs = [frames[k][b][1].copy() for k in range(n)]
        for k in range(1, n):                       # one hemisphere, so averaging is local
            if qs[k].dot(qs[k - 1]) < 0:
                qs[k].negate()
        Q = filt(np.array([[q.w, q.x, q.y, q.z] for q in qs]), sigma_frames)
        for k in range(n):
            q = Quaternion(Q[k]).normalized()
            out[k][b] = (frames[k][b][0], q)
    rs = sigma_frames if root_sigma_frames is None else root_sigma_frames
    if root in which and rs > 0:
        L = filt(np.array([list(frames[k][root][0]) for k in range(n)]), rs)
        for k in range(n):
            out[k][root] = (Vector(L[k]), out[k][root][1])
    new = _new_action(armature, name or f"{action.name}_smooth", meta, out)
    if repin:
        stats = _lock(armature, new, meta, shift_root=False, planted=_contacts(armature, action, meta))
    else:
        stats = {}
        stored = _stored_contacts(action)                # same timing: pass the record on, unread
        if stored is not None:
            _store_contacts(new, stored)
    return _report(new, meta, sigma_frames=sigma_frames,
                   jitter_p95_deg={"before": _jitter(frames, which),
                                   "after": _jitter(_sample(armature, new, which, [k + 1 for k in range(n)]), which)},
                   **stats)
