"""Which human joints an imported character has, their limits, and how its rig bends.

Pure Python (numpy, math, re), with no dycraft imports, so the sandboxed Blender check
worker can load it from `/tools` as well as the importer: an ObjectIR is
content-addressed, so a rig imported before `ANATOMY_VERSION` keeps its stored block,
and `verify.current_anatomy` derives the current one from the armature in the scene
being checked. `skeleton_anatomy` reads the bone rows `_import_gltf_blender` reports
(name, parent, head, tail, deform, and the rest frame).
"""
from __future__ import annotations

import math
import re

import numpy as np


# ---------------------------------------------------------------------------
# Anatomy: joint limits by skeleton family
# ---------------------------------------------------------------------------

#: Skeleton families recognized from bone names. A pattern is tried on the
#: name without an exporter's `prefix:` and then, failing that, also without a
#: trailing `_<n>` / `.<n>` (`_role_name`). Each pattern yields a side (`L`/`R`,
#: or none) and a family-specific bone word, mapped to one of the canonical
#: roles below. Biped is 3ds Max's (Rocketbox, and any rig named
#: `<prefix> L UpperArm`, Valve's `ValveBiped.Bip01_L_UpperArm` too); mixamo is
#: Adobe's (also ReadyPlayerMe and the many Sketchfab exports of it); unreal is
#: the Epic mannequin's lower-case naming, which Renderpeople's rigged scans use
#: with `upperleg`/`lowerleg`. The rest were found in the bone names of the
#: 45,692 rigged Objaverse characters and animals that the families above left
#: unmapped (counts are rigs carrying the family's thigh bone):
#: blender, the metarig's `thigh.L` and hand-built rigs' `UpperLeg.L` (~2,600);
#: source2, Valve's `bip_hip_L` (1,446); character_creator, Reallusion's
#: `CC_Base_L_Thigh` (779); advanced_skeleton, Maya AdvancedSkeleton's
#: joint-named `Hip_L`/`Knee_L`/`Ankle_L` (686); rigify, the generated rig's
#: deform bones `DEF-thigh.L`, never its ORG-/MCH-/control bones (620); mmd,
#: MikuMikuDance's English `Left leg`/`Left knee`/`Left ankle` (596); bn and
#: jnt, two game exporters' `L_Leg_01_Bn` and `L_UpperLeg_jnt` (502, 408); and
#: unity, Unity humanoid / VRM `LeftUpperLeg` and VRoid `J_Bip_L_UpperLeg`.
#: Several name a bone after the joint it starts at (`Shoulder_L` is the upper
#: arm, not the clavicle); for those, `_linked` keeps a mapping only where the
#: chain the names claim is the chain the parents form.
#: Biped, mixamo, and unreal also name the finger joints, for `bend_axes`.
_LIMBS_UNREAL = {"upperarm": "upper_arm", "lowerarm": "forearm", "hand": "hand", "thigh": "thigh",
                 "upperleg": "thigh", "calf": "shin", "lowerleg": "shin", "foot": "foot"}
SKELETON_FAMILIES = {
    "biped": [(r"^(?:.+ )?(?P<side>[LR]) (?P<word>UpperArm|Forearm|Hand|Thigh|Calf|Foot)$",
               {"UpperArm": "upper_arm", "Forearm": "forearm", "Hand": "hand",
                "Thigh": "thigh", "Calf": "shin", "Foot": "foot"}),
              (r"^ValveBiped\.Bip\d+_(?P<side>[LR])_(?P<word>UpperArm|Forearm|Hand|Thigh|Calf|Foot)$",
               {"UpperArm": "upper_arm", "Forearm": "forearm", "Hand": "hand",
                "Thigh": "thigh", "Calf": "shin", "Foot": "foot"}),
              (r"^(?:.+ )?(?P<word>Neck|Head)$", {"Neck": "neck", "Head": "head"}),
              (r"^ValveBiped\.Bip\d+_(?P<word>Neck|Head)1$", {"Neck": "neck", "Head": "head"}),
              (r"^(?:.+ )?(?P<side>[LR]) (?P<word>Finger[0-4][12]?)$",
               {f"Finger{d}{p}": f"{name}_{int(p or 0) + 1}"
                for d, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                for p in ("", "1", "2")})],
    "mixamo": [(r"^(?:[Mm]ixamorig\d*_)?(?P<side>Left|Right)(?P<word>Arm|ForeArm|Hand|UpLeg|Leg|Foot)$",
                {"Arm": "upper_arm", "ForeArm": "forearm", "Hand": "hand",
                 "UpLeg": "thigh", "Leg": "shin", "Foot": "foot"}),
               (r"^(?:[Mm]ixamorig\d*_)?(?P<word>Neck|Head)$", {"Neck": "neck", "Head": "head"}),
               (r"^(?:[Mm]ixamorig\d*_)?(?P<side>Left|Right)(?P<word>Hand(?:Thumb|Index|Middle|Ring|Pinky)[123])$",
                {f"Hand{d}{p}": f"{d.lower()}_{p}"
                 for d in ("Thumb", "Index", "Middle", "Ring", "Pinky") for p in "123"})],
    "unreal": [(r"^(?P<word>upperarm|lowerarm|hand|thigh|upperleg|calf|lowerleg|foot)_(?P<side>[lr])$",
                _LIMBS_UNREAL),
               (r"^(?P<word>neck|head)$", {"neck": "neck", "head": "head"}),
               (r"^(?P<word>(?:thumb|index|middle|ring|pinky)_0[123])_(?P<side>[lr])$",
                {f"{d}_0{p}": f"{d}_{p}"
                 for d in ("thumb", "index", "middle", "ring", "pinky") for p in "123"})],
    "blender": [(r"^(?i:(?P<word>upper_?arm|fore_?arm|lower_?arm|hand|thigh|upper_?leg|shin|calf|lower_?leg|foot))"
                 r"\.(?P<side>[LR])$",
                 {"upper_arm": "upper_arm", "upperarm": "upper_arm", "forearm": "forearm", "fore_arm": "forearm",
                  "lowerarm": "forearm", "lower_arm": "forearm", "hand": "hand", "thigh": "thigh",
                  "upperleg": "thigh", "upper_leg": "thigh", "shin": "shin", "calf": "shin",
                  "lowerleg": "shin", "lower_leg": "shin", "foot": "foot"}),
                (r"^(?i:(?P<word>neck|head))$", {"neck": "neck", "head": "head"})],
    "rigify": [(r"^DEF-(?P<word>upper_arm|forearm|hand|thigh|shin|foot)\.(?P<side>[LR])$",
                {"upper_arm": "upper_arm", "forearm": "forearm", "hand": "hand",
                 "thigh": "thigh", "shin": "shin", "foot": "foot"}),
               (r"^DEF-(?P<word>spine\.004|spine\.006|neck|head)$",
                {"spine.004": "neck", "spine.006": "head", "neck": "neck", "head": "head"})],
    "character_creator": [(r"^CC_Base_(?P<side>[LR])_(?P<word>Upperarm|Forearm|Hand|Thigh|Calf|Foot)$",
                           {"Upperarm": "upper_arm", "Forearm": "forearm", "Hand": "hand",
                            "Thigh": "thigh", "Calf": "shin", "Foot": "foot"}),
                          (r"^CC_Base_(?P<word>NeckTwist01|Head)$", {"NeckTwist01": "neck", "Head": "head"})],
    "source2": [(r"^bip_(?P<word>upperArm|upperarm|lowerArm|lowerarm|hand|hip|thigh|knee|foot)_(?P<side>[LR])$",
                 {"upperArm": "upper_arm", "upperarm": "upper_arm", "lowerArm": "forearm", "lowerarm": "forearm",
                  "hand": "hand", "hip": "thigh", "thigh": "thigh", "knee": "shin", "foot": "foot"}),
                (r"^bip_(?P<word>neck|head)$", {"neck": "neck", "head": "head"})],
    "advanced_skeleton": [(r"^(?P<word>Shoulder|Elbow|Wrist|Hip|Knee|Ankle)_(?P<side>[LR])$",
                           {"Shoulder": "upper_arm", "Elbow": "forearm", "Wrist": "hand",
                            "Hip": "thigh", "Knee": "shin", "Ankle": "foot"}),
                          (r"^(?P<word>Neck|Head)_M$", {"Neck": "neck", "Head": "head"})],
    "mmd": [(r"^(?P<side>Left|Right) (?P<word>arm|elbow|wrist|leg|knee|ankle)$",
             {"arm": "upper_arm", "elbow": "forearm", "wrist": "hand",
              "leg": "thigh", "knee": "shin", "ankle": "foot"}),
            (r"^(?P<word>Neck|Head)$", {"Neck": "neck", "Head": "head"})],
    "bn": [(r"^(?P<side>[LR])_(?P<word>Arm_01|Arm_02|Arm_03|Leg_01|Leg_02|Leg_03)_Bn$",
            {"Arm_01": "upper_arm", "Arm_02": "forearm", "Arm_03": "hand",
             "Leg_01": "thigh", "Leg_02": "shin", "Leg_03": "foot"}),
           (r"^(?P<word>Neck|Head)_Bn$", {"Neck": "neck", "Head": "head"})],
    "jnt": [(r"^(?P<side>[LR])_(?P<word>Shoulder|Elbow|Wrist|UpperLeg|Knee|Ankle)_jnt$",
             {"Shoulder": "upper_arm", "Elbow": "forearm", "Wrist": "hand",
              "UpperLeg": "thigh", "Knee": "shin", "Ankle": "foot"}),
            (r"^(?P<word>Neck|Head)_jnt$", {"Neck": "neck", "Head": "head"})],
    "unity": [(r"^(?:J_Bip_)?(?P<side>Left|Right|L_|R_)(?P<word>UpperArm|LowerArm|Hand|UpperLeg|LowerLeg|Foot)$",
               {"UpperArm": "upper_arm", "LowerArm": "forearm", "Hand": "hand",
                "UpperLeg": "thigh", "LowerLeg": "shin", "Foot": "foot"}),
              (r"^(?:J_Bip_C_)?(?P<word>Neck|Head)$", {"Neck": "neck", "Head": "head"})],
}
#: The roles that make a family a body: `skeleton_roles` needs four of them, so
#: finger bones (or a quadruped's `Neck` and `Head`) do not make one.
LIMB_ROLES = ("upper_arm", "forearm", "hand", "thigh", "shin", "foot")
#: Fingers whose joints `bend_axes` describes; the thumb folds across the palm
#: along a plane its rest pose does not give, so it is left out.
FINGERS = ("index", "middle", "ring", "pinky")
#: Which role each limb role hangs from, for `_linked`.
_CHAIN = {"forearm": "upper_arm", "hand": "forearm", "shin": "thigh", "foot": "shin"}
#: Families that name bones after joints, where a word alone does not say which bone it is (`Shoulder_L` is the
#: clavicle in some rigs): only their mappings must form the chains the names claim. The others keep any
#: hierarchy -- a flattened export, every bone under the root, measures its joints just the same.
_CHAIN_CHECKED = {"advanced_skeleton", "mmd", "bn", "jnt"}

#: The human joints checked, in degrees. A hinge folds from the straight limb
#: toward `fold` (body axes after normalization: +Y front, +Z up): the elbow
#: forward, the knee backward. `flexion` below its range is the limb bending
#: backwards; `off_plane` is the limb swinging out of its hinge plane. A cone
#: bounds how far a segment turns from its rest direction relative to its
#: parent, without a twist component. Deliberately generous -- these flag poses
#: no body reaches, not poses an animator would avoid -- and they are
#: advisory, as every AnatomicalJointLimits finding is. The cones are set from
#: measured poses: an open-palm "stay" signal reaches 128 degrees at the wrist
#: and kneeling on the instep 79 at the ankle, both real poses, while a hand
#: folded back onto its forearm reaches 161. A cone is measured from the rest
#: pose, so a rig whose rest hand is already flexed reads that much higher,
#: hence the wide wrist bound.
HUMAN_JOINTS = (
    {"name": "elbow", "kind": "hinge", "parent": "upper_arm", "child": "forearm",
     "end": "hand", "fold": (0.0, 1.0, 0.0), "flexion_deg": [-10.0, 160.0], "off_plane_deg": 35.0},
    {"name": "knee", "kind": "hinge", "parent": "thigh", "child": "shin",
     "end": "foot", "fold": (0.0, -1.0, 0.0), "flexion_deg": [-10.0, 160.0], "off_plane_deg": 25.0},
    {"name": "wrist", "kind": "cone", "parent": "forearm", "child": "hand",
     "vector": "segment", "max_deg": 145.0},
    {"name": "ankle", "kind": "cone", "parent": "shin", "child": "foot",
     "vector": (0.0, 1.0, 0.0), "max_deg": 85.0},
    {"name": "head", "kind": "cone", "parent": "neck", "child": "head",
     "vector": (0.0, 0.0, 1.0), "max_deg": 75.0},
)
#: A cone's reference vector is fixed to the child bone at rest and compared
#: with its rest direction. Bone tails are not used for it: Sketchfab exports
#: carry end bones whose tails sit at the origin or a meter below the floor
#: (`head_end` at z = 0, `eye_end_l` at z = -1.7 on a Renderpeople scan), and
#: a bone's own axis inherits that. So the wrist uses the forearm's line
#: continued through the hand ("segment", elbow head to wrist head), which a
#: forearm twist -- pronation, which is not a wrist motion -- leaves alone; the
#: ankle the figure's forward axis (where the foot points); the head its up
#: axis.
ANATOMY_POLICIES = ("auto", "human", "none")


def _role_name(name: str) -> str:
    """A bone name without an exporter's namespace or a trailing copy/index suffix."""
    name = name.rsplit(":", 1)[-1]
    return re.sub(r"(?:[_.]\d+)+$", "", name)


def _linked(bones: list, found: dict, depth: int = 4) -> dict:
    """`found` less the limb roles whose bone does not hang from the role above it.

    A forearm must descend from its upper arm, a hand from its forearm, a shin
    from its thigh and a foot from its shin, within `depth` parents (twist and
    share bones may sit between). Bones without parent information are trusted.
    A role that fails takes the roles below it on that side with it.
    """
    parent = {b["name"]: b.get("parent") for b in bones}
    if not any(parent.values()):
        return found
    out = dict(found)
    for role in ("forearm", "hand", "shin", "foot"):           # top down, so a failure propagates
        for side in ("L", "R"):
            key, up = (role, side), (_CHAIN[role], side)
            if key not in out:
                continue
            ok, b = False, parent.get(out[key])
            for _ in range(depth):
                if b is None:
                    break
                if up in out and b == out[up]:
                    ok = True
                    break
                b = parent.get(b)
            if not ok:
                out.pop(key)
    return out


def skeleton_roles(bones: list) -> tuple:
    """`(family, {(role, side): bone name})` for the best-matching family, or `(None, {})`.

    A role two bones claim goes to the one whose full name matched (Rigify's
    `DEF-upper_arm.L` over its second segment `DEF-upper_arm.L.001`); with no
    such bone it is dropped rather than guessed: twist and helper bones are
    named apart in every family above, so a clash means the naming is not what
    the family assumes. In the joint-named families, limb roles must also form
    the chains their names claim (`_linked`).
    """
    best = (None, {})
    for family, patterns in SKELETON_FAMILIES.items():
        claims = {}
        for bone in bones:
            full = bone["name"].rsplit(":", 1)[-1]
            short = _role_name(bone["name"])
            for exact, name in ((True, full), (False, short)):
                hit = None
                for pattern, words in patterns:
                    m = re.match(pattern, name)
                    if m:
                        hit = m
                        break
                if hit is None:
                    continue
                side = (hit.groupdict().get("side") or "")[:1].upper() or None
                word = hit.group("word")
                role = words.get(word) or words[word.lower()]      # case-insensitive patterns
                claims.setdefault((role, side), []).append((exact, bone["name"]))
                break
        found = {}
        for key, cands in claims.items():
            exact = [n for e, n in cands if e]
            if len(cands) == 1:
                found[key] = cands[0][1]
            elif len(exact) == 1:
                found[key] = exact[0]
        if family in _CHAIN_CHECKED:
            found = _linked(bones, found)
        # A family is its limbs: "Neck" and "Head" alone match half the
        # quadrupeds on Sketchfab (a labrador rig among them).
        limbs = sum(1 for role, _ in found if role in LIMB_ROLES)
        if limbs >= 4 and len(found) > len(best[1]):
            best = (family, found)
    return best


def _humanlike(bones: list, roles: dict, height: float) -> tuple:
    """Whether the rest figure is built like a standing human, and why not.

    Bone names say "biped" for a crow as readily as for a person -- the
    library's crow is a 3ds Max Biped -- and a bird's wing and knee fold differently, so
    the name match alone cannot apply human limits. Rest proportions can tell
    them apart: on a standing human the head is at the top of the figure and
    the hip joints about halfway up, and each leg spans roughly half its
    height. Read on bone *heads* against the mesh's height (`height`, feet on
    z = 0 after normalization): end-bone tails are not reliable, see
    `HUMAN_JOINTS`. The torso must stand too, shoulders well above the hips: a
    quadruped auto-rigged with a Mixamo skeleton can hold its head high with
    its hind legs spanning half its height, but its shoulders sit level with
    its hips, where a person's are a quarter to two fifths of the height above
    them.
    """
    need = [("upper_arm", s) for s in "LR"] + [("forearm", s) for s in "LR"] + \
           [("hand", s) for s in "LR"] + [("thigh", s) for s in "LR"] + \
           [("shin", s) for s in "LR"] + [("foot", s) for s in "LR"] + [("head", None)]
    missing = [f"{r}{'_' + s if s else ''}" for r, s in need if (r, s) not in roles]
    if missing:
        return False, f"skeleton lacks {', '.join(missing)}"
    by_name = {b["name"]: b for b in bones}
    if not height or height <= 0:
        return False, "degenerate rest pose"
    def z(role, side=None):
        return by_name[roles[(role, side)]]["head"][2] / height
    hip = (z("thigh", "L") + z("thigh", "R")) / 2
    head = z("head")
    leg = hip - (z("foot", "L") + z("foot", "R")) / 2
    if not (head >= 0.75 and 0.4 <= hip <= 0.65 and leg >= 0.35):
        return False, (f"rest proportions are not a standing human's (head at {head:.2f}, "
                       f"hips at {hip:.2f}, leg span {leg:.2f} of the height)")
    shoulders = (z("upper_arm", "L") + z("upper_arm", "R")) / 2
    if shoulders - hip < 0.12:
        return False, (f"the torso does not stand: shoulders {shoulders - hip:.2f} of the height above "
                       f"the hips (a horizontal body on a humanoid skeleton: a four-legged animal, a dinosaur)")
    return True, ""


#: `native_rig.anatomy.version`: 2 adds `body` (the whole-body measures below),
#: `hands` and `bend_axes`.
ANATOMY_VERSION = 2


def _body_references(by_name: dict, roles: dict, joints: list) -> tuple:
    """`(body, hands, bend_axes, skipped)`: what the whole-body measures need, and how the rig bends.

    The measures were chosen because none depends on how a rig rolls its bones (checked
    against blind review on 130 library rigs): elbow and knee bend and swivel (the roll the upper bone would need for the bend
    to lie in its hinge plane), shoulder and hip roll against how far the limb is raised,
    forearm rotation from the anatomical neutral, wrist flexion and tilt on the palm's axes,
    ankle flexion, and the head's and spine's anatomical (Cardan) split. Every vector here is
    in the rest frame of the bone that carries it, so `verify.body_measures` needs only the
    evaluated bone orientations. The palm normal points out of the palm, the side authored
    clips curl the fingers toward; reversed, wrist flexion reads as extension.

    `hands` gives each palm's normal and `bend_axes` each elbow's, knee's, and finger joint's
    axis in the child bone's own frame that flexes it when turned positively: +X on a Mixamo
    finger, -Z on a 3ds Max Biped's.
    """
    parent = {name: b.get("parent") for name, b in by_name.items()}
    order = list(by_name)
    def R(bone):
        return np.asarray(by_name[bone]["rotation"], dtype=float)
    def head(bone):
        return np.asarray(by_name[bone]["head"], dtype=float)
    def unit(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None
    def local(bone, v):
        return None if v is None else R(bone).T @ v
    r6 = lambda v: [round(float(x), 6) for x in np.ravel(v)]
    hinge = {j["name"]: j for j in joints if j["kind"] == "hinge"}
    body, hands, axes, skipped = {}, {}, {}, []
    thighs = [roles.get(("thigh", side)) for side in "LR"]
    arms = [roles.get(("upper_arm", side)) for side in "LR"]
    neck, skull = roles.get(("neck", None)), roles.get(("head", None))
    # Toward the figure's right: across the shoulders for the head and spine, across the
    # hips for the hip's split.
    shoulders = head(arms[1]) - head(arms[0]) if all(arms) else None
    hips_across = head(thighs[1]) - head(thighs[0]) if all(thighs) else None
    def cardan_frame(long_axis, lateral):
        z = unit(long_axis)
        if z is None or lateral is None:
            return None
        x = unit(lateral - (lateral @ z) * z)
        return None if x is None else np.stack([x, np.cross(z, x), z], axis=1)
    for side, mirror in (("L", -1.0), ("R", 1.0)):
        # Elbow and knee: the hinge plus what its swivel needs.
        for name, roll in (("elbow", "shoulder"), ("knee", "hip")):
            j = hinge.get(f"{name}_{side}")
            if j is None:
                continue
            up, low = j["parent"], j["child"]
            j["bend_axis_local"] = r6(R(low).T @ (R(up) @ np.cross(j["zero"], j["fold"])))
            grand = parent.get(up)
            if grand not in by_name:
                skipped.append(f"{roll}_{side}")
                continue
            entry = {"hinge": j["name"], "upper": up, "lower": low, "grandparent": grand,
                     "grand_rest_relative": r6(R(grand).T @ R(up)),
                     "upper_segment": r6(local(up, head(low) - head(up)))}
            if name == "elbow":
                # Against the chest (the neck's parent), down being the line from the neck to
                # the hips; without a neck, against the bone the arm hangs from, down being -Z.
                ref = parent.get(neck) if neck else grand
                down = (sum(head(t) for t in thighs) / 2 - head(neck)) if neck and all(thighs) else \
                    np.array([0.0, 0.0, -1.0])
                if ref in by_name and unit(down) is not None:
                    entry.update(elevation_ref=ref, elevation_down=r6(local(ref, unit(down))))
            else:
                entry.update(elevation_ref=grand, elevation_down=r6(local(grand, np.array([0.0, 0.0, -1.0]))))
                B = cardan_frame(head(low) - head(up), hips_across)
                if B is not None:
                    entry.update(cardan_rest=r6(R(up)), cardan_basis=r6(B))
            body[f"{roll}_{side}"] = entry
        # Forearm rotation and the wrist, from the index and little fingers' knuckles.
        up, fo, ha = (roles.get((r, side)) for r in ("upper_arm", "forearm", "hand"))
        index, little = roles.get(("index_1", side)), roles.get(("pinky_1", side))
        middle = roles.get(("middle_1", side)) or index
        if up and fo and ha and index and little and f"elbow_{side}" in hinge:
            body[f"wrist_{side}"] = {
                "upper": up, "forearm": fo, "hand": ha, "hinge": f"elbow_{side}", "mirror": mirror,
                "upper_segment": r6(local(up, head(fo) - head(up))),
                "forearm_segment": r6(local(fo, head(ha) - head(fo))),
                "hand_length": r6(local(ha, (head(index) + head(little)) / 2 - head(ha))),
                "radial": r6(local(ha, head(index) - head(little)))}
        elif ha:
            skipped.append(f"wrist_{side} (no index and little-finger knuckles)")
        # The ankle: the foot's turn about the knee's hinge axis from rest.
        th, sh, ft = (roles.get((r, side)) for r in ("thigh", "shin", "foot"))
        toe = next((b for b in order if parent.get(b) == ft), None) if ft else None
        kn = hinge.get(f"knee_{side}")
        if th and sh and ft and toe and kn:
            n = np.cross(kn["zero"], kn["fold"])
            n0, shin0, foot0 = unit(R(th) @ n), unit(head(ft) - head(sh)), unit(head(toe) - head(ft))
            if n0 is not None and shin0 is not None and foot0 is not None:
                rest_angle = math.degrees(math.atan2(float(n0 @ np.cross(-shin0, foot0)), float(-shin0 @ foot0)))
                body[f"ankle_{side}"] = {"thigh": th, "shin": sh, "foot": ft, "hinge_normal": r6(n),
                                         "shin_segment": r6(local(sh, shin0)), "foot_segment": r6(local(ft, foot0)),
                                         "rest_angle": round(rest_angle, 4)}
        # The palm, and each finger joint's bend axis toward it.
        if ha and index and little and middle:
            along = unit(head(middle) - head(ha))
            palm = unit(np.cross(along, head(index) - head(little))) if along is not None else None
            if palm is not None:
                palm = palm * (1.0 if side == "L" else -1.0)
                hands[side] = {"palm_normal": r6(palm), "palm_normal_local": r6(R(ha).T @ palm)}
                for digit in FINGERS:
                    chain = [roles.get((f"{digit}_{k}", side)) for k in (1, 2, 3)]
                    chain = [b for b in chain[:chain.index(None)]] if None in chain else chain
                    for k, bone in enumerate(chain):
                        nxt = chain[k + 1] if k + 1 < len(chain) else next(
                            (b for b in order if parent.get(b) == bone), None)
                        seg = unit(head(nxt) - head(bone)) if nxt else unit(R(bone)[:, 1])
                        if seg is None:
                            continue
                        axis = unit(np.cross(seg, palm))
                        if axis is not None:
                            axes[f"{digit}_{('knuckle', 'pip', 'dip')[k]}_{side}"] = r6(R(bone).T @ axis)
    # The head against the neck's parent, and the spine from the hips to that parent.
    base = parent.get(neck) if neck else None
    if neck and skull and base in by_name:
        B = cardan_frame(head(skull) - head(neck), shoulders)
        if B is not None:
            body["head"] = {"head": skull, "base": base, "rest_relative": r6(R(base).T @ R(skull)),
                            "cardan_rest": r6(R(skull)), "cardan_basis": r6(B)}
        hips = base
        while parent.get(hips) in by_name and parent.get(parent.get(hips)) is not None:
            hips = parent[hips]
        B = cardan_frame(head(neck) - head(hips), shoulders)
        if hips != base and B is not None:
            body["spine"] = {"chest": base, "hips": hips, "rest_relative": r6(R(hips).T @ R(base)),
                             "cardan_rest": r6(R(base)), "cardan_basis": r6(B)}
    for j in joints:
        if j["kind"] == "hinge" and "bend_axis_local" in j:
            axes[j["name"]] = j["bend_axis_local"]
    return body, hands, axes, skipped


def skeleton_anatomy(bones: list, policy: str = "auto", height: float | None = None,
                     roles: dict | None = None) -> dict:
    """The `native_rig.anatomy` block: recognized skeleton and per-joint limits.

    `bones` are the Blender child's report rows, which carry each bone's rest
    `head`, `tail`, and `rotation` (its axes as columns, armature space -- the
    character's own frame after normalization). Every reference direction is
    stored in the *parent bone's* rest frame and the child's segment in the
    child bone's, so `animation.blender.verify.anatomical_measure` needs only
    the two bones' evaluated orientations, which the evaluation records.
    `height` is the figure's (the mesh's, feet on z = 0); without it, the
    highest bone head stands in. `roles` ({(role, side): bone}) replaces the
    roles read from the bone names, for a rig whose roles were written by hand;
    its skeleton is then "given".
    """
    family, roles = skeleton_roles(bones) if roles is None else ("given", dict(roles))
    if height is None:
        height = max((b["head"][2] for b in bones), default=0.0)
    out = {"skeleton": family or "unknown", "policy": policy, "joints": []}
    if policy == "none":
        out["reason"] = "--anatomy none"
        return out
    if family is None:
        out["reason"] = "no known skeleton naming"
        return out
    if policy == "auto":
        ok, why = _humanlike(bones, roles, height)
        if not ok:
            out["reason"] = why + "; pass --anatomy human to apply human limits anyway"
            return out
    by_name = {b["name"]: b for b in bones}
    def unit(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None
    def frame(bone):
        return np.asarray(by_name[bone]["rotation"], dtype=float)
    def head(bone):
        return np.asarray(by_name[bone]["head"], dtype=float)
    rounded = lambda v: [round(float(x), 6) for x in v]
    skipped = []
    for spec in HUMAN_JOINTS:
        sides = ("L", "R") if spec["name"] != "head" else (None,)
        for side in sides:
            label = spec["name"] + (f"_{side}" if side else "")
            parent = roles.get((spec["parent"], side))
            child = roles.get((spec["child"], side))
            end = roles.get((spec.get("end"), side)) if spec.get("end") else None
            if parent is None or child is None or (spec.get("end") and end is None):
                skipped.append(label)
                continue
            Rp, Rc = frame(parent), frame(child)
            if spec["kind"] == "hinge":
                upper = unit(head(child) - head(parent))
                lower = unit(head(end) - head(child))
                if upper is None or lower is None:
                    skipped.append(label)
                    continue
                zero = Rp.T @ upper
                fold = Rp.T @ np.asarray(spec["fold"], dtype=float)
                fold = unit(fold - (fold @ zero) * zero)
                if fold is None:
                    skipped.append(label)
                    continue
                axis = np.cross(zero, fold)
                child_vector = Rc.T @ lower
                d = Rp.T @ lower
                p = d - (d @ axis) * axis
                rest = math.degrees(math.atan2(float(p @ fold), float(p @ zero)))
                out["joints"].append({
                    "name": label, "kind": "hinge", "parent": parent, "child": child,
                    "axis": rounded(axis), "zero": rounded(zero), "fold": rounded(fold),
                    "child_vector": rounded(child_vector),
                    "flexion_deg": list(spec["flexion_deg"]),
                    "off_plane_deg": spec["off_plane_deg"],
                    "rest_flexion_deg": round(rest, 2)})
            else:
                if spec["vector"] == "segment":
                    segment = unit(head(child) - head(parent))
                else:
                    segment = unit(spec["vector"])
                if segment is None:
                    skipped.append(label)
                    continue
                out["joints"].append({
                    "name": label, "kind": "cone", "parent": parent, "child": child,
                    "child_vector": rounded(Rc.T @ segment),
                    "rest_direction": rounded(Rp.T @ segment),
                    "max_deg": spec["max_deg"]})
    body, hands, bend_axes, more = _body_references(by_name, roles, out["joints"])
    out.update(version=ANATOMY_VERSION, body=body, hands=hands, bend_axes=bend_axes)
    skipped += more
    if skipped:
        out["skipped"] = skipped
    return out
