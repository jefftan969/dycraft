"""Anatomical joint limits of imported characters: the body's measures, their ranges, and per-frame bone rows."""
from __future__ import annotations

import math

import numpy as np

try:
    from .continuity import frame_runs, quaternion_matrices
    from .rig_anatomy import ANATOMY_VERSION, skeleton_anatomy
except ImportError:  # Read-only modules staged into isolated Blender.
    from continuity import frame_runs, quaternion_matrices
    from rig_anatomy import ANATOMY_VERSION, skeleton_anatomy


def rig_anatomy_table(scene_ir, catalog):
    """`{armature object: anatomy}` for every placed character whose rig declares one.

    The armature is `rig/<instance>/<part>` (`worker.RIG_PREFIX` + the part
    identity, as `_swap_in_native_rig` names it), and the joints come from the
    character's `native_rig.anatomy`, written at import by
    `rig_anatomy.skeleton_anatomy`. A rig without one -- an unrecognized
    skeleton, a non-human figure, an asset imported before it existed -- simply
    has no entry.
    """
    table = {}
    for inst in (scene_ir or {}).get("instances", ()):
        rig = ((catalog or {}).get(inst.get("object_ref")) or {}).get("native_rig") or {}
        anatomy = rig.get("anatomy") or {}
        if anatomy.get("joints"):
            table[f"rig/{inst['name']}/{rig['part']}"] = anatomy
    return table


def current_anatomy(table, rigs):
    """`table` with each rig's anatomy as `ANATOMY_VERSION` declares it.

    An ObjectIR is content-addressed, so a character imported before the
    current version keeps the anatomy it froze: five joints, no `body`. Its
    joints are derived here instead, from the rest frames the evaluation
    recorded for its armature (`worker.armature_table`), the same rows the
    importer derives them from. Human limits apply because the stored block
    already passed the import's human-proportions test; a rig whose evaluation
    has no rest frames keeps what it stored.
    """
    out = {}
    for armature, anatomy in table.items():
        spec = (rigs or {}).get(armature) or {}
        if anatomy.get("version", 1) >= ANATOMY_VERSION or not spec.get("rest"):
            out[armature] = anatomy
            continue
        rows = [dict(rest, name=name, parent=spec["bones"][parent] if parent >= 0 else None)
                for name, parent, rest in zip(spec["bones"], spec["parents"], spec["rest"])]
        derived = skeleton_anatomy(rows, "human")
        out[armature] = dict(derived, policy=anatomy.get("policy", "human")) if derived.get("joints") else anatomy
    return out


def anatomical_measure(joint, parent_rotation, child_rotation):
    """One joint's angles, degrees, from its two bones' armature-space rotations.

    The child's segment (`child_vector`, in the child bone's rest frame) is
    carried into the parent bone's frame, where the import stored the
    reference directions. A `hinge` (elbow, knee) reports `flexion`, the signed
    angle about the hinge axis from the straight limb toward the side the
    joint folds to -- negative is the limb bending backwards -- and
    `off_plane`, how far out of the hinge plane it swung. A `cone` (wrist,
    ankle, neck) reports `deviation` from its rest direction.
    """
    d = (parent_rotation.T @ child_rotation) @ np.asarray(joint["child_vector"], dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    if joint["kind"] == "hinge":
        n = np.asarray(joint["axis"], dtype=float)
        zero = np.asarray(joint["zero"], dtype=float)
        fold = np.asarray(joint["fold"], dtype=float)
        off = math.degrees(math.asin(float(np.clip(d @ n, -1, 1))))
        p = d - (d @ n) * n
        flexion = math.degrees(math.atan2(float(p @ fold), float(p @ zero)))
        return {"flexion": flexion, "off_plane": off}
    rest = np.asarray(joint["rest_direction"], dtype=float)
    return {"deviation": math.degrees(math.acos(float(np.clip(d @ rest, -1, 1))))}


#: Whole-body limits for an imported person, degrees: a proposal, not yet
#: certified. Clinical ranges of motion with about 15 degrees for the rig and
#: the measurement (rest offsets on the library's rigs reach 13 to 17), widened
#: only where inspected authored motion couples joints the clinical tables
#: measure one at a time. A roll is `[(elevation, limit), ...]`, linear between,
#: held beyond the ends, and counts only while its hinge is bent `ROLL_RANGE`.
BODY_LIMITS = {
    # internal 70 (hands behind the back read -116 hanging), external 90; raised, an arm turns further outward
    # and less inward (authored clips: outward 73 hanging, 123 at shoulder height; inward -70 hanging, -20 overhead)
    "shoulder_roll": ([(30, -110), (90, -100), (150, -75)], [(60, 100), (120, 140)]),
    # 45 each way; sitting cross-legged reads 106 with the thighs 60 degrees up
    "hip_rot": ([(0, -65)], [(30, 75), (60, 115)]),
    "elbow_bend": (None, 155), "knee_bend": (None, 160),        # flexion 150; the knee's heel to buttock
    "forearm_rot": (-105, 110),         # pronation 80-85, supination 85-90, plus some 10 the wrist's carpals add
    "wrist": {"flex": 90, "ext": 80, "radial": 30, "ulnar": 50},     # flexion 80, extension 70, radial 20, ulnar 35-40
    "ankle_flex": (-80, 45),            # plantarflexion 50 (kneeling on the instep 70), dorsiflexion 20 (40-45 squatting)
    "head_axial": (-95, 95), "head_tilt": (-60, 60), "head_nod": (-85, 70),   # rotation 80; side 45; back 70, forward 45-60
}
ROLL_LIMITS = {"shoulder_roll": ("elbow_bend", "shoulder_elev", "shoulder", "roll"),
               "hip_rot": ("knee_bend", "hip_elev", "hip", "rotation")}
#: A hinge straighter than this reads its roll unreliably, and one more folded
#: than the upper end swings it far for a few degrees across; below the lower
#: end, a hinge bent backwards past HYPEREXTENSION_DEG is read from its flexion.
ROLL_RANGE = (30.0, 140.0)
HYPEREXTENSION_DEG = 15.0
#: Within ROLL_RANGE, a hinge whose bend sits more than this far round from the way it folds is bent backwards
#: (`*_off_fold`). Authored clips on 130 library rigs reach 66 degrees and agent-authored film motion 120; a
#: hinge bent straight backwards reads 180.
AGAINST_HINGE_DEG = 150.0


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _twist_about(m, axis):
    """Signed twist (degrees) of rotation matrix `m` about unit `axis`, swing removed."""
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    p = float(np.dot([x, y, z], axis))
    return math.degrees(2 * math.atan2(p, w)) if abs(p) > 1e-12 or w else 0.0


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _signed_angle(a, b, axis):
    return math.degrees(math.atan2(float(axis @ np.cross(a, b)), float(a @ b)))


def _cardan(change, rest_child, basis):
    """A joint change split the anatomical way: flexion about the body's left-right line
    (+ backward), side bend (+ toward the figure's right), axial rotation (+ toward its left).
    Past 70 degrees of side bend the split degenerates and only the side bend is kept."""
    rest_child = np.asarray(rest_child, dtype=float).reshape(3, 3)
    B = np.asarray(basis, dtype=float).reshape(3, 3)
    M = B.T @ (rest_child @ change @ rest_child.T) @ B
    side = math.degrees(math.asin(float(np.clip(M[0, 2], -1, 1))))
    if abs(M[0, 2]) > 0.94:
        return None, side, None
    return (math.degrees(math.atan2(-M[1, 2], M[2, 2])), side, math.degrees(math.atan2(-M[0, 1], M[0, 0])))


def body_measures(anatomy, rotation):
    """The whole-body measures of one pose, degrees, from `native_rig.anatomy.body`.

    `rotation` maps a bone to its evaluated armature-space orientation. None of the measures
    depends on how a rig rolls its bones (checked against blind review on 130 library rigs);
    see `rig_anatomy._body_references`.
    Keys end in `_L` / `_R` per side; the head's and spine's have none.
    """
    out = {}
    body = anatomy.get("body") or {}
    hinges = {j["name"]: j for j in anatomy.get("joints", ()) if j["kind"] == "hinge"}
    def posed(bone, vec):
        return rotation[bone] @ np.asarray(vec, dtype=float)
    def have(*bones):
        return all(b in rotation for b in bones)
    for key in ("shoulder_L", "shoulder_R", "hip_L", "hip_R"):
        e = body.get(key)
        if not e or not have(e["upper"], e["lower"], e["grandparent"]):
            continue
        side, hinge = key[-1], hinges[e["hinge"]]
        name = "elbow" if key.startswith("shoulder") else "knee"
        Rp, Rc = rotation[e["upper"]], rotation[e["lower"]]
        d = _unit((Rp.T @ Rc) @ np.asarray(hinge["child_vector"], dtype=float))
        zero, fold = np.asarray(hinge["zero"], dtype=float), np.asarray(hinge["fold"], dtype=float)
        bend = math.degrees(math.acos(float(np.clip(d @ zero, -1, 1))))
        out[f"{name}_bend_{side}"] = bend
        out[f"{name}_flexion_{side}"] = anatomical_measure(hinge, Rp, Rc)["flexion"]
        change = np.asarray(e["grand_rest_relative"], dtype=float).reshape(3, 3).T @ (rotation[e["grandparent"]].T @ Rp)
        if bend >= 10:
            p = d - (d @ zero) * zero
            phi = math.degrees(math.atan2(float(p @ np.cross(zero, fold)), float(p @ fold)))
            # How far round the hinge the bend sits from the way it folds: 0 bending normally, 180 straight
            # backwards. The roll below adds it to the upper bone's own turn and wraps, so a limb bent backwards
            # whose upper bone is turned a quarter round reads like a natural one; this keeps it apart.
            out[f"{name}_off_fold_{side}"] = abs(phi)
            mirror = 1.0 if side == "R" else -1.0
            if name == "elbow":
                out[f"shoulder_roll_{side}"] = mirror * _wrap(
                    _twist_about(change, _unit(e["upper_segment"])) + phi)
            elif "cardan_basis" in e:
                turn = _cardan(change, e["cardan_rest"], e["cardan_basis"])[2]
                if turn is not None:
                    out[f"hip_rot_{side}"] = mirror * _wrap(turn + phi)
        if "elevation_ref" in e and have(e["elevation_ref"]):
            up = _unit(posed(e["upper"], e["upper_segment"]))
            down = _unit(posed(e["elevation_ref"], e["elevation_down"]))
            if up is not None and down is not None:
                key_ = "shoulder_elev" if name == "elbow" else "hip_elev"
                out[f"{key_}_{side}"] = math.degrees(math.acos(float(np.clip(up @ down, -1, 1))))
    for side in "LR":
        e = body.get(f"wrist_{side}")
        if e and have(e["upper"], e["forearm"], e["hand"]):
            u = _unit(posed(e["upper"], e["upper_segment"]))
            f = _unit(posed(e["forearm"], e["forearm_segment"]))
            length = _unit(posed(e["hand"], e["hand_length"]))
            radial = posed(e["hand"], e["radial"])
            if u is not None and f is not None and length is not None:
                mirror = e["mirror"]
                bend = math.degrees(math.acos(float(np.clip(u @ f, -1, 1))))
                j = hinges[e["hinge"]]
                hinge_axis = rotation[e["upper"]] @ np.cross(np.asarray(j["zero"], float), np.asarray(j["fold"], float))
                h = np.cross(u, f) if bend >= 30 else hinge_axis
                # An elbow bent onto the back half of its hinge has no forearm rotation to read: the plane it is
                # bent in turns over, which read the forearm half a turn round (an elbow bent straight backwards
                # with no twist read 180). The elbow's own findings cover it.
                if h @ hinge_axis < 0:
                    h = None
                h = None if h is None else _unit(h)
                if h is not None:
                    a = _unit(radial - (radial @ f) * f)
                    if a is not None:
                        out[f"forearm_rot_{side}"] = mirror * _signed_angle(-np.cross(f, h), a, f)
                r = _unit(radial - (radial @ length) * length)
                if r is not None:
                    normal = -mirror * np.cross(length, r)            # out of the palm
                    cn, cr, cl = -float(f @ normal), -float(f @ r), float(f @ length)
                    lateral = math.hypot(cn, cr)
                    beta = math.degrees(math.atan2(lateral, cl))
                    out[f"wrist_flex_{side}"] = beta * cn / lateral if lateral > 1e-9 else 0.0
                    out[f"wrist_tilt_{side}"] = beta * cr / lateral if lateral > 1e-9 else 0.0
        e = body.get(f"ankle_{side}")
        if e and have(e["thigh"], e["shin"], e["foot"]):
            n = _unit(rotation[e["thigh"]] @ np.asarray(e["hinge_normal"], dtype=float))
            shin, foot = _unit(posed(e["shin"], e["shin_segment"])), _unit(posed(e["foot"], e["foot_segment"]))
            if n is not None and shin is not None and foot is not None:
                out[f"ankle_flex_{side}"] = -_wrap(_signed_angle(-shin, foot, n) - e["rest_angle"])
    e = body.get("head")
    if e and have(e["head"], e["base"]):
        change = np.asarray(e["rest_relative"], dtype=float).reshape(3, 3).T @ (rotation[e["base"]].T @ rotation[e["head"]])
        back, tilt, turn = _cardan(change, e["cardan_rest"], e["cardan_basis"])
        out.update({k: v for k, v in (("head_axial", turn), ("head_tilt", tilt),
                                      ("head_nod", None if back is None else -back)) if v is not None})
    e = body.get("spine")
    if e and have(e["chest"], e["hips"]):
        change = np.asarray(e["rest_relative"], dtype=float).reshape(3, 3).T @ (rotation[e["hips"]].T @ rotation[e["chest"]])
        turn = _cardan(change, e["cardan_rest"], e["cardan_basis"])[2]
        if turn is not None:
            out["spine_axial"] = turn
    return out


def _limit_at(curve, x):
    if not isinstance(curve, list):
        return curve
    if x <= curve[0][0]:
        return curve[0][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def body_findings(measured):
    """`(joint, measure, value, [lo, hi], reason)` for every `BODY_LIMITS` range a pose leaves,
    named per joint."""
    out = []
    def check(joint, measure, value, lo, hi, below, above):
        if value is None:
            return
        if lo is not None and value < lo:
            out.append((joint, measure, value, [lo, hi], below))
        elif hi is not None and value > hi:
            out.append((joint, measure, value, [lo, hi], above))
    for side in "LR":
        g = lambda k: measured.get(f"{k}_{side}")
        for roll, (bend_key, elev_key, joint, measure) in ROLL_LIMITS.items():
            bend, elev, value = g(bend_key), g(elev_key), g(roll)
            if value is None or bend is None or elev is None or not ROLL_RANGE[0] <= bend <= ROLL_RANGE[1]:
                continue
            lo, hi = BODY_LIMITS[roll]
            check(f"{joint}_{side}", measure, value, _limit_at(lo, elev), _limit_at(hi, elev),
                  "rolled inward past its range", "rolled outward past its range")
        for hinge in ("elbow", "knee"):
            bend = g(f"{hinge}_bend")
            check(f"{hinge}_{side}", "bend", bend, None, BODY_LIMITS[f"{hinge}_bend"][1], "", "over-flexed")
            flexion = g(f"{hinge}_flexion")
            if bend is not None and flexion is not None and bend < ROLL_RANGE[0]:
                check(f"{hinge}_{side}", "flexion", flexion, -HYPEREXTENSION_DEG, None, "bent backwards", "")
            if bend is not None and ROLL_RANGE[0] <= bend <= ROLL_RANGE[1]:
                check(f"{hinge}_{side}", "bend direction", g(f"{hinge}_off_fold"), None, AGAINST_HINGE_DEG,
                      "", "bent backwards")
        check(f"forearm_{side}", "rotation", g("forearm_rot"), *BODY_LIMITS["forearm_rot"],
              "pronated past its range", "supinated past its range")
        flex, tilt = g("wrist_flex"), g("wrist_tilt")
        if flex is not None and tilt is not None:
            w = BODY_LIMITS["wrist"]
            a, b = (w["flex"] if flex >= 0 else w["ext"]), (w["radial"] if tilt >= 0 else w["ulnar"])
            if (flex / a) ** 2 + (tilt / b) ** 2 > 1:
                if abs(tilt) >= b:
                    check(f"wrist_{side}", "tilt", tilt, -w["ulnar"], w["radial"],
                          "bent toward the little finger past its range", "bent toward the thumb past its range")
                else:
                    room = math.sqrt(1 - (tilt / b) ** 2)
                    check(f"wrist_{side}", "flexion", flex, -w["ext"] * room, w["flex"] * room,
                          "bent back past its range", "bent toward the palm past its range")
        check(f"ankle_{side}", "flexion", g("ankle_flex"), *BODY_LIMITS["ankle_flex"],
              "pointed past its range", "bent toward the shin past its range")
    check("head", "turn", measured.get("head_axial"), *BODY_LIMITS["head_axial"],
          "turned past its range", "turned past its range")
    check("head", "tilt", measured.get("head_tilt"), *BODY_LIMITS["head_tilt"],
          "tilted past its range", "tilted past its range")
    check("head", "nod", measured.get("head_nod"), *BODY_LIMITS["head_nod"],
          "tipped back past its range", "nodded forward past its range")
    # The spine's twist is measured but not judged: it read 100 degrees on
    # one rig's running clip that shows none (a rig artifact), and reviewers could not see it.
    return out


#: Which whole-body measure stands in for each per-bone joint check; where the body
#: block cannot build it (a hand without finger knuckles, a neck without a parent),
#: the per-bone check stays. The per-bone elbow and knee checks are replaced
#: outright: read in the upper bone's own frame, they call a rolled arm "bent
#: backwards".
_SUPERSEDED_BY = {"elbow": "shoulder", "knee": "hip", "wrist": "wrist", "ankle": "ankle", "head": "head"}


def bone_anatomy(frame, record, rigs, rig_anatomy):
    """Every declared bone limit exceeded at `frame`: one row per (armature, joint, measure).

    Per frame, like every row `measure` keeps, so the rows of a parallel
    check's chunks concatenate (`chunking.merge_rows`) and a re-check replaces
    exactly the frames it re-verified (`chunking.merge_scoped`, which keys rows
    by `frame`). `bone_anatomy_rows` groups them, in `aggregate`. A rig whose
    anatomy carries a `body` block (version 2) is judged by `body_findings`.
    """
    out = []
    rotations = record.get("bone_rotations") or {}
    for armature, anatomy in rig_anatomy.items():
        rows, spec = rotations.get(armature), (rigs or {}).get(armature)
        if rows is None or spec is None or len(rows) != len(spec["bones"]):
            continue
        slots = {name: k for k, name in enumerate(spec["bones"])}
        body = anatomy.get("body") or {}
        if body:
            mats = quaternion_matrices(rows)
            rotation = {name: mats[k] for name, k in slots.items()}
            for joint, measure, value, (lo, hi), reason in body_findings(body_measures(anatomy, rotation)):
                out.append({"frame": frame, "rig": armature, "joint": joint, "measure": measure,
                            "reason": reason, "value_deg": round(float(value), 3), "limits_deg": [lo, hi]})
        for joint in anatomy["joints"]:
            if joint["parent"] not in slots or joint["child"] not in slots:
                continue
            base, _, side = joint["name"].partition("_")
            if f"{_SUPERSEDED_BY.get(base, '?')}{'_' + side if side else ''}" in body:
                continue
            parent, child = quaternion_matrices([rows[slots[joint["parent"]]],
                                                 rows[slots[joint["child"]]]])
            measured = anatomical_measure(joint, parent, child)
            checks = []
            if joint["kind"] == "hinge":
                lo, hi = joint["flexion_deg"]
                checks.append(("flexion", measured["flexion"], [lo, hi],
                               "bent backwards" if measured["flexion"] < lo else "over-flexed"))
                checks.append(("off_plane", abs(measured["off_plane"]),
                               [0, joint["off_plane_deg"]], "swung out of its hinge plane"))
            else:
                checks.append(("deviation", measured["deviation"], [0, joint["max_deg"]],
                               "turned past its range"))
            for measure, value, (lo, hi), reason in checks:
                if lo - 1e-6 <= value <= hi + 1e-6:
                    continue
                out.append({"frame": frame, "rig": armature, "joint": joint["name"],
                            "measure": measure, "reason": reason,
                            "value_deg": round(float(value), 3), "limits_deg": [lo, hi]})
    return out


def bone_anatomy_rows(rows, fps):
    """`bone_anatomy`'s per-frame rows as evidence: one row per (armature, joint, measure,
    reason), its frames as runs and its worst excess, most frames first."""
    groups = {}
    for row in rows:
        lo, hi = row["limits_deg"]
        value = row["value_deg"]
        # A range can be open at one end (a bend has only a most, a hyperextension only a least).
        excess = lo - value if lo is not None and value < lo else value - (hi if hi is not None else value)
        key = (row["rig"], row["joint"], row["measure"], row["reason"])
        held = groups.setdefault(key, {"frames": [], "worst": value, "excess": excess,
                                       "worst_frame": row["frame"], "limits": [lo, hi]})
        held["frames"].append(row["frame"])
        if excess > held["excess"]:
            held.update(worst=value, excess=excess, worst_frame=row["frame"])
    out = []
    for (armature, joint, measure, reason), entry in groups.items():
        frames = sorted(set(entry["frames"]))
        out.append({"rig": armature, "joint": joint, "measure": measure,
                    "reason": reason, "frame_count": len(frames),
                    "frames": frame_runs(frames),
                    "window": [round(frames[0] / fps, 4), round(frames[-1] / fps, 4)],
                    "worst_deg": round(entry["worst"], 1), "worst_frame": entry["worst_frame"],
                    "limits_deg": entry["limits"]})
    out.sort(key=lambda r: (-r["frame_count"], r["rig"], r["joint"], r["measure"]))
    return out
