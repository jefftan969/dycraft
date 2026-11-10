"""Query module: stateless functions over a finalized object IR dict.

All functions take an IR dict (as returned by Object.finalize()) and reference
parts/joints by name. Names are unique within their namespace, so name lookup
is unambiguous.

Forward kinematics walks the joint tree from `ir["root"]` with each joint
evaluated at its `default` value (rest pose). All angles in the IR are in
radians; query outputs return Euler XYZ degrees (never quaternions, never
radians). The IR itself is unchanged.
"""

import math
from typing import NamedTuple

from . import _rotation as _rot
from . import _anchors as _anchors_mod
from . import _primitives as _prim


_PENETRATION_EPS = 1e-6
_AXIS_EPS = 1e-6
_TIEBREAK_EPS = 1e-9
_RAD_TO_DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------
# Penetration report record (object + scene layer share this shape)
# ---------------------------------------------------------------------------

class PenetrationPair(NamedTuple):
    """Single intersecting-pair record produced by the SAT narrow phase.

    Field semantics match the GJK+EPA upgrade so the verifier API is
    forward-compatible:

      * a, b              — entity names (parts for object layer, instances
                            for scene layer), ordered lexicographically.
      * depth             — penetration depth in meters (>= EPS_REPORT).
      * contact_normal    — unit world-frame vector pointing from a to b;
                            translating b by +depth*n separates the pair.
      * contact_point     — world-frame point inside the overlap volume.
      * signed_distance   — -depth for reported pairs (always negative;
                            >=0 reserved for v2 near-miss reporting).
      * a_frame, b_frame  — ((tx, ty, tz), (rx_deg, ry_deg, rz_deg))
                            world-frame pose of each entity at rest.
      * a_bbox, b_bbox    — ((lo, hi)) world-axis-aligned AABBs, kept as
                            diagnostic and broad-phase remnant.

    The trailing four fields are filled in only when the exact-geometry
    adjudicator (``verify/_exact_adjudicator.py``) has re-checked this pair
    against real B-rep / analytic geometry and confirmed it. They are additive:
    ``depth`` / ``contact_normal`` / ``contact_point`` always remain the SAT
    values, because an exact query yields a separation and a shared volume but
    no minimum translation vector.

    **The SAT fields do not reach a verifier report:**
    ``verify/object_verify._pair_evidence`` emits the exact fields only. They remain on this record, and on the ``pair_report``
    probe, as the low-level box diagnostics the SDK prompt already labels them
    — a caller that asks for them by name gets what it asked for, which is a
    different thing from a verifier handing them to an author as a finding.

      * adjudicated       — True once exact geometry confirmed the overlap.
                            Absent/False means SAT's (conservative) verdict
                            stands, either because adjudication was unavailable
                            or because the pair had no exact geometry to
                            query: an `imported` part on the B-rep backend,
                            which is the object layer's, or an asset whose
                            collision mesh is not staged.
      * exact_distance    — exact separation in meters, **signed**: positive is
                            the gap, negative is the measured overlap depth.
                            The B-rep backend has no signed distance to give and
                            reports 0.0 when overlapping; the mesh narrow phase
                            measures the depth on the triangles, so above the
                            object layer this is the one number that runs
                            continuously through contact.
      * exact_volume      — shared volume in m^3.
      * closest_points    — ((ax,ay,az), (bx,by,bz)) closest surface point on
                            each body, the true contact location.
      * separating_shift  — (dx, dy, dz) meters: a translation of ``b`` that
                            was TESTED to end the overlap, found by bisecting
                            the exact boolean rather than inferred from a box.
                            None when no single-axis translation separated the
                            pair inside the search bound, which is a real
                            answer: the fix is then a reshape, not a nudge.
                            Measured on a recorded corpus, that is the
                            outcome for 63% of adjudicated pairs — an assembly
                            interference often has no small translation out of
                            it in any direction, which is why the absence is
                            reported rather than back-filled from a box.

    ``a_sub`` / ``b_sub`` are scene-layer only. There, ``a``/``b`` are instance
    names and the record aggregates the deepest of ``|parts_a| x |parts_b|``
    part-pairs; these name that part-pair. Without them the adjudicator has no
    way to know which geometry to place, since an instance is not a single body.
    """

    a: str
    b: str
    depth: float
    contact_normal: tuple
    contact_point: tuple
    signed_distance: float
    a_frame: tuple
    b_frame: tuple
    a_bbox: tuple
    b_bbox: tuple
    witnessing_pose: "dict | None" = None
    adjudicated: bool = False
    exact_distance: "float | None" = None
    exact_volume: "float | None" = None
    closest_points: "tuple | None" = None
    separating_shift: "tuple | None" = None
    a_sub: "str | None" = None
    b_sub: "str | None" = None


class SATResult(NamedTuple):
    """Total SAT result for OBB-OBB tests.


      * collided         — True iff signed_distance < 0 (modulo SAT eps).
      * signed_distance  — +margin if separated, -depth if penetrating.
      * axis             — unit world-frame vector. MTV direction (a→b) if
                           collided; max-separating-axis if separated.
      * contact_point    — world-frame point inside the overlap volume,
                           or None when separated.
    """

    collided: bool
    signed_distance: float
    axis: tuple
    contact_point: "tuple | None"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _index_parts(ir):
    return {p["name"]: p for p in ir["parts"]}


def _index_joints_by_parent(ir):
    out: dict = {}
    for j in ir["joints"]:
        out.setdefault(j["parent"], []).append(j)
    return out


def _se3_from_anchor(anchor: dict):
    """Anchor dict -> (translation_tuple, quat_wxyz_tuple)."""
    t = anchor["translation"]
    q = anchor["rotation"]
    return ((t[0], t[1], t[2]), (q[0], q[1], q[2], q[3]))


# ---------------------------------------------------------------------------
# Public list-style queries
# ---------------------------------------------------------------------------

def list_parts(ir) -> list:
    return [p["name"] for p in ir["parts"]]


def list_joints(ir) -> list:
    return [j["name"] for j in ir["joints"]]


def _anchor_with_euler(anchor: dict) -> dict:
    """Convert an IR anchor dict (translation + quat) into the LLM-facing
    shape (translation + Euler XYZ degrees)."""
    q = anchor["rotation"]
    rx, ry, rz = _rot.quat_to_euler_xyz_deg((q[0], q[1], q[2], q[3]))
    return {
        "translation": list(anchor["translation"]),
        "rotation_euler_deg": [rx, ry, rz],
    }


def joint_table(ir) -> list:
    """Return a list of joint records (deep-copied) in creation order.

    Rotations are returned as intrinsic Euler XYZ in degrees under the key
    "rotation_euler_deg" (the IR's quaternions are never exposed). Revolute
    `limits` and `default` are returned in degrees; prismatic in meters, with
    `unit` naming which — `"deg"` or `"m"`, the same key and the same two tokens
    `scene_joint_limits` reports, so one fact has one spelling on both layers.

    Joint `axis` is reported in the parent-anchor's local frame (its
    authored frame); a non-identity rotation on parent_anchor implies
    the axis is rotated when expressed in parent-part-local. Use
    `joint_axis_world` for the resolved world-frame axis at a pose.
    """
    out = []
    for j in ir["joints"]:
        row = {
            "name": j["name"],
            "type": j["type"],
            "parent": j["parent"],
            "child": j["child"],
            "parent_anchor": _anchor_with_euler(j["parent_anchor"]),
            "child_anchor": _anchor_with_euler(j["child_anchor"]),
        }
        if j["type"] == "revolute":
            row["axis"] = list(j["axis"])
            row["limits"] = [j["limits"][0] * _RAD_TO_DEG, j["limits"][1] * _RAD_TO_DEG]
            row["default"] = j["default"] * _RAD_TO_DEG
            row["unit"] = "deg"
        elif j["type"] == "prismatic":
            row["axis"] = list(j["axis"])
            row["limits"] = list(j["limits"])
            row["default"] = j["default"]
            row["unit"] = "m"
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Local geometry
# ---------------------------------------------------------------------------

def _part_record(ir, part_name: str) -> dict:
    for p in ir.get("parts", ()):
        if p["name"] == part_name:
            return p
    raise KeyError(part_name)


def part_shape(ir, part_name: str):
    """Return an asset part's derived shape fact table, or None.

    The table is what lets a caller reason about geometry the bounding box hides:
    exact ``volume`` / ``area`` / ``center`` (the true centroid, not the AABB
    midpoint), ``inertia_diag``, ``fill_ratio`` (volume over bounding-box volume
    — a shelled carcass reports ~0.13, telling you 87% of its AABB is air), the
    ``face_census``, ``topo``, and the ``cylinders`` inventory, where a face
    marked ``concave`` is a bore.

    It also carries the staged render mesh's ``qc`` (``bounds_volume``,
    ``is_manifold``, ``winding_consistent``, ``n_shells``, ``n_boundary_edges``,
    ``volume_m3``, ``bbox``; None if the QC pass itself failed) — a fact about
    the geometry, and the only way a caller that sees just the catalog IR, like
    the scene agent through ``objects[…].part_shapes()``, can ask whether a part
    encloses a volume. ``bounds_volume`` is that question; ``is_manifold`` is the
    stronger one no consumer here asks, reported for algorithms that would.

    What it deliberately does NOT carry is what the *compile* did: which
    duplicate vertices were welded, which stage raised, whether the render mesh
    was coarsened to stay under the triangle budget. Those describe one run, not
    the part, and the IR is content-hashed whole — so they would make a
    diagnostic's prose part of the object's identity. They travel in the compile
    report's ``geometry_notes`` instead.

    None for a `box`, which is fully described by its params.

    An ``imported`` part has this table too — integrated from its triangles at
    import time rather than measured by the CAD kernel — so the same questions
    are answerable about it. The optional keys differ: it carries no
    ``face_census``/``topo``/``cylinders`` (there are no B-rep faces to census),
    and a caller reads them with ``.get``.
    """
    part = _part_record(ir, part_name)
    if part.get("primitive") not in _prim.ASSET_PRIMITIVES:
        return None
    return (part.get("params") or {}).get("shape")


def object_source_path(ir):
    """Run-relative path to this object's authored `object.py`, or None.

    One program builds every part, so this is an **object-level** fact. None
    for an object with no authored parts.
    """
    src = ir.get("source")
    return src if isinstance(src, str) and src else None


def part_color(ir, part_name: str):
    """Return the part's RGB color as a 3-tuple of ints in [0, 255].

    Reads from the part's `material` field (see sdk._material). Returns
    None if the part has no material.

    **A `shader` material has no authored color, and this still answers with
    one.** `material_from_ir` deliberately leaves the scalar absent there — the
    graph decides the surface — but the question this function is asked is "what
    color is this part", and it is asked by the probe API the object agent
    reads and by the scene layer's palette reasoning. The module default is the
    honest answer, because it is exactly what every consumer that cannot
    evaluate a shader will draw. `part_material(...).color` still reports the
    absence for a caller that needs to know.
    """
    mat = part_material(ir, part_name)
    if mat is None:
        return None
    if mat.color is None:
        from ._material import DEFAULT_PART_COLOR
        return DEFAULT_PART_COLOR
    return mat.color


def part_material(ir, part_name: str):
    """Return the part's Material.

    Returns None if the part has no material attached. **Reports the scalars
    exactly as authored, absences included**: a `shader` material leaves
    `color`, `metallic`, and `roughness` as None because the graph owns them.
    That is deliberate — `part_color` is the accessor that substitutes a
    default, and a caller that needs to know a surface is procedural asks here.
    """
    from ._material import material_from_ir, DEFAULT_PART_COLOR
    parts = _index_parts(ir)
    if part_name not in parts:
        raise KeyError(part_name)
    d = parts[part_name].get("material")
    if d is None:
        return None
    return material_from_ir(d, default_color=DEFAULT_PART_COLOR)


def local_bbox(ir, part_name: str, parts=None):
    # `parts` lets a caller that already needs the part dict for something else
    # pass its index in rather than pay a second `_index_parts` walk. This is on
    # the animation layer's hottest pure-Python path (see
    # `_scene_kernel.instance_part_world_aabbs`), so an extra dict build per
    # part is an extra O(P) per part.
    if parts is None:
        parts = _index_parts(ir)
    if part_name not in parts:
        raise KeyError(part_name)
    p = parts[part_name]
    # Per-primitive AABB derivation lives in _primitives. A `box` is
    # origin-centered; an asset part (`solid` or `imported`) returns its stored
    # full AABB, which MAY be off-center. Downstream stays correct for either:
    # obb_of_part places the collider center at t + R·c (c = AABB midpoint) and
    # the 27-name anchor vocabulary reads the true AABB min/max/midpoint — both
    # reduce to the origin-centered case when c = 0.
    return _prim.local_aabb(p["primitive"], p["params"])


def collide_margin(part: dict) -> float:
    """How far this part's collision geometry escapes its stored AABB, in meters.

    `0.0` for everything except an `imported` part converted since the key
    existed. An asset ships two meshes, the stored box is the *visual* one's so
    that anchors land on the visible surface, and the narrow phase measures the
    other — which on the committed Panda pokes outside that box on 9 of 11
    parts, worst 5.554 mm. The broad phase drops a pair whose boxes separate by
    more than its tolerance on the argument that the box contains the part
    (`_probes._gap_candidates`), so without this the argument does
    not hold and a touching pair can go unexamined.

    Only `import_urdf` sets it, and that is measured rather than assumed: a
    glTF character's collision body is its visual mesh or a hull of it, so it
    cannot leave that mesh's box. On the committed 146-part `soldier` the worst
    escape is 4.8e-09 m — the OBJ write/read round-trip, not geometry.
    """
    params = part.get("params") or {}
    try:
        return max(0.0, float(params.get("collide_escape_m") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _inflate_local_box(box, inflate, margin: float = 0.0):
    """Grow a part's local AABB by a displacement bound `(lo3, hi3)`.

    `inflate` is the componentwise
    min/max of the rig's control displacements at one frame, which every vertex's
    own displacement is inside because the LBS weights are a partition of unity.
    So `lo + inflate.lo` / `hi + inflate.hi` is a box that provably contains the
    deformed geometry, where the rest box does not.

    Applied here rather than inside `local_bbox`, deliberately. `local_bbox` also
    feeds the 27-name anchor vocabulary (`sdk/_anchors.py`), and an anchor is a
    *landmark*, not a bound — `top_center` of an inflated box is not where
    anything is. Keeping the inflation in the collision/bound producers means the
    anchor path provably cannot pick it up. That anchors go stale under
    deformation is a known, larger question.

    `margin` is the isotropic version of the same idea, for geometry that
    escapes its box at rest rather than because a rig moved it — see
    `collide_margin`. It rides here for the same reason: a bound may grow, a
    landmark may not.
    """
    if inflate is None and not margin:
        return box
    (lo, hi) = box
    dlo, dhi = inflate if inflate is not None else ((0.0,) * 3, (0.0,) * 3)
    m = float(margin or 0.0)
    return ((lo[0] + dlo[0] - m, lo[1] + dlo[1] - m, lo[2] + dlo[2] - m),
            (hi[0] + dhi[0] + m, hi[1] + dhi[1] + m, hi[2] + dhi[2] + m))


def _inflate_for(deform_bounds, part_name: str):
    """This part's displacement bound out of a `{part: (lo3, hi3)}` map."""
    if not deform_bounds:
        return None
    return deform_bounds.get(part_name)


# ---------------------------------------------------------------------------
# Forward kinematics (internal)
# ---------------------------------------------------------------------------


def _mount_delta_for(joint: dict, mount_deltas):
    """The mount delta `D` for this joint, or None.

    `D` is the motion of the *mount frame* expressed in the parent part's local
    frame — what the parent's deformation did to the point the joint hangs off.
    """
    if not mount_deltas:
        return None
    return mount_deltas.get(joint["name"])


def _apply_mount(rel, joint: dict, mount_deltas):
    """Left-compose the mount delta onto a child-in-parent transform.

    LEFT, and before `parent_anchor`, so that the articulation still happens
    about the (now moved) mount frame. That placement is what keeps a joint
    value meaning what it means without deformation: the child's
    orientation relative to the parent becomes `D.R · R(q) · rest.R`, so the
    rotation from q=0 to q is `D.R · R(q) · D.R⁻¹` — the same *angle* q, about
    an axis that rides the deformation. Joint limits, `joint_at`, `moved_by`
    and every event predicate therefore keep their meaning.

    Applying `D` after `R(q)` instead would make q stop being the observable
    relative rotation, which would invalidate that whole vocabulary.
    """
    D = _mount_delta_for(joint, mount_deltas)
    if D is None:
        return rel
    return _rot.se3_compose(D, rel)


def _child_in_parent(joint: dict, poses=None, mount_deltas=None):
    """Compute child part frame expressed in parent part frame at rest (q=default).

    Anchor semantics:
      * `parent_anchor` and `child_anchor` are full SE3s in their parts'
        local frames; their rotations come from any non-identity
        rotation offset baked into the AnchorRef at build time.
      * Joint records carry no separate `offset` field — all placement
        information lives in the two anchor SE3s.
      * For revolute / prismatic, the pivot (rotation origin / line of
        action through axis) sits at `parent_anchor.translation`. The
        joint's `axis` is in the parent_anchor's local frame, so a
        non-identity rotation on parent_anchor rotates the axis with it.

    Rest-pose composition:
        rest        = parent_anchor ∘ inverse(child_anchor)
        axis_parent = parent_anchor.R · axis_raw         (axis_raw in pa-local)

    `mount_deltas` is `{joint_name: SE3}` and is how a deforming parent part
    moves the frames of its children — see `_apply_mount`. `None` (the case for
    every object- and scene-layer caller) leaves the composition exactly as it
    was.
    """
    parent_anchor = _se3_from_anchor(joint["parent_anchor"])
    child_anchor = _se3_from_anchor(joint["child_anchor"])
    inv_child_anchor = _rot.se3_inverse(child_anchor)
    rest = _rot.se3_compose(parent_anchor, inv_child_anchor)

    jtype = joint["type"]
    if jtype == "fixed":
        return _apply_mount(rest, joint, mount_deltas)

    # Revolute / prismatic: `axis` is stored in PARENT-ANCHOR-local frame;
    # rotate it through parent_anchor.R to get the axis in parent-part-
    # local frame, then apply revolute / prismatic articulation about the
    # pivot at parent_anchor.translation.
    axis_raw = joint["axis"]
    norm = (axis_raw[0] ** 2 + axis_raw[1] ** 2 + axis_raw[2] ** 2) ** 0.5
    axis_pa_unit = (axis_raw[0] / norm, axis_raw[1] / norm, axis_raw[2] / norm)
    axis_in_parent = _rot.quat_rotate_vec(parent_anchor[1], axis_pa_unit)

    # Pose override takes priority over the authored default. Pose keys
    # that don't match this joint are silently ignored.
    if poses is not None and joint["name"] in poses:
        q = float(poses[joint["name"]])
    else:
        q = joint["default"]

    if jtype == "revolute":
        # Pivot at parent_anchor.translation; rotation about axis_in_parent.
        origin = parent_anchor[0]
        rq = _rot.quat_from_axis_angle_rad(axis_in_parent, q)
        rotated_origin = _rot.quat_rotate_vec(rq, origin)
        t_rot_about_origin = (
            (origin[0] - rotated_origin[0],
             origin[1] - rotated_origin[1],
             origin[2] - rotated_origin[2]),
            rq,
        )
        return _apply_mount(_rot.se3_compose(t_rot_about_origin, rest),
                            joint, mount_deltas)

    if jtype == "prismatic":
        delta = (q * axis_in_parent[0], q * axis_in_parent[1],
                 q * axis_in_parent[2])
        t_translate = (delta, (1.0, 0.0, 0.0, 0.0))
        return _apply_mount(_rot.se3_compose(t_translate, rest),
                            joint, mount_deltas)

    raise ValueError(f"unknown joint type {jtype!r}")


class MountDeltaUnknownJoint(ValueError):
    """`_compute_frames` was handed mount deltas for a joint this object lacks.

    A plumbing bug, not an authoring error — hence a plain `ValueError` like
    `AnchorProvenanceMismatch` rather than a `BuildError` the agent could repair
    from. It fires when a caller pairs one object's IR with another actor's
    mount deltas, or forwards scene-level `"<instance>.<joint>"` keys without
    unqualifying them first. Both would otherwise resolve child part frames off
    the undeformed mount and answer confidently with the wrong geometry.
    """

    def __init__(self, message, *, object_name=None, unknown=(), known=()):
        super().__init__(message)
        self.object_name = object_name
        self.unknown = list(unknown)
        self.known = list(known)


class AnchorProvenanceMismatch(ValueError):
    """A joint's embedded transform disagrees with its semantic anchor.

    `PartIR.anchors` is the formal definition; the joint's `parent_anchor` /
    `child_anchor` are compatibility snapshots the builder derives from it.
    They cannot diverge when the builder writes them, so this only fires on IR
    that was hand-edited, produced by a different tool, or corrupted.
    """


def validate_anchor_provenance(ir, *, tol: float = 1e-9) -> None:
    """Check every joint's embedded transform against its semantic anchor.

    **Call this ONCE when an ObjectIR enters the system, not per FK frame.**
    FK runs thousands of times across a CCD sweep; this is a whole-object
    structural check whose answer cannot change between frames, so putting it
    in `_compute_frames` would be pure waste.

    A no-op for objects without semantic anchors: their IR carries no
    `*_anchor_ref`, so the loop body never runs.
    """
    parts = {p["name"]: p for p in ir.get("parts") or ()}
    for joint in ir.get("joints") or ():
        for side, key in (("parent", "parent_anchor_ref"),
                          ("child", "child_anchor_ref")):
            prov = joint.get(key)
            if not prov:
                continue
            part_name = joint[side]
            part = parts.get(part_name)
            if part is None:
                raise AnchorProvenanceMismatch(
                    f"joint {joint.get('name')!r} references unknown part "
                    f"{part_name!r}")
            aabb = local_bbox(ir, part_name)
            base = _anchors_mod.resolve_base_frame(
                aabb, part.get("anchors"), prov["anchor"], part_name=part_name)
            offset = prov.get("offset")
            if offset:
                resolved = _rot.se3_compose(
                    base, (tuple(offset["translation"]),
                           tuple(offset["rotation"])))
            else:
                resolved = base
            embedded = _se3_from_anchor(joint[key.replace("_ref", "")])
            dt = max(abs(a - b) for a, b in zip(resolved[0], embedded[0]))
            dq = min(
                max(abs(a - b) for a, b in zip(resolved[1], embedded[1])),
                max(abs(a + b) for a, b in zip(resolved[1], embedded[1])),
            )
            if dt > tol or dq > tol:
                raise AnchorProvenanceMismatch(
                    f"joint {joint.get('name')!r} {side} anchor "
                    f"{prov['anchor']!r} on part {part_name!r} resolves to a "
                    f"different frame than the transform embedded in the "
                    f"joint (dt={dt:.3e}, dq={dq:.3e}). The part's anchor "
                    f"record is authoritative; the embedded transform is a "
                    f"snapshot and has drifted.")


def _compute_frames(ir, poses=None, mount_deltas=None) -> dict:
    """Return dict: part_name -> SE3 in object root frame.

    `poses` is an optional `{joint_name: q}` mapping. For each joint
    along the kinematic tree, the joint's articulation value is taken
    from `poses[joint_name]` if present, else its authored `default`.
    Keys not matching any joint name are silently ignored (consistent
    with the universal pose-tag rule).

    `mount_deltas` is an optional `{joint_name: SE3}` mapping carrying the
    deformation-induced motion of each joint's mount frame — see
    `_apply_mount`. **Unlike `poses`, an unrecognised key raises.** The
    asymmetry is deliberate: a `poses` dict is a scene-level tag sprayed at many
    objects, so unknown keys there are the normal case, whereas `mount_deltas`
    is resolved per-object by its caller, so a key
    that matches no joint means the caller paired the wrong object with the
    wrong actor. That is precisely the silent-wrong-geometry failure this
    parameter would otherwise introduce, and it costs nothing to make loud.

    **UNITS: this function takes IR-INTERNAL units — RADIANS for revolute,
    meters for prismatic.** The LLM-facing surface (`Object.assert_*`,
    `_probes.*`, joint `limits` / `default`) uses DEGREES for revolute, and
    converts at its own boundary via `_rotation.convert_poses_to_internal`.
    Both dicts have the identical shape `{joint_name: float}` with nothing to
    distinguish them, so passing a degree value here silently produces a wildly
    wrong pose rather than an error. Callers inside the kernel, the CCD sampler
    and the verifiers are already in radians; anything crossing from the agent
    surface must convert first.
    """
    by_parent = _index_joints_by_parent(ir)
    if mount_deltas:
        known = {j["name"] for j in ir.get("joints") or ()}
        unknown = sorted(set(mount_deltas) - known)
        if unknown:
            raise MountDeltaUnknownJoint(
                f"mount deltas name joints this object does not have: "
                f"{', '.join(unknown)}",
                object_name=ir.get("name"),
                unknown=unknown,
                known=sorted(known),
            )

    # The root part IS the object's local origin. There is no object-level
    # placement transform: a scene places an instance by aligning a chosen
    # anchor to a world pose, so nothing downstream needs one, and no builder,
    # importer or catalog writes a root pose. Object-level root placement is
    # deliberately NOT introduced here; if it is ever wanted it should arrive
    # with a writer, a schema field and tests, not as a silent default.
    frames: dict = {ir["root"]: _rot.IDENTITY_SE3}
    frontier = [ir["root"]]
    while frontier:
        next_frontier = []
        for parent_name in frontier:
            parent_frame = frames[parent_name]
            for joint in by_parent.get(parent_name, []):
                rel = _child_in_parent(joint, poses=poses,
                                       mount_deltas=mount_deltas)
                child_frame = _rot.se3_compose(parent_frame, rel)
                frames[joint["child"]] = child_frame
                next_frontier.append(joint["child"])
        frontier = next_frontier
    return frames


def _frame_to_surface(se3):
    """Internal SE3 ``((t),(q_wxyz))`` -> surface ``((t),(rx,ry,rz) deg)``."""
    t, q = se3
    rx, ry, rz = _rot.quat_to_euler_xyz_deg(q)
    return ((t[0], t[1], t[2]), (rx, ry, rz))


def frame_of(ir, part_name: str, poses=None):
    """Part frame in object root frame.

    Returns ((tx, ty, tz), (rx_deg, ry_deg, rz_deg)) — translation in meters,
    rotation as intrinsic Euler XYZ in degrees. Quaternions are never
    surfaced.

    `poses` is an optional `{joint_name: q}` mapping; defaults to the rest
    pose (all joints at their authored defaults). Passing it matters for
    evidence produced at a CCD witnessing pose: without it the reported
    frame describes the rest-pose geometry, which breaks the documented
    world-shift -> parent-local repair recipe.
    """
    frames = _compute_frames(ir, poses=poses)
    if part_name not in frames:
        raise KeyError(part_name)
    return _frame_to_surface(frames[part_name])


# ---------------------------------------------------------------------------
# Global (object-root-frame) AABBs
# ---------------------------------------------------------------------------

def _box_corners(local):
    (xmn, ymn, zmn), (xmx, ymx, zmx) = local
    return (
        (xmn, ymn, zmn), (xmx, ymn, zmn),
        (xmn, ymx, zmn), (xmx, ymx, zmn),
        (xmn, ymn, zmx), (xmx, ymn, zmx),
        (xmn, ymx, zmx), (xmx, ymx, zmx),
    )


def _aabb_of_transformed_corners(corners, se3):
    t, q = se3
    xs = []
    ys = []
    zs = []
    for c in corners:
        rx, ry, rz = _rot.quat_rotate_vec(q, c)
        xs.append(t[0] + rx)
        ys.append(t[1] + ry)
        zs.append(t[2] + rz)
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _global_aabb_of_part(ir, part_name: str, frames: dict, inflate=None):
    if part_name not in frames:
        raise KeyError(part_name)
    return _aabb_of_transformed_corners(
        _box_corners(_inflate_local_box(local_bbox(ir, part_name), inflate)),
        frames[part_name],
    )


def global_bbox(ir, part_name: str | None = None, poses=None,
                mount_deltas=None, deform_bounds=None):
    """World AABB of one part, or of the whole object.

    `deform_bounds` is `{part_name: (lo3, hi3)}` — see `_inflate_local_box`.
    Without it a deformed part is bounded by its *rest* box, which its moved
    vertices can leave, so every consumer of this answer inherits a bound that
    is not one.
    """
    frames = _compute_frames(ir, poses=poses, mount_deltas=mount_deltas)
    if part_name is not None:
        return _global_aabb_of_part(ir, part_name, frames,
                                    _inflate_for(deform_bounds, part_name))
    xs_min, ys_min, zs_min = [], [], []
    xs_max, ys_max, zs_max = [], [], []
    for p in ir["parts"]:
        (xmn, ymn, zmn), (xmx, ymx, zmx) = _global_aabb_of_part(
            ir, p["name"], frames, _inflate_for(deform_bounds, p["name"]))
        xs_min.append(xmn); ys_min.append(ymn); zs_min.append(zmn)
        xs_max.append(xmx); ys_max.append(ymx); zs_max.append(zmx)
    return ((min(xs_min), min(ys_min), min(zs_min)),
            (max(xs_max), max(ys_max), max(zs_max)))


def centroid(ir, part_name: str | None = None):
    (xmn, ymn, zmn), (xmx, ymx, zmx) = global_bbox(ir, part_name)
    return (0.5 * (xmn + xmx), 0.5 * (ymn + ymx), 0.5 * (zmn + zmx))


# ---------------------------------------------------------------------------
# Penetration (SAT narrow phase on OBBs, AABB broad-phase prefilter)
# ---------------------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vscale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _vnorm(v):
    return math.sqrt(_dot(v, v))


def obb_of_part(ir, part_name, frames=None, inflate=None):
    """Return ((cx, cy, cz), (R0, R1, R2), (hx, hy, hz)) for a part's OBB.

    R0, R1, R2 are unit vectors — the OBB's local x/y/z axes expressed in
    world frame (columns of the rotation matrix). half_extents are along
    the OBB's local axes. The OBB center is the part's world translation
    plus the rotated local-AABB center ``t + R·c``: an analytic primitive
    (``box``) has an origin-centered AABB so ``c = 0`` and the
    center reduces to ``t`` (byte-identical to the prior behavior), while a
    mesh authored off its local origin has ``c != 0`` and its collider is
    correctly placed at the mesh's true center rather than the part origin.

    ``inflate`` is this part's displacement bound `(lo3, hi3)` when it is being
    deformed on the frame in question — see ``_inflate_local_box``. It grows the
    half extents and shifts the center, which is what keeps the SAT narrow phase
    from testing the rest shape of a part whose vertices have moved.
    """
    if frames is None:
        frames = _compute_frames(ir)
    t, q = frames[part_name]
    parts = _index_parts(ir)
    lo, hi = _inflate_local_box(
        local_bbox(ir, part_name, parts), inflate,
        margin=collide_margin(parts.get(part_name) or {}))
    half = ((hi[0] - lo[0]) * 0.5,
            (hi[1] - lo[1]) * 0.5,
            (hi[2] - lo[2]) * 0.5)
    c = ((lo[0] + hi[0]) * 0.5,
         (lo[1] + hi[1]) * 0.5,
         (lo[2] + hi[2]) * 0.5)
    cw = _rot.quat_rotate_vec(q, c)   # R·c ; (0,0,0) for the analytic primitives
    center = (t[0] + cw[0], t[1] + cw[1], t[2] + cw[2])
    r0 = _rot.quat_rotate_vec(q, (1.0, 0.0, 0.0))
    r1 = _rot.quat_rotate_vec(q, (0.0, 1.0, 0.0))
    r2 = _rot.quat_rotate_vec(q, (0.0, 0.0, 1.0))
    return center, (r0, r1, r2), half


def _aabbs_overlap(bb_a, bb_b):
    """Strict-positive AABB overlap predicate. Matches the prior broad
    phase: a face-flush configuration (overlap == 0) returns False."""
    (axmn, aymn, azmn), (axmx, aymx, azmx) = bb_a
    (bxmn, bymn, bzmn), (bxmx, bymx, bzmx) = bb_b
    return (axmx - bxmn > _PENETRATION_EPS and bxmx - axmn > _PENETRATION_EPS
            and aymx - bymn > _PENETRATION_EPS and bymx - aymn > _PENETRATION_EPS
            and azmx - bzmn > _PENETRATION_EPS and bzmx - azmn > _PENETRATION_EPS)


def _project_obb_half(R_cols, half, L_hat):
    """Half-projection length of an OBB onto unit world axis L_hat."""
    return (abs(_dot(R_cols[0], L_hat)) * half[0]
            + abs(_dot(R_cols[1], L_hat)) * half[1]
            + abs(_dot(R_cols[2], L_hat)) * half[2])


def sat_obb_obb(obb_a, obb_b) -> SATResult:
    """OBB-OBB Separating Axis Theorem test (total).

    Returns a `SATResult(collided, signed_distance, axis, contact_point)`:

      * `collided`        — True iff every candidate axis shows overlap
                            (positive depth-along beyond _PENETRATION_EPS).
                            False iff some axis separates the OBBs.
      * `signed_distance` — +margin (separated; largest positive margin
                            across all axes) OR -depth (colliding; the
                            minimum overlap = MTV depth).
      * `axis`             — unit world-frame vector. For colliding pairs:
                            the MTV direction pointing from A to B (so
                            translating B by +depth*axis separates them).
                            For separated pairs: the max-separating axis
                            (oriented A→B by convention; translating B by
                            +signed_distance*axis brings the pair to contact).
      * `contact_point`   — world-frame point inside the overlap volume
                            (collided) or None (separated).

    The 15 candidates are the 3 face normals of A, the 3 face normals of
    B, and the 9 pairwise cross products of edge directions. Degenerate
    cross axes (magnitude < EPS_AXIS, from near-parallel face normals)
    are skipped; the face-normal axes alone cover that case. Among
    overlapping axes the smallest overlap is the MTV; among separating
    axes the largest margin is the signed_distance. Face-normal axes
    are preferred over cross-product axes on ties within EPS_TIEBREAK.
    """
    c_a, R_a, h_a = obb_a
    c_b, R_b, h_b = obb_b
    delta = _vsub(c_b, c_a)

    # Build candidate axes: face axes first (kind=0), cross axes (kind=1).
    candidates = []
    for ra in R_a:
        candidates.append((ra, 0))
    for rb in R_b:
        candidates.append((rb, 0))
    for ra in R_a:
        for rb in R_b:
            candidates.append((_cross(ra, rb), 1))

    best_overlap = None  # (depth, kind, L_hat, signed_d) — for colliding case
    best_margin = None   # (margin, kind, L_hat, signed_d) — for separated case
    for L, kind in candidates:
        mag2 = _dot(L, L)
        if mag2 < _AXIS_EPS * _AXIS_EPS:
            continue  # Degenerate cross — parallel face normals.
        inv = 1.0 / math.sqrt(mag2)
        L_hat = _vscale(L, inv)
        d = _dot(delta, L_hat)
        r_a = _project_obb_half(R_a, h_a, L_hat)
        r_b = _project_obb_half(R_b, h_b, L_hat)
        depth_along = (r_a + r_b) - abs(d)
        if depth_along <= _PENETRATION_EPS:
            # Separating axis. Track LARGEST margin (most-separating axis).
            margin = -depth_along  # >= 0 by construction (could be tiny)
            if best_margin is None:
                best_margin = (margin, kind, L_hat, d)
            else:
                prev_margin, prev_kind, _, _ = best_margin
                if margin > prev_margin + _TIEBREAK_EPS:
                    best_margin = (margin, kind, L_hat, d)
                elif abs(margin - prev_margin) <= _TIEBREAK_EPS and kind < prev_kind:
                    best_margin = (margin, kind, L_hat, d)
        else:
            # Overlap axis. Track SMALLEST overlap (MTV).
            if best_overlap is None:
                best_overlap = (depth_along, kind, L_hat, d)
            else:
                prev_depth, prev_kind, _, _ = best_overlap
                if depth_along + _TIEBREAK_EPS < prev_depth:
                    best_overlap = (depth_along, kind, L_hat, d)
                elif abs(depth_along - prev_depth) <= _TIEBREAK_EPS and kind < prev_kind:
                    best_overlap = (depth_along, kind, L_hat, d)

    # SAT theorem: OBBs are SEPARATED iff at least one separating axis exists.
    if best_margin is not None:
        margin, _, L_hat, d = best_margin
        # Orient the axis from A to B so translating B by +margin*axis
        # brings the OBBs into contact (rather than further apart).
        if d > 0.0:
            normal = L_hat
        elif d < 0.0:
            normal = _vscale(L_hat, -1.0)
        else:
            dmag = _vnorm(delta)
            if dmag > _AXIS_EPS:
                normal = _vscale(delta, 1.0 / dmag)
            else:
                normal = (1.0, 0.0, 0.0)
        return SATResult(collided=False, signed_distance=margin,
                         axis=normal, contact_point=None)

    # All axes overlap → colliding. best_overlap is guaranteed non-None
    # for non-degenerate OBBs (face-normal axes always present).
    depth, _, L_hat, d = best_overlap
    if d > 0.0:
        normal = L_hat
    elif d < 0.0:
        normal = _vscale(L_hat, -1.0)
    else:
        # Coincident centers along this axis. Tie-break: align with
        # center-to-center direction if non-zero, else world +x.
        dmag = _vnorm(delta)
        if dmag > _AXIS_EPS:
            normal = _vscale(delta, 1.0 / dmag)
        else:
            normal = (1.0, 0.0, 0.0)
    cp = _contact_point(obb_a, obb_b, normal)
    return SATResult(collided=True, signed_distance=-depth,
                     axis=normal, contact_point=cp)


def _line_clip_obb(p0, dp, obb):
    """Slab-clip the parametric line p0 + t * dp against an OBB.

    Returns (t_enter, t_exit) or None if the line never enters the OBB.
    """
    c, R_cols, half = obb
    rel = _vsub(p0, c)
    local_p0 = (_dot(R_cols[0], rel),
                _dot(R_cols[1], rel),
                _dot(R_cols[2], rel))
    local_dp = (_dot(R_cols[0], dp),
                _dot(R_cols[1], dp),
                _dot(R_cols[2], dp))
    t_min = -float("inf")
    t_max = float("inf")
    for i in range(3):
        h = half[i]
        if abs(local_dp[i]) < _AXIS_EPS:
            if local_p0[i] < -h - _AXIS_EPS or local_p0[i] > h + _AXIS_EPS:
                return None
            continue
        t1 = (-h - local_p0[i]) / local_dp[i]
        t2 = (h - local_p0[i]) / local_dp[i]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
        if t2 < t_max:
            t_max = t2
        if t_min > t_max:
            return None
    return t_min, t_max


def _contact_point(obb_a, obb_b, normal_world):
    """Compute a contact point inside the overlap volume.

    Primary recipe: midpoint of the overlap segment on the line from c_A
    to c_B. Fallback (used when the center-to-center line does not pass
    through the intersection — possible for two convex shapes whose
    intersection lies off the centers' axis): midpoint of the support
    points along the MTV axis.
    """
    c_a = obb_a[0]
    c_b = obb_b[0]
    dp = _vsub(c_b, c_a)
    if _vnorm(dp) > _AXIS_EPS:
        ta = _line_clip_obb(c_a, dp, obb_a)
        tb = _line_clip_obb(c_a, dp, obb_b)
        if ta is not None and tb is not None:
            t_enter = max(ta[0], tb[0])
            t_exit = min(ta[1], tb[1])
            if t_enter <= t_exit:
                t_mid = 0.5 * (t_enter + t_exit)
                return (c_a[0] + t_mid * dp[0],
                        c_a[1] + t_mid * dp[1],
                        c_a[2] + t_mid * dp[2])
    # Fallback: midpoint of MTV-axis support points.
    r_a = _project_obb_half(obb_a[1], obb_a[2], normal_world)
    r_b = _project_obb_half(obb_b[1], obb_b[2], normal_world)
    s_a = _vadd(c_a, _vscale(normal_world, r_a))
    s_b = _vsub(c_b, _vscale(normal_world, r_b))
    return (0.5 * (s_a[0] + s_b[0]),
            0.5 * (s_a[1] + s_b[1]),
            0.5 * (s_a[2] + s_b[2]))


def penetration_pairs(ir, poses=None, mount_deltas=None, deform_bounds=None,
                      skip_pairs=None):
    """Intra-object self-penetration pairs.

    Returns a list of `PenetrationPair` records ordered lexicographically
    by (a, b). The narrow phase is OBB-OBB SAT; the broad phase is
    AABB-AABB overlap. Parent/child pairs are NOT excluded — an
    intentional overlap at the evaluated pose is real authoring signal
    worth surfacing.

    `poses` is an optional `{joint_name: q}` mapping; defaults to the
    rest pose (all joints at their authored defaults). `mount_deltas` moves
    child part frames under a deforming parent — see `_apply_mount`; without it
    an animation's self-collision sweep measures every child at its rest mount.

    `skip_pairs` is a set of `frozenset({a, b})` never to test. The caller uses
    it for pairs it would discard anyway — a blanket `allow_collision` exempts
    its pair at every pose, so testing it is pure cost. Both phases are skipped,
    not just the narrow one: the AABB test is the cheaper half but it runs on
    every pair of every pose, which is where the count lives.

    `deform_bounds` (`{part: (lo3, hi3)}`) grows the box and the OBB of a part
    that is *itself* deforming. Both phases need it, not just one: a deforming
    parent can reach a child that its rest box does not, so without it the broad
    phase drops the pair before the narrow phase ever sees it.
    """
    frames = _compute_frames(ir, poses=poses, mount_deltas=mount_deltas)
    obbs = {p["name"]: obb_of_part(ir, p["name"], frames,
                                   _inflate_for(deform_bounds, p["name"]))
            for p in ir["parts"]}
    aabbs = {
        name: _aabb_of_transformed_corners(
            _box_corners(_inflate_local_box(
                local_bbox(ir, name), _inflate_for(deform_bounds, name))),
            frames[name])
        for name in obbs
    }
    names = sorted(obbs.keys())
    skip = skip_pairs or frozenset()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if skip and frozenset((a, b)) in skip:
                continue
            if not _aabbs_overlap(aabbs[a], aabbs[b]):
                continue
            res = sat_obb_obb(obbs[a], obbs[b])
            if not res.collided:
                continue
            pairs.append(PenetrationPair(
                a=a, b=b,
                depth=-res.signed_distance,
                contact_normal=res.axis,
                contact_point=res.contact_point,
                signed_distance=res.signed_distance,
                a_frame=_frame_to_surface(frames[a]),
                b_frame=_frame_to_surface(frames[b]),
                a_bbox=(tuple(aabbs[a][0]), tuple(aabbs[a][1])),
                b_bbox=(tuple(aabbs[b][0]), tuple(aabbs[b][1])),
                witnessing_pose=dict(poses) if poses else None,
            ))
    return pairs


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Anchor positions (shared-frame, viewer-relative convention)
# ---------------------------------------------------------------------------


def anchor_position(ir, part_name: str, anchor_name: str, poses=None,
                    anchor_offset=None):
    """World-frame position of an anchor on `part`.

    Anchors are named points on the part's AABB (the 27-name vocabulary
    in sdk/_anchors.py). The position is the AABB landmark transformed
    by the part's pose-frame, then by any `anchor_offset` translation
    in part-local. Returns a 3-tuple (x, y, z).

    `anchor_offset` is `((tx,ty,tz),(rx,ry,rz))` — the surface form of
    an AnchorRef's offset. Only the translation contributes to the
    position; the rotation is irrelevant for point queries. Pass
    `None` for an identity offset.

    `poses` may be a `{joint_name: q}` mapping (or `None` for rest pose).
    Joint values override their authored `default` along the part's
    ancestor chain; keys not on the chain are silently ignored.
    """
    part_local_aabb = local_bbox(ir, part_name)
    # Same resolver the builder uses, so a semantic anchor means the same
    # frame in an assertion as it does in a joint. A built-in resolves to the
    # AABB landmark with identity rotation, exactly as before.
    part_record = next((p for p in ir["parts"] if p["name"] == part_name), None)
    base_t, base_q = _anchors_mod.resolve_base_frame(
        part_local_aabb, (part_record or {}).get("anchors"), anchor_name,
        part_name=part_name)
    if anchor_offset is not None:
        (otx, oty, otz), _orot = anchor_offset
        # The offset translation is in the ANCHOR's frame, so it must be
        # rotated by the base orientation before being added. Identity for a
        # built-in, which keeps the historical result unchanged.
        rot_off = _rot.quat_rotate_vec(base_q, (otx, oty, otz))
        anchor_local = (base_t[0] + rot_off[0],
                        base_t[1] + rot_off[1],
                        base_t[2] + rot_off[2])
    else:
        anchor_local = base_t
    frames = _compute_frames(ir, poses=poses)
    if part_name not in frames:
        raise KeyError(part_name)
    t, q = frames[part_name]
    rotated = _rot.quat_rotate_vec(q, anchor_local)
    return (t[0] + rotated[0], t[1] + rotated[1], t[2] + rotated[2])


def outward(ir, part_name: str, anchor_name: str, poses=None,
            anchor_offset=None):
    """Unit outward direction of an anchor in the object root frame.

    For face anchors this is the outward face normal; for edges and
    corners the normalized bisector / diagonal. If `anchor_offset`
    carries a non-identity rotation, the part-local outward is first
    rotated by it (in part-local); the result is then transformed
    through the part's pose-frame.

    Raises ValueError for the `center` anchor (no outward direction).
    """
    local_dir = _anchors_mod.outward_direction(anchor_name)
    if anchor_offset is not None:
        _otrans, (rx, ry, rz) = anchor_offset
        if rx != 0.0 or ry != 0.0 or rz != 0.0:
            q_off = _rot.euler_xyz_deg_to_quat_wxyz(rx, ry, rz)
            q_off = _rot.sign_canonicalize_quat(_rot.normalize_quat(q_off))
            local_dir = _rot.quat_rotate_vec(q_off, local_dir)
    frames = _compute_frames(ir, poses=poses)
    if part_name not in frames:
        raise KeyError(part_name)
    _t, q = frames[part_name]
    return _rot.quat_rotate_vec(q, local_dir)


def part_world_aabb(ir, part_name: str, poses=None):
    """World-frame axis-aligned bounding box of `part` at the given pose.

    Returns `((xmin, ymin, zmin), (xmax, ymax, zmax))` in the object's
    root frame. The 8 corners of the part's local AABB are transformed
    by the part's pose-frame and re-axis-aligned.
    """
    frames = _compute_frames(ir, poses=poses)
    if part_name not in frames:
        raise KeyError(part_name)
    return _aabb_of_transformed_corners(
        _box_corners(local_bbox(ir, part_name)),
        frames[part_name],
    )


def resolve_entity_aabb(ir, entity_dict):
    """Object-layer entity-AABB resolver for `_eval_overlap`.

    Object-layer only supports `kind="part"`; other kinds raise.
    """
    kind = entity_dict.get("kind")
    if kind != "part":
        raise ValueError(
            f"object layer cannot resolve EntityRef of kind {kind!r}; "
            f"only 'part' is supported"
        )
    return part_world_aabb(ir, entity_dict["name"],
                           poses=entity_dict.get("poses") or None)


def resolve_direction(ir, direction_dict):
    """Object-layer direction resolver for `_eval_angle`."""
    src = direction_dict["source"]
    poses = direction_dict.get("poses") or None
    if src == "world":
        return tuple(direction_dict["vec"])
    if src == "joint_axis":
        return joint_axis_world(ir, direction_dict["joint"], poses=poses)
    if src == "anchor_outward":
        if direction_dict.get("entity_kind") != "part":
            raise ValueError(
                "object layer anchor_outward requires entity_kind='part'"
            )
        anchor_offset = direction_dict.get("anchor_offset")
        return outward(ir, direction_dict["entity"],
                       direction_dict["anchor"], poses=poses,
                       anchor_offset=anchor_offset)
    raise ValueError(f"unknown direction source {src!r}")


def joint_axis_world(ir, joint_name: str, poses=None):
    """Unit world-frame axis of a revolute/prismatic joint at the
    requested pose (rest pose if `poses` is None).

    The authored axis is in the joint's parent_anchor's local frame.
    Composition to world: rotate through parent_anchor.R (parent-anchor
    -> parent-part-local), then through the parent's world-frame
    orientation at the pose.

    Raises KeyError if joint name is unknown. Raises ValueError if the
    joint is fixed (no axis).
    """
    joints_by_name = {j["name"]: j for j in ir["joints"]}
    if joint_name not in joints_by_name:
        raise KeyError(joint_name)
    joint = joints_by_name[joint_name]
    if joint["type"] == "fixed":
        raise ValueError(
            f"joint {joint_name!r} is fixed and has no kinematic axis"
        )
    axis = joint["axis"]
    norm = (axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2) ** 0.5
    axis_pa_unit = (axis[0] / norm, axis[1] / norm, axis[2] / norm)
    # parent_anchor's rotation lifts the axis from pa-local to parent-part-local.
    pa = _se3_from_anchor(joint["parent_anchor"])
    axis_in_parent = _rot.quat_rotate_vec(pa[1], axis_pa_unit)
    frames = _compute_frames(ir, poses=poses)
    parent_name = joint["parent"]
    if parent_name not in frames:
        raise KeyError(parent_name)
    _t, q = frames[parent_name]
    return _rot.quat_rotate_vec(q, axis_in_parent)


def ir_snapshot(ir):
    """Return `ir` unchanged."""
    return ir
