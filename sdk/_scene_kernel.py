"""Scene query module: stateless functions over a finalized scene IR dict.

All functions take a scene IR (returned by Scene.finalize()) and a catalog
mapping `object_ref` hash -> finalized ObjectIR dict. The catalog is required
because the scene IR references object IRs by hash and per-part / FK lookups
must consult the underlying ObjectIR. Names are unique within their namespace
so name lookup is unambiguous.

All rotation outputs are intrinsic Euler XYZ in degrees. Quaternions never
appear in any function output (the IR retains quaternions for hash stability).

Forward kinematics for the attachment forest evaluates at the scene's
**committed** configuration: each instance's saved `initial_joints` where it
has them, each object's joint `default` for every joint it does not name, and
every attach edge resolved through the forest. `poses=None` means that
committed state, not the definitions' defaults — `effective_object_poses` is
the single place the precedence between the two is resolved, and every query
here goes through it.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import _rotation as _rot
from . import _anchors as _anchors_mod
from . import _object_kernel as _object_query


_PENETRATION_EPS = 1e-6


# ---------------------------------------------------------------------------
# Constants matching scene_builder.py
# ---------------------------------------------------------------------------

_ROOM_RESERVED_NAME = "room"


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _index_instances(ir) -> dict:
    return {i["name"]: i for i in ir["instances"]}


def _index_groups(ir) -> dict:
    return {g["name"]: g for g in ir["groups"]}


# ---------------------------------------------------------------------------
# Public list-style queries
# ---------------------------------------------------------------------------


def list_instances(ir) -> list:
    return [i["name"] for i in ir["instances"]]


def list_groups(ir) -> list:
    return [g["name"] for g in ir["groups"]]


def list_assertions(ir) -> list:
    return [a["name"] for a in ir["assertions"]]


# ---------------------------------------------------------------------------
# Catalog / object AABB lookup
# ---------------------------------------------------------------------------


def _unknown_entity(ir, name: str, *, catalog=None) -> KeyError:
    """A `KeyError` that says what the caller could have asked for.

    A bare `KeyError(name)` cannot tell "no such instance" from "no such part"
    from "this probe does not take dotted names". The dotted form is supported unevenly across the probe
    surface, and until that is uniform the error is what has to carry the
    difference.

    Still a `KeyError`, so every caller and `except` clause behaves as before;
    only the message grows.
    """
    inst = sorted(i["name"] for i in (ir.get("instances") or []))
    head, _sep, tail = str(name).partition(".")
    if _sep and head in inst:
        parts = []
        if catalog is not None:
            try:
                obj = _object_ir_for(
                    catalog,
                    next(i["object_ref"] for i in ir["instances"]
                         if i["name"] == head))
                parts = sorted(q.get("name") for q in (obj.get("parts") or []))
            except Exception:                                  # noqa: BLE001
                parts = []
        detail = (f"instance {head!r} exists but has no part {tail!r}"
                  if parts and tail not in parts else
                  f"this query does not accept the dotted "
                  f"'<instance>.<part>' form; pass {head!r} and scope the "
                  f"part another way")
        return KeyError(f"{name!r}: {detail}"
                        + (f". Parts of {head!r}: {parts}" if parts else ""))
    return KeyError(f"{name!r}: no such instance or group. "
                    f"Instances: {inst[:12]}"
                    + (" ..." if len(inst) > 12 else ""))


def _object_ir_for(catalog, object_ref: str) -> dict:
    """The ObjectIR a SceneIR instance names, or a diagnosable failure.

    A bare `KeyError(object_ref)` is a 32-character content hash and nothing
    else. A miss can be transient: a compile that runs while a snapshot build
    is still writing the catalog holds a mapping loaded before the entry
    existed. The per-file write is atomic (`scene_context.atomic_write_text`),
    so the race is on the reader side.

    So say which instance wanted it, which refs were available, and that a
    retry may succeed. `DanglingReference` is the SDK's shape for "this
    reference does not resolve".
    """
    if object_ref not in catalog:
        from ._errors import DanglingReference
        known = sorted(catalog)[:6]
        raise DanglingReference(
            f"no ObjectIR in the catalog for object_ref {object_ref!r}",
            refs=[],
            evidence={"object_ref": object_ref,
                      "n_catalog_entries": len(catalog),
                      "known_refs_sample": known,
                      "reason": "catalog_miss"},
            suggested_repairs=[
                "if a build or snapshot was running concurrently, this "
                "reference may not have been written yet — repeat the "
                "operation on unchanged source before treating it as a "
                "geometry failure",
                "otherwise the instance names an object that was never "
                "committed to the catalog; check the object built and "
                "finalized",
            ],
        )
    return catalog[object_ref]


def _object_global_bbox(catalog, object_ref: str, part_name=None, poses=None,
                        mount_deltas=None, deform_bounds=None):
    return _object_query.global_bbox(_object_ir_for(catalog, object_ref),
                                     part_name=part_name, poses=poses,
                                     mount_deltas=mount_deltas,
                                     deform_bounds=deform_bounds)


# ---------------------------------------------------------------------------
# Placement resolution: walk the attachment forest to a world pose.
# ---------------------------------------------------------------------------


def _se3_from_ir_pose(pose_dict):
    t = pose_dict["translation"]
    q = pose_dict["rotation"]
    return ((float(t[0]), float(t[1]), float(t[2])),
            (float(q[0]), float(q[1]), float(q[2]), float(q[3])))


def _identity_se3():
    return ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _resolve_world_pose(ir, catalog, entity_name: str, memo: dict,
                        *, poses=None, base_deltas=None,
                        mount_deltas=None) -> tuple:
    """Return the world-frame SE3 of the entity at the given scene pose.

    `poses` is an optional scene-level `{joint_name: q}` / qualified
    `{"<instance>.<joint>": q}` mapping. When supplied, every attach
    edge is resolved against the *posed* frame of its parent anchor, so
    an instance bolted to a moving part of its parent rides that joint
    (kinematic-follow). `poses=None` resolves the attach forest at the scene's
    committed configuration — each instance's `initial_joints` over its
    object's authored `default`s. An attach edge authored with `follow=False`
    stays pinned to its parent's *committed* frame regardless of `poses`.

    `base_deltas` is an optional `{entity_name: se3}` map of *whole-body*
    rigid deltas in world axes, left-composed onto the resolved placement:
    ``world = delta ∘ placement``. It is how the dynamics layer moves an
    instance the SceneIR pins by an attach edge — the SceneIR itself needs
    no floating-base placement kind: base motion is a delta from the declared
    placement. Because it is
    applied here, *before* memoisation, and because `_resolve_placement`
    reaches a parent through this same function, a child rides its parent's
    delta for free: move the tray and the cup on it follows. A child that
    also carries its own delta is then moved relative to the tray.

    Deltas propagate across `follow=False` edges too. `follow` governs
    whether a child tracks its parent's *articulation*; it says nothing
    about the parent's whole body translating. If the table moves,
    everything on it moves.

    The result is in internal SE3 form (translation, quaternion) — caller
    is responsible for converting rotation to Euler degrees if returning
    to the LLM.
    """
    if entity_name == _ROOM_RESERVED_NAME:
        return _identity_se3()
    if entity_name in memo:
        return memo[entity_name]
    pose = _resolve_placed_pose(ir, catalog, entity_name, memo,
                                poses=poses, base_deltas=base_deltas,
                                mount_deltas=mount_deltas)
    if base_deltas:
        delta = base_deltas.get(entity_name)
        if delta is not None:
            pose = _rot.se3_compose(delta, pose)
    memo[entity_name] = pose
    return pose


def _resolve_placed_pose(ir, catalog, entity_name: str, memo: dict,
                         *, poses=None, base_deltas=None,
                         mount_deltas=None) -> tuple:
    """`_resolve_world_pose` without this entity's own `base_deltas` entry.

    Split out so the delta is applied exactly once, on every return path,
    rather than repeated at each of the branches below. Recursive lookups
    go back through `_resolve_world_pose`, so ancestors keep their deltas.
    Callers other than `_resolve_world_pose` want that function instead.
    """
    instances = _index_instances(ir)
    groups = _index_groups(ir)

    if entity_name in instances:
        inst = instances[entity_name]
        # If this instance is the root of an attached group, its own
        # placement is null and the authoritative pose comes from the
        # group's placement.
        placement = inst["placement"]
        if placement is None:
            # Find the group whose root is this instance.
            owning_group_name = None
            for g in ir["groups"]:
                if g["root"] == entity_name and g["placement"] is not None:
                    owning_group_name = g["name"]
                    break
            if owning_group_name is None:
                # Shadowed but no group with placement found — IR
                # integrity error.
                raise ValueError(
                    f"instance {entity_name!r} placement is null but no "
                    f"containing group provides a placement")
            return _resolve_world_pose(ir, catalog, owning_group_name, memo,
                                       poses=poses, base_deltas=base_deltas,
                                       mount_deltas=mount_deltas)
        return _resolve_placement(ir, catalog, placement, entity_name, memo,
                                  entity_kind="instance", poses=poses,
                                  base_deltas=base_deltas,
                                  mount_deltas=mount_deltas)

    if entity_name in groups:
        g = groups[entity_name]
        placement = g["placement"]
        if placement is None:
            # Unattached group: its pose equals its root member's pose
            # (using the root's own placement). Recurse via the root's
            # instance/group record.
            root_name = g["root"]
            # Bypass the null-shadow logic for the root member by
            # consulting the instance/group record directly.
            if root_name in instances:
                root_placement = instances[root_name]["placement"]
            else:
                root_placement = groups[root_name]["placement"]
            if root_placement is None:
                raise ValueError(
                    f"group {entity_name!r} has null placement and root "
                    f"{root_name!r} also has null placement")
            return _resolve_placement(ir, catalog, root_placement,
                                      root_name, memo, entity_kind="instance"
                                      if root_name in instances else "group",
                                      poses=poses, base_deltas=base_deltas,
                                      mount_deltas=mount_deltas)
        # The group's placement is authoritative; resolve relative to
        # parent. We treat the group's local frame as the root member's
        # local frame (anchors are in the group's frame; the union AABB
        # is also in the group's frame).
        return _resolve_placement(ir, catalog, placement, entity_name,
                                  memo, entity_kind="group", poses=poses,
                                  base_deltas=base_deltas,
                                  mount_deltas=mount_deltas)

    raise KeyError(entity_name)


def _resolve_placement(ir, catalog, placement, entity_name: str,
                       memo: dict, *, entity_kind: str, poses=None,
                       base_deltas=None, mount_deltas=None) -> tuple:
    if placement["kind"] == "attach":
        parent_name = placement["parent"]
        # `follow` (default True) makes this edge track the parent
        # anchor's *posed* frame, so a child bolted to a moving part
        # rides the joint. `follow=False` pins the child (and its whole
        # sub-forest) to the parent's rest frame — the historical
        # behavior, kept for "place near a moving part but stay put".
        #
        # `base_deltas` is forwarded on BOTH branches: `follow` scopes
        # articulation only, so a `follow=False` child still rides its
        # parent's whole-body motion.
        #
        # `mount_deltas` joins `poses` in gating the follow branch, for the
        # same reason: a mount delta moves the parent part's FRAME, so an
        # instance attached to a part that rode its parent's deformation has
        # to ride it too. Leaving it out of the gate would resolve that child
        # against a rest frame that nothing is at any more.
        follow = placement.get("follow", True)
        if follow:
            parent_world = _resolve_world_pose(ir, catalog, parent_name, memo,
                                               poses=poses,
                                               base_deltas=base_deltas,
                                               mount_deltas=mount_deltas)
            pa_poses = effective_object_poses(ir, parent_name, poses)
            pa_mounts = _object_mounts_for_instance(mount_deltas, parent_name)
        else:
            # Resolve the parent (and parent anchor) at the scene's COMMITTED
            # state — its `initial_joints`, not the definition's defaults and
            # not the evaluation poses. `follow=False` means "aim at where this
            # articulation sits in the delivered scene and stay there", so an
            # author who placed a spindle at 30 degrees gets a fixture aligned
            # to 30 degrees that then ignores the animation driving it.
            #
            # A *posed* shared memo must be bypassed so committed-state poses
            # never contaminate it — but when `poses` is falsy the shared memo
            # already IS a committed-state memo, so reusing it is both safe and
            # much cheaper (otherwise every such attach edge re-walks its
            # parent's chain from scratch). `base_deltas` does not affect that
            # reasoning: it is passed identically down both branches, so a
            # delta-bearing memo stays internally consistent.
            rest_memo = {} if (poses or mount_deltas) else memo
            parent_world = _resolve_world_pose(ir, catalog, parent_name,
                                               rest_memo, poses=None,
                                               base_deltas=base_deltas)
            pa_poses = effective_object_poses(ir, parent_name, None)
            pa_mounts = None
        pa_local = _anchor_local_pose(ir, catalog, parent_name,
                                      placement["parent_anchor"],
                                      poses=pa_poses,
                                      mount_deltas=pa_mounts)
        # The child anchor is the grab-point on the child's own body, resolved
        # in the configuration the child was PLACED in — its `initial_joints`,
        # supplied by `_anchor_local_pose` when `poses` is None. Deliberately
        # not the evaluation poses: alignment is decided once, at placement, so
        # opening this instance's door for a snapshot does not re-derive where
        # its body sits and slide it sideways to keep a growing bbox bottom on
        # the floor.
        ca_local = _anchor_local_pose(ir, catalog, entity_name,
                                      placement["child_anchor"])
        # child_world = parent_world ∘ pa_local ∘ inverse(ca_local)
        # Each anchor's pose carries its own translation + rotation in
        # its entity-local frame; offsets baked into the AnchorRef at
        # build time are already part of pa_local / ca_local.
        ca_local_inv = _rot.se3_inverse(ca_local)
        local = _rot.se3_compose(pa_local, ca_local_inv)
        return _rot.se3_compose(parent_world, local)
    raise ValueError(f"unknown placement kind: {placement['kind']!r}")


# ---------------------------------------------------------------------------
# Anchor positions in their entity's local frame.
# ---------------------------------------------------------------------------


def _anchor_local_position(ir, catalog, entity_name: str,
                           anchor_name: str, poses=None,
                           anchor_offset=None, mount_deltas=None) -> tuple:
    """Return the anchor's position (3-tuple) in the entity's local frame.

    For instances: local frame = the object's root frame. For groups: local
    frame = the group root member's local frame. For the room: local frame
    = world frame (room is at identity).

    `anchor_offset` is `((tx,ty,tz),(rx,ry,rz))` (or `None` = identity).
    Only the translation contributes; rotation is irrelevant for points.

    `poses` reaches the **part-scoped** branch only: a part's AABB — and so
    every anchor on it — travels with the joints above it, and a query that
    ignored that would silently report the rest-pose location of a door
    handle that has swung open. A *whole-instance* auto-AABB anchor is
    deliberately still resolved against the object's canonical extent, though
    at the instance's **committed** configuration: those 27 names are defined
    against the canonical bbox, and letting them drift under an *evaluation*
    pose would move `table.top` sideways the moment a drawer opens, silently
    re-placing everything attached to it. An instance's `initial_joints` is
    not drift — it is static for the life of the scene — so it does apply.
    Ask for `"<part>.<name>"` when you want the posed answer.

    `mount_deltas` reaches the same part-scoped branch, for the same reason and
    on the same terms as `poses`: a mount delta moves the part's *frame*, so a
    landmark on it is somewhere else. This is also what keeps this function
    agreeing with its SE3 sibling `_anchor_local_pose`, which has been
    mount-aware since mounts existed — the two answer the same question and one
    of them silently answering at the rest mount is the worst of both.

    **The landmark itself is still taken on the part's REST AABB**, deliberately
    and unlike a bound: `deform_bounds` is not threaded here at all. `top_center`
    of an inflated box is not where anything is; the 27 names are defined against
    canonical extents. A part that deforms therefore reports a stale landmark,
    which is a known limitation rather than something this
    parameter could fix.
    """
    if entity_name == _ROOM_RESERVED_NAME:
        pos = _room_anchor_position(ir["room"], anchor_name)
    else:
        instances = _index_instances(ir)
        groups = _index_groups(ir)
        if entity_name in instances:
            inst = instances[entity_name]
            # Part-scoped anchor?
            if "." in anchor_name:
                part_name, _dot, sub = anchor_name.partition(".")
                inst_poses = effective_object_poses(ir, entity_name, poses)
                inst_mounts = _object_mounts_for_instance(mount_deltas,
                                                          entity_name)
                if sub not in _anchors_mod.AUTO_AABB_ANCHOR_NAMES:
                    # A **declared** anchor — an author's `contact_frame` /
                    # `hinge_axis`, or an importer's URDF frame. It is a point
                    # in the part's OWN local frame, not a landmark on a box,
                    # so it rides the part's pose-aware frame rather than the
                    # object-root AABB: the box landmark is a rest-only
                    # artifact and does not carry the joint rotation (see
                    # `_part_anchor_pose_in_object_root`, which transports an
                    # attach record the same way).
                    #
                    # This branch makes the anchors the object layer records
                    # (the grasp frame a URDF import carries, a roller's
                    # `floor_contact`) visible to scene and animation queries,
                    # so an agent does not re-derive an arm's forward
                    # kinematics by hand to find one.
                    return _declared_part_anchor_position(
                        catalog, inst["object_ref"], part_name, sub,
                        poses=inst_poses, mount_deltas=inst_mounts,
                        anchor_offset=anchor_offset)
                aabb = _object_global_bbox(
                    catalog, inst["object_ref"], part_name,
                    poses=inst_poses, mount_deltas=inst_mounts)
            else:
                # The committed state, not `poses` — see the docstring. This
                # is what makes `place(cabinet, initial_joints={"door": 90})`
                # land the *open* cabinet's bbox-bottom on the floor.
                aabb = _object_global_bbox(
                    catalog, inst["object_ref"],
                    poses=instance_initial_joints(ir, entity_name))
                sub = anchor_name
            pos = _aabb_anchor_position(aabb, sub)
        elif entity_name in groups:
            g = groups[entity_name]
            aabb = _group_union_aabb_in_root_local(ir, catalog, g)
            pos = _aabb_anchor_position(aabb, anchor_name)
        else:
            raise KeyError(entity_name)
    if anchor_offset is not None:
        (otx, oty, otz), _orot = anchor_offset
        pos = (pos[0] + otx, pos[1] + oty, pos[2] + otz)
    return pos


def _anchor_dict_offset_surface(anchor_dict) -> tuple | None:
    """Extract the surface-form offset ((t),(euler_xyz_deg)) from an
    IR anchor placement dict, or None when the placement is a bare string
    (a landmark name with no offset).

    `Mapping`, not `dict`: the probe overlays hand sandboxed code a
    `_readonly.freeze()` view of the IR, and a `_ReadOnlyMapping` fails
    `isinstance(x, dict)`. A `dict` check would return None there and silently
    drop the anchor's translation, so a probe and the verifier it predicts
    (which sees the raw dict) would place the same instance differently.
    """
    if not isinstance(anchor_dict, Mapping):
        return None
    t = anchor_dict.get("translation")
    q = anchor_dict.get("rotation")
    if t is None or q is None:
        return None
    rx, ry, rz = _rot.quat_to_euler_xyz_deg(
        (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    )
    return ((float(t[0]), float(t[1]), float(t[2])),
            (float(rx), float(ry), float(rz)))


def _part_anchor_base_in_part(obj_ir, part_name: str, anchor_name: str) -> tuple:
    """The `Anchor -> Part` SE3 of `anchor_name` on `part_name`, before offsets.

    One line of delegation, and the point of it is that the line is shared:
    `_anchors.resolve_base_frame` is the single resolution point for both
    anchor kinds, so a built-in name yields its landmark on the part's local
    AABB with identity rotation, and a declared name yields the full frame —
    translation *and* rotation — its author registered.

    Reaching it from here is what makes a declared frame usable for scene
    attachment at all: deriving the AABB landmark here instead, in
    `_part_anchor_pose_in_object_root` (the one attachment goes through), would
    raise `KeyError` on every declared name.
    """
    part_record = next((p for p in obj_ir.get("parts") or ()
                        if p.get("name") == part_name), None)
    if part_record is None:
        raise KeyError(part_name)
    return _anchors_mod.resolve_base_frame(
        _object_query.local_bbox(obj_ir, part_name),
        part_record.get("anchors"), anchor_name, part_name=part_name)


def _compose_anchor_offset_internal(base, translation, rotation_quat) -> tuple:
    """`base ∘ (t, q)` — the anchor-offset composition, in internal SE3 form.

    **The object layer's convention, not a scene-layer variant of it.**
    `object_builder._anchor_to_se3` composes an AnchorRef's offset onto the
    resolved base frame exactly this way, so an offset means the same thing
    whether it lands on a joint anchor or a scene attachment. For a built-in
    landmark the base rotation is identity and this reduces to
    `landmark + translation` with the offset rotation standing — byte-identical
    to what this module did before. On a declared frame the base rotation
    participates, so "5 mm along the mounting face's normal" follows the face.

    Both components may be `None` (= identity for that component), which is how
    an IR anchor record with no baked offset reads.
    """
    t = ((0.0, 0.0, 0.0) if translation is None
         else (float(translation[0]), float(translation[1]),
               float(translation[2])))
    q = ((1.0, 0.0, 0.0, 0.0) if rotation_quat is None
         else (float(rotation_quat[0]), float(rotation_quat[1]),
               float(rotation_quat[2]), float(rotation_quat[3])))
    if t == (0.0, 0.0, 0.0) and q == (1.0, 0.0, 0.0, 0.0):
        # Skip the compose rather than multiply by identity: it keeps every
        # offset-free anchor bit-for-bit on its historical arithmetic.
        return base
    return _rot.se3_compose(base, (t, q))


def _compose_anchor_offset(base, anchor_offset) -> tuple:
    """`_compose_anchor_offset_internal` for a surface-form offset.

    `anchor_offset` is `((tx,ty,tz),(rx,ry,rz))` with Euler XYZ degrees — the
    shape `_anchor_dict_offset_surface` and `AnchorRef._offset` carry. `None`
    returns `base` untouched.
    """
    if anchor_offset is None:
        return base
    (otx, oty, otz), (orx, ory, orz) = anchor_offset
    return _compose_anchor_offset_internal(
        base, (otx, oty, otz),
        _rot.euler_xyz_deg_to_quat_wxyz(float(orx), float(ory), float(orz)))


def _declared_part_anchor_position(catalog, object_ref: str, part_name: str,
                                   anchor_name: str, *, poses=None,
                                   mount_deltas=None,
                                   anchor_offset=None) -> tuple:
    """Object-root position of a declared anchor on `part_name`.

    Only the translation is returned, matching `_anchor_local_position`'s
    contract. A caller that needs the anchor's *frame* wants
    `_anchor_local_pose`.

    `anchor_offset` is composed here rather than added by the caller: added
    afterwards it would land in object-root axes, while the SE3 query one
    function below resolves it through the anchor's own rotation. Both share
    `_compose_anchor_offset`, so a position query and an SE3 query agree.
    """
    obj_ir = _object_ir_for(catalog, object_ref)
    base = _part_anchor_base_in_part(obj_ir, part_name, anchor_name)
    frames = _object_query._compute_frames(obj_ir, poses=poses,
                                           mount_deltas=mount_deltas)
    if part_name not in frames:
        raise KeyError(part_name)
    return _rot.se3_compose(frames[part_name],
                            _compose_anchor_offset(base, anchor_offset))[0]


def _part_anchor_pose_in_object_root(obj_ir, anchor_record, poses,
                                     mount_deltas=None) -> tuple:
    """Full SE3 of a part-scoped anchor in the object's root frame.

    For `"<part>.<landmark>"` the landmark is taken on the part's *own* local
    AABB with identity rotation; for `"<part>.<declared_name>"` the base frame
    is the full rigid frame the object's author registered, rotation included.
    Either way it is then transported by the part's frame under `poses` (the
    object kernel's pose-aware forward kinematics). That transport is what lets
    an attachment to a moving part both translate *and rotate* with the joint —
    the AABB-of-the-rotated-part landmark used elsewhere is a rest-only
    artifact and does not carry the joint rotation.

    Preserving the declared rotation is the difference between mounting a
    sensor and aiming one. A translation-only resolution puts it at the right
    point on the door with the object-root's orientation, so a `wall_mount`
    frame authored to face outward faces whatever way the object happens to be
    modeled — and every downstream check passes, because the position is
    right.

    The anchor record's own `translation` / `rotation` are the AnchorRef offset
    baked at build time, composed onto that base frame by
    `_compose_anchor_offset` — the same convention `Object`'s joint anchors
    use.
    """
    name = anchor_record["name"]
    part_name, _dot, sub = name.partition(".")
    frames = _object_query._compute_frames(obj_ir, poses=poses,
                                           mount_deltas=mount_deltas)
    if part_name not in frames:
        raise KeyError(part_name)
    base = _part_anchor_base_in_part(obj_ir, part_name, sub)
    # The record's quaternion is composed as-is. Going through the surface
    # (Euler) form would round-trip a rotation through a non-canonical
    # representation — euler(180,180,180) does not come back as the triple it
    # went in as — for no gain, since nothing here needs degrees.
    anchor_in_part = _compose_anchor_offset_internal(
        base, anchor_record.get("translation"), anchor_record.get("rotation"))
    return _rot.se3_compose(frames[part_name], anchor_in_part)


def _anchor_local_pose(ir, catalog, entity_name: str,
                       anchor_record, poses=None,
                       mount_deltas=None) -> tuple:
    """Resolve an IR anchor placement record (dict) to a full SE3 in the
    entity's local frame.

    `anchor_record` shape: `{"name": str, "translation": [tx,ty,tz],
    "rotation": [qw,qx,qy,qz]}`. The translation is in entity-local; the
    rotation is the anchor's frame orientation at the (landmark +
    translation) point.

    Part-scoped anchors — `"<part>.<landmark>"` and `"<part>.<declared_name>"`
    alike — on an instance are resolved through the part's pose-aware frame so
    attachments to a moving part follow the joint; see
    `_part_anchor_pose_in_object_root`. Whole-object / group / room anchors use
    the union-AABB landmark with the baked rotation.

    `poses` is an already-projected per-instance dict in internal units.
    `None` is not "the definition's defaults" but **this instance's committed
    configuration**; see the substitution below.

    **A deformed part's landmark does not move.** `mount_deltas` transports
    the part's *frame*, so an attachment to a part that rode its parent's
    deformation follows. The landmark itself is still taken on the part's rest
    AABB, because that is what the 27-name vocabulary is defined against — so
    an instance resting on a surface that sagged under it keeps sitting at the
    rest height. Recomputing anchors from deformed geometry is a larger
    question than this parameter.
    """
    name = anchor_record["name"]
    instances = _index_instances(ir)
    # `poses=None` means the scene's COMMITTED configuration for this instance,
    # not the definition's defaults — see `effective_object_poses`. Substituting
    # it here is what gives `_resolve_placement`'s child-anchor call, the group
    # walk in `_child_in_parent_via_anchors` and the URDF exporter the same
    # answer without each of them resolving the state itself.
    if poses is None:
        poses = instance_initial_joints(ir, entity_name)
    if entity_name in instances and "." in name:
        obj_ir = _object_ir_for(catalog, instances[entity_name]["object_ref"])
        return _part_anchor_pose_in_object_root(obj_ir, anchor_record, poses,
                                               mount_deltas=mount_deltas)
    surface_offset = _anchor_dict_offset_surface(anchor_record)
    pos = _anchor_local_position(ir, catalog, entity_name, name,
                                 poses=poses,
                                 anchor_offset=surface_offset)
    q = anchor_record.get("rotation")
    if q is None:
        return (pos, (1.0, 0.0, 0.0, 0.0))
    return (pos, (float(q[0]), float(q[1]), float(q[2]), float(q[3])))


def _pose_in_local_frame(ir, catalog, entity_name: str,
                         ancestor_name: str) -> tuple:
    """Pose of entity_name in the local frame of ancestor_name, walking
    attach edges *within* the sub-forest rooted at ancestor_name.

    Used to compute group union AABBs without going through world
    coordinates — avoids the world-pose <-> group-AABB recursion when a
    group's anchor position is being queried as part of resolving the
    group's own world pose.

    Returns an internal SE3 (translation, quaternion).
    """
    if entity_name == ancestor_name:
        return _identity_se3()
    instances = _index_instances(ir)
    groups = _index_groups(ir)
    # Locate the placement of entity_name. If entity_name is the root of
    # an attached group, its own placement is null and the group's
    # placement is authoritative.
    if entity_name in instances:
        placement = instances[entity_name]["placement"]
        if placement is None:
            for g in ir["groups"]:
                if g["root"] == entity_name and g["placement"] is not None:
                    placement = g["placement"]
                    break
            if placement is None:
                raise ValueError(
                    f"_pose_in_local_frame: instance {entity_name!r} has "
                    f"null placement and no containing group provides one")
    elif entity_name in groups:
        placement = groups[entity_name]["placement"]
        if placement is None:
            # An unattached group's pose-in-frame equals its root
            # member's pose-in-frame.
            return _pose_in_local_frame(ir, catalog,
                                        groups[entity_name]["root"],
                                        ancestor_name)
    else:
        raise KeyError(entity_name)
    if placement["kind"] != "attach":
        raise ValueError(
            f"_pose_in_local_frame: expected an attach placement walking "
            f"from {entity_name!r} toward ancestor {ancestor_name!r}, got "
            f"{placement['kind']!r}")
    parent_name = placement["parent"]
    parent_pose = _pose_in_local_frame(ir, catalog, parent_name, ancestor_name)
    child_in_parent = _child_in_parent_via_anchors(ir, catalog, placement,
                                                    entity_name)
    return _rot.se3_compose(parent_pose, child_in_parent)


def _child_in_parent_via_anchors(ir, catalog, placement, child_name: str) -> tuple:
    """Child SE3 in parent's local frame, given an attach placement record.

    Each anchor's IR dict carries its name, translation (in entity-local),
    and rotation (anchor frame orientation). Composition:
        child_in_parent = pa_local ∘ inverse(ca_local)
    """
    parent_name = placement["parent"]
    pa_local = _anchor_local_pose(ir, catalog, parent_name,
                                  placement["parent_anchor"])
    ca_local = _anchor_local_pose(ir, catalog, child_name,
                                  placement["child_anchor"])
    ca_local_inv = _rot.se3_inverse(ca_local)
    return _rot.se3_compose(pa_local, ca_local_inv)


def _group_union_aabb_in_root_local(ir, catalog, group_record) -> tuple:
    """Union AABB of all members, expressed in the group root's local
    frame. Uses _pose_in_local_frame to walk the attach forest *within*
    the group without ever calling _resolve_world_pose.
    """
    root_name = group_record["root"]
    members = group_record["members"]
    instances = _index_instances(ir)
    groups = _index_groups(ir)

    # Root member's AABB in its own frame, at the state the scene committed
    # for it — a group of two cabinets, one placed open, has a union box that
    # contains the open door. Evaluation poses deliberately do not reach here
    # (the group's own 27 landmarks are canonical extents, like an instance's),
    # but `initial_joints` is static and is part of what was placed.
    if root_name in instances:
        root_aabb = _object_global_bbox(
            catalog, instances[root_name]["object_ref"],
            poses=instance_initial_joints(ir, root_name))
    else:
        root_aabb = _group_union_aabb_in_root_local(
            ir, catalog, groups[root_name])

    xs_min = [root_aabb[0][0]]; ys_min = [root_aabb[0][1]]; zs_min = [root_aabb[0][2]]
    xs_max = [root_aabb[1][0]]; ys_max = [root_aabb[1][1]]; zs_max = [root_aabb[1][2]]

    for m in members:
        if m == root_name:
            continue
        m_in_root = _pose_in_local_frame(ir, catalog, m, root_name)
        if m in instances:
            m_aabb = _object_global_bbox(
                catalog, instances[m]["object_ref"],
                poses=instance_initial_joints(ir, m))
        else:
            m_aabb = _group_union_aabb_in_root_local(ir, catalog, groups[m])
        for c in _aabb_corners(m_aabb):
            rx, ry, rz = _rot.quat_rotate_vec(m_in_root[1], c)
            wx = m_in_root[0][0] + rx
            wy = m_in_root[0][1] + ry
            wz = m_in_root[0][2] + rz
            xs_min.append(wx); ys_min.append(wy); zs_min.append(wz)
            xs_max.append(wx); ys_max.append(wy); zs_max.append(wz)

    return ((min(xs_min), min(ys_min), min(zs_min)),
            (max(xs_max), max(ys_max), max(zs_max)))


def _aabb_corners(aabb):
    (xmn, ymn, zmn), (xmx, ymx, zmx) = aabb
    return (
        (xmn, ymn, zmn), (xmx, ymn, zmn),
        (xmn, ymx, zmn), (xmx, ymx, zmn),
        (xmn, ymn, zmx), (xmx, ymn, zmx),
        (xmn, ymx, zmx), (xmx, ymx, zmx),
    )


def _aabb_anchor_position(aabb, anchor_name: str) -> tuple:
    """Position of an auto-AABB anchor in the AABB's local frame.

    Uses the shared 27-name vocabulary from `sdk/_anchors.py` (viewer-
    relative axis convention: +X=left, +Y=front, +Z=top).
    """
    return _anchors_mod.auto_aabb_anchor_position(aabb, anchor_name)


def _room_anchor_position(room: dict, anchor_name: str) -> tuple:
    """Position of an anchor on the room AABB, in world frame.

    The room is centered horizontally at the origin and sits on the
    floor: AABB = ((-w/2, -d/2, 0), (w/2, d/2, h)). Compass-style room
    anchor names (floor_center, wall_north_center, …) have been dropped
    in favor of the unified 27-name vocabulary.
    """
    w = room["width"]; d = room["depth"]; h = room["height"]
    room_aabb = ((-w / 2.0, -d / 2.0, 0.0), (w / 2.0, d / 2.0, h))
    return _anchors_mod.auto_aabb_anchor_position(room_aabb, anchor_name)


# ---------------------------------------------------------------------------
# Public pose / bbox queries
# ---------------------------------------------------------------------------


def _pose_to_api(pose) -> tuple:
    t, q = pose
    rx, ry, rz = _rot.quat_to_euler_xyz_deg(q)
    return ((t[0], t[1], t[2]), (rx, ry, rz))


def instance_pose(ir, catalog, name: str, poses=None, *, memo=None,
                  base_deltas=None, mount_deltas=None):
    """World pose of an instance.

    Returns ((tx, ty, tz), (rx_deg, ry_deg, rz_deg)).

    `poses` is an optional scene-level pose mapping; `None` (the default)
    resolves the attach forest at rest, the historical behavior. Passing
    it matters for evidence produced at a CCD witnessing pose — otherwise
    the reported pose describes rest-pose geometry.

    `base_deltas` applies whole-body rigid deltas — see `_resolve_world_pose`.

    `mount_deltas` carries the deformation channel, as every sibling query in
    this module does — `instance_world_aabb`, `instance_part_world_aabbs`,
    `instance_part_world_se3`. `_probes.animation_instance_pose_at` forwards it;
    dropping it here breaks that probe at the call, not at the type checker.

    `memo` lets a caller share the `_resolve_world_pose` memo across many
    queries; without it each call re-walks the attach chain from scratch.
    A shared memo is only valid for one `(poses, base_deltas, mount_deltas)`
    triple — never reuse one across animation frames.
    """
    if name not in {i["name"] for i in ir["instances"]}:
        raise _unknown_entity(ir, name, catalog=catalog)
    pose = _resolve_world_pose(ir, catalog, name,
                               {} if memo is None else memo, poses=poses,
                               base_deltas=base_deltas,
                               mount_deltas=mount_deltas)
    return _pose_to_api(pose)


def group_pose(ir, catalog, name: str):
    if name not in {g["name"] for g in ir["groups"]}:
        raise _unknown_entity(ir, name, catalog=catalog)
    pose = _resolve_world_pose(ir, catalog, name, {})
    return _pose_to_api(pose)


def instance_bbox(ir, catalog, name: str):
    """World-frame AABB of an instance at rest pose.

    The rest-pose case of `instance_world_aabb`, and delegated to it rather
    than recomputed: this had its own copy of the box math and inherited the
    same double axis-alignment, so `instance_bbox` and `instance_world_aabb(…,
    poses=None)` could report different boxes for the same resting instance.
    """
    if name not in _index_instances(ir):
        raise _unknown_entity(ir, name, catalog=catalog)
    return instance_world_aabb(ir, catalog, name)


def instance_part_world_aabbs(ir, catalog, name: str, poses=None, *,
                              memo=None, base_deltas=None, mount_deltas=None,
                              deform_bounds=None):
    """World-frame AABB of every part of an instance at the given pose.

    Part-granular analogue of `instance_bbox`: returns a list of
    `(part_name, ((xmin, ymin, zmin), (xmax, ymax, zmax)))`. The
    scene-layer SupportConnected graph uses these so a recessed/inset
    fixture grounds via a real part-to-part face contact instead of its
    whole-object union box, which buries the fixture inside the host's
    AABB and exposes no coplanar face.

    Each box is tight to the part's own oriented extent — see the loop below
    for what that replaced. `instance_world_aabb` is the union of these, so the
    instance-level and part-level answers cannot disagree.

    `memo` shares the `_resolve_world_pose` memo with a caller sweeping many
    instances at one pose, as `instance_world_aabb` does. Valid for one
    `(poses, base_deltas, mount_deltas)` tuple only.
    """
    instances = _index_instances(ir)
    if name not in instances:
        raise _unknown_entity(ir, name, catalog=catalog)
    inst = instances[name]
    obj_ref = inst["object_ref"]
    obj_ir = _object_ir_for(catalog, obj_ref)
    obj_poses = effective_object_poses(ir, name, poses)
    # `mount_deltas` here as well as in `_compute_frames` above: the frames
    # call places this instance's own parts, while this one resolves where the
    # instance *is* — and an instance attached to a part anchor of a DEFORMING
    # parent rides that deformation through `_resolve_placement`. Passing it to
    # only one of the two made that path unreachable from every public query.
    world_pose = _resolve_world_pose(ir, catalog, name,
                                     {} if memo is None else memo, poses=poses,
                                     base_deltas=base_deltas,
                                     mount_deltas=mount_deltas)
    # One forward-kinematics walk for the whole object, not one per part.
    # `global_bbox(part_name=...)` recomputes `_compute_frames` on every call
    # and nothing caches it, so calling it per part would cost P full FK walks
    # per invocation, in the hottest pure-Python function on the animation path
    # (`_contact_candidates` alone calls it twice per frame, and the room,
    # floor and prewarm paths all reach it). `_instance_part_obbs`
    # below already hoists it the same way.
    frames = _object_query._compute_frames(
        obj_ir, poses=obj_poses,
        mount_deltas=_object_mounts_for_instance(mount_deltas, name))
    # A deforming part's rest AABB is not a bound on it, so every consumer of
    # this function — the contact broad phase, the room walk, the floor and room
    # gap queries, the prewarm — would inherit a box its own geometry can leave.
    obj_bounds = _object_deform_bounds_for_instance(deform_bounds, name)
    out = []
    for p in obj_ir["parts"]:
        part_name = p["name"]
        # **One axis-alignment, not two.** Taking the part's AABB in the
        # *object root* frame and then rotating that box by the instance pose
        # would align twice, so a part turned inside its object and an instance
        # turned in the scene would pay the inflation twice (a rod whose true
        # world box is 0.10 x 1.00 m would come back as 1.10 x 1.10). Composing
        # the two transforms first and carrying the part's **local** box through
        # once is tight, and is the same composition `_instance_part_obbs` does
        # for the OBB, so the box and the oriented box bound the same body.
        local = _object_query._inflate_local_box(
            _object_query.local_bbox(obj_ir, part_name),
            _object_query._inflate_for(obj_bounds, part_name),
            margin=_object_query.collide_margin(p))
        out.append((part_name, _aabb_of_transformed_corners(
            _aabb_corners(local),
            _rot.se3_compose(world_pose, frames[part_name]))))
    return out


def group_bbox(ir, catalog, name: str):
    """World-frame union AABB of a group at rest pose."""
    groups = _index_groups(ir)
    if name not in groups:
        raise _unknown_entity(ir, name, catalog=catalog)
    g = groups[name]
    local_aabb = _group_union_aabb_in_root_local(ir, catalog, g)
    pose = _resolve_world_pose(ir, catalog, name, {})
    return _aabb_of_transformed_corners(_aabb_corners(local_aabb), pose)


def scene_bbox(ir, catalog):
    """World-frame AABB union over all instances at rest pose."""
    if not ir["instances"]:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    xs_min, ys_min, zs_min = [], [], []
    xs_max, ys_max, zs_max = [], [], []
    for inst in ir["instances"]:
        (xmn, ymn, zmn), (xmx, ymx, zmx) = instance_bbox(ir, catalog, inst["name"])
        xs_min.append(xmn); ys_min.append(ymn); zs_min.append(zmn)
        xs_max.append(xmx); ys_max.append(ymx); zs_max.append(zmx)
    return ((min(xs_min), min(ys_min), min(zs_min)),
            (max(xs_max), max(ys_max), max(zs_max)))


def _aabb_of_transformed_corners(corners, se3):
    t, q = se3
    xs, ys, zs = [], [], []
    for c in corners:
        rx, ry, rz = _rot.quat_rotate_vec(q, c)
        xs.append(t[0] + rx)
        ys.append(t[1] + ry)
        zs.append(t[2] + rz)
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


# ---------------------------------------------------------------------------
# Anchor world position
# ---------------------------------------------------------------------------


def anchor_position(ir, catalog, entity_name: str, anchor_name: str,
                    poses=None, anchor_offset=None, *, base_deltas=None,
                    mount_deltas=None):
    """World-frame position of an anchor.

    `poses` is an optional `{joint_name: q}` map scoped to the
    entity's owning object's joints. For groups, qualified keys
    `<instance>.<joint>` target a specific instance; unqualified
    keys apply to every member-instance whose object has that joint.
    Unrelated keys are silently ignored.

    `anchor_offset` is `((tx,ty,tz),(rx,ry,rz))` (or `None` for
    identity); only the translation contributes to the position.

    `mount_deltas` moves a part-scoped anchor whose part rode its parent's
    deformation — see `_anchor_local_position`, which also documents why the
    landmark is still taken on the rest AABB.
    """
    if entity_name == _ROOM_RESERVED_NAME:
        pos = _room_anchor_position(ir["room"], anchor_name)
        if anchor_offset is not None:
            (otx, oty, otz), _orot = anchor_offset
            pos = (pos[0] + otx, pos[1] + oty, pos[2] + otz)
        return pos
    instances = _index_instances(ir)
    groups = _index_groups(ir)
    if entity_name not in instances and entity_name not in groups:
        raise _unknown_entity(ir, entity_name, catalog=catalog)
    local_pos = _anchor_local_position(ir, catalog, entity_name, anchor_name,
                                       poses=poses,
                                       anchor_offset=anchor_offset,
                                       mount_deltas=mount_deltas)
    pose = _resolve_world_pose(ir, catalog, entity_name, {}, poses=poses,
                               base_deltas=base_deltas,
                               mount_deltas=mount_deltas)
    t, q = pose
    rx, ry, rz = _rot.quat_rotate_vec(q, local_pos)
    return (t[0] + rx, t[1] + ry, t[2] + rz)


def _outward_local(leaf: str, anchor_name: str) -> tuple:
    """`outward_direction(leaf)`, but saying which *kind* of no this is.

    A declared semantic anchor has an orientation and no outward face normal,
    which is a different answer from "you misspelled a landmark" and needs a
    different fix. `outward` means "the AABB face this landmark sits on points
    that way"; an authored `wall_mount` sits on no face, and deriving a normal
    for it would be inventing one — so this stays a refusal, matching
    `AnchorRef.outward()`'s refusal at build time. It names the query that
    does answer: `anchor_pose`, whose rotation is the frame the author
    declared. A bare `KeyError('door.sensor_mount')` would read as "no such
    anchor" for an anchor that resolves everywhere else in this module.
    """
    try:
        return _anchors_mod.outward_direction(leaf)
    except KeyError:
        raise _anchors_mod.UnknownAnchor(
            f"outward() is defined only for the built-in AABB landmarks, and "
            f"{anchor_name!r} resolves to {leaf!r}, which is not one. If it is "
            f"a declared semantic anchor, it carries a full frame rather than a "
            f"face normal: ask anchor_pose({anchor_name!r}) and read its "
            f"rotation. If it was meant to be a landmark, use 'center' or up to "
            f"one token per axis: x=left/right, y=front/back, z=top/bottom."
        ) from None


def anchor_pose(ir, catalog, entity_name: str, anchor_name: str,
                poses=None, anchor_offset=None, *, base_deltas=None,
                mount_deltas=None):
    """World-frame **frame** of an anchor: `((x,y,z),(rx,ry,rz))` degrees.

    `anchor_position`'s orientation-carrying sibling, and it exists because a
    declared semantic anchor is a frame rather than a point. `add_anchor`
    records a rotation, `Scene.attach` composes it, and the whole difference
    between mounting a sensor and *aiming* one lives in that rotation.
    `anchor_position` answers only about the point, and `outward` refuses a
    declared name (rightly — see below), so this is the query that shows
    whether a sensor attached by `body.mount` points the right way.

    For a built-in AABB landmark the rotation is the entity's own orientation
    (identity in entity-local), so this reduces to `anchor_position` plus the
    body's rotation, and `anchor_offset`'s rotation still shows up in it.

    Same channels and precedence as every sibling: `poses=None` resolves the
    scene's committed configuration.
    """
    record = {"name": anchor_name}
    if anchor_offset is not None:
        (tx, ty, tz), (rx, ry, rz) = anchor_offset
        q = _rot.sign_canonicalize_quat(_rot.normalize_quat(
            _rot.euler_xyz_deg_to_quat_wxyz(rx, ry, rz)))
        record["translation"] = [float(tx), float(ty), float(tz)]
        record["rotation"] = [float(v) for v in q]
    if entity_name == _ROOM_RESERVED_NAME:
        local = _anchor_local_pose(ir, catalog, entity_name, record)
        return _pose_to_api(local)
    instances = _index_instances(ir)
    groups = _index_groups(ir)
    if entity_name not in instances and entity_name not in groups:
        raise _unknown_entity(ir, entity_name, catalog=catalog)
    inst_poses = (effective_object_poses(ir, entity_name, poses)
                  if entity_name in instances else poses)
    local = _anchor_local_pose(
        ir, catalog, entity_name, record, poses=inst_poses,
        mount_deltas=_object_mounts_for_instance(mount_deltas, entity_name))
    world = _resolve_world_pose(ir, catalog, entity_name, {}, poses=poses,
                                base_deltas=base_deltas,
                                mount_deltas=mount_deltas)
    return _pose_to_api(_rot.se3_compose(world, local))


def outward(ir, catalog, entity_name: str, anchor_name: str, poses=None,
            anchor_offset=None, *, base_deltas=None, mount_deltas=None):
    """Unit outward direction of an anchor in world frame at rest pose.

    Resolves the entity-local outward direction (from sdk/_anchors.py)
    through the entity's rest-pose rotation in world. If `anchor_offset`
    carries a non-identity rotation, the entity-local outward is first
    rotated by it before transformation. The room's local frame
    coincides with the world frame so room anchors return their
    rotated-outward directions directly.

    `base_deltas` applies whole-body rigid deltas — see `_resolve_world_pose`.
    A delta's *rotation* turns the outward direction; its translation cannot,
    which is why this returns the same vector for a body that only slides.

    `mount_deltas` carries the deformation channel, as `anchor_position` does:
    the two resolve the same anchor and take the same channels.

    Raises ValueError for the `center` anchor (no outward direction).
    """
    # Part-scoped instance anchor (`<part>.<anchor>`): the outward
    # direction is the part-anchor outward, transformed through the
    # instance's world pose.
    if entity_name != _ROOM_RESERVED_NAME and "." in anchor_name:
        _part_name, _dot, sub = anchor_name.partition(".")
        local_dir = _outward_local(sub, anchor_name)
    else:
        local_dir = _outward_local(anchor_name, anchor_name)
    if anchor_offset is not None:
        _otrans, (rx, ry, rz) = anchor_offset
        if rx != 0.0 or ry != 0.0 or rz != 0.0:
            q_off = _rot.euler_xyz_deg_to_quat_wxyz(rx, ry, rz)
            q_off = _rot.sign_canonicalize_quat(_rot.normalize_quat(q_off))
            local_dir = _rot.quat_rotate_vec(q_off, local_dir)
    if entity_name == _ROOM_RESERVED_NAME:
        return local_dir
    pose = _resolve_world_pose(ir, catalog, entity_name, {}, poses=poses,
                               base_deltas=base_deltas,
                               mount_deltas=mount_deltas)
    _t, q = pose
    return _rot.quat_rotate_vec(q, local_dir)


def _object_poses_for_instance(scene_poses, instance_name: str):
    """Project scene-level poses to an instance's object-local joint poses.

    Scene `poses` keys are either qualified `"<instance>.<joint>"` (only
    applies to the matching instance) or unqualified `"<joint>"` (applies
    to any instance whose object has that joint). Qualified keys for
    other instances are filtered out; unrelated keys are silently
    forwarded — `_compute_frames` ignores joint names not on the ref's
    ancestor chain.

    **A qualified key beats a bare one regardless of insertion order.** Both
    spellings can name the same joint — `{"hinge": 0, "cab_left.hinge": 40}` is
    the ordinary way to say "all shut except this one" — and a single pass that
    wrote each key as it came would let `dict` ordering decide. Bare keys are
    laid down first and qualified ones overwrite them.

    Projection only: this does **not** apply an instance's saved
    `initial_joints`. Callers want `effective_object_poses`, which layers that
    underneath; this stays separate because a few call sites hold a pose dict
    that has already been projected and merged.

    Units are the IR's own (radians for revolute), like everything else in this
    module. A caller holding surface-unit poses converts first — see
    `_probes._scene_poses_to_internal`.
    """
    if not scene_poses:
        return None
    out: dict = {}
    qualified: dict = {}
    for key, val in scene_poses.items():
        if "." in key:
            inst_name, _, joint_name = key.partition(".")
            if inst_name == instance_name:
                qualified[joint_name] = val
        else:
            out[key] = val
    out.update(qualified)
    return out or None


def instance_initial_joints(ir, instance_name: str):
    """The instance's saved initial joint configuration, or `None`.

    A fresh dict in IR-internal units (radians for revolute), keyed by bare
    joint name — the `initial_joints` the author selected at `place` / `attach`
    time. `None` for an instance that selected none, for a group or room name,
    and for a name this scene does not have, so a caller can hand the result
    straight to `_compute_frames`.

    This is the scene's **committed** configuration, and it sits between two
    things it is not:

    * the object definition's `default=`, which is a property of the shared
      ObjectIR and identical for every instance placed from it; and
    * an evaluation override (`.at({...})`, `poses=`, an animation track),
      which asks a question about a configuration without changing what the
      scene committed.

    A fresh dict rather than the stored one because the stored one lives in a
    frozen IR that other queries read: `_compute_frames` does not mutate its
    `poses`, but a caller that adds a key to what it got back from here would
    otherwise be editing the scene.
    """
    if not ir:
        return None
    for inst in ir.get("instances") or ():
        if inst.get("name") == instance_name:
            saved = inst.get("initial_joints")
            return {k: float(v) for k, v in saved.items()} if saved else None
    return None


def merge_instance_initial(ir, instance_name: str, object_poses):
    """Layer an already-projected per-instance pose dict over saved state.

    `object_poses` is bare-keyed and in internal units — the output of
    `_object_poses_for_instance`, or a dict a caller projected and converted
    itself. Returns `None` when neither source has anything, so the result is
    still a valid `poses=` for `_compute_frames`.

    The merge direction is the whole contract: an explicit override wins, and
    every joint it does not name keeps its saved value. A caller that instead
    reached for `dict(saved, **overrides)`-by-hand in each of the eight places
    this is needed is how a partial override ends up resetting the other
    joints to the definition's defaults in one consumer and not another.
    """
    saved = instance_initial_joints(ir, instance_name)
    if not saved:
        return object_poses or None
    if not object_poses:
        return saved
    saved.update(object_poses)
    return saved


def effective_object_poses(ir, instance_name: str, scene_poses):
    """The configuration one instance is actually in, for `_compute_frames`.

    Scene-level `scene_poses` (bare and/or `"<instance>.<joint>"` keys, internal
    units) projected onto this instance and layered over its saved
    `initial_joints`. The single resolution path for the documented precedence,
    highest first:

      1. an instance-qualified evaluation override;
      2. a bare evaluation override (broadcast);
      3. the instance's saved `initial_joints`;
      4. the object definition's `default=` — reached by omission, inside
         `_object_kernel._child_in_parent`.

    `scene_poses=None` therefore means **the scene's committed state**, not the
    definition's defaults. That is the load-bearing consequence: every kernel
    query, verifier, renderer, and exporter that already passed no poses to ask
    "where is this scene at rest" now gets the state the author committed,
    without each of them learning about `initial_joints` separately.
    """
    return merge_instance_initial(
        ir, instance_name, _object_poses_for_instance(scene_poses,
                                                      instance_name))


def _object_mounts_for_instance(scene_mounts, instance_name: str):
    """Project scene-level mount deltas to one instance's object-local joints.

    Deliberately **not** the same contract as `_object_poses_for_instance`.
    That one forwards unqualified keys to every instance and lets
    `_compute_frames` ignore what does not apply, because a scene pose tag is
    meaningfully broadcast. Mount deltas are the opposite: each one is a
    measured consequence of one specific part of one specific instance
    deforming, so the scene-level dict is keyed by instance name and nothing is
    broadcast. Handing another instance's mounts to `_compute_frames` would
    trip its unknown-joint check, which is the intended outcome if a caller
    ever tries.
    """
    if not scene_mounts:
        return None
    per_instance = scene_mounts.get(instance_name)
    return dict(per_instance) if per_instance else None


def _object_deform_bounds_for_instance(scene_bounds, instance_name: str):
    """Project scene-level deformation bounds to one instance's parts.

    Same no-broadcast contract as `_object_mounts_for_instance`, and for the same
    reason: a displacement bound is a measured consequence of one part of one
    instance deforming, so it is meaningless on any other body. Two instances of
    the same object with only one of them rigged is the ordinary case.
    """
    if not scene_bounds:
        return None
    per_instance = scene_bounds.get(instance_name)
    return dict(per_instance) if per_instance else None


def instance_world_aabb(ir, catalog, instance_name: str, poses=None,
                        *, memo=None, base_deltas=None,
                        mount_deltas=None, deform_bounds=None) -> tuple:
    """World-frame axis-aligned bounding box of an instance.

    Computed as the union of the underlying object's parts transformed
    by the instance's world pose. Scene-level `poses` are projected to
    the instance's object-local joints via `_object_poses_for_instance`,
    so articulated instances expand their AABB under the named pose.

    `base_deltas` applies whole-body rigid deltas — see `_resolve_world_pose`.

    `memo` lets a caller share the `_resolve_world_pose` memo across many
    queries (e.g. one broad-phase sweep over every instance); without it
    each call re-walks the attach chain from scratch. A shared memo is only
    valid for one `(poses, base_deltas)` pair.
    """
    # **The union of the part boxes, not a box around a box.** A union in the
    # *object root* frame rotated into the world would axis-align twice and
    # union before rotating. Taking the union of `instance_part_world_aabbs` is
    # tight, and leaves one
    # derivation of "where is this instance" instead of two that drift.
    boxes = [box for _name, box in instance_part_world_aabbs(
        ir, catalog, instance_name, poses, memo=memo, base_deltas=base_deltas,
        mount_deltas=mount_deltas, deform_bounds=deform_bounds)]
    if not boxes:
        # An object with no parts cannot be placed, but the historical answer
        # here was a degenerate box at the instance's origin rather than a
        # raise, and some callers sweep every instance.
        world_pose = _resolve_world_pose(
            ir, catalog, instance_name, {} if memo is None else memo,
            poses=poses, base_deltas=base_deltas, mount_deltas=mount_deltas)
        return (tuple(world_pose[0]), tuple(world_pose[0]))
    return (tuple(min(b[0][k] for b in boxes) for k in range(3)),
            tuple(max(b[1][k] for b in boxes) for k in range(3)))


def group_world_aabb(ir, catalog, group_name: str, poses=None, *,
                     base_deltas=None, mount_deltas=None,
                     deform_bounds=None) -> tuple:
    """World-frame AABB of a group: union of its member instances'
    world AABBs.

    `base_deltas` applies whole-body rigid deltas — see `_resolve_world_pose`.
    A group is only a name for its members, so a delta on any member moves the
    union; there is no delta keyed on the group itself. `mount_deltas` and
    `deform_bounds` reach the members, since the union is `instance_world_aabb`
    over them.
    """
    groups = _index_groups(ir)
    if group_name not in groups:
        raise _unknown_entity(ir, group_name, catalog=catalog)
    g = groups[group_name]
    # Walk member instances and unify their world AABBs.
    return _group_world_union(ir, catalog, g, poses=poses,
                              base_deltas=base_deltas,
                              mount_deltas=mount_deltas,
                              deform_bounds=deform_bounds)


def room_world_aabb(ir) -> tuple:
    """Room's world AABB: [-w/2, w/2] x [-d/2, d/2] x [0, h]."""
    room = ir["room"]
    w = float(room["width"])
    d = float(room["depth"])
    h = float(room["height"])
    return ((-w / 2, -d / 2, 0.0), (w / 2, d / 2, h))


def _group_world_union(ir, catalog, group_record, poses=None, *,
                       base_deltas=None, mount_deltas=None,
                       deform_bounds=None):
    instances = _index_instances(ir)
    groups = _index_groups(ir)
    xs_min, ys_min, zs_min = [], [], []
    xs_max, ys_max, zs_max = [], [], []
    # `[*members, root]` rather than `members + [root]`: a probe reads the IR
    # through a `_readonly.freeze()` view, whose sequences are
    # `_ReadOnlySequence` and define no `__add__` — so the concatenation raised
    # `TypeError: unsupported operand type(s) for +` for every group query made
    # from sandboxed code, while the identical call inside a verifier worked.
    # Unpacking asks only for iteration, which both spellings support.
    for m in [*(group_record.get("members") or ()), group_record["root"]]:
        if m in instances:
            (lo, hi) = instance_world_aabb(ir, catalog, m, poses=poses,
                                           base_deltas=base_deltas,
                                           mount_deltas=mount_deltas,
                                           deform_bounds=deform_bounds)
        elif m in groups:
            (lo, hi) = _group_world_union(ir, catalog, groups[m], poses=poses,
                                          base_deltas=base_deltas,
                                          mount_deltas=mount_deltas,
                                          deform_bounds=deform_bounds)
        else:
            continue
        xs_min.append(lo[0]); ys_min.append(lo[1]); zs_min.append(lo[2])
        xs_max.append(hi[0]); ys_max.append(hi[1]); zs_max.append(hi[2])
    return ((min(xs_min), min(ys_min), min(zs_min)),
            (max(xs_max), max(ys_max), max(zs_max)))


def resolve_entity_aabb(ir, catalog, entity_dict):
    """Scene-layer entity-AABB resolver for `_eval_overlap`."""
    kind = entity_dict.get("kind")
    name = entity_dict["name"]
    poses = entity_dict.get("poses") or None
    if kind == "instance":
        return instance_world_aabb(ir, catalog, name, poses=poses)
    if kind == "group":
        return group_world_aabb(ir, catalog, name, poses=poses)
    if kind == "room":
        return room_world_aabb(ir)
    raise ValueError(
        f"scene layer cannot resolve EntityRef of kind {kind!r}; "
        f"expected 'instance', 'group', or 'room'"
    )


def resolve_direction(ir, catalog, direction_dict):
    """Scene-layer direction resolver for `_eval_angle`.

    Scene-layer joint_axis directions require an instance-scoped
    joint (named as `<instance>.<joint>` in `direction_dict["joint"]`).
    The kernel-side scoping is handled by `_relational._eval_angle`.
    """
    src = direction_dict["source"]
    poses = direction_dict.get("poses") or None
    if src == "world":
        return tuple(direction_dict["vec"])
    if src == "anchor_outward":
        anchor_offset = direction_dict.get("anchor_offset")
        return outward(ir, catalog,
                       direction_dict["entity"],
                       direction_dict["anchor"],
                       poses=poses,
                       anchor_offset=anchor_offset)
    if src == "joint_axis":
        # Joint name format: '<instance_name>.<joint_name>' (scene layer
        # requires qualification). For now, error if unqualified.
        joint_token = direction_dict["joint"]
        if "." not in joint_token:
            raise ValueError(
                "scene-layer joint_axis direction requires a qualified "
                f"name '<instance>.<joint>', got {joint_token!r}"
            )
        inst_name, _, joint_name = joint_token.partition(".")
        instances = _index_instances(ir)
        if inst_name not in instances:
            raise KeyError(inst_name)
        inst = instances[inst_name]
        # Resolve in the instance's underlying object frame, then
        # rotate through the instance's world pose.
        from . import _object_kernel as _oq
        obj_ir = _object_ir_for(catalog, inst["object_ref"])
        local_axis = _oq.joint_axis_world(obj_ir, joint_name, poses=poses)
        world_pose = _resolve_world_pose(ir, catalog, inst_name, {})
        _t, q = world_pose
        return _rot.quat_rotate_vec(q, local_axis)
    raise ValueError(f"unknown direction source {src!r}")


# ---------------------------------------------------------------------------
# Penetration (cross-instance, per-part SAT narrow phase)
# ---------------------------------------------------------------------------


def _instance_part_obbs(ir, catalog, name: str, memo: dict, poses=None,
                        base_deltas=None, mount_deltas=None,
                        deform_bounds=None):
    """World-frame OBBs of every part of an instance.

    Returns a list of `(part_name, obb)` tuples where `obb` is
    `((cx, cy, cz), (R0, R1, R2), (hx, hy, hz))` — the part's
    object-local-frame box transformed by the instance's world pose
    composed with the part's pose-aware frame in the object root.

    The OBB center is the composed translation plus the rotated
    part-local AABB center ``t + R·c``, mirroring
    ``_object_kernel.obb_of_part``. A ``box`` is origin-centered
    so ``c = 0`` and the center reduces to ``t``; a mesh authored off its local origin has ``c != 0``
    and would otherwise get a collider displaced by that offset — i.e.
    the scene layer would silently disagree with the object layer.
    """
    instances = _index_instances(ir)
    inst = instances[name]
    obj_ir = _object_ir_for(catalog, inst["object_ref"])
    obj_poses = effective_object_poses(ir, name, poses)
    frames = _object_query._compute_frames(
        obj_ir, poses=obj_poses,
        mount_deltas=_object_mounts_for_instance(mount_deltas, name))
    inst_pose = _resolve_world_pose(ir, catalog, name, memo, poses=poses,
                                    base_deltas=base_deltas,
                                    mount_deltas=mount_deltas)
    obj_bounds = _object_deform_bounds_for_instance(deform_bounds, name)
    out = []
    for p in obj_ir["parts"]:
        part_name = p["name"]
        # A deforming part's collider is its rest box grown by the displacement
        # bound: SAT on the rest shape of a part whose vertices have moved is a
        # broad phase that can miss, and this one gates the whole narrow phase.
        lo, hi = _object_query._inflate_local_box(
            _object_query.local_bbox(obj_ir, part_name),
            _object_query._inflate_for(obj_bounds, part_name))
        half = (0.5 * (hi[0] - lo[0]),
                0.5 * (hi[1] - lo[1]),
                0.5 * (hi[2] - lo[2]))
        c_local = (0.5 * (lo[0] + hi[0]),
                   0.5 * (lo[1] + hi[1]),
                   0.5 * (lo[2] + hi[2]))
        composed = _rot.se3_compose(inst_pose, frames[part_name])
        t = composed[0]
        q = composed[1]
        cw = _rot.quat_rotate_vec(q, c_local)  # R·c ; (0,0,0) for analytic prims
        c = (t[0] + cw[0], t[1] + cw[1], t[2] + cw[2])
        r0 = _rot.quat_rotate_vec(q, (1.0, 0.0, 0.0))
        r1 = _rot.quat_rotate_vec(q, (0.0, 1.0, 0.0))
        r2 = _rot.quat_rotate_vec(q, (0.0, 0.0, 1.0))
        out.append((part_name, ((c[0], c[1], c[2]), (r0, r1, r2), half)))
    return out


def penetration_part_pairs(ir, catalog, poses=None, *, base_deltas=None,
                           mount_deltas=None, deform_bounds=None):
    """All SAT-positive cross-instance *part pairs* at one scene pose.

    Per-part SAT: each instance's parts are converted to world-frame
    OBBs via the instance's world pose composed with each part's
    pose-aware object-root frame. A pair of instances is tested by
    first filtering with the instance world AABB (broad phase) and
    then running OBB-OBB SAT on every part-pair across the two instances.
    Every SAT-positive part pair is returned; callers that perform an exact
    narrow phase must use this API so a deep proxy false positive cannot hide
    a shallower real collision. `poses` keys may be qualified
    `"<instance>.<joint>"` or unqualified `"<joint>"`; only joints on
    each instance's underlying object affect that instance.
    Returns a list of `object_query.PenetrationPair` records sorted
    lexicographically by (a, b). Parent/child attach pairs are NOT
    excluded — an authored interpenetration is real signal.

    `base_deltas` adds whole-body rigid deltas (the dynamics layer's per-frame
    actor motion) — see `_resolve_world_pose`. `deform_bounds` grows the boxes
    and colliders of parts that are deforming, without which both phases test the
    rest shape of geometry that has moved. The memo below is built per call, so it
    is already scoped to this one `(poses, base_deltas, mount_deltas,
    deform_bounds)` tuple; an animation must call this once per frame rather than
    sharing a memo.

    """
    memo: dict = {}
    part_obbs = {}
    aabbs = {}
    for inst in ir["instances"]:
        name = inst["name"]
        part_obbs[name] = _instance_part_obbs(ir, catalog, name, memo,
                                              poses=poses,
                                              base_deltas=base_deltas,
                                              mount_deltas=mount_deltas,
                                              deform_bounds=deform_bounds)
        aabbs[name] = instance_world_aabb(ir, catalog, name, poses=poses,
                                          memo=memo, base_deltas=base_deltas,
                                          mount_deltas=mount_deltas,
                                          deform_bounds=deform_bounds)
    names = sorted(part_obbs.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if not _object_query._aabbs_overlap(aabbs[a], aabbs[b]):
                continue
            for pa_name, obb_a in part_obbs[a]:
                for pb_name, obb_b in part_obbs[b]:
                    res = _object_query.sat_obb_obb(obb_a, obb_b)
                    if not res.collided:
                        continue
                    pairs.append(_object_query.PenetrationPair(
                        a=a, b=b,
                        depth=-res.signed_distance,
                        contact_normal=res.axis,
                        contact_point=res.contact_point,
                        signed_distance=res.signed_distance,
                        a_frame=instance_pose(
                            ir, catalog, a, poses=poses, memo=memo,
                            base_deltas=base_deltas),
                        b_frame=instance_pose(
                            ir, catalog, b, poses=poses, memo=memo,
                            base_deltas=base_deltas),
                        a_bbox=(tuple(aabbs[a][0]), tuple(aabbs[a][1])),
                        b_bbox=(tuple(aabbs[b][0]), tuple(aabbs[b][1])),
                        witnessing_pose=dict(poses) if poses else None,
                        a_sub=pa_name,
                        b_sub=pb_name,
                    ))
    return pairs


def penetration_pairs(ir, catalog, poses=None, *, base_deltas=None,
                      mount_deltas=None, deform_bounds=None):
    """Compatibility instance-pair view of :func:`penetration_part_pairs`.

    Returns only the deepest SAT part pair for each instance pair, preserving
    the historical public probe contract. Exact verification must consume the
    raw API above and aggregate only after adjudication.
    """
    worst = {}
    for pair in penetration_part_pairs(ir, catalog, poses=poses,
                                       base_deltas=base_deltas,
                                       mount_deltas=mount_deltas,
                                       deform_bounds=deform_bounds):
        key = (pair.a, pair.b)
        existing = worst.get(key)
        if existing is None or pair.signed_distance < existing.signed_distance:
            worst[key] = pair
    return [worst[key] for key in sorted(worst)]


def instance_part_world_se3(ir, catalog, instance_name: str, part_name: str,
                            poses=None, *, memo=None, base_deltas=None,
                            mount_deltas=None):
    """World-frame SE3 of one part of one instance, in internal form.

    Same composition `_instance_part_obbs` does — instance world pose composed
    with the part's pose-aware frame in the object root — exposed on its own so
    the exact adjudicator can place a single part's geometry without rebuilding
    every OBB of the instance.
    """
    instances = _index_instances(ir)
    if instance_name not in instances:
        raise _unknown_entity(ir, instance_name, catalog=catalog)
    inst = instances[instance_name]
    obj_ir = _object_ir_for(catalog, inst["object_ref"])
    obj_poses = effective_object_poses(ir, instance_name, poses)
    frames = _object_query._compute_frames(
        obj_ir, poses=obj_poses,
        mount_deltas=_object_mounts_for_instance(mount_deltas, instance_name))
    if part_name not in frames:
        raise KeyError(f"{instance_name}.{part_name}")
    inst_pose = _resolve_world_pose(ir, catalog, instance_name,
                                    {} if memo is None else memo, poses=poses,
                                    base_deltas=base_deltas,
                                    mount_deltas=mount_deltas)
    return _rot.se3_compose(inst_pose, frames[part_name])


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def ir_snapshot(ir):
    """Identity passthrough. API-symmetry shim."""
    return ir
