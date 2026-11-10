"""Shared opaque ref types — AnchorRef, AssertionRef, DirectionRef.

These types are produced by builder methods (`Object.anchor` /
`Scene.anchor` / `joint.axis()` / `anchor.outward()` / `Direction.world(...)`)
and consumed by other builder methods and assertion validation. They carry
an `entity_kind` discriminator so the verifier kernel can route resolution
to the right layer.

Anchor offsets
--------------
An `AnchorRef` carries a translation + rotation offset in the entity's
local frame. The offset is `((tx,ty,tz),(rx,ry,rz))` with rotation in
intrinsic Euler XYZ degrees (the SDK's canonical surface form). Default
is identity — `obj.anchor(part, "front")` reproduces the prior pure-
landmark behavior. The offset can be set at construction
(`obj.anchor(part, "front", translation=(0.03,0,0))`) or chained via
`.offset_by(...)`. The 27-name vocabulary picks an AABB landmark; the
offset is applied in the entity's local frame on top.

For placement (joint `attach_*` / `Scene.attach`), parent_anchor and
child_anchor offsets fully determine the rest pose — the joint API no
longer carries an offset argument. For revolute / prismatic joints the
joint `axis` is interpreted in the parent_anchor's local frame, so a
non-identity rotation on parent_anchor rotates the joint axis with it.

For position-only assertion consumers (`distance` / `collinear` /
`overlap`), only the translation part of the offset is observable —
the rotation is allowed but does not affect a point query. For
direction-typed consumers via `anchor.outward()`, the anchor rotation
rotates the outward vector.

Pose tagging
------------
Every ref type supports `.at(poses)` returning a copy with joint pose
overrides applied. `poses` is a mapping `{JointRef|str: float}`. Pose
keys not on the ref's ancestor kinematic chain are silently ignored at
evaluation time — this lets agents define a single pose dict (e.g.
``open_microwave = {door: pi/2, latch: 0.05}``) and reuse it across
multiple refs whose chains include subsets of those joints.

Refs are hashable and equality-by-value (including poses and the offset
6-tuple). Each ref exposes a small set of public read-back properties
that the agent sandbox can reach — `name`, `kind`/`source`/`entity_*`,
`poses`, plus per-class additions like `AnchorRef.translation`/`.rotation`
or `DirectionRef.world_vec()`. All other state is stored under leading-
underscore slot names; the AST allowlist rejects reads on `_`-prefixed
attributes, keeping internals unreachable from sandboxed code.
"""

from __future__ import annotations

import math

from ._readonly import is_mapping, is_sequence


__all__ = [
    "AnchorRef",
    "AssertionRef",
    "DirectionRef",
    "Direction",
    "freeze_poses",
    "poses_to_dict",
    "IDENTITY_ANCHOR_OFFSET",
    "freeze_anchor_offset",
    "compose_anchor_offsets",
]


# ---------------------------------------------------------------------------
# Anchor offset normalization
# ---------------------------------------------------------------------------

# Identity offset in surface form: ((tx,ty,tz),(rx,ry,rz)) — degrees, intrinsic XYZ.
IDENTITY_ANCHOR_OFFSET = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _bad_ref_input(summary: str, *, reason: str, value, repairs):
    """A structured error for malformed author input to a ref operation.

    Structured rather than a bare `TypeError` / `ValueError`, which would reach
    the agent as a `SourceRuntimeError` carrying the interpreter's own words and
    no repair ("expected a DirectionRef, got tuple"). Structured, it arrives in
    the same shape as every other authoring error, so the fix travels with the
    complaint.

    `InvalidAnchor` because every caller here is an anchor-frame operation — an
    offset, a pose tag, an outward direction. Deliberately not a new class: a
    class earns its keep by selecting different advice, and the advice is on the
    error itself. `refs` is empty because this layer holds only the ref's own
    names, not the object that owns it; `compile` fills in the failing line.
    """
    from ._errors import InvalidAnchor
    return InvalidAnchor(
        summary,
        refs=[],
        evidence={"reason": reason, "value": repr(value)},
        suggested_repairs=list(repairs),
    )


def _check_finite_number(x, label: str):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise _bad_ref_input(
            f"{label} must be a number, got {type(x).__name__}",
            reason="not_a_number", value=x,
            repairs=[f"pass {label} as a float or int in meters/degrees"])
    f = float(x)
    if not math.isfinite(f):
        raise _bad_ref_input(
            f"{label} must be finite, got {x}",
            reason="not_finite", value=x,
            repairs=[
                f"{label} came out NaN or infinite, which means an arithmetic "
                f"error upstream — a division by a dimension that is zero is "
                f"the usual cause",
            ])
    return f


def freeze_anchor_offset(translation=None, rotation=None) -> tuple:
    """Normalize a (translation, rotation) pair into a canonical 6-tuple of floats.

    Accepts either component as None (= identity for that component) or as
    a 3-tuple of finite numbers. Returns ((tx,ty,tz),(rx,ry,rz)) with
    rotation in intrinsic Euler XYZ degrees. Component validation rejects
    bools, non-numbers, NaN, and inf.
    """
    if translation is None:
        t = (0.0, 0.0, 0.0)
    else:
        if not (is_sequence(translation) and len(translation) == 3):
            raise _bad_ref_input(
                f"translation must be 3 numbers, got {translation!r}",
                reason="translation_shape", value=translation,
                repairs=["pass translation=(dx, dy, dz) in meters, in the "
                         "anchor's local frame"])
        t = (
            _check_finite_number(translation[0], "translation[0]"),
            _check_finite_number(translation[1], "translation[1]"),
            _check_finite_number(translation[2], "translation[2]"),
        )
    if rotation is None:
        r = (0.0, 0.0, 0.0)
    else:
        if not (is_sequence(rotation) and len(rotation) == 3):
            raise _bad_ref_input(
                f"rotation must be 3 numbers, got {rotation!r}",
                reason="rotation_shape", value=rotation,
                repairs=["pass rotation=(rx, ry, rz) as intrinsic Euler XYZ "
                         "degrees"])
        r = (
            _check_finite_number(rotation[0], "rotation[0]"),
            _check_finite_number(rotation[1], "rotation[1]"),
            _check_finite_number(rotation[2], "rotation[2]"),
        )
    return (t, r)


def _is_identity_anchor_offset(offset) -> bool:
    (tx, ty, tz), (rx, ry, rz) = offset
    return tx == 0.0 and ty == 0.0 and tz == 0.0 and rx == 0.0 and ry == 0.0 and rz == 0.0


def _surface_offset_to_se3(offset):
    """Convert ((tx,ty,tz),(rx,ry,rz)) → ((tx,ty,tz),(qw,qx,qy,qz))."""
    from . import _rotation as _rot
    (tx, ty, tz), (rx, ry, rz) = offset
    q = _rot.euler_xyz_deg_to_quat_wxyz(float(rx), float(ry), float(rz))
    q = _rot.sign_canonicalize_quat(_rot.normalize_quat(q))
    return ((float(tx), float(ty), float(tz)), q)


def _se3_to_surface_offset(se3):
    """Convert ((tx,ty,tz),(qw,qx,qy,qz)) → ((tx,ty,tz),(rx,ry,rz))."""
    from . import _rotation as _rot
    t, q = se3
    rx, ry, rz = _rot.quat_to_euler_xyz_deg(q)
    return ((float(t[0]), float(t[1]), float(t[2])),
            (float(rx), float(ry), float(rz)))


def compose_anchor_offsets(base, override) -> tuple:
    """Compose two surface-form offsets as SE3s in the entity-local frame.

    Semantics: `base` is applied first (the anchor's current offset from
    the AABB landmark), then `override` is applied on top in the same
    entity-local frame — `T_new = T_base ∘ T_override`. Translation
    composes as `t_new = t_base + R_base · t_override`; rotation
    composes as `R_new = R_base · R_override` (right-multiplication).

    `.offset_by(translation=(dx, 0, 0))` therefore moves the anchor
    `dx` along the anchor's *current* x-axis (the x-axis after any
    prior rotation), which matches how successive frame transforms
    compose in robotics conventions.
    """
    from . import _rotation as _rot
    se3_base = _surface_offset_to_se3(base)
    se3_over = _surface_offset_to_se3(override)
    se3_new = _rot.se3_compose(se3_base, se3_over)
    return _se3_to_surface_offset(se3_new)


# ---------------------------------------------------------------------------
# Pose-dict normalization
# ---------------------------------------------------------------------------

def _normalize_pose_key(key) -> str:
    """Extract joint name from either a JointRef-like object or a str.

    Duck-typed against JointRef via the `_name` attribute to avoid
    circular imports with object_builder.
    """
    if isinstance(key, str):
        return key
    name = getattr(key, "_name", None)
    if isinstance(name, str) and getattr(key, "_id", None) is not None:
        return name
    raise _bad_ref_input(
        f"pose dict key must be a JointRef or joint-name str, got "
        f"{type(key).__name__}",
        reason="pose_key", value=key,
        repairs=["key the poses dict by the joint's name, or by the JointRef "
                 "that `attach_*` returned"])


def freeze_poses(poses) -> tuple:
    """Normalize a poses mapping to a hashable sorted tuple ((name, value), ...).

    Accepts None, {}, or a {JointRef|str: number} mapping. Values must
    be finite real numbers; booleans are rejected. Empty input returns ().
    """
    if poses is None:
        return ()
    if isinstance(poses, tuple):
        # Already frozen.
        return poses
    if not is_mapping(poses):
        raise _bad_ref_input(
            f"poses must be a dict, got {type(poses).__name__}",
            reason="poses_shape", value=poses,
            repairs=["pass poses as {joint: angle_or_distance, ...}"])
    if not poses:
        return ()
    out: dict[str, float] = {}
    for k, v in poses.items():
        kn = _normalize_pose_key(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise _bad_ref_input(
                f"pose value for {kn!r} must be a number, got "
                f"{type(v).__name__}",
                reason="pose_value", value=v,
                repairs=[f"give {kn!r} a number: degrees for a revolute "
                         f"joint, meters for a prismatic one"])
        f = float(v)
        if not math.isfinite(f):
            raise _bad_ref_input(
                f"pose value for {kn!r} must be finite, got {v}",
                reason="pose_value_not_finite", value=v,
                repairs=[f"{kn!r} came out NaN or infinite — check the "
                         f"arithmetic that produced it"])
        out[kn] = f
    return tuple(sorted(out.items()))


def poses_to_dict(frozen) -> dict:
    """Inverse of freeze_poses for IR serialization."""
    return {k: v for k, v in frozen}


def _merge_pose_overrides(existing: tuple, overrides) -> tuple:
    """Merge `overrides` into `existing`, per-key replacement on conflict."""
    merged = dict(existing)
    for k, v in freeze_poses(overrides):
        merged[k] = v
    return tuple(sorted(merged.items()))


# ---------------------------------------------------------------------------
# AnchorRef
# ---------------------------------------------------------------------------

class AnchorRef:
    """(entity_kind, entity_name, anchor_name, offset) bundle, optionally pose-tagged.

    `entity_kind`:
        "part"      — object-layer anchor (entity is a part name)
        "instance"  — scene-layer anchor on an Instance
        "group"     — scene-layer anchor on a Group
        "room"      — scene-layer anchor on the Room

    `offset` is `((tx,ty,tz),(rx,ry,rz))` applied in the entity's local
    frame on top of the named AABB landmark. Defaults to identity.

    **Transform direction: ANCHOR -> PART.** The SE3 this resolves to
    (`Object._anchor_to_se3`) is the anchor frame *expressed in* the part's
    local frame, so composing it maps anchor-local coordinates into part-local
    coordinates:

        p_part = T_anchor->part * p_anchor

    That follows from `_rotation.se3_compose(a, b)`, which maps b-frame points
    into a's parent frame. Verified: an anchor at t=(1,0,0) with R=Rz(90)
    carries anchor-local (1,0,0) to part-local (1,1,0).

    The direction matters because the joint formula uses the INVERSE of the
    child anchor (`rest = parent_anchor o inverse(child_anchor)`); reading the
    direction backwards makes that inverse look like a bug.

    Pose-tagged copies are produced via `.at({joint: q, ...})`.
    Offset-extended copies are produced via `.offset_by(...)`.
    """

    __slots__ = (
        "_entity_kind", "_entity_name", "_anchor_name", "_owner_id",
        "_poses", "_offset", "_entity_id",
    )

    def __init__(self, entity_kind: str, entity_name: str,
                 anchor_name: str, owner_id: int, poses=(),
                 offset=IDENTITY_ANCHOR_OFFSET, entity_id=None):
        self._entity_kind = entity_kind
        self._entity_name = entity_name
        self._anchor_name = anchor_name
        self._owner_id = owner_id
        # Content identity of the entity this anchor is bound to, when the
        # entity has one — today only `Scene.anchor(<definition>, ...)`, which
        # records the ObjectIR's hash. `_entity_name` is a *display* name and
        # two different definitions may share it, so `Scene.attach` checks the
        # child anchor against this instead: the anchor path was validated
        # against one definition's parts and declared frames, and silently
        # accepting a same-named different definition would resolve the anchor
        # on geometry it was never checked against. None everywhere else, so
        # including it in equality below narrows nothing that existed.
        self._entity_id = entity_id
        self._poses = freeze_poses(poses) if not isinstance(poses, tuple) else poses
        # offset is already canonical when produced by builder code; tolerate
        # surface-form input from internal callers.
        if offset is IDENTITY_ANCHOR_OFFSET:
            self._offset = IDENTITY_ANCHOR_OFFSET
        else:
            (t, r) = offset
            self._offset = ((float(t[0]), float(t[1]), float(t[2])),
                            (float(r[0]), float(r[1]), float(r[2])))

    @property
    def name(self) -> str:
        """The anchor's vocabulary name (e.g. `"top"`, `"front_right"`)."""
        return self._anchor_name

    @property
    def entity_name(self) -> str:
        """The author-supplied name of the part / instance / group / room."""
        return self._entity_name

    @property
    def entity_kind(self) -> str:
        """One of `"part"`, `"instance"`, `"group"`, `"room"`."""
        return self._entity_kind

    @property
    def poses(self) -> dict:
        """Pose overrides applied via `.at({...})`. Empty dict if untagged."""
        return poses_to_dict(self._poses)

    @property
    def translation(self) -> tuple:
        """The translation component of the anchor's offset, `(tx, ty, tz)`
        in the entity's local frame. `(0.0, 0.0, 0.0)` if no `translation=`
        was passed to `obj.anchor(...)`."""
        return self._offset[0]

    @property
    def rotation(self) -> tuple:
        """The rotation component of the anchor's offset, intrinsic Euler
        XYZ in degrees, `(rx, ry, rz)`. `(0.0, 0.0, 0.0)` if no `rotation=`
        was passed to `obj.anchor(...)`."""
        return self._offset[1]

    def at(self, poses) -> "AnchorRef":
        """Return a copy with pose overrides merged into existing poses."""
        return AnchorRef(
            self._entity_kind, self._entity_name, self._anchor_name,
            self._owner_id,
            poses=_merge_pose_overrides(self._poses, poses),
            offset=self._offset,
            entity_id=self._entity_id,
        )

    def offset_by(self, *, translation=None, rotation=None) -> "AnchorRef":
        """Return a copy with an additional offset composed onto this one.

        Composition is SE3-style in the entity-local frame:
            T_new = T_self ∘ T_override
        so `translation` is interpreted in this anchor's current
        (already-offset) local frame, and `rotation` post-multiplies the
        anchor's rotation. Chaining `.offset_by(...).offset_by(...)`
        composes as expected.

        If `translation` and `rotation` are both None, returns an
        equivalent copy.
        """
        override = freeze_anchor_offset(translation, rotation)
        new_offset = compose_anchor_offsets(self._offset, override)
        return AnchorRef(
            self._entity_kind, self._entity_name, self._anchor_name,
            self._owner_id,
            poses=self._poses,
            offset=new_offset,
            entity_id=self._entity_id,
        )

    def outward(self) -> "DirectionRef":
        """Return a DirectionRef for this anchor's outward direction.

        If the anchor carries a rotation offset, the outward vector is
        rotated by it (in the entity's local frame) before being
        transformed through the entity's world pose.
        """
        from . import _anchors as _anchors_mod
        # Scene anchors may be part-scoped: `tabletop.front` means the built-in
        # `front` landmark on the instance's `tabletop` part. Only the leaf is
        # a vocabulary name. Object-layer anchor names cannot contain a dot
        # (`_check_name` enforces an identifier), so splitting here can never
        # mistake a semantic anchor for a scoped path.
        leaf = self._anchor_name.rpartition(".")[2] or self._anchor_name
        if leaf == "center":
            raise _bad_ref_input(
                "anchor 'center' has no outward direction; outward() is "
                "only defined for the 26 face/edge/corner anchors",
                reason="outward_on_center", value="center",
                repairs=["take outward() from the face this should point out "
                         "of — e.g. anchor(part, 'top').outward() — or state "
                         "the direction with Direction.world((x, y, z))"])
        if leaf not in _anchors_mod.AUTO_AABB_ANCHOR_NAMES:
            # A semantic anchor HAS an orientation, but `outward()` means "the
            # AABB face normal this landmark sits on", which is meaningless for
            # an authored frame. Orientation-consuming assertions on semantic
            # anchors are deferred past M3; refusing here keeps that a visible
            # gap rather than a direction silently derived from nothing.
            raise _bad_ref_input(
                f"outward() is defined only for the built-in AABB anchors; "
                f"{self._anchor_name!r} resolves to {leaf!r}, which is a "
                f"semantic anchor",
                reason="outward_on_semantic_anchor", value=self._anchor_name,
                repairs=["`outward()` means 'the AABB face normal this "
                         "landmark sits on', which an authored frame does not "
                         "have",
                         "use joint.axis() for a joint's own direction, or "
                         "Direction.world((x, y, z)) to state one outright"])
        return DirectionRef(
            source="anchor_outward",
            owner_id=self._owner_id,
            data={
                "entity_kind": self._entity_kind,
                "entity": self._entity_name,
                "anchor": self._anchor_name,
                "anchor_offset": self._offset,
            },
            poses=self._poses,
        )

    def __eq__(self, other):
        return (isinstance(other, AnchorRef)
                and self._entity_kind == other._entity_kind
                and self._entity_name == other._entity_name
                and self._anchor_name == other._anchor_name
                and self._owner_id == other._owner_id
                and self._poses == other._poses
                and self._offset == other._offset
                and self._entity_id == other._entity_id)

    def __hash__(self):
        return hash(("AnchorRef", self._entity_kind, self._entity_name,
                     self._anchor_name, self._owner_id, self._poses,
                     self._offset, self._entity_id))

    def __repr__(self):
        base = (f"AnchorRef(entity={self._entity_name!r}, "
                f"anchor={self._anchor_name!r}")
        if not _is_identity_anchor_offset(self._offset):
            (tx, ty, tz), (rx, ry, rz) = self._offset
            base += f", translation=({tx}, {ty}, {tz})"
            if rx != 0.0 or ry != 0.0 or rz != 0.0:
                base += f", rotation=({rx}, {ry}, {rz})"
        if self._poses:
            base += f", poses={poses_to_dict(self._poses)}"
        return base + ")"


# ---------------------------------------------------------------------------
# AssertionRef
# ---------------------------------------------------------------------------

class AssertionRef:
    """Reference to an authored assertion. Identity = (owner, integer id)."""

    __slots__ = ("_id", "_owner_id", "_name", "_kind")

    def __init__(self, id_: int, owner_id: int, name: str, kind: str):
        self._id = id_
        self._owner_id = owner_id
        self._name = name
        self._kind = kind

    @property
    def name(self) -> str:
        """The author-supplied name passed to `assert_*` (including any sugar suffix)."""
        return self._name

    @property
    def kind(self) -> str:
        """The assertion record kind: `"distance"` / `"collinear"` /
        `"overlap"` / `"angle"` / `"axis_aligned"`.
        """
        return self._kind

    def __eq__(self, other):
        return (isinstance(other, AssertionRef)
                and self._id == other._id
                and self._owner_id == other._owner_id)

    def __hash__(self):
        return hash(("AssertionRef", self._id, self._owner_id))

    def __repr__(self):
        return (f"AssertionRef(name={self._name!r}, "
                f"id={self._id}, kind={self._kind!r})")


# ---------------------------------------------------------------------------
# DirectionRef
# ---------------------------------------------------------------------------

# Sentinel owner-id used for literal world directions (no owning builder).
WORLD_DIRECTION_OWNER_ID = -1


class DirectionRef:
    """A pose-taggable unit-direction reference. Sources:

        * "joint_axis"     — world-frame axis of a revolute or prismatic joint
                             (data: {"joint": <joint_name>})
        * "anchor_outward" — world-frame outward direction of an anchor
                             (data: {"entity_kind", "entity", "anchor"})
        * "world"          — literal world-frame unit vector
                             (data: {"vec": (ux, uy, uz)}, owner_id=-1)

    Construct via `joint.axis()`, `anchor.outward()`, or
    `Direction.world((x, y, z))` (also `Direction.world_x()` / etc.).
    """

    _VALID_SOURCES = frozenset({"joint_axis", "anchor_outward", "world"})

    __slots__ = ("_source", "_owner_id", "_data", "_poses")

    def __init__(self, *, source: str, owner_id: int, data: dict, poses=()):
        if source not in DirectionRef._VALID_SOURCES:
            raise ValueError(f"unknown DirectionRef source {source!r}")
        self._source = source
        self._owner_id = owner_id
        # Defensive copy so callers can't mutate the stored data dict.
        self._data = dict(data)
        self._poses = freeze_poses(poses) if not isinstance(poses, tuple) else poses

    @property
    def source(self) -> str:
        """One of `"joint_axis"` / `"anchor_outward"` / `"world"`."""
        return self._source

    @property
    def poses(self) -> dict:
        """Pose overrides applied via `.at({...})`. Empty dict if untagged."""
        return poses_to_dict(self._poses)

    def world_vec(self) -> tuple:
        """Return the unit world vector `(ux, uy, uz)` for a `Direction.world(...)`
        literal. Raises `ValueError` for `joint_axis` / `anchor_outward` sources —
        those resolve only against a finalized IR; query them with the
        `joint_axis_world(ir, joint_name, poses=...)` or
        `outward(ir, entity_name, anchor_name, poses=...)` probes after
        `compile()`.
        """
        if self._source != "world":
            raise _bad_ref_input(
                f"world_vec() is only defined for Direction.world(...) "
                f"literals; this DirectionRef has source={self._source!r}",
                reason="world_vec_source", value=self._source,
                repairs=["for a joint_axis / anchor_outward direction, read it "
                         "with the `joint_axis_world` / `outward` probes on a "
                         "finalized IR"])
        return tuple(self._data["vec"])

    def at(self, poses) -> "DirectionRef":
        """Return a copy with pose overrides merged into existing poses."""
        return DirectionRef(
            source=self._source,
            owner_id=self._owner_id,
            data=self._data,
            poses=_merge_pose_overrides(self._poses, poses),
        )

    def __eq__(self, other):
        return (isinstance(other, DirectionRef)
                and self._source == other._source
                and self._owner_id == other._owner_id
                and self._data == other._data
                and self._poses == other._poses)

    def __hash__(self):
        return hash(("DirectionRef", self._source, self._owner_id,
                     tuple(sorted(self._data.items())), self._poses))

    def __repr__(self):
        return (f"DirectionRef(source={self._source!r}, "
                f"data={self._data!r}"
                + (f", poses={poses_to_dict(self._poses)}"
                   if self._poses else "")
                + ")")


class Direction:
    """Factory for literal world-frame direction refs.

    Joint-axis and anchor-outward direction refs are built via
    `joint.axis()` and `anchor.outward()` on their owning builder's
    refs. Use this class only for world-axis-aligned directions.
    """

    @staticmethod
    def world(vec) -> DirectionRef:
        """Return a unit-normalized world-frame direction ref.

        `vec` is 3 numbers, or an axis name — `"z"`, `"-y"`. This is the third
        place in the SDK that spells a world axis (joint `axis=`, assertion
        `axis=`, here), and they share one vocabulary
        (`_assertions.AXIS_LITERALS`) so a spelling learned at one is not
        refused at another.
        """
        from . import _assertions as _asrt
        from ._errors import InvalidAnchor

        if isinstance(vec, str):
            unit = _asrt.AXIS_LITERALS.get(vec.strip().lower())
            if unit is None:
                raise _bad_ref_input(
                    f"Direction.world({vec!r}): unknown axis name",
                    reason="axis_name", value=vec,
                    repairs=["name a world axis as 'x', 'y', 'z', or a signed "
                             "form like '-z', or pass (x, y, z)"])
            vec = unit
        if not (is_sequence(vec) and len(vec) == 3):
            raise _bad_ref_input(
                f"Direction.world(vec): vec must be 3 numbers or an axis "
                f"name, got {vec!r}",
                reason="direction_shape", value=vec,
                repairs=["pass Direction.world((x, y, z)), or name an axis: "
                         "Direction.world('z')"])
        x, y, z = vec
        for label, c in (("x", x), ("y", y), ("z", z)):
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                raise _bad_ref_input(
                    f"Direction.world(vec): {label}-component must be a "
                    f"number, got {type(c).__name__}",
                    reason="direction_component", value=c,
                    repairs=["every component must be a float or int"])
            if not math.isfinite(float(c)):
                raise _bad_ref_input(
                    f"Direction.world(vec): {label}-component must be "
                    f"finite, got {c}",
                    reason="direction_not_finite", value=c,
                    repairs=["a NaN or infinite component means an arithmetic "
                             "error upstream"])
        norm = math.sqrt(float(x) * float(x) + float(y) * float(y)
                         + float(z) * float(z))
        if norm < 1.0e-12:
            raise _bad_ref_input(
                f"Direction.world(vec): vec norm too small ({norm})",
                reason="direction_zero", value=list(vec),
                repairs=["pass a non-zero direction"])
        u = (float(x) / norm, float(y) / norm, float(z) / norm)
        return DirectionRef(
            source="world",
            owner_id=WORLD_DIRECTION_OWNER_ID,
            data={"vec": u},
        )

    @staticmethod
    def world_x() -> DirectionRef:
        return Direction.world((1.0, 0.0, 0.0))

    @staticmethod
    def world_y() -> DirectionRef:
        return Direction.world((0.0, 1.0, 0.0))

    @staticmethod
    def world_z() -> DirectionRef:
        return Direction.world((0.0, 0.0, 1.0))
