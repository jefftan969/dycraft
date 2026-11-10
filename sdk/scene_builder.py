"""Scene builder for the DyCraft scene SDK.

This module exposes the single global `Scene` that the sandbox injects, plus
the LLM-facing `ObjectHandle` wrapper around upstream ObjectIR dicts, plus the
opaque ref types returned by builder methods.

Constant-time validations raise at the offending call. Structural
(O(n)) checks fire inside `finalize()`.
"""

from __future__ import annotations

import math
import re
import threading

from collections.abc import Mapping

from . import _rotation as _rot
from . import _anchors as _anchors_mod
from . import _assertions as _assertions
from . import _sugar as _sugar
from . import _object_kernel as _object_query
from ._readonly import freeze as _freeze
from ._material import (
    Material,
    check_material,
    DEFAULT_FLOOR_COLOR,
    DEFAULT_WALL_COLOR,
)
from ._refs import AnchorRef, AssertionRef
from .ir import SCENE_SCHEMA_VERSION, SCENE_SCHEMA_VERSION_V3, insert_id
from ._errors import (
    BuildError,
    InvalidScene,
    InvalidRoom,
    RoomNotDeclared,
    NoInstances,
    InvalidInstance,
    InvalidGroup,
    InvalidAnchor,
    InvalidAssertion,
    InvalidJoint,
)
from . import _readonly as _ro


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_MAX_LEN = 64
_AXIS_MIN_NORM = 1e-12

_ROOM_RESERVED_NAME = "room"

# Default room materials. Color defaults imported from sdk._material; the
# scalar PBR fields default to the Material class defaults (roughness=0.9,
# metallic=0.0, alpha=1.0), preserving the renderer's prior matte look.
DEFAULT_FLOOR_MATERIAL = Material(color=DEFAULT_FLOOR_COLOR)
DEFAULT_WALL_MATERIAL = Material(color=DEFAULT_WALL_COLOR)

#: The six surfaces of the default room shell, by their canonical names.
#: Axis-signed rather than viewer-relative because these name geometry, and
#: geometry does not turn around when the camera does.
ROOM_SURFACES = ("floor", "ceiling",
                 "wall_xmin", "wall_xmax", "wall_ymin", "wall_ymax")

#: How much of the shell a room actually has.
#:
#: **An extent is not an enclosure**, and conflating the two is what turned an
#: outdoor scene into a tunnel. `set_room` declares how big the world is and
#: where the ground is; whether that world has a lid is a separate fact. A
#: courtyard, an alley, a street, and a garden all have a ground plane, a
#: bounded extent and no ceiling, and one of them (a hutong alley 4.9 x 18.4 x
#: 5.5 m) was measured at 3.3/255 mean luminance once the shell sealed it —
#: the world dome was shut out by a ceiling the author never asked for.
#:
#:   "interior"  all six surfaces. A room.
#:   "open_top"  floor and four walls, no ceiling. A courtyard, a light well.
#:   "open_air"  floor only. A street or a garden: the extent still frames the
#:               scene and the ground still catches shadows, but nothing
#:               encloses it and the sky reaches everything.
ROOM_ENCLOSURES = ("interior", "open_top", "open_air")
DEFAULT_ROOM_ENCLOSURE = "interior"

#: Which shell surfaces each enclosure drops. The renderer reads this through
#: `room_surfaces_suppressed_by`, so the SDK and both backends cannot disagree
#: about what "open_top" means.
_ENCLOSURE_SUPPRESSES = {
    "interior": (),
    "open_top": ("ceiling",),
    "open_air": ("ceiling", "wall_xmin", "wall_xmax", "wall_ymin", "wall_ymax"),
}


def room_surfaces_suppressed_by(enclosure: str | None) -> tuple:
    """Shell surfaces an enclosure removes. Unknown or None -> none removed."""
    return _ENCLOSURE_SUPPRESSES.get(enclosure or DEFAULT_ROOM_ENCLOSURE, ())


#: Viewer-relative aliases, accepted because the anchor vocabulary an author
#: already uses is viewer-relative (`back` = -Y, `front` = +Y, `left` = +X,
#: `right` = -X) and forcing two conventions on one scene invites silent
#: mistakes. Normalized on the way in, so the IR only ever carries the
#: canonical names.
_ROOM_SURFACE_ALIASES = {
    "back": "wall_ymin",
    "front": "wall_ymax",
    "left": "wall_xmax",
    "right": "wall_xmin",
}


def _check_room_surface(value, room_refs) -> str:
    """Normalize and validate one room-surface name.

    **Replacement is declared, never inferred.** Deciding that an authored box
    "is" the north wall because it happens to overlap where the shell would go
    would be a guess that fails silently in both directions: a bookshelf
    standing flat against a wall would delete it, and a wall modeled 20 mm
    proud would leave the shell standing behind it. So the scene names the
    surface it has taken over, and takes responsibility for it.
    """
    if not isinstance(value, str):
        raise InvalidRoom(
            f"room surface must be a string, got {type(value).__name__}",
            refs=list(room_refs),
            evidence={"param": "surface", "value": type(value).__name__,
                      "known": list(ROOM_SURFACES)},
            suggested_repairs=[f"one of {', '.join(ROOM_SURFACES)}"])
    name = value.strip().lower()
    name = _ROOM_SURFACE_ALIASES.get(name, name)
    if name not in ROOM_SURFACES:
        raise InvalidRoom(
            f"unknown room surface {value!r}",
            refs=list(room_refs),
            evidence={"param": "surface", "value": value,
                      "known": list(ROOM_SURFACES),
                      "aliases": sorted(_ROOM_SURFACE_ALIASES)},
            suggested_repairs=[
                f"use one of {', '.join(ROOM_SURFACES)}, or a viewer-relative "
                f"alias ({', '.join(sorted(_ROOM_SURFACE_ALIASES))})"])
    return name


# Auto-AABB anchor vocabulary on instance / group / room AABBs. The
# unified 27-name vocabulary is shared with the object SDK; viewer-relative
# axis convention (+X=left, +Y=front, +Z=top). See sdk/_anchors.py.
_AABB_ANCHOR_NAMES = _anchors_mod.AUTO_AABB_ANCHOR_NAMES


# One Scene per sandbox execution. The agent layer is expected to reset
# this counter (call _reset_scene_singleton) between sandbox runs.
_scene_singleton_lock = threading.Lock()
_scene_singleton_constructed = False


def _reset_scene_singleton():
    """Test / host hook: clear the one-Scene-per-sandbox guard."""
    global _scene_singleton_constructed
    with _scene_singleton_lock:
        _scene_singleton_constructed = False


# ---------------------------------------------------------------------------
# Ref-dict helpers (for structured error refs)
# ---------------------------------------------------------------------------


def _ref(kind: str, name=None, id_=None) -> dict:
    return {"kind": kind, "name": name, "id": id_}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_name(name, kind_label: str, error_cls, *, refs):
    if not isinstance(name, str):
        raise error_cls(
            f"{kind_label} name must be a string, got {type(name).__name__}",
            refs=refs,
            evidence={"param": f"{kind_label}.name", "value": repr(name),
                      "constraint": "type:str"},
            suggested_repairs=[f"pass a string for the {kind_label} name"],
        )
    if len(name) == 0 or len(name) > _NAME_MAX_LEN:
        raise error_cls(
            f"{kind_label} name length must be 1..{_NAME_MAX_LEN}, got {len(name)}",
            refs=refs,
            evidence={"param": f"{kind_label}.name", "value": name,
                      "constraint": f"length in 1..{_NAME_MAX_LEN}"},
            suggested_repairs=["shorten the name to 64 chars or fewer"],
        )
    if not _NAME_RE.match(name):
        raise error_cls(
            f"{kind_label} name {name!r} does not match {_NAME_RE.pattern}",
            refs=refs,
            evidence={"param": f"{kind_label}.name", "value": name,
                      "constraint": f"regex {_NAME_RE.pattern}"},
            suggested_repairs=["start name with a letter or underscore; "
                               "use only letters, digits, underscores"],
        )


def _check_finite_scalar(x, label: str, error_cls, *, refs):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise error_cls(
            f"{label} must be a number, got {type(x).__name__}",
            refs=refs,
            evidence={"param": label, "value": repr(x), "constraint": "type:number"},
        )
    if not math.isfinite(float(x)):
        raise error_cls(
            f"{label} must be finite, got {x}",
            refs=refs,
            evidence={"param": label, "value": x, "constraint": "finite"},
        )


def _check_vec3(v, label: str, error_cls, *, refs):
    if not (isinstance(v, tuple) and len(v) == 3):
        raise error_cls(
            f"{label} must be a 3-tuple, got {v!r}",
            refs=refs,
            evidence={"param": label, "value": repr(v), "constraint": "3-tuple"},
        )
    for i, c in enumerate(v):
        _check_finite_scalar(c, f"{label}[{i}]", error_cls, refs=refs)


def _check_se3(se3, label: str, error_cls, *, refs):
    if not (isinstance(se3, tuple) and len(se3) == 2):
        raise error_cls(
            f"{label} must be a 2-tuple ((tx,ty,tz),(rx,ry,rz)), got {se3!r}",
            refs=refs,
            evidence={"param": label, "value": repr(se3),
                      "constraint": "((tx,ty,tz),(rx_deg,ry_deg,rz_deg))"},
        )
    _check_vec3(se3[0], f"{label}.translation", error_cls, refs=refs)
    _check_vec3(se3[1], f"{label}.rotation_euler_deg", error_cls, refs=refs)


def _parts_declaring(part_anchors: dict, anchor_name: str) -> list:
    """Which parts declare `anchor_name`, in the object's part order.

    Powers the "you forgot to part-qualify it" branch below. The unqualified
    spelling is the common mistake: `wall_mount` is a real frame the author has
    in mind, and `scene.anchor(inst, "wall_mount")` would otherwise report
    `unknown instance anchor 'wall_mount'`, sending them to look for a name
    that exists. Returns a list rather than a single hit
    because part-qualification is precisely what distinguishes a `mount` on the
    body from a `mount` on the lid, and naming both is the answer.
    """
    return [part for part, names in part_anchors.items() if anchor_name in names]


def _raise_unqualified_declared_anchor(anchor_name, parts, refs, *, label,
                                       where: str):
    """Refuse a bare declared-anchor name, telling the author to qualify it."""
    prefix = f"{label}: " if label else ""
    qualified = [f"{p}.{anchor_name}" for p in parts]
    raise InvalidAnchor(
        f"{prefix}anchor {anchor_name!r} on {where} is a **declared** frame "
        f"and must be part-qualified: write "
        f"{' or '.join(repr(q) for q in qualified)}",
        refs=refs,
        evidence={"reason": "declared_anchor_not_part_qualified",
                  "value": anchor_name, "declared_on": list(parts),
                  "qualified_forms": qualified},
        suggested_repairs=[
            f"use {qualified[0]!r}"
            + (f"; the name is declared on {len(parts)} parts, and the "
               f"part decides which frame you mean" if len(parts) > 1 else ""),
        ],
    )


def _raise_unknown_part_anchor(anchor_name, part_name, sub, declared, *,
                               refs, where: str, label):
    """Refuse a `<part>.<anchor>` path, naming **both** vocabularies.

    The two are genuinely separate namespaces — the 27 built-in AABB landmarks
    that exist on every part, and whatever the object's author registered with
    `Object.add_anchor` — and a message that named only the built-ins is what
    made a shipped contact frame look unsupported rather than misspelled. So
    this lists the part's declared names when it has any, and says plainly that
    it has none when it does not, before falling back to the built-in advice.
    """
    prefix = f"{label}: " if label else ""
    if declared:
        detail = (f"part {part_name!r} declares "
                  f"{', '.join(sorted(declared))}")
        repairs = [
            f"use one of {part_name!r}'s declared anchors: "
            f"{', '.join(sorted(declared))}",
            "or an auto-AABB landmark (top/bottom/left/right/front/back/"
            "center or an edge/corner name)",
        ]
    else:
        detail = f"part {part_name!r} declares no semantic anchors"
        repairs = [
            "use an auto-AABB landmark (top/bottom/left/right/front/back/"
            "center or an edge/corner name)",
            f"or register the frame in the object definition with "
            f"obj.add_anchor(<{part_name}>, {sub!r}, translation=..., "
            f"rotation=...)",
        ]
    raise InvalidAnchor(
        f"{prefix}unknown anchor {sub!r} in part-scoped path "
        f"{anchor_name!r} on {where}: it is neither one of the "
        f"{len(_AABB_ANCHOR_NAMES)} built-in AABB landmarks nor a semantic "
        f"anchor on that part — {detail}",
        refs=refs,
        evidence={"entity_kind": "instance", "value": anchor_name,
                  "part": part_name, "anchor": sub,
                  "declared_anchors": sorted(declared),
                  "builtin_names": _anchors_mod.builtin_anchor_groups()},
        suggested_repairs=repairs,
    )


def _handle_joint_table(handle) -> dict:
    """`{joint_name: (type, lo_internal, hi_internal)}` for an ObjectHandle.

    Straight off the wrapped ObjectIR, in the IR's own units, because the one
    caller (`_normalize_initial_joints`) compares a converted value against
    them. `ObjectHandle.joint_table()` reports the same facts in surface units
    for the author; going through it here would convert twice.
    """
    out: dict = {}
    for j in handle._ir.get("joints", []) or ():
        limits = j.get("limits") or (float("-inf"), float("inf"))
        out[j["name"]] = (j.get("type"), float(limits[0]), float(limits[1]))
    return out


def _handle_part_anchors(handle) -> dict:
    """`{part_name: frozenset(declared anchor names)}` for an ObjectHandle.

    Parts with no declared anchors get an empty frozenset rather than being
    omitted, so a validator can tell "this part has none" from "no such part".
    """
    out: dict = {}
    for part in handle._ir.get("parts", []) or ():
        names = {a["name"] for a in (part.get("anchors") or ())
                 if isinstance(a, dict) and a.get("name")}
        out[part["name"]] = frozenset(names)
    return out


def _normalize_initial_joints(state, joint_table: dict, *,
                              instance_name: str, refs) -> dict:
    """Validate an author's `initial_joints=` mapping into IR-internal units.

    `joint_table` is `{joint_name: (joint_type, lo_internal, hi_internal)}` for
    the instance's definition. Returns a fresh `{joint_name: value}` dict in
    **internal** units (radians for revolute, meters for prismatic), or `{}`
    for empty/omitted input.

    **A copy, and the conversion happens here once.** The author's dict is read
    and discarded, so mutating it afterwards cannot reach the committed scene;
    and the degrees-to-radians step lives at this one boundary rather than at
    each of the eight consumers that later read the stored value, which is the
    same discipline `_rotation.convert_poses_to_internal` enforces for every
    other pose dict crossing into the IR.

    Rejected, each with the joint named: an unknown joint, a `fixed` joint
    (there is no value to select), a qualified `"<inst>.<joint>"` key (this
    mapping is already scoped to one instance, so a dotted key means the author
    reached for the evaluation-override spelling), a non-number or non-finite
    value, and a value outside the joint's authored limits — the same strict
    range rule `Object.attach_revolute` applies to `default=`.
    """
    if state is None:
        return {}
    if not isinstance(state, Mapping):
        raise InvalidJoint(
            f"initial_joints must be a mapping of joint name to value, got "
            f"{type(state).__name__}",
            refs=refs,
            evidence={"reason": "initial_joints_shape",
                      "param": "initial_joints",
                      "value": repr(state)},
            suggested_repairs=[
                'pass initial_joints={"door_hinge": 40} — degrees for a '
                'revolute joint, meters for a prismatic one',
            ],
        )
    if not state:
        return {}
    from ._refs import _normalize_pose_key
    out: dict = {}
    for raw_key, raw_val in state.items():
        try:
            jname = _normalize_pose_key(raw_key)
        except Exception:
            raise InvalidJoint(
                f"initial_joints key must be a joint name or JointRef, got "
                f"{type(raw_key).__name__}",
                refs=refs,
                evidence={"reason": "initial_joints_key_type",
                          "value": repr(raw_key)},
                suggested_repairs=[
                    "key initial_joints by the joint's name, or by the "
                    "JointRef that attach_revolute / attach_prismatic returned",
                ],
            ) from None
        if "." in jname:
            raise InvalidJoint(
                f"initial_joints key {jname!r} is instance-qualified; "
                f"initial_joints is already scoped to instance "
                f"{instance_name!r}",
                refs=refs,
                evidence={"reason": "initial_joints_qualified_key",
                          "value": jname, "instance": instance_name},
                suggested_repairs=[
                    f'use the bare joint name: '
                    f'{jname.partition(".")[2]!r}',
                    "the '<instance>.<joint>' spelling is for evaluation "
                    "overrides (.at({...}) / poses=), not for saved state",
                ],
            )
        if jname not in joint_table:
            known = sorted(joint_table)
            raise InvalidJoint(
                f"initial_joints names joint {jname!r}, which object "
                f"{instance_name!r} does not have",
                refs=refs,
                evidence={"reason": "initial_joints_unknown_joint",
                          "value": jname, "instance": instance_name,
                          "known": known},
                suggested_repairs=[
                    f"joints on this definition: {', '.join(known) or '(none)'}",
                ],
            )
        jtype, lo, hi = joint_table[jname]
        if jtype == "fixed":
            raise InvalidJoint(
                f"initial_joints names {jname!r}, which is a fixed joint and "
                f"has no configuration to select",
                refs=refs,
                evidence={"reason": "initial_joints_fixed_joint",
                          "value": jname, "instance": instance_name},
                suggested_repairs=[
                    "drop it, or make the joint revolute / prismatic in the "
                    "object definition if it is meant to move",
                ],
            )
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise InvalidJoint(
                f"initial_joints[{jname!r}] must be a number, got "
                f"{type(raw_val).__name__}",
                refs=refs,
                evidence={"reason": "initial_joints_value_type",
                          "param": f"initial_joints[{jname}]",
                          "value": repr(raw_val)},
                suggested_repairs=[
                    f"give {jname!r} a number: degrees for a revolute joint, "
                    f"meters for a prismatic one",
                ],
            )
        val = float(raw_val)
        if not math.isfinite(val):
            raise InvalidJoint(
                f"initial_joints[{jname!r}] must be finite, got {raw_val}",
                refs=refs,
                evidence={"reason": "initial_joints_value_not_finite",
                          "param": f"initial_joints[{jname}]",
                          "value": raw_val},
                suggested_repairs=[
                    f"{jname!r} came out NaN or infinite — check the "
                    f"arithmetic that produced it",
                ],
            )
        internal = _rot.convert_pose_value_to_internal(jtype, val)
        # Compared in internal units against the IR's own limits, then
        # REPORTED in surface units: the author wrote degrees, so an error
        # naming radians would send them looking for a bug in their own
        # arithmetic.
        if not (lo <= internal <= hi):
            unit = "m" if jtype == "prismatic" else "deg"
            lo_s = _rot.convert_pose_value_to_surface(jtype, lo)
            hi_s = _rot.convert_pose_value_to_surface(jtype, hi)
            raise InvalidJoint(
                f"initial_joints[{jname!r}] = {val:g} {unit} is outside the "
                f"joint's authored limits [{lo_s:g}, {hi_s:g}] {unit}",
                refs=refs,
                evidence={"reason": "initial_joints_out_of_range",
                          "param": f"initial_joints[{jname}]",
                          "value": val,
                          "limits": [lo_s, hi_s], "units": unit,
                          "instance": instance_name},
                suggested_repairs=[
                    f"pick a value in [{lo_s:g}, {hi_s:g}] {unit}",
                    "or widen the joint's limits= in the object definition",
                ],
            )
        out[jname] = internal
    return out


def _anchor_ref_to_ir(ref: AnchorRef) -> dict:
    """Convert an AnchorRef (in surface form) to its IR placement dict.

    Shape:
        {"name": <anchor name>, "translation": [tx, ty, tz],
         "rotation": [qw, qx, qy, qz]}

    The rotation is converted to the canonical IR quaternion form
    (normalize + sign-canonicalize) so two anchors with semantically
    equal Euler offsets serialize to byte-equal IR. Identity-offset
    anchors still emit the full translation/rotation fields for IR
    schema regularity.
    """
    (tx, ty, tz), (rx, ry, rz) = ref._offset
    q = _rot.euler_xyz_deg_to_quat_wxyz(rx, ry, rz)
    q = _rot.normalize_quat(q)
    q = _rot.sign_canonicalize_quat(q)
    return {
        "name": ref._anchor_name,
        "translation": [float(tx), float(ty), float(tz)],
        "rotation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


# Inf-tolerant float serializer is shared with the object SDK in
# _assertions.float_for_ir.
_float_for_ir = _assertions.float_for_ir


# ---------------------------------------------------------------------------
# ObjectHandle — LLM-facing wrapper around an upstream ObjectIR dict.
# ---------------------------------------------------------------------------


class ObjectHandle:
    """Opaque, LLM-facing view of an upstream finalized ObjectIR dict.

    The raw IR is held in a private slot and is never reachable via
    attribute access from sandboxed code (no underscore-prefixed attribute
    access is permitted by the AST allowlist). All public attributes and
    methods return data in API-surface units: meters, Euler XYZ degrees,
    strings, floats. Quaternions are never surfaced.
    """

    __slots__ = ("_ir", "_hash", "_anchor_names_cache", "_part_anchor_names_cache")

    def __init__(self, ir: dict):
        # The host constructs these; sandbox user code does not. Minimal
        # shape check so callers get a useful error early.
        if not isinstance(ir, dict) or "id" not in ir or "name" not in ir \
                or "parts" not in ir or "joints" not in ir or "root" not in ir:
            raise ValueError("ObjectHandle: ir must be a finalized ObjectIR dict")
        self._ir = ir
        self._hash = ir["id"]
        self._anchor_names_cache = None
        self._part_anchor_names_cache = None

    # ---- identity --------------------------------------------------------

    def __eq__(self, other):
        return isinstance(other, ObjectHandle) and self._hash == other._hash

    def __hash__(self):
        return hash(("ObjectHandle", self._hash))

    def __repr__(self):
        w, l, h = self.dims
        return (f"ObjectHandle(name={self.name!r}, "
                f"dims=({w:g}, {l:g}, {h:g}))")

    # ---- basic attributes ------------------------------------------------

    @property
    def name(self) -> str:
        return self._ir["name"]

    @property
    def parts(self) -> tuple:
        return tuple(p["name"] for p in self._ir["parts"])

    @property
    def joints(self) -> tuple:
        return tuple(j["name"] for j in self._ir["joints"])

    @property
    def bbox(self):
        return _object_query.global_bbox(self._ir)

    @property
    def dims(self):
        (xmn, ymn, zmn), (xmx, ymx, zmx) = self.bbox
        return (xmx - xmn, ymx - ymn, zmx - zmn)

    def source(self):
        """Catalog-relative path to this object's authored `object.py`, or None.

        **A path, not the text, and that is the point.** The scene layer places
        objects rather than re-authoring them, so a couple of hundred lines of
        build123d per placed object would dominate its context with information
        it usually cannot act on — which is why the handle carried no source at
        all for a long time. Returning the path keeps that cost at zero while
        making the recipe *reachable*: `read_file` on the returned path is an
        explicit, opt-in decision, taken only when knowing how an object was
        built actually matters (a clearance that depends on a wall thickness the
        fact table does not name, or a refine pass that edits the recipe rather
        than regenerating it).

        None for an object with no CAD-authored parts. The file travels with the
        object into this run: `referenced_object_assets` lists it and
        `stage_object_assets` copies it under the catalog, paths preserved.
        """
        return _object_query.object_source_path(self._ir)

    # ---- part-scoped queries --------------------------------------------

    def part_dims(self, part_name: str):
        (xmn, ymn, zmn), (xmx, ymx, zmx) = _object_query.global_bbox(self._ir, part_name)
        return (xmx - xmn, ymx - ymn, zmx - zmn)

    def part_bbox(self, part_name: str):
        return _object_query.global_bbox(self._ir, part_name)

    def part_frame(self, part_name: str):
        return _object_query.frame_of(self._ir, part_name)

    def part_color(self, part_name: str):
        """Return the part's RGB color as a 3-tuple of ints in [0, 255]."""
        return _object_query.part_color(self._ir, part_name)

    def part_material(self, part_name: str) -> Material:
        """Return the part's Material (color + PBR scalars)."""
        return _object_query.part_material(self._ir, part_name)

    def part_shape(self, part_name: str):
        """Derived geometry facts for a CAD-authored part, or None.

        The table carries exact ``volume`` / ``area`` / ``center`` (the true
        centroid), ``fill_ratio`` (volume over bounding-box volume — read it
        before trusting ``part_bbox`` for clearance: a shelled carcass reports
        ~0.13, i.e. 87% of that box is air), the ``face_census``, and the
        ``planes`` / ``cylinders`` inventories.

        None only for `box`, which its params fully describe. An `imported`
        part has this table too — integrated from its triangles at import
        rather than measured by the CAD kernel — but only
        ``volume``/``center``/``inertia_diag`` plus vertex and triangle
        counts and a ``closed`` flag; every B-rep-derived key, ``fill_ratio``
        included, is absent, so read them with ``.get``.

        Frozen: the underlying table is `ir["parts"][i]["params"]["shape"]`
        itself, so an unwrapped return would let sandboxed code edit the run's
        catalog IR through `objects['x'].part_shape('p').clear()`.
        """
        return _freeze(_object_query.part_shape(self._ir, part_name),
                       f"objects[{self._ir['name']!r}].part_shape({part_name!r})")

    @property
    def part_shapes(self) -> dict:
        """Mapping from part name to its shape table, for parts that have one.

        The mapping is rebuilt per access, but its values alias into the IR —
        see `part_shape` — so they are frozen.
        """
        out = {}
        for p in self._ir["parts"]:
            table = _object_query.part_shape(self._ir, p["name"])
            if table is not None:
                out[p["name"]] = _freeze(
                    table, f"objects[{self._ir['name']!r}].part_shapes[{p['name']!r}]")
        return out

    @property
    def colors(self) -> dict:
        """Mapping from part name to its RGB color (3-tuple of ints in
        [0, 255])."""
        return {p["name"]: _object_query.part_color(self._ir, p["name"])
                for p in self._ir["parts"]
                if p.get("material") is not None}

    @property
    def materials(self) -> dict:
        """Mapping from part name to its Material.

        A part with a procedural surface reports `color` / `metallic` /
        `roughness` as None — the shader graph owns them. `colors` routes
        through `part_color`, which substitutes the neutral default, and is the
        safe choice for palette reasoning.
        """
        return {p["name"]: _object_query.part_material(self._ir, p["name"])
                for p in self._ir["parts"]
                if p.get("material") is not None}

    def joint_table(self):
        return _object_query.joint_table(self._ir)

    # ---- anchor vocabulary ----------------------------------------------

    def anchor_names(self) -> tuple:
        if self._anchor_names_cache is None:
            self._anchor_names_cache = tuple(sorted(_AABB_ANCHOR_NAMES))
        return self._anchor_names_cache

    def part_anchor_names(self, part_name: str | None = None) -> tuple:
        """Anchor names resolvable on a part — the 27 built-ins, plus, when
        `part_name` is given, that part's **declared** anchors.

        The bare call answers about every part at once and so can only report
        the built-ins. Name a part and you also get what its author (or, for an
        imported asset, its URDF) registered by name: `<part>.<name>` resolves
        those from this layer and from the animation layer, and it has since
        declared anchors reached the scene kernel.
        """
        if part_name is None:
            if self._part_anchor_names_cache is None:
                self._part_anchor_names_cache = tuple(sorted(_AABB_ANCHOR_NAMES))
            return self._part_anchor_names_cache
        for p in self._ir["parts"]:
            if p["name"] == part_name:
                declared = [a["name"] for a in (p.get("anchors") or ())
                            if isinstance(a, dict) and a.get("name")]
                return tuple(sorted(set(_AABB_ANCHOR_NAMES) | set(declared)))
        known = [p["name"] for p in self._ir["parts"]]
        raise KeyError(
            f"{self._ir['name']!r} has no part named {part_name!r}; "
            f"has: {', '.join(known)}")


# ---------------------------------------------------------------------------
# Opaque refs
# ---------------------------------------------------------------------------


class _Ref:
    """Scene-layer ref base with optional `_poses` for pose tagging.

    Pose-tagging on a scene ref (Instance/Group) means: when this ref's
    AABB is resolved, the underlying object's kinematic tree is
    evaluated under the supplied joint poses. Pose keys not on the
    ref's ancestor chain are silently ignored.
    """

    __slots__ = ("_id", "_owner_id", "_name", "_poses")

    def __init__(self, id_: int, owner_id: int, name: str, poses=()):
        from ._refs import freeze_poses
        self._id = id_
        self._owner_id = owner_id
        self._name = name
        self._poses = freeze_poses(poses) if not isinstance(poses, tuple) else poses

    @property
    def name(self) -> str:
        """The author-supplied name of this ref's entity (instance / group / room)."""
        return self._name

    @property
    def poses(self) -> dict:
        """Pose overrides applied via `.at({...})`. Empty dict if untagged."""
        from ._refs import poses_to_dict
        return poses_to_dict(self._poses)

    def __eq__(self, other):
        return (type(self) is type(other)
                and self._id == other._id
                and self._owner_id == other._owner_id
                and self._poses == other._poses)

    def __hash__(self):
        return hash((type(self).__name__, self._id, self._owner_id, self._poses))


class RoomRef(_Ref):
    """Reference to the scene's room. Singleton per scene. Fixed id = 0.

    `.at(...)` is accepted for API uniformity but the room has no
    kinematics — all pose keys are silently ignored at evaluation.
    """
    __slots__ = ("_width", "_depth", "_height")

    def __init__(self, owner_id: int, width: float, depth: float, height: float,
                 poses=()):
        super().__init__(0, owner_id, _ROOM_RESERVED_NAME, poses=poses)
        self._width = width
        self._depth = depth
        self._height = height

    @property
    def width(self) -> float:
        """Room width in meters (x extent)."""
        return self._width

    @property
    def depth(self) -> float:
        """Room depth in meters (y extent)."""
        return self._depth

    @property
    def height(self) -> float:
        """Room height in meters (z extent)."""
        return self._height

    def at(self, poses) -> "RoomRef":
        from ._refs import _merge_pose_overrides
        return RoomRef(self._owner_id, self._width, self._depth, self._height,
                       _merge_pose_overrides(self._poses, poses))

    def __repr__(self):
        return (f"RoomRef(width={self._width:g}, "
                f"depth={self._depth:g}, height={self._height:g})")


class InstanceRef(_Ref):
    __slots__ = ()

    def at(self, poses) -> "InstanceRef":
        """Return a pose-tagged copy.

        Pose keys (`<joint_name>` or a `JointRef` from the instance's
        owning Object) apply to that joint when the instance's effective
        AABB is computed under articulation.
        """
        from ._refs import _merge_pose_overrides
        return InstanceRef(self._id, self._owner_id, self._name,
                           _merge_pose_overrides(self._poses, poses))

    def __repr__(self):
        base = f"InstanceRef(name={self._name!r}, id={self._id}"
        if self._poses:
            from ._refs import poses_to_dict
            base += f", poses={poses_to_dict(self._poses)}"
        return base + ")"


class GroupRef(_Ref):
    __slots__ = ()

    def at(self, poses) -> "GroupRef":
        """Return a pose-tagged copy.

        Pose keys may be either `<joint_name>` (applies to every
        instance in the group whose underlying Object declares that
        joint) or `<instance_name>.<joint_name>` (applies only to the
        named instance). Unmatched keys are silently ignored.
        """
        from ._refs import _merge_pose_overrides
        return GroupRef(self._id, self._owner_id, self._name,
                        _merge_pose_overrides(self._poses, poses))

    def __repr__(self):
        base = f"GroupRef(name={self._name!r}, id={self._id}"
        if self._poses:
            from ._refs import poses_to_dict
            base += f", poses={poses_to_dict(self._poses)}"
        return base + ")"


# AnchorRef and AssertionRef are imported from ._refs (shared with the
# object SDK).


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------


class Scene:
    """Builder for a single scene composed of object instances, groups,
    and LLM-authored relational assertions.

    At most one Scene may be constructed per sandbox execution. Builder
    methods mutate internal state; `finalize()` produces a frozen,
    content-addressed scene IR dict.
    """

    def __init__(self, name: str):
        global _scene_singleton_constructed
        with _scene_singleton_lock:
            if _scene_singleton_constructed:
                raise InvalidScene(
                    "only one Scene may be constructed per sandbox execution",
                    refs=[_ref("scene", name if isinstance(name, str) else None)],
                    evidence={"reason": "singleton_already_constructed"},
                    suggested_repairs=[
                        "construct exactly one Scene per scene.py",
                    ],
                )
            _scene_singleton_constructed = True

        _check_name(name, "scene", InvalidScene,
                    refs=[_ref("scene", name if isinstance(name, str) else None)])
        self._name = name
        self._owner_id = id(self)

        self._room: dict | None = None
        self._room_ref: RoomRef | None = None

        # Instances: list of dicts in creation order, plus indices.
        self._instances: list[dict] = []
        self._instance_name_to_id: dict[str, int] = {}
        self._instance_refs: dict[int, InstanceRef] = {}

        # Groups: list of dicts, plus indices.
        self._groups: list[dict] = []
        self._group_name_to_id: dict[str, int] = {}
        self._group_refs: dict[int, GroupRef] = {}

        # Shared entity namespace (instances + groups + room reserved).
        # name -> (entity_kind, ref).
        self._entities: dict[str, tuple] = {}

        # Attachment forest tracking. For each entity (instance or group):
        # parent name (string) or None if it is a "root" (room-placed or
        # group-shadowed root). Used for cycle detection on attach_group.
        self._parent_of: dict[str, str] = {}

        # Which entities have been grouped already (by entity name).
        # Used to enforce "a member belongs to at most one group" and to
        # forbid independent place/attach after grouping.
        self._grouped: set[str] = set()

        # Which groups have already been attached via attach_group (each
        # group can be attached at most once).
        self._group_attached: set[str] = set()

        # Assertions namespace (separate from entities).
        self._assertions: list[dict] = []
        #: Observable claims for the independent reviewer. See `add_check`.
        self._checks: list[dict] = []
        self._check_claims: set = set()
        self._lighting = None
        self._light_names: set = set()

        self._replaced_surfaces: frozenset = frozenset()
        self._assertion_name_to_id: dict[str, int] = {}

        # instance_name -> tuple of part names. Recorded at place/attach
        # time so anchor validation can run without holding a catalog
        # reference. Avoids the LLM ever seeing the raw ObjectIR fields.
        self._instance_parts: dict[str, tuple] = {}

        # instance_name -> {part_name: frozenset(declared anchor names)}.
        # Recorded alongside `_instance_parts` and for the same reason: a
        # `<part>.<declared_name>` anchor path has to be checkable at the
        # offending call without a catalog lookup.
        self._instance_part_anchors: dict[str, dict] = {}

        # instance_name -> {joint_name: joint_type}. Recorded at place/attach
        # time. Used by the assertion-record builders to convert surface-unit
        # pose values (degrees for revolute) into IR-internal units (radians)
        # at serialization time. Pose keys on scene refs / assertion-level
        # poses= are either bare (`"hinge"`) or qualified (`"<inst>.<hinge>"`);
        # _joint_types_map flattens both forms into one lookup table.
        self._instance_joint_types: dict[str, dict[str, str]] = {}

        self._finalized = False
        self._cached_ir: dict | None = None

    # ---- helpers ---------------------------------------------------------

    def _joint_types_map(self) -> dict:
        """Return a flat `{key: joint_type}` map for assertion-record pose
        conversion. Keys are both bare (`"hinge"`) and qualified
        (`"<instance>.<hinge>"`); for bare keys, the first-seen joint type
        across instances wins (matches the runtime "applies to any instance
        with that joint" semantics — a name collision with conflicting
        types would surface as a verifier error elsewhere).
        """
        out: dict = {}
        for inst_name, joints in self._instance_joint_types.items():
            for jname, jtype in joints.items():
                out.setdefault(jname, jtype)
                out[f"{inst_name}.{jname}"] = jtype
        return out

    def _check_not_finalized(self):
        if self._finalized:
            raise BuildError(
                "scene builder is finalized; no further calls allowed",
                refs=[_ref("scene", self._name)],
                evidence={"reason": "builder_already_finalized"},
                suggested_repairs=["do not call builder methods after finalize()"],
            )

    def _check_room_declared(self, op_label: str):
        if self._room is None:
            raise RoomNotDeclared(
                f"set_room must be called before {op_label}",
                refs=[_ref("scene", self._name)],
                evidence={"reason": "room_not_declared", "operation": op_label},
                suggested_repairs=["call scene.set_room(width, depth, height) "
                                   "before any place / attach / create_group / "
                                   "assert_* call"],
            )

    def _check_entity_name_available(self, name: str, kind_label: str, error_cls):
        if name == _ROOM_RESERVED_NAME:
            raise error_cls(
                f"name {_ROOM_RESERVED_NAME!r} is reserved for the scene's room",
                refs=[_ref("scene", self._name),
                      _ref(kind_label, name, None)],
                evidence={"reason": "reserved_name", "value": name},
                suggested_repairs=[f"pick a {kind_label} name other than "
                                   f"{_ROOM_RESERVED_NAME!r}"],
            )
        if name in self._entities:
            existing_kind, _existing = self._entities[name]
            raise error_cls(
                f"entity name {name!r} already used by a {existing_kind}",
                refs=[_ref("scene", self._name),
                      _ref(kind_label, name, None)],
                evidence={"reason": "entity_name_collision",
                          "value": name, "existing_kind": existing_kind},
                suggested_repairs=["choose a unique entity name"],
            )

    def _entity_kind_of(self, ref) -> str:
        if isinstance(ref, InstanceRef):
            return "instance"
        if isinstance(ref, GroupRef):
            return "group"
        if isinstance(ref, RoomRef):
            return "room"
        raise InvalidInstance(
            f"expected InstanceRef / GroupRef / RoomRef, got "
            f"{type(ref).__name__}",
            refs=[_ref("scene", self._name)],
            evidence={"reason": "bad_ref_type", "value": type(ref).__name__},
            suggested_repairs=["pass a ref returned by this scene's builder"],
        )

    def _check_owned(self, ref, error_cls):
        kind = self._entity_kind_of(ref)
        if ref._owner_id != self._owner_id:
            raise error_cls(
                f"{kind} ref {ref!r} was not produced by this Scene",
                refs=[_ref("scene", self._name), _ref(kind, ref._name, None)],
                evidence={"reason": "foreign_ref",
                          "owner_scene": self._name,
                          "ref_kind": kind, "ref_name": ref._name},
                suggested_repairs=["use only refs returned by this scene's "
                                   "builder methods"],
            )

    # ---- room ------------------------------------------------------------

    def set_room(self, width: float, depth: float, height: float,
                 floor_material: Material | None = None,
                 wall_material: Material | None = None,
                 ceiling_material: Material | None = None,
                 enclosure: str = DEFAULT_ROOM_ENCLOSURE) -> RoomRef:
        """Declare the room: its dimensions, and the shell's three materials.

        The shell is six surfaces — floor, ceiling, and four walls — synthesized
        per camera from these dimensions. **All six exist whatever the camera
        sees**: the cutaway hides the ones standing between the eye and the
        room, and a hidden surface still bounces, shadows, and blocks. Deleting
        them instead would light the interior as an open-topped three-sided
        box.

        Each material is a full `Material`, so a room surface can carry a
        procedural `shader=` graph or an image texture exactly as a part does.
        `ceiling_material` defaults to `wall_material`.

        `enclosure` says how much of that shell the scene actually has:
        `"interior"` (all six), `"open_top"` (no ceiling — a courtyard or light
        well) or `"open_air"` (floor only — a street, an alley, a garden). **An
        extent is not an enclosure.** Declaring a room says how big the world
        is and where the ground is; whether it has a lid is a separate fact,
        and an outdoor scene that gets one loses the sky.

        To take a surface over with authored geometry — a wall with a real
        opening in it — see `replace_room_surface`.
        """
        self._check_not_finalized()
        if self._room is not None:
            raise InvalidRoom(
                "set_room may be called at most once per Scene",
                refs=[_ref("scene", self._name), _ref("room", _ROOM_RESERVED_NAME)],
                evidence={"reason": "room_already_declared"},
                suggested_repairs=["call set_room exactly once"],
            )
        if self._instances or self._groups or self._assertions:
            raise InvalidRoom(
                "set_room must be called before any place / attach / "
                "create_group / assert_* call",
                refs=[_ref("scene", self._name), _ref("room", _ROOM_RESERVED_NAME)],
                evidence={"reason": "set_room_after_other_calls"},
                suggested_repairs=["call set_room first"],
            )
        room_refs = [_ref("scene", self._name), _ref("room", _ROOM_RESERVED_NAME)]
        for label, val in (("width", width), ("depth", depth), ("height", height)):
            _check_finite_scalar(val, label, InvalidRoom, refs=room_refs)
            if float(val) <= 0.0:
                raise InvalidRoom(
                    f"room {label} must be > 0, got {val}",
                    refs=room_refs,
                    evidence={"param": label, "value": val, "constraint": "> 0"},
                    suggested_repairs=[f"pass a strictly positive value for "
                                       f"room {label}"],
                )
        if floor_material is None:
            floor_mat = DEFAULT_FLOOR_MATERIAL
        else:
            floor_mat = check_material(floor_material, "floor_material",
                                       error_cls=InvalidRoom, refs=room_refs)
        if wall_material is None:
            wall_mat = DEFAULT_WALL_MATERIAL
        else:
            wall_mat = check_material(wall_material, "wall_material",
                                      error_cls=InvalidRoom, refs=room_refs)
        ceiling_mat = (None if ceiling_material is None else
                       check_material(ceiling_material, "ceiling_material",
                                      error_cls=InvalidRoom, refs=room_refs))
        self._room = {
            "width": float(width),
            "depth": float(depth),
            "height": float(height),
            "floor_material": floor_mat.to_ir(),
            "wall_material": wall_mat.to_ir(),
        }
        # Optional, and omitted when unset so an unchanged scene hashes to the
        # id it always did. Surface replacement is declared separately, by
        # `replace_room_surface`, because it names an instance that cannot
        # exist yet — `set_room` runs before any `place`.
        if ceiling_mat is not None:
            self._room["ceiling_material"] = ceiling_mat.to_ir()
        if enclosure not in ROOM_ENCLOSURES:
            raise InvalidRoom(
                f"unknown room enclosure {enclosure!r}",
                refs=room_refs,
                evidence={"param": "enclosure", "value": enclosure,
                          "known": list(ROOM_ENCLOSURES)},
                suggested_repairs=[
                    "interior (a room), open_top (a courtyard), or open_air "
                    "(a street or garden)"])
        # Written only when it is not the default, so an unchanged scene hashes
        # to the id it always did.
        if enclosure != DEFAULT_ROOM_ENCLOSURE:
            self._room["enclosure"] = enclosure
        self._room_ref = RoomRef(self._owner_id, float(width), float(depth),
                                 float(height))
        return self._room_ref

    @property
    def room(self) -> RoomRef:
        if self._room_ref is None:
            raise RoomNotDeclared(
                "scene.room is not yet defined; call set_room first",
                refs=[_ref("scene", self._name)],
                evidence={"reason": "room_not_declared"},
                suggested_repairs=["call scene.set_room(width, depth, height)"],
            )
        return self._room_ref

    # ---- place / attach (new instance) -----------------------------------

    def replace_room_surface(self, surface: str, instance=None) -> None:
        """Take one surface of the default shell out of the renderer's hands.

        The room is a convenience: five numbers, drawn as six thin slabs with
        one material for the walls, no thickness worth the name, and nowhere to
        cut an opening. Anything architecturally real — a wall with a window
        punched through it, a coffered ceiling, a plank floor whose boards are
        geometry — is an ordinary catalog object. This is how you say so::

            wall = scene.place(objects["back_wall"], name="back_wall",
                               pose=((0.0, -2.19, 0.0), (0, 0, 0)))
            scene.replace_room_surface("wall_ymin", wall)

        After this the shell's `wall_ymin` is **not drawn at all** — not
        visibly, and not as a hidden bounce or shadow surface either. That
        matters: leaving it would put an invisible slab behind the authored
        wall, shadowing the room and sealing whatever opening the object cut.

        `surface` is one of `ROOM_SURFACES`, or a viewer-relative alias
        (`back`, `front`, `left`, `right`).

        `instance` is the object standing in for it, and naming it buys one
        thing: `RoomContainment` lets that instance straddle the room boundary
        by its own thickness, because a replacement wall is *supposed* to
        occupy the shell's plane. Containment is not relaxed for
        anything else — furniture that escapes the room is still an error.
        Passing `None` suppresses the surface with no replacement, which is how
        you get an open-topped room or a missing fourth wall on purpose.
        """
        self._check_not_finalized()
        self._check_room_declared("replace_room_surface")
        refs = [_ref("scene", self._name), _ref("room", _ROOM_RESERVED_NAME)]
        name = _check_room_surface(surface, refs)

        instance_name = None
        if instance is not None:
            if isinstance(instance, InstanceRef):
                self._check_owned(instance, InvalidRoom)
                instance_name = instance._name
            elif isinstance(instance, str):
                instance_name = instance
            else:
                raise InvalidRoom(
                    f"replace_room_surface instance must be an InstanceRef "
                    f"from place/attach, got {type(instance).__name__}",
                    refs=refs,
                    evidence={"param": "instance",
                              "value": type(instance).__name__},
                    suggested_repairs=["pass the ref returned by scene.place"])
            known = {i["name"] for i in self._instances}
            if instance_name not in known:
                raise InvalidRoom(
                    f"replace_room_surface: no instance named "
                    f"{instance_name!r}; place it first",
                    refs=refs,
                    evidence={"param": "instance", "value": instance_name,
                              "known": sorted(known)},
                    suggested_repairs=[
                        "call scene.place(...) before declaring the "
                        "replacement"])

        surfaces = list(self._room.get("suppressed_surfaces") or ())
        if name not in surfaces:
            surfaces.append(name)
        self._room["suppressed_surfaces"] = surfaces

        if instance_name is not None:
            mapping = dict(self._room.get("surface_replacements") or {})
            mapping[name] = instance_name
            self._room["surface_replacements"] = mapping
        self._replaced_surfaces = frozenset(surfaces)

    def _validate_object_handle(self, handle, ctx_label: str, refs):
        """Normalize what an author passed into an ObjectHandle, or raise.

        **Three spellings, one meaning.** A run authors its objects in the
        same file, so what it has in hand is an `Object` builder or the
        finalized IR dict that builder returned, beside an `ObjectHandle` it
        made itself — and requiring
        `ObjectHandle(make_counter(...).finalize())` at every call site would
        be ceremony over a distinction the author has no reason to care about.
        All three are accepted and normalized here, so `place` and `attach`
        below still work on exactly one type.

        An `Object` is recognized by having a callable `finalize`, not by
        importing the class: `object_builder` imports nothing from this module
        today and an import the other way would close that loop for a check
        that duck-typing answers exactly as well.
        """
        if isinstance(handle, ObjectHandle):
            return handle
        finalize = getattr(handle, "finalize", None)
        if callable(finalize) and not isinstance(handle, (dict, str, bytes)):
            handle = finalize()
        if isinstance(handle, dict):
            try:
                return ObjectHandle(handle)
            except ValueError:
                raise InvalidInstance(
                    f"{ctx_label}: got a dict that is not a finalized ObjectIR "
                    f"(needs id/name/parts/joints/root)",
                    refs=refs,
                    evidence={"reason": "bad_handle_ir",
                              "value": sorted(handle)[:8]},
                    suggested_repairs=[
                        "pass the value `obj.finalize()` returned, unmodified",
                    ],
                ) from None
        raise InvalidInstance(
            f"{ctx_label}: expected ObjectHandle, Object, or finalized ObjectIR, "
            f"got {type(handle).__name__}",
            refs=refs,
            evidence={"reason": "bad_handle_type",
                      "value": type(handle).__name__},
            suggested_repairs=["pass an ObjectHandle from the `objects` "
                               "catalog, or an Object you built in this file"],
        )

    def place(self, handle: ObjectHandle, *, name: str, pose,
              anchor: AnchorRef | None = None,
              initial_joints=None) -> InstanceRef:
        """Place a new instance so a chosen anchor lands at `pose`.

        Placement is anchor-based: `pose = ((x, y, z), (rx, ry, rz))` gives
        the world point (and orientation) at which the object's `anchor`
        sits. `anchor` defaults to `scene.anchor(handle, "bottom")` — the
        object's overall bbox-bottom-center — so `pose`'s `z` is the
        supporting-surface height (0 for floor-standing). The object's
        authored local origin is irrelevant; only the anchor matters.

        This is exactly `scene.attach(handle, name=name, parent=scene.room,
        parent_anchor=scene.anchor(scene.room, "bottom", translation=t,
        rotation=r), child_anchor=anchor)` — i.e. attaching the anchor to
        the room's floor-center, offset to the requested pose.

        `initial_joints={"door_hinge": 40}` selects **this instance's** initial
        configuration without touching the shared definition — see
        `Scene.attach`, which documents the units and the precedence rules.
        """
        self._check_not_finalized()
        self._check_room_declared("place")
        inst_refs = [_ref("scene", self._name), _ref("instance", name)]
        handle = self._validate_object_handle(handle, "place", inst_refs)
        _check_se3(pose, "pose", InvalidInstance, refs=inst_refs)
        (tx, ty, tz), (rx, ry, rz) = pose
        parent_anchor = self.anchor(self.room, "bottom",
                                    translation=(tx, ty, tz),
                                    rotation=(rx, ry, rz))
        child_anchor = (self._as_anchor_ref(anchor, handle)
                        if anchor is not None
                        else self.anchor(handle, "bottom"))
        return self.attach(handle, name=name, parent=self.room,
                           parent_anchor=parent_anchor,
                           child_anchor=child_anchor,
                           initial_joints=initial_joints)

    def attach(self, handle: ObjectHandle, *, name: str,
               parent, parent_anchor: AnchorRef,
               child_anchor: AnchorRef,
               follow: bool = True,
               initial_joints=None) -> InstanceRef:
        """Attach a new instance to `parent` aligning the two anchors.

        `parent_anchor` and `child_anchor` each take an anchor name or an
        `AnchorRef` from `Scene.anchor(...)`. A ref's `translation`/`rotation`
        offset is interpreted in its entity's local frame; there is no
        separate `offset` argument.

        `parent_anchor` must be on `parent`. `child_anchor` must be on
        the new instance — pass an instance-style ref. Because the new
        instance doesn't exist yet, callers should produce it via
        `Scene.anchor(handle, ...)` where `handle` is the same
        `ObjectHandle` being attached; see the worked example in
        `agent/prompts/scene.md` for the pattern.

        `follow` (default True): when `parent_anchor` is a part-scoped
        anchor (`"<part>.<landmark>"`) on a part that moves under one of
        the parent's joints, the attached instance **rides that joint** —
        e.g. a workpiece bolted to a trunnion plate tilts with the
        trunnion's A-axis, so the verifiers see them move together
        instead of the plate sweeping through a static child. Attaching
        to a static part (or the room) is unaffected: its frame is
        pose-independent, so `follow` is a no-op there. Set
        `follow=False` to pin the child to the parent's *rest* frame
        regardless of articulation — e.g. a fixture aimed at where a
        spindle sits at rest without physically riding it.

        `initial_joints={"door_hinge": 40}` selects **this instance's** initial
        joint configuration. It is saved on the instance record and is the
        scene's committed state for those joints — snapshots, probes, scene
        assertions, collision, support, rendering, URDF/sim export, and any
        animation channel the author does not drive all evaluate it. Values are
        in surface units (**degrees** for a revolute joint, **meters** for a
        prismatic one) and must lie within the joint's authored `limits`.

        **It does not touch the definition.** Two `place` calls with different
        `initial_joints` share one `object_ref`, one ObjectIR, and one set of
        meshes; nothing is cloned, mutated, or re-finalized. That is the whole
        point — the alternative an author reaches for otherwise is building the
        object twice with different `default=` values, which doubles the
        geometry cost and gives the two cabinets different identities.

        Joints left out inherit normally, so a partial mapping is exactly a
        partial mapping: it never resets the others. Effective value of a
        joint, highest priority first:

        1. an instance-qualified evaluation override (`.at({"cab.hinge": 10})`,
           `poses={"cab.hinge": 10}`);
        2. a bare evaluation override (`.at({"hinge": 10})`), which broadcasts
           to every instance that has such a joint;
        3. this instance's `initial_joints` value;
        4. the object definition's `default=`.

        The author's mapping is copied, so mutating it afterwards cannot change
        the committed scene.

        **Initial alignment is fixed at the state you place in.** The child
        anchor — including a bbox landmark like the default `"bottom"` — is
        resolved in the configuration `initial_joints` selects, and then stays
        put. Opening that instance's door later for a snapshot or an animation
        does not re-derive its placement, so the body does not slide sideways
        to keep a growing bounding box's bottom on the floor.
        """
        self._check_not_finalized()
        self._check_room_declared("attach")
        _check_name(name, "instance", InvalidInstance,
                    refs=[_ref("scene", self._name), _ref("instance", name)])
        self._check_entity_name_available(name, "instance", InvalidInstance)
        inst_refs = [_ref("scene", self._name), _ref("instance", name)]
        handle = self._validate_object_handle(handle, "attach", inst_refs)
        self._check_owned(parent, InvalidInstance)

        # Validate parent_anchor and child_anchor are AnchorRefs.
        parent_kind = self._entity_kind_of(parent)
        parent_anchor = self._as_anchor_ref(parent_anchor, parent)
        child_anchor = self._as_anchor_ref(child_anchor, handle)
        self._validate_attach_anchor(
            parent_anchor, parent, parent_kind, inst_refs,
            label="parent_anchor")
        self._validate_attach_anchor_handle(
            child_anchor, handle, inst_refs, label="child_anchor")

        # Grouping check: a grouped parent's individual members cannot
        # serve as attach parents for new instances (only the group can).
        if parent_kind in ("instance", "group") and parent._name in self._grouped:
            raise InvalidInstance(
                f"parent {parent._name!r} has been grouped; attach via the "
                f"containing group instead",
                refs=inst_refs + [_ref(parent_kind, parent._name)],
                evidence={"reason": "parent_already_grouped",
                          "value": parent._name},
                suggested_repairs=["attach to the group, or restructure so the "
                                   "parent is not a member of a group"],
            )

        joint_state = _normalize_initial_joints(
            initial_joints, _handle_joint_table(handle),
            instance_name=name, refs=inst_refs)

        inst_id = len(self._instances)
        placement = {
            "kind": "attach",
            "parent": parent._name,
            "parent_anchor": _anchor_ref_to_ir(parent_anchor),
            "child_anchor": _anchor_ref_to_ir(child_anchor),
        }
        # Only emit `follow` when it differs from the default (True) so
        # existing scenes' IR hashes are unchanged; the kernel reads
        # `placement.get("follow", True)`.
        if not follow:
            placement["follow"] = False
        record = {
            "id": inst_id,
            "name": name,
            "object_ref": handle._hash,
            "object_name": handle.name,
            "placement": placement,
        }
        # Omitted when the author selected nothing, on the same terms as
        # `follow` above and for the same reason: the IR is content-hashed
        # whole, so an unconditional key would re-id every scene ever built.
        # `finalize` bumps the schema tag only when some instance has one.
        if joint_state:
            record["initial_joints"] = joint_state
        self._instances.append(record)
        self._instance_name_to_id[name] = inst_id
        ref = InstanceRef(inst_id, self._owner_id, name)
        self._instance_refs[inst_id] = ref
        self._entities[name] = ("instance", ref)
        self._instance_parts[name] = handle.parts
        # Declared semantic anchors, per part, recorded at place/attach time
        # for the same reason `_instance_parts` is: anchor validation runs
        # without holding a catalog reference, and the raw ObjectIR never
        # reaches the author.
        self._instance_part_anchors[name] = _handle_part_anchors(handle)
        self._instance_joint_types[name] = {
            j["name"]: j["type"] for j in handle._ir.get("joints", [])
        }
        self._parent_of[name] = parent._name
        return ref

    # ---- groups ----------------------------------------------------------

    def create_group(self, *, name: str,
                     members,
                     root=None) -> GroupRef:
        self._check_not_finalized()
        self._check_room_declared("create_group")
        _check_name(name, "group", InvalidGroup,
                    refs=[_ref("scene", self._name), _ref("group", name)])
        self._check_entity_name_available(name, "group", InvalidGroup)
        grp_refs = [_ref("scene", self._name), _ref("group", name)]

        if not _ro.is_sequence(members) or len(members) == 0:
            raise InvalidGroup(
                "members must be a non-empty list/tuple of InstanceRef / GroupRef",
                refs=grp_refs,
                evidence={"reason": "empty_or_bad_members",
                          "value": repr(members)},
                suggested_repairs=["pass at least one InstanceRef or GroupRef"],
            )

        member_names: list[str] = []
        member_kinds: list[str] = []
        for m in members:
            self._check_owned(m, InvalidGroup)
            kind = self._entity_kind_of(m)
            if kind == "room":
                raise InvalidGroup(
                    "the room cannot be a group member",
                    refs=grp_refs + [_ref("room", m._name)],
                    evidence={"reason": "room_as_member"},
                    suggested_repairs=["remove the room from members"],
                )
            if m._name in self._grouped:
                raise InvalidGroup(
                    f"member {m._name!r} already belongs to another group",
                    refs=grp_refs + [_ref(kind, m._name)],
                    evidence={"reason": "member_already_grouped",
                              "value": m._name},
                    suggested_repairs=["Each entity can belong to at most one group. "
                                       "Revise the earlier create_group membership or "
                                       "choose another member; there is no ungroup "
                                       "method."],
                )
            member_names.append(m._name)
            member_kinds.append(kind)

        # Determine the group root.
        if root is None:
            # Auto-pick: the unique member whose parent is outside the
            # group (or is None, meaning room-placed).
            external = []
            for n in member_names:
                parent_name = self._parent_of.get(n)
                if parent_name is None or parent_name not in member_names:
                    external.append(n)
            if len(external) != 1:
                raise InvalidGroup(
                    f"cannot auto-pick group root: {len(external)} members "
                    f"have parents outside the group "
                    f"(expected exactly 1)",
                    refs=grp_refs,
                    evidence={"reason": "ambiguous_root",
                              "candidates": external,
                              "members": member_names},
                    suggested_repairs=["pass an explicit root=<member>"],
                )
            root_name = external[0]
        else:
            self._check_owned(root, InvalidGroup)
            if root._name not in member_names:
                raise InvalidGroup(
                    f"root {root._name!r} must be in members",
                    refs=grp_refs + [_ref(self._entity_kind_of(root), root._name)],
                    evidence={"reason": "root_not_in_members",
                              "value": root._name, "members": member_names},
                    suggested_repairs=["pick a root that is in members, or "
                                       "omit root to auto-select"],
                )
            root_name = root._name

        # Verify that every non-root member is transitively attached to the
        # root via existing attach edges that stay within `members`.
        member_set = set(member_names)
        reachable = {root_name}
        # We walk parent pointers upward; each member must eventually reach
        # root via member-internal edges.
        for n in member_names:
            cur = n
            seen = set()
            while cur != root_name:
                if cur in seen:
                    raise InvalidGroup(
                        f"member {n!r} appears in an attachment cycle",
                        refs=grp_refs,
                        evidence={"reason": "member_cycle", "value": n},
                        suggested_repairs=["fix the attach graph"],
                    )
                seen.add(cur)
                p = self._parent_of.get(cur)
                if p is None or p not in member_set:
                    raise InvalidGroup(
                        f"member {n!r} is not transitively attached to root "
                        f"{root_name!r} via members of the group",
                        refs=grp_refs,
                        evidence={"reason": "member_not_connected",
                                  "value": n, "root": root_name,
                                  "first_external_parent": p},
                        suggested_repairs=["attach all non-root members to the "
                                           "root via attach() calls before "
                                           "create_group, or include the "
                                           "missing intermediate parents in "
                                           "members"],
                    )
                cur = p
            reachable.add(n)

        group_id = len(self._groups)
        group_record = {
            "id": group_id,
            "name": name,
            "members": list(member_names),
            "root": root_name,
            # placement is null on the group at create time; if/when
            # attach_group is called this is filled in. The group root's
            # own instance.placement is also "shadowed" — see finalize().
            "placement": None,
        }
        self._groups.append(group_record)
        self._group_name_to_id[name] = group_id
        ref = GroupRef(group_id, self._owner_id, name)
        self._group_refs[group_id] = ref
        self._entities[name] = ("group", ref)
        for n in member_names:
            self._grouped.add(n)
        # The group itself has the same parent as its root member.
        self._parent_of[name] = self._parent_of.get(root_name)
        return ref

    def attach_group(self, group: GroupRef, *,
                     parent, parent_anchor: AnchorRef,
                     child_anchor: AnchorRef) -> None:
        """Attach `group` to `parent` aligning the two anchors.

        Both anchors are `AnchorRef`s. `parent_anchor` must be on
        `parent`; `child_anchor` must be on `group`. Each ref's
        translation/rotation offset is in its entity's local frame.
        """
        self._check_not_finalized()
        self._check_room_declared("attach_group")
        self._check_owned(group, InvalidGroup)
        if not isinstance(group, GroupRef):
            raise InvalidGroup(
                "attach_group: first argument must be a GroupRef",
                refs=[_ref("scene", self._name)],
                evidence={"reason": "bad_group_type",
                          "value": type(group).__name__},
                suggested_repairs=["pass the GroupRef returned by create_group"],
            )
        if group._name in self._group_attached:
            raise InvalidGroup(
                f"group {group._name!r} is already attached",
                refs=[_ref("scene", self._name), _ref("group", group._name)],
                evidence={"reason": "group_already_attached",
                          "value": group._name},
                suggested_repairs=["each group can be attached at most once"],
            )
        self._check_owned(parent, InvalidGroup)
        grp_refs = [_ref("scene", self._name), _ref("group", group._name)]

        parent_kind = self._entity_kind_of(parent)
        parent_anchor = self._as_anchor_ref(parent_anchor, parent)
        child_anchor = self._as_anchor_ref(child_anchor, group)
        self._validate_attach_anchor(
            parent_anchor, parent, parent_kind, grp_refs,
            label="parent_anchor")
        self._validate_attach_anchor(
            child_anchor, group, "group", grp_refs,
            label="child_anchor")

        # Cycle check: walking from `parent` upward must not enter the
        # group's closure (its members + their transitive descendants).
        descendants = self._descendants_of(group._name)
        cur = parent._name if parent_kind != "room" else None
        guard = 0
        while cur is not None:
            if cur in descendants:
                raise InvalidGroup(
                    f"attach_group would create a cycle via {cur!r}",
                    refs=grp_refs + [_ref(parent_kind, parent._name)],
                    evidence={"reason": "cycle", "value": cur,
                              "group": group._name},
                    suggested_repairs=["attach the group to an entity that is "
                                       "not a descendant of the group"],
                )
            cur = self._parent_of.get(cur)
            guard += 1
            if guard > len(self._parent_of) + 2:  # safety net
                break  # pragma: no cover

        record = self._groups[self._group_name_to_id[group._name]]
        record["placement"] = {
            "kind": "attach",
            "parent": parent._name,
            "parent_anchor": _anchor_ref_to_ir(parent_anchor),
            "child_anchor": _anchor_ref_to_ir(child_anchor),
        }
        # The group root's own instance placement is shadowed — finalize()
        # writes null in its place.
        self._group_attached.add(group._name)
        self._parent_of[group._name] = parent._name

    def _descendants_of(self, entity_name: str) -> set:
        """Return the closure of entities logically inside `entity_name`
        — i.e., the set the proposed attach_group parent must NOT
        intersect.

        Closure includes:
          - members of the group (recursively for nested groups);
          - every entity whose parent_of chain reaches any member or
            the group itself via an attach edge.
        """
        out: set[str] = set()

        # Step 1: expand group membership transitively.
        def expand_membership(name: str):
            if name in out:
                return
            out.add(name)
            if name in self._group_name_to_id:
                g = self._groups[self._group_name_to_id[name]]
                for m in g["members"]:
                    expand_membership(m)

        expand_membership(entity_name)

        # Step 2: walk attach-edge children from every entity in out
        # and add anything whose parent points into the closure.
        children: dict[str, list[str]] = {}
        for child_name, parent_name in self._parent_of.items():
            if parent_name is None:
                continue
            children.setdefault(parent_name, []).append(child_name)
        frontier = list(out)
        while frontier:
            n = frontier.pop()
            for c in children.get(n, ()):
                if c not in out:
                    out.add(c)
                    frontier.append(c)

        # The original entity itself is already in `out`; callers may
        # remove it or include it depending on intent. The caller
        # (attach_group) wants to forbid `entity_name` too, so we leave
        # it in.
        return out

    # ---- anchors ---------------------------------------------------------

    def _as_anchor_ref(self, value, entity):
        """An `AnchorRef` for `value`, resolving a bare anchor NAME on `entity`.

        `place(anchor="top.datum")` and `attach(parent_anchor="bottom")` are
        what authors write, so the name is resolved here rather than refused
        with `InvalidAnchor / bad_anchor_ref_type`.

        There is no ambiguity to resolve: each of these parameters names one
        entity, so a string can only mean an anchor on it. A non-string falls
        through untouched and the validators below refuse it as before, which
        keeps the diagnosis for a genuinely wrong TYPE (a PartRef, an int)
        exactly what it was.
        """
        if isinstance(value, str):
            return self.anchor(entity, value)
        return value

    def _validate_attach_anchor(self, ref, expected_entity, expected_kind: str,
                                refs, *, label: str):
        """Ensure `ref` is an AnchorRef on this builder, pointing at the
        expected entity (by name + kind).
        """
        if not isinstance(ref, AnchorRef):
            raise InvalidAnchor(
                f"{label} must be an anchor name or AnchorRef, "
                f"got {type(ref).__name__}",
                refs=refs,
                evidence={"reason": "bad_anchor_ref_type", "param": label,
                          "value": type(ref).__name__},
                suggested_repairs=[
                    f"pass an anchor name on {expected_entity._name!r}, or an "
                    f"AnchorRef from scene.anchor(...) on that entity, as {label}",
                ],
            )
        if ref._owner_id != self._owner_id:
            raise InvalidAnchor(
                f"{label} AnchorRef was not produced by this scene",
                refs=refs,
                evidence={"reason": "foreign_anchor_ref", "param": label,
                          "entity": ref._entity_name,
                          "anchor": ref._anchor_name},
                suggested_repairs=[
                    f"use this scene's .anchor(...) for {label}",
                ],
            )
        if ref._entity_kind != expected_kind:
            raise InvalidAnchor(
                f"{label}: AnchorRef entity_kind {ref._entity_kind!r} does "
                f"not match expected entity kind {expected_kind!r}",
                refs=refs,
                evidence={"reason": "anchor_wrong_entity_kind",
                          "value": ref._entity_kind,
                          "expected": expected_kind},
                suggested_repairs=[
                    f"use an anchor on {expected_entity._name!r} as {label}",
                ],
            )
        if ref._entity_name != expected_entity._name:
            raise InvalidAnchor(
                f"{label}: AnchorRef references entity {ref._entity_name!r}, "
                f"but {label} must reference {expected_entity._name!r}",
                refs=refs,
                evidence={"reason": "anchor_entity_mismatch",
                          "anchor_entity": ref._entity_name,
                          "expected_entity": expected_entity._name},
                suggested_repairs=[
                    f"use an anchor on {expected_entity._name!r} as {label}",
                ],
            )

    def _validate_attach_anchor_handle(self, ref, handle, refs, *, label: str):
        """Variant of _validate_attach_anchor for the child_anchor at
        Scene.attach time: the new instance doesn't exist as a ref yet,
        so the anchor must be authored against the ObjectHandle's part
        vocabulary using a part-scoped anchor name.

        Accepts AnchorRefs whose `_entity_kind` is "instance_handle"
        (produced by `Scene.anchor(handle, "<part>.<anchor>")`).
        """
        if not isinstance(ref, AnchorRef):
            raise InvalidAnchor(
                f"{label} must be an anchor name or AnchorRef, "
                f"got {type(ref).__name__}",
                refs=refs,
                evidence={"reason": "bad_anchor_ref_type", "param": label,
                          "value": type(ref).__name__},
                suggested_repairs=[
                    f"pass an anchor name on the object being attached, or "
                    f"scene.anchor(<that object>, <anchor_name>), as {label}",
                ],
            )
        if ref._owner_id != self._owner_id:
            raise InvalidAnchor(
                f"{label} AnchorRef was not produced by this scene",
                refs=refs,
                evidence={"reason": "foreign_anchor_ref", "param": label},
                suggested_repairs=[
                    f"use this scene's .anchor(...) for {label}",
                ],
            )
        if ref._entity_kind != "instance_handle":
            raise InvalidAnchor(
                f"{label}: AnchorRef must be authored against the "
                f"ObjectHandle of the object being attached, got "
                f"entity_kind={ref._entity_kind!r}",
                refs=refs,
                evidence={"reason": "anchor_wrong_entity_kind",
                          "value": ref._entity_kind,
                          "expected": "instance_handle"},
                suggested_repairs=[
                    "pass scene.anchor(<handle>, <anchor_name>) where "
                    "<handle> is the same ObjectHandle being attached",
                ],
            )
        # **Identity, not display name.** Two definitions may legitimately be
        # called "bracket" — a `scene.py` that builds a variant per
        # station does exactly that — and the anchor path was validated against
        # one of them. Comparing names would accept the other and resolve the
        # anchor against parts and declared frames it was never checked
        # against; a `<part>.<declared>` path makes that a KeyError deep in the
        # kernel rather than a build error here. Two *identical* definitions
        # share a content hash and stay interchangeable, which is the case an
        # author actually hits (`make_cabinet()` called twice).
        if ref._entity_id is not None and ref._entity_id != handle._hash:
            same_name = ref._entity_name == handle.name
            raise InvalidAnchor(
                f"{label}: AnchorRef is bound to a different definition than "
                f"the object being attached"
                + (f" — both are named {handle.name!r}, but they are not the "
                   f"same object" if same_name
                   else f" (anchor on {ref._entity_name!r}, attaching "
                        f"{handle.name!r})"),
                refs=refs,
                evidence={"reason": "anchor_handle_mismatch",
                          "anchor_object": ref._entity_name,
                          "anchor_object_ref": ref._entity_id,
                          "handle_object": handle.name,
                          "handle_object_ref": handle._hash},
                suggested_repairs=[
                    f"take {label} from the same value you pass to attach: "
                    f"child_anchor=scene.anchor(<that object>, ...)",
                ],
            )
        if ref._entity_id is None and ref._entity_name != handle.name:
            raise InvalidAnchor(
                f"{label}: AnchorRef bound to object {ref._entity_name!r}, "
                f"but attach handle is {handle.name!r}",
                refs=refs,
                evidence={"reason": "anchor_handle_mismatch",
                          "anchor_object": ref._entity_name,
                          "handle_object": handle.name},
                suggested_repairs=[
                    f"use scene.anchor(<handle for {handle.name!r}>, ...) "
                    f"as {label}",
                ],
            )

    def _validate_anchor_name(self, anchor_name: str, entity,
                              entity_kind: str, refs):
        if not isinstance(anchor_name, str):
            raise InvalidAnchor(
                f"anchor name must be a string, got {type(anchor_name).__name__}",
                refs=refs,
                evidence={"reason": "bad_anchor_type",
                          "value": repr(anchor_name)},
                suggested_repairs=["pass an anchor name string"],
            )
        if entity_kind == "room":
            # Room uses the unified 27-anchor vocabulary (compass-style
            # names like wall_north_center were dropped).
            if anchor_name not in _AABB_ANCHOR_NAMES:
                raise InvalidAnchor(
                    f"unknown room anchor {anchor_name!r}",
                    refs=refs + [_ref("anchor", anchor_name)],
                    evidence={"entity_kind": "room", "value": anchor_name,
                              "builtin_names": _anchors_mod.builtin_anchor_groups()},
                    suggested_repairs=["pick from the auto-AABB anchor "
                                       "vocabulary (top/bottom/front/back/"
                                       "left/right or edge/corner names)"],
                )
            return
        if entity_kind == "group":
            if anchor_name not in _AABB_ANCHOR_NAMES:
                raise InvalidAnchor(
                    f"unknown group anchor {anchor_name!r}",
                    refs=refs + [_ref("anchor", anchor_name)],
                    evidence={"entity_kind": "group", "value": anchor_name,
                              "builtin_names": _anchors_mod.builtin_anchor_groups()},
                    suggested_repairs=["pick from the auto-AABB anchor "
                                       "vocabulary; groups do not support "
                                       "part-scoped anchors"],
                )
            return
        # instance: AABB anchor OR part-scoped <part>.<anchor>
        if anchor_name in _AABB_ANCHOR_NAMES:
            return
        if "." in anchor_name:
            part_name, _dot, sub = anchor_name.partition(".")
            parts = self._instance_parts.get(entity._name, ())
            if part_name not in parts:
                raise InvalidAnchor(
                    f"part {part_name!r} not in object instance {entity._name!r}",
                    refs=refs + [_ref("anchor", anchor_name)],
                    evidence={"entity_kind": "instance",
                              "value": anchor_name,
                              "instance_name": entity._name,
                              "parts": list(parts)},
                    suggested_repairs=[f"pick a part from {list(parts)}"],
                )
            declared = self._instance_part_anchors.get(
                entity._name, {}).get(part_name, frozenset())
            if sub not in _AABB_ANCHOR_NAMES and sub not in declared:
                _raise_unknown_part_anchor(
                    anchor_name, part_name, sub, declared,
                    refs=refs + [_ref("anchor", anchor_name)],
                    where=f"object instance {entity._name!r}",
                    label=None)
            return
        declaring = _parts_declaring(
            self._instance_part_anchors.get(entity._name, {}), anchor_name)
        if declaring:
            _raise_unqualified_declared_anchor(
                anchor_name, declaring, refs + [_ref("anchor", anchor_name)],
                label=None, where=f"instance {entity._name!r}")
        raise InvalidAnchor(
            f"unknown instance anchor {anchor_name!r}",
            refs=refs + [_ref("anchor", anchor_name)],
            evidence={"entity_kind": "instance", "value": anchor_name,
                      "builtin_names": _anchors_mod.builtin_anchor_groups()},
            suggested_repairs=["use an auto-AABB anchor name, or part-scope "
                               "as '<part_name>.<anchor>' — which is also how "
                               "a frame declared with obj.add_anchor is named"],
        )

    def _validate_instance_anchor_path(self, handle: ObjectHandle,
                                       anchor_name: str, refs, *, label: str):
        """Variant of _validate_anchor_name that takes an ObjectHandle
        directly — used at attach() time before the new instance ref
        exists in the entity map."""
        if not isinstance(anchor_name, str):
            raise InvalidAnchor(
                f"{label} must be a string, got {type(anchor_name).__name__}",
                refs=refs,
                evidence={"reason": "bad_anchor_type",
                          "value": repr(anchor_name)},
                suggested_repairs=[f"pass a {label} name string"],
            )
        if anchor_name in _AABB_ANCHOR_NAMES:
            return
        if "." in anchor_name:
            part_name, _dot, sub = anchor_name.partition(".")
            if part_name not in handle.parts:
                raise InvalidAnchor(
                    f"{label}: part {part_name!r} not in object "
                    f"{handle.name!r}",
                    refs=refs + [_ref("anchor", anchor_name)],
                    evidence={"entity_kind": "instance",
                              "value": anchor_name,
                              "object_name": handle.name,
                              "parts": list(handle.parts)},
                    suggested_repairs=[f"pick a part from {list(handle.parts)}"],
                )
            declared = _handle_part_anchors(handle).get(part_name, frozenset())
            if sub not in _AABB_ANCHOR_NAMES and sub not in declared:
                _raise_unknown_part_anchor(
                    anchor_name, part_name, sub, declared,
                    refs=refs + [_ref("anchor", anchor_name)],
                    where=f"object {handle.name!r}",
                    label=label)
            return
        declaring = _parts_declaring(_handle_part_anchors(handle), anchor_name)
        if declaring:
            _raise_unqualified_declared_anchor(
                anchor_name, declaring, refs + [_ref("anchor", anchor_name)],
                label=label, where=f"object {handle.name!r}")
        raise InvalidAnchor(
            f"{label}: unknown anchor {anchor_name!r}",
            refs=refs + [_ref("anchor", anchor_name)],
            evidence={"entity_kind": "instance", "value": anchor_name,
                      "builtin_names": _anchors_mod.builtin_anchor_groups()},
            suggested_repairs=["use an auto-AABB anchor name, or part-scope "
                               "as '<part_name>.<anchor>' — which is also how "
                               "a frame declared with obj.add_anchor is named"],
        )

    # ---- anchor builder --------------------------------------------------

    # -----------------------------------------------------------------
    # Lighting (optional; absent means the frame is lit only by emissive
    # materials — there is no automatic rig, see `set_lighting`)
    # -----------------------------------------------------------------

    def set_lighting(self, block) -> None:
        """Bind this scene's `@lighting` block — its lights, World, and
        photographic settings, as Blender Python.

        This is the whole lighting surface. There is no `add_light`,
        `set_sky`, `set_environment`, or `set_auto_lighting`, and the renderer
        adds no sun, sky, fill, or automatic rig of its own: what the block
        builds is what the frame is lit by. If a scene needs daylight through a
        window, something visible outside it, or practicals after dark, the
        block is where that gets built.

        The block runs in Blender, last — after geometry, cameras, and
        materials — and nothing writes lights, World, exposure, view transform,
        look, or white balance after it. See `sdk/_lighting_block.lighting` for
        the authored form and `render/_blender_child._apply_lighting` for the
        façade it is handed.

        A scene with no block renders unlit. That is deliberate and is the
        replacement for the old automatic rig: an unlit frame is a visible,
        reportable fact, whereas a rig silently supplying light the author
        never asked for is how a scene comes to be lit twice.
        """
        from ._lighting_block import LightingBlock
        self._check_not_finalized()
        if not isinstance(block, LightingBlock):
            raise InvalidScene(
                "set_lighting takes the value a @lighting-decorated function "
                f"evaluates to, got {type(block).__name__}",
                evidence={"param": "block",
                          "constraint": "@lighting function value"},
                suggested_repairs=[
                    "decorate the function with @lighting and pass its name: "
                    "`@lighting\ndef rig(L): ...` then `scene.set_lighting(rig)`",
                ])
        self._lighting = block

    def anchor(self, entity, anchor_name: str, *,
               translation=None, rotation=None) -> AnchorRef:
        """Return an AnchorRef for a named anchor on `entity`.

        `entity` may be an existing entity ref (InstanceRef / GroupRef /
        RoomRef), or an object **definition** in any of the three spellings
        `place` and `attach` accept — an `ObjectHandle` from the injected
        `objects` catalog, an `Object` builder authored in this file, or the
        finalized ObjectIR dict a builder returned. Definition-bound refs carry
        entity_kind `"instance_handle"`; they are what `Scene.attach` takes as
        `child_anchor`, since the instance being attached does not exist yet.

        **Passing an `Object` builder finalizes it.** That is the same
        lifecycle `place` / `attach` already have, and it has to happen here
        too: the anchor path is checked against the definition's parts and
        declared anchors, which are only settled at `finalize`. `finalize` is
        idempotent, so the later `attach` call costs nothing — but the object
        is frozen from this call onward, so finish adding parts, joints, and
        anchors before taking an anchor on it.

        `anchor_name` is an auto-AABB landmark (`"bottom"`), or part-scoped as
        `"<part>.<landmark>"`, or a **declared** semantic frame as
        `"<part>.<declared_name>"` — the frame an author registered with
        `Object.add_anchor`, carrying its authored rotation as well as its
        position. Part-qualification is what makes `"body.mount"` and
        `"lid.mount"` two different frames.

        Optional `translation` (meters) and `rotation` (intrinsic Euler XYZ
        degrees) bake an offset into the ref, composed onto the resolved anchor
        frame as `T_anchor ∘ T(translation) ∘ R(rotation)`. For a built-in
        landmark the anchor frame has identity rotation, so the translation
        reads as entity-local axes; on a declared frame with an authored
        rotation the offset is applied **in that frame's own axes**, which is
        the convention `Object`'s joint anchors already use (see
        `object_builder._anchor_to_se3`) and is what lets "5 mm out along the
        mounting face's normal" mean that regardless of how the face is turned.
        For chained composition use `.offset_by(...)` on the returned ref.
        """
        self._check_not_finalized()
        self._check_room_declared("anchor")
        if not isinstance(entity, (InstanceRef, GroupRef, RoomRef)):
            entity = self._validate_object_handle(
                entity, "anchor",
                [_ref("scene", self._name), _ref("instance_handle", None)])
        entity_id = None
        if isinstance(entity, ObjectHandle):
            kind = "instance_handle"
            entity_name = entity.name
            entity_id = entity._hash
            anchor_refs = [_ref("scene", self._name),
                           _ref("instance_handle", entity_name)]
            self._validate_instance_anchor_path(
                entity, anchor_name, anchor_refs, label="anchor_name")
        else:
            self._check_owned(entity, InvalidAnchor)
            kind = self._entity_kind_of(entity)
            entity_name = entity._name
            anchor_refs = [_ref("scene", self._name), _ref(kind, entity_name)]
            self._validate_anchor_name(anchor_name, entity, kind, anchor_refs)
        from ._refs import freeze_anchor_offset, IDENTITY_ANCHOR_OFFSET
        if translation is None and rotation is None:
            offset = IDENTITY_ANCHOR_OFFSET
        else:
            try:
                offset = freeze_anchor_offset(translation, rotation)
            except (TypeError, ValueError) as e:
                raise InvalidAnchor(
                    f"anchor offset on {anchor_name!r}: {e}",
                    refs=anchor_refs + [_ref("anchor", anchor_name)],
                    evidence={"reason": "bad_anchor_offset",
                              "translation": repr(translation),
                              "rotation": repr(rotation)},
                    suggested_repairs=[
                        "pass translation as a finite 3-tuple in meters; "
                        "pass rotation as a finite 3-tuple of Euler XYZ degrees",
                    ],
                ) from None
        return AnchorRef(kind, entity_name, anchor_name, self._owner_id,
                         offset=offset, entity_id=entity_id)

    # ---- assertions ------------------------------------------------------
    #
    # Validation + record building uses the shared `sdk/_assertions.py`
    # helpers — same logic drives the object SDK's assertion API.

    def assert_distance(self, *, a: AnchorRef, b: AnchorRef,
                        min, max, axis=None, poses=None,
                        name: str) -> AssertionRef:
        """Distance between `a` and `b` in [min, max].

        `axis=None` -> Euclidean magnitude |b - a| (must have min >= 0).
        `axis=(x, y, z)` -> signed projection (b - a) . unit(axis); `min`
        may be negative.
        """
        self._check_not_finalized()
        self._check_room_declared("assert_distance")
        a_refs = [_ref("scene", self._name), _ref("assertion", name)]
        _assertions.check_assertion_name(
            name, self._assertion_name_to_id, InvalidAssertion, refs=a_refs)
        _assertions.check_anchor_ref_owned(
            a, self._owner_id, "a", InvalidAssertion, refs=a_refs)
        _assertions.check_anchor_ref_owned(
            b, self._owner_id, "b", InvalidAssertion, refs=a_refs)

        if axis is None:
            axis_unit = None
            _assertions.check_finite_or_inf(min, "min", InvalidAssertion,
                                            refs=a_refs, allow_neg_inf=False)
            _assertions.check_finite_or_inf(max, "max", InvalidAssertion,
                                            refs=a_refs, allow_neg_inf=False)
            if float(min) < 0.0:
                raise InvalidAssertion(
                    f"min {min} must be >= 0 (Euclidean distance is non-negative)",
                    refs=a_refs,
                    evidence={"reason": "negative_min", "value": float(min)},
                    suggested_repairs=[
                        "use min >= 0 for Euclidean distance; for signed "
                        "directional bounds pass axis=(x, y, z)"
                    ],
                )
        else:
            axis_unit = _assertions.check_axis(axis, InvalidAssertion, refs=a_refs)
            _assertions.check_finite_or_inf(min, "min", InvalidAssertion, refs=a_refs)
            _assertions.check_finite_or_inf(max, "max", InvalidAssertion, refs=a_refs)
        if not (float(min) <= float(max)):
            raise InvalidAssertion(
                f"min {min} must be <= max {max}",
                refs=a_refs,
                evidence={"reason": "min_gt_max", "min": float(min),
                          "max": float(max)},
                suggested_repairs=["swap min and max so min <= max"],
            )
        assertion_id = len(self._assertions)
        from ._refs import freeze_poses as _freeze_poses
        record = _assertions.build_distance_record(
            assertion_id=assertion_id, name=name,
            a=a, b=b, axis_unit=axis_unit, lo=min, hi=max,
            assertion_poses=_freeze_poses(poses),
            joint_types=self._joint_types_map(),
        )
        self._assertions.append(record)
        self._assertion_name_to_id[name] = assertion_id
        return AssertionRef(assertion_id, self._owner_id, name, "distance")

    def assert_collinear(self, *anchors, axis, tol: float = 1.0e-3,
                         poses=None, name: str) -> AssertionRef:
        self._check_not_finalized()
        self._check_room_declared("assert_collinear")
        a_refs = [_ref("scene", self._name), _ref("assertion", name)]
        _assertions.check_assertion_name(
            name, self._assertion_name_to_id, InvalidAssertion, refs=a_refs)
        if len(anchors) < 2:
            raise InvalidAssertion(
                f"assert_collinear requires at least 2 anchors, got {len(anchors)}",
                refs=a_refs,
                evidence={"reason": "too_few_anchors", "value": len(anchors)},
                suggested_repairs=["pass two or more AnchorRefs"],
            )
        for i, ar in enumerate(anchors):
            _assertions.check_anchor_ref_owned(
                ar, self._owner_id, f"anchors[{i}]",
                InvalidAssertion, refs=a_refs)
        axis_unit = _assertions.check_axis(axis, InvalidAssertion, refs=a_refs)
        _assertions.check_finite_scalar(tol, "tol", InvalidAssertion, refs=a_refs)
        if float(tol) < 0.0:
            raise InvalidAssertion(
                f"tol {tol} must be >= 0",
                refs=a_refs,
                evidence={"reason": "negative_tol", "value": float(tol)},
                suggested_repairs=["use tol >= 0"],
            )
        assertion_id = len(self._assertions)
        from ._refs import freeze_poses as _freeze_poses
        record = _assertions.build_collinear_record(
            assertion_id=assertion_id, name=name,
            anchors=list(anchors), axis_unit=axis_unit, tol=tol,
            assertion_poses=_freeze_poses(poses),
            joint_types=self._joint_types_map(),
        )
        self._assertions.append(record)
        self._assertion_name_to_id[name] = assertion_id
        return AssertionRef(assertion_id, self._owner_id, name, "collinear")

    def assert_overlap(self, *, a, b, axis, min, max=None, poses=None,
                       name: str) -> AssertionRef:
        """Signed interval overlap of two entity AABBs along `axis`.

        Entities may be Instance / Group / Room refs. Positive = overlap
        depth; negative = gap distance. Result must lie in [min, max].
        `max=None` defaults to +inf.
        """
        self._check_not_finalized()
        self._check_room_declared("assert_overlap")
        a_refs = [_ref("scene", self._name), _ref("assertion", name)]
        _assertions.check_assertion_name(
            name, self._assertion_name_to_id, InvalidAssertion, refs=a_refs)
        _assertions.check_entity_ref_owned(
            a, self._owner_id, "a", InvalidAssertion, refs=a_refs,
            allowed_kinds=("InstanceRef", "GroupRef", "RoomRef"))
        _assertions.check_entity_ref_owned(
            b, self._owner_id, "b", InvalidAssertion, refs=a_refs,
            allowed_kinds=("InstanceRef", "GroupRef", "RoomRef"))
        axis_unit = _assertions.check_axis(axis, InvalidAssertion, refs=a_refs)
        _assertions.check_finite_or_inf(min, "min", InvalidAssertion, refs=a_refs)
        if max is None:
            max = float("inf")
        _assertions.check_finite_or_inf(max, "max", InvalidAssertion, refs=a_refs)
        if not (float(min) <= float(max)):
            raise InvalidAssertion(
                f"min {min} must be <= max {max}",
                refs=a_refs,
                evidence={"reason": "min_gt_max", "min": float(min),
                          "max": float(max)},
                suggested_repairs=["swap min and max so min <= max"],
            )
        assertion_id = len(self._assertions)
        from ._refs import freeze_poses as _freeze_poses
        record = _assertions.build_overlap_record(
            assertion_id=assertion_id, name=name,
            a=a, b=b, axis_unit=axis_unit, lo=min, hi=max,
            assertion_poses=_freeze_poses(poses),
            joint_types=self._joint_types_map(),
        )
        self._assertions.append(record)
        self._assertion_name_to_id[name] = assertion_id
        return AssertionRef(assertion_id, self._owner_id, name, "overlap")

    def assert_angle(self, *, a, b, min_deg: float, max_deg: float,
                     poses=None, name: str) -> AssertionRef:
        """Angle (degrees) between two DirectionRefs in [min_deg, max_deg]."""
        self._check_not_finalized()
        self._check_room_declared("assert_angle")
        a_refs = [_ref("scene", self._name), _ref("assertion", name)]
        _assertions.check_assertion_name(
            name, self._assertion_name_to_id, InvalidAssertion, refs=a_refs)
        _assertions.check_direction_ref_owned(
            a, self._owner_id, "a", InvalidAssertion, refs=a_refs)
        _assertions.check_direction_ref_owned(
            b, self._owner_id, "b", InvalidAssertion, refs=a_refs)
        _assertions.check_finite_scalar(min_deg, "min_deg", InvalidAssertion,
                                         refs=a_refs)
        _assertions.check_finite_scalar(max_deg, "max_deg", InvalidAssertion,
                                         refs=a_refs)
        lo, hi = float(min_deg), float(max_deg)
        if lo < 0.0 or hi > 180.0 or lo > hi:
            raise InvalidAssertion(
                f"angle bounds must satisfy 0 <= min_deg <= max_deg <= 180, "
                f"got [{lo}, {hi}]",
                refs=a_refs,
                evidence={"reason": "bad_angle_bounds",
                          "min_deg": lo, "max_deg": hi},
                suggested_repairs=[
                    "use bounds within [0, 180] with min_deg <= max_deg"
                ],
            )
        assertion_id = len(self._assertions)
        from ._refs import freeze_poses as _freeze_poses
        record = _assertions.build_angle_record(
            assertion_id=assertion_id, name=name,
            dir_a=a, dir_b=b, min_deg=lo, max_deg=hi,
            assertion_poses=_freeze_poses(poses),
            joint_types=self._joint_types_map(),
        )
        self._assertions.append(record)
        self._assertion_name_to_id[name] = assertion_id
        return AssertionRef(assertion_id, self._owner_id, name, "angle")

    # ---- review criteria -------------------------------------------------

    MAX_CHECKS = 16

    def add_check(self, claim: str) -> None:
        """Commit to an observable claim the finished scene must satisfy.

        The scene counterpart of `Object.add_check`, and the same contract: an
        independent reviewer sees the rendered scene and this list, and answers
        each claim `met`, `not_met`, or `cant_tell`. It never blocks `finish`.

        Write each claim as a **relation a viewer could see**, not a tolerance
        and not a recipe:

          - good: "the dishwasher front sits flush with the cabinet run"
          - too much: "the dishwasher front is within 3 mm of the cabinet line"
          - too little: "the kitchen looks right"

        Scene claims are about relationships between things — alignment,
        clearance, reachability, whether a run reads as continuous — because
        that is what a scene adds over the objects in it. Anything measurable
        belongs in the `assert_*` family, which computes it exactly; a check is
        for what only looking can settle.

        At most `MAX_CHECKS`, and duplicates are rejected: both keep the list
        short enough to be read rather than skimmed.
        """
        self._check_not_finalized()
        refs = [_ref("scene", self._name)]
        if not isinstance(claim, str):
            raise InvalidScene(
                f"add_check: claim must be a string, got "
                f"{type(claim).__name__}",
                refs=refs,
                evidence={"reason": "claim_not_string",
                          "value": type(claim).__name__},
                suggested_repairs=["pass one observable claim as a string"],
            )
        text = " ".join(claim.split())
        if not text:
            raise InvalidScene(
                "add_check: claim is empty", refs=refs,
                evidence={"reason": "claim_empty"},
                suggested_repairs=[
                    "state what a viewer would see, e.g. "
                    "\"the sink is centered under the window\""],
            )
        if text in self._check_claims:
            raise InvalidScene(
                f"add_check: duplicate claim {text!r}", refs=refs,
                evidence={"reason": "duplicate_claim", "value": text},
                suggested_repairs=["drop the duplicate, or narrow one of them"],
            )
        if len(self._checks) >= self.MAX_CHECKS:
            raise InvalidScene(
                f"add_check: at most {self.MAX_CHECKS} checks per scene "
                f"({len(self._checks)} already declared)",
                refs=refs,
                evidence={"reason": "too_many_checks",
                          "limit": self.MAX_CHECKS,
                          "declared": len(self._checks)},
                suggested_repairs=[
                    "keep the claims that would be invisible in a compile "
                    "error and fatal to the scene, and drop the rest",
                    "move anything numeric to the assert_* family instead"],
            )
        self._checks.append({"id": f"c{len(self._checks) + 1:02d}",
                             "claim": text})
        self._check_claims.add(text)

    # ---- sugar assertions ------------------------------------------------

    def assert_contact(self, *, a, b, tol: float, name: str):
        return _sugar.assert_contact(self, a=a, b=b, tol=tol, name=name)

    def assert_within(self, *, inner, outer, axes=("x", "y", "z"),
                      margin: float = 0.0, name: str):
        return _sugar.assert_within(self, inner=inner, outer=outer,
                                    axes=axes, margin=margin, name=name)

    def assert_aabb_overlap(self, *, a, b, axes, min_overlap: float, name: str):
        return _sugar.assert_aabb_overlap(self, a=a, b=b, axes=axes,
                                          min_overlap=min_overlap, name=name)

    def assert_parallel(self, *, a, b, tol_deg: float = 1.0, name: str):
        return _sugar.assert_parallel(self, a=a, b=b,
                                       tol_deg=tol_deg, name=name)

    def assert_antiparallel(self, *, a, b, tol_deg: float = 1.0, name: str):
        return _sugar.assert_antiparallel(self, a=a, b=b,
                                           tol_deg=tol_deg, name=name)

    def assert_perpendicular(self, *, a, b, tol_deg: float = 1.0, name: str):
        return _sugar.assert_perpendicular(self, a=a, b=b,
                                            tol_deg=tol_deg, name=name)

    def assert_flush(self, *, face_a, face_b, axis_world,
                     tol_pos: float = 1.0e-3, tol_deg: float = 1.0,
                     opposing: bool = True, name: str):
        return _sugar.assert_flush(
            self, face_a=face_a, face_b=face_b, axis_world=axis_world,
            tol_pos=tol_pos, tol_deg=tol_deg, opposing=opposing, name=name)

    # ---- finalize --------------------------------------------------------

    def finalize(self) -> dict:
        if self._finalized:
            return self._cached_ir  # idempotent

        if self._room is None:
            raise RoomNotDeclared(
                f"scene {self._name!r}: set_room was never called",
                refs=[_ref("scene", self._name)],
                evidence={"reason": "room_never_declared"},
                suggested_repairs=["call scene.set_room(...) before finalize()"],
            )
        if not self._instances:
            raise NoInstances(
                f"scene {self._name!r}: no instances placed",
                refs=[_ref("scene", self._name)],
                evidence={"reason": "no_instances"},
                suggested_repairs=["call scene.place(...) or scene.attach(...)"
                                   " at least once before finalize()"],
            )

        # Walk every entity's parent chain to verify it terminates at None
        # (a room-placed root) or at the room. No cycles, no orphans.
        # Cycle prevention is already enforced eagerly at attach_group, but
        # this is a defensive double-check.
        for n in list(self._instance_name_to_id.keys()) + \
                list(self._group_name_to_id.keys()):
            seen = set()
            cur = n
            while True:
                if cur in seen:
                    raise InvalidGroup(
                        f"cycle detected in attachment forest at {cur!r}",
                        refs=[_ref("scene", self._name)],
                        evidence={"reason": "cycle_at_finalize",
                                  "value": cur},
                        suggested_repairs=["restructure the attach forest"],
                    )
                seen.add(cur)
                p = self._parent_of.get(cur)
                if p is None or p == _ROOM_RESERVED_NAME:
                    break
                cur = p

        # Materialize instances list: a group root's own placement is
        # shadowed (set to null in the IR) once the containing group has
        # been attached. We also normalize each placement's float fields.
        # Group roots that have NOT been attached keep their own
        # placement.
        instances_out = []
        for inst in self._instances:
            # Determine whether this instance is the root of any
            # attached group; if so, shadow its placement.
            placement = inst["placement"]
            for g in self._groups:
                if g["root"] == inst["name"] and g["placement"] is not None:
                    placement = None
                    break
            inst_out = {
                "id": inst["id"],
                "name": inst["name"],
                "object_ref": inst["object_ref"],
                "object_name": inst["object_name"],
                "placement": placement,
            }
            # Carried through even for a group root whose placement is
            # shadowed: grouping decides where the body is, not what
            # configuration it is in.
            if inst.get("initial_joints"):
                inst_out["initial_joints"] = dict(inst["initial_joints"])
            instances_out.append(inst_out)

        # Groups: same shadow logic for nested groups (if a group is the
        # root of an outer group with a placement).
        groups_out = []
        for g in self._groups:
            placement = g["placement"]
            # Check if this group is the root of an outer attached group.
            for outer in self._groups:
                if outer is g:
                    continue
                if outer["root"] == g["name"] and outer["placement"] is not None:
                    placement = None
                    break
            groups_out.append({
                "id": g["id"],
                "name": g["name"],
                "members": list(g["members"]),
                "root": g["root"],
                "placement": placement,
            })

        # v2 exactly when some instance carries a saved configuration, so a
        # scene without the feature serializes and hashes byte-for-byte as it
        # did before `initial_joints` existed. See `sdk/ir.py`.
        schema = (SCENE_SCHEMA_VERSION_V3
                  if any(i.get("initial_joints") for i in instances_out)
                  else SCENE_SCHEMA_VERSION)
        ir_no_id = {
            "schema": schema,
            "name": self._name,
            "room": dict(self._room),
            "instances": instances_out,
            "groups": groups_out,
            "assertions": list(self._assertions),
        }
        # Omitted when absent, like `Material.texture`: the IR is content-
        # hashed whole, so an unconditional key would re-id every scene that
        # has ever been built. A v2 scene carries `lighting` and never
        # `lights` / `sky` / `environment` / `auto_lighting` — those are v1
        # keys, and the renderer branches on the schema version rather than on
        # which keys happen to be present, so the two can never be mixed.
        #
        # `checks` is omitted when the author declared none, for the reason the
        # v1 lighting keys were: the IR is content-hashed whole, so an
        # unconditional key would re-id every scene that has ever been built.
        if self._checks:
            ir_no_id["checks"] = list(self._checks)
        if self._lighting is not None:
            ir_no_id["lighting"] = self._lighting.to_ir()

        self._cached_ir = insert_id(ir_no_id)
        self._finalized = True
        return self._cached_ir

