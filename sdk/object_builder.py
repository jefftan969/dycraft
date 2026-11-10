"""Object builder for the DyCraft object SDK.

This module exposes the single global `Object` that the sandbox injects,
plus the opaque `PartRef` / `JointRef` wrappers that builder methods return.

Constant-time validations raise at the offending call. Structural
(O(n)) checks fire inside `finalize()`.
"""

import math
import re
import threading
from pathlib import Path

from . import _rotation as _rot
from . import _anchors as _anchors_mod
from . import _assertions as _assertions
from . import _sugar as _sugar
from . import _primitives as _prim
from . import _mesh_assets as _mesh_assets
from ._material import (Material, check_material, DEFAULT_PART_COLOR,
                        SHADER_UV_SCALE_M)
from . import _refs as _refs_mod
from ._refs import AnchorRef, AssertionRef
from .ir import OBJECT_SCHEMA_VERSION, content_hash, insert_id
from ._errors import (
    BuildError,
    InvalidPrimitive,
    InvalidJoint,
    DanglingReference,
    RootNotDeclared,
    NotConnectedToRoot,
    InvalidAnchor,
    InvalidAssertion,
)
from . import _readonly as _ro


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_MAX_LEN = 64
_AXIS_MIN_NORM = 1e-12

_JOINT_FIXED = "fixed"
_JOINT_REVOLUTE = "revolute"
_JOINT_PRISMATIC = "prismatic"

_PRIMITIVE_BOX = "box"
_PRIMITIVE_SOLID = "solid"


def _jsonify_shape(value):
    """Recursively normalize a solid's ``shape`` table to JSON-native types.

    The canonical form that produces the object id is ``json.dumps(...)``, and
    a tuple and a list serialize identically — but only after this pass, since
    the runner's JSON round-trip yields lists while a hand-built or
    ``_primitives``-validated fragment can hold tuples. Normalizing here keeps
    the object id independent of which path the table arrived by. Values are
    otherwise untouched: the runner is responsible for the fixed-point rounding
    that makes the floats stable across OCCT versions.
    """
    if _ro.is_mapping(value):
        return {str(k): _jsonify_shape(v) for k, v in value.items()}
    if _ro.is_sequence(value):
        return [_jsonify_shape(v) for v in value]
    return value

# Default material when none is supplied. Matches the renderer's prior
# flat-gray Lambertian-looking appearance (color 176 ≈ 0.69 in [0,1] floats,
# roughness 0.9, metallic 0.0). Imported from sdk._material.
DEFAULT_MATERIAL = Material(color=DEFAULT_PART_COLOR)

def _texture_scale_m(mat):
    """The tile size the UV baker needs for `mat`, or None for a bare surface.

    Three cases, and the middle one is the one that is easy to miss.

    A material with maps carries a `Mapping`, and its `scale_m` says how much
    geometry one tile of the image spans. A material with neither maps nor a
    shader needs no coordinates at all, and passing None keeps its mesh bytes
    byte-identical to a pre-texture build — which is what keeps every stored
    object's id stable.

    **A `shader` material carries no `Mapping` and still needs UVs.** There is
    no image to lay out, so there is nothing for a `Mapping` to size, but a
    graph reading `ShaderNodeTexCoord`'s `UV` output needs a layer to read —
    and with None nothing is baked and every UV-driven graph renders one flat
    color, silently. `SHADER_UV_SCALE_M` is 1.0, so the coordinates are meters
    and the graph does its own scaling with a Mapping node.
    """
    mapping = getattr(mat, "mapping", None)
    if mapping is not None:
        return mapping.scale_m
    if getattr(mat, "shader", None) is not None:
        return SHADER_UV_SCALE_M
    return None


#: The only value `anchor.source` has ever held. Retained in the IR for hash
#: compatibility with already-staged objects; see `Object.add_anchor`.
_ANCHOR_SOURCE = "authored"

# How many Objects one sandbox execution may construct, and how many it has.
# The agent layer resets both between sandbox runs (_reset_object_singleton).
#
# **The limit is per-layer, not universal.** `object.py` builds exactly one
# object and assigns it to `ir`, so a second `Object(...)` there is a mistake
# worth naming — the guard has caught it since the layer existed. A
# `scene.py` builds every object in the scene, so the same guard would forbid
# the whole point of that layer. `limit=None` lifts it; the object layer keeps
# passing 1 and gets the identical error it always did.
_object_singleton_lock = threading.Lock()
_objects_constructed = 0
_object_limit: "int | None" = 1

# Monotonic owner ids, handed out under the same lock.
#
# **Not `id(self)`, and the difference only bites once a build makes more than
# one Object.** `id()` is an address, and CPython reuses an address as soon as
# the object at it is collected — so a `scene.py` that builds cabinets in a
# loop, dropping each builder as it finalizes, can hand two different Objects
# the same `_owner_id`. Every cross-object check in this file
# (`_check_owned`, the anchor and joint validators) is an `_owner_id`
# comparison, so the collision does not raise: it silently accepts a PartRef
# from one object inside another and builds a wrong joint tree. A counter
# cannot be reused, so the check means what it says however many objects a
# build makes. `-1` is `WORLD_DIRECTION_OWNER_ID`, so this starts at 1.
_next_owner_id = 0

# Every ObjectIR finalized during this execution, keyed by content hash.
#
# **This is how a `scene.py` produces a catalog.** A SceneIR names its
# objects by hash and nothing else (`instances[].object_ref`), so the object
# definitions have to reach the harness by some path other than the scene IR.
# No catalog is staged before a scene build, because the objects are what the
# run is authoring.
# The builders drain into a module-level registry, keyed by digest because the
# IR carries only the digest.
#
# Keyed by hash rather than by name so that placing one definition twice
# contributes ONE catalog entry — which is the sharing an author asks for by
# reusing a definition — and so that two definitions that happen to share a
# name cannot silently overwrite each other. Names stay advisory; the bundle
# manifest is where they have to be made unique.
_finalized_objects: dict = {}


def take_finalized_objects() -> dict:
    """Drain and return `{ir_hash: ObjectIR}` for this execution's objects.

    Draining rather than reading: a second build in the same process must not
    inherit the previous one's catalog.
    """
    global _finalized_objects
    with _object_singleton_lock:
        out = _finalized_objects
        _finalized_objects = {}
    return out


def register_external_object(ir: dict) -> dict:
    """Adopt an already-finalized ObjectIR into this execution's catalog.

    ``Object.finalize()`` is the only other writer of ``_finalized_objects``,
    and that is a deliberate closure: a definition reaches the catalog because
    this build authored it. An *imported* asset breaks the premise rather than
    the rule. It carries no ``source`` and no ``brep``, and no authoring
    surface can emit its ``imported`` primitive at all (``_primitives`` states
    the freeze structurally), so there is no program to run and no builder to
    finalize. Adoption is the only path its definition can take, and without
    one the asset is placeable but uncatalogued — ``Scene.place`` accepts the
    dict and records a correct ``object_ref``, then every catalog consumer
    either skips the instance or raises ``DanglingReference`` on it.

    The hash is recomputed here rather than trusted. This is the boundary
    where a dict assembled outside the SDK enters trusted state, and the
    launcher's check ran against the file on disk, which is not necessarily
    the object handed over now. Same rule the catalog gate applies: the id is
    the hash of everything else.

    Returns the IR unchanged, so a caller can place what it registered.
    """
    if not isinstance(ir, dict):
        raise InvalidPrimitive(
            f"register_external_object expects an ObjectIR dict, got "
            f"{type(ir).__name__}")
    if ir.get("schema") != OBJECT_SCHEMA_VERSION:
        raise InvalidPrimitive(
            f"external object schema is {ir.get('schema')!r}, expected "
            f"{OBJECT_SCHEMA_VERSION!r}")
    if not ir.get("parts"):
        raise InvalidPrimitive("external object has no parts")
    stated = ir.get("id")
    recomputed = content_hash({k: v for k, v in ir.items() if k != "id"})
    if stated != recomputed:
        raise InvalidPrimitive(
            f"external object content hash mismatch: it states id={stated!r} "
            f"but its content hashes to {recomputed!r}. The IR was edited "
            f"after it was frozen; re-run the converter rather than patching "
            f"the file.")
    with _object_singleton_lock:
        _finalized_objects[stated] = ir
    return ir


def _reset_object_singleton(limit: "int | None" = 1):
    """Test / host hook: clear the per-execution Object count.

    `limit` is how many Objects the next execution may construct — 1 for the
    object layer's one-object-per-`object.py` contract, None for a scene
    build that authors every object it places. Owner ids deliberately do
    NOT reset: they must stay unique across executions in a warm CAD worker,
    where a ref captured by one build must never validate against the next.
    """
    global _objects_constructed, _object_limit, _finalized_objects
    with _object_singleton_lock:
        _objects_constructed = 0
        _object_limit = limit
        # A build that raised part-way through may have finalized objects the
        # scene never placed. Clearing at the start of every build — rather
        # than only on the drain — is what keeps a failed attempt out of the
        # next one's catalog.
        _finalized_objects = {}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_name(name, kind: str, *, obj_name: str | None = None):
    refs = [_object_ref(obj_name)] if obj_name is not None else []
    if not isinstance(name, str):
        raise InvalidPrimitive(
            f"{kind} name must be a string, got {type(name).__name__}",
            refs=refs,
            evidence={"param": f"{kind}.name", "value": repr(name),
                      "constraint": "type:str"},
            suggested_repairs=[f"pass a string for the {kind} name"],
        )
    if len(name) == 0 or len(name) > _NAME_MAX_LEN:
        raise InvalidPrimitive(
            f"{kind} name length must be 1..{_NAME_MAX_LEN}, got {len(name)}",
            refs=refs,
            evidence={"param": f"{kind}.name", "value": name,
                      "constraint": f"length in 1..{_NAME_MAX_LEN}"},
            suggested_repairs=["shorten the name to 64 chars or fewer"],
        )
    if not _NAME_RE.match(name):
        raise InvalidPrimitive(
            f"{kind} name {name!r} does not match {_NAME_RE.pattern}",
            refs=refs,
            evidence={"param": f"{kind}.name", "value": name,
                      "constraint": f"regex {_NAME_RE.pattern}"},
            suggested_repairs=["start name with a letter or underscore; "
                               "use only letters, digits, underscores"],
        )


def _check_finite_scalar(x, label: str, *, refs=()):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise InvalidPrimitive(
            f"{label} must be a number, got {type(x).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": repr(x), "constraint": "type:number"},
        )
    if not math.isfinite(float(x)):
        raise InvalidPrimitive(
            f"{label} must be finite, got {x}",
            refs=list(refs),
            evidence={"param": label, "value": x, "constraint": "finite"},
        )


def _check_vec3(v, label: str, *, refs=()):
    if not (isinstance(v, tuple) and len(v) == 3):
        raise InvalidPrimitive(
            f"{label} must be a 3-tuple, got {v!r}",
            refs=list(refs),
            evidence={"param": label, "value": repr(v), "constraint": "3-tuple"},
        )
    for i, c in enumerate(v):
        _check_finite_scalar(c, f"{label}[{i}]", refs=refs)


def _check_se3(se3, label: str, *, refs=()):
    if not (isinstance(se3, tuple) and len(se3) == 2):
        raise InvalidPrimitive(
            f"{label} must be a 2-tuple ((tx,ty,tz),(rx,ry,rz)), got {se3!r}",
            refs=list(refs),
            evidence={"param": label, "value": repr(se3),
                      "constraint": "((tx,ty,tz),(rx,ry,rz))"},
        )
    _check_vec3(se3[0], f"{label}.translation", refs=refs)
    _check_vec3(se3[1], f"{label}.rotation_euler_deg", refs=refs)


# ---------------------------------------------------------------------------
# refs helpers
# ---------------------------------------------------------------------------

def _solid_failure_repairs(compiler, err) -> list:
    """Repair advice for a `SolidCompileError`, most specific first.

    Three tiers, and the order is the point — the agent acts on
    ``suggested_repairs[0]``:

      1. a **recognized kernel message** (`failure_hint`), which names the exact
         move for a failure OCCT has already been observed to produce;
      2. the **kind's** advice (`kind_repairs`), which says what to do about
         this class of failure;
      3. nothing further — an unrecognized kind adds no invented advice, for the
         same reason `failure_hint` returns None rather than guessing.

    Both are reached through the injected compiler rather than by importing the
    agent layer, which would invert the sdk -> agent layering.
    """
    out: list = []
    hint_fn = getattr(compiler, "failure_hint", None)
    hint = hint_fn(err.message) if hint_fn else None
    if hint:
        out.append(hint)
    repairs_fn = getattr(compiler, "kind_repairs", None)
    if repairs_fn is not None:
        out.extend(repairs_fn(err.kind))
    return out


def _object_ref(name, id_=None):
    return {"kind": "object", "name": name, "id": id_}


def _part_ref_dict(name, id_=None):
    return {"kind": "part", "name": name, "id": id_}


def _joint_ref_dict(name, id_=None):
    return {"kind": "joint", "name": name, "id": id_}


def _anchor_provenance(anchor) -> dict | None:
    """Symbolic record of which semantic anchor produced a joint transform.

    Returns None for a built-in AABB anchor, so a joint that uses only built-ins
    gains no field. Anchors are part of the content hash, so an empty record
    would move the object id of every such joint for no semantic gain.

    The offset is serialized as a canonical WXYZ quaternion rather than the
    Euler triple assertions use. Euler is not canonical — measured,
    (180, 180, 180) round-trips to a different triple denoting the same frame —
    so a byte comparison of Euler provenance would report false mismatches in
    the consistency check. The quaternion round-trips to ~1e-16.
    """
    if anchor._anchor_name in _anchors_mod.AUTO_AABB_ANCHOR_NAMES:
        return None
    out = {"anchor": anchor._anchor_name}
    if anchor._offset != _refs_mod.IDENTITY_ANCHOR_OFFSET:
        (tx, ty, tz), (rx, ry, rz) = anchor._offset
        q = _rot.sign_canonicalize_quat(
            _rot.normalize_quat(_rot.euler_xyz_deg_to_quat_wxyz(rx, ry, rz)))
        out["offset"] = {
            "translation": [float(tx), float(ty), float(tz)],
            "rotation": [float(v) for v in q],
        }
    return out


def _se3_internal_to_ir(se3):
    """Internal SE3 ((t),(quat_wxyz)) -> IR dict form.

    Anchors resolve to a quaternion directly, so routing them through the
    API-surface Euler form would add a non-canonical round trip for nothing.
    """
    (tx, ty, tz), q = se3
    q = _rot.sign_canonicalize_quat(_rot.normalize_quat(q))
    return {
        "translation": [float(tx), float(ty), float(tz)],
        "rotation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


# ---------------------------------------------------------------------------
# Opaque refs
# ---------------------------------------------------------------------------

class _Ref:
    """Base ref with optional `_poses` for pose-tagging via `.at(...)`.

    Identity (eq/hash) includes `_poses` so a pose-tagged copy is value-
    distinct from the untagged ref. Subclasses must keep `_poses` in
    sync with `__slots__`.
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
        """The author-supplied name of this ref's entity (part / joint)."""
        return self._name

    @property
    def poses(self) -> dict:
        """Pose overrides applied via `.at({...})`. Empty dict if untagged.

        Read-only view; mutating the returned dict has no effect on the ref.
        """
        from ._refs import poses_to_dict
        return poses_to_dict(self._poses)

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._id == other._id
            and self._owner_id == other._owner_id
            and self._poses == other._poses
        )

    def __hash__(self):
        return hash((type(self).__name__, self._id, self._owner_id, self._poses))


class PartRef(_Ref):
    __slots__ = ()

    def at(self, poses) -> "PartRef":
        """Return a pose-tagged copy.

        Used as an EntityRef in `assert_overlap(...)`: pose-tagging a
        PartRef means the part's world-frame AABB is recomputed under
        the supplied joint poses (others at their authored defaults).
        Pose keys not on this part's ancestor chain are silent no-ops.
        """
        from ._refs import _merge_pose_overrides
        return PartRef(self._id, self._owner_id, self._name,
                       _merge_pose_overrides(self._poses, poses))

    def __repr__(self):
        base = f"PartRef(name={self._name!r}, id={self._id}"
        if self._poses:
            from ._refs import poses_to_dict
            base += f", poses={poses_to_dict(self._poses)}"
        return base + ")"


# Joint-type constants used by JointRef to gate `.axis()` (fixed joints
# raise) and by record builders. Mirrors the strings in
# `_JOINT_FIXED/_JOINT_REVOLUTE/_JOINT_PRISMATIC` at module top.
_JOINT_KIND_FIXED = "fixed"
_JOINT_KIND_REVOLUTE = "revolute"
_JOINT_KIND_PRISMATIC = "prismatic"
_JOINT_KINDS_WITH_AXIS = frozenset({_JOINT_KIND_REVOLUTE, _JOINT_KIND_PRISMATIC})


class JointRef(_Ref):
    """Reference to a joint. Carries `_joint_type` so `.axis()` can
    eagerly reject fixed joints at build time.
    """

    __slots__ = ("_joint_type",)

    def __init__(self, id_: int, owner_id: int, name: str,
                 joint_type: str, poses=()):
        super().__init__(id_, owner_id, name, poses=poses)
        self._joint_type = joint_type

    @property
    def kind(self) -> str:
        """The joint kind: `"fixed"`, `"revolute"`, or `"prismatic"`."""
        return self._joint_type

    def axis(self):
        """Return a DirectionRef for this joint's kinematic axis in
        world frame at this ref's pose tag (rest pose by default).

        Build-time error for fixed joints (which have no kinematic axis):
        use `anchor.outward()` or `Direction.world(...)` instead.
        """
        from ._refs import DirectionRef
        if self._joint_type not in _JOINT_KINDS_WITH_AXIS:
            raise InvalidJoint(
                f"joint {self._name!r} has type {self._joint_type!r} and "
                f"no kinematic axis; .axis() is only valid for "
                f"revolute/prismatic joints. "
                f"Use anchor.outward() or Direction.world(...) instead.",
                refs=[_joint_ref_dict(self._name, self._id)],
                evidence={"reason": "fixed_joint_has_no_axis",
                          "joint": self._name,
                          "joint_type": self._joint_type},
                suggested_repairs=[
                    "remove the .axis() call; the joint is rigid",
                    "use anchor.outward() on a face of the parent or child",
                    "use Direction.world((x, y, z)) for a literal world axis",
                ],
            )
        return DirectionRef(
            source="joint_axis",
            owner_id=self._owner_id,
            data={"joint": self._name},
            poses=self._poses,
        )

    def at(self, poses) -> "JointRef":
        from ._refs import _merge_pose_overrides
        return JointRef(self._id, self._owner_id, self._name,
                        self._joint_type,
                        _merge_pose_overrides(self._poses, poses))

    def __eq__(self, other):
        return (super().__eq__(other)
                and self._joint_type == other._joint_type)

    def __hash__(self):
        return hash((type(self).__name__, self._id, self._owner_id,
                     self._joint_type, self._poses))

    def __repr__(self):
        return (f"JointRef(name={self._name!r}, id={self._id}, "
                f"type={self._joint_type!r})")


# ---------------------------------------------------------------------------
# Object builder
# ---------------------------------------------------------------------------

class Object:
    """Builder for a single articulated object.

    At most one Object may be constructed per sandbox execution. Builder
    methods mutate internal state; `finalize()` produces a frozen,
    content-addressed IR dict.
    """

    def __init__(self, name: str):
        global _objects_constructed, _next_owner_id
        with _object_singleton_lock:
            if _object_limit is not None and _objects_constructed >= _object_limit:
                raise BuildError(
                    "only one Object may be constructed per sandbox execution",
                    refs=[_object_ref(name if isinstance(name, str) else None)],
                    evidence={"reason": "singleton_already_constructed"},
                    suggested_repairs=[
                        "construct exactly one Object per object.py",
                    ],
                )
            _objects_constructed += 1
            _next_owner_id += 1
            owner_id = _next_owner_id

        _check_name(name, "object", obj_name=None)
        self._name = name
        self._owner_id = owner_id

        # Parts: list of dicts in creation order, with PartRefs for lookup.
        self._parts: list[dict] = []
        self._part_name_to_id: dict[str, int] = {}

        # Joints: list of dicts in creation order.
        self._joints: list[dict] = []
        self._joint_name_to_id: dict[str, int] = {}

        # Child-id -> joint-id (enforces at-most-one-incoming-joint).
        self._parent_of_child: dict[int, int] = {}
        # Parent-id -> list of child-ids (for reachability walk).
        self._children_of_parent: dict[int, list[int]] = {}

        self._root_part_id: int | None = None
        self._root_part_name: str | None = None

        # Assertions namespace (separate from parts/joints).
        self._assertions: list[dict] = []
        self._assertion_name_to_id: dict[str, int] = {}

        # Plan checks: observable claims the author commits to before building,
        # read by the independent reviewer. See `add_check()`.
        self._checks: list[dict] = []
        self._check_claims: set[str] = set()

        # Collision allowances: explicit (a, b) part-pairs the agent has
        # declared as intentional self-overlap, exempt from
        # NoSelfPenetration. Stored canonically with `a < b` lexicographic
        # ordering. See `allow_collision()` and the verifier.
        self._allowances: list[dict] = []
        self._allowance_pair_keys: set[tuple[str, str]] = set()

        #: Run-relative path of this object's staged `object.py`, or None for an
        #: object with no CAD-authored parts (nothing was executed to build a
        #: `box`). Set by `_stage_registered_parts`, emitted as `ir["source"]`.
        self._source_asset = None
        # Persistent Part-owned semantic anchors: part_id -> {name: record}.
        # Named mechanical frames the author registers explicitly; serialized
        # into `PartIR.anchors` at finalize. The 27 AABB landmarks are NOT in
        # here — they stay virtual formulas over the baked AABB, so this holds
        # only what an author named explicitly.
        self._part_anchors: dict[int, dict] = {}

        # Live build123d Shapes registered through `add_part`, keyed by part id
        # and compiled to artifacts at `finalize()`.
        #
        # Deliberately NOT inside the `_parts` records: those are handed
        # straight to `canonical_json_bytes` (`json.dumps(..., allow_nan=False)`)
        # to compute the object's content hash, and a live OCCT object there
        # would raise a TypeError at hash time. Keeping them in a parallel store
        # means the serializable records stay serializable at every moment,
        # rather than needing to be scrubbed before finalize.
        self._registered_shapes: dict[int, dict] = {}

        self._finalized = False
        self._cached_ir: dict | None = None

    # ---- internal ref helpers --------------------------------------------

    def _check_owned(self, ref, kind: str):
        """Validate `ref` and RETURN it, resolving a bare name first.

        `obj.allow_collision("lid", "body")` and
        `obj.attach_fixed("j", "base", "arm", ...)` are what authors write. A
        name is unambiguous here — this builder owns exactly one part and one
        joint namespace, and `add_part` / `attach_*` already index both — so a
        lookup is all it takes.

        It returns the ref because that is what makes the resolution reach the
        caller; every call site assigns the result. A name this builder does
        not know raises the same `DanglingReference` an unowned ref does, with
        the known names listed, rather than being silently dropped.

        **The assertion surface deliberately does NOT accept names**, so
        `allow_collision("lid", "body")` works while
        `assert_overlap(a="lid", b="body")` still refuses. That is a known
        asymmetry, not an oversight: `assert_*` validates through
        `_assertions.check_entity_ref_owned`, which is shared with the Scene
        builder and holds no name table, so coercion would have to be repeated
        at ~13 call sites — and several of them take `AnchorRef`, where a bare
        string has no single meaning. Extend it when runs show agents passing
        names there.
        """
        expected_cls = PartRef if kind == "part" else JointRef
        if isinstance(ref, str):
            table = (self._part_name_to_id if kind == "part"
                     else self._joint_name_to_id)
            if ref not in table:
                raise DanglingReference(
                    f"this Object has no {kind} named {ref!r}",
                    refs=[_object_ref(self._name)],
                    evidence={"ref_kind": expected_cls.__name__,
                              "owner_object_name": self._name,
                              "ref_name": ref,
                              "known": sorted(table)},
                    suggested_repairs=[
                        f"name one of this Object's {kind}s: "
                        f"{', '.join(sorted(table)) or '(none yet)'}",
                    ],
                )
            ref_id = table[ref]
            if kind == "part":
                ref = PartRef(ref_id, self._owner_id, ref)
            else:
                ref = JointRef(ref_id, self._owner_id, ref,
                               joint_type=self._joints[ref_id]["type"])
        if not isinstance(ref, expected_cls):
            raise DanglingReference(
                f"expected {kind} name or {expected_cls.__name__}, "
                f"got {type(ref).__name__}",
                refs=[_object_ref(self._name)],
                evidence={
                    "ref_kind": expected_cls.__name__,
                    "owner_object_name": self._name,
                    "got_type": type(ref).__name__,
                },
                suggested_repairs=[
                    f"pass a {kind} name or {expected_cls.__name__} from this Object",
                ],
            )
        if ref._owner_id != self._owner_id:
            raise DanglingReference(
                f"{kind} ref {ref!r} was not produced by this Object",
                refs=[
                    _object_ref(self._name),
                    {"kind": kind, "name": ref._name, "id": ref._id},
                ],
                evidence={
                    "ref_kind": expected_cls.__name__,
                    "owner_object_name": self._name,
                    "ref_name": ref._name,
                    "ref_id": ref._id,
                },
                suggested_repairs=[
                    "use only refs returned by this Object's builder methods",
                ],
            )
        return ref

    def _check_not_finalized(self):
        if self._finalized:
            raise BuildError(
                "builder is finalized; no further calls allowed",
                refs=[_object_ref(self._name)],
                evidence={"reason": "builder_already_finalized"},
                suggested_repairs=["do not call builder methods after finalize()"],
            )

    # ---- part construction ------------------------------------------------

    def make_box(
        self,
        name: str,
        width: float,
        length: float,
        height: float,
        material: Material | None = None,
    ) -> PartRef:
        self._check_not_finalized()
        _check_name(name, "part", obj_name=self._name)
        if name in self._part_name_to_id:
            raise InvalidPrimitive(
                f"duplicate part name {name!r}",
                refs=[
                    _object_ref(self._name),
                    _part_ref_dict(name, self._part_name_to_id[name]),
                ],
                evidence={"param": "part.name", "value": name,
                          "constraint": "unique in part namespace"},
                suggested_repairs=["choose a different part name"],
            )

        part_refs = [_object_ref(self._name), _part_ref_dict(name)]
        for label, val in (("width", width), ("length", length), ("height", height)):
            _check_finite_scalar(val, label, refs=part_refs)
            if float(val) <= 0.0:
                raise InvalidPrimitive(
                    f"{label} must be > 0, got {val}",
                    refs=part_refs,
                    evidence={"param": label, "value": val, "constraint": "> 0"},
                    suggested_repairs=[
                        f"pass a strictly positive value for {label}",
                    ],
                )

        if material is None:
            mat = DEFAULT_MATERIAL
        else:
            mat = check_material(material, "material", refs=part_refs)

        part_id = len(self._parts)
        self._parts.append({
            "id": part_id,
            "name": name,
            "primitive": _PRIMITIVE_BOX,
            "params": {
                "width": float(width),
                "length": float(length),
                "height": float(height),
            },
            "material": mat.to_ir(),
        })
        self._part_name_to_id[name] = part_id
        return PartRef(part_id, self._owner_id, name)

    def add_part(self, name: str, shape, *,
                 material: Material | None = None,
                 tolerance: float = 1e-3,
                 angular_tolerance: float = 0.1,
                 planar_max_edge: float | None = None,
                 expect_solids: int | None = None) -> PartRef:
        """Register a **live build123d Shape** as one rigid part.

        The one way to author geometry: ``object.py`` constructs the Shape and
        hands it straight over, in a single program.

            body = obj.add_part("body", Box(0.5, 0.4, 0.6) - inner, material=wood)

        **A Part is one rigid body.** ``shape`` may be a ``Solid``, a build123d
        ``Part``, or a ``Compound`` holding several Solids that are fixed
        relative to each other. Geometry that moves relative to other geometry
        belongs to a different Part, joined by a joint. Faces, Wires, Sketches
        and empty Shapes are construction geometry and are rejected — only what
        is explicitly registered here becomes a Part, so intermediate cutting
        tools and helper shapes never leak into the IR.

        **Registration is the geometry commit point.** The Shape is validated,
        tessellated, measured, and serialized to a B-rep *now*, so the part's
        AABB, shape facts, and exact geometry are one consistent snapshot taken
        at the same instant. Anchors resolve against real geometry immediately,
        before ``finalize()``. The snapshot holds no reference to the Shape, so
        mutating the Shape afterwards cannot reach any artifact — a structural
        guarantee rather than a convention.

        Work is split across two stages by what it needs, not by what it
        produces: everything requiring the live Shape happens here, and
        everything requiring only numbers happens at ``finalize()`` (mesh
        post-processing, and writing the three assets to disk). The B-rep bytes
        staged at finalize are the ones captured here.

        The resulting part is an ordinary ``solid`` part, so anchors, joints,
        the renderer, the verifiers, the exact-collision adjudicator, and the
        URDF exporter all consume it with no special casing.

        ``tolerance`` / ``angular_tolerance`` control only the render mesh; the
        exact B-rep is unaffected. There is no ``scale``: the authored geometry
        is authoritative about size, and scaling afterwards would desync the
        exact mass properties from the geometry they describe.

        ``planar_max_edge`` bounds the longest triangle edge on **flat** faces,
        in **meters** — unlike ``tolerance``, which is relative. Neither
        tolerance can put a vertex inside a plane (chord and angular error are
        both zero there, so a pure box is 24 vertices at any value), so this is
        the only way to get interior vertices on a flat face, and a deformation
        rig binds to exactly those. ``None`` (the default) leaves flat faces
        boundary-triangulated. Pass it when the part will be deformed; the cost
        is triangles, and `geometry_notes` reports ``planar_max_edge_used`` if
        the budget forced a coarser grid.

        ``expect_solids=N`` requires exactly N Solids and reports a mismatch
        with the expected/actual counts and repair guidance. Use 1 to catch
        an unintended disconnected fuse or a cut that split the body.
        Without it, multiple Solids are accepted as one rigid part::

            obj.add_part("seat", seat_shape, material=WOOD, expect_solids=1)
        """
        self._check_not_finalized()
        _check_name(name, "part", obj_name=self._name)
        part_refs = [_object_ref(self._name), _part_ref_dict(name)]
        if planar_max_edge is not None:
            _check_finite_scalar(planar_max_edge, "planar_max_edge",
                                 refs=part_refs)
            if float(planar_max_edge) <= 0.0:
                raise InvalidPrimitive(
                    f"add_part({name!r}): planar_max_edge must be > 0, got "
                    f"{planar_max_edge!r}",
                    refs=part_refs,
                    evidence={"param": "planar_max_edge",
                              "value": planar_max_edge,
                              "constraint": "positive"},
                    suggested_repairs=[
                        "pass a length in meters, e.g. planar_max_edge=0.004",
                        "or omit it to leave flat faces boundary-triangulated",
                    ])
        if expect_solids is not None:
            # `bool` is an `int` subclass, and `expect_solids=True` would
            # otherwise silently mean "exactly one solid".
            if (isinstance(expect_solids, bool)
                    or not isinstance(expect_solids, int)
                    or expect_solids < 1):
                raise InvalidPrimitive(
                    f"add_part({name!r}): expect_solids must be an integer "
                    f">= 1, got {expect_solids!r}",
                    refs=part_refs,
                    evidence={"param": "expect_solids",
                              "value": repr(expect_solids),
                              "constraint": "integer >= 1"},
                    suggested_repairs=[
                        "pass the number of disconnected solids this part is "
                        "meant to contain, e.g. expect_solids=1",
                        "or omit it to accept any number >= 1",
                    ])
        if name in self._part_name_to_id:
            raise InvalidPrimitive(
                f"duplicate part name {name!r}",
                refs=[
                    _object_ref(self._name),
                    _part_ref_dict(name, self._part_name_to_id[name]),
                ],
                evidence={"param": "part.name", "value": name,
                          "constraint": "unique in part namespace"},
                suggested_repairs=["choose a different part name"],
            )

        compiler = _mesh_assets.get_active_solid_compiler()
        if compiler is None:
            raise InvalidPrimitive(
                "add_part requires the build123d solid compiler, which is not "
                "available in this interpreter; live-Shape parts need the harness CAD worker",
                refs=part_refs,
                evidence={"param": "shape", "value": type(shape).__name__,
                          "reason": "no_active_solid_compiler"},
                suggested_repairs=[
                    "run this source through the object worker, which executes "
                    "it in the frozen build123d environment and installs the "
                    "compiler",
                ],
            )

        if material is None:
            mat = DEFAULT_MATERIAL
        else:
            mat = check_material(material, "material", refs=part_refs)

        # Validate + tessellate + measure eagerly. A Part is a rigid body, so
        # the compiler checks "contains at least one Solid" unconditionally,
        # rejecting construction geometry (a Face, a Wire, an empty boolean
        # result) at the line that registered it.
        try:
            # The bound material's tile size, and only when it has one. UVs
            # are derived at staging (see `_solid_compile`'s texture-coordinate
            # section), so this is the one place the compiler needs to know
            # anything about the surface: `scale_m` fixes how much real
            # geometry one tile of the image spans. An untextured part passes
            # None and its mesh bytes are unchanged.
            mapping = getattr(mat, "mapping", None)
            entry = compiler.measure_shape(
                shape, name=name, tolerance=tolerance,
                angular_tolerance=angular_tolerance,
                planar_max_edge=planar_max_edge,
                texture_scale_m=_texture_scale_m(mat))
        except compiler.SolidCompileError as e:
            # Two sources of advice, most specific first: a recognized kernel
            # message names the exact move that resolves it, and the error's
            # `kind` selects what to do about that class of failure, so "your
            # boolean did not overlap" and "OCCT could not triangulate this
            # surface" get different advice.
            raise InvalidPrimitive(
                f"add_part({name!r}): {e.message}",
                refs=part_refs,
                evidence={"param": "shape", "value": type(shape).__name__,
                          "reason": e.kind},
                suggested_repairs=_solid_failure_repairs(compiler, e),
            ) from None

        # Checked here rather than inside the compiler because only the author
        # knows the intended count — `measure_shape` has already established
        # ">= 1", which is all a rigid body requires. The measured number comes
        # off the fact table the compiler just built, so this costs nothing.
        if expect_solids is not None:
            got = ((entry.get("table") or {}).get("topo") or {}).get("solids")
            if got is not None and int(got) != int(expect_solids):
                raise InvalidPrimitive(
                    f"add_part({name!r}): expected {expect_solids} solid(s), "
                    f"the shape contains {got}",
                    refs=part_refs,
                    evidence={"param": "shape", "value": type(shape).__name__,
                              "reason": "solid_count_mismatch",
                              "expected": int(expect_solids),
                              "measured": int(got)},
                    suggested_repairs=(
                        [
                            f"{got} disconnected bodies where {expect_solids} "
                            f"was intended. Check for a disconnected fuse "
                            f"or a cut that split the body. Face contact can "
                            f"fuse successfully; if that interface failed, "
                            f"try overlap scaled to the feature and recheck",
                            "if the extra bodies are intended and rigidly "
                            f"fixed to each other, pass "
                            f"expect_solids={got} instead",
                        ] if int(got) > int(expect_solids) else [
                            f"{got} solid(s) where {expect_solids} was "
                            f"intended means a boolean removed more than you "
                            f"meant, or the pieces merged. Compare the volume "
                            f"before and after the last boolean",
                            f"if one merged body is correct, pass "
                            f"expect_solids={got}",
                        ]),
                )

        # Params are PARTIAL until finalize: the three asset paths and the
        # digest do not exist yet. `_validate_solid_params` requires them
        # non-empty, so it deliberately is NOT run here — it runs in
        # `_stage_registered_parts` once staging has filled them in. What is
        # present now is exactly what anchors need: the kernel's AABB.
        part_id = len(self._parts)
        self._parts.append({
            "id": part_id,
            "name": name,
            "primitive": _PRIMITIVE_SOLID,
            "params": {
                "aabb_min": [float(v) for v in entry["aabb_min"]],
                "aabb_max": [float(v) for v in entry["aabb_max"]],
                "shape": _jsonify_shape(entry["table"]),
            },
            "material": mat.to_ir(),
        })
        self._part_name_to_id[name] = part_id
        self._registered_shapes[part_id] = entry
        return PartRef(part_id, self._owner_id, name)

    def _stage_registered_parts(self) -> None:
        """Turn every `add_part` snapshot into staged artifacts + full params.

        Stage 2 of two. Nothing here touches a live Shape — `measure_shape`
        dropped it at registration — so this works purely on numbers and bytes.
        What it adds is everything that needs the *whole* object: one
        content-addressed source blob shared by all N parts, which is not
        knowable until the last part is registered, and the three staged assets
        that complete each part's params.

        A no-op for an object with no registered snapshots.

        Runs before the structural checks in `finalize()` so that a geometry
        failure is reported as a geometry failure, rather than surfacing later
        as a confusing schema error about a missing `brep` path.
        """
        if not self._registered_shapes:
            return

        compiler = _mesh_assets.get_active_solid_compiler()
        session = _mesh_assets.get_active_mesh_session()
        refs = [_object_ref(self._name)]
        if compiler is None or session is None:
            raise InvalidPrimitive(
                "finalize cannot stage the registered parts: the solid compiler "
                "or the asset-staging session is no longer installed",
                refs=refs,
                evidence={"reason": "no_active_compiler_or_session",
                          "registered_parts": len(self._registered_shapes)},
                suggested_repairs=[
                    "finalize inside the same object run that registered the "
                    "parts",
                ],
            )

        # One source artifact for the whole object, recorded at the object
        # level. `_stage_bytes` is content-addressed, so re-staging identical
        # text is a no-op.
        #
        # None when the executing program is not this object's own recipe — a
        # `scene.py` builds many objects, and naming it as each one's
        # source both misdescribes it and, since `source` is hashed, re-ids
        # every object in the file whenever any line of it changes.
        source_text = self._source_text_for_staging()
        self._source_asset = (session.stage_source(source_text)
                              if source_text is not None else None)

        for part_id, entry in sorted(self._registered_shapes.items()):
            record = self._parts[part_id]
            name = record["name"]
            part_refs = [_object_ref(self._name), _part_ref_dict(name, part_id)]
            try:
                compiled = compiler.compile_entry(entry)
            except compiler.SolidCompileError as e:
                raise InvalidPrimitive(
                    f"finalize could not compile part {name!r}: {e.message}",
                    refs=part_refs,
                    evidence={"param": "shape", "reason": e.kind},
                    suggested_repairs=_solid_failure_repairs(compiler, e),
                ) from None
            try:
                frag = session.stage_solid(
                    obj_bytes=compiled["obj_bytes"],
                    brep_bytes=compiled["brep_bytes"],
                    aabb_min=compiled["aabb_min"],
                    aabb_max=compiled["aabb_max"],
                    shape=compiled["shape"],
                )
            except _mesh_assets.MeshStagingError as e:
                raise InvalidPrimitive(
                    f"finalize could not stage the assets for {name!r}: {e}",
                    refs=part_refs,
                    evidence={"param": "shape", "reason": str(e)},
                    suggested_repairs=["check the run directory is writable"],
                ) from None

            # What this compile did, as opposed to what the part is. The agent
            # layer drains these after the exec; nothing here reaches the IR, so
            # a coarsened render mesh or a stage that raised cannot change the
            # object's identity.
            #
            # **Only what an author could act on.** The repairs this compiler
            # applies are lossless and complete — welded coincident vertices,
            # dropped zero-area triangles, a whole-mesh re-wind — so reporting
            # them asks the author to think about a problem that is already
            # solved, on every part of every compile, in the one channel that
            # should be reserved for things needing a decision. Vertex and
            # triangle counts stay out too: they describe the tessellator, not
            # the design, and `qc` already carries them for anyone probing.
            #
            # What survives is what the compiler could NOT fix and the author
            # can: shells that disagree on orientation, geometry with no
            # thickness, a mesh coarsened off the requested tolerance, an
            # imprint that could not be honored, and a stage that raised. A
            # part with none of these produces no note at all, so a clean
            # compile reports an empty list rather than one reassuring line per
            # part.
            note = {"part": name, "warnings": list(compiled.get("warnings") or ())}
            # A status is only worth carrying when it is not "ok": it exists to
            # distinguish "ran and found nothing" from "did not run", and the
            # first of those is the silent case by design.
            for key in ("repair_status", "qc_status"):
                if compiled.get(key) != "ok":
                    note[key] = compiled.get(key)
            # `tolerance_used` is withdrawn with the coarsening warning it went
            # with (see `_solid_compile`): a bare number the agent cannot act on
            # is the same invitation to go hunting, minus the sentence saying
            # the B-rep is fine. `shape.qc` still carries the mesh it produced.
            # **The planar imprint is the exception, and it is a principled
            # one.** The fields dropped above were unconditional noise: they
            # appeared on every part of every compile whether or not anything
            # happened. `planar_max_edge` is different in kind — it only exists
            # because the author explicitly asked for it, and what they got
            # back is reported nowhere else. `planar_max_edge_used` is the
            # target after the budget may have doubled it, and
            # `planar_max_edge_achieved` is MEASURED off the resulting mesh, so
            # neither is recoverable from `qc` or from the OBJ the way vertex
            # counts are. So: silent for a part that never asked, and complete
            # for one that did — including the honored case, because "you got
            # what you requested" is the answer to a question that was asked.
            if entry.get("planar_max_edge") is not None:
                note["planar_max_edge_used"] = compiled.get("planar_max_edge_used")
                note["planar_max_edge_achieved"] = \
                    compiled.get("planar_max_edge_achieved")
                note["n_faces_imprinted"] = compiled.get("n_faces_imprinted") or 0
            if note["warnings"] or len(note) > 2:
                session.add_geometry_note(note)

            # Now the params are complete, so the ordinary solid validation
            # applies.
            try:
                params = _prim.validate_params(_PRIMITIVE_SOLID, frag)
            except ValueError as e:
                raise InvalidPrimitive(
                    f"finalize produced malformed solid params for {name!r}: {e}",
                    refs=part_refs,
                    evidence={"param": "shape", "reason": str(e)},
                    suggested_repairs=[
                        "this is an internal inconsistency; report it",
                    ],
                ) from None
            record["params"] = {
                "brep": params["brep"],
                "mesh": params["mesh"],
                "digest": params["digest"],
                "aabb_min": [float(v) for v in params["aabb_min"]],
                "aabb_max": [float(v) for v in params["aabb_max"]],
                "shape": _jsonify_shape(params["shape"]),
            }

        # The snapshots have served their purpose. Dropping them makes it
        # structurally impossible for one to reach serialization.
        self._registered_shapes.clear()

    def _source_text_for_staging(self) -> "str | None":
        """The authored program to stage as this object's `source` asset.

        **The executed bytes, when they are available.**
        `compile.build_ir_with_layer` puts the exact text it is about to `exec`
        on the session, so the staged artifact is the program that actually
        produced this geometry. That matters for the same reason `measure_shape`
        serializes the B-rep eagerly instead of holding the Shape: a provenance
        artifact must depend on the execution, not on the later state of a file
        the execution happened to be read from.

        The disk read below is the fallback for a session built outside a
        harness compile (a test staging a solid directly). It is second, not
        first, because re-reading `<run_dir>/object.py` is a second read of a
        mutable file — and if that file has moved on, the staged "source" would
        document a program that did not build this object.
        """
        session = _mesh_assets.get_active_mesh_session()
        if not getattr(session, "source_owns_object", True):
            # The running program builds more than this object. It is recorded
            # for the bundle as a whole, not claimed as this object's recipe.
            return None
        executed = getattr(session, "source_text", None)
        if isinstance(executed, str) and executed.strip():
            return executed
        run_dir = getattr(session, "run_dir", None)
        if run_dir is not None:
            src = Path(run_dir) / "object.py"
            try:
                text = src.read_text(encoding="utf-8")
                if text.strip():
                    return text
            except OSError:
                pass
        return (f"# object.py for {self._name!r} was not readable at finalize; "
                f"its live-Shape parts were built in-process.\n")

    # ---- joint construction (shared helpers) -----------------------------

    def _validate_anchors(self, parent_anchor, child_anchor, joint_name, parent, child):
        """Validate the joint's anchor inputs.

        Both parent_anchor and child_anchor must be AnchorRefs produced by
        this Object's .anchor() method. The AnchorRef must reference the
        matching part (parent_anchor → parent, child_anchor → child).
        """
        refs = [
            _object_ref(self._name),
            _joint_ref_dict(joint_name),
            _part_ref_dict(parent._name, parent._id),
            _part_ref_dict(child._name, child._id),
        ]
        self._validate_anchor_ref_for_joint(
            parent_anchor, parent, "parent_anchor", refs)
        self._validate_anchor_ref_for_joint(
            child_anchor, child, "child_anchor", refs)

    def _validate_anchor_ref_for_joint(self, ref, expected_part: PartRef,
                                       label: str, refs):
        """Validate an AnchorRef passed to attach_* as the joint anchor.

        The ref must be owned by this Object and must address the expected
        part (i.e. parent_anchor must reference parent; child_anchor must
        reference child).
        """
        if not isinstance(ref, AnchorRef):
            raise InvalidJoint(
                f"{label} must be an AnchorRef (from obj.anchor(...)), got "
                f"{type(ref).__name__}",
                refs=refs,
                evidence={"reason": "bad_anchor_type", "param": label,
                          "value": type(ref).__name__},
                suggested_repairs=[
                    f"pass obj.anchor(part, <anchor_name>) for {label}, "
                    f"where part is the PartRef named {expected_part._name!r}",
                ],
            )
        if ref._owner_id != self._owner_id:
            raise InvalidJoint(
                f"{label}: AnchorRef was not produced by this Object",
                refs=refs,
                evidence={"reason": "foreign_anchor_ref", "param": label,
                          "entity": ref._entity_name,
                          "anchor": ref._anchor_name},
                suggested_repairs=["use this Object's .anchor(part, name)"],
            )
        if ref._entity_kind != "part":
            raise InvalidJoint(
                f"{label}: AnchorRef must reference a part, got "
                f"entity_kind={ref._entity_kind!r}",
                refs=refs,
                evidence={"reason": "anchor_wrong_entity_kind",
                          "value": ref._entity_kind},
                suggested_repairs=["pass obj.anchor(<PartRef>, <anchor_name>)"],
            )
        if ref._entity_name != expected_part._name:
            raise InvalidJoint(
                f"{label}: AnchorRef references part {ref._entity_name!r}, "
                f"but the joint's matched part is {expected_part._name!r}",
                refs=refs,
                evidence={"reason": "anchor_part_mismatch",
                          "anchor_part": ref._entity_name,
                          "joint_part": expected_part._name},
                suggested_repairs=[
                    f"use obj.anchor(part, <anchor_name>) for {label}, "
                    f"where part is the PartRef named {expected_part._name!r}",
                ],
            )

    def _anchor_to_se3(self, anchor: AnchorRef, part: PartRef):
        """Resolve an AnchorRef to an API-shaped SE3 in the part's local frame.

        The anchor's frame is the AABB-landmark point, transformed by the
        ref's offset SE3 `T_offset = T(t) ∘ R(r)` interpreted in part-
        local: the offset's translation `t` is in part-local axes; its
        rotation `r` then rotates the anchor's frame about (landmark + t).

        Result shape: internal SE3 `((tx,ty,tz),(qw,qx,qy,qz))`, ready for
        `_se3_internal_to_ir`.
        """
        part_id = self._part_name_to_id[part._name]
        part_record = self._parts[part_id]
        # Anchors resolve against the primitive's enclosing AABB (centered at
        # origin). box/cylinder/sphere all derive their AABB in _primitives, so
        # the 27-name vocabulary applies uniformly to round parts.
        aabb = _prim.local_aabb(part_record["primitive"], part_record["params"])
        try:
            base = _anchors_mod.resolve_base_frame(
                aabb, self._part_anchors.get(part_id), anchor._anchor_name,
                part_name=part._name)
        except _anchors_mod.UnknownAnchor as e:
            raise InvalidAnchor(
                str(e).strip("'"),
                refs=[_object_ref(self._name),
                      _part_ref_dict(part._name, part._id)],
                evidence={"param": "anchor.name",
                          "value": anchor._anchor_name,
                          "reason": "unknown_anchor"},
                suggested_repairs=[
                    "use a built-in AABB anchor name, or register the anchor "
                    "with obj.add_anchor(part, name, ...) before referencing it",
                ],
            ) from None

        # Compose the ref's offset onto the base frame:  T_base o T_offset.
        # For a built-in the base rotation is identity, so this reduces to
        # `landmark + offset.translation` with the offset rotation standing —
        # byte-identical to the pre-M3 behavior. For a semantic anchor the
        # base rotation participates, which is what makes an authored frame
        # actually orientable.
        (otx, oty, otz), (orx, ory, orz) = anchor._offset
        offset = ((otx, oty, otz),
                  _rot.euler_xyz_deg_to_quat_wxyz(orx, ory, orz))
        # Return the INTERNAL form (translation, quaternion). Converting to
        # Euler and back would round-trip through a representation that is
        # not canonical — measured: euler(180,180,180)
        # comes back as a different triple for the same frame — which would
        # perturb existing object ids for no reason.
        return _rot.se3_compose(base, offset)

    def _validate_joint_endpoints(self, parent: PartRef, child: PartRef,
                                  joint_name: str):
        """Validate the endpoints and RETURN them, names resolved.

        Returned rather than checked in place because `attach_*` goes on to
        use `parent` / `child` for the anchors and the joint record, so a
        coercion that stopped here would validate a name and then record a
        string.
        """
        parent = self._check_owned(parent, "part")
        child = self._check_owned(child, "part")
        refs_base = [_object_ref(self._name), _joint_ref_dict(joint_name)]
        if parent._id == child._id:
            raise InvalidJoint(
                f"joint {joint_name!r}: cannot attach a part to itself",
                refs=refs_base + [_part_ref_dict(parent._name, parent._id)],
                evidence={"reason": "self_attach", "value": parent._name},
                suggested_repairs=["pick distinct parent and child parts"],
            )
        if child._id in self._parent_of_child:
            existing = self._joints[self._parent_of_child[child._id]]["name"]
            raise InvalidJoint(
                f"joint {joint_name!r}: child part {child._name!r} already has "
                f"incoming joint {existing!r}",
                refs=refs_base + [
                    _part_ref_dict(child._name, child._id),
                    _joint_ref_dict(existing, self._parent_of_child[child._id]),
                ],
                evidence={"reason": "multi_parent",
                          "value": {"child": child._name, "existing_joint": existing}},
                suggested_repairs=[
                    "each part may have at most one incoming joint; "
                    "remove the existing joint or pick a different child",
                ],
            )
        if self._root_part_id is not None and child._id == self._root_part_id:
            raise InvalidJoint(
                f"joint {joint_name!r}: declared root {child._name!r} cannot be a child",
                refs=refs_base + [_part_ref_dict(child._name, child._id)],
                evidence={"reason": "root_as_child", "value": child._name},
                suggested_repairs=[
                    "the root part has no incoming joints; pick a non-root child",
                ],
            )
        return parent, child

    def _joint_types_map(self) -> dict:
        """Return `{joint_name: joint_type}` for assertion-record pose conversion.

        Pose-tagged refs and assertion-level `poses=` carry surface-unit
        values (degrees for revolute, meters for prismatic); the IR stores
        IR-internal units (radians for revolute). The assertion-record
        builders use this map to convert at serialization time.
        """
        return {j["name"]: j["type"] for j in self._joints}

    def _record_joint(
        self,
        name: str,
        kind: str,
        parent: PartRef,
        child: PartRef,
        parent_anchor,
        child_anchor,
        extra: dict | None = None,
    ) -> JointRef:
        """Resolve anchor inputs to canonical SE3 frames and append the
        joint record.

        Anchor offsets carry the full placement information; a joint record has
        no separate `offset` field. Each AnchorRef's offset
        `((tx,ty,tz),(rx,ry,rz))` is in the entity's local frame: translation
        in part-local axes, rotation about the (landmark + translation) point.

        IR layout:
          {
            "parent_anchor": SE3 of the parent_anchor's pose in parent-local,
            "child_anchor":  SE3 of the child_anchor's pose in child-local,
            ...
          }

        The FK code (object_query._child_in_parent) composes:
          rest    = parent_anchor ∘ inverse(child_anchor)
          pivot   = parent_anchor.translation  (revolute / prismatic only)
          axis    = parent_anchor.R · axis_raw  (axis_raw is in parent-anchor-local frame)
        """
        pa_se3 = self._anchor_to_se3(parent_anchor, parent)
        ca_se3 = self._anchor_to_se3(child_anchor, child)

        joint_id = len(self._joints)
        record = {
            "id": joint_id,
            "name": name,
            "type": kind,
            "parent": parent._name,
            "child": child._name,
            "parent_anchor": _se3_internal_to_ir(pa_se3),
            "child_anchor": _se3_internal_to_ir(ca_se3),
        }
        # Symbolic provenance, semantic anchors only. The embedded transforms
        # above stay the compatibility snapshot every downstream consumer
        # already reads; this records WHICH authored frame produced them, so
        # the relationship is inspectable rather than implicit.
        for key, ref in (("parent_anchor_ref", parent_anchor),
                         ("child_anchor_ref", child_anchor)):
            prov = _anchor_provenance(ref)
            if prov is not None:
                record[key] = prov
        if extra:
            record.update(extra)
        self._joints.append(record)
        self._joint_name_to_id[name] = joint_id
        self._parent_of_child[child._id] = joint_id
        self._children_of_parent.setdefault(parent._id, []).append(child._id)
        return JointRef(joint_id, self._owner_id, name, joint_type=kind)

    def _validate_joint_name(self, name: str):
        _check_name(name, "joint", obj_name=self._name)
        if name in self._joint_name_to_id:
            raise InvalidJoint(
                f"duplicate joint name {name!r}",
                refs=[
                    _object_ref(self._name),
                    _joint_ref_dict(name, self._joint_name_to_id[name]),
                ],
                evidence={"param": "joint.name", "value": name,
                          "constraint": "unique in joint namespace"},
                suggested_repairs=["choose a different joint name"],
            )

    def _validate_axis_and_limits(self, axis, limits, default, units_label: str,
                                   *, refs):
        # axis — a 3-sequence or an axis name ('z', '-y'). One vocabulary with
        # the assertions, so an author cannot learn a spelling here that is
        # refused there.
        axis = _assertions.coerce_axis(axis, InvalidJoint, refs=refs)
        for i, c in enumerate(axis):
            _check_finite_scalar(c, f"axis[{i}]", refs=refs)
        norm = math.sqrt(sum(float(c) * float(c) for c in axis))
        if norm < _AXIS_MIN_NORM:
            raise InvalidJoint(
                f"axis norm too small ({norm}); must be > {_AXIS_MIN_NORM}",
                refs=list(refs),
                evidence={"reason": "axis_zero", "value": list(axis), "norm": norm},
                suggested_repairs=["pass a non-zero axis vector"],
            )
        axis_unit = tuple(float(c) / norm for c in axis)

        # limits + default (units_label is informational only)
        if not (isinstance(limits, tuple) and len(limits) == 2):
            raise InvalidJoint(
                f"limits must be (lo, hi), got {limits!r}",
                refs=list(refs),
                evidence={"reason": "limits_shape", "value": repr(limits)},
                suggested_repairs=["pass limits as (lo, hi)"],
            )
        lo, hi = limits
        _check_finite_scalar(lo, "limits[0]", refs=refs)
        _check_finite_scalar(hi, "limits[1]", refs=refs)
        if float(lo) > float(hi):
            raise InvalidJoint(
                f"limits lo > hi ({lo} > {hi}) [{units_label}]",
                refs=list(refs),
                evidence={"reason": "limits_inverted",
                          "value": [float(lo), float(hi)], "units": units_label},
                suggested_repairs=["swap limits so lo <= hi"],
            )
        _check_finite_scalar(default, "default", refs=refs)
        if not (float(lo) <= float(default) <= float(hi)):
            raise InvalidJoint(
                f"default {default} not in limits [{lo}, {hi}] [{units_label}]",
                refs=list(refs),
                evidence={"reason": "default_out_of_range",
                          "value": float(default),
                          "limits": [float(lo), float(hi)], "units": units_label},
                suggested_repairs=["pick a default within limits"],
            )
        return axis_unit, (float(lo), float(hi)), float(default)

    # ---- attach_fixed ----------------------------------------------------

    def attach_fixed(
        self,
        name: str,
        parent: PartRef,
        child: PartRef,
        parent_anchor,
        child_anchor,
    ) -> JointRef:
        self._check_not_finalized()
        self._validate_joint_name(name)
        parent, child = self._validate_joint_endpoints(parent, child, name)
        self._validate_anchors(parent_anchor, child_anchor, name, parent, child)
        return self._record_joint(
            name, _JOINT_FIXED, parent, child,
            parent_anchor, child_anchor,
        )

    # ---- attach_revolute -------------------------------------------------

    def attach_revolute(
        self,
        name: str,
        parent: PartRef,
        child: PartRef,
        axis,
        limits,
        default,
        parent_anchor,
        child_anchor,
    ) -> JointRef:
        self._check_not_finalized()
        self._validate_joint_name(name)
        parent, child = self._validate_joint_endpoints(parent, child, name)
        self._validate_anchors(parent_anchor, child_anchor, name, parent, child)
        joint_refs = [
            _object_ref(self._name),
            _joint_ref_dict(name),
            _part_ref_dict(parent._name, parent._id),
            _part_ref_dict(child._name, child._id),
        ]
        axis_unit, (lo_deg, hi_deg), default_deg = self._validate_axis_and_limits(
            axis, limits, default, "degrees", refs=joint_refs
        )
        # axis_unit is in the parent_anchor's local frame.
        # Convert revolute limits/default from degrees -> radians for IR.
        extra = {
            "axis": [axis_unit[0], axis_unit[1], axis_unit[2]],
            "limits": [_rot.deg_to_rad(lo_deg), _rot.deg_to_rad(hi_deg)],
            "default": _rot.deg_to_rad(default_deg),
        }
        return self._record_joint(
            name, _JOINT_REVOLUTE, parent, child,
            parent_anchor, child_anchor, extra=extra,
        )

    # ---- attach_prismatic ------------------------------------------------

    def attach_prismatic(
        self,
        name: str,
        parent: PartRef,
        child: PartRef,
        axis,
        limits,
        default,
        parent_anchor,
        child_anchor,
    ) -> JointRef:
        self._check_not_finalized()
        self._validate_joint_name(name)
        parent, child = self._validate_joint_endpoints(parent, child, name)
        self._validate_anchors(parent_anchor, child_anchor, name, parent, child)
        joint_refs = [
            _object_ref(self._name),
            _joint_ref_dict(name),
            _part_ref_dict(parent._name, parent._id),
            _part_ref_dict(child._name, child._id),
        ]
        axis_unit, (lo_m, hi_m), default_m = self._validate_axis_and_limits(
            axis, limits, default, "meters", refs=joint_refs
        )
        # axis_unit is in the parent_anchor's local frame.
        extra = {
            "axis": [axis_unit[0], axis_unit[1], axis_unit[2]],
            "limits": [lo_m, hi_m],
            "default": default_m,
        }
        return self._record_joint(
            name, _JOINT_PRISMATIC, parent, child,
            parent_anchor, child_anchor, extra=extra,
        )

    # ---- anchor + assertion API -----------------------------------------
    #
    # Anchors are named points on a part's AABB. Under the shared-frame
    # convention they have identity rotation in the part's local frame, so
    # an AnchorRef carries only `(entity, anchor_name)`. The 27-name
    # vocabulary is shared with the Scene SDK via sdk/_anchors.py.

    def _anchor_name_and_offset(self, anchor_name, translation, rotation,
                                refs, entity_kind, *, part_id=None):
        """Validate an anchor name and freeze the optional offset.

        A name is valid if it is one of the 27 built-in AABB landmarks OR a
        semantic anchor already registered on this part. Referencing a semantic
        anchor by name — rather than only through the ref `add_anchor` returned
        — is what lets a joint read as `obj.anchor(boom, "base_hinge")`, which
        is the whole ergonomic point of naming frames.
        """
        if not isinstance(anchor_name, str):
            raise InvalidAnchor(
                f"anchor_name must be a string, got {type(anchor_name).__name__}",
                refs=refs,
                evidence={"reason": "bad_anchor_type",
                          "value": repr(anchor_name)},
                suggested_repairs=["pass an anchor name string"],
            )
        known_semantic = (self._part_anchors.get(part_id) or {}
                          ) if part_id is not None else {}
        if (anchor_name not in _anchors_mod.AUTO_AABB_ANCHOR_NAMES
                and anchor_name not in known_semantic):
            raise InvalidAnchor(
                f"unknown anchor {anchor_name!r}: not one of the 27 built-in "
                f"AABB names ({_anchors_mod.builtin_anchor_names_text()}), and "
                f"no semantic anchor of that name is registered on this part"
                + (f" (registered: {', '.join(sorted(known_semantic))})"
                   if known_semantic else ""),
                refs=refs,
                evidence={"entity_kind": entity_kind, "value": anchor_name,
                          "registered_semantic": sorted(known_semantic),
                          "builtin_names": _anchors_mod.builtin_anchor_groups()},
                suggested_repairs=[
                    "use top/bottom/front/back/left/right, an edge name like "
                    "top_front, or a corner name like top_front_left",
                    "or register the frame first with "
                    "obj.add_anchor(part, name, translation=..., rotation=...)",
                ],
            )
        from ._refs import freeze_anchor_offset, IDENTITY_ANCHOR_OFFSET
        if translation is None and rotation is None:
            return IDENTITY_ANCHOR_OFFSET
        try:
            return freeze_anchor_offset(translation, rotation)
        except (TypeError, ValueError) as e:
            raise InvalidAnchor(
                f"anchor offset on {anchor_name!r}: {e}",
                refs=refs,
                evidence={"reason": "bad_anchor_offset",
                          "translation": repr(translation),
                          "rotation": repr(rotation)},
                suggested_repairs=[
                    "pass translation as a finite 3-tuple in meters; "
                    "pass rotation as a finite 3-tuple of Euler XYZ degrees",
                ],
            ) from None

    def anchor(self, part: PartRef, anchor_name: str, *,
               translation=None, rotation=None) -> AnchorRef:
        """Return an AnchorRef for a named anchor on `part`.

        Optional `translation` (3-tuple, meters, in part-local frame) and
        `rotation` (3-tuple, intrinsic Euler XYZ degrees) bake an offset
        into the ref. The offset is interpreted as the SE3
        `T(translation) ∘ R(rotation)` applied to the AABB landmark:
        translation moves the anchor point in part-local axes, then
        rotation rotates the anchor's frame about that translated point.

        For further offset composition use `.offset_by(...)` on the
        returned ref.
        """
        self._check_not_finalized()
        part = self._check_owned(part, "part")
        refs = [_object_ref(self._name),
                _part_ref_dict(part._name, part._id),
                {"kind": "anchor", "name": anchor_name, "id": None}]
        offset = self._anchor_name_and_offset(
            anchor_name, translation, rotation, refs, "part",
            part_id=part._id)
        return AnchorRef("part", part._name, anchor_name, self._owner_id,
                         offset=offset)

    def add_anchor(self, part: PartRef, name: str, *,
                   translation=(0.0, 0.0, 0.0),
                   rotation=(0.0, 0.0, 0.0)) -> AnchorRef:
        """Register a **persistent, named semantic anchor** on `part`.

        A semantic anchor is a complete rigid frame fixed to the part and
        expressed in that part's local coordinates — a hinge line, a mounting
        face, a grasp point. Unlike the 27 built-in AABB landmarks it is
        *authored knowledge*: it says where a mechanically meaningful frame is,
        which no bounding box can tell you.

            hinge = obj.add_anchor(boom, "base_hinge",
                                   translation=(-0.60, 0.0, 0.04),
                                   rotation=(90, 0, 0))
            obj.attach_revolute("boom_pitch", parent=house, child=boom,
                                axis=(0, 0, 1), limits=(-20, 60), default=0,
                                parent_anchor=mount, child_anchor=hinge)

        **An anchor does not define motion.** It defines a frame; the joint
        defines how two parts move once their frames are brought together. The
        joint axis is always explicit on the joint.

        The anchor's meaning lives in its **name**, which is the only label
        anything reads. There was a `kind=` parameter for a descriptive tag
        ("hinge_axis", "mount_frame"); it was withdrawn because nothing in the
        system ever read it, so it was work the author did for no effect. Pick a
        name that says what the frame is and that is the whole vocabulary.

        `translation` is meters in part-local axes; `rotation` is intrinsic
        Euler XYZ degrees, exactly as `anchor()` takes them. Together they are
        the anchor's `Anchor -> Part` transform.

        Registered after `add_part` and before `finalize`. This does not weaken
        the M2 snapshot contract: geometry is frozen at `add_part`; an anchor
        adds *metadata about* that frozen geometry, never geometry itself.

        Returns an ordinary `AnchorRef` — the same type `anchor()` returns —
        so it composes with `.offset_by(...)` / `.at(...)` and is accepted
        anywhere an anchor is, by joints and assertions alike.
        """
        self._check_not_finalized()
        part = self._check_owned(part, "part")
        refs = [_object_ref(self._name),
                _part_ref_dict(part._name, part._id),
                {"kind": "anchor", "name": name, "id": None}]
        _check_name(name, "anchor", obj_name=self._name)

        # Reserved: the built-in vocabulary is the single source of truth, so a
        # semantic anchor can never shadow one and the resolver can tell them
        # apart by name alone.
        if name in _anchors_mod.AUTO_AABB_ANCHOR_NAMES:
            raise InvalidAnchor(
                f"anchor name {name!r} is one of the {len(_anchors_mod.AUTO_AABB_ANCHOR_NAMES)} "
                f"built-in AABB anchor names and cannot be reused for a "
                f"semantic anchor; the reserved names are "
                f"{_anchors_mod.builtin_anchor_names_text()}",
                refs=refs,
                evidence={"param": "anchor.name", "value": name,
                          "reason": "reserved_builtin_name",
                          "builtin_names": _anchors_mod.builtin_anchor_groups()},
                suggested_repairs=[
                    "choose a mechanically descriptive name such as "
                    "'base_hinge', 'boom_mount' or 'grasp_frame'",
                ],
            )

        existing = self._part_anchors.setdefault(part._id, {})
        if name in existing:
            raise InvalidAnchor(
                f"part {part._name!r} already has a semantic anchor named "
                f"{name!r}",
                refs=refs,
                evidence={"param": "anchor.name", "value": name,
                          "reason": "duplicate_anchor_name",
                          "existing": sorted(existing)},
                suggested_repairs=[
                    f"pick a name not in {sorted(existing)}; anchor names are "
                    f"unique within a part but may repeat across parts",
                ],
            )

        t = tuple(translation) if translation is not None else (0.0, 0.0, 0.0)
        r = tuple(rotation) if rotation is not None else (0.0, 0.0, 0.0)
        _check_vec3(t, "translation", refs=refs)
        _check_vec3(r, "rotation", refs=refs)
        # Canonical WXYZ, exactly as `scene_builder._anchor_ref_to_ir` does it,
        # so two semantically equal frames serialize byte-identically and the
        # object hash is stable.
        quat = _rot.sign_canonicalize_quat(
            _rot.normalize_quat(_rot.euler_xyz_deg_to_quat_wxyz(*r)))

        record = {
            "name": name,
            "translation": [float(v) for v in t],
            "rotation": [float(v) for v in quat],
            # A constant, and deliberately still emitted. `source` was an
            # authoring parameter that no author ever passed in the entire run
            # history, and that nothing reads — `resolve_base_frame`, the joint
            # provenance check and every consumer look only at `translation` and
            # `rotation`. It is withdrawn from the API so nobody writes a dead
            # field, and kept in the record because it appears in every anchor
            # of every object already staged: dropping it would change the
            # content hash of each of them for no gain.
            "source": _ANCHOR_SOURCE,
        }
        existing[name] = record

        return AnchorRef("part", part._name, name, self._owner_id)

    def assert_distance(self, *, a: AnchorRef, b: AnchorRef,
                        min, max, axis=None, poses=None,
                        name: str) -> AssertionRef:
        """Distance between `a` and `b` in [min, max].

        `axis=None` -> Euclidean magnitude |b - a| (must have min >= 0).
        `axis=(x, y, z)` -> signed projection (b - a) . unit(axis); `min`
        may be negative (signed gap / overlap depth semantics).

        Evaluated at the rest pose of each ref unless the ref carries
        pose tags via `.at({joint: q, ...})`.
        """
        self._check_not_finalized()
        a_refs = [_object_ref(self._name),
                  {"kind": "assertion", "name": name, "id": None}]
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
        """Assert that anchors lie on a common line parallel to `axis`."""
        self._check_not_finalized()
        a_refs = [_object_ref(self._name),
                  {"kind": "assertion", "name": name, "id": None}]
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

    def assert_overlap(self, *, a: PartRef, b: PartRef,
                       axis, min, max=None, poses=None,
                       name: str) -> AssertionRef:
        """Signed interval overlap of two part AABBs along `axis`.

        value = min(a.max, b.max) - max(a.min, b.min)

        Positive = overlap depth (parts share extent on `axis`);
        negative = gap distance. The value must lie in [min, max].

        `max=None` defaults to +inf (allow any overlap depth).
        """
        self._check_not_finalized()
        a_refs = [_object_ref(self._name),
                  {"kind": "assertion", "name": name, "id": None}]
        _assertions.check_assertion_name(
            name, self._assertion_name_to_id, InvalidAssertion, refs=a_refs)
        _assertions.check_entity_ref_owned(
            a, self._owner_id, "a", InvalidAssertion, refs=a_refs,
            allowed_kinds=("PartRef",))
        _assertions.check_entity_ref_owned(
            b, self._owner_id, "b", InvalidAssertion, refs=a_refs,
            allowed_kinds=("PartRef",))
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
        """Angle (degrees) between two unit DirectionRefs in [min_deg, max_deg].

        Each direction may be a joint axis (`joint.axis()`), an anchor
        outward normal (`anchor.outward()`), or a literal world vector
        (`Direction.world((x, y, z))`).
        """
        self._check_not_finalized()
        a_refs = [_object_ref(self._name),
                  {"kind": "assertion", "name": name, "id": None}]
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
            # Out of RANGE and out of ORDER are different mistakes, so they get
            # different advice. The range case is the one that misleads: this is
            # the *unsigned* angle between two directions, so 181 deg is not a
            # wider bound than 180 — it is 179 deg again, and an author reaching
            # for a symmetric tolerance band around 180 is asking for something
            # the measurement cannot express. Saying only "must be within
            # [0, 180]" leaves them to rediscover that.
            repairs = ["use bounds within [0, 180] with min_deg <= max_deg"]
            if lo < 0.0 or hi > 180.0:
                repairs.insert(0, (
                    "This unsigned angle is in [0, 180] degrees. Use one-sided "
                    "tolerance bands at the limits: [178, 180] for nearly opposite "
                    "directions or [0, 2] for nearly aligned directions."))
            raise InvalidAssertion(
                f"angle bounds must satisfy 0 <= min_deg <= max_deg <= 180, "
                f"got [{lo}, {hi}]",
                refs=a_refs,
                evidence={"reason": "bad_angle_bounds",
                          "min_deg": lo, "max_deg": hi},
                suggested_repairs=repairs,
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

    def _assert_axis_aligned(self, *, a, b, tol_deg: float,
                              poses=None, name: str) -> AssertionRef:
        """Private: emit an unsigned axis-aligned record.

        Consumed by the `assert_axis_world_aligned` sugar; not part of
        the public surface (use the sugar instead).
        """
        self._check_not_finalized()
        a_refs = [_object_ref(self._name),
                  {"kind": "assertion", "name": name, "id": None}]
        _assertions.check_assertion_name(
            name, self._assertion_name_to_id, InvalidAssertion, refs=a_refs)
        _assertions.check_direction_ref_owned(
            a, self._owner_id, "a", InvalidAssertion, refs=a_refs)
        _assertions.check_direction_ref_owned(
            b, self._owner_id, "b", InvalidAssertion, refs=a_refs)
        _assertions.check_finite_scalar(tol_deg, "tol_deg", InvalidAssertion,
                                         refs=a_refs)
        tol = float(tol_deg)
        if tol < 0.0 or tol > 90.0:
            raise InvalidAssertion(
                f"tol_deg must lie in [0, 90], got {tol}",
                refs=a_refs,
                evidence={"reason": "bad_axis_aligned_tol", "tol_deg": tol},
                suggested_repairs=[
                    "pick a tolerance in [0, 90] degrees (typically <= 5)",
                ],
            )
        assertion_id = len(self._assertions)
        from ._refs import freeze_poses as _freeze_poses
        record = _assertions.build_axis_aligned_record(
            assertion_id=assertion_id, name=name,
            dir_a=a, dir_b=b, tol_deg=tol,
            assertion_poses=_freeze_poses(poses),
            joint_types=self._joint_types_map(),
        )
        self._assertions.append(record)
        self._assertion_name_to_id[name] = assertion_id
        return AssertionRef(assertion_id, self._owner_id, name, "axis_aligned")

    # ---- sugar assertions ------------------------------------------------
    #
    # Each sugar expands to one or more base records via `_sugar.*`. The
    # IR kernel never sees a sugar kind; failure reports preserve intent
    # through auto-suffixed names of the form
    # `<user_name>__<sugar_kind>_<aspect>`.

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

    def assert_axis_world_aligned(self, *, joint, world_axis,
                                   tol_deg: float = 1.0, name: str):
        return _sugar.assert_axis_world_aligned(
            self, joint=joint, world_axis=world_axis,
            tol_deg=tol_deg, name=name)

    def assert_flush(self, *, face_a, face_b, axis_world,
                     tol_pos: float = 1.0e-3, tol_deg: float = 1.0,
                     opposing: bool = True, name: str):
        return _sugar.assert_flush(
            self, face_a=face_a, face_b=face_b, axis_world=axis_world,
            tol_pos=tol_pos, tol_deg=tol_deg, opposing=opposing, name=name)

    def assert_opens_toward(self, *, moving_part, direction,
                            open_pose, closed_pose=None,
                            min_displacement: float = 0.001,
                            probe_anchor: str = "center",
                            name: str):
        return _sugar.assert_opens_toward(
            self, moving_part=moving_part, direction=direction,
            open_pose=open_pose, closed_pose=closed_pose,
            min_displacement=min_displacement, probe_anchor=probe_anchor,
            name=name)

    def assert_motion_sign(self, *, joint, moving_part, toward,
                           min_displacement: float = 0.005, name: str):
        return _sugar.assert_motion_sign(
            self, joint=joint, moving_part=moving_part, toward=toward,
            min_displacement=min_displacement, name=name)

    # ---- sugar helpers (consumed by _sugar.*) ----------------------------

    def _joint_default_lower_upper(self, joint):
        """Return (default_q, lower, upper) for a JointRef."""
        if not isinstance(joint, JointRef) or joint._owner_id != self._owner_id:
            raise InvalidJoint(
                f"_joint_default_lower_upper requires a JointRef from "
                f"this Object",
                refs=[_object_ref(self._name)],
                evidence={"reason": "foreign_joint_ref",
                          "value": type(joint).__name__},
                suggested_repairs=["pass a JointRef from this builder"],
            )
        j = self._joints[joint._id]
        if j["type"] == "fixed":
            raise InvalidJoint(
                f"joint {joint._name!r} is fixed; no range",
                refs=[_object_ref(self._name),
                      _joint_ref_dict(joint._name, joint._id)],
                evidence={"reason": "fixed_joint_no_range"},
                suggested_repairs=[
                    "use assert_motion_sign only on revolute/prismatic joints"
                ],
            )
        limits = j["limits"]
        return float(j["default"]), float(limits[0]), float(limits[1])

    # ---- root declaration ------------------------------------------------

    #: Ceiling on `add_check`. The five-lens planning walk yields 80-110
    #: criteria for a typical object, which is the right number to author
    #: against and far too many to review: every check costs the reviewer a
    #: verdict and a cited view, and a list that long is skimmed rather than
    #: checked. The cap exists to force selection at plan time rather than to
    #: express an opinion about which checks matter.
    MAX_CHECKS = 16

    def add_check(self, claim: str) -> None:
        """Commit to an observable claim the finished geometry must satisfy.

        Checks are the object's own review criteria. At verification an
        independent reviewer sees the rendered geometry and this list, and
        nothing else — not the user's prompt, not the assertions, not the
        source. It answers each claim `met`, `not_met`, or `cant_tell` and
        cites the view it judged from.

        Write each claim as a **relation a viewer could see**, not as a
        tolerance and not as a recipe:

          - good: "the lid closes with its tongue tucked under the front lip"
          - too much: "the tongue tip sits 1.5 mm below the lip's lower edge"
          - too little: "the lid closes properly"

        The first is falsifiable from an image. The second fails on a
        millimeter the reviewer cannot measure, and the third cannot fail at
        all. Numbers and tolerances belong in the `assert_*` family, which
        computes them exactly; a check is for what only looking can settle.

        Deliberately *not* passed to the reviewer: what the correct build
        looks like in detail, and what the wrong build would look like.
        Describing the intended geometry to someone about to look for it
        primes them to see it, so a claim carries the relation and stops
        there.

        The reviewer never blocks a commit — its verdict is advisory — so an
        unmet check is information for the next iteration rather than a gate.

        At most `MAX_CHECKS` per object, and duplicate claims are rejected:
        both keep the list short enough to be read rather than skimmed.
        Checks are recorded in declaration order and, being part of the IR,
        are covered by its content hash.
        """
        self._check_not_finalized()
        if not isinstance(claim, str):
            raise BuildError(
                f"add_check: claim must be a string, got "
                f"{type(claim).__name__}",
                refs=[_object_ref(self._name)],
                evidence={"reason": "claim_not_string",
                          "value": type(claim).__name__},
                suggested_repairs=["pass one observable claim as a string"],
            )
        text = " ".join(claim.split())
        if not text:
            raise BuildError(
                "add_check: claim is empty",
                refs=[_object_ref(self._name)],
                evidence={"reason": "claim_empty"},
                suggested_repairs=[
                    "state what a viewer would see, e.g. "
                    "\"the lid's tongue tucks under the front lip when closed\"",
                ],
            )
        if text in self._check_claims:
            raise BuildError(
                f"add_check: duplicate claim {text!r}",
                refs=[_object_ref(self._name)],
                evidence={"reason": "duplicate_claim", "value": text},
                suggested_repairs=["drop the duplicate, or narrow one of them"],
            )
        if len(self._checks) >= self.MAX_CHECKS:
            raise BuildError(
                f"add_check: at most {self.MAX_CHECKS} checks per object "
                f"({len(self._checks)} already declared)",
                refs=[_object_ref(self._name)],
                evidence={"reason": "too_many_checks",
                          "limit": self.MAX_CHECKS,
                          "declared": len(self._checks)},
                suggested_repairs=[
                    "keep the checks that would be invisible in a compile "
                    "error and fatal to the object, and drop the rest",
                    "move anything numeric to the assert_* family instead",
                ],
            )
        self._checks.append({"id": f"c{len(self._checks) + 1:02d}",
                             "claim": text})
        self._check_claims.add(text)

    def declare_root(self, part: PartRef) -> None:
        """Designate `part` as the kinematic root.

        The root has no incoming joint; its frame is the object's local
        origin and all other parts are positioned relative to it via the
        joint chain. The object's local origin is not observable downstream
        — scenes place an instance by aligning a chosen anchor (default the
        object's bbox-bottom-center) to a world pose, so where the origin
        falls within the geometry does not matter.
        """
        self._check_not_finalized()
        part = self._check_owned(part, "part")
        if self._root_part_id is not None:
            raise BuildError(
                f"root already declared as {self._root_part_name!r}; "
                f"cannot redeclare as {part._name!r}",
                refs=[
                    _object_ref(self._name),
                    _part_ref_dict(self._root_part_name, self._root_part_id),
                    _part_ref_dict(part._name, part._id),
                ],
                evidence={"reason": "root_already_declared",
                          "existing_root": self._root_part_name,
                          "redeclare_as": part._name},
                suggested_repairs=["call declare_root exactly once"],
            )
        if part._id in self._parent_of_child:
            incoming = self._joints[self._parent_of_child[part._id]]["name"]
            raise InvalidJoint(
                f"cannot declare {part._name!r} as root: it has incoming joint {incoming!r}",
                refs=[
                    _object_ref(self._name),
                    _part_ref_dict(part._name, part._id),
                    _joint_ref_dict(incoming, self._parent_of_child[part._id]),
                ],
                evidence={"reason": "root_has_incoming_joint",
                          "value": {"part": part._name, "incoming_joint": incoming}},
                suggested_repairs=[
                    "declare a part with no incoming joint as the root",
                ],
            )
        self._root_part_id = part._id
        self._root_part_name = part._name

    # ---- collision allowances --------------------------------------------

    def allow_collision(self, a: "PartRef", b: "PartRef", *,
                        reason: str,
                        joint: "JointRef | None" = None,
                        q_range: "tuple[float, float] | None" = None) -> None:
        """Declare an intentional self-overlap between `a` and `b`.

        `NoSelfPenetration` will skip this pair during its SAT sweep
        for any CCD sample whose pose matches the allowance scope.

        Scoping:
          - `joint=None`, `q_range=None` (default): blanket allowance
            — every sample pose is exempt for this pair.
          - `joint=<JointRef>`: the allowance only applies when the
            sample's `joint` value lies in `q_range`. Use this to
            permit the overlap during one joint's sweep without
            masking real bugs at other poses.
          - `q_range=(lo, hi)`: required when `joint` is set. Surface
            units — degrees for revolute, meters for prismatic.
            Half-open is fine: pass the joint's full `limits` to
            allow at any q on that joint.

        You may register multiple scoped allowances for the same `(a, b)`
        pair as long as the (joint, q_range) tuples differ. A blanket
        allowance and a scoped allowance for the same pair both raise.

        Allowances are symmetric: `(a, b)` and `(b, a)` collapse.
        Self-pair `a == b` is rejected at build time.

        `reason` is a short author-supplied string preserved in the IR
        so the rationale stays visible to readers, the renderer, and
        downstream verifier reports.
        """
        self._check_not_finalized()
        a = self._check_owned(a, "part")
        b = self._check_owned(b, "part")
        if a._id == b._id:
            raise BuildError(
                f"allow_collision: self-pair {a._name!r} not allowed",
                refs=[_object_ref(self._name),
                      _part_ref_dict(a._name, a._id)],
                evidence={"reason": "self_pair", "value": a._name},
                suggested_repairs=["pass two distinct parts"],
            )
        if not isinstance(reason, str) or len(reason) == 0:
            raise BuildError(
                "allow_collision: reason must be a non-empty string",
                refs=[_object_ref(self._name),
                      _part_ref_dict(a._name, a._id),
                      _part_ref_dict(b._name, b._id)],
                evidence={"reason": "bad_reason_type",
                          "value": repr(reason)},
                suggested_repairs=["pass a short non-empty reason string"],
            )
        if len(reason) > 256:
            raise BuildError(
                "allow_collision: reason exceeds 256 characters",
                refs=[_object_ref(self._name),
                      _part_ref_dict(a._name, a._id),
                      _part_ref_dict(b._name, b._id)],
                evidence={"reason": "reason_too_long",
                          "value": len(reason)},
                suggested_repairs=["shorten the reason to <=256 chars"],
            )
        # Validate optional scope.
        scope_joint_name = None
        scope_q_lo = None
        scope_q_hi = None
        if joint is not None or q_range is not None:
            if joint is None:
                raise BuildError(
                    "allow_collision: q_range requires a joint=",
                    refs=[_object_ref(self._name),
                          _part_ref_dict(a._name, a._id),
                          _part_ref_dict(b._name, b._id)],
                    evidence={"reason": "q_range_without_joint",
                              "value": q_range},
                    suggested_repairs=[
                        "pass joint=<joint name or JointRef> alongside q_range, or omit both"
                    ],
                )
            # A joint NAME is accepted here for the same reason `a` and `b`
            # accept part names: this builder owns one joint namespace, so the
            # string can only mean one thing. `internet_cafe` hit exactly this
            # with `joint="hinge"`.
            if isinstance(joint, str):
                joint = self._check_owned(joint, "joint")
            if not isinstance(joint, JointRef) or joint._owner_id != self._owner_id:
                raise BuildError(
                    "allow_collision: joint must be a joint name or JointRef from this Object",
                    refs=[_object_ref(self._name),
                          _part_ref_dict(a._name, a._id),
                          _part_ref_dict(b._name, b._id)],
                    evidence={"reason": "foreign_joint_ref"},
                    suggested_repairs=[
                        "pass a joint name or JointRef from this Object's attach_*"
                    ],
                )
            if joint._joint_type not in _JOINT_KINDS_WITH_AXIS:
                raise BuildError(
                    f"allow_collision: joint {joint._name!r} is fixed; "
                    f"scoped allowances only apply to revolute/prismatic joints",
                    refs=[_object_ref(self._name),
                          _joint_ref_dict(joint._name, joint._id)],
                    evidence={"reason": "scope_joint_is_fixed",
                              "joint": joint._name},
                    suggested_repairs=[
                        "omit joint=/q_range= for a blanket allowance, or "
                        "pick a revolute/prismatic joint"
                    ],
                )
            if q_range is None:
                # Default to the joint's full limits in surface units
                # (degrees for revolute, meters for prismatic).
                from . import _rotation as _rot
                j_rec = self._joints[joint._id]
                lo_int, hi_int = j_rec["limits"]
                if j_rec["type"] == _JOINT_REVOLUTE:
                    lo = _rot.rad_to_deg(lo_int)
                    hi = _rot.rad_to_deg(hi_int)
                else:  # prismatic
                    lo, hi = float(lo_int), float(hi_int)
            else:
                if (not isinstance(q_range, (tuple, list)) or len(q_range) != 2):
                    raise BuildError(
                        "allow_collision: q_range must be (lo, hi)",
                        refs=[_object_ref(self._name),
                              _part_ref_dict(a._name, a._id),
                              _part_ref_dict(b._name, b._id)],
                        evidence={"reason": "bad_q_range",
                                  "value": repr(q_range)},
                        suggested_repairs=[
                            "pass q_range=(lo, hi) — degrees for revolute, "
                            "meters for prismatic"
                        ],
                    )
                lo, hi = float(q_range[0]), float(q_range[1])
                if lo > hi:
                    raise BuildError(
                        f"allow_collision: q_range lo > hi ({lo} > {hi})",
                        refs=[_object_ref(self._name),
                              _part_ref_dict(a._name, a._id),
                              _part_ref_dict(b._name, b._id)],
                        evidence={"reason": "q_range_inverted",
                                  "value": [lo, hi]},
                        suggested_repairs=[
                            "ensure q_range[0] <= q_range[1]"
                        ],
                    )
            scope_joint_name = joint._name
            scope_q_lo, scope_q_hi = lo, hi
        a_name, b_name = a._name, b._name
        if a_name > b_name:
            a_name, b_name = b_name, a_name
        key = (a_name, b_name, scope_joint_name,
               None if scope_q_lo is None else (scope_q_lo, scope_q_hi))
        if key in self._allowance_pair_keys:
            raise BuildError(
                f"allow_collision: duplicate allowance for "
                f"({a_name!r}, {b_name!r}, joint={scope_joint_name!r}, "
                f"q_range={None if scope_q_lo is None else (scope_q_lo, scope_q_hi)!r})",
                refs=[_object_ref(self._name),
                      _part_ref_dict(a._name, a._id),
                      _part_ref_dict(b._name, b._id)],
                evidence={"reason": "duplicate_allowance",
                          "value": [a_name, b_name, scope_joint_name,
                                    [scope_q_lo, scope_q_hi]
                                    if scope_q_lo is not None else None]},
                suggested_repairs=["remove the duplicate allow_collision call"],
            )
        # Reject mixing a blanket allowance with a scoped one for the
        # same pair — the blanket already exempts every pose.
        same_pair_keys = {k for k in self._allowance_pair_keys
                          if k[0] == a_name and k[1] == b_name}
        if same_pair_keys:
            has_blanket = any(k[2] is None for k in same_pair_keys)
            new_is_blanket = scope_joint_name is None
            if has_blanket or new_is_blanket:
                raise BuildError(
                    f"allow_collision: pair ({a_name!r}, {b_name!r}) "
                    f"already has an allowance — a blanket allowance "
                    f"cannot coexist with scoped ones",
                    refs=[_object_ref(self._name),
                          _part_ref_dict(a._name, a._id),
                          _part_ref_dict(b._name, b._id)],
                    evidence={"reason": "blanket_scoped_conflict",
                              "value": [a_name, b_name]},
                    suggested_repairs=[
                        "drop the blanket allowance and use only scoped ones",
                        "or drop the scoped allowance and rely on the blanket",
                    ],
                )
        self._allowance_pair_keys.add(key)
        record = {"a": a_name, "b": b_name, "reason": reason}
        if scope_joint_name is not None:
            record["joint"] = scope_joint_name
            record["q_range"] = [scope_q_lo, scope_q_hi]
        self._allowances.append(record)

    # ---- finalize --------------------------------------------------------

    def finalize(self) -> dict:
        if self._finalized:
            return self._cached_ir  # idempotent

        # Turn the snapshots `add_part` registered into staged artifacts and
        # complete params. A no-op for an object with none.
        self._stage_registered_parts()

        if self._root_part_id is None:
            raise RootNotDeclared(
                f"object {self._name!r}: declare_root was never called",
                refs=[_object_ref(self._name)],
                evidence={"reason": "root_never_declared"},
                suggested_repairs=["call obj.declare_root(part) before finalize()"],
            )

        # Reachability BFS from root via parent->child edges.
        reachable = {self._root_part_id}
        frontier = [self._root_part_id]
        while frontier:
            next_frontier = []
            for pid in frontier:
                for cid in self._children_of_parent.get(pid, ()):
                    if cid not in reachable:
                        reachable.add(cid)
                        next_frontier.append(cid)
            frontier = next_frontier

        if len(reachable) != len(self._parts):
            unreachable_parts = [p for p in self._parts if p["id"] not in reachable]
            unreachable = [p["name"] for p in unreachable_parts]
            raise NotConnectedToRoot(
                f"object {self._name!r}: parts not connected to root "
                f"{self._root_part_name!r}: {unreachable}",
                refs=[_object_ref(self._name)] + [
                    _part_ref_dict(p["name"], p["id"]) for p in unreachable_parts
                ],
                evidence={"unreachable": unreachable,
                          "root": self._root_part_name},
                suggested_repairs=[
                    "attach unreachable parts to the root via a joint chain, "
                    "or remove them",
                ],
            )

        # Attach semantic anchors to their owning parts, sorted by name.
        # Sorted rather than insertion-ordered so serialization is
        # deterministic; the key is omitted entirely when a part has none, so an
        # AABB-only part carries no `anchors` field at all.
        for part in self._parts:
            anchors = self._part_anchors.get(part["id"])
            if anchors:
                part["anchors"] = [anchors[n] for n in sorted(anchors)]

        ir_no_id = {
            "schema": OBJECT_SCHEMA_VERSION,
            "name": self._name,
            "root": self._root_part_name,
            "parts": list(self._parts),
            "joints": list(self._joints),
            "assertions": list(self._assertions),
            "collision_allowances": list(self._allowances),
        }
        # The authored program, once per object. Omitted entirely when there is
        # none — a placeholder object of analytic boxes was never executed from
        # a program — so the key's presence means "this geometry was authored"
        # rather than defaulting to an empty string that reads the same as a
        # program that happened to be blank.
        if self._source_asset is not None:
            ir_no_id["source"] = self._source_asset
        # Omitted entirely when the author declared none, for the same reason
        # `source` is: an absent key reads as "this object set no review
        # criteria", where an empty list would read as "it set none on
        # purpose". It also keeps the IR hash of an object with no checks
        # independent of this field.
        if self._checks:
            ir_no_id["checks"] = list(self._checks)
        self._cached_ir = insert_id(ir_no_id)
        self._finalized = True
        # Register for `take_finalized_objects`. Inside the finalized guard, so
        # the idempotent second call does not re-register, and unconditional so
        # that an object built but never placed still reaches the catalog —
        # the author may be inspecting it before deciding where it goes, and a
        # definition that vanishes when it is not yet referenced is worse to
        # debug than one unused entry.
        with _object_singleton_lock:
            _finalized_objects[self._cached_ir["id"]] = self._cached_ir
        return self._cached_ir
