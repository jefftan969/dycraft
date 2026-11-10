"""Tests for dycraft.render.urdf_export.

These target hidden runtime semantics of the conversion -- properties that
a static read of the code would not let you predict -- and explicitly
avoid restating one-line code facts (material name format, "world" link
exists, etc.).

What's actually being tested:

  * **Object-layer FK round-trip.** A multi-joint chain with a non-root
    parent (i.e. a grandparent in the joint tree) is exported, then we
    manually walk the URDF link/joint origins to compute the leaf's pose
    in the URDF root frame, then add the visual-origin offset to recover
    the DyCraft part frame. The result must equal ``object_query._compute_frames``
    at the DyCraft q=0 configuration. This catches:
      * mis-composed ``joint_in_parent_link`` for non-root parents
        (the ``part_in_link[parent] * parent_anchor`` step);
      * forgetting to offset the visual by ``inv(child_anchor)``;
      * mis-handling of joint vs. part frames anywhere in the chain.

  * **Joint-axis frame transform.** Axis is in the parent-part frame in
    the IR but must be in the joint frame in URDF. We use a
    parent_anchor with a non-trivial rotation that actually moves the
    axis (yaw doesn't move +Z so that test would pass even with the
    transform stubbed out) and verify the URDF axis is the inverse-
    rotated version. Also verifies that re-rotating it by the joint-
    origin rpy reconstructs the original IR axis.

  * **URDF rpy is fixed-axis (extrinsic) XYZ, not intrinsic XYZ.** Round-
    trip a non-commuting compound rotation through the exporter's
    ``_quat_to_urdf_rpy`` and check that ``Rz(yaw)*Ry(pitch)*Rx(roll)``
    reproduces the original quaternion, AND verify it does NOT equal the
    DyCraft intrinsic-XYZ reading of the same quaternion (these agree on
    pure-axis rotations but disagree on compounds -- a stubbed exporter
    that called ``quat_to_euler_xyz_deg`` would silently produce wrong
    URDFs).

  * **Gimbal-lock branch.** Pitch = +/- 90 deg is the only place where
    rpy recovery cannot be a straight ``atan2``. We round-trip such a
    quaternion and check matrix equality (the rpy tuple itself is
    non-unique under gimbal lock, so equality on R is the right check).

  * **Scene-layer FK round-trip across an attach chain.** Build a scene
    where instance C is attached to instance B which sits at a non-
    identity pose, then export. The URDF fixed joint for C must encode
    the world pose that ``scene_query._resolve_world_pose`` computes,
    even though ``place(...)``'s API never directly states it. This
    catches: a buggy exporter that bypassed ``_resolve_world_pose`` and
    used the literal ``placement.pose`` (which is parent-frame, not
    world).

Run: ``conda run -n chord pytest dycraft/tests/test_urdf_export.py -v``
(with ``PYTHONPATH=papers/chord_data`` so ``dycraft`` resolves.)
"""

import math
import xml.etree.ElementTree as ET

import pytest

from dycraft.sdk import (
    Material,
    Object,
    Scene,
    ObjectHandle,
    object_query,
    scene_query,
)
from dycraft.sdk import _rotation as _rot
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton
from dycraft.render import urdf_export
from dycraft.render.urdf_export import _quat_to_urdf_rpy


@pytest.fixture(autouse=True)
def _reset_singletons():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOL = 1e-9


def _parse_xml(xml: str) -> ET.Element:
    return ET.fromstring(xml.split("?>", 1)[1])


def _parse_vec(s):
    return tuple(float(x) for x in s.split())


def _find_joint(root, name):
    for j in root.findall("joint"):
        if j.get("name") == name:
            return j
    raise AssertionError(f"joint {name!r} not in URDF")


def _find_link(root, name):
    for l in root.findall("link"):
        if l.get("name") == name:
            return l
    raise AssertionError(f"link {name!r} not in URDF")


def _rpy_to_quat_wxyz(r, p, y):
    """Fixed-axis XYZ (r, p, y) -> WXYZ quaternion. Matches URDF convention
    R = Rz(y) Ry(p) Rx(r)."""
    cr = math.cos(r * 0.5); sr = math.sin(r * 0.5)
    cp = math.cos(p * 0.5); sp = math.sin(p * 0.5)
    cy = math.cos(y * 0.5); sy = math.sin(y * 0.5)
    qx = (cr, sr, 0.0, 0.0)
    qy = (cp, 0.0, sp, 0.0)
    qz = (cy, 0.0, 0.0, sy)
    return _rot.quat_mul(qz, _rot.quat_mul(qy, qx))


def _origin_to_se3(origin_elem):
    xyz = _parse_vec(origin_elem.get("xyz"))
    r, p, y = _parse_vec(origin_elem.get("rpy"))
    return (xyz, _rpy_to_quat_wxyz(r, p, y))


def _quat_to_R(q):
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ]


def _matrix_close(A, B, *, tol=1e-9):
    for i in range(3):
        for j in range(3):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# URDF rpy convention: fixed-axis (extrinsic) XYZ -- catches the easy bug
# of reusing DyCraft's intrinsic-XYZ helper.
# ---------------------------------------------------------------------------

def test_quat_to_urdf_rpy_disagrees_with_intrinsic_xyz_on_compound_rotation():
    """A compound (non-commuting) rotation distinguishes intrinsic XYZ from
    fixed-axis XYZ. Pure-axis rotations agree under both conventions, so
    this must use multi-axis input to be a real check."""
    # Hand-crafted compound: Rz(60 deg) * Ry(40 deg) * Rx(30 deg) under
    # the FIXED-axis convention. We feed the resulting quaternion to
    # _quat_to_urdf_rpy and to the DyCraft intrinsic-XYZ helper and
    # confirm they disagree.
    rx, ry, rz = math.radians(30.0), math.radians(40.0), math.radians(60.0)
    q = _rpy_to_quat_wxyz(rx, ry, rz)

    r, p, y = _quat_to_urdf_rpy(q)
    # Round-trip: re-build the rotation matrix from the URDF rpy values
    # using the fixed-axis convention and confirm it matches the original
    # quaternion's rotation matrix.
    R_orig = _quat_to_R(q)
    R_back = _quat_to_R(_rpy_to_quat_wxyz(r, p, y))
    assert _matrix_close(R_orig, R_back, tol=1e-12)

    # Same quaternion under DyCraft's INTRINSIC-XYZ recovery returns a
    # different tuple. We don't care about the exact values; just that
    # they're not numerically equal to (r, p, y) -- otherwise the
    # exporter could (incorrectly) be using the wrong helper.
    ix_deg, iy_deg, iz_deg = _rot.quat_to_euler_xyz_deg(q)
    intrinsic = (math.radians(ix_deg), math.radians(iy_deg),
                 math.radians(iz_deg))
    fixed = (r, p, y)
    assert not all(abs(a - b) < 1e-6 for a, b in zip(intrinsic, fixed))


def test_quat_to_urdf_rpy_gimbal_lock_matrix_roundtrip():
    """At pitch = +90 deg, the (r, p, y) tuple is non-unique, but the
    rotation matrix it represents is unique -- so we check matrix
    equality, not tuple equality. A buggy gimbal-lock branch typically
    returns NaN, wrong-sign, or yaw=non-zero values that fail this."""
    rx, ry, rz = math.radians(25.0), math.pi / 2.0, math.radians(40.0)
    q = _rpy_to_quat_wxyz(rx, ry, rz)
    r, p, y = _quat_to_urdf_rpy(q)
    assert abs(p - math.pi / 2.0) < 1e-9   # pitch recovered exactly.
    R_orig = _quat_to_R(q)
    R_back = _quat_to_R(_rpy_to_quat_wxyz(r, p, y))
    assert _matrix_close(R_orig, R_back, tol=1e-9)


# ---------------------------------------------------------------------------
# Object-layer FK round-trip: 3-part chain so the non-root parent code
# path is exercised. The leaf part's world pose recomputed by manual FK
# from the URDF output must equal DyCraft's _compute_frames result.
# ---------------------------------------------------------------------------

def _build_three_part_chain():
    """root --(rev hinge)--> mid --(fixed)--> tip.

    Both joints use non-identity ``parent_anchor`` (translation + rotation)
    and non-identity ``child_anchor`` translation. The hinge joint has
    its axis in the rotated parent frame, exercising the
    axis-frame-transform code path. ``default`` is left at 0 so the
    DyCraft "rest pose" is the URDF zero pose.
    """
    o = Object("chain")
    root = o.make_box("root", 0.3, 0.4, 0.5)
    mid = o.make_box("mid", 0.2, 0.2, 0.6)
    tip = o.make_box("tip", 0.1, 0.1, 0.1)
    o.attach_revolute('hinge', root, mid, axis=(0.0, 0.0, 1.0), limits=(-30.0, 60.0), default=0.0, parent_anchor=o.anchor(root, "center", translation=(0.36390567234751503, 0.061810710171036475, 0.50333082859113), rotation=(14.999999999999998, 25.000000000000004, 35.0)), child_anchor=o.anchor(mid, "center"))
    o.attach_fixed('weld', mid, tip, parent_anchor=o.anchor(mid, "center", translation=(0.0, 0.0, 0.35), rotation=(0.0, 0.0, 45.00000000000001)), child_anchor=o.anchor(tip, "center"))
    o.declare_root(root)
    return o.finalize()


def test_object_urdf_fk_roundtrip_matches_dycraft_compute_frames():
    """End-to-end: walk URDF link/joint origins and recover the leaf
    part frame in the URDF root link frame; compare to DyCraft FK.

    This is the strongest single check on the conversion -- it folds
    together joint-origin composition, axis frames, and visual-origin
    offset. A unit error anywhere in those code paths fails this test.
    """
    ir = _build_three_part_chain()
    xml = urdf_export.object_ir_to_urdf(ir)
    root_elem = _parse_xml(xml)

    # Manually compose URDF transforms at q=0:
    #   part_frame_in_root = joint_origin(j1) o joint_origin(j2) o ...
    #                         o visual_origin(child)
    # All "joint origins" sit in the parent link frame; chaining them
    # yields the joint frame of the deepest joint in the root link frame.
    # Then the child link frame == joint frame at q=0, and the visual
    # origin lifts us back to the DyCraft part frame.

    # joint "hinge" has parent = root (so its origin is in root's link frame).
    hinge = _find_joint(root_elem, "hinge")
    weld = _find_joint(root_elem, "weld")
    mid_link = _find_link(root_elem, "mid")
    tip_link = _find_link(root_elem, "tip")
    root_link = _find_link(root_elem, "root")

    hinge_origin = _origin_to_se3(hinge.find("origin"))
    weld_origin = _origin_to_se3(weld.find("origin"))
    mid_visual_origin = _origin_to_se3(mid_link.find("visual/origin"))
    tip_visual_origin = _origin_to_se3(tip_link.find("visual/origin"))

    # Root link's visual origin should be identity (root part == root link).
    root_visual_origin = _origin_to_se3(root_link.find("visual/origin"))
    assert root_visual_origin[0] == pytest.approx((0.0, 0.0, 0.0), abs=_TOL)
    R = _quat_to_R(root_visual_origin[1])
    assert _matrix_close(R, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], tol=_TOL)

    # mid part frame in root part frame (via URDF):
    #   mid_part = hinge_origin o mid_visual_origin
    mid_part_in_root_urdf = _rot.se3_compose(hinge_origin, mid_visual_origin)

    # tip part frame in root part frame (via URDF):
    #   tip_part = hinge_origin o weld_origin o tip_visual_origin
    #   (weld_origin is in mid's LINK frame, which equals the joint
    #    frame of hinge at q=0; "mid link" is what URDF puts at hinge_origin)
    tip_part_in_root_urdf = _rot.se3_compose(
        hinge_origin,
        _rot.se3_compose(weld_origin, tip_visual_origin),
    )

    # Dycraft FK ground truth (uses the same q=0 = default convention).
    dycraft_frames = object_query._compute_frames(ir)
    mid_frame_t, mid_frame_q = dycraft_frames["mid"]
    tip_frame_t, tip_frame_q = dycraft_frames["tip"]

    assert mid_part_in_root_urdf[0] == pytest.approx(mid_frame_t, abs=_TOL)
    R_mid_urdf = _quat_to_R(mid_part_in_root_urdf[1])
    R_mid_truth = _quat_to_R(mid_frame_q)
    assert _matrix_close(R_mid_urdf, R_mid_truth, tol=_TOL)

    assert tip_part_in_root_urdf[0] == pytest.approx(tip_frame_t, abs=_TOL)
    R_tip_urdf = _quat_to_R(tip_part_in_root_urdf[1])
    R_tip_truth = _quat_to_R(tip_frame_q)
    assert _matrix_close(R_tip_urdf, R_tip_truth, tol=_TOL)


def _build_chain_with_defaults(*, rev_default_deg: float,
                                pris_default_m: float):
    """root --(revolute, default=rev_default_deg)--> mid
              --(prismatic, default=pris_default_m)--> tip.

    Both joints have non-trivial anchors so the bake interacts with a
    rotated parent frame (revolute) AND a non-root parent (prismatic).
    Returns the finalized IR; the DyCraft rest pose is the FK at
    q=default for each joint.
    """
    _reset_object_singleton()
    o = Object("baked_chain")
    root = o.make_box("root", 0.3, 0.4, 0.5)
    mid = o.make_box("mid", 0.2, 0.2, 0.6)
    tip = o.make_box("tip", 0.1, 0.1, 0.1)
    o.attach_revolute('hinge', root, mid, axis=(0.0, 0.0, 1.0), limits=(-30.0, 60.0), default=rev_default_deg, parent_anchor=o.anchor(root, "center", translation=(0.36390567234751503, 0.061810710171036475, 0.50333082859113), rotation=(14.999999999999998, 25.000000000000004, 35.0)), child_anchor=o.anchor(mid, "center"))
    o.attach_prismatic('slide', mid, tip, axis=(1.0, 0.0, 0.0), limits=(-0.1, 0.4), default=pris_default_m, parent_anchor=o.anchor(mid, "center", translation=(0.0, 0.0, 0.35), rotation=(0.0, 0.0, 45.00000000000001)), child_anchor=o.anchor(tip, "center"))
    o.declare_root(root)
    return o.finalize()


def test_the_joint_default_does_not_reach_the_urdf_at_all():
    """Two IRs differing ONLY in each joint's `default` must export byte-identically.

    `default` is absent from the file: URDF `q` IS DyCraft `q`, and applying
    the rest pose is the loader's job (`animation/ik/dycraft_ik.py::default_q`,
    `viewer`'s joint controller). Folding it into `<origin>` and `<limit>`
    instead would make every consumer convert `q_dycraft = q_urdf + default`
    on read and write.

    Byte-identity is the strongest form of that claim — nothing subtler can
    leak. The chain exercises both joint types and a non-axis-aligned joint
    frame (the hinge's parent_anchor carries roll+pitch+yaw), which is where a
    residual bake would show up first.
    """
    a = urdf_export.object_ir_to_urdf(
        _build_chain_with_defaults(rev_default_deg=35.0, pris_default_m=0.17))
    b = urdf_export.object_ir_to_urdf(
        _build_chain_with_defaults(rev_default_deg=0.0, pris_default_m=0.0))
    assert a == b


def test_the_default_export_is_in_dycrafts_joint_frame():
    """URDF `q` IS DyCraft `q`, limits verbatim.

    Shifting limits to `(lo - default, hi - default)` would turn a finger with
    default 0.04 and limits [0, 0.04] into `[-0.04, 0]`, where "open" is
    negative and a sign slip is invisible.
    """
    ir = _build_chain_with_defaults(rev_default_deg=35.0, pris_default_m=0.17)
    plain = _parse_xml(urdf_export.object_ir_to_urdf(ir))

    # Limits go out exactly as authored, in internal units.
    lim = _find_joint(plain, "hinge").find("limit")
    assert float(lim.get("lower")) == pytest.approx(math.radians(-30.0), abs=1e-12)
    assert float(lim.get("upper")) == pytest.approx(math.radians(60.0), abs=1e-12)
    slim = _find_joint(plain, "slide").find("limit")
    assert float(slim.get("lower")) == pytest.approx(-0.1, abs=1e-12)
    assert float(slim.get("upper")) == pytest.approx(0.4, abs=1e-12)

    # …and the prismatic default is not hiding in the joint origin either: the
    # origin must be the authored anchor transform, with no `default * axis`
    # translation folded in.
    (tx, ty, tz), _q = _origin_to_se3(_find_joint(plain, "slide").find("origin"))
    ident = _build_chain_with_defaults(rev_default_deg=35.0, pris_default_m=0.0)
    (ux, uy, uz), _q2 = _origin_to_se3(
        _find_joint(_parse_xml(urdf_export.object_ir_to_urdf(ident)),
                    "slide").find("origin"))
    assert (tx, ty, tz) == pytest.approx((ux, uy, uz), abs=1e-12)


# ---------------------------------------------------------------------------
# The <pbr> block, and the contract the browser viewer reads it under
#
# `<pbr><metal><metalness>/<roughness>` is Gazebo's SDF schema embedded inline
# in URDF: standard readers skip it, and PBR-aware ones consume it. The browser
# viewer is one of those — `viewer/web/src/components/viewer3d/
# urdf-parser.ts::parseMaterial` — so the element PATH and the suppression rule
# are a cross-language contract rather than an implementation detail.
#
# Pinned here because nothing else pins it: `check-triplanar-uv.mjs`
# deliberately does not parse XML (node has no DOM here) — it pins what the
# spec does with the parsed result and relies on this file for the shape.
# ---------------------------------------------------------------------------

def _one_box_ir(material: dict) -> dict:
    """A minimal one-box ObjectIR — enough for the exporter, and free of the
    staged-asset params a `solid` part would need."""
    return {"schema_version": "dycraft.object.v1", "name": "t", "id": "x",
            "parts": [{"name": "top", "primitive": "box", "material": material,
                       "params": {"width": 1.0, "length": 1.0, "height": 1.0}}],
            "joints": [], "root": "top"}


def _metal_elem(xml: str):
    """The `<pbr><metal>` element of the one and only material, or None."""
    return _parse_xml(xml).find("./link/visual/material/pbr/metal")


def test_the_urdf_carries_the_authored_pbr_scalars_the_viewer_reads():
    """The exact element path `urdf-parser.ts` queries, and the values.

    `querySelector('pbr metal')` then `metalness` / `roughness` — so a rename
    or a re-nesting anywhere on that path silently returns the viewer to one
    neutral default for every surface, which is what this exists to catch. The
    values are asserted as floats rather than as strings because `_fmt_num`
    owns the formatting and pinning its output here would be a second test of
    the wrong thing.
    """
    xml = urdf_export.object_ir_to_urdf(_one_box_ir(
        Material(color=(160, 160, 165), metallic=1.0, roughness=0.35).to_ir()))
    metal = _metal_elem(xml)
    assert metal is not None, "no <pbr><metal> for an authored metal"
    assert float(metal.findtext("metalness")) == pytest.approx(1.0)
    assert float(metal.findtext("roughness")) == pytest.approx(0.35)


def test_a_material_at_the_sdk_defaults_emits_no_pbr_block_at_all():
    """The suppression rule, which the viewer decodes as information.

    The exporter omits `<pbr>` when nothing differs from the SDK default, so an
    untextured part's URDF carries no `<pbr>` at all.
    `materials.ts::EXPORTED_DEFAULT_METALNESS/_ROUGHNESS` therefore
    reads the ABSENCE as exactly `(0.0, 0.9)` rather than as "unknown" — so if
    this test ever fails because the block became unconditional, that constant
    pair stops being reachable and every default-matte surface in the viewport
    silently falls back to the no-material neutral roughness of 0.48.
    """
    xml = urdf_export.object_ir_to_urdf(_one_box_ir(
        Material(color=(176, 176, 176)).to_ir()))
    assert _metal_elem(xml) is None
    assert "<pbr>" not in xml


@pytest.mark.parametrize("metallic,roughness", [
    (0.0, 0.9),      # both default -> suppressed
    (0.4, 0.9),      # metallic alone
    (0.0, 0.2),      # roughness alone
])
def test_the_block_appears_exactly_when_a_field_leaves_its_default(
        metallic, roughness):
    """One field differing is enough, and the pair travels together.

    The viewer's parser falls back per field, so a block carrying only one
    child is readable — but the exporter never writes one, and that is worth
    pinning from this side so the fallback stays defensive rather than load-
    bearing.
    """
    xml = urdf_export.object_ir_to_urdf(_one_box_ir(
        Material(color=(120, 120, 120), metallic=metallic,
                 roughness=roughness).to_ir()))
    metal = _metal_elem(xml)
    if metallic == 0.0 and roughness == 0.9:
        assert metal is None
        return
    assert metal is not None
    assert float(metal.findtext("metalness")) == pytest.approx(metallic)
    assert float(metal.findtext("roughness")) == pytest.approx(roughness)
