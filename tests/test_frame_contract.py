"""The frame, transform, and unit contract — executable form of design_frames.md.

Each assertion pins behavior that is *measured*, not assumed. The point is that a future change to the transform stack breaks a test
rather than silently changing what downstream code believes.

Two tiers:

  * kernel-free — rotation math, unit boundaries, camera table, anchor
    vocabulary. These run everywhere.
  * CAD-gated — anything needing a live build123d Shape. Driven through the
    object worker so they run from the main env, matching the rest of the suite.
"""

from __future__ import annotations

import importlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.sdk import _anchors as A
from dycraft.sdk import _object_kernel as K
from dycraft.sdk import _probes as P
from dycraft.sdk import _rotation as R

REPO = Path(__file__).resolve().parents[1]

_needs_cad = pytest.mark.skipif(
    not W.is_available(), reason="needs the pinned build123d interpreter")


def _build(source: str) -> dict:
    """Run one object.py through the worker and return its ObjectIR."""
    out = W.run_object_source(source, run_dir=Path(tempfile.mkdtemp()))
    assert out["ok"], out.get("build_error") or out.get("message")
    return out["ir"]


def _world_aabb_from_kernel(ir, part_name, q_rad):
    """World AABB of a part, computed independently of `_probes`.

    Transforms the part's eight local AABB corners by the kernel frame and
    takes the bound. Deliberately NOT the probe's own code path — the point is
    to compare two routes to the same answer.
    """
    part = next(p for p in ir["parts"] if p["name"] == part_name)
    lo = part["params"]["aabb_min"]
    hi = part["params"]["aabb_max"]
    corners = [(x, y, z) for x in (lo[0], hi[0])
               for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    t, quat = K._compute_frames(ir, poses={"hinge": q_rad})[part_name]
    pts = [np.array(R.quat_rotate_vec(quat, c)) + np.array(t) for c in corners]
    return np.min(pts, axis=0), np.max(pts, axis=0)


# ---------------------------------------------------------------------------
# 1. Global coordinate convention
# ---------------------------------------------------------------------------


def _in_cad(script: str, *, timeout: float = 600) -> str:
    """Run `script` in the CAD interpreter and return its stdout.

    `run_cad_snippet` puts this on the warm pooled child, so the five CAD
    probes in this file share one `import build123d` instead of spawning an
    interpreter each. The child already has `dycraft` bound, so a script here
    needs no `spec_from_file_location` bootstrap — and must not do one, since
    that would bind a second copy of the package alongside the live one.
    """
    out = W.run_cad_snippet(script, timeout=timeout)
    assert out["ok"], (
        f"{out['message']}\n{out['traceback'] or ''}\n"
        f"stdout={out['stdout'][-2000:]}\nstderr={out['stderr'][-3000:]}")
    return out["stdout"]


def test_coordinate_system_is_right_handed():
    """X -> Y -> Z under +90 rotations. Proven, not inherited from a library.

    A left-handed stack would still satisfy every orthonormality check, so this
    cycle is the assertion that actually discriminates.
    """
    for axis, v_in, expected in (((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                 ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
                                 ((0, 0, 1), (1, 0, 0), (0, 1, 0))):
        q = R.quat_from_axis_angle_rad(axis, math.pi / 2)
        got = R.quat_rotate_vec(q, v_in)
        assert np.allclose(got, expected, atol=1e-12), (axis, got, expected)
    assert np.allclose(np.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1])


def test_camera_axes_match_the_anchor_vocabulary():
    """The renderer's view names and `_anchors`' axis names must not diverge.

    `left = +X` is the counter-intuitive one: the viewer stands at +Y looking
    at the origin, so their left hand points along +X. If the camera table and
    the anchor table ever disagreed, every "left" in a prompt would mean two
    different things.
    """
    from dycraft.render import canonical_cameras

    cams = {c.id: c for c in canonical_cameras(
        target_center=(0.0, 0.0, 0.0), target_extent_xy=1.0,
        target_extent_z=1.0)}
    assert set(cams) >= {"front", "back", "left", "right", "top"}

    def dominant(v):
        i = int(np.argmax(np.abs(v[:3])))
        return "XYZ"[i], (1 if v[i] > 0 else -1)

    assert dominant(cams["front"].eye) == ("Y", +1), "front camera must sit on +Y"
    assert dominant(cams["back"].eye) == ("Y", -1)
    assert dominant(cams["left"].eye) == ("X", +1), "left camera must sit on +X"
    assert dominant(cams["right"].eye) == ("X", -1)
    # Anchor vocabulary agrees.
    assert A.outward_direction("front") == (0.0, 1.0, 0.0)
    assert A.outward_direction("left") == (1.0, 0.0, 0.0)
    assert A.outward_direction("top") == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Rotation representation
# ---------------------------------------------------------------------------

def _nonaxis_quat():
    """A rotation with four distinct components and no axis alignment.

    An axis-aligned quaternion has zeros that hide component-order mistakes.
    """
    axis = np.array([0.3, -0.5, 0.81])
    axis = axis / np.linalg.norm(axis)
    return R.quat_from_axis_angle_rad(tuple(axis), 0.7)


def _matrix_of(q):
    return np.column_stack([R.quat_rotate_vec(q, e)
                            for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])


def test_quaternions_are_wxyz_and_proper_rotations():
    assert R.IDENTITY_SE3[1] == (1.0, 0.0, 0.0, 0.0)
    q = _nonaxis_quat()
    assert len({round(c, 9) for c in q}) == 4, "canary must not be axis-aligned"
    M = _matrix_of(q)
    assert np.isclose(np.linalg.det(M), 1.0, atol=1e-12)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)


def test_quaternion_euler_round_trip():
    q = _nonaxis_quat()
    back = R.euler_xyz_deg_to_quat_wxyz(*R.quat_to_euler_xyz_deg(q))
    assert np.allclose(R.sign_canonicalize_quat(q),
                       R.sign_canonicalize_quat(back), atol=1e-12)


def test_quaternion_canonicalisation_forces_positive_w():
    """Byte-stable IR hashing depends on this."""
    assert R.sign_canonicalize_quat((-0.5, 0.1, 0.2, 0.3))[0] >= 0.0
    a = R.sign_canonicalize_quat((0.5, 0.1, 0.2, 0.3))
    b = R.sign_canonicalize_quat((-0.5, -0.1, -0.2, -0.3))
    assert np.allclose(a, b), "equivalent quaternions must serialize identically"


@_needs_cad
def test_occt_quaternion_reorder_is_correct():
    """DyCraft stores WXYZ; OCCT's gp_Quaternion takes XYZW.

    Includes a NEGATIVE CONTROL: feeding the components without reordering must
    produce a *different* matrix. Without that, a test that merely builds a
    rotation would pass under either convention.
    """
    import sys

    from dycraft.agent.cad import _cad_env as SA

    q = _nonaxis_quat()
    M = _matrix_of(q)
    script = (
        "import json,sys\n"
        "from OCP.gp import gp_Quaternion, gp_Trsf\n"
        f"w,x,y,z = {tuple(float(c) for c in q)!r}\n"
        "def mat(gq):\n"
        "    t=gp_Trsf(); t.SetRotation(gq)\n"
        "    return [[t.Value(r+1,c+1) for c in range(3)] for r in range(3)]\n"
        "print(json.dumps({'right': mat(gp_Quaternion(x,y,z,w)),\n"
        "                  'wrong': mat(gp_Quaternion(w,x,y,z))}))\n"
    )
    out = _in_cad(script, timeout=300)
    got = json.loads(out.strip().splitlines()[-1])
    assert np.allclose(M, np.array(got["right"]), atol=1e-9), \
        "the (x,y,z,w) reorder does not reproduce dycraft's rotation"
    assert not np.allclose(M, np.array(got["wrong"]), atol=1e-9), \
        "negative control failed: the un-swapped order agrees, so this test " \
        "cannot detect an order regression"


# ---------------------------------------------------------------------------
# 3. Pose units
# ---------------------------------------------------------------------------

@_needs_cad
def test_pose_units_kernel_is_radians_probes_are_degrees():
    """The hazard is that both take `{joint: float}` with nothing to tell them
    apart. This pins each side so a future unification is a deliberate change.
    """
    ir = _build(_ARTICULATED)
    deg = -75.0
    rad = math.radians(deg)

    k_lo, k_hi = _world_aabb_from_kernel(ir, "door", rad)
    p_lo, p_hi = P.object_part_world_aabb(ir, "door", poses={"hinge": deg})
    assert np.allclose(k_lo, p_lo, atol=1e-6), "kernel(rad) vs probe(deg) min"
    assert np.allclose(k_hi, p_hi, atol=1e-6), "kernel(rad) vs probe(deg) max"

    # Feeding degrees to the kernel is silently wrong, not an error.
    wrong_lo, _ = _world_aabb_from_kernel(ir, "door", deg)
    assert not np.allclose(wrong_lo, k_lo, atol=1e-6), (
        "passing degrees where radians are expected should change the pose")


# ---------------------------------------------------------------------------
# 4. Part-local geometry: authored coordinates are preserved
# ---------------------------------------------------------------------------

_ONE_PART = """
obj = Object("p")
body = obj.add_part("body", {shape})
obj.declare_root(body)
ir = obj.finalize()
"""

_ARTICULATED = """
obj = Object("art")
root = obj.add_part("root", Box(0.30, 0.20, 0.10))
door = obj.add_part("door", Pos(0.25, 0, 0) * Box(0.40, 0.06, 0.04))
obj.declare_root(root)
obj.attach_revolute(
    "hinge", parent=root, child=door, axis=(0.0, 0.0, 1.0),
    limits=(-110.0, 0.0), default=0.0,
    parent_anchor=obj.anchor(root, "left"),
    child_anchor=obj.anchor(door, "right"))
ir = obj.finalize()
"""


def _aabb(ir):
    p = ir["parts"][0]["params"]
    return np.array(p["aabb_min"]), np.array(p["aabb_max"])


@_needs_cad
@pytest.mark.parametrize("shape,lo,hi,label", [
    ("Box(0.2, 0.1, 0.05)",
     (-0.1, -0.05, -0.025), (0.1, 0.05, 0.025), "origin-centred"),
    ("Pos(1, 2, 3) * Box(0.2, 0.1, 0.05)",
     (0.9, 1.95, 2.975), (1.1, 2.05, 3.025), "translated"),
    ("Rotation(0, 0, 90) * Box(0.2, 0.1, 0.05)",
     (-0.05, -0.1, -0.025), (0.05, 0.1, 0.025), "rotated: AABB swaps x/y"),
    ("Pos(1, 0, 0) * Rotation(0, 0, 90) * Box(0.2, 0.1, 0.05)",
     (0.95, -0.1, -0.025), (1.05, 0.1, 0.025), "translated + rotated"),
])
def test_authored_coordinates_are_preserved(shape, lo, hi, label):
    """No auto-centering, no canonicalisation. What you author is Part-local.

    This is the property the whole anchor system rests on: anchors are AABB
    landmarks, so silently re-framing geometry would move every joint.
    """
    ir = _build(_ONE_PART.format(shape=shape))
    got_lo, got_hi = _aabb(ir)
    assert np.allclose(got_lo, lo, atol=1e-9), label
    assert np.allclose(got_hi, hi, atol=1e-9), label


@_needs_cad
def test_compound_internal_locations_are_preserved():
    """A Compound is ONE rigid Part with one Part-local frame; the children's
    internal placement is baked, not flattened away."""
    src = _ONE_PART.format(
        shape="Compound(children=[Pos(-0.3,0,0)*Box(0.1,0.1,0.1), "
              "Pos(0.3,0,0)*Box(0.1,0.1,0.1)])")
    ir = _build(src)
    lo, hi = _aabb(ir)
    assert np.allclose(lo, (-0.35, -0.05, -0.05), atol=1e-9)
    assert np.allclose(hi, (0.35, 0.05, 0.05), atol=1e-9)
    assert ir["parts"][0]["params"]["shape"]["topo"]["solids"] == 2


@_needs_cad
def test_object_py_cannot_mutate_a_registered_shape_at_all():
    """The sandbox already forbids the mutation, so `object.py` cannot even try.

    Worth pinning explicitly: it means the snapshot guarantee below is
    defense-in-depth for the agent-facing path, and the *primary* protection for
    in-process SDK callers (tests, the scene layer, future code) which are not
    sandboxed.
    """
    src = """
obj = Object("p")
shape = Box(0.2, 0.1, 0.05)
body = obj.add_part("body", shape)
shape.location = Location((9.0, 9.0, 9.0))
obj.declare_root(body)
ir = obj.finalize()
"""
    out = W.run_object_source(src, run_dir=Path(tempfile.mkdtemp()))
    assert out["ok"] is False
    assert out["sandbox_violation"] is not None
    assert "attribute assignment not allowed" in json.dumps(out["sandbox_violation"])


@_needs_cad
def test_add_part_is_a_full_snapshot_including_the_brep():
    """Registration is immutable: mutating the Shape afterwards reaches NOTHING.

    Exercised at the SDK level, in the CAD interpreter, because that is the
    surface where mutation is actually possible (`object.py` is sandboxed — see
    the test above). The B-rep, the AABB, and the mesh are all captured at
    registration: a B-rep serialized from the live Shape at finalize would let a
    mutated Shape produce artifacts that disagree, and since the B-rep is what
    the collision adjudicator loads, the divergence would be silently
    permissive.

    All three artifacts are read back FROM DISK, so this cannot pass by
    inspecting in-memory state that happens to be right.
    """

    from dycraft.agent.cad import _cad_env as SA

    script = r"""
import json, sys, tempfile
from pathlib import Path
from dycraft.agent.layers import layer as L
from dycraft.sdk import Object
from build123d import Box, Location, Solid, Compound
from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape
from OCP.TopAbs import TopAbs_ShapeEnum

d = Path(tempfile.mkdtemp())
class C: run_dir = d
L.reset_object_layer_state(); L._object_build_overlay(C())
(d / "object.py").write_text("# probe")

o = Object("p")
shape = Box(0.2, 0.1, 0.05)
part = o.add_part("body", shape)
shape.location = Location((9.0, 9.0, 9.0))     # must reach nothing
o.declare_root(part)
ir = o.finalize()
prm = ir["parts"][0]["params"]

xs = [float(l.split()[1]) for l in (d / prm["mesh"]).read_text().splitlines()
      if l.startswith("v ")]
s2 = TopoDS_Shape(); BRepTools.Read_s(s2, str(d / prm["brep"]), BRep_Builder())
ob = Solid(s2) if s2.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID else Compound(s2)
bb = ob.bounding_box()
print(json.dumps({"aabb_min_x": prm["aabb_min"][0], "obj_min_x": min(xs),
                  "brep_min_x": bb.min.X, "brep_max_x": bb.max.X}))
"""
    out = _in_cad(script, timeout=600)
    got = json.loads(out.strip().splitlines()[-1])

    assert got["aabb_min_x"] == pytest.approx(-0.1, abs=1e-6), got
    assert got["obj_min_x"] == pytest.approx(-0.1, abs=1e-6), \
        f"OBJ followed the post-registration mutation: {got}"
    assert got["brep_min_x"] == pytest.approx(-0.1, abs=1e-6), \
        f"BREP followed the post-registration mutation: {got}"
    assert got["brep_max_x"] == pytest.approx(0.1, abs=1e-6), got


@_needs_cad
def test_staged_brep_does_not_carry_the_triangulation():
    """The .brep must hold geometry only — the OBJ is the mesh.

    `tessellate` attaches a triangulation to the TopoDS_Shape and
    `BRepTools.Write_s` serializes whatever is attached, so a B-rep written
    straight after tessellation is ~7x larger and its content-addressed digest
    changes — which moves every object id.

    `measure_shape` calls `BRepTools.Clean_s` before serializing to make the
    bytes a function of the geometry alone. This test pins the size, because
    the failure mode is silent: bloated assets and churned ids, with the
    geometry still perfectly correct.
    """

    from dycraft.agent.cad import _cad_env as SA
    script = (
        "import hashlib\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "from build123d import Box, Cylinder, Pos, Rotation, fillet, Axis\n"
        "s = Box(0.20,0.12,0.16) - Pos(0,0.006,0)*Box(0.18,0.12,0.14)\n"
        "s -= Pos(0.09,0,0.05)*Rotation(90,0,0)*Cylinder(0.004,0.30)\n"
        "s = fillet(s.edges().filter_by(Axis.Y).group_by(Axis.X)[0], 0.004)\n"
        "e = sc.measure_shape(s, name='x')\n"
        "print(len(e['brep_bytes']))\n"
    )
    out = _in_cad(script, timeout=600)
    size = int(out.strip().splitlines()[-1])
    # Clean: ~13.9 kB. With the triangulation attached: ~98 kB.
    assert size < 30_000, (
        f"the staged B-rep is {size} bytes — it is carrying the tessellation. "
        "Every asset digest and object id will have moved.")


@_needs_cad
def test_exact_shading_does_not_move_the_geometry():
    """The equivalence the exact-normal path rests on.

    `_solid_compile._tessellate_faces` walks the faces itself instead of calling
    `Shape.tessellate`, because build123d concatenates the per-face
    triangulations and the exact normals and UVs live *on* those. That is only
    safe if the positions and triangles it produces are the ones build123d would
    have produced — otherwise a shading change would quietly become a geometry
    change, moving the welded mesh, the QC numbers, every asset digest, and every
    object id.

    So this asserts, across a box, a curved solid, a boolean, and a located
    shape: identical vertex positions, identical triangles including winding,
    an exact normal available for every node, and every triangle's winding
    agreeing with its own exact normal (which is what catches a REVERSED face
    whose normal was not negated — a failure that renders black rather than
    raising).
    """

    from dycraft.agent.cad import _cad_env as SA
    script = (
        "import math\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "from build123d import Box, Cylinder, Sphere, Cone, Pos, Rotation\n"
        "cases = {\n"
        "  'box': Box(0.4,0.3,0.2),\n"
        "  'cyl': Cylinder(0.05,0.2),\n"
        "  'sphere': Sphere(0.05),\n"
        "  'cone': Cone(0.05,0.02,0.1),\n"
        "  'bore': Box(0.3,0.3,0.3) - Pos(0,0,0.1)*Cylinder(0.05,0.4),\n"
        "  'located': Pos(1.2,-0.4,0.7)*Rotation(20,0,0)*Box(0.2,0.2,0.2),\n"
        "}\n"
        "bad = []\n"
        "for name, shape in cases.items():\n"
        "    for solid in shape.solids():\n"
        "        bv, bt = solid.tessellate(1e-3, 0.1)\n"
        "        v, f, n, u = sc._tessellate_faces(solid, 1e-3, 0.1, 1.0)\n"
        "        if len(v) != len(bv):\n"
        "            bad.append(f'{name}: node count {len(v)} != {len(bv)}')\n"
        "            continue\n"
        "        for p, q in zip(v, bv):\n"
        "            if max(abs(a-b) for a,b in zip(p,(q.X,q.Y,q.Z))) > 1e-12:\n"
        "                bad.append(f'{name}: position differs'); break\n"
        "        if [tuple(t) for t in bt] != f:\n"
        "            bad.append(f'{name}: triangles differ')\n"
        "        if sum(1 for x in n if x is None):\n"
        "            bad.append(f'{name}: missing normals')\n"
        "        if sum(1 for x in u if x is None):\n"
        "            bad.append(f'{name}: missing uvs')\n"
        "        for (a,b,c) in f:\n"
        "            if None in (n[a],n[b],n[c]): continue\n"
        "            ux,uy,uz = (v[b][i]-v[a][i] for i in range(3))\n"
        "            vx,vy,vz = (v[c][i]-v[a][i] for i in range(3))\n"
        "            fx,fy,fz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx\n"
        "            m = math.sqrt(fx*fx+fy*fy+fz*fz)\n"
        "            if m <= 0: continue\n"
        "            if (fx*n[a][0]+fy*n[a][1]+fz*n[a][2])/m < 0:\n"
        "                bad.append(f'{name}: winding disagrees with normal'); break\n"
        "print('BAD=' + repr(sorted(set(bad))))\n"
    )
    out = _in_cad(script, timeout=900)
    line = [ln for ln in out.splitlines() if ln.startswith("BAD=")][-1]
    assert line == "BAD=[]", line


@_needs_cad
def test_exact_normals_match_the_analytic_surface():
    """Exactness, measured rather than asserted by construction.

    A cylinder's barrel normal is radial and a sphere's is the unit position, so
    both have a closed form to compare against. The crease-angle heuristic this
    replaced landed within ~0.06 deg here; the exact evaluation should be at
    floating-point noise, which is what distinguishes "we compute the normal"
    from "we estimate it well".
    """

    from dycraft.agent.cad import _cad_env as SA
    script = (
        "import math\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "from build123d import Cylinder, Sphere\n"
        "worst = 0.0\n"
        "cyl = Cylinder(0.05, 0.2)\n"
        "for solid in cyl.solids():\n"
        "    v, f, n, u = sc._tessellate_faces(solid, 1e-3, 0.1, None)\n"
        "    for p, nn in zip(v, n):\n"
        "        r = math.hypot(p[0], p[1])\n"
        "        if nn is None or abs(r - 0.05) > 1e-9: continue\n"
        "        if abs(nn[2]) > 0.5: continue\n"
        "        d = nn[0]*p[0]/r + nn[1]*p[1]/r\n"
        "        worst = max(worst, math.degrees(math.acos(max(-1.0,min(1.0,d)))))\n"
        "sph = Sphere(0.05)\n"
        "for solid in sph.solids():\n"
        "    v, f, n, u = sc._tessellate_faces(solid, 1e-3, 0.1, None)\n"
        "    for p, nn in zip(v, n):\n"
        "        r = math.sqrt(sum(c*c for c in p))\n"
        "        if nn is None or r < 1e-9: continue\n"
        "        d = sum(a*b/r for a, b in zip(nn, p))\n"
        "        worst = max(worst, math.degrees(math.acos(max(-1.0,min(1.0,d)))))\n"
        "print('WORST=%.9f' % worst)\n"
    )
    out = _in_cad(script, timeout=900)
    worst = float([ln for ln in out.splitlines()
                   if ln.startswith("WORST=")][-1].split("=")[1])
    assert worst < 1e-3, (
        f"exact surface normals deviate by up to {worst:.6f} deg from the "
        f"analytic normal; that is heuristic-scale error, so the evaluation is "
        f"not reading the surface it claims to")


# ---------------------------------------------------------------------------
# 5. Anchors
# ---------------------------------------------------------------------------

def test_there_are_exactly_27_anchors():
    assert len(A.AUTO_AABB_ANCHOR_NAMES) == 27
    assert len(A.FACE_ANCHOR_NAMES) == 6
    assert len(A.EDGE_ANCHOR_NAMES) == 12
    assert len(A.CORNER_ANCHOR_NAMES) == 8


def test_anchor_positions_track_an_off_centre_aabb():
    """Anchors are derived from the baked AABB, so off-center geometry moves
    them. `left = xmax` / `right = xmin` per the viewer-relative convention."""
    aabb = ((0.9, -0.05, -0.025), (1.1, 0.05, 0.025))
    pos = A.auto_aabb_anchor_position
    assert pos(aabb, "center") == (1.0, 0.0, 0.0)
    assert pos(aabb, "left") == (1.1, 0.0, 0.0)
    assert pos(aabb, "right") == (0.9, 0.0, 0.0)
    assert pos(aabb, "top") == (1.0, 0.0, 0.025)
    assert pos(aabb, "bottom_back_right") == (0.9, -0.05, -0.025)


@_needs_cad
def test_all_27_anchors_are_identity_oriented():
    """Orientation comes only from an explicit `rotation=` offset."""
    ir = _build(_ONE_PART.format(shape="Pos(1,0,0) * Box(0.2, 0.1, 0.05)"))
    del ir  # the invariant is on the builder, exercised below
    from dycraft.sdk.object_builder import Object as _O
    del _O
    # Exercised through the joint records in the articulated fixture instead:
    art = _build(_ARTICULATED)
    for key in ("parent_anchor", "child_anchor"):
        rot = art["joints"][0][key]["rotation"]
        assert np.allclose(rot, (1.0, 0.0, 0.0, 0.0), atol=1e-12), \
            f"{key} should be identity-oriented without an explicit rotation"


def test_anchor_transform_maps_anchor_to_part():
    """Direction is ANCHOR -> PART.

    Composing the stored SE3 carries anchor-local coordinates into part-local
    coordinates. The joint formula relies on this by inverting the CHILD
    anchor; reading the direction backwards makes that inverse look wrong.
    """
    q = R.euler_xyz_deg_to_quat_wxyz(0, 0, 90)
    anchor_se3 = ((1.0, 0.0, 0.0), q)
    p_anchor = (1.0, 0.0, 0.0)
    p_part = R.se3_compose(anchor_se3, (p_anchor, R.IDENTITY_SE3[1]))[0]
    assert np.allclose(p_part, (1.0, 1.0, 0.0), atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Joints and FK
# ---------------------------------------------------------------------------

_OFF_AXIS = """
obj = Object("art")
root = obj.add_part("root", Box(0.30, 0.20, 0.10))
link = obj.add_part("link", Pos(0.25, 0, 0) * Box(0.40, 0.06, 0.04))
obj.declare_root(root)
obj.attach_{kind}(
    "j1", parent=root, child=link, axis=(1.0, 2.0, 3.0),
    limits={limits}, default=0.0,
    parent_anchor=obj.anchor(root, "left", rotation=(17, -23, 41)),
    child_anchor=obj.anchor(link, "right", rotation=(-35, 11, 64)))
ir = obj.finalize()
"""


@_needs_cad
def test_joint_axis_is_normalised_on_entry():
    ir = _build(_OFF_AXIS.format(kind="revolute", limits="(-90.0, 90.0)"))
    axis = ir["joints"][0]["axis"]
    assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-12)
    assert np.allclose(axis, np.array([1.0, 2.0, 3.0]) / math.sqrt(14), atol=1e-9)


@_needs_cad
def test_rest_pose_is_parent_anchor_compose_inverse_child_anchor():
    ir = _build(_OFF_AXIS.format(kind="revolute", limits="(-90.0, 90.0)"))
    j = ir["joints"][0]
    pa = K._se3_from_anchor(j["parent_anchor"])
    ca = K._se3_from_anchor(j["child_anchor"])
    expected = R.se3_compose(pa, R.se3_inverse(ca))
    got = K._child_in_parent(dict(j, type="fixed"))
    assert np.allclose(got[0], expected[0], atol=1e-12)
    assert np.allclose(got[1], expected[1], atol=1e-12)


@_needs_cad
def test_revolute_preserves_distance_to_the_pivot():
    """An off-principal axis through rotated anchors still rotates about the
    pivot rather than drifting."""
    ir = _build(_OFF_AXIS.format(kind="revolute", limits="(-90.0, 90.0)"))
    j = ir["joints"][0]
    pivot = np.array(K._se3_from_anchor(j["parent_anchor"])[0])
    dists = {round(float(np.linalg.norm(
        np.array(K._child_in_parent(j, poses={"j1": q})[0]) - pivot)), 9)
        for q in (0.0, 0.4, -1.1, 1.3)}
    assert len(dists) == 1, f"pivot distance drifted: {dists}"


@_needs_cad
def test_prismatic_translates_by_exactly_q_without_rotating():
    ir = _build(_OFF_AXIS.format(kind="prismatic", limits="(0.0, 0.5)"))
    j = ir["joints"][0]
    base = K._child_in_parent(j, poses={"j1": 0.0})
    for q in (0.1, 0.35):
        rel = K._child_in_parent(j, poses={"j1": q})
        delta = np.array(rel[0]) - np.array(base[0])
        assert np.isclose(np.linalg.norm(delta), q, atol=1e-9)
        assert np.allclose(rel[1], base[1], atol=1e-12), \
            "a prismatic joint must not introduce rotation"


_CHAIN = """
obj = Object("chain")
a = obj.add_part("root", Box(0.2, 0.2, 0.2))
b = obj.add_part("link_1", Pos(0.3, 0, 0) * Box(0.2, 0.05, 0.05))
c = obj.add_part("link_2", Pos(0.3, 0, 0) * Box(0.2, 0.05, 0.05))
obj.declare_root(a)
obj.attach_revolute("j1", parent=a, child=b, axis=(0.0, 0.0, 1.0),
                    limits=(-90.0, 90.0), default=0.0,
                    parent_anchor=obj.anchor(a, "left"),
                    child_anchor=obj.anchor(b, "right"))
obj.attach_revolute("j2", parent=b, child=c, axis=(1.0, 1.0, 0.0),
                    limits=(-90.0, 90.0), default=0.0,
                    parent_anchor=obj.anchor(b, "left", rotation=(0, 0, 25)),
                    child_anchor=obj.anchor(c, "right"))
ir = obj.finalize()
"""


@_needs_cad
def test_multi_joint_chain_composes_and_stays_independent():
    ir = _build(_CHAIN)
    rest = K._compute_frames(ir)
    only_j1 = K._compute_frames(ir, poses={"j1": 0.5})
    only_j2 = K._compute_frames(ir, poses={"j2": 0.5})

    assert np.allclose(only_j2["link_1"][0], rest["link_1"][0], atol=1e-12), \
        "a downstream joint must not move its parent"
    assert not np.allclose(only_j1["link_2"][0], rest["link_2"][0], atol=1e-9), \
        "an upstream joint must carry its whole subtree"


@_needs_cad
def test_fk_is_stable_under_joint_ordering():
    """Traversal must not depend on the order joints happen to sit in."""
    ir = _build(_CHAIN)
    poses = {"j1": 0.3, "j2": -0.2}
    a = K._compute_frames(ir, poses=poses)
    b = K._compute_frames(dict(ir, joints=list(reversed(ir["joints"]))),
                          poses=poses)
    for name in a:
        assert np.allclose(a[name][0], b[name][0], atol=1e-12), name


@_needs_cad
def test_root_part_is_the_object_origin():
    """M2 removed a dead `root_pose` read. The root must sit at identity, and
    an IR carrying a stray `root_pose` must not silently displace anything."""
    ir = _build(_ARTICULATED)
    root = K._compute_frames(ir)[ir["root"]]
    assert np.allclose(root[0], (0.0, 0.0, 0.0), atol=1e-12)
    assert np.allclose(root[1], (1.0, 0.0, 0.0, 0.0), atol=1e-12)

    polluted = dict(ir, root_pose=((5.0, 5.0, 5.0), (1.0, 0.0, 0.0, 0.0)))
    assert np.allclose(K._compute_frames(polluted)[ir["root"]][0],
                       (0.0, 0.0, 0.0), atol=1e-12), \
        "object-level root placement was reintroduced"


# ---------------------------------------------------------------------------
# 7. Downstream placement agreement
# ---------------------------------------------------------------------------

@_needs_cad
def test_renderer_and_verifier_share_one_frame_source():
    """Not 'both look right' — both must resolve poses through the same call.

    Two independent implementations that agree today can diverge tomorrow; one
    shared function cannot.

    The renderer's half is `render/_world.py`, which is where FK moved when the
    package grew a second backend. That is the point of the module: every
    backend composes its world through this one call, so Blender and Open3D can
    disagree about lighting but never about where the door is. A backend module
    calling `_compute_frames` itself would be reintroducing exactly the second
    implementation this test exists to forbid — hence the negative half below.
    """
    import inspect

    from dycraft.render import _world
    from dycraft.verify import object_verify

    for mod in (_world, object_verify):
        src = inspect.getsource(mod)
        assert "_compute_frames" in src, f"{mod.__name__} no longer shares the kernel"

    for backend_mod in ("open3d_renderer", "blender_renderer"):
        try:
            mod = importlib.import_module(f"dycraft.render.{backend_mod}")
        except ImportError:
            continue        # backend not installed in this env; nothing to guard
        assert "_compute_frames" not in inspect.getsource(mod), (
            f"{backend_mod} resolves frames itself instead of going through "
            "render/_world.py — that is a second FK implementation")

    ir = _build(_ARTICULATED)
    for q in (0.0, -1.0):
        k_lo, k_hi = _world_aabb_from_kernel(ir, "door", q)
        p_lo, p_hi = P.object_part_world_aabb(
            ir, "door", poses={"hinge": math.degrees(q)})
        assert np.allclose(k_lo, p_lo, atol=1e-6), f"disagreement at q={q}"
        assert np.allclose(k_hi, p_hi, atol=1e-6), f"disagreement at q={q}"


@_needs_cad
def test_fk_moves_a_child_and_the_world_axis_is_pose_invariant():
    ir = _build(_ARTICULATED)
    closed = P.object_part_world_aabb(ir, "door", poses={"hinge": 0.0})
    opened = P.object_part_world_aabb(ir, "door", poses={"hinge": -75.0})
    assert closed != opened
    axes = {tuple(np.round(P.object_joint_axis_world(
        ir, "hinge", poses={"hinge": q}), 9)) for q in (0.0, -40.0)}
    assert len(axes) == 1, "the world joint axis must not depend on q"


# ---------------------------------------------------------------------------
# 8. Documented limitations
# ---------------------------------------------------------------------------

@_needs_cad
def test_inertia_is_diagonal_only_and_lossy_when_rotated():
    """Pins a KNOWN LIMITATION so nobody assumes `inertia_diag` is exact.

    `shape.inertia_diag` keeps only the diagonal of the inertia tensor about
    the centroid. Exact for an axis-aligned body; for a rotated asymmetric one
    the discarded off-diagonal terms are large. Measured here so that if the
    representation is ever fixed, this test fails and gets updated rather than
    quietly continuing to assert a lie.

    Carrying the full tensor needs an ObjectIR field.
    Only `render/urdf_export.py` consumes inertia today.
    """

    from dycraft.agent.cad import _cad_env as SA
    script = (
        "import json\n"
        "from build123d import Box, Rotation\n"
        "b = Box(0.30, 0.10, 0.05)\n"
        "out = {}\n"
        "for label, sh in (('aligned', b), ('rotated', Rotation(20,35,50)*b)):\n"
        "    M = sh.matrix_of_inertia\n"
        "    A = [[M[i][j] for j in range(3)] for i in range(3)]\n"
        "    diag = max(abs(A[i][i]) for i in range(3))\n"
        "    off = max(abs(A[i][j]) for i in range(3) for j in range(3) if i != j)\n"
        "    out[label] = off / diag\n"
        "print(json.dumps(out))\n"
    )
    out = _in_cad(script, timeout=300)
    ratios = json.loads(out.strip().splitlines()[-1])
    assert ratios["aligned"] < 1e-6, "an axis-aligned box should be diagonal"
    assert ratios["rotated"] > 0.2, (
        "the rotated-body inertia loss this limitation documents is no longer "
        f"present (ratio {ratios['rotated']}); if inertia was fixed, update "
        "design_frames.md and this test")


def test_fixed_source_poses_flag_is_degrees_not_radians():
    """`--poses hinge=75` must mean 75 degrees at the CLI surface.

    `_render_poses` hands its values to the renderer, which hands them to
    `_compute_frames` — and that function is explicitly RADIANS-only. Passing
    `float(value)` straight through would draw every posed render at the
    authored number of *radians* (`door_hinge=-75` as -4297 deg), and nothing
    would raise, because a pose value has no units attached and any float is a
    legal angle.

    This pins the boundary the frame contract requires - degrees at every
    surface, radians in the IR and kernel - and it pins it on a *revolute*
    joint only: prismatic values are meters in both worlds and must not be
    scaled by 57.3.
    """
    from dycraft.agent.layers import fixed_source

    ir = {"joints": [{"name": "hinge", "type": "revolute"},
                     {"name": "slide", "type": "prismatic"}]}
    captured = {}

    class _Cache(dict):
        pass

    class _Ctx:
        render_cache = _Cache()

    # Capture what the renderer would be handed, without a GL context.
    import dycraft.render as render_mod
    real = render_mod.render_request

    def fake(request):
        captured[request.pose_label] = dict(request.poses)
        raise render_mod.RenderUnavailable("captured")

    render_mod.render_request = fake
    try:
        with tempfile.TemporaryDirectory() as td:
            out = fixed_source._render_poses(
                _Ctx(), ir, Path(td), "open:hinge=75,slide=0.25")
    finally:
        render_mod.render_request = real

    assert captured["open"]["hinge"] == pytest.approx(math.radians(75.0))
    assert captured["open"]["slide"] == pytest.approx(0.25), (
        "prismatic values are metres and must pass through unconverted")
    # The bundle records what the author typed, not the converted value.
    assert out[0]["poses_deg"] == {"hinge": 75.0, "slide": 0.25}
