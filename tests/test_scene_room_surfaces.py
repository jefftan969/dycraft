"""The room shell as authored content: materials, ceiling, replacement.

Three properties keep a generated interior from reading flat beside the objects
standing in it:

  * A room surface can carry a `shader`: the scene sandbox's overlay binds it,
    `_world._material_to_render` packs one for the room, `_box_spec` projects
    UVs for a shell surface and `_build_material` execs it.
  * There is a ceiling: `room_wall_boxes` draws all six surfaces and hides the
    camera-side wall rather than deleting it, so an interior is lit with all of
    its bounce surfaces.
  * An authored object can replace the shell's version of a wall, so a wall
    with a window cut through it is not sealed by the shell's own slab.

No GPU: everything here is the parent-side half, which is pure.
"""

from __future__ import annotations

import json
import tempfile

import pytest

np = pytest.importorskip("numpy")

from pathlib import Path

from dycraft.render import CameraSpec, RenderRequest
from dycraft.render._world import _build_world_geometry, room_wall_boxes
from dycraft.sdk import Material, Object, ObjectHandle, Scene
from dycraft.sdk import _mesh_assets
from dycraft.sdk._errors import InvalidRoom
from dycraft.sdk._shader import shader
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import (ROOM_SURFACES,
                                       _reset_scene_singleton,
                                       room_surfaces_suppressed_by)

_SHADER_SRC = "@shader\ndef surf(nodes, links, bsdf):\n    pass\n"


@pytest.fixture(autouse=True)
def _reset():
    _reset_object_singleton()
    _reset_scene_singleton()
    _mesh_assets.set_active_mesh_session(
        _mesh_assets.MeshAssetSession(tempfile.mkdtemp(),
                                      source_text=_SHADER_SRC))
    yield
    _mesh_assets.clear_active_mesh_session()
    _reset_object_singleton()
    _reset_scene_singleton()


def _box_object(name="filler", size=(0.5, 0.5, 0.5)):
    _reset_object_singleton()
    o = Object(name)
    o.declare_root(o.make_box("body", *size))
    return o.finalize()


def _room_info(ir, catalog):
    req = RenderRequest(layer="scene", ir=ir, catalog=catalog, target="all",
                        out_dir=Path(tempfile.mkdtemp()), step=0, cache={})
    return _build_world_geometry(req)[1]


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def test_a_room_surface_can_carry_a_procedural_shader():
    """The one-line capability gap: `shader` reaching the scene sandbox.

    Verified all the way to the packed renderer material, because that is the
    boundary the Cycles child reads — the SDK accepting it was never the
    question.
    """
    @shader
    def surf(nodes, links, bsdf):
        pass

    box = _box_object()
    _reset_scene_singleton()
    s = Scene("shaded")
    s.set_room(4.0, 4.4, 2.6, floor_material=Material(shader=surf))
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
    ir = s.finalize()

    assert "shader" in ir["room"]["floor_material"]
    packed = _room_info(ir, {box["id"]: box})
    assert "shader" in packed["floor"]


def test_the_scene_sandbox_binds_the_whole_material_vocabulary():
    """The overlay IS the vocabulary — a name absent from it is unreachable."""
    from dycraft.agent.layers.layer import _scene_build_overlay

    class _Ctx:
        run_dir = None
        layer_payload = type("P", (), {"object_handles": {},
                                       "asset_base_dir": None})()

    overlay = _scene_build_overlay(_Ctx())
    for name in ("shader", "Material", "Emission"):
        assert name in overlay, f"{name} unreachable from a scene source"
    # `Environment` is deliberately NOT bound: it is a photometric IR reader with
    # no `Scene` method to take it, so an author could construct one and have
    # nowhere to pass it. Pinned so it does not drift back in.
    assert "Environment" not in overlay, (
        "Environment is exposed again but still has no consumer")


# ---------------------------------------------------------------------------
# The ceiling, and existence vs visibility
# ---------------------------------------------------------------------------

def test_the_shell_has_six_surfaces_and_a_ceiling_material():
    box = _box_object()
    _reset_scene_singleton()
    s = Scene("six")
    s.set_room(4.0, 4.4, 2.6,
               ceiling_material=Material(color=(250, 250, 250), roughness=0.9))
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
    ir = s.finalize()

    info = _room_info(ir, {box["id"]: box})
    assert info["ceiling"] is not None
    cam = CameraSpec(id="front", eye=(0, 9, 1.5), target=(0, 0, 1.0),
                     up=(0, 0, 1), fov_deg=45)
    assert [b.tag for b in room_wall_boxes(cam, info)] == list(ROOM_SURFACES)


def test_a_ceiling_defaults_to_the_wall_material():
    """A room authored before ceilings existed still gets one."""
    box = _box_object()
    _reset_scene_singleton()
    s = Scene("nofcm")
    s.set_room(4.0, 4.0, 2.6, wall_material=Material(color=(200, 190, 180)))
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
    info = _room_info(s.finalize(), {box["id"]: box})
    assert info["ceiling"] is None          # nothing authored
    cam = CameraSpec(id="front", eye=(0, 9, 1.5), target=(0, 0, 1.0),
                     up=(0, 0, 1), fov_deg=45)
    by_tag = {b.tag: b for b in room_wall_boxes(cam, info)}
    assert by_tag["ceiling"].material == info["wall"]


def test_hidden_surfaces_still_exist_and_replaced_ones_do_not():
    """The distinction the whole change turns on.

    A camera-invisible surface bounces, shadows and contributes to every
    indirect path — it just is not drawn. A replaced one is gone: leaving it
    would put an invisible slab behind the authored wall, shadowing the room
    and sealing whatever opening the object cut.
    """
    box = _box_object()
    wall = _box_object("wall", (4.0, 0.12, 2.6))

    _reset_scene_singleton()
    s = Scene("mixed")
    s.set_room(4.0, 4.4, 2.6)
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
    w = s.place(ObjectHandle(wall), name="back_wall",
                pose=((0.0, -2.14, 0.0), (0, 0, 0)))
    s.replace_room_surface("back", w)
    ir = s.finalize()

    info = _room_info(ir, {box["id"]: box, wall["id"]: wall})
    cam = CameraSpec(id="front", eye=(0, 9, 1.5), target=(0, 0, 1.0),
                     up=(0, 0, 1), fov_deg=45)
    tags = {b.tag: b for b in room_wall_boxes(cam, info)}

    # Replaced: absent entirely.
    assert "wall_ymin" not in tags
    # Hidden: present, and not drawn.
    assert tags["wall_ymax"].camera_visible is False
    # Everything else: present and drawn.
    assert tags["ceiling"].camera_visible is True
    assert tags["floor"].camera_visible is True


# ---------------------------------------------------------------------------
# Replacement is declared, never inferred
# ---------------------------------------------------------------------------

def test_replacement_is_declared_and_names_a_real_instance():
    box = _box_object()
    _reset_scene_singleton()
    s = Scene("decl")
    s.set_room(4.0, 4.0, 2.6)
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))

    with pytest.raises(InvalidRoom, match="unknown room surface"):
        s.replace_room_surface("nrothwall")
    with pytest.raises(InvalidRoom, match="no instance named"):
        s.replace_room_surface("ceiling", "not_placed")

    # Viewer-relative aliases normalize to the canonical axis-signed names, so
    # the IR only ever carries one spelling.
    s.replace_room_surface("left")
    assert s.finalize()["room"]["suppressed_surfaces"] == ["wall_xmax"]


def test_only_a_declared_replacement_escapes_room_containment():
    """The exemption is keyed off the declaration, not off looking like a wall.

    Inferring it from overlap would fail in both directions: a bookshelf flat
    against a wall would delete it, and a wall modeled 20 mm proud would leave
    the shell standing behind it.
    """
    from dycraft.verify import scene_verify

    box = _box_object()
    wall = _box_object("wall", (4.0, 0.12, 2.6))
    _reset_scene_singleton()
    s = Scene("exempt")
    s.set_room(4.0, 4.4, 2.6)
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
    w = s.place(ObjectHandle(wall), name="back_wall",
                pose=((0.0, -2.16, 0.0), (0, 0, 0)))
    s.replace_room_surface("back", w)
    ir = s.finalize()

    exempt = set((ir["room"].get("surface_replacements") or {}).values())
    assert exempt == {"back_wall"}
    # The source of the exemption is the declaration, and nothing else is in it.
    assert "b" not in exempt
    assert hasattr(scene_verify, "ROOM_CONTAINMENT")


# ---------------------------------------------------------------------------
# Enclosure: an extent is not an enclosure
# ---------------------------------------------------------------------------

def _room_info_for(ir, catalog, override=None):
    req = RenderRequest(layer="scene", ir=ir, catalog=catalog, target="all",
                        out_dir=Path(tempfile.mkdtemp()), step=0, cache={},
                        room_enclosure=override)
    return _build_world_geometry(req)[1]


def _front_cam():
    return CameraSpec(id="front", eye=(0, 40, 1.5), target=(0, 0, 1.0),
                      up=(0, 0, 1), fov_deg=45)


@pytest.mark.parametrize("enclosure,expected", [
    ("interior", list(ROOM_SURFACES)),
    ("open_top", ["floor", "wall_xmin", "wall_xmax", "wall_ymin", "wall_ymax"]),
    ("open_air", ["floor"]),
])
def test_enclosure_decides_which_shell_surfaces_exist(enclosure, expected):
    """Declaring how big the world is must not decide whether it has a lid.

    An outdoor scene uses `set_room` for its ground plane and its extent, and
    must not get a ceiling and four walls with it: sealing an outdoor scene
    shuts out the world dome and darkens the view almost to black.
    """
    box = _box_object()
    _reset_scene_singleton()
    s = Scene(f"enc_{enclosure}")
    s.set_room(4.0, 4.4, 2.6, enclosure=enclosure)
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
    info = _room_info_for(s.finalize(), {box["id"]: box})
    assert [b.tag for b in room_wall_boxes(_front_cam(), info)] == expected


def test_only_a_capped_room_gets_the_interior_rig():
    """The interior rig hangs three area lights under a ceiling.

    With no ceiling there is nothing for them to be under and nothing for them
    to bounce off, so an open-air extent is lit like a subject instead.
    """
    box = _box_object()
    for enclosure, capped in (("interior", True), ("open_top", False),
                              ("open_air", False)):
        _reset_scene_singleton()
        s = Scene(f"cap_{enclosure}")
        s.set_room(4.0, 4.4, 2.6, enclosure=enclosure)
        s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
        info = _room_info_for(s.finalize(), {box["id"]: box})
        assert info["capped"] is capped, enclosure


def test_enclosure_is_absent_from_the_ir_unless_it_is_not_the_default():
    """Writing it unconditionally would re-id every scene ever built."""
    box = _box_object()
    for enclosure, present in (("interior", False), ("open_air", True)):
        _reset_scene_singleton()
        s = Scene(f"id_{enclosure}")
        s.set_room(4.0, 4.4, 2.6, enclosure=enclosure)
        s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
        assert ("enclosure" in s.finalize()["room"]) is present


def test_a_render_time_override_corrects_a_run_that_predates_enclosure():
    """An override for a run whose IR carries no enclosure, and it stores nothing.

    An IR with no enclosure key reads as `interior`. Editing a run's ir.json
    would falsify the run, so the correction is an argument to the renderer.
    """
    box = _box_object()
    _reset_scene_singleton()
    s = Scene("legacy")
    s.set_room(4.9, 18.4, 5.5)              # no enclosure: an older scene
    s.place(ObjectHandle(box), name="b", pose=((0, 0, 0), (0, 0, 0)))
    ir = s.finalize()
    assert "enclosure" not in ir["room"]

    cat = {box["id"]: box}
    assert _room_info_for(ir, cat)["capped"] is True
    assert _room_info_for(ir, cat, override="open_air")["capped"] is False
    # The override changed nothing stored.
    assert "enclosure" not in ir["room"]


def test_the_enclosure_vocabulary_is_shared_with_the_renderer():
    """One table, so the SDK and both backends cannot disagree."""
    assert room_surfaces_suppressed_by("interior") == ()
    assert room_surfaces_suppressed_by("open_top") == ("ceiling",)
    assert set(room_surfaces_suppressed_by("open_air")) == set(ROOM_SURFACES) - {"floor"}
    # Unknown or absent is "no opinion", never a guess.
    assert room_surfaces_suppressed_by(None) == ()
    assert room_surfaces_suppressed_by("nonsense") == ()


def test_an_unknown_enclosure_is_refused_at_authoring_time():
    _reset_scene_singleton()
    s = Scene("bad")
    with pytest.raises(InvalidRoom, match="unknown room enclosure"):
        s.set_room(4.0, 4.0, 2.6, enclosure="outdoors")


# ---------------------------------------------------------------------------
# Authored architecture: the camera cutaway for `replace_room_surface`
# ---------------------------------------------------------------------------

def _replaced_room_info(replaced=ROOM_SURFACES):
    """A room whose named surfaces are all authored by one instance."""
    return {"aabb": ((-2.2, -2.3, 0.0), (2.2, 2.3, 2.75)),
            "wall": {"color": (238, 233, 224)},
            "floor": {"color": (176, 138, 96)},
            "ceiling": None,
            "suppressed": tuple(replaced),
            "replacements": {s: "shell" for s in replaced}}


def _cam(eye):
    return CameraSpec(id="c", eye=eye, target=(0.0, 0.0, 1.37),
                      up=(0, 0, 1), fov_deg=45)


def test_a_fully_replaced_room_emits_no_shell_but_still_cuts_away():
    """The gap this cutaway closes, stated as the two halves that made it.

    `replace_room_surface` takes a surface out of `room_wall_boxes` entirely,
    so there is nothing left to carry a `camera_visible` flag — and the
    authored object standing in its place is an ordinary instance. A room whose
    six surfaces were all authored therefore rendered as a closed box from
    outside, which is exactly where the canonical views stand.
    """
    from dycraft.render._world import room_cutaway_planes

    info = _replaced_room_info()
    cam = _cam((0.0, 5.61, 1.37))
    assert room_wall_boxes(cam, info) == [], (
        "a fully replaced room must emit no shell geometry")
    planes = room_cutaway_planes(cam, info)
    assert [h["tag"] for h in planes["shell"]] == ["wall_ymax"], (
        "the eye is outside +Y only, so only that surface may be cut")


def test_the_cutaway_names_only_surfaces_the_eye_is_behind():
    """Same geometric rule as the shell's own, including the diagonal case."""
    from dycraft.render._world import room_cutaway_planes

    info = _replaced_room_info()
    def tags(eye):
        return sorted(h["tag"] for h in
                      room_cutaway_planes(_cam(eye), info).get("shell", []))

    assert tags((0.0, 5.0, 1.37)) == ["wall_ymax"]
    assert tags((0.0, -5.0, 1.37)) == ["wall_ymin"]
    # Diagonally outside two spans: both, and nothing else. The pre-`b3` shell
    # got exactly this wrong and drew a flanking wall across the shot.
    assert tags((5.0, 5.0, 1.37)) == ["wall_xmax", "wall_ymax"]
    # Inside the room there is nothing between the eye and the contents.
    assert tags((0.0, 1.0, 1.37)) == []
    # A surface the scene did NOT replace is the shell's business, not this.
    only_floor = _replaced_room_info(replaced=("floor",))
    assert sorted(h["tag"] for h in room_cutaway_planes(
        _cam((0.0, 5.0, 1.37)), only_floor).get("shell", [])) == []


def test_the_cutaway_plane_contains_the_wall_slabs_inner_face():
    """The bias, pinned as the failure it prevents.

    An authored wall's inner face sits exactly ON the room's interior bound.
    With a strict comparison the outer face is inside the half-space and the
    inner face is not, so a camera ray passed through the outside of the wall,
    hit the inside of the same wall from behind and rendered its unlit
    backface — a black block with the window as its only opening.
    """
    from dycraft.render._world import room_cutaway_slots

    slots = {s["tag"]: s for s in room_cutaway_slots(_replaced_room_info())["shell"]}

    ymax = slots["wall_ymax"]          # sign +1, room face at y = +2.3
    assert ymax["sign"] == 1
    assert 2.3 > ymax["bound"], "the +Y plane must sit inside y = 2.3"
    assert 2.3 - ymax["bound"] < 1e-3, "and only just inside it"

    ymin = slots["wall_ymin"]          # sign -1, room face at y = -2.3
    assert ymin["sign"] == -1
    assert ymin["bound"] > -2.3, "the -Y plane must sit inside y = -2.3"
    assert ymin["bound"] + 2.3 < 1e-3

    # The comparison the renderer makes is strict, so the inner face at exactly
    # the room bound has to satisfy it.
    for slot, face in ((ymax, 2.3), (ymin, -2.3)):
        inside = (face > slot["bound"] if slot["sign"] > 0
                  else face < slot["bound"])
        assert inside, f"{slot['tag']}: the wall's inner face is not cut"


def test_only_the_replacement_instances_parts_carry_a_cutaway():
    """A chair standing against the wall must not be cut with it."""
    from dycraft.render.blender_renderer import _instance_of

    assert _instance_of("shell/walls") == "shell"
    assert _instance_of("desk/desk_body") == "desk"
    assert _instance_of("body") == "", "an object-layer part has no instance"


# ---------------------------------------------------------------------------
# The same cutaway, in Blender
# ---------------------------------------------------------------------------

_needs_blender = pytest.mark.skipif(
    __import__("dycraft.render._blender_env", fromlist=["x"]).blender_python()
    is None,
    reason="needs a configured Blender interpreter")


_CUTAWAY_PROBE = r"""
import sys, json, importlib.util, math, bpy
REPO = sys.argv[-2]
OUT = sys.argv[-1]
sys.path.insert(0, REPO + "/render")
_s = importlib.util.spec_from_file_location(
    "child", REPO + "/render/_blender_child.py")
child = importlib.util.module_from_spec(_s)
sys.modules["child"] = child
_s.loader.exec_module(child)

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 64
sc.render.resolution_x = sc.render.resolution_y = 96
sc.view_settings.view_transform = "Standard"

def slab(name, loc, scale, colour):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    m = bpy.data.materials.new(name + "_mat")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = colour
    o.data.materials.append(m)
    return o

# A room whose +Y wall is a SLAB: outer face y=2.16, inner face y=2.00, so the
# inner face sits exactly on the interior bound the cutaway plane is built from.
slab("floor", (0, 0, -0.05), (2, 2, 0.05), (0.85, 0.85, 0.85, 1))
slab("far", (0, -2.08, 1), (2, 0.08, 1), (0.85, 0.85, 0.85, 1))
marker = slab("marker", (0, 0.6, 0.6), (0.35, 0.35, 0.6), (0.95, 0.08, 0.08, 1))
# Saturated green ONLY on the wall under test, so any green reaching the
# neutral floor can only have bounced off it.
near = slab("near", (0, 2.08, 1), (2, 0.08, 1), (0.02, 0.85, 0.02, 1))

light = bpy.data.lights.new("key", type="AREA")
light.energy = 1200.0
light.size = 2.0
lo = bpy.data.objects.new("key", light)
bpy.context.collection.objects.link(lo)
lo.location = (0, 0, 2.3)
lo.rotation_euler = (math.pi, 0, 0)

cam_d = bpy.data.cameras.new("c")
cam = bpy.data.objects.new("c", cam_d)
bpy.context.collection.objects.link(cam)
sc.camera = cam
cam.location = (0, 8, 0.9)
cam.rotation_euler = (math.pi / 2, 0, math.pi)

def shot(tag):
    sc.render.filepath = "/tmp/_cutaway_probe_" + tag
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load("/tmp/_cutaway_probe_" + tag + ".png")
    px = list(img.pixels)
    w, h = img.size
    def at(fx, fy, r=3):
        # A patch mean, not a pixel. The light-participation check compares two
        # renders a few percent apart, which single-pixel Cycles noise can
        # swamp; averaging (2r+1)^2 samples puts the noise well under it.
        cx, cy = int(fx * w), int(fy * h)
        acc = [0.0, 0.0, 0.0]
        cnt = 0
        for yy in range(max(0, cy - r), min(h, cy + r + 1)):
            for xx in range(max(0, cx - r), min(w, cx + r + 1)):
                i = (yy * w + xx) * 4
                for c in range(3):
                    acc[c] += px[i + c]
                cnt += 1
        return [v / cnt for v in acc]
    res = {"centre": at(0.5, 0.52), "floor": at(0.5, 0.30)}
    bpy.data.images.remove(img)
    return res

result = {"before": shot("before")}
controls = child._install_camera_cutaway(
    near, [{"tag": "wall_ymax", "axis": 1, "sign": 1, "bound": 2.0 - 1e-4}])
child._set_camera_cutaway({k: [v] for k, v in controls.items()}, ["wall_ymax"])
result["after"] = shot("after")
result["rays"] = {r: bool(getattr(near, r)) for r in
                  ("visible_camera", "visible_diffuse", "visible_glossy",
                   "visible_transmission", "visible_shadow")}
result["in_scene"] = near.name in sc.objects
result["verts"] = len(near.data.vertices)
# With the wall REMOVED instead of cut, the room loses its bounce and its
# occlusion. Same camera, so any difference is the wall's light contribution.
bpy.data.objects.remove(near, do_unlink=True)
result["deleted"] = shot("deleted")
open(OUT, "w").write(json.dumps(result))
"""


@_needs_blender
def test_an_authored_wall_is_cut_for_the_camera_and_still_carries_light(tmp_path):
    """The whole contract, in one render: see in, without letting light out.

    Three shots of the same room from the same camera outside its +Y wall:

      before  - the wall is drawn, so the camera sees painted plaster
      after   - the wall is cut, so the camera sees the red marker inside
      deleted - the wall is gone entirely

    `after` must differ from `before` (the camera got in) AND from `deleted`
    (the wall is still bouncing and occluding). That second comparison is the
    one that distinguishes this from alpha, `visible_transmission` or removing
    the geometry: all three would make `after` and `deleted` agree.
    """
    import subprocess
    from dycraft.render._blender_env import blender_child_env, blender_python

    probe = tmp_path / "probe.py"
    probe.write_text(_CUTAWAY_PROBE)
    out = tmp_path / "result.json"
    repo = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [blender_python(), str(probe), repo, str(out)],
        env=blender_child_env(), capture_output=True, text=True, timeout=900)
    assert out.is_file(), f"probe failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    r = json.loads(out.read_text())

    def red(px):
        return px[0] - 0.5 * (px[1] + px[2])

    # 1. The camera gets in: the marker behind the wall is now the brightest
    #    red thing at frame center, rather than painted plaster.
    assert red(r["after"]["centre"]) > red(r["before"]["centre"]) + 0.05, (
        f"the camera still cannot see the interior: {r}")

    # 2. Nothing was hidden the blunt way. Every non-camera ray is untouched
    #    and the geometry is still there with its vertices.
    assert r["rays"] == {"visible_camera": True, "visible_diffuse": True,
                         "visible_glossy": True, "visible_transmission": True,
                         "visible_shadow": True}, (
        f"the cutaway must not clear any ray-visibility flag: {r['rays']}")
    assert r["in_scene"] and r["verts"] == 8

    # 3. **The wall is still lighting the room**, shown by color bleed. The
    #    wall is the only saturated green surface in the scene, so green
    #    arriving at the neutral floor is indirect light off the very wall the
    #    camera is seeing through. Deleting the wall must take that green with
    #    it; alpha, `visible_transmission`, or removing the geometry would all
    #    have taken it already.
    def green(px):
        return px[1] - 0.5 * (px[0] + px[2])

    bleed_cut = green(r["after"]["floor"])
    bleed_gone = green(r["deleted"]["floor"])
    assert bleed_cut > 0.01, (
        f"the cut wall casts no green onto the floor ({bleed_cut:.4f}), so it "
        f"has stopped bouncing light")
    # A difference, not a ratio: with the wall gone the floor's green excess
    # goes NEGATIVE (it is lit by the red marker and a neutral world), and a
    # ratio against a negative baseline is not a threshold at all.
    assert bleed_cut - bleed_gone > 0.02, (
        f"a cut wall and a deleted wall bleed the same green onto the floor "
        f"({bleed_cut:.4f} vs {bleed_gone:.4f}), so the cut is passing light")


def test_every_wall_tag_the_planner_emits_is_one_a_backend_can_remove():
    """`open3d_renderer` keeps ONE Filament scene across a camera batch, so it
    removes the previous camera's surfaces by tag before adding this camera's,
    so its tag list must cover every surface `room_wall_boxes` can emit.

    A missed tag fails silently: Filament refuses a duplicate geometry id,
    warns, and IGNORES the add, so a surface stays frozen at whatever the FIRST
    camera of the batch needed and every later view renders that camera's
    cutaway, degrading exactly the snapshots inspection and review read.

    Asserting the two lists are equal would pass on two copies of the same
    stale list. This asserts the relation that matters: whatever
    `room_wall_boxes` can emit, from any camera, the backend can name.
    """
    import math

    from dycraft.render._world import ROOM_SURFACE_TAGS
    from dycraft.render.open3d_renderer import _ROOM_SURFACE_TAGS

    mat = {"color": [200, 200, 200]}
    info = {"aabb": ((-3.0, -2.5, 0.0), (3.0, 2.5, 3.0)),
            "wall": mat, "floor": mat}

    emitted: set = set()
    for azimuth, elevation in ((0, 15), (90, 15), (180, 15), (270, 15),
                               (45, 40), (0, 89), (0, -89)):
        radius = 8.0
        eye = (radius * math.cos(math.radians(azimuth))
               * math.cos(math.radians(elevation)),
               radius * math.sin(math.radians(azimuth))
               * math.cos(math.radians(elevation)),
               radius * math.sin(math.radians(elevation)))
        cam = CameraSpec(id=f"c{azimuth}_{elevation}", eye=eye,
                         target=(0, 0, 1.5), up=(0, 0, 1), fov_deg=60.0)
        emitted |= {b.tag for b in room_wall_boxes(cam, info)}

    assert emitted, "the planner emitted nothing; the fixture is wrong"
    unremovable = emitted - set(_ROOM_SURFACE_TAGS)
    assert not unremovable, (
        f"open3d_renderer cannot remove {sorted(unremovable)}; those surfaces "
        f"persist into every later camera in the batch")
    assert set(ROOM_SURFACE_TAGS) == set(_ROOM_SURFACE_TAGS), (
        "the backend must read the planner's vocabulary, not keep a copy")


