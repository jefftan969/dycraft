"""The Blender backend's parent half: job spec, cache, availability.

Everything here stubs the child, so the suite needs no bpy and no GPU. That is
the right place to draw the line: what the parent *sends* is checkable and is
where the silent bugs live (a dropped wall, an unresolved texture path, a frame
that reuses another's filename), while what Blender does with a correct spec is
Blender's business and only a real render can tell you about it.

The one test that does spawn bpy is marked `expensive`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dycraft import render as R
from dycraft.render import _world


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixtures: a scene small enough to read in a failure message
# ---------------------------------------------------------------------------

_MAT = {"color": (0.5, 0.4, 0.3), "metallic": 0.0, "roughness": 0.8,
        "alpha": 1.0}


def _part(name="p", primitive="box", **kw):
    return _world._WorldPart(
        name=name, size=(1.0, 2.0, 3.0),
        se3=((0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0)),
        material=kw.pop("material", dict(_MAT)),
        primitive=primitive, **kw)


def _cam(cid="front"):
    return R.CameraSpec(id=cid, eye=(0.0, 5.0, 1.0), target=(0.0, 0.0, 0.0),
                        up=(0.0, 0.0, 1.0))


def _plan(parts=None, room=None, cams=None, ir_hash="hash0"):
    return _world._Plan(
        world_parts=(parts if parts is not None else [_part()]),
        room_info=room,
        cameras=(cams if cams is not None else [_cam()]),
        target_bbox=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ir_hash=ir_hash,
        catalog_fp=None,
    )


def _req(tmp_path, **kw):
    base = dict(layer="object", ir={"id": "hash0"}, catalog=None, target="all",
                out_dir=tmp_path / "renders", step=0, backend="blender")
    base.update(kw)
    return R.RenderRequest(**base)


def _load_arrays(job):
    """Every `*_npy` sidecar the job references, keyed by path.

    Read HERE, inside the fake child, because that is the only moment they
    exist: the parent writes them into a `TemporaryDirectory` that is gone by
    the time `render()` returns. A test that reloaded them afterwards would be
    asserting against dangling paths — which is also the reason the real child
    must consume them before it reports.
    """
    out = {}
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k.endswith("_npy") and isinstance(v, str):
                    out[v] = np.load(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(job)
    return out


@pytest.fixture
def spy(monkeypatch, tmp_path):
    """Capture the job the parent would send, and fake the PNGs it expects."""
    from dycraft.render import blender_renderer as B

    jobs = []

    def fake_child(job, scratch, tag=""):
        job = dict(job, _arrays=_load_arrays(job))
        jobs.append(job)
        for frame in job["frames"]:
            for cam in frame["cameras"]:
                p = Path(cam["out"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x89PNG-stub")
        return {"ok": True, "written": []}

    monkeypatch.setattr(B, "_run_child", fake_child)
    monkeypatch.setattr(B, "blender_python", lambda: "/nonexistent/python")
    monkeypatch.setattr(_world, "_prepare", lambda r: r._test_plan)
    return jobs


def _render(tmp_path, plan, **kw):
    from dycraft.render import blender_renderer as B
    req = _req(tmp_path, **kw)
    req._test_plan = plan
    # `_prepare` is imported by name into blender_renderer, so patching it on
    # `_world` alone would miss; patch the module-local binding too.
    import dycraft.render.blender_renderer as mod
    orig = mod._prepare
    mod._prepare = lambda r: r._test_plan
    try:
        return req, B.render(req)
    finally:
        mod._prepare = orig


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_no_interpreter_is_render_unavailable(monkeypatch, tmp_path):
    """Not an ImportError, not a crash.

    Every render call site already catches `RenderUnavailable` and degrades —
    `finalize_render_result` still writes the URDF. A box with no bpy env must lose its pictures, not its run.
    """
    from dycraft.render import blender_renderer as B

    monkeypatch.setattr(B, "blender_python", lambda: None)
    req = _req(tmp_path)
    req._test_plan = _plan()
    monkeypatch.setattr(B, "_prepare", lambda r: r._test_plan)
    with pytest.raises(R.RenderUnavailable, match="DYCRAFT_BLENDER_PYTHON"):
        B.render(req)


def test_env_resolution_prefers_the_var_over_the_pointer_file(monkeypatch,
                                                              tmp_path):
    from dycraft.render import _blender_env as E

    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("DYCRAFT_BLENDER_PYTHON", str(fake))
    assert E.blender_python() == str(fake)


def test_an_empty_env_var_is_a_deliberate_no_interpreter(monkeypatch):
    """Same sentinel `_cad_env` uses — how a caller says "pretend it is absent"
    without deleting the pointer file the rest of the box relies on."""
    from dycraft.render import _blender_env as E

    monkeypatch.setenv("DYCRAFT_BLENDER_PYTHON", "  ")
    assert E.blender_python() is None
    assert E.is_available() is False


def test_the_child_env_drops_the_software_gl_pins(monkeypatch):
    from dycraft.render import _blender_env as E

    base = {"LIBGL_ALWAYS_SOFTWARE": "1", "MESA_LOADER_DRIVER_OVERRIDE":
            "llvmpipe", "EGL_PLATFORM": "surfaceless", "PYTHONPATH": "/harness",
            "KEEP": "yes"}
    env = E.blender_child_env(base)
    assert env == {"KEEP": "yes"}


# ---------------------------------------------------------------------------
# Job spec
# ---------------------------------------------------------------------------

def test_every_part_and_camera_reaches_the_child(spy, tmp_path):
    parts = [_part("a"), _part("b"), _part("c")]
    cams = [_cam("front"), _cam("top")]
    _render(tmp_path, _plan(parts=parts, cams=cams))

    frame = spy[0]["frames"][0]
    assert len(frame["parts"]) == 3
    assert [c["id"] for c in frame["cameras"]] == ["front", "top"]


def test_a_box_arrives_as_arrays_the_parent_generated(spy, tmp_path):
    """Not as a size the child would re-derive a mesh from.

    Boxes are the geometry with no authored UVs, so something has to project
    them — and it must be the same `_triplanar_uvs` both backends use, or a
    tiled floor gets different grain depending on who drew it.
    """
    mat = dict(_MAT, mapping={"scale_m": 0.5},
               texture={"base_color": {"path": "/x.png", "digest": "d"}})
    _render(tmp_path, _plan(parts=[_part(material=mat)]))

    spec = spy[0]["frames"][0]["parts"][0]
    arrays = spy[0]["_arrays"]
    assert spec["kind"] == "mesh"
    verts = arrays[spec["verts_npy"]]
    tris = arrays[spec["tris_npy"]]
    uvs = arrays[spec["uvs_npy"]]
    assert verts.shape == (8, 3) and tris.shape == (12, 3)
    assert uvs.shape == (36, 2)                       # per CORNER, 3 per tri
    assert np.allclose(uvs, _world._triplanar_uvs(verts, tris, 0.5))


def test_an_untextured_box_ships_no_uvs(spy, tmp_path):
    """`scale_m` absent means nothing to sample — the same early return
    `_apply_texture_uvs` takes on the other backend."""
    _render(tmp_path, _plan(parts=[_part()]))
    assert "uvs_npy" not in spy[0]["frames"][0]["parts"][0]


def test_deformed_geometry_travels_as_npy_not_json(spy, tmp_path):
    """A rigged frame can carry tens of thousands of vertices per part.

    Inlining them would make the job spec megabytes of JSON to encode and
    reparse for every frame of a clip.
    """
    verts = np.random.default_rng(0).random((5000, 3))
    tris = np.zeros((4998, 3), dtype=np.int32)
    part = _part(primitive="solid", deformed=(verts, tris),
                 params={"mesh": "/rest.obj"})
    _render(tmp_path, _plan(parts=[part]))

    spec = spy[0]["frames"][0]["parts"][0]
    assert spec["kind"] == "mesh"
    assert np.allclose(spy[0]["_arrays"][spec["verts_npy"]], verts)
    # The rest OBJ travels too: LBS leaves topology alone, so its per-corner
    # UVs are still valid and are the only place they exist.
    assert spec["uv_from_obj"] == "/rest.obj"
    assert len(json.dumps({k: v for k, v in spy[0].items()
                           if k != "_arrays"})) < 100_000


def test_an_asset_part_ships_its_path_and_scale(spy, tmp_path):
    part = _part(primitive="solid",
                 params={"mesh": "/staged/abc.obj", "scale": (2.0, 2.0, 2.0)})
    _render(tmp_path, _plan(parts=[part]))

    spec = spy[0]["frames"][0]["parts"][0]
    assert spec["kind"] == "asset"
    assert spec["obj"] == "/staged/abc.obj"
    assert spec["scale"] == [2.0, 2.0, 2.0]


def test_an_asset_with_no_file_falls_back_to_its_box(spy, tmp_path):
    """The same fallback `_part_mesh` takes, so a broken asset produces one
    wrong-but-legible picture rather than two different ones."""
    _render(tmp_path, _plan(parts=[_part(primitive="solid", params={})]))
    assert spy[0]["frames"][0]["parts"][0]["kind"] == "mesh"


def test_an_asset_carries_its_bounding_box_for_the_child_to_fall_back_on(
        spy, tmp_path):
    """A path that exists as a *string* may still not load, and only the
    importer finds out — so the AABB has to travel with the asset.

    Found by rendering a real clip with the wrong `asset_base_dir`: Open3D drew
    every part as its bounding box (visibly wrong, obviously still a scene) and
    Blender drew an **empty room**, because a dropped part leaves nothing at
    all. Losing the object is strictly worse than drawing a crude one.
    """
    part = _part(primitive="solid", params={"mesh": "/gone/nope.obj"})
    _render(tmp_path, _plan(parts=[part]))

    spec = spy[0]["frames"][0]["parts"][0]
    assert spec["kind"] == "asset"
    fb = spec.get("fallback")
    assert fb, "an asset spec with no fallback can only vanish"
    verts = spy[0]["_arrays"][fb["verts_npy"]]
    assert verts.shape == (8, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    assert np.allclose(hi - lo, part.size), "the fallback is not the part's AABB"


def test_material_paths_arrive_already_resolved(spy, tmp_path):
    """The child cannot resolve a run-relative path — it has no run dir and no
    DyCraft. `_material_to_render` did it parent-side; this checks it survives
    the trip rather than being flattened away."""
    mat = dict(_MAT, texture={"base_color": {"path": "/run/assets/w.jpg",
                                             "digest": "d1"}})
    _render(tmp_path, _plan(parts=[_part(material=mat)]))
    spec = spy[0]["frames"][0]["parts"][0]
    assert spec["material"]["texture"]["base_color"]["path"] == "/run/assets/w.jpg"


# ---------------------------------------------------------------------------
# Room cutaway
# ---------------------------------------------------------------------------

_ROOM = {"aabb": ((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
         "floor": dict(_MAT), "wall": dict(_MAT)}


def test_a_side_camera_gets_all_six_surfaces_with_the_near_wall_hidden(
        spy, tmp_path):
    """Six surfaces exist; the cutaway is a flag on one of them.

    Every surface is built and `camera_visible` decides what the camera may
    see. Leaving out the camera-side wall and the ceiling would remove two of
    the six bounce surfaces, so an interior would be lit inside an open-topped
    three-sided box and read flat however it was lit.
    """
    _render(tmp_path, _plan(room=_ROOM, cams=[_cam("front")]))
    walls = spy[0]["frames"][0]["cameras"][0]["walls"]
    assert [w["tag"] for w in walls] == [
        "floor", "ceiling", "wall_xmin", "wall_xmax", "wall_ymin", "wall_ymax"]
    by_tag = {w["tag"]: w for w in walls}
    # `front` looks from +Y, so the +Y wall is the one between eye and room.
    assert by_tag["wall_ymax"]["camera_visible"] is False
    assert all(by_tag[t]["camera_visible"] is True
               for t in ("floor", "ceiling", "wall_xmin", "wall_xmax",
                         "wall_ymin"))


def test_a_camera_outside_two_spans_hides_both_of_those_walls(spy, tmp_path):
    """The diagonal-exterior case: every surface the eye is outside of is hidden.

    A per-axis rule that picks one far wall on the dominant axis and draws both
    flanking walls opaque puts one of those flanks straight across the view of
    an eye placed diagonally outside two spans at once. The rule is
    per-surface instead.
    """
    diagonal = R.CameraSpec(id="diag", eye=(6.0, 6.0, 1.6),
                            target=(0.0, 0.0, 1.0), up=(0, 0, 1))
    _render(tmp_path, _plan(room=_ROOM, cams=[diagonal]))
    by_tag = {w["tag"]: w for w in spy[0]["frames"][0]["cameras"][0]["walls"]}
    assert by_tag["wall_xmax"]["camera_visible"] is False
    assert by_tag["wall_ymax"]["camera_visible"] is False
    assert by_tag["wall_xmin"]["camera_visible"] is True
    assert by_tag["wall_ymin"]["camera_visible"] is True


def test_every_surface_takes_its_material_own_alpha(spy, tmp_path):
    """No surface gets an alpha override: an unconditional one would discard an
    authored translucent room material, so the material's own alpha (1.0 by
    default) applies."""
    _render(tmp_path, _plan(room=_ROOM, cams=[_cam("front")]))
    walls = spy[0]["frames"][0]["cameras"][0]["walls"]
    assert all(w["alpha"] is None for w in walls)


def test_the_top_camera_still_sees_the_floor_only(spy, tmp_path):
    """Plan views keep what they always showed, but now they are lit properly.

    Every surface is still built — the room bounces as a room — and the four
    walls plus the ceiling are simply hidden from this camera, so a plan view
    is not lit through an open box.
    """
    _render(tmp_path, _plan(room=_ROOM,
                            cams=[R.CameraSpec(id="top", eye=(0, 0, 9),
                                               target=(0, 0, 0), up=(0, -1, 0))]))
    walls = spy[0]["frames"][0]["cameras"][0]["walls"]
    visible = [w["tag"] for w in walls if w["camera_visible"]]
    assert visible == ["floor"]
    assert len(walls) == 6


def test_walls_are_generated_like_any_other_box(spy, tmp_path):
    """A floor's tiling has to match a floor-coloured box part's, and it only
    does if one function laid out both."""
    _render(tmp_path, _plan(room=_ROOM, cams=[_cam("front")]))
    floor = spy[0]["frames"][0]["cameras"][0]["walls"][0]
    assert floor["kind"] == "mesh"
    assert spy[0]["_arrays"][floor["verts_npy"]].shape == (8, 3)


def test_no_room_means_no_walls(spy, tmp_path):
    _render(tmp_path, _plan(room=None))
    assert spy[0]["frames"][0]["cameras"][0]["walls"] == []


# ---------------------------------------------------------------------------
# Cache and filenames
# ---------------------------------------------------------------------------

def test_a_full_cache_hit_never_spawns_a_child(spy, tmp_path):
    from dycraft.render.blender_renderer import BLENDER_CONFIG_VERSION

    cached = tmp_path / "cached.png"
    cached.write_bytes(b"\x89PNG-cached")
    # Read the version from the module rather than spelling it: a literal
    # fails on every recipe bump for no reason of its own, and a *stale* one
    # can keep passing while testing a key the backend does not produce.
    key = _world.plan_outputs(
        _req(tmp_path, cache={}), _plan(),
        config_version=BLENDER_CONFIG_VERSION)[0].key

    req, result = _render(tmp_path, _plan(), cache={key: cached})
    assert spy == []                                  # no job at all
    assert result.cache_hits == {"front": True}
    assert result.render_paths["front"].read_bytes() == b"\x89PNG-cached"


def test_a_miss_is_recorded_only_after_the_png_exists(spy, tmp_path):
    cache = {}
    req, result = _render(tmp_path, _plan(), cache=cache)
    assert len(cache) == 1
    assert next(iter(cache.values())).exists()


def test_a_failed_child_leaves_no_cache_entry(monkeypatch, tmp_path):
    """Otherwise every later render of that (ir, camera, pose) hardlinks a
    file that was never written."""
    import dycraft.render.blender_renderer as B

    monkeypatch.setattr(B, "blender_python", lambda: "/nonexistent/python")
    monkeypatch.setattr(B, "_run_child",
                        lambda job, scratch: {"ok": True, "written": []})
    orig = B._prepare
    B._prepare = lambda r: r._test_plan
    try:
        cache = {}
        req = _req(tmp_path, cache=cache)
        req._test_plan = _plan()
        B.render(req)
        assert cache == {}
    finally:
        B._prepare = orig


def test_the_result_reports_the_framing_the_plan_computed(spy, tmp_path):
    plan = _plan()
    _, result = _render(tmp_path, plan)
    assert result.target_bbox == plan.target_bbox


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------

@pytest.mark.expensive
def test_a_real_blender_render_writes_a_png(tmp_path):
    """Spawns bpy. Proves the child can be driven end to end at all."""
    from dycraft.render import _blender_env as E

    if not E.is_available():
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")

    ir = {
        "id": "smoke",
        "name": "smoke",
        "schema": "object/1",
        "root": "b",
        "parts": [{"name": "b", "primitive": "box",
                   "params": {"width": 1.0, "length": 1.0, "height": 1.0},
                   "material": {"color": [200, 60, 60]}}],
        "joints": [],
    }
    result = R.render_request(R.RenderRequest(
        layer="object", ir=ir, catalog=None, target="all",
        out_dir=tmp_path, step=0, camera_ids=("front",), backend="blender"))

    png = result.render_paths["front"]
    assert png.exists() and png.stat().st_size > 1000

    from PIL import Image
    arr = np.asarray(Image.open(png).convert("RGB"))
    assert arr.shape == (R.RENDER_HEIGHT, R.RENDER_WIDTH, 3)
    # The authored red must survive to the pixels — the sRGB->linear conversion
    # is what makes that true, and without it a warm tan came out cream.
    #
    # This does NOT discriminate between view transforms, though its comment
    # claimed it did: measured under AgX the same swatch is (187,75,68) against
    # Standard's (201,68,68), saturation 0.64 vs 0.66, and the assertion holds
    # either way. Red is the hue AgX moves least. If a future change needs to
    # pin the transform, test a teal (0.58 -> 0.37) or a green (0.60 -> 0.46).
    h, w, _ = arr.shape
    centre = arr[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40]
    r, g, b = centre.reshape(-1, 3).mean(axis=0)
    assert r > g + 40 and r > b + 40, f"centre is not red: {(r, g, b)}"


# ---------------------------------------------------------------------------
# Concurrent workers
# ---------------------------------------------------------------------------


def test_worker_count_is_bounded_by_frames_and_cpu(monkeypatch):
    import dycraft.render.blender_renderer as B

    monkeypatch.delenv("DYCRAFT_BLENDER_WORKERS", raising=False)
    monkeypatch.setattr(B.os, "cpu_count", lambda: 12)
    assert B._worker_count(1) == 1, "a single frame must not fan out"
    assert B._worker_count(802) == B._MAX_WORKERS
    monkeypatch.setattr(B.os, "cpu_count", lambda: 2)
    assert B._worker_count(802) == 1, "a 2-core box cannot host 3 children"


def test_the_worker_override_wins_and_is_sane(monkeypatch):
    import dycraft.render.blender_renderer as B

    monkeypatch.setenv("DYCRAFT_BLENDER_WORKERS", "6")
    assert B._worker_count(802) == 6
    assert B._worker_count(2) == 2, "never more workers than frames"
    monkeypatch.setenv("DYCRAFT_BLENDER_WORKERS", "garbage")
    assert B._worker_count(802) >= 1, "a bad override must not crash the render"


def test_one_child_failing_does_not_strand_its_siblings(monkeypatch, tmp_path):
    """Every future is waited on before the first error is re-raised.

    Returning early would leave sibling Blender processes running, holding GPU
    memory that nothing will ever reclaim.
    """
    import dycraft.render.blender_renderer as B

    monkeypatch.setenv("DYCRAFT_BLENDER_WORKERS", "3")
    finished = []

    def flaky(job, scratch, tag=""):
        if tag == "_w0":
            raise R.RenderUnavailable("boom")
        finished.append(tag)
        return {"ok": True}

    monkeypatch.setattr(B, "_run_child", flaky)
    with pytest.raises(R.RenderUnavailable, match="boom"):
        B._run_children({}, list(range(9)), tmp_path)
    assert sorted(finished) == ["_w1", "_w2"], "siblings were not awaited"


# ---------------------------------------------------------------------------
# Map kinds -> Principled sockets: metallic, height -> Bump, normal -> NormalMap
# ---------------------------------------------------------------------------

#: Built and run inside the bpy interpreter, because the question is a node
#: topology and only Blender can answer it. Renders nothing — `_build_material`
#: needs no scene, so this costs a `bpy` import rather than a path trace.
_NODE_PROBE = r'''
import json, sys
sys.path.insert(0, %(child_dir)r)
import importlib.util
spec = importlib.util.spec_from_file_location("_bc", %(child)r)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

def graph(texture, has_uv=True):
    mat = bc._build_material("probe", {
        "color": (0.5, 0.5, 0.5), "metallic": 0.0, "roughness": 0.8,
        "alpha": 1.0, "texture": texture}, has_uv=has_uv)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    types = sorted(n.type for n in nt.nodes)
    # Compared by (node, socket name), NOT by socket identity: bpy hands out a
    # fresh RNA wrapper per access, so `l.to_socket is node.inputs["Normal"]`
    # is False even for the link you just made.
    def feeding(node, socket="Normal"):
        return [l.from_node.type for l in nt.links
                if l.to_node == node and l.to_socket.name == socket]

    into_normal = feeding(bsdf)
    bump = next((n for n in nt.nodes if n.type == "BUMP"), None)
    into_bump = feeding(bump) if bump is not None else []
    dist = bump.inputs["Distance"].default_value if bump else None
    return {"types": types, "into_normal": into_normal,
            "into_bump_normal": into_bump, "distance": dist,
            "into_metallic": feeding(bsdf, "Metallic"),
            "metallic_value": bsdf.inputs["Metallic"].default_value}

H = {"path": %(height)r, "digest": "h" * 32, "strength": 0.003}
N = {"path": %(normal)r, "digest": "n" * 32}
M = {"path": %(metallic)r, "digest": "m" * 32}
print("RESULT" + json.dumps({
    "height_only": graph({"height": H}),
    "normal_only": graph({"normal": N}),
    "both": graph({"height": H, "normal": N}),
    "height_no_strength": graph({"height": {k: v for k, v in H.items()
                                           if k != "strength"}}),
    "metallic_only": graph({"metallic": M}),
    "metallic_none": graph({}),
    # The UV guard. `has_uv=False` is what `_build_part` passes for a mesh with
    # no UV layer; only the normal map may react to it.
    "normal_no_uv": graph({"normal": N}, has_uv=False),
    "both_no_uv": graph({"height": H, "normal": N}, has_uv=False),
    "base_color_no_uv": graph({"base_color": M}, has_uv=False),
}))
'''


@pytest.mark.expensive
def test_the_map_kinds_reach_the_sockets_they_belong_to(tmp_path):
    """`height` drives Bump; `normal` drives NormalMap; together they CHAIN.

    Principled has one `Normal` input, and Blender replaces an input's existing
    link rather than adding to it — so wiring both maps straight to the BSDF
    would silently drop whichever was built first. Bump's own `Normal` input is
    the composition point: the normal map supplies the surface frame and Bump
    perturbs it.

    Also pins the `Distance` floor. Blender's default is 1.0, i.e. a meter of
    relief, so a map arriving without a `strength` must not inherit it.

    And pins the UV guard: `has_uv=False` must drop the normal map and nothing
    else, because it is the only kind whose failure mode with no UV layer is a
    wrong picture rather than a plain one.
    """
    from dycraft.render import _blender_env as E
    from dycraft.render import blender_renderer as B

    interp = E.blender_python()
    if not interp:
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")

    from PIL import Image as PILImage
    import subprocess

    h = tmp_path / "h.png"
    n = tmp_path / "n.png"
    mt = tmp_path / "m.png"
    PILImage.new("L", (32, 32), 128).save(h)
    PILImage.new("RGB", (32, 32), (128, 128, 255)).save(n)
    PILImage.new("L", (32, 32), 230).save(mt)

    child = Path(B._CHILD)
    script = tmp_path / "probe.py"
    script.write_text(_NODE_PROBE % {
        "child_dir": str(child.parent), "child": str(child),
        "height": str(h), "normal": str(n), "metallic": str(mt)})

    out = subprocess.run([interp, str(script)], capture_output=True, text=True,
                         timeout=300, env=E.blender_child_env())
    assert "RESULT" in out.stdout, f"probe failed:\n{out.stdout}\n{out.stderr}"
    got = json.loads(out.stdout.split("RESULT", 1)[1].splitlines()[0])

    # Height alone: Bump into the BSDF, nothing feeding Bump's own normal.
    assert "BUMP" in got["height_only"]["types"]
    assert got["height_only"]["into_normal"] == ["BUMP"]
    assert got["height_only"]["into_bump_normal"] == []
    assert got["height_only"]["distance"] == pytest.approx(0.003)

    # Normal alone: the normal map goes straight in, with no Bump.
    assert got["normal_only"]["into_normal"] == ["NORMAL_MAP"]
    assert "BUMP" not in got["normal_only"]["types"]

    # Both: one link into the BSDF, from Bump, with NormalMap feeding Bump.
    both = got["both"]
    assert both["into_normal"] == ["BUMP"], (
        f"expected the chain to terminate in Bump, got {both['into_normal']}")
    assert both["into_bump_normal"] == ["NORMAL_MAP"], (
        "the normal map is not feeding Bump — it would be silently dropped")

    # A strength-less map takes the explicit floor, never Blender's 1.0.
    assert got["height_no_strength"]["distance"] == pytest.approx(
        0.002), "a height map with no strength inherited Blender's 1 m default"

    # A metalness map from the generator must actually drive the Metallic
    # socket. Without it a generated brass renders as a brown dielectric, which
    # is the defect binding the map exists to fix.
    assert got["metallic_only"]["into_metallic"] == ["TEX_IMAGE"], (
        "the metallic map is not wired to Principled's Metallic socket")
    # And with no map the authored scalar still governs — the map REPLACES its
    # scalar in Blender rather than multiplying it, so the untextured case has
    # to keep working unchanged.
    assert got["metallic_none"]["into_metallic"] == []
    assert got["metallic_none"]["metallic_value"] == pytest.approx(0.0)

    # --- the UV guard ---------------------------------------------------
    # A normal map on a mesh with no UV layer must NOT be wired. Cycles derives
    # the tangent basis from the active UV map, so with none there is no basis
    # and the perturbation is undefined rather than flat — a confident picture
    # of the wrong surface, where every other kind degrades to a bad picture of
    # the right one.
    assert got["normal_no_uv"]["into_normal"] == [], (
        "a normal map was wired with no UV layer; its tangent basis is "
        "undefined and the lighting is garbage rather than merely flat")
    assert "NORMAL_MAP" not in got["normal_no_uv"]["types"]

    # Height is deliberately NOT gated, so the composition survives the guard:
    # with no UVs the chain loses its normal map and terminates in Bump alone.
    assert got["both_no_uv"]["into_normal"] == ["BUMP"]
    assert got["both_no_uv"]["into_bump_normal"] == [], (
        "Bump is still being fed a normal map that the guard should have "
        "dropped")

    # And no other kind reacts to the flag at all.
    assert got["base_color_no_uv"]["into_metallic"] == []
    assert "TEX_IMAGE" in got["base_color_no_uv"]["types"], (
        "the UV guard leaked past `normal` and dropped a base-colour map")


# ---------------------------------------------------------------------------
# Agent-authored shader bodies, executed in the real bpy
# ---------------------------------------------------------------------------

#: A plausible thing an object agent would write: coordinates -> noise ->
#: ColorRamp into Base Color, plus a Bump off the same noise.
_GOOD_SHADER = '''def marble(nodes, links, bsdf):
    tc = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[1].color = (0.28, 0.29, 0.31, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    links.new(tc.outputs["UV"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.18
'''

#: The failure the author-time checks cannot catch: a socket that does not
#: exist. Blender raises KeyError naming the key, which is what the agent needs.
_BAD_SOCKET_SHADER = '''def bad(nodes, links, bsdf):
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Detial"].default_value = 8.0
    links.new(noise.outputs["Fac"], bsdf.inputs["Base Color"])
'''

_SHADER_PROBE = r'''
import json, sys
sys.path.insert(0, %(child_dir)r)
import importlib.util
spec = importlib.util.spec_from_file_location("_bc", %(child)r)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

GOOD = %(good)r
BAD = %(bad)r
EM = {"color": (255, 200, 150), "strength": 12.0}

def build(shader=None, alpha_override=None, emission=None, name="probe"):
    bc._WARNINGS.clear()
    packed = {"color": (0.9, 0.9, 0.85), "metallic": 0.0, "roughness": 0.7,
              "alpha": 1.0}
    if shader is not None:
        packed["shader"] = shader
    if emission is not None:
        packed["emission"] = emission
    mat = bc._build_material(name, packed, alpha_override)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    return {
        "types": sorted(n.type for n in nt.nodes),
        "n_nodes": len(nt.nodes),
        "base_color_linked": bsdf.inputs["Base Color"].is_linked,
        "normal_linked": bsdf.inputs["Normal"].is_linked,
        "base_color": list(bsdf.inputs["Base Color"].default_value)[:3],
        "roughness": bsdf.inputs["Roughness"].default_value,
        "alpha": bsdf.inputs["Alpha"].default_value,
        "emission_strength": bsdf.inputs["Emission Strength"].default_value,
        "surface_linked": nt.nodes.get("Material Output").inputs["Surface"].is_linked,
        "warnings": list(bc._WARNINGS),
    }

print("RESULT" + json.dumps({
    "no_shader": build(),
    "good": build({"name": "marble", "body": GOOD}),
    "bad_socket": build({"name": "bad", "body": BAD}),
    "missing_fn": build({"name": "nope", "body": GOOD}),
    "empty_body": build({"name": "x", "body": ""}),
    "alpha_override": build({"name": "marble", "body": GOOD}, alpha_override=0.25),
    "emission_ok": build({"name": "marble", "body": GOOD}, emission=EM),
    "emission_after_failure": build({"name": "bad", "body": BAD}, emission=EM),
}))
'''


@pytest.mark.expensive
def test_an_authored_shader_builds_its_graph_and_owns_the_bsdf(tmp_path):
    """The whole shader path, against the bpy that will actually run it.

    Pins the four things the design rests on: a valid body builds its nodes and
    its links REPLACE the scalars (Blender replaces a socket's link rather than
    adding to it, which is what makes "the shader owns the appearance sockets"
    true without any arbitration code); a body that raises leaves a usable
    scalar material rather than a half-built graph; the two harness-owned
    fields, `alpha_override` and `emission`, survive both paths; and every
    failure reports itself, because a shader that fails still produces a
    picture and a warning is the only way anyone finds out.
    """
    from dycraft.render import _blender_env as E
    from dycraft.render import blender_renderer as B

    interp = E.blender_python()
    if not interp:
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")

    import subprocess

    child = Path(B._CHILD)
    script = tmp_path / "shader_probe.py"
    script.write_text(_SHADER_PROBE % {
        "child_dir": str(child.parent), "child": str(child),
        "good": _GOOD_SHADER, "bad": _BAD_SOCKET_SHADER})

    out = subprocess.run([interp, str(script)], capture_output=True, text=True,
                         timeout=300, env=E.blender_child_env())
    assert "RESULT" in out.stdout, f"probe failed:\n{out.stdout}\n{out.stderr}"
    got = json.loads(out.stdout.split("RESULT", 1)[1].splitlines()[0])

    # Baseline: with no shader, the material is the plain one.
    base = got["no_shader"]
    assert base["warnings"] == []
    assert not base["base_color_linked"]
    assert base["roughness"] == pytest.approx(0.7)

    # A valid shader builds its nodes and drives the sockets it links.
    good = got["good"]
    assert good["warnings"] == [], good["warnings"]
    assert "TEX_NOISE" in good["types"] and "VALTORGB" in good["types"]
    assert good["base_color_linked"], (
        "the shader's ColorRamp is not driving Base Color — the scalar won")
    assert good["normal_linked"], "the shader's Bump is not driving Normal"
    # A socket the shader set directly still takes its value, not the packed one.
    assert good["roughness"] == pytest.approx(0.18)

    # A raising body restores the scalar material WHOLE: the graph it half-built
    # is thrown away, so nothing renders as a material nobody authored.
    bad = got["bad_socket"]
    assert bad["n_nodes"] == 2, (
        f"expected a bare Principled + Output after a failed shader, got "
        f"{bad['types']}")
    assert bad["surface_linked"], "the fallback material is not connected"
    assert not bad["base_color_linked"]
    assert bad["base_color"][0] == pytest.approx(
        got["no_shader"]["base_color"][0])
    assert bad["roughness"] == pytest.approx(0.7)
    assert len(bad["warnings"]) == 1
    assert "Detial" in bad["warnings"][0], bad["warnings"][0]
    assert "bad" in bad["warnings"][0] and "probe" in bad["warnings"][0]

    # A body whose function is not the one the IR names, and an empty body.
    assert len(got["missing_fn"]["warnings"]) == 1
    assert "nope" in got["missing_fn"]["warnings"][0]
    assert got["missing_fn"]["n_nodes"] == 2
    assert len(got["empty_body"]["warnings"]) == 1
    assert got["empty_body"]["n_nodes"] == 2

    # `alpha_override` is the renderer's (the room-wall cutaway), not the
    # shader's, and it has to survive a shader taking over the tree.
    assert got["alpha_override"]["alpha"] == pytest.approx(0.25)
    assert got["alpha_override"]["base_color_linked"]

    # `emission` stays harness-applied under a shader — it is a validated
    # physical quantity, not an appearance choice — and still lands after a
    # shader failure, because a fallback surface that should glow still should.
    assert got["emission_ok"]["emission_strength"] == pytest.approx(12.0)
    assert got["emission_ok"]["base_color_linked"]
    assert got["emission_after_failure"]["emission_strength"] == pytest.approx(
        12.0)
