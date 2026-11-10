"""Render-recipe propagation and recording for the native animation worker.

Two bugs are pinned here, both of which shipped reachable-looking code that
did nothing:

  * `worker.py` read `DYCRAFT_BLENDER_*` from its environment, but
    `runtime.run_worker` hands `Popen` a closed `env=` dict, so the variables
    never crossed the sandbox and the engine could not actually be selected.
  * `bundle.make_manifest` rebuilt the same recipe from `os.environ` in the
    *daemon*, so a manifest could assert an engine the worker never used, and
    `validate_manifest` hashes the manifest rather than the pixels.
"""
import importlib.util
import itertools
import sys
from types import SimpleNamespace

import pytest

from dycraft.animation.blender.bundle import make_manifest
from dycraft.animation.blender.runtime import REPO

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")

_VARS = ("DYCRAFT_BLENDER_ENGINE", "DYCRAFT_BLENDER_EEVEE_SAMPLES",
         "DYCRAFT_BLENDER_WIDTH", "DYCRAFT_BLENDER_HEIGHT",
         "DYCRAFT_BLENDER_WORKBENCH_AA")


def _load_worker(monkeypatch, **env):
    """The real worker module, with the bpy collaborators stubbed out.

    Reloaded per test because the recipe is read into module-level constants at
    import — deliberately, so one job cannot change engine halfway through.
    """
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils",
                        SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    monkeypatch.setitem(sys.modules, "_blender_child", SimpleNamespace())
    # The worker imports the modules beside it by name; drop any a previous test cached, so they see this bpy.
    for sibling in ("surface_colors", "frame_export", "animation_diff"):
        monkeypatch.delitem(sys.modules, sibling, raising=False)
    spec = importlib.util.spec_from_file_location(
        "render_config_worker", REPO / "animation/blender/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene():
    aa = [SimpleNamespace(identifier=v)
          for v in ("OFF", "FXAA", "5", "8", "11", "16", "32")]
    return SimpleNamespace(
        render=SimpleNamespace(
            engine=None, resolution_x=0, resolution_y=0,
            resolution_percentage=0,
            image_settings=SimpleNamespace(file_format=None)),
        eevee=SimpleNamespace(taa_render_samples=None),
        display=SimpleNamespace(
            render_aa="8",
            shading=SimpleNamespace(color_type=None, show_shadows=False),
            bl_rna=SimpleNamespace(
                properties={"render_aa": SimpleNamespace(enum_items=aa)})))


_material_seq = itertools.count()


def _material(base_color=None, diffuse=(0.8, 0.8, 0.8, 1.0)):
    # `surface_color` caches on the material name, so each fixture needs its own.
    name = f"fixture_{next(_material_seq)}"
    if base_color is None:
        return SimpleNamespace(name=name, use_nodes=False,
                               diffuse_color=list(diffuse))
    bsdf = SimpleNamespace(inputs={
        "Base Color": SimpleNamespace(default_value=list(base_color) + [1.0],
                                      is_linked=False, links=[]),
        "Alpha": SimpleNamespace(default_value=1.0, is_linked=False)})
    return SimpleNamespace(
        name=name, use_nodes=True, diffuse_color=list(diffuse),
        node_tree=SimpleNamespace(nodes={"Principled BSDF": bsdf}))


# --------------------------------------------------------------- propagation

def test_sandbox_forwards_the_render_recipe(tmp_path, monkeypatch):
    """`run_worker`'s closed `env=` means bwrap must carry these explicitly."""
    from dycraft.animation.blender import runtime
    monkeypatch.setattr(runtime, "blender_python", lambda: sys.executable)
    # Pin the wheel branch. `shutil.which` is stubbed just below so bwrap and
    # unshare resolve, and `shutil` is one module object, so that stub also
    # answers `blender_binary()`'s PATH backstop -- which would then hand back
    # this interpreter as "a Blender" and take the binary branch instead. The
    # binary tail is asserted in `test_physics_bake_audit.py`.
    monkeypatch.setattr(runtime, "blender_binary", lambda: None)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: sys.executable)
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DYCRAFT_BLENDER_ENGINE", "BLENDER_WORKBENCH")
    monkeypatch.setenv("DYCRAFT_BLENDER_WIDTH", "640")
    cmd = runtime.command(tmp_path / "work", tmp_path / "out", tmp_path / "job")
    pairs = {cmd[i + 1]: cmd[i + 2]
             for i, part in enumerate(cmd) if part == "--setenv"}
    assert pairs["DYCRAFT_BLENDER_ENGINE"] == "BLENDER_WORKBENCH"
    assert pairs["DYCRAFT_BLENDER_WIDTH"] == "640"
    # Unset names must not be forwarded as empty strings: `worker.py` treats
    # "" as falsy and would fall back anyway, but an explicit empty --setenv
    # would make `os.environ.get(...) or default` silently differ from "unset".
    assert "DYCRAFT_BLENDER_HEIGHT" not in pairs
    # The interpreter invocation still comes last.
    assert cmd[-3:] == [sys.executable, "-B", "/tools/worker.py"]


def test_the_diagnostic_engine_default_is_workbench_and_the_knob_still_moves_it(monkeypatch):
    """The default is Workbench, and the env knob overrides it either way.

    A diagnostic frame answers coverage, blocking, contact, and framing, none of
    which shading changes, and Workbench draws one in under a second where
    EEVEE takes over a minute. The knob must be honored rather than ignored,
    in both directions.
    """
    assert _load_worker(monkeypatch).render_config() == {
        "engine": "BLENDER_WORKBENCH", "width": 1280, "height": 720,
        "samples": 64}
    worker = _load_worker(monkeypatch,
                          DYCRAFT_BLENDER_ENGINE="BLENDER_EEVEE_NEXT")
    assert worker.render_config()["engine"] == "BLENDER_EEVEE_NEXT"
    # The recipe the saved blend keeps is a separate constant and does not move
    # with the diagnostic default: the diagnostic engine is Workbench, the
    # stored and delivered one is denoised Cycles. See `persist_render_settings`
    # and `test_the_stored_recipe_is_the_one_delivery_uses`.
    assert worker._DELIVERY["engine"] == "CYCLES"


# ------------------------------------------------------------------ recording

def test_manifest_records_what_the_worker_reported_not_the_daemon_env(
        tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "evaluation").mkdir(parents=True)
    (bundle / "animation.blend").write_bytes(b"blend")
    (bundle / "evaluation/index.json").write_text("{}")
    # A daemon environment that disagrees with the worker in every field.
    monkeypatch.setenv("DYCRAFT_BLENDER_ENGINE", "BLENDER_WORKBENCH")
    monkeypatch.setenv("DYCRAFT_BLENDER_WIDTH", "640")
    monkeypatch.setenv("DYCRAFT_BLENDER_HEIGHT", "360")
    monkeypatch.setenv("DYCRAFT_BLENDER_EEVEE_SAMPLES", "8")
    info = {"fps": 24.0, "n_frames": 10, "frame_start": 1,
            "blender_version": "test",
            "render": {"engine": "BLENDER_EEVEE_NEXT", "width": 1280,
                       "height": 720, "samples": 64}}
    manifest = make_manifest(bundle, scene_ref="scene", evaluation=info)
    assert manifest["render"] == info["render"]


def test_manifest_falls_back_for_a_record_without_a_recipe(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "evaluation").mkdir(parents=True)
    (bundle / "animation.blend").write_bytes(b"blend")
    (bundle / "evaluation/index.json").write_text("{}")
    info = {"fps": 24.0, "n_frames": 10, "frame_start": 1,
            "blender_version": "test"}
    manifest = make_manifest(bundle, scene_ref="scene", evaluation=info)
    # Unchanged on purpose: `_LEGACY_RENDER` is the recipe a record without
    # its own `render` block was drawn with, a fact about that record rather
    # than a default anything chooses.
    assert manifest["render"] == {"engine": "BLENDER_EEVEE_NEXT", "width": 1280,
                                  "height": 720, "samples": 64}


# ------------------------------------------------------- settings application

def test_workbench_aa_is_independent_of_eevee_samples(monkeypatch):
    """Feeding `_SAMPLES` into `render_aa` measured slower than the EEVEE frame
    Workbench exists to undercut, so the two knobs stay separate."""
    worker = _load_worker(monkeypatch,
                          DYCRAFT_BLENDER_ENGINE="BLENDER_WORKBENCH",
                          DYCRAFT_BLENDER_EEVEE_SAMPLES="64")
    scene = _scene()
    worker.bpy = SimpleNamespace(context=SimpleNamespace(scene=scene))
    worker.render_settings()
    assert scene.render.engine == "BLENDER_WORKBENCH"
    assert scene.display.shading.color_type == "MATERIAL"
    assert scene.display.render_aa == "8"          # untouched by samples=64
    assert "workbench_aa" not in worker.render_config()

    worker = _load_worker(monkeypatch,
                          DYCRAFT_BLENDER_ENGINE="BLENDER_WORKBENCH",
                          DYCRAFT_BLENDER_WORKBENCH_AA="16")
    scene = _scene()
    worker.bpy = SimpleNamespace(context=SimpleNamespace(scene=scene))
    worker.render_settings()
    assert scene.display.render_aa == "16"
    assert worker.render_config()["workbench_aa"] == "16"


def test_invalid_workbench_aa_is_refused(monkeypatch):
    worker = _load_worker(monkeypatch,
                          DYCRAFT_BLENDER_ENGINE="BLENDER_WORKBENCH",
                          DYCRAFT_BLENDER_WORKBENCH_AA="64")
    scene = _scene()
    worker.bpy = SimpleNamespace(context=SimpleNamespace(scene=scene))
    with pytest.raises(ValueError, match="WORKBENCH_AA"):
        worker.render_settings()


def test_saved_blend_keeps_the_delivery_recipe(monkeypatch):
    """A Workbench preview must not change the committed artifact.

    `initialize`/`edit` save the blend; if the review engine were in force the
    stored engine — and therefore the file hash `validate_manifest` checks —
    would depend on which engine someone last previewed with.
    """
    worker = _load_worker(monkeypatch,
                          DYCRAFT_BLENDER_ENGINE="BLENDER_WORKBENCH",
                          DYCRAFT_BLENDER_WIDTH="640")
    scene = _scene()
    worker.bpy = SimpleNamespace(context=SimpleNamespace(scene=scene))
    worker.persist_render_settings()
    assert scene.render.engine == "CYCLES"
    assert (scene.render.resolution_x, scene.render.resolution_y) == (1920, 1080)


def test_the_stored_recipe_is_the_one_delivery_uses(monkeypatch):
    """`_DELIVERY` restates `final_render_settings()`; they must not drift.

    The worker cannot import `render_config` by its flat name outside the
    sandbox, so the four values are written out in `worker.py` rather than
    derived. That is the only reason they are duplicated, and this is what
    stops the duplicate going stale.
    """
    from dycraft.animation.blender.render_config import final_render_settings

    delivery = final_render_settings()
    worker = _load_worker(monkeypatch)
    assert worker._DELIVERY == {k: delivery[k] for k in
                                ("engine", "width", "height", "samples")}


# ------------------------------------------------------------ display colors

def test_display_colors_bind_surface_color_only_under_workbench(monkeypatch):
    """Workbench MATERIAL shading reads `diffuse_color`, which nothing else in
    the repo writes — without this the scene renders in one flat gray."""
    green, grey = (0.1, 0.6, 0.2), (0.8, 0.8, 0.8, 1.0)
    worker = _load_worker(monkeypatch,
                          DYCRAFT_BLENDER_ENGINE="BLENDER_WORKBENCH")
    materials = [_material(green), _material(None, (0.3, 0.3, 0.3, 1.0))]
    worker.bpy = SimpleNamespace(data=SimpleNamespace(materials=materials))
    # One, not two: the second material's swatch is not Blender's factory
    # 0.8, so somebody set it deliberately and the guard leaves it
    # alone. The observable result is the same either way — writing
    # `surface_color` back to a node-less material returns its own swatch —
    # but skipping it says so, rather than relying on the write being a no-op.
    assert worker.apply_display_colors() == 1
    assert materials[0].diffuse_color == pytest.approx([*green, 1.0])
    # A material with no node tree keeps whatever swatch it already had.
    assert materials[1].diffuse_color == pytest.approx([0.3, 0.3, 0.3, 1.0])

    # The engine gate is the CALLER's, not this function's. `render_frames`
    # runs it under `if s.render.engine == "BLENDER_WORKBENCH"`, read off the
    # scene that was actually staged — which is the only place that knows: a
    # final-film request picks its own engine, so neither this process's
    # `_ENGINE` nor the requested recipe answers the question. A guard here
    # keyed on `_ENGINE` would get it wrong in the direction that matters: a
    # Workbench film rendered by a Cycles-default worker would bind nothing
    # and draw gray.
    #
    # So the function itself is engine-blind, and this asserts that. Binding
    # under EEVEE would be harmless if it ever happened — `diffuse_color` is a
    # viewport property that EEVEE and Cycles never sample — it is simply not
    # paid for, because the call site does not make the call.
    worker = _load_worker(monkeypatch, DYCRAFT_BLENDER_ENGINE="BLENDER_EEVEE_NEXT")
    materials = [_material(green, grey)]
    worker.bpy = SimpleNamespace(data=SimpleNamespace(materials=materials))
    assert worker.apply_display_colors() == 1
    assert materials[0].diffuse_color == pytest.approx([*green, 1.0])

# ------------------------------------------- representative color extraction
#
# `surface_color` is the only color a consumer that cannot run the shader ever
# sees: Workbench's `diffuse_color` swatch and the viewer's evaluated-geometry
# record. A shader material may not carry a color (`check_material` refuses
# `shader` beside `color`), so the Base Color socket holds
# `DEFAULT_PART_COLOR` -> linear 0.4342 gray, and a shader that *links* the
# socket leaves that gray behind. Over half of shader-bearing parts link it.

_GREY = 0.4342


class _Sockets:
    """Name-keyed like `node.inputs["Base Color"]`, iterable like `node.inputs`."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def __contains__(self, name):
        return any(n == name for n, _ in self._pairs)

    def __getitem__(self, name):
        for n, socket in self._pairs:
            if n == name:
                return socket
        raise KeyError(name)

    def __iter__(self):
        return iter(socket for _n, socket in self._pairs)


def _socket(kind="RGBA", value=(0, 0, 0, 1), upstream=None):
    held = list(value) if isinstance(value, (list, tuple)) else float(value)
    socket = SimpleNamespace(type=kind, default_value=held,
                             is_linked=upstream is not None, links=[])
    if upstream is not None:
        socket.links = [SimpleNamespace(from_node=upstream)]
    return socket


def _node(kind, inputs=(), outputs=(), ramp=None):
    node = SimpleNamespace(type=kind, inputs=_Sockets(inputs),
                           outputs=list(outputs))
    if ramp is not None:
        node.color_ramp = SimpleNamespace(
            elements=[SimpleNamespace(color=list(c) + [1.0]) for c in ramp])
    return node


def _mat(name="m", base_upstream=None, base_value=(_GREY,) * 3 + (1.0,),
         alpha=1.0, use_nodes=True, diffuse=(0.8, 0.8, 0.8, 1.0)):
    bsdf = _node("BSDF_PRINCIPLED", [
        ("Base Color", _socket("RGBA", base_value, base_upstream)),
        ("Alpha", _socket("VALUE", alpha)),
    ])
    bsdf.inputs["Alpha"].default_value = alpha
    return SimpleNamespace(
        name=name, use_nodes=use_nodes, diffuse_color=list(diffuse),
        node_tree=SimpleNamespace(nodes={"Principled BSDF": bsdf}))


@pytest.fixture
def worker(monkeypatch):
    w = _load_worker(monkeypatch)
    w.reset_surface_color_cache()
    return w


def test_linked_color_ramp_averages_its_stops(worker):
    """The oak/stone/floor case: a ColorRamp drives Base Color, so the socket
    keeps a gray no author chose."""
    ramp = _node("VALTORGB", ramp=[(0.2, 0.1, 0.05), (0.6, 0.4, 0.2)])
    got = worker.surface_color(_mat("oak", base_upstream=ramp))
    assert got[:3] == pytest.approx([0.4, 0.25, 0.125])
    assert got[3] == pytest.approx(1.0)


def test_linked_mix_averages_its_constant_inputs(worker):
    mix = _node("MIX_RGB", [("Fac", _socket("VALUE", 0.5)),
                            ("Color1", _socket("RGBA", (1.0, 0.0, 0.0, 1))),
                            ("Color2", _socket("RGBA", (0.0, 0.0, 1.0, 1)))])
    got = worker.surface_color(_mat("mix", base_upstream=mix))
    assert got[:3] == pytest.approx([0.5, 0.0, 0.5])


def test_linked_rgb_node_uses_its_value(worker):
    rgb = _node("RGB", outputs=[SimpleNamespace(default_value=[0.1, 0.2, 0.3, 1])])
    assert worker.surface_color(_mat("rgb", base_upstream=rgb))[:3] == \
        pytest.approx([0.1, 0.2, 0.3])


def test_walk_recurses_through_an_intermediate_node(worker):
    """An unrecognized node is not a dead end: its RGBA inputs are followed."""
    ramp = _node("VALTORGB", ramp=[(0.8, 0.4, 0.0)])
    passthrough = _node("HUE_SAT", [("Fac", _socket("VALUE", 1.0)),
                                    ("Color", _socket("RGBA", upstream=ramp))])
    assert worker.surface_color(_mat("hs", base_upstream=passthrough))[:3] == \
        pytest.approx([0.8, 0.4, 0.0])


class _Pixels:
    def __init__(self, flat):
        self._flat = flat

    def foreach_get(self, out):
        out[:] = self._flat


def _image(name, rgb_bytes, space="sRGB", is_float=False, size=(2, 2)):
    """A fake `bpy.types.Image` whose pixels read the way Blender's do: an
    8-bit image as stored, byte/255."""
    import numpy as np
    n = size[0] * size[1]
    px = np.tile(np.array([c / 255.0 for c in rgb_bytes] + [1.0], dtype=np.float32), n)
    return SimpleNamespace(name=name, size=size, channels=4, is_float=is_float,
                           colorspace_settings=SimpleNamespace(name=space),
                           pixels=_Pixels(px))


def _srgb_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def test_an_image_texture_is_read_as_its_linear_mean(worker):
    """A textured asset -- every imported character -- linked Base Color to an
    image, found no constant upstream, and drew the socket's 0.8 white in
    Workbench drafts and the viewer: a black crow rendered as a white mesh.
    The image's mean is measurable; and since `pixels` returns an 8-bit sRGB
    map encoded, each pixel is linearized before averaging."""
    tex = _node("TEX_IMAGE", [("Vector", _socket("VECTOR", (0, 0, 0)))])
    tex.image = _image("feathers", (46, 46, 45))
    got = worker.surface_color(_mat("crow", base_upstream=tex))[:3]
    assert got == pytest.approx([_srgb_lin(46 / 255)] * 2 + [_srgb_lin(45 / 255)],
                                abs=1e-6)
    assert max(got) < 0.05, "a near-black map must not read as mid-grey"


def test_a_non_colour_image_is_not_linearised(worker):
    tex = _node("TEX_IMAGE", [("Vector", _socket("VECTOR", (0, 0, 0)))])
    tex.image = _image("data", (128, 128, 128), space="Non-Color")
    assert worker.surface_color(_mat("data", base_upstream=tex))[:3] == \
        pytest.approx([128 / 255] * 3, abs=1e-6)


def test_an_image_node_with_no_image_falls_back_to_the_socket_scalar(worker):
    """No measurable color upstream keeps the socket scalar rather than
    inventing one."""
    tex = _node("TEX_IMAGE", [("Vector", _socket("VECTOR", (0, 0, 0)))])
    tex.image = None
    assert worker.surface_color(_mat("img", base_upstream=tex))[:3] == \
        pytest.approx([_GREY] * 3)


def test_a_cyclic_graph_terminates(worker):
    a = _node("MIX_RGB", [("Color1", _socket("RGBA", (0.5, 0.5, 0.5, 1)))])
    b = _node("MIX_RGB", [("Color1", _socket("RGBA", upstream=a))])
    # close the loop
    a.inputs = _Sockets([("Color1", _socket("RGBA", upstream=b))])
    got = worker.surface_color(_mat("cycle", base_upstream=b))
    assert len(got) == 4 and all(0.0 <= v <= 1.0 for v in got)


def test_deep_chain_is_bounded(worker):
    node = _node("VALTORGB", ramp=[(1.0, 1.0, 1.0)])
    for _ in range(sys.modules["surface_colors"]._COLOR_WALK_DEPTH + 4):
        node = _node("MIX_RGB", [("Color1", _socket("RGBA", upstream=node))])
    got = worker.surface_color(_mat("deep", base_upstream=node))
    # The ramp is out of reach, so the socket scalar stands; the point is that
    # it returns at all.
    assert len(got) == 4


def test_unlinked_and_nodeless_materials_are_unchanged(worker):
    # an authored scalar color: read straight off the socket, alpha honored
    got = worker.surface_color(_mat("sage", base_value=(0.23, 0.3, 0.22, 1.0),
                                    alpha=0.5))
    assert got == pytest.approx([0.23, 0.3, 0.22, 0.5])
    # a legitimately gray material stays gray
    assert worker.surface_color(
        _mat("robot", base_value=(0.445, 0.445, 0.445, 1.0)))[:3] == \
        pytest.approx([0.445] * 3)
    # no node tree at all: the viewport swatch
    assert worker.surface_color(
        _mat("flat", use_nodes=False, diffuse=(0.1, 0.2, 0.3, 1.0))) == \
        pytest.approx([0.1, 0.2, 0.3, 1.0])
    # and no material
    assert worker.surface_color(None) == pytest.approx([0.6, 0.6, 0.6, 1])


def test_result_is_cached_per_material_and_resettable(worker, monkeypatch):
    """`evaluated_frame` asks once per object per frame — over a hundred
    thousand times for a long clip — so the walk must happen once per
    material, not once per ask."""
    walks = []
    colors = sys.modules["surface_colors"]
    original = colors._constant_colors
    monkeypatch.setattr(colors, "_constant_colors",
                        lambda *a, **k: walks.append(1) or original(*a, **k))
    ramp = _node("VALTORGB", ramp=[(0.4, 0.2, 0.1)])
    mat = _mat("oak", base_upstream=ramp)
    first = worker.surface_color(mat)
    for _ in range(50):
        assert worker.surface_color(mat) == pytest.approx(first)
    assert len(walks) == 1
    # a returned list must not be the cached object, or a caller could poison it
    worker.surface_color(mat)[0] = 99.0
    assert worker.surface_color(mat) == pytest.approx(first)
    worker.reset_surface_color_cache()
    worker.surface_color(mat)
    assert len(walks) == 2
