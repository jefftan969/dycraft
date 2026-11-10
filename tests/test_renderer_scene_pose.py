"""The scene renderer places instances where the kernel says they are.

`_build_scene_geometry` resolves every instance's world pose itself, so it can
disagree with `scene_query._resolve_world_pose` — the function every verifier,
probe and assertion goes through — and nothing else would notice. An instance
attached to a parent's **moving part** must be drawn where the posed kernel puts
it, not at its rest location; the picture is the only consumer that would show
the difference.

This is the renderer analog of `test_urdf_export.py`'s scene-layer FK
round-trip.

No GPU: `_build_scene_geometry` is pure geometry assembly, and open3d is
imported lazily inside `_open_renderer`, so this stays in the fast suite.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pathlib import Path

from dycraft.render import open3d_renderer as R
from dycraft.sdk import Object, ObjectHandle, Scene, scene_query
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton


@pytest.fixture(autouse=True)
def _reset():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


def _cabinet_with_cup_on_the_door():
    """A cup attached to a cabinet's *door* — a part-scoped parent anchor.

    That is the only shape where a disagreement shows: the child's world pose
    depends on a joint, because `_resolve_placement` resolves a `follow=True`
    edge against the parent's *posed* anchor frame. A cup on the cabinet's
    `top` (a whole-instance anchor) would not move, by design.
    """
    _reset_object_singleton()
    o = Object("cabinet")
    body = o.make_box("body", 0.6, 0.4, 0.8)
    door = o.make_box("door", 0.02, 0.4, 0.8)
    o.declare_root(body)
    o.attach_revolute(
        "hinge", body, door, axis=(0.0, 0.0, 1.0), limits=(0.0, 90.0),
        default=0.0,
        parent_anchor=o.anchor(body, "top_front_left"),
        child_anchor=o.anchor(door, "top_front_left"),
    )
    cabinet = o.finalize()

    _reset_object_singleton()
    o2 = Object("cup")
    o2.declare_root(o2.make_box("body", 0.1, 0.1, 0.1))
    cup = o2.finalize()

    catalog = {cabinet["id"]: cabinet, cup["id"]: cup}

    _reset_scene_singleton()
    s = Scene("cabinet_scene")
    s.set_room(width=4.0, depth=4.0, height=3.0)
    ch, uh = ObjectHandle(cabinet), ObjectHandle(cup)
    cab = s.place(ch, name="cabinet", pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    s.attach(uh, name="cup", parent=cab,
             parent_anchor=s.anchor(cab, "door.top"),
             child_anchor=s.anchor(uh, "bottom"))
    return s.finalize(), catalog


def _cup_drawn_at(world_parts):
    """Where the renderer actually puts the cup's geometry.

    `_WorldPart.name` is `"<instance>/<part>"` and `se3[0]` is its world
    translation. The cup has a single part at its object root, so that
    translation is exactly the instance world pose the kernel should agree with.
    """
    pts = [p.se3[0] for p in world_parts if p.name.split("/")[0] == "cup"]
    assert len(pts) == 1, f"expected one cup part, drew {len(pts)}"
    return tuple(pts[0])


OPEN = {"cabinet.hinge": np.pi / 2}


def test_renderer_agrees_with_the_kernel_when_a_joint_is_closed():
    scene, catalog = _cabinet_with_cup_on_the_door()
    parts, _room = R._build_scene_geometry(scene, catalog, poses=None)
    truth = scene_query._resolve_world_pose(scene, catalog, "cup", {})[0]
    assert _cup_drawn_at(parts) == pytest.approx(truth, abs=1e-6)


def test_renderer_agrees_with_the_kernel_when_the_joint_moves():
    """The renderer passes `poses` into `_resolve_world_pose`, so it draws the
    cup where the kernel swings it with the door, not at its rest place."""
    scene, catalog = _cabinet_with_cup_on_the_door()
    parts, _room = R._build_scene_geometry(scene, catalog, poses=OPEN)
    truth = scene_query._resolve_world_pose(
        scene, catalog, "cup", {}, poses=OPEN)[0]
    assert _cup_drawn_at(parts) == pytest.approx(truth, abs=1e-6)


def test_the_fixture_actually_moves_the_cup():
    """Guards the two tests above from going vacuous.

    If a future change stops `door.top` tracking the hinge, both would still
    pass — renderer and kernel would agree on the wrong answer. This asserts the
    joint moves the cup by a distance far larger than any tolerance above.
    """
    scene, catalog = _cabinet_with_cup_on_the_door()
    closed = scene_query._resolve_world_pose(scene, catalog, "cup", {})[0]
    opened = scene_query._resolve_world_pose(
        scene, catalog, "cup", {}, poses=OPEN)[0]
    moved = max(abs(closed[k] - opened[k]) for k in range(3))
    assert moved > 0.1, f"fixture no longer exercises the bug (moved {moved} m)"


def _renderer_freed_on(monkeypatch, call):
    """Run `call()` with a stub renderer and report the thread that freed it.

    Returns `(thread_name_or_None, render_thread_name)`. `None` means the stub
    was still alive when the drawing call returned — i.e. it was left for a
    later automatic collection on some other thread.
    """
    import threading
    freed = {}

    class _Stub:
        """In a reference cycle, like the real OffscreenRenderer, so that
        dropping the last name is not enough to free it — only gc is."""

        def __init__(self):
            self.self_ref = self          # the cycle
            self.scene = self

        def clear_geometry(self):
            pass

        def __del__(self):
            freed["thread"] = threading.current_thread().name

    monkeypatch.setattr(R, "_open_renderer", lambda *a, **k: _Stub())
    monkeypatch.setattr(R, "_populate_static", lambda *a, **k: None)
    monkeypatch.setattr(R, "_draw_plan", lambda *a, **k: None)
    monkeypatch.setattr(R, "_prepare",
                        lambda req: R._Plan([], None, [], ((0, 0, 0), (1, 1, 1)),
                                            "h", None))
    call()
    where = R._on_render_thread(lambda: threading.current_thread().name)
    return freed.get("thread"), where


def test_render_frees_its_context_on_the_owning_thread(monkeypatch):
    """Filament aborts the process — `terminate()`, not an exception — if an
    Engine is destroyed from a thread other than the one that made it.

    The teardown therefore has to drop the reference and collect **in the frame
    that holds the renderer**. Factoring those three lines into a
    `_release(renderer)` helper silently breaks it: rebinding the parameter
    leaves the caller's name bound, the collect frees nothing, and the cycle
    survives to whatever thread next trips the gc threshold. That regression
    passes every existing test and only shows up as an occasional hard crash,
    which is why it is pinned here.
    """
    from dycraft.render import RenderRequest
    req = RenderRequest(layer="scene", ir={"id": "x"}, catalog={},
                        target="all", out_dir=Path("."), step=0)
    freed_on, render_thread = _renderer_freed_on(
        monkeypatch, lambda: R.render(req))
    assert freed_on is not None, (
        "the renderer outlived render(); it will be collected later on an "
        "arbitrary thread and Filament will abort the process")
    assert freed_on == render_thread, (freed_on, render_thread)


def test_png_compress_level_none_uses_the_module_default(tmp_path, monkeypatch):
    """`None` means the module default, and the module default is 1.

    Level 1 writes a 1280x720 frame several times faster than Pillow's default
    of 6, and a request can still override it, so `None` has to resolve to a
    number rather than to an absence."""
    seen = {}

    class _Img:
        @staticmethod
        def fromarray(arr):
            class _S:
                @staticmethod
                def save(path, **kw):
                    seen.update(kw)
                    open(path, "wb").close()
            return _S()

    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=_Img))
    monkeypatch.setitem(sys.modules, "PIL.Image", _Img)

    R._save_png(np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / "a.png")
    assert seen["compress_level"] == R._PNG_COMPRESS_LEVEL == 1, seen
    # An explicit level still wins; 6 is Pillow's own default.
    R._save_png(np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / "b.png", 6)
    assert seen["compress_level"] == 6


def test_base_deltas_also_reach_the_drawn_geometry():
    """Same failure one field over: `base_deltas` is the animation layer's
    per-frame whole-body motion, and a renderer that dropped it would draw every
    frame of a clip at the scene's rest pose."""
    scene, catalog = _cabinet_with_cup_on_the_door()
    delta = {"cup": ((0.0, 0.0, 0.25), (1.0, 0.0, 0.0, 0.0))}
    parts, _room = R._build_scene_geometry(scene, catalog, base_deltas=delta)
    truth = scene_query._resolve_world_pose(
        scene, catalog, "cup", {}, base_deltas=delta)[0]
    drawn = _cup_drawn_at(parts)
    assert drawn == pytest.approx(truth, abs=1e-6)
    rest = scene_query._resolve_world_pose(scene, catalog, "cup", {})[0]
    assert drawn[2] == pytest.approx(rest[2] + 0.25, abs=1e-6)


def test_scene_renderer_omits_instances_outside_the_exists_set():
    scene, catalog = _cabinet_with_cup_on_the_door()
    parts, _room = R._build_scene_geometry(
        scene, catalog, include_instances={"cabinet"})
    names = {p.name.split("/", 1)[0] for p in parts}
    assert names == {"cabinet"}

    empty, _room = R._build_scene_geometry(
        scene, catalog, include_instances=set())
    assert empty == []
