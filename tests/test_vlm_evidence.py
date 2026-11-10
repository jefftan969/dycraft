"""Unit tests for the image evidence given to the reviewer (`review`)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from PIL import Image

from dycraft.render import CANONICAL_CAMERA_IDS, canonical_cameras
from dycraft.sdk import Object, ObjectHandle, Scene
from dycraft.sdk.object_builder import _reset_object_singleton
from dycraft.sdk.scene_builder import _reset_scene_singleton
from dycraft.verify.review_images import render_review_images


@pytest.fixture(autouse=True)
def _reset_singleton():
    _reset_object_singleton()
    _reset_scene_singleton()
    yield
    _reset_object_singleton()
    _reset_scene_singleton()


def test_bottom_camera_is_always_rendered_and_looks_up():
    """The underside is part of the canonical set, not an opt-in extra.

    Nothing else can infer it, and the reviewer judges against it — so an
    agent that renders the canonical views has to see what the reviewer saw.
    """
    center = (1.0, 2.0, 3.0)
    cameras = canonical_cameras(center, 0.5, 0.25)

    assert {camera.id for camera in cameras} == set(CANONICAL_CAMERA_IDS)
    assert len(cameras) == 6

    bottom = cameras[-1]
    assert bottom.id == "bottom"
    assert bottom.eye[2] < bottom.target[2]
    assert bottom.target == center
    assert bottom.up == (0.0, 1.0, 0.0)


def test_scene_judge_passes_explicit_catalog_asset_root(tmp_path, monkeypatch):
    obj = Object("catalog_object")
    obj.declare_root(obj.make_box("body", 0.4, 0.4, 0.4))
    obj_ir = obj.finalize()
    _reset_scene_singleton()
    scene = Scene("scene")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(ObjectHandle(obj_ir), name="item",
                pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    scene_ir = scene.finalize()

    from dycraft import render as render_module
    from dycraft.render import CANONICAL_CAMERA_IDS, RenderResult

    catalog_root = tmp_path / "catalog_assets"
    seen_roots = []

    def fake_render(request):
        seen_roots.append(request.asset_base_dir)
        paths = {}
        for camera in CANONICAL_CAMERA_IDS:
            path = request.out_dir / f"scene_{camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 16), (100, 100, 100)).save(path)
            paths[camera] = path
        return RenderResult(render_paths=paths)

    monkeypatch.setattr(render_module, "render_request", fake_render)
    payload = SimpleNamespace(
        catalog={obj_ir["id"]: obj_ir},
        asset_base_dir=catalog_root,
    )
    ctx = SimpleNamespace(
        run_dir=tmp_path, step=9, render_cache={}, layer_payload=payload)

    images, articulated, _layout, error = render_review_images(
        scene_ir, ctx, "scene")
    assert error is None
    assert articulated is False
    assert len(images) == 6, "a scene renders the same six canonical views"
    assert [image.label for image in images.images] == list(CANONICAL_CAMERA_IDS), (
        "a scene's images must name all six defaults, bottom included, or "
        "every later image is mislabelled")
    assert seen_roots == [catalog_root]


def test_articulated_scene_legend_labels_line_up_with_its_ten_images(
        tmp_path, monkeypatch):
    """The legend names the images in the order they are sent.

    An articulated scene sends six default views then `front_lo`, `top_lo`,
    `front_hi`, `top_hi` — ten images. A legend off by one would tell the
    reviewer image 6 was `front_lo` when it was `bottom`, and every pose
    comparison it made would be between the wrong two pictures.
    """
    obj = Object("cabinet")
    body = obj.make_box("body", 0.6, 0.4, 0.8)
    door = obj.make_box("door", 0.6, 0.02, 0.8)
    obj.attach_revolute(
        "door_hinge", body, door, axis=(0, 0, 1), limits=(0, 90), default=0,
        parent_anchor=obj.anchor(body, "front_left"),
        child_anchor=obj.anchor(door, "left"),
    )
    obj.declare_root(body)
    obj_ir = obj.finalize()

    _reset_scene_singleton()
    scene = Scene("room")
    scene.set_room(4.0, 4.0, 3.0)
    scene.place(ObjectHandle(obj_ir), name="cab",
                pose=((0.5, 0.5, 0.0), (0.0, 0.0, 0.0)))
    scene_ir = scene.finalize()

    from dycraft import render as render_module
    from dycraft.render import CANONICAL_CAMERA_IDS, RenderResult

    def fake_render(request):
        paths = {}
        for camera in request.camera_ids or CANONICAL_CAMERA_IDS:
            path = request.out_dir / f"{request.pose_label}_{camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 16), (60, 60, 60)).save(path)
            paths[camera] = path
        return RenderResult(render_paths=paths)

    monkeypatch.setattr(render_module, "render_request", fake_render)
    payload = SimpleNamespace(catalog={obj_ir["id"]: obj_ir},
                             asset_base_dir=tmp_path)
    ctx = SimpleNamespace(run_dir=tmp_path, step=2, render_cache={},
                         layer_payload=payload)

    images, articulated, layout, error = render_review_images(
        scene_ir, ctx, "scene")
    assert error is None and articulated is True
    assert layout["kind"] == "all_limit_views"
    assert [image.label for image in images.images] == [
        "front", "right", "back", "left", "top", "bottom",
        "front_lo", "top_lo", "front_hi", "top_hi",
    ]
    # Each label names the pose and camera of the file it is paired with, so a
    # legend entry cannot point at a different picture than the model receives.
    for label, path in zip([image.label for image in images.images], images.paths()):
        if label in CANONICAL_CAMERA_IDS:
            assert path.name == f"default_{label}.png"
        else:
            camera, pose = label.rsplit("_", 1)
            assert path.name == f"{pose}_{camera}.png"


# ---------------------------------------------------------------------------
# Mixed-resolution comparison
#
# The review renders default views at FULL_RESOLUTION and joint states at
# CARD_RESOLUTION. `ImageChops.difference` does not rescale — it compares the
# overlapping top-left rectangle and reports the rest as changed — so a
# comparison across the two would read a resolution mismatch as motion.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The two-sheet layout
# ---------------------------------------------------------------------------

def _ctx(run_dir):
    """The attributes `render_review_images` actually reads off a ToolContext."""
    return SimpleNamespace(run_dir=run_dir, step=7, render_cache={},
                           layer_payload=None)


def _sheet_stub(monkeypatch, tmp_path):
    """Render stub that writes a real distinguishable PNG per camera+pose."""
    from PIL import Image, ImageDraw
    from dycraft import render as render_module
    from dycraft.render import CANONICAL_CAMERA_IDS, RenderResult

    seen = {"requests": [], "renders": 0}

    def fake(request):
        ids = list(request.camera_ids or CANONICAL_CAMERA_IDS)
        paths = {}
        request.out_dir.mkdir(parents=True, exist_ok=True)
        for cid in ids:
            p = request.out_dir / f"{request.pose_label}_{cid}.png"
            im = Image.new("RGB", request.resolution, "white")
            ImageDraw.Draw(im).text((20, 20), f"{cid}/{request.pose_label}",
                                    fill="black")
            im.save(p)
            paths[cid] = p
        seen["requests"].append((request.pose_label, tuple(ids),
                                 request.resolution))
        seen["renders"] += len(ids)
        return RenderResult(render_paths=paths, cameras=[],
                            target_bbox=((0, 0, 0), (1, 1, 1)),
                            cache_hits={c: False for c in ids})

    monkeypatch.setattr(render_module, "render_request", fake)
    return seen


def _rig(n_joints):
    # One Object per sandbox execution is a hard guard, so a test that builds
    # several has to clear it between them.
    _reset_object_singleton()
    obj = Object("rig")
    root = obj.make_box("root", 0.30, 0.20, 0.10)
    for i in range(n_joints):
        arm = obj.make_box(f"arm{i}", 0.10, 0.02, 0.02)
        obj.attach_revolute(
            f"swing{i}", root, arm, axis=(0, 0, 1), limits=(-60, 60),
            default=0, parent_anchor=obj.anchor(root, "top"),
            child_anchor=obj.anchor(arm, "center"))
    obj.declare_root(root)
    return obj.finalize()


def test_the_sheet_layout_sends_two_images_however_many_joints(tmp_path,
                                                               monkeypatch):
    """Two images, flat in the joint count — that is the whole point.

    The card layout scales as `7 + min(J, 12)` images and `18 + 12J` renders,
    so an 18-joint object would draw 234 renders to send 19 images, most of
    them camera-selection candidates that reach no model.
    """
    from dycraft.verify import review_images as J
    ctx = _ctx(tmp_path)
    for n in (1, 3, 8):
        room = tmp_path / f"n{n}"
        room.mkdir(parents=True, exist_ok=True)
        seen = _sheet_stub(monkeypatch, room)
        images, articulated, layout, unavail = J.render_review_images(
            _rig(n), _ctx(room), "object")
        assert unavail is None
        assert len(images) == 2, f"{n} joints -> {len(images)} images"
        assert layout["kind"] == "view_sheets"
        # Six default views plus three cameras at each of two limits.
        assert seen["renders"] == 12, f"{n} joints -> {seen['renders']} renders"


def test_a_static_object_gets_only_the_default_sheet(tmp_path, monkeypatch):
    from dycraft.verify import review_images as J
    _sheet_stub(monkeypatch, tmp_path)
    _reset_object_singleton()
    obj = Object("block")
    root = obj.make_box("root", 0.2, 0.2, 0.2)
    obj.declare_root(root)
    images, articulated, layout, _ = J.render_review_images(
        obj.finalize(), _ctx(tmp_path), "object")
    assert articulated is False
    assert [image.label for image in images.images] == ["default_views"]
    assert "articulation_views" not in layout


def test_the_articulation_sheet_pairs_lo_beside_hi_per_camera(tmp_path,
                                                              monkeypatch):
    """Each ROW is one camera at both limits, because that is the comparison.

    Grouping by pose instead would put the two halves of every comparison in
    different rows of a two-column sheet.
    """
    from dycraft.verify import review_images as J
    _sheet_stub(monkeypatch, tmp_path)
    _images, _art, layout, _ = J.render_review_images(
        _rig(2), _ctx(tmp_path), "object")
    assert layout["articulation_views"] == [
        "front_lo", "front_hi", "top_lo", "top_hi", "left_lo", "left_hi"]


def test_only_the_cameras_that_reach_the_sheet_are_rendered(tmp_path,
                                                            monkeypatch):
    """The card path drew all six cameras at each limit and used four of the
    twelve. The sheet path asks for the three it will paste."""
    from dycraft.verify import review_images as J
    seen = _sheet_stub(monkeypatch, tmp_path)
    J.render_review_images(_rig(2), _ctx(tmp_path), "object")
    by_pose = {label: ids for label, ids, _res in seen["requests"]}
    assert len(by_pose["default"]) == 6
    assert by_pose["lo"] == ("front", "top", "left")
    assert by_pose["hi"] == ("front", "top", "left")


def test_panel_labels_are_drawn_into_the_sheet(tmp_path):
    """A prompt legend cannot tell the reviewer which quadrant it is looking at."""
    import numpy as np
    from PIL import Image
    from dycraft.verify._vlm_evidence import build_view_sheet
    panels = []
    for i, name in enumerate(("front", "right", "back")):
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (1280, 720), (40, 40, 40)).save(p)
        panels.append((name, p))
    out = build_view_sheet(panels, tmp_path / "sheet.png")
    a = np.asarray(Image.open(out).convert("L"))
    # Labels are white-stroked, so a dark sheet must now contain bright pixels.
    assert a.max() > 200
    with Image.open(out) as im:
        assert max(im.size) <= 2048


def test_two_columns_beat_three_for_six_panels(tmp_path):
    """The cap is on the LONGEST side, so a tall sheet spends fewer of its
    pixels on width: 2x3 gives 1024x576 panels, 3x2 gives 682x384."""
    from PIL import Image
    from dycraft.verify._vlm_evidence import build_view_sheet
    panels = []
    for i in range(6):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (1280, 720), "white").save(p)
        panels.append((f"p{i}", p))
    tall = build_view_sheet(panels, tmp_path / "tall.png", columns=2)
    wide = build_view_sheet(panels, tmp_path / "wide.png", columns=3)
    with Image.open(tall) as t, Image.open(wide) as w:
        assert (t.width // 2) * (t.height // 3) > \
               2.0 * (w.width // 3) * (w.height // 2)
def test_the_judge_carries_render_warnings_into_its_evidence_layout(
        tmp_path, monkeypatch):
    """A shader that raised has to be legible in the report it caused.

    The reviewer is what usually sends an agent back to a material, and a
    shader body that failed renders the part's flat scalars — so the review
    will say the surface looks unfinished and the agent has no way to learn
    that its graph never ran unless the renderer's complaint travels with it.

    Deduplicated across pose renders: the same material is rebuilt for every
    pose, so one broken shader would otherwise report itself once per render.

    The sheet path returns through `_sheet_evidence`, which does not see the
    accumulator, so the warnings are stamped onto its layout by the caller.
    """
    obj = Object("plate")
    root = obj.make_box("root", 0.2, 0.2, 0.02)
    obj.declare_root(root)
    ir = obj.finalize()

    from dycraft import render as render_module
    from dycraft.render import CANONICAL_CAMERA_IDS, RenderResult

    warning = ("material 'part_0_mat': shader 'oak' raised KeyError: "
               "'bpy_prop_collection[key]: key \"Scail\" not found'")

    def fake_render(request):
        paths = {}
        for camera in request.camera_ids or CANONICAL_CAMERA_IDS:
            path = request.out_dir / f"fake_{request.pose_label}_{camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 16), (128, 128, 128)).save(path)
            paths[camera] = path
        return RenderResult(render_paths=paths, warnings=[warning])

    monkeypatch.setattr(render_module, "render_request", fake_render)
    ctx = SimpleNamespace(
        run_dir=tmp_path, step=1, render_cache={}, layer_payload=None)

    _paths, _articulated, layout, error = render_review_images(ir, ctx, "object")
    assert error is None
    assert layout["render_warnings"] == [warning], layout
    # `kind` still has to be there — the caller branches on it.
    assert layout["kind"] == "view_sheets"


def test_a_clean_render_adds_no_warnings_key(tmp_path, monkeypatch):
    """Presence of the key is itself the signal, so a clean judgment's evidence
    carries no `render_warnings` key at all."""
    obj = Object("plate")
    root = obj.make_box("root", 0.2, 0.2, 0.02)
    obj.declare_root(root)
    ir = obj.finalize()

    from dycraft import render as render_module
    from dycraft.render import CANONICAL_CAMERA_IDS, RenderResult

    def fake_render(request):
        paths = {}
        for camera in request.camera_ids or CANONICAL_CAMERA_IDS:
            path = request.out_dir / f"c_{request.pose_label}_{camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 16), (128, 128, 128)).save(path)
            paths[camera] = path
        return RenderResult(render_paths=paths)

    monkeypatch.setattr(render_module, "render_request", fake_render)
    ctx = SimpleNamespace(
        run_dir=tmp_path, step=1, render_cache={}, layer_payload=None)
    _paths, _articulated, layout, error = render_review_images(ir, ctx, "object")
    assert error is None
    assert "render_warnings" not in layout


# ---------------------------------------------------------------------------
# One op, several models: the evidence must not share a filename
# ---------------------------------------------------------------------------

def test_two_targets_under_one_step_do_not_share_evidence_paths(tmp_path,
                                                                monkeypatch):
    """`run_review` renders every target through one `ctx`.

    Named by step alone — `renders/<step>_<camera>.png` and
    `renders/<step>_vlm_views/` — the targets would collide, since a step is
    one OP, not one render: each would overwrite the previous one's images and
    record the shared path as its own. The reviewer, told "there are N separate
    models below, judge each one only against its own renders", would be
    handed one model's pictures N times and describe the wrong model.
    """
    from dycraft.verify import review_images as J
    _sheet_stub(monkeypatch, tmp_path)
    ctx = _ctx(tmp_path)

    first, _a, _l, err_a = J.render_review_images(
        _rig(2), ctx, "object", evidence_key="folding_counter")
    second, _b, _l2, err_b = J.render_review_images(
        _rig(2), ctx, "object", evidence_key="sorting_hamper")

    assert err_a is None and err_b is None
    a, b = set(map(str, first.paths())), set(map(str, second.paths()))
    assert a and b
    assert not (a & b), (
        f"two targets of one review share evidence files: {sorted(a & b)}")
    for path in first.paths():
        assert path.exists(), (
            "the first target's sheet was overwritten or removed by the "
            "second — the reviewer would read the second model's pixels "
            "under the first model's name")


def test_a_single_target_render_keeps_its_historical_paths(tmp_path,
                                                           monkeypatch):
    """No key, no subdirectory: a single-target review and every existing run
    bundle keep the same layout."""
    from dycraft.verify import review_images as J
    _sheet_stub(monkeypatch, tmp_path)

    images, _art, _layout, err = J.render_review_images(
        _rig(2), _ctx(tmp_path), "object")

    assert err is None
    for path in images.paths():
        assert path.parent == tmp_path / "renders" / "7_vlm_views", path

