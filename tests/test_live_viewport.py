"""The viewer's live Blender viewport: resolution, wire format, coalescing.

Everything here stubs the resident child, following `test_blender_backend.py`:
what the parent *resolves and sends* is where the silent bugs live — a joint
name that does not survive the URDF's `<instance>__` prefixing, a settled frame
that paints over a camera the user already left, a backlog of stale states — and
none of that needs bpy or a GPU.

The three tests that do spawn Blender are marked `expensive`. One of them is
there specifically to fail if `use_persistent_data` is ever removed from
`_serve_build`: without it the frame rate drops by an order of magnitude and
nothing else in the suite would notice.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient

from dycraft import render as R
from dycraft.render.interactive_session import InteractiveFrame, InteractiveSession
from dycraft.viewer import live as live_mod

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BOX = {"width": 0.1, "length": 0.1, "height": 0.05}
_MAT = {"color": [180, 170, 160], "alpha": 1.0, "metallic": 0.0,
        "roughness": 0.6}


def _object_ir(obj_id: str = "obj1") -> dict:
    identity = {"rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.05]}
    return {
        "id": obj_id, "name": "widget", "root": "body", "schema": "object/v1",
        "parts": [
            {"name": "body", "primitive": "box", "params": dict(_BOX),
             "material": dict(_MAT)},
            {"name": "lid", "primitive": "box", "params": dict(_BOX),
             "material": dict(_MAT)},
        ],
        "joints": [
            {"name": "hinge", "type": "revolute", "parent": "body",
             "child": "lid", "axis": [1.0, 0.0, 0.0], "limits": [0.0, 1.57],
             "default": 0.0, "id": 1,
             "parent_anchor": dict(identity),
             "child_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                              "translation": [0.0, 0.0, 0.0]}},
        ],
        "anchors": [],
    }


def _write_object_run(tmp_path: Path, name: str = "run") -> Path:
    run = tmp_path / name
    run.mkdir(parents=True)
    (run / "ir.json").write_text(json.dumps(_object_ir()))
    return run


def _write_scene_run(tmp_path: Path, name: str = "scene_run") -> Path:
    run = tmp_path / name
    (run / "catalog").mkdir(parents=True)
    obj = _object_ir("cat-hash-1")
    (run / "catalog" / "cat-hash-1.json").write_text(json.dumps(obj))
    identity = {"rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0]}
    scene = {
        "id": "scene1", "name": "stack", "schema": "dycraft.scene.v1",
        "room": {"width": 2.0, "depth": 2.0, "height": 2.2,
                 "floor_material": dict(_MAT), "wall_material": dict(_MAT)},
        "groups": [], "assertions": [],
        "instances": [
            {"id": 0, "name": "base", "object_name": "widget",
             "object_ref": "cat-hash-1",
             "placement": {"kind": "attach", "parent": "room",
                           "parent_anchor": {**identity, "name": "bottom"},
                           "child_anchor": {**identity, "name": "bottom"}}},
        ],
    }
    (run / "ir.json").write_text(json.dumps(scene))
    return run


class _FakeSession:
    """Stands in for the resident child. Records what it was asked to draw."""

    #: Seconds each frame "takes". The coalescing test raises this.
    delay = 0.0

    def __init__(self, request, **kwargs):
        self.request = request
        self.kwargs = kwargs
        self.started = None
        self.calls: list[dict] = []
        self.alive = True
        self.closed = False

    def start(self, camera):
        self.started = camera
        return {"device": "CPU", "engine": "CYCLES", "parts": 2, "built": 2,
                "warnings": ["one shader raised; drew its scalars"]}

    def frame(self, camera, *, width, height, poses=None, samples=None, seq=0):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({"camera": camera, "width": width, "height": height,
                           "poses": poses, "seq": seq})
        return InteractiveFrame(seq=seq, jpeg=b"\xff\xd8jpeg" + bytes([seq % 256]),
                                width=width, height=height, render_ms=1.5)

    def close(self):
        self.closed = True
        self.alive = False


def _live_app(run_dir: Path, factory) -> FastAPI:
    """`serve()` behind a socket, with the child replaced."""
    app = FastAPI()

    @app.websocket("/live")
    async def route(websocket: WebSocket) -> None:
        await live_mod.serve(websocket, run_dir, session_factory=factory)

    return app


def _decode(blob: bytes) -> tuple[dict, bytes]:
    n = struct.unpack(">I", blob[:4])[0]
    return json.loads(blob[4:4 + n]), blob[4 + n:]


def _hello(**kw) -> str:
    payload = {"protocol": live_mod.PROTOCOL_VERSION, "seq": 0,
               "tier": "interactive", "width": 320, "height": 180,
               "camera": {"eye": [1.0, 1.0, 1.0], "target": [0.0, 0.0, 0.0],
                          "up": [0.0, 0.0, 1.0], "fov_deg": 45.0}}
    payload.update(kw)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Resolution: run dir -> RenderRequest inputs
# ---------------------------------------------------------------------------

def test_an_object_run_resolves_with_identity_joint_names(tmp_path):
    """`object_ir_to_urdf` prefixes nothing, so the map is identity.

    Worth pinning rather than assuming: the scene layer's names go through a
    translation and the object layer's must not, and a stray prefix would send
    a joint the kernel has never heard of.
    """
    target = live_mod.resolve_target(_write_object_run(tmp_path))
    assert target.layer == "object"
    assert target.catalog is None
    assert target.joint_keys == {"hinge": "hinge"}


def test_a_fixed_joint_is_not_posable_and_cannot_be_forwarded(tmp_path):
    """The browser sends the FULL resolved pose, fixed joints included.

    `resolveJointValues` covers every joint in the URDF, so the client's map
    names the fixed ones too. A fixed joint has no value to set, and excluding
    it here means a stray non-zero value on one can never reach the kernel —
    rather than trusting the client to have sent the authored default.
    """
    run = tmp_path / "withfixed"
    run.mkdir()
    ir = _object_ir()
    ir["parts"].append({"name": "foot", "primitive": "box",
                        "params": dict(_BOX), "material": dict(_MAT)})
    ir["joints"].append({
        "name": "foot_mount", "type": "fixed", "parent": "body",
        "child": "foot", "axis": [0.0, 0.0, 1.0], "default": 0.0, "id": 2,
        "parent_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                          "translation": [0.0, 0.0, 0.0]},
        "child_anchor": {"rotation": [1.0, 0.0, 0.0, 0.0],
                         "translation": [0.0, 0.0, 0.05]}})
    (run / "ir.json").write_text(json.dumps(ir))

    target = live_mod.resolve_target(run)
    assert target.joint_keys == {"hinge": "hinge"}
    assert live_mod._poses_from(
        {"joints": {"hinge": 0.3, "foot_mount": 9.9}}, target) == {"hinge": 0.3}


def test_a_scene_run_translates_urdf_joint_names_to_pose_keys(tmp_path):
    """`inst__joint` (what the URDF and the sliders say) -> `inst.joint`.

    The two namings are set by different modules — `urdf_export` uses `__` for
    XML-name uniqueness, `_scene_kernel` uses `.` for pose lookup — and the
    browser only ever sees the first.
    """
    target = live_mod.resolve_target(_write_scene_run(tmp_path))
    assert target.layer == "scene"
    assert target.joint_keys == {"base__hinge": "base.hinge"}
    assert target.asset_base_dir.name == "catalog"


def test_an_animation_run_is_refused_rather_than_drawn_at_rest(tmp_path):
    """The one narrowing against `runs.run_urdf`, and it must be loud.

    An animation frame is per-frame actor motion plus deformed geometry, and
    this path transports neither. Rendering anyway would produce a *correct
    looking* picture of the scene at rest, which is precisely the
    silently-wrong-picture failure the live path exists to avoid.
    """
    run = tmp_path / "anim"
    run.mkdir()
    (run / "ir.json").write_text(json.dumps({
        "id": "a1", "schema": "dycraft.animation.v1", "fps": 30,
        "frames": [], "scene_ref": "s1"}))
    with pytest.raises(live_mod.LiveUnavailable) as exc:
        live_mod.resolve_target(run)
    assert "actor motion" in str(exc.value)


def test_a_run_with_no_ir_is_refused_with_the_reason(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    with pytest.raises(live_mod.LiveUnavailable, match="no ir.json"):
        live_mod.resolve_target(run)


# ---------------------------------------------------------------------------
# Camera and pose translation
# ---------------------------------------------------------------------------

def test_the_camera_arrives_as_a_vertical_fov_camera_spec():
    cam = live_mod._camera_from({"eye": [1.0, 2.0, 3.0],
                                 "target": [0.0, 0.0, 0.5],
                                 "up": [0.0, 0.0, 1.0], "fov_deg": 45.0})
    assert isinstance(cam, R.CameraSpec)
    assert (cam.eye, cam.target, cam.fov_deg) == ((1.0, 2.0, 3.0),
                                                  (0.0, 0.0, 0.5), 45.0)


@pytest.mark.parametrize("payload", [
    {"eye": [1.0, 2.0], "target": [0, 0, 0]},
    {"eye": [1.0, 2.0, float("nan")], "target": [0, 0, 0]},
    {"eye": [1.0, 2.0, float("inf")], "target": [0, 0, 0]},
    {"eye": [1, 1, 1], "target": [0, 0, 0], "fov_deg": 0.0},
    {"eye": [1, 1, 1], "target": [0, 0, 0], "fov_deg": 200.0},
])
def test_a_malformed_camera_is_refused_not_normalized(payload):
    """A NaN eye renders black with nothing saying why; refuse it instead."""
    with pytest.raises(live_mod.LiveUnavailable):
        live_mod._camera_from(payload)


def test_unknown_joint_names_are_dropped_not_forwarded(tmp_path):
    """The URDF carries synthetic placement joints with no IR counterpart.

    `scene_ir_to_urdf` emits `world__<instance>` fixed joints to parent each
    instance; the browser's joint map can contain them. Forwarding one makes
    the kernel resolve a joint that does not exist.
    """
    target = live_mod.resolve_target(_write_scene_run(tmp_path))
    poses = live_mod._poses_from(
        {"joints": {"base__hinge": 0.4, "world__base": 0.0, "nope": 1.0}},
        target)
    assert poses == {"base.hinge": 0.4}


def test_an_empty_joint_map_means_the_authored_default_pose(tmp_path):
    """`None`, not `{}` — the two mean different things to `RenderRequest`."""
    target = live_mod.resolve_target(_write_object_run(tmp_path))
    assert live_mod._poses_from({}, target) is None
    assert live_mod._poses_from({"joints": {}}, target) is None
    assert live_mod._poses_from({"joints": {"unknown": 1.0}}, target) is None


def test_a_frame_size_is_bounded_without_changing_its_aspect_ratio():
    """The aspect is what makes the frame land on the viewport.

    The client stretches what it receives to fill the canvas, so clamping the
    two sides independently would return a squashed picture rather than a
    smaller one — visible on any display taller than `MAX_HEIGHT`.
    """
    def aspect(size):
        return size[0] / size[1]

    # A 21:9 request past the width ceiling.
    assert aspect(live_mod._clamp_size({"width": 5040, "height": 2160})) == \
        pytest.approx(5040 / 2160, rel=1e-3)
    # A portrait request past the height ceiling.
    tall = live_mod._clamp_size({"width": 1600, "height": 2560})
    assert tall[1] <= live_mod.MAX_HEIGHT and tall[0] <= live_mod.MAX_WIDTH
    assert aspect(tall) == pytest.approx(1600 / 2560, rel=1e-3)
    # Below the floor, scaled up rather than squared off.
    assert aspect(live_mod._clamp_size({"width": 32, "height": 8})) == \
        pytest.approx(4.0, rel=1e-3)
    assert live_mod._clamp_size({"width": 0, "height": -4}) == (
        live_mod.MIN_SIDE, live_mod.MIN_SIDE)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

def test_a_frame_is_one_message_so_a_header_cannot_meet_the_wrong_picture():
    """Header and pixels travel together, length-prefixed.

    Two messages would have to be paired by arrival order, and a header
    attached to the previous frame's bytes is a wrong picture with a
    plausible-looking label on it.
    """
    blob = live_mod.encode_frame({"type": "frame", "seq": 7}, b"\xff\xd8xyz")
    header, payload = _decode(blob)
    assert header == {"type": "frame", "seq": 7}
    assert payload == b"\xff\xd8xyz"


def test_a_protocol_mismatch_is_reported_rather_than_desynchronising(tmp_path):
    app = _live_app(_write_object_run(tmp_path), _FakeSession)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello(protocol=999))
        message = json.loads(ws.receive_text())
    assert message["type"] == "error" and message["fatal"] is True
    assert "protocol mismatch" in message["message"]


def test_an_unresolvable_run_reports_on_the_socket_not_as_an_http_error(tmp_path):
    """The client's contract on failure is to keep its own viewport and say why.

    An HTTP error would leave it guessing between "no such run", "no Blender"
    and "the shader would not build" — and the wrong guess is the one where it
    shows an approximation instead.
    """
    run = tmp_path / "empty"
    run.mkdir()
    app = _live_app(run, _FakeSession)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        message = json.loads(ws.receive_text())
    assert message["type"] == "error" and message["fatal"] is True
    assert "no ir.json" in message["message"]


def test_a_session_that_cannot_start_is_fatal_and_says_why(tmp_path):
    """No Blender, no GPU, a shader that will not build: all fatal, all named.

    The client's documented behavior on a fatal error is to keep the three.js
    viewport and show the reason. That is only better than an approximation if
    the reason actually travels — an opaque failure would leave it choosing
    between "retry" and "show something plausible".
    """
    def factory(request, **kwargs):
        class _Dead:
            alive = False

            def start(self, camera):
                raise R.RenderUnavailable(
                    "no Blender interpreter: set $DYCRAFT_BLENDER_PYTHON")

            def close(self):
                pass

        return _Dead()

    app = _live_app(_write_object_run(tmp_path), factory)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello())
        message = json.loads(ws.receive_text())
    assert message["type"] == "error" and message["fatal"] is True
    assert "DYCRAFT_BLENDER_PYTHON" in message["message"]


def test_a_run_id_with_a_path_separator_reaches_its_sub_run(tmp_path, monkeypatch):
    """A run inside a folder, or a sub-run, is addressed as `<parent>/<child>`.

    The route takes a `:path` param so it accepts them, which means the client
    must not collapse the separator — `encodeURIComponent` on the whole id
    would turn it into `%2F` and make every such run unreachable. The browser
    side uses `api.ts`'s shared `encodeRunId`; this pins the server end.
    """
    from dycraft.render import interactive_session as I
    from dycraft.viewer.app import create_app

    monkeypatch.setattr(I, "InteractiveSession", _FakeSession)
    runs = tmp_path / "runs"
    runs.mkdir()
    child = _write_object_run(runs, "parent/child")
    # `resolve_run` requires it; a directory without one is not a run.
    (child / "meta.json").write_text(json.dumps({"layer": "object"}))
    app = create_app(runs_dir=runs)
    with TestClient(app) as client:
        with client.websocket_connect("/api/runs/parent/child/live") as ws:
            ws.send_text(_hello())
            message = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "close"}))
    assert message["type"] == "ready", message


def test_the_route_reports_an_unknown_run_on_the_socket(tmp_path):
    from dycraft.viewer.app import create_app

    (tmp_path / "runs").mkdir()
    app = create_app(runs_dir=tmp_path / "runs")
    with TestClient(app) as client:
        with client.websocket_connect("/api/runs/ghost/live") as ws:
            message = json.loads(ws.receive_text())
    assert message["type"] == "error" and "unknown run ghost" in message["message"]


# ---------------------------------------------------------------------------
# The session loop
# ---------------------------------------------------------------------------

def test_the_hello_doubles_as_the_first_frame_request(tmp_path):
    """One round trip to a picture: the child needs a camera to build against
    anyway, so asking for the current one means the first frame is the right
    frame rather than a default view that then jumps."""
    sessions: list[_FakeSession] = []

    def factory(request, **kw):
        sessions.append(_FakeSession(request, **kw))
        return sessions[-1]

    app = _live_app(_write_object_run(tmp_path), factory)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello(joints={"hinge": 0.75}))
        ready = json.loads(ws.receive_text())
        header, payload = _decode(ws.receive_bytes())
        ws.send_text(json.dumps({"type": "close"}))

    assert ready["type"] == "ready" and ready["layer"] == "object"
    assert ready["warnings"] == ["one shader raised; drew its scalars"]
    assert header["seq"] == 0 and header["width"] == 320
    assert payload.startswith(b"\xff\xd8")
    session = sessions[0]
    assert session.started.eye == (1.0, 1.0, 1.0)
    # The pose travels on the *request*, so the resident scene is built already
    # articulated rather than built at rest and then moved.
    assert session.request.poses == {"hinge": 0.75}
    assert session.closed


def test_a_pose_change_reaches_the_session_as_a_pose_not_a_rebuild(tmp_path):
    sessions: list[_FakeSession] = []

    def factory(request, **kw):
        sessions.append(_FakeSession(request, **kw))
        return sessions[-1]

    app = _live_app(_write_object_run(tmp_path), factory)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello())
        json.loads(ws.receive_text())
        _decode(ws.receive_bytes())
        ws.send_text(_hello(seq=1, joints={"hinge": 1.2}))
        header, _ = _decode(ws.receive_bytes())
        ws.send_text(json.dumps({"type": "close"}))

    assert header["seq"] == 1
    assert len(sessions) == 1, "a pose change must not restart the child"
    assert sessions[0].calls[-1]["poses"] == {"hinge": 1.2}


def test_superseded_states_are_dropped_rather_than_queued(tmp_path):
    """The coalescing rule, which is the whole reason this is not a queue.

    A pointer drag emits states far faster than a frame takes. Queueing them
    renders a trail of camera positions the user has already left and falls
    further behind with every frame; overwriting one pending slot means the
    next frame is always of *now*.
    """
    sessions: list[_FakeSession] = []

    def factory(request, **kw):
        session = _FakeSession(request, **kw)
        session.delay = 0.08
        sessions.append(session)
        return session

    app = _live_app(_write_object_run(tmp_path), factory)
    n = 12
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello())
        json.loads(ws.receive_text())
        _decode(ws.receive_bytes())
        for k in range(1, n + 1):
            ws.send_text(_hello(seq=k, camera={
                "eye": [1.0 + 0.01 * k, 1.0, 1.0], "target": [0, 0, 0],
                "up": [0, 0, 1], "fov_deg": 45.0}))
        seen = []
        while not seen or seen[-1] != n:
            header, _ = _decode(ws.receive_bytes())
            seen.append(header["seq"])
        ws.send_text(json.dumps({"type": "close"}))

    assert seen[-1] == n, "the newest state must always be the one rendered"
    assert len(seen) < n, f"nothing was coalesced: rendered all of {seen}"


def test_the_tier_the_client_asked_for_comes_back_with_the_frame(tmp_path):
    """The client decides whether a late frame may be painted, and it needs the
    tier to do it: a stale *settled* frame is a jump backwards, a stale
    *interactive* frame is the newest thing there is."""
    app = _live_app(_write_object_run(tmp_path), _FakeSession)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello())
        json.loads(ws.receive_text())
        _decode(ws.receive_bytes())
        ws.send_text(_hello(seq=5, tier="settled", width=1280, height=720))
        header, _ = _decode(ws.receive_bytes())
        ws.send_text(json.dumps({"type": "close"}))
    assert header["tier"] == "settled"
    assert (header["width"], header["height"]) == (1280, 720)


def test_a_refused_frame_does_not_take_the_session_down(tmp_path):
    """One bad request costs that frame. The scene took a second and a half to
    build and a client that sent a NaN camera should not have to pay for it
    again."""
    app = _live_app(_write_object_run(tmp_path), _FakeSession)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello())
        json.loads(ws.receive_text())
        _decode(ws.receive_bytes())
        ws.send_text(_hello(seq=3, camera={"eye": [0, 0], "target": [0, 0, 0]}))
        message = json.loads(ws.receive_text())
        assert message["type"] == "error" and message["fatal"] is False
        # Still alive: the next good request is answered.
        ws.send_text(_hello(seq=4))
        header, _ = _decode(ws.receive_bytes())
        ws.send_text(json.dumps({"type": "close"}))
    assert header["seq"] == 4


def test_the_live_path_writes_nothing_into_the_run_directory(tmp_path):
    """No artifacts and, above all, no render-cache entry.

    The cache key omits resolution (`plan_outputs`), so a 640x360 interactive
    frame keyed on (ir, camera, pose) would hardlink itself over the canonical
    1280x720 still. The interactive path is uncached for that reason.
    """
    run = _write_object_run(tmp_path)
    before = sorted(p.relative_to(run) for p in run.rglob("*"))
    app = _live_app(run, _FakeSession)
    with TestClient(app) as client, client.websocket_connect("/live") as ws:
        ws.send_text(_hello())
        json.loads(ws.receive_text())
        _decode(ws.receive_bytes())
        ws.send_text(json.dumps({"type": "close"}))
    assert sorted(p.relative_to(run) for p in run.rglob("*")) == before


# ---------------------------------------------------------------------------
# InteractiveSession, parent half
# ---------------------------------------------------------------------------

def _request(run: Path, tmp_path: Path, **kw) -> R.RenderRequest:
    return R.RenderRequest(
        layer="object", ir=json.loads((run / "ir.json").read_text()),
        catalog=None, target="all", out_dir=tmp_path / "out", step=0,
        asset_base_dir=run, backend="blender", **kw)


def test_a_missing_blender_interpreter_names_the_fix(tmp_path, monkeypatch):
    from dycraft.render import interactive_session as I

    monkeypatch.setattr(I, "blender_python", lambda: None)
    session = InteractiveSession(_request(_write_object_run(tmp_path), tmp_path))
    with pytest.raises(R.RenderUnavailable, match="setup_blender.sh"):
        session.start(R.CameraSpec(id="live", eye=(1.0, 1.0, 1.0),
                                   target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)))


def test_the_packed_job_is_one_frame_of_the_canonical_part_specs(tmp_path):
    """The job the child builds from must be indistinguishable from a still's.

    Same `_part_spec`, same UV projection, same lighting resolution — the whole
    claim of this feature is that the appearance comes from the canonical path,
    and it only holds if the *input* is canonical too.
    """
    run = _write_object_run(tmp_path)
    session = InteractiveSession(_request(run, tmp_path))
    session._scratch = tmp_path / "scratch"
    session._scratch.mkdir()
    camera = R.CameraSpec(id="live", eye=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.0),
                          up=(0.0, 0.0, 1.0), fov_deg=45.0)
    job = session._build_job(camera)

    assert len(job["frames"]) == 1, "a live session holds exactly one frame"
    frame = job["frames"][0]
    assert len(frame["cameras"]) == 1
    assert len(frame["parts"]) == 2
    assert all(p["kind"] == "mesh" for p in frame["parts"])
    assert frame["cameras"][0]["fov_deg"] == 45.0
    assert job["jpeg_quality"] == 85 and job["stream_dir"]
    # `framing` is what sizes the automatic lighting rig, and it has to be the
    # bbox the canonical path would use or the object is lit differently here.
    assert len(frame["framing"]) == 2


def test_a_part_list_that_changes_shape_is_refused_not_misapplied(tmp_path):
    """The child addresses parts by position.

    A reordered or resized list would place every part after the first
    difference on another part's transform — a wrong picture, not an error, so
    it is made into one.
    """
    run = _write_object_run(tmp_path)
    session = InteractiveSession(_request(run, tmp_path))
    session._part_names = ["body", "lid", "ghost"]
    with pytest.raises(R.RenderUnavailable, match="changed shape"):
        session._world_se3({"hinge": 0.4})


def test_a_pose_key_is_stable_under_reordering_and_int_float_mixing():
    """It decides whether the per-part SE3 list is resent, so a false negative
    costs a whole frame's articulation and a false positive costs nothing."""
    from dycraft.render.interactive_session import _pose_key

    assert _pose_key({"a": 1, "b": 2.0}) == _pose_key({"b": 2, "a": 1.0})
    assert _pose_key(None) is None
    assert _pose_key({"a": 1}) != _pose_key({"a": 1.5})


# ---------------------------------------------------------------------------
# Against real Blender
# ---------------------------------------------------------------------------

def _blender_or_skip():
    from dycraft.render import _blender_env as E

    interp = E.blender_python()
    if not interp:
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")
    return interp


@pytest.mark.expensive
def test_an_interactive_frame_matches_the_canonical_render(tmp_path):
    """The requirement the whole feature rests on, measured.

    Renders one camera and pose twice — once through
    `blender_renderer.render()` and once through the resident session — and
    compares. The bar is the JPEG floor: whatever q85 encoding costs the
    canonical PNG itself, plus a margin for the sample-count difference. Any
    real divergence (a different material, a different light rig, a different
    FOV) is an order of magnitude above that.
    """
    _blender_or_skip()
    import io

    import numpy as np
    from PIL import Image

    from dycraft.render import blender_renderer as B

    run = _write_object_run(tmp_path)
    camera = R.CameraSpec(id="live", eye=(0.30, 0.34, 0.22),
                          target=(0.0, 0.0, 0.04), up=(0.0, 0.0, 1.0),
                          fov_deg=45.0)
    poses = {"hinge": 0.45}
    width, height = B.RENDER_WIDTH, B.RENDER_HEIGHT

    canonical = B.render(_request(run, tmp_path / "canon", cameras=(camera,),
                                  poses=poses, pose_label="parity"))
    png = canonical.render_paths[camera.id]
    reference = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)

    session = InteractiveSession(_request(run, tmp_path / "live"))
    session.start(camera)
    try:
        frame = session.frame(camera, width=width, height=height, poses=poses)
    finally:
        session.close()
    got = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB"),
                     dtype=np.float32)
    assert got.shape == reference.shape

    buf = io.BytesIO()
    Image.open(png).convert("RGB").save(buf, "JPEG", quality=85)
    floor = np.abs(np.asarray(Image.open(io.BytesIO(buf.getvalue()))
                              .convert("RGB"), dtype=np.float32)
                   - reference).mean()
    delta = np.abs(got - reference).mean()
    assert delta < max(floor * 2.0, 1.5), (
        f"interactive frame diverges from the canonical render: "
        f"mean |d| {delta:.4f}/255 against a JPEG floor of {floor:.4f}")


@pytest.mark.expensive
def test_articulating_a_joint_actually_changes_the_pixels(tmp_path):
    """Guards the failure mode `use_persistent_data` could plausibly have.

    Persistent data keeps the exported scene and the BVHs resident between
    renders, so the question that has to be answered by pixels rather than by
    reasoning is whether a transform change invalidates it. It does — but a
    version of this check that moved a part outside the frame reported "no
    change" and looked exactly like a stale cache, so this one sweeps a joint
    whose child part fills a good share of the shot.
    """
    _blender_or_skip()
    import io

    import numpy as np
    from PIL import Image

    run = _write_object_run(tmp_path)
    camera = R.CameraSpec(id="live", eye=(0.30, 0.34, 0.22),
                          target=(0.0, 0.0, 0.04), up=(0.0, 0.0, 1.0),
                          fov_deg=45.0)
    session = InteractiveSession(_request(run, tmp_path))
    session.start(camera)
    try:
        frames = []
        for value in (0.0, 0.5, 1.0, 0.5, 0.0):
            got = session.frame(camera, width=320, height=180,
                                poses={"hinge": value})
            frames.append(np.asarray(
                Image.open(io.BytesIO(got.jpeg)).convert("RGB"),
                dtype=np.float32))
    finally:
        session.close()

    moved = np.abs(frames[2] - frames[0]).mean()
    returned = np.abs(frames[4] - frames[0]).mean()
    assert moved > 0.5, f"the joint swept and nothing moved ({moved:.4f}/255)"
    assert returned < moved / 4.0, (
        f"returning to the original pose did not return the picture "
        f"({returned:.4f}/255 vs {moved:.4f})")


@pytest.mark.expensive
def test_the_resident_session_stays_interactive_across_frames(tmp_path):
    """Fails if `use_persistent_data` is ever dropped from `_serve_build`.

    Without it Cycles re-exports the scene and rebuilds every BVH per call,
    a fixed cost per frame before a single sample. The flag is read off the
    scene the session serves frames from, so the check holds on any device:
    a frame-time bound tuned on a GPU fails on a CPU-only host whether or not
    the flag is set.
    """
    _blender_or_skip()

    run = _write_object_run(tmp_path)
    camera = R.CameraSpec(id="live", eye=(0.30, 0.34, 0.22),
                          target=(0.0, 0.0, 0.04), up=(0.0, 0.0, 1.0),
                          fov_deg=45.0)
    session = InteractiveSession(_request(run, tmp_path))
    ready = session.start(camera)
    try:
        assert ready["persistent_data"] is True, ready
        for k in range(2):
            moved = R.CameraSpec(id="live", eye=(0.30 + 0.004 * k, 0.34, 0.22),
                                 target=(0.0, 0.0, 0.04), up=(0.0, 0.0, 1.0),
                                 fov_deg=45.0)
            session.frame(moved, width=480, height=270)
    finally:
        session.close()
