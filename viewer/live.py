"""The viewer's live Blender viewport: run dir -> resident Cycles session.

Two jobs, and the split is the whole design:

  * **Resolution** — turn a run directory into the `RenderRequest` the
    canonical renderer would use for it, so the interactive frames are the same
    scene the stills show. Layer handling follows
    `runs.run_urdf`'s, with one deliberate narrowing: an **animation** run is
    refused rather than rendered, because its viewport is driven by per-frame
    actor motion and deformation that this path does not transport. See
    `resolve_target`.
  * **Coalescing** — a pointer drag produces events far faster than 32 ms, and
    a queue of them would render a trail of stale camera positions and fall
    further behind with every frame. `serve()` therefore keeps exactly ONE
    pending state, overwritten in place, and the render loop always picks up
    the newest. Nothing queues.

Staleness is decided by the *client*, not here, and deliberately: the browser
knows what the user is looking at now, and a frame's own request `seq` travels
back with it so the client can drop a settled high-resolution frame that the
world moved out from under. See `useBlenderStream.ts`.

Nothing in this module writes to the run directory or the render cache. A
render's PNG path (`plan_outputs`) omits resolution, so a 640x360 interactive
frame written there would replace the canonical 1280x720 still for that camera;
the interactive path writes nothing for that reason.
"""

from __future__ import annotations

import asyncio
import json
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import anyio

from . import scene as scene_mod

#: Bumped when the wire format below changes. The client sends its own, and a
#: mismatch is reported rather than left to desynchronize silently.
PROTOCOL_VERSION = 1

#: Hard ceiling on a requested frame. Guards against a client asking for
#: something that would take minutes and hold the single GPU; the settled tier
#: asks for the canvas size, which on a large display can exceed this.
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
MIN_SIDE = 16


class LiveUnavailable(RuntimeError):
    """The run cannot be rendered interactively, with the reason."""


@dataclass
class LiveTarget:
    """A run resolved to everything a `RenderRequest` needs."""

    layer: str
    ir: dict
    catalog: dict | None
    asset_base_dir: Path
    #: URDF joint name -> `RenderRequest.poses` key. Identity on the object
    #: layer; `inst__joint` -> `inst.joint` on the scene layer, which is the
    #: translation between `scene_ir_to_urdf`'s naming and
    #: `_scene_kernel._object_poses_for_instance`'s.
    joint_keys: dict[str, str]


def resolve_target(run_dir: Path) -> LiveTarget:
    """Everything needed to render `run_dir` interactively.

    Raises `LiveUnavailable` rather than returning a request that would draw
    nothing — the same contract, and the same three failure modes, as
    `runs.run_urdf`.
    """
    ir_path = run_dir / "ir.json"
    if not ir_path.is_file():
        raise LiveUnavailable(
            "run has no ir.json — geometry is written on commit")
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    kind = scene_mod.ir_kind(ir)

    if kind == "animation":
        # **Refused, deliberately, rather than approximated.** An animation
        # run's viewport is driven by per-frame whole-body deltas and deformed
        # geometry (`RenderRequest.base_deltas` / `deformed`), and this session
        # transports neither — it applies joint poses and a camera. A frame
        # that silently dropped the actor motion would look like a correct
        # render of a scene at rest, which is the one failure mode the whole
        # live path exists to avoid. The scrubber and three.js playback are
        # untouched; they stay the way to view a clip.
        raise LiveUnavailable(
            "the live Blender viewport does not carry per-frame actor motion "
            "or deformation, so an animation run would render at rest — use "
            "the timeline scrubber with the preview viewport instead")

    if kind == "object":
        return LiveTarget("object", ir, None, run_dir, _joint_keys(ir, ""))

    if kind != "scene":
        raise LiveUnavailable(
            "ir.json is neither an ObjectIR (parts) nor a SceneIR (instances)")

    catalog = scene_mod.load_catalog(run_dir)
    resolved, total = scene_mod.catalog_coverage(ir, catalog)
    if total and not resolved:
        raise LiveUnavailable(
            f"scene has {total} instance(s) but none resolve against "
            f"{scene_mod.CATALOG_DIR}/")
    joints = {}
    for inst in ir.get("instances") or ():
        if not isinstance(inst, dict):
            continue
        name = inst.get("name")
        obj_ir = catalog.get(inst.get("object_ref"))
        if not name or not isinstance(obj_ir, dict):
            continue
        # `<instance>__<joint>` is what `scene_ir_to_urdf` emits and therefore
        # what the browser's joint map is keyed by; `<instance>.<joint>` is what
        # `_scene_kernel._object_poses_for_instance` looks up.
        joints.update({f"{name}__{k}": f"{name}.{v}"
                       for k, v in _joint_keys(obj_ir, "").items()})
    return LiveTarget("scene", ir, catalog,
                      run_dir / scene_mod.CATALOG_DIR, joints)


def _joint_keys(obj_ir: dict, prefix: str) -> dict[str, str]:
    """Posable joints only, URDF name -> IR name.

    **Fixed joints are excluded.** The browser sends the full resolved pose,
    which covers every joint in the URDF including the fixed ones — and a fixed
    joint has no value to set by definition. Dropping them here means a stray
    non-zero value on one can never reach the kernel, rather than relying on
    the client to have sent the authored default.
    """
    return {f"{prefix}{j['name']}": f"{prefix}{j['name']}"
            for j in obj_ir.get("joints") or ()
            if isinstance(j, dict) and j.get("name")
            and j.get("type") != "fixed"}


def _camera_from(payload: dict):
    """A `CameraSpec` from the browser's three.js camera, in dycraft world axes.

    The client is responsible for undoing the viewer's own change of basis (the
    robot group's z-up rotation and ground-normalizing offset) before sending,
    which it does by inverting that group's `matrixWorld` — the same
    conversion `SceneCanvas` already uses for `fitBounds`. Doing it there
    rather than here keeps the one place that knows about the three.js frame in
    the three.js code.

    `fov_deg` is the VERTICAL field of view, which is what
    `THREE.PerspectiveCamera.fov` is and what `CameraSpec.fov_deg` means
    (`sensor_fit = "VERTICAL"` in the child). Matching aspect ratio and
    vertical FOV is what makes the streamed frame land on the viewport without
    a framing jump.
    """
    from ..render import CameraSpec

    def vec3(key, default=None):
        raw = payload.get(key)
        if raw is None and default is not None:
            return default
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise LiveUnavailable(f"camera.{key} must be three numbers")
        out = tuple(float(v) for v in raw)
        if not all(v == v and abs(v) != float("inf") for v in out):
            raise LiveUnavailable(f"camera.{key} is not finite")
        return out

    fov = float(payload.get("fov_deg", 45.0))
    if not 1.0 <= fov <= 175.0:
        raise LiveUnavailable(f"camera.fov_deg out of range: {fov}")
    return CameraSpec(id="live", eye=vec3("eye"), target=vec3("target"),
                      up=vec3("up", (0.0, 0.0, 1.0)), fov_deg=fov)


def _clamp_size(payload: dict) -> tuple[int, int]:
    """Bound the frame **without changing its aspect ratio**.

    Clamping the two sides independently is the tempting version and it is
    wrong: the client stretches the frame to fill the viewport, so a request
    from a display taller than `MAX_HEIGHT` would come back squashed rather
    than merely smaller. The requested aspect is the one thing here that must
    survive, because it is half of what makes the streamed frame land on the
    viewport without a framing jump (vertical FOV is the other half).
    """
    width = int(payload.get("width", 640))
    height = int(payload.get("height", 360))
    if width < 1 or height < 1:
        return (MIN_SIDE, MIN_SIDE)
    scale = min(1.0, MAX_WIDTH / width, MAX_HEIGHT / height)
    # ...and back up if that, or the request itself, is below the floor.
    scale = max(scale, MIN_SIDE / width, MIN_SIDE / height)
    return (max(MIN_SIDE, round(width * scale)),
            max(MIN_SIDE, round(height * scale)))


def _poses_from(payload: dict, target: LiveTarget) -> dict | None:
    """The client's joint map, keyed the way `RenderRequest.poses` wants.

    Unknown names are dropped rather than passed through: the URDF carries
    synthetic placement joints (`world__<instance>`) that are fixed and have no
    IR counterpart, and forwarding one would make the kernel resolve a joint
    that does not exist.
    """
    raw = payload.get("joints")
    if not isinstance(raw, dict) or not raw:
        return None
    poses = {}
    for name, value in raw.items():
        key = target.joint_keys.get(str(name))
        if key is None:
            continue
        try:
            poses[key] = float(value)
        except (TypeError, ValueError):
            continue
    return poses or None


def encode_frame(header: dict, jpeg: bytes) -> bytes:
    """One WebSocket binary message: `[u32 header length][JSON][JPEG]`.

    A single message rather than a JSON text frame followed by a binary one, so
    the client never has to pair two deliveries — and so a dropped or reordered
    delivery cannot attach a header to the wrong picture.
    """
    blob = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(blob)) + blob + jpeg


async def serve(websocket, run_dir: Path, *, session_factory=None) -> None:
    """Drive one browser's live viewport until it disconnects.

    The session is blocking and lives on a worker thread; this coroutine owns
    the socket and the single pending-state slot. The render loop and the
    receive loop run concurrently, which is what makes coalescing free: while a
    frame is in flight the receiver keeps overwriting `pending`, and the loop
    picks up whatever is newest when it comes back.
    """
    from ..render import RenderRequest, RenderUnavailable
    from ..render.interactive_session import InteractiveSession

    await websocket.accept()
    try:
        target = resolve_target(run_dir)
    except LiveUnavailable as exc:
        await websocket.send_text(json.dumps(
            {"type": "error", "fatal": True, "message": str(exc)}))
        await websocket.close()
        return

    try:
        first = json.loads(await websocket.receive_text())
    except Exception:       # noqa: BLE001 — a client that hangs up mid-hello
        await websocket.close()
        return
    if int(first.get("protocol", 0)) != PROTOCOL_VERSION:
        await websocket.send_text(json.dumps(
            {"type": "error", "fatal": True,
             "message": f"live protocol mismatch: server speaks "
                        f"{PROTOCOL_VERSION}, client sent "
                        f"{first.get('protocol')}"}))
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    tmp = tempfile.TemporaryDirectory(prefix="dycraft-live-viewer-")
    session = None
    try:
        try:
            camera = _camera_from(first.get("camera") or {})
            request = RenderRequest(
                layer=target.layer, ir=target.ir, catalog=target.catalog,
                target="all", out_dir=Path(tmp.name) / "renders", step=0,
                asset_base_dir=target.asset_base_dir, backend="blender",
                poses=_poses_from(first, target))
            factory = session_factory or InteractiveSession
            session = factory(request)
            info = await loop.run_in_executor(None, session.start, camera)
        except Exception as exc:      # noqa: BLE001
            # Every failure here is fatal *and* must be legible: a missing
            # Blender interpreter, a shader that would not compile, a GPU
            # already held by another session. The client's contract is to keep
            # the existing viewer rather than show an approximation, so it
            # needs to be told which.
            await websocket.send_text(json.dumps(
                {"type": "error", "fatal": True,
                 "message": f"{type(exc).__name__}: {exc}"}))
            await websocket.close()
            return

        await websocket.send_text(json.dumps(
            {"type": "ready", "protocol": PROTOCOL_VERSION,
             "layer": target.layer, "device": info.get("device"),
             "engine": info.get("engine"), "parts": info.get("parts"),
             "built": info.get("built"),
             "warnings": info.get("warnings") or []}))

        pending: dict | None = dict(first)
        wake = asyncio.Event()
        wake.set()
        closed = asyncio.Event()

        async def receive() -> None:
            nonlocal pending
            try:
                while True:
                    message = json.loads(await websocket.receive_text())
                    if message.get("type") == "close":
                        break
                    # **Overwritten, never appended.** This single assignment
                    # is the whole coalescing policy.
                    pending = message
                    wake.set()
            except Exception:       # noqa: BLE001 — a disconnect is normal
                pass
            finally:
                closed.set()
                wake.set()

        receiver = asyncio.create_task(receive())
        try:
            while not closed.is_set():
                await wake.wait()
                wake.clear()
                state, pending = pending, None
                if state is None:
                    continue
                try:
                    camera = _camera_from(state.get("camera") or {})
                    width, height = _clamp_size(state)
                    poses = _poses_from(state, target)
                except LiveUnavailable as exc:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "fatal": False,
                         "seq": state.get("seq"), "message": str(exc)}))
                    continue
                try:
                    frame = await loop.run_in_executor(
                        None,
                        lambda: session.frame(
                            camera, width=width, height=height, poses=poses,
                            seq=int(state.get("seq", 0))))
                except RenderUnavailable as exc:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "fatal": not session.alive,
                         "message": str(exc)}))
                    if not session.alive:
                        break
                    continue
                if closed.is_set():
                    break
                await websocket.send_bytes(encode_frame({
                    "type": "frame", "seq": frame.seq,
                    "tier": state.get("tier", "interactive"),
                    "width": frame.width, "height": frame.height,
                    "render_ms": round(frame.render_ms, 2),
                    "warnings": frame.warnings,
                }, frame.jpeg))
        finally:
            receiver.cancel()
    finally:
        # Shielded: Starlette's test client cancels this task the moment the
        # client leaves, and a server shutdown can too. An unshielded await
        # here would drop the wait on the Blender child's quit and skip the
        # cleanup. `close` is bounded by its own timeouts, so the shield
        # cannot hang a shutdown.
        with anyio.CancelScope(shield=True):
            if session is not None:
                await loop.run_in_executor(None, session.close)
            tmp.cleanup()
        try:
            await websocket.close()
        except Exception:       # noqa: BLE001 — already gone
            pass
