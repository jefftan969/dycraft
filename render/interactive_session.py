"""A resident Blender child that answers frame requests from live viewer state.

This is the viewer's live Cycles viewport. The viewer draws shader parts in flat colors by default; the baked atlas
assets (`render/viewer_bake.py`) and this exact stream are opt-in. `Material(shader=...)` makes a Cycles node graph the
sole authority for appearance, and three.js cannot evaluate one; a tileable bake would misplace a graph driven
from object-space coordinates, which are uncorrelated with a part's UVs (two glaze bands at two heights on a vase
would repeat every meter). The atlas bake avoids that by sampling the graph at each surface point; this module
renders the graph itself.

It is fast enough because of one setting:

    scene.render.use_persistent_data = True

Without it Cycles re-exports the scene, rebuilds the BVH, and re-uploads device buffers on every
`bpy.ops.render.render()`, hundreds of milliseconds before a sample; with it, under a millisecond. On a recent GPU
and a 16-part object:

    to first pixel (spawn + bpy + scene build + cold BVH)   1.52 s
    steady state, 640x360, 32 spp, denoised, JPEG            ~32 ms   (31 fps)
    steady state, 1280x720                                   ~115 ms  (8.7 fps)
    re-posing ALL 16 parts, 640x360                          ~48 ms   (21 fps)

Rigid re-posing rewrites instance transforms only, so per-object BVHs stay resident and articulating everything
costs about as much as articulating one part, which is what makes Preview Motion streamable.

**Nothing here is a second renderer.** The scene is built by the same `_blender_child` functions a canonical
still uses, from a job packed by the same `_part_spec` / `_box_spec`, so a part's material comes from the one
`_build_material` / `_run_shader`. The differences are a one-time build, JPEG instead of PNG, and resolution as a
parameter. `blender_renderer.render()` does not use this module, and it writes nothing into the run or the render
cache: a render's PNG path (`_world.plan_outputs`) omits resolution, so a 640x360 frame written there would replace
the 1280x720 canonical still. Nothing in the IR, the staged assets, or the URDF changes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import CameraSpec, RenderRequest, RenderUnavailable
from .. import _oom
from ._blender_env import blender_child_env, blender_python
from ._world import (_build_world_geometry, _prepare, room_cutaway_planes,
                     room_cutaway_slots, room_wall_boxes)

_CHILD = Path(__file__).resolve().parent / "_blender_child.py"

#: Seconds to wait for the child's scene build. Generous: a small object builds
#: in seconds, but a scene run imports every instance's meshes.
START_TIMEOUT_S = float(os.environ.get("DYCRAFT_LIVE_START_TIMEOUT_S", "180"))
#: Seconds to wait for one frame. A 1280x720 frame is ~115 ms; this is a
#: liveness bound, not a budget.
FRAME_TIMEOUT_S = float(os.environ.get("DYCRAFT_LIVE_FRAME_TIMEOUT_S", "60"))

#: Cycles samples for interactive frames. Not a quality reduction: with
#: `use_adaptive_sampling` at the production `adaptive_threshold=0.01`, 1 spp
#: and 128 spp cost the same within noise on these scenes (8.1-13.3 ms at
#: 640x360), so the cap is never what the renderer is spending its time on.
#: 32 is the point past which nothing observably changes.
SAMPLES = int(os.environ.get("DYCRAFT_LIVE_SAMPLES", "32"))
JPEG_QUALITY = int(os.environ.get("DYCRAFT_LIVE_JPEG_QUALITY", "85"))


@dataclass
class InteractiveFrame:
    """One rendered frame, as the bytes a browser can display directly."""

    seq: int
    jpeg: bytes
    width: int
    height: int
    render_ms: float
    warnings: list[str] = field(default_factory=list)


class InteractiveSession:
    """One resident Blender child, holding one scene at one joint pose.

    Not thread-safe and deliberately not internally queued: `frame()` is a
    blocking request/response over a pipe, and *coalescing is the caller's job*
    because only the caller knows which state is current. See
    `viewer/live.py`, which drops every superseded request rather than
    building a backlog.
    """

    def __init__(self, request: RenderRequest, *,
                 samples: int = SAMPLES,
                 jpeg_quality: int = JPEG_QUALITY,
                 device: str | None = None,
                 engine: str = "CYCLES",
                 background: tuple[float, float, float] = (0.95, 0.95, 0.95)):
        self._request = request
        self._samples = int(samples)
        self._jpeg_quality = int(jpeg_quality)
        self._device = device or os.environ.get("DYCRAFT_BLENDER_DEVICE",
                                                "OPTIX")
        self._engine = engine
        self._background = background

        self._proc: subprocess.Popen | None = None
        self._req = None            # writable text stream to the child
        self._rep_fd: int | None = None   # raw reply descriptor
        self._rbuf = bytearray()         # see `_fill`
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._scratch: Path | None = None
        self._stderr_path: Path | None = None

        self._part_names: list[str] = []
        self._room_info = None
        self._pose_key: object = _UNSET
        self._wall_cache: dict[tuple, list] = {}
        self._wall_key: object = _UNSET
        self.info: dict = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self, camera: CameraSpec) -> dict:
        """Spawn the child and build the scene. Returns its ready report."""
        if self._proc is not None:
            raise RuntimeError("session already started")
        interp = blender_python()
        if not interp:
            raise RenderUnavailable(
                "no Blender interpreter: set $DYCRAFT_BLENDER_PYTHON or write "
                "one to .blender_python (see setup_blender.sh)")

        self._tmp = tempfile.TemporaryDirectory(prefix="dycraft-live-")
        self._scratch = Path(self._tmp.name)
        job = self._build_job(camera)
        job_path = self._scratch / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")

        # Dedicated descriptors for the protocol. **Cycles writes render
        # progress to stdout**, one line per frame, which would interleave with
        # a length-prefixed reply channel and desynchronize it permanently.
        req_r, req_w = os.pipe()
        rep_r, rep_w = os.pipe()
        self._stderr_path = self._scratch / "child.stderr"
        try:
            # A file rather than a pipe, and closed here rather than held:
            # nobody drains the child's output, and Cycles is chatty enough
            # that a full pipe would deadlock the render loop. `Popen` dups the
            # descriptor, so our copy has no further use.
            with open(self._stderr_path, "wb") as errors:
                self._proc = subprocess.Popen(
                    _oom.tiered([interp, str(_CHILD), "--serve", str(job_path),
                                 str(req_r), str(rep_w)], _oom.RESIDENT),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                    pass_fds=(req_r, rep_w),
                    env=blender_child_env())
        except OSError as e:
            for fd in (req_r, req_w, rep_r, rep_w):
                _close(fd)
            self._cleanup()
            raise RenderUnavailable(
                f"blender child could not start: {e}") from e
        # The child's ends belong to the child now; holding them open here
        # would mean our own reader never sees EOF when it exits.
        _close(req_r)
        _close(rep_w)
        self._req = os.fdopen(req_w, "w", encoding="utf-8")
        self._rep_fd = rep_r
        self._rbuf.clear()

        header, _ = self._read_reply(START_TIMEOUT_S, what="scene build")
        if not header.get("ok"):
            self.close()
            raise RenderUnavailable(
                "blender interactive session failed to build the scene: "
                + str(header.get("error") or "no reason given"))
        self.info = header
        return header

    def close(self) -> None:
        if self._req is not None:
            try:
                self._req.write(json.dumps({"op": "quit"}) + "\n")
                self._req.flush()
            except OSError:
                pass
            try:
                self._req.close()
            except OSError:
                pass
            self._req = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
            self._proc = None
        if self._rep_fd is not None:
            _close(self._rep_fd)
            self._rep_fd = None
        self._cleanup()

    def _cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
            self._scratch = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- rendering ---------------------------------------------------------

    def frame(self, camera: CameraSpec, *, width: int, height: int,
              poses: dict | None = None, samples: int | None = None,
              seq: int = 0) -> InteractiveFrame:
        """Render the given camera and joint pose. Blocks until the frame lands.

        Only the deltas travel. A camera-only move sends four vectors; a pose
        change additionally sends one SE3 per part, recomputed here by the same
        `_build_world_geometry` the canonical path uses, so the forward
        kinematics cannot diverge between the two.
        """
        if not self.alive:
            raise RenderUnavailable(
                "blender interactive session is not running"
                + self._stderr_tail())

        req = {"op": "frame", "seq": int(seq),
               "width": int(width), "height": int(height),
               "samples": int(self._samples if samples is None else samples),
               "camera": _camera_json(camera)}

        pose_key = _pose_key(poses)
        if pose_key != self._pose_key:
            req["se3"] = self._world_se3(poses)
            self._pose_key = pose_key

        walls = self._walls_for(camera)
        if walls is not None:
            req["walls"] = walls
        # Cheap and unconditional: an orbit crosses an authored wall's plane
        # exactly as it crosses a shell wall's, and the wall cache's early
        # return means a diffed send would sometimes never happen at all.
        req["cutaway_hidden"] = self._cutaway_for(camera)

        started = time.time()
        try:
            self._req.write(json.dumps(req) + "\n")
            self._req.flush()
        except OSError as e:
            raise RenderUnavailable(
                f"blender interactive session closed its pipe: {e}"
                + self._stderr_tail()) from e

        header, payload = self._read_reply(FRAME_TIMEOUT_S, what="frame")
        if not header.get("ok"):
            raise RenderUnavailable(
                "blender interactive frame failed: "
                + str(header.get("error") or "no reason given"))
        return InteractiveFrame(
            seq=int(header.get("seq", seq)), jpeg=payload,
            width=int(header.get("width", width)),
            height=int(header.get("height", height)),
            render_ms=(time.time() - started) * 1000.0,
            warnings=list(header.get("warnings") or []))

    # -- job assembly ------------------------------------------------------

    def _build_job(self, camera: CameraSpec) -> dict:
        """The one-frame job the child builds its resident scene from.

        Geometry and UVs are packed by `_part_spec`, as for a still. The frame
        carries no lighting keys, so the child lights it with its automatic rig.
        """
        from .blender_renderer import _part_spec

        request = dataclasses.replace(self._request, cameras=(camera,),
                                      camera_ids=None)
        plan = _prepare(request)
        self._room_info = plan.room_info
        self._part_names = [p.name for p in plan.world_parts]
        self._pose_key = _pose_key(self._request.poses)

        cutaway_slots = room_cutaway_slots(plan.room_info)
        parts = [_part_spec(p, self._scratch, f"p{i}", cutaway_slots)
                 for i, p in enumerate(plan.world_parts)]

        lo, hi = plan.target_bbox
        frame = {
            "parts": parts,
            "cameras": [{**_camera_json(camera), "id": camera.id,
                         "walls": self._walls_for(camera) or [],
                         "cutaway_hidden": self._cutaway_for(camera)}],
            "framing": [[float(v) for v in lo], [float(v) for v in hi]],
        }
        if plan.room_info is not None:
            rlo, rhi = plan.room_info["aabb"]
            frame["room"] = [[float(v) for v in rlo], [float(v) for v in rhi]]

        return {
            "engine": self._engine,
            "device": self._device,
            "samples": self._samples,
            "width": 640, "height": 360,
            "background": list(self._background),
            "jpeg_quality": self._jpeg_quality,
            "stream_dir": str(self._scratch),
            "frames": [frame],
        }

    def _world_se3(self, poses: dict | None) -> list[dict | None]:
        """Per-part world SE3 at `poses`, aligned with the child's part list."""
        from .blender_renderer import _se3

        request = dataclasses.replace(self._request, poses=poses)
        world_parts, _ = _build_world_geometry(request)
        names = [p.name for p in world_parts]
        if names != self._part_names:
            # The child indexes parts by position. A reordered or resized list
            # would silently place every part after the first difference on
            # another part's transform, which is a wrong picture rather than an
            # error — so it is one.
            raise RenderUnavailable(
                "part list changed shape between poses "
                f"({len(names)} vs {len(self._part_names)}); the interactive "
                "session cannot apply that incrementally")
        return [_se3(p.se3) for p in world_parts]

    def _cutaway_for(self, camera: CameraSpec) -> list:
        """Which authored room surfaces this camera must see through.

        The authored counterpart of `_walls_for`'s `camera_visible`.
        """
        if self._room_info is None:
            return []
        return sorted({h["tag"]
                       for hs in room_cutaway_planes(
                           camera, self._room_info).values()
                       for h in hs})

    def _walls_for(self, camera: CameraSpec) -> list | None:
        """Wall specs for this camera, or None when the child's are still right.

        Which room surfaces exist is a per-camera answer, so a free orbit can
        change the set — but only between a handful of discrete outcomes, so
        the packed specs (and the `.npy` files they name) are cached by
        signature instead of rewritten every frame.
        """
        if self._room_info is None:
            return None
        from .blender_renderer import _box_spec, _se3

        boxes = room_wall_boxes(camera, self._room_info)
        # `camera_visible` is in the key because it changes with the camera:
        # orbiting past a wall's plane flips it, and a cache keyed only on
        # geometry would keep serving the previous camera's answer.
        key = tuple((w.tag, tuple(w.size), _se3_tuple(w.se3), w.camera_visible)
                    for w in boxes)
        if key == self._wall_key:
            return None
        specs = self._wall_cache.get(key)
        if specs is None:
            specs = []
            for wi, w in enumerate(boxes):
                spec = {"tag": w.tag, "se3": _se3(w.se3),
                        "material": w.material, "alpha": w.alpha,
                        "camera_visible": w.camera_visible}
                specs.append(_box_spec(spec, w.size, (0.0, 0.0, 0.0),
                                       w.material, self._scratch,
                                       f"w{len(self._wall_cache)}_{wi}"))
            self._wall_cache[key] = specs
        self._wall_key = key
        return specs

    # -- protocol ----------------------------------------------------------
    #
    # **Raw `os.read` with our own buffer, not a file object.** A buffered
    # reader is unusable here: its first read pulls everything available into
    # the buffer, after which `select()` on the descriptor reports "no data"
    # while the bytes we want are sitting in userspace. That deadlocks on the
    # very first reply, which is exactly how this was found. So the buffer is
    # ours, and `select()` is consulted only when it is empty.

    def _read_reply(self, timeout_s: float, *, what: str):
        """One JSON header line plus its length-prefixed payload."""
        deadline = time.time() + timeout_s
        header = json.loads(self._read_line(deadline, what))
        payload = self._read_exact(int(header.get("bytes", 0)), deadline, what)
        return header, payload

    def _read_line(self, deadline: float, what: str) -> str:
        while True:
            nl = self._rbuf.find(b"\n")
            if nl >= 0:
                line = bytes(self._rbuf[:nl])
                del self._rbuf[:nl + 1]
                return line.decode("utf-8")
            self._fill(deadline, what)

    def _read_exact(self, n: int, deadline: float, what: str) -> bytes:
        while len(self._rbuf) < n:
            self._fill(deadline, what)
        out = bytes(self._rbuf[:n])
        del self._rbuf[:n]
        return out

    def _fill(self, deadline: float, what: str) -> None:
        """Block until the child says something, or explain why it never will."""
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RenderUnavailable(
                    f"blender interactive session timed out waiting for {what}"
                    + self._stderr_tail())
            ready, _, _ = select.select([self._rep_fd], [], [],
                                        min(remaining, 1.0))
            if ready:
                chunk = os.read(self._rep_fd, 1 << 16)
                if not chunk:
                    raise RenderUnavailable(
                        f"blender interactive session closed its reply pipe "
                        f"during {what}" + self._stderr_tail())
                self._rbuf += chunk
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RenderUnavailable(
                    f"blender interactive session exited "
                    f"({self._proc.returncode}) during {what}"
                    + self._stderr_tail())

    def _stderr_tail(self, lines: int = 12) -> str:
        """The child's last words. It is the only evidence a crash leaves."""
        if self._stderr_path is None or not self._stderr_path.exists():
            return ""
        try:
            text = self._stderr_path.read_text(errors="replace")
        except OSError:
            return ""
        tail = [ln for ln in text.strip().splitlines() if ln.strip()][-lines:]
        return ("\n  " + "\n  ".join(tail)) if tail else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Unset:
    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()

def _camera_json(camera: CameraSpec) -> dict:
    return {"eye": [float(v) for v in camera.eye],
            "target": [float(v) for v in camera.target],
            "up": [float(v) for v in camera.up],
            "fov_deg": float(camera.fov_deg)}


def _se3_tuple(se3) -> tuple:
    t, q = se3
    return (tuple(float(v) for v in t), tuple(float(v) for v in q))


def _pose_key(poses: dict | None):
    """A comparable, hashable digest of a joint pose."""
    if poses is None:
        return None
    return tuple(sorted((str(k), float(v)) for k, v in poses.items()))


def _close(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
