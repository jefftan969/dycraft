"""FastAPI app: JSON API over `runs/` plus the built SPA on one port.

Route surface. Read-only, with one duplex exception (`/live`):

    GET /api/runs                        run summaries, ordered by path
    GET /api/runs/{id}                   meta + IR summary + compile reports
                                         + sub_runs (the runs it started)
    GET /api/runs/{id}/urdf              URDF generated from ir.json
                                         (from scene.json on an animation run)
    GET /api/runs/{id}/animation         per-frame timeline; animation runs only
    GET /api/runs/{id}/native-track      whole-clip motion; native bundles only
    GET /api/runs/{id}/native-mesh/{d}   one evaluated mesh, by content digest
    GET /api/runs/{id}/native-positions/{d}  its vertices only, and its topology hash
    GET /api/runs/{id}/native-atlas      per-object baked atlas UVs, so a
                                         native clip can wear its real maps
                                         (plus `rigs`: the deformation control
                                          table, when the run has any)
    GET /api/runs/{id}/cutaway           a native film's walls and ceilings,
                                         which the free camera cuts away
    WS  /api/runs/{id}/live              interactive Cycles frames, driven
                                         by the viewport's own camera/pose
    GET /api/runs/{id}/derived           baked-atlas viewer asset status, one
                                         per object; starts absent bakes
    GET /api/runs/{id}/trajectory        normalized LLM turns + token totals
    GET /api/runs/{id}/agent_timeline    this run's own actions, in lanes
    GET /api/runs/{id}/mesh-bundle       how many tar chunks this run's meshes
                                         pack into, and how big they are
    GET /api/runs/{id}/mesh-bundle/{n}   chunk n, as a tar of run-relative
                                         mesh paths (see viewer/mesh_bundle.py)
    GET /api/runs/{id}/files/{path}      any file inside the run dir
    GET /api/runs/{id}/text/{path}       same, as JSON text (for the code panel)
    GET /api/config                      the viewer's settings: the native clip's memory budget
    GET /health

`{id}` is a run's path **relative to `runs/`**, not a bare directory name:
`runs/` is discovered recursively, so a run filed under a folder is addressed as
`achieve/kettle`. For a run sitting directly in `runs/` the two are the same.

`files/` is the base URL the three.js loader resolves `<mesh filename>` against,
so `assets/meshes/<digest>.obj` lands here.

The SPA-serving pattern (index + catch-all with an `api/` guard) follows
Articraft's `viewer/api/frontend.py`.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse, Response

from . import derived as derived_mod
from . import films as films_mod
from . import live as live_mod
from . import mesh_bundle as mesh_bundle_mod
from . import runs as runs_mod
from ._payload_pass import PayloadTimeout, cached_payload

# Vite's dev server, for `--dev` where the SPA is served by vite and the API here.
_DEV_ORIGINS = [
    "http://127.0.0.1:5173", "http://localhost:5173",
    "http://127.0.0.1:8770", "http://localhost:8770",
]

# Text files the code panel may request. Keeping this narrow stops the endpoint
# from becoming a way to slurp arbitrary bytes as JSON.
_TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".urdf", ".txt", ".xml"}
_MAX_TEXT_BYTES = 4 * 1024 * 1024

# The filename the vendored three.js loader appends to its base URL.
_MODEL_URDF_NAME = "model.urdf"

#: The env var, and the default, for the most built native-clip geometry the page keeps, in MB.
CLIP_BUDGET_ENV = "DYCRAFT_VIEWER_CLIP_BUDGET_MB"
DEFAULT_CLIP_BUDGET_MB = 1024

# Asset directories whose filenames are content digests. `sdk/_mesh_assets.py`
# names every file in these after a hash of its own bytes, so the URL cannot
# outlive the content it points at and revalidating one is pure cost.
_CONTENT_ADDRESSED_DIRS = frozenset(
    {"meshes", "collision", "brep", "textures", "src"})
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"


def _is_content_addressed(file_path: str) -> bool:
    """Whether `file_path` names an immutable, content-addressed run asset.

    Both halves have to hold: the file sits directly under one of the digest
    directories of an `assets/` tree — a scene run's catalog nests a second one
    at `catalog/assets/...` — and its stem is a hash rather than a name. The
    second test is what keeps a hand-dropped `assets/meshes/notes.obj` on
    `no-cache`, and it is cheap because a wrong answer here is not a stale
    render but a permanently stale one: nothing busts a year-long cache.

    Everything else stays `no-cache`, and must. `ir.json`, the compile reports,
    and the renders are rewritten in place while a run is live, under URLs that
    do not change.
    """
    parts = PurePosixPath(file_path).parts
    if len(parts) < 3 or parts[-3] != "assets":
        return False
    if parts[-2] not in _CONTENT_ADDRESSED_DIRS:
        return False
    stem = PurePosixPath(parts[-1]).stem
    return len(stem) == 32 and all(c in "0123456789abcdef" for c in stem)


#: Route segments whose responses are run *assets*: meshes, baked GLBs,
#: textures, renders, native motion, and meshes. Never compressed — see
#: `_SelectiveGZip`.
_UNCOMPRESSED_SEGMENTS = ("/files/", "/native-track",
                          "/native-mesh/", "/native-positions/", "/native-atlas", "/mesh-bundle")


def _skips_compression(path: str) -> bool:
    return any(segment in path for segment in _UNCOMPRESSED_SEGMENTS)


def _gzip_level_1(request: Request, data: bytes) -> tuple[bytes, dict[str, str]]:
    """`data` gzipped at level 1 when the browser takes gzip, with the headers that say so; else as it is.

    For the assets that compress well though the middleware passes them through: the mesh bundle's chunks (OBJ text,
    3.6 times smaller) and whole native meshes (their triangle indices, 2.5 times). Called in the request's worker
    thread, never on the event loop, and at level 1, about a tenth of a second for a 16 MB chunk, so loading on the
    same machine pays next to nothing for what a remote browser saves. A browser on another machine reaches the
    viewer through a tunnel to the same loopback address, so the two cannot be told apart and both get it.
    """
    if "gzip" not in request.headers.get("accept-encoding", ""):
        return data, {}
    return gzip.compress(data, compresslevel=1, mtime=0), {"Content-Encoding": "gzip", "Vary": "Accept-Encoding"}


class _SelectiveGZip:
    """GZip the JSON and XML, never the assets — and cheaply either way.

    `GZipMiddleware` applied app-wide, at Starlette's default `compresslevel=9`,
    would be the largest cost of loading a scene: every mesh would go through it
    on the event loop, at about thirty times level 1's CPU to save a few percent
    of the size, and since that compression is synchronous on the one
    event-loop thread it serializes the *whole* API behind it, long enough for
    the browser to give up on a big scene's mesh requests.

    So assets are passed through untouched here. Those that compress well,
    the mesh bundle's OBJ chunks, native meshes, and the native track, are
    compressed by their own routes off the event loop (`_gzip_level_1`,
    `track_gzip`). Everything else — run summaries, IR, trajectories, the
    generated URDF, the SPA bundle — still compresses here, at level 1, a
    quarter of level 9's time for a slightly larger result.
    """

    def __init__(self, app, minimum_size: int = 1024,
                 compresslevel: int = 1) -> None:
        self._plain = app
        self._gzip = GZipMiddleware(app, minimum_size=minimum_size,
                                    compresslevel=compresslevel)

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http" and _skips_compression(
                str(scope.get("path") or "")):
            await self._plain(scope, receive, send)
            return
        await self._gzip(scope, receive, send)


def _resolve_runs_dir(runs_dir: Path | None) -> Path:
    if runs_dir is not None:
        return Path(runs_dir).resolve()
    configured = os.getenv("DYCRAFT_RUNS_DIR")
    if configured:
        return Path(configured).resolve()
    # The CLI's own default: runs/ under the user-state root.
    from .. import _home
    return _home.default_runs_dir()


def create_app(*, runs_dir: Path | None = None,
               dist_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="dycraft Viewer API")
    resolved_runs = _resolve_runs_dir(runs_dir)
    app.state.runs_dir = resolved_runs

    app.add_middleware(
        CORSMiddleware, allow_origins=_DEV_ORIGINS, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.add_middleware(_SelectiveGZip, minimum_size=1024)

    def _run_dir(run_id: str) -> Path:
        run_dir = runs_mod.resolve_run(resolved_runs, run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return run_dir

    #: `DYCRAFT_VIEWER_CLIP_BUDGET_MB`, default 1024.
    try:
        clip_budget_mb = max(64, int(os.environ.get(CLIP_BUDGET_ENV, "") or DEFAULT_CLIP_BUDGET_MB))
    except ValueError:
        clip_budget_mb = DEFAULT_CLIP_BUDGET_MB

    @app.get("/health", include_in_schema=False)
    async def health() -> dict:
        return {"ok": True, "runs_dir": str(resolved_runs)}

    @app.get("/api/config")
    async def api_config() -> dict:
        # The most built native-clip geometry the page keeps (`native-stream.ts`). A long film's deforming objects
        # carry a mesh per frame, so the page loads them around the playhead within this; a machine with memory to
        # spare can raise it.
        return {"clip_budget_mb": clip_budget_mb}

    # Every handler below that reads the filesystem or parses a transcript
    # runs in a threadpool, and has to.
    #
    # These bodies are synchronous and some of them are slow — `list_runs`
    # walks the whole tree (2.9 s over a 1828-run `runs/` served from a parent
    # directory), `run_urdf` on a 5403-mesh scene is 1.35 s across 1658
    # instances, and a trajectory means parsing a multi-MB rollout. Called
    # directly from `async def` they hold the event loop for that whole time,
    # which stalls *every other request* — including the several hundred mesh
    # fetches the viewport has in flight while a scene is loading, which is how
    # a slow scene turned into "failed to fetch" in the browser rather than
    # into a slow scene.
    @app.get("/api/runs")
    async def api_runs() -> dict:
        runs = await run_in_threadpool(runs_mod.list_runs, resolved_runs)
        return {"runs_dir": str(resolved_runs), "runs": runs}

    #: Generated URDFs, keyed by what they are generated *from*. Small and
    #: bounded: the viewport asks for one document per run, but it asks for it
    #: again on every re-selection and once more when the baked variant lands,
    #: and generating a large scene's document costs about a second each time. The key carries `ir.json`'s size and mtime, so a
    #: live run rewriting its IR invalidates itself rather than serving a
    #: document that no longer matches the meshes beside it.
    urdf_cache: dict[tuple, tuple[str, str]] = {}
    #: Mesh-bundle packings, under the same key — one `stat` per mesh, which is
    #: 222 ms on a 5403-mesh scene, and the manifest request plus every
    #: chunk request all need the same answer.
    bundle_cache: dict[tuple, list] = {}

    #: Everything `run_urdf` reads to build a document. An object run has the
    #: first; a scene adds the catalog it draws its geometry out of; an
    #: animation is generated from the staged scene instead.
    urdf_sources = ("ir.json", "catalog/manifest.json", "scene.json")

    def _urdf_cache_key(run_dir: Path, derived: bool) -> tuple | None:
        stamps = []
        for name in urdf_sources:
            try:
                stat = (run_dir / name).stat()
            except OSError:
                continue
            stamps.append((name, stat.st_mtime_ns, stat.st_size))
        if not stamps:
            return None
        # A finished shader color pass changes what the document draws, so it starts a new key.
        return (str(run_dir), derived, tuple(stamps), derived_mod.flat_generation())

    def _urdf_with_flat(run_dir: Path, derived: bool = False) -> tuple[str, str]:
        """The document, and the status of its shader parts' flat colors (`derived.flat_colors`), which the
        viewport reads from `X-Dycraft-Flat-Colors` to fetch the document again once they are read."""
        key = _urdf_cache_key(run_dir, derived)
        if key is not None:
            cached = urdf_cache.get(key)
            if cached is not None:
                return cached
        try:
            flat_status, flat = derived_mod.flat_colors(run_dir)
            entry = (runs_mod.run_urdf(run_dir, derived=derived, flat=flat), flat_status)
            if key is not None:
                # Two entries per run (preview and derived) and a handful of
                # runs looked at in a session; dropped whole rather than by
                # age, because the cost of a miss is one regeneration.
                if len(urdf_cache) >= 16:
                    urdf_cache.clear()
                urdf_cache[key] = entry
            return entry
        except runs_mod.UrdfUnavailable as exc:
            # The reason is layer-dependent (no IR, no catalog, unknown IR
            # shape); `UrdfUnavailable` carries it so the viewport can say
            # which rather than always blaming a missing ir.json.
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _urdf(run_dir: Path, derived: bool = False) -> str:
        return _urdf_with_flat(run_dir, derived)[0]

    # `run_id` is a runs-dir-relative *path* (`achieve/kettle`), so it has to be
    # a `:path` param. Those are greedy, so the suffixed routes below must be
    # registered BEFORE the bare `/api/runs/{run_id:path}` or it would swallow
    # `/urdf` and friends as part of the id. Registration order is match order.
    #
    # The one residual ambiguity: `/files/{file_path:path}` splits at the *last*
    # `/files/` in the URL, so a run whose artifact tree contained its own
    # `files/` directory would mis-split. Nothing the harness writes does.
    @app.get("/api/runs/{run_id:path}/prepare")
    async def api_prepare(run_id: str, start: bool = True) -> dict:
        """Whether the run is ready to open, preparing an installed film the first time it is asked (`films`).

        A GET with a side effect, like `/derived`: it fills the viewer's own folder, never the film's. A run that
        needs no preparing is `{"state": "ready"}`.
        """
        film = runs_mod.resolve_film(resolved_runs, run_id)
        if film is None:
            _run_dir(run_id)
            return {"state": "ready"}
        return await run_in_threadpool(films_mod.status, film, start=start)

    @app.get("/api/runs/{run_id:path}/urdf", response_class=PlainTextResponse)
    async def api_urdf(run_id: str, derived: bool = False) -> str:
        return await run_in_threadpool(_urdf, _run_dir(run_id), derived)

    @app.get("/api/runs/{run_id:path}/derived")
    async def api_derived(run_id: str, build: bool = True) -> dict:
        """Status of this run's baked-atlas viewer assets, building on demand.

        A GET with a side effect, for the same reason `/live` spawns a Blender
        child: it fills a content-addressed cache rather than mutating anything
        the user owns, and the browser needs one round trip to both ask and
        start. `build=0` makes it a pure query.

        **Aggregate.** An object run needs one asset; a scene run needs one per
        catalog object it instances, and `ready` means all of them — with
        `objects` / `objects_ready` carrying the progress, and `partial` for
        "some baked, the rest are not coming".

        Never raises for an unsupported run — an animation run gets
        `status: "unavailable"` with the reason, and the viewport keeps its
        existing appearance rather than showing an approximation of a shader.
        """
        return await run_in_threadpool(
            derived_mod.status, _run_dir(run_id), start=build)

    @app.get("/api/runs/{run_id:path}/animation")
    async def api_animation(run_id: str) -> dict:
        try:
            return await run_in_threadpool(
                runs_mod.run_animation, _run_dir(run_id))
        except runs_mod.UrdfUnavailable as exc:
            # Shared vocabulary with /urdf on purpose: the viewport asks for
            # both, and "this run has no staged scene" is the same fact.
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id:path}/native-track")
    async def api_native_track(run_id: str, request: Request) -> Response:
        """The clip's motion, once: object table plus per-frame matrices.

        `no-cache` rather than immutable, because the URL names a run and a
        live run recompiles its bundle under that same URL. The expensive part
        is cached server-side against the manifest id instead, which does move
        when the bundle does. Gzipped when the browser takes it (`track_gzip`),
        which is every browser.
        """
        from .native_animation import track, track_gzip
        packed = "gzip" in request.headers.get("accept-encoding", "")
        try:
            data = await run_in_threadpool(track_gzip if packed else track, _run_dir(run_id))
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        headers = {"Cache-Control": "no-cache", "Vary": "Accept-Encoding"}
        if packed:
            headers["Content-Encoding"] = "gzip"
        return Response(content=data, media_type="application/octet-stream", headers=headers)

    @app.get("/api/runs/{run_id:path}/native-atlas")
    async def api_native_atlas(run_id: str) -> Response:
        """Baked atlas UVs per object, pairing the bake to evaluated geometry.

        `no-cache` for the same reason as `/native-track`: the URL names a run,
        and both the bundle and the bake under it can move. An empty payload is
        a normal answer — see `native_animation.atlas`.
        """
        from .native_animation import atlas
        try:
            data = await run_in_threadpool(atlas, _run_dir(run_id))
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs/{run_id:path}/storyboard")
    async def api_storyboard(run_id: str) -> dict:
        """The film's storyboard keyframes and its first frame (`animation.storyboard`), which the timeline marks;
        no keyframes without a storyboard."""
        from .animation import storyboard
        return await run_in_threadpool(storyboard, _run_dir(run_id))

    @app.get("/api/runs/{run_id:path}/storyboard/{index}.jpg")
    async def api_storyboard_thumb(run_id: str, index: int) -> Response:
        """Keyframe `index`'s rendered view, small (`animation.storyboard_thumb`)."""
        from .animation import storyboard_thumb
        data = await run_in_threadpool(storyboard_thumb, _run_dir(run_id), index)
        if data is None:
            raise HTTPException(status_code=404, detail=f"no storyboard keyframe {index}")
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs/{run_id:path}/film-cameras")
    async def api_film_cameras(run_id: str) -> dict:
        """The film's camera at every frame (`animation.film_cameras`), which the film view follows as it plays.

        Asked for once the clip is on screen, apart from `/animation`, so a long film's opening does not wait on it.
        A film with no camera the viewer can reproduce has none: 404, and the film view stays on its opening shot.
        """
        from .animation import film_camera_track
        track = await run_in_threadpool(film_camera_track, _run_dir(run_id))
        if track is None:
            raise HTTPException(status_code=404, detail="this film has no camera the viewer can follow")
        return track

    @app.get("/api/runs/{run_id:path}/cutaway")
    async def api_cutaway(run_id: str) -> dict:
        """A native film's walls and ceilings, which the free camera cuts away when they are in its way.

        Worked out from the film's own geometry the first time it is asked for, and cached (`viewer/cutaway.py`).
        A run with no native clip has none: 404, and the viewport cuts nothing away.
        """
        from .cutaway import cutaway
        try:
            return await run_in_threadpool(cutaway, _run_dir(run_id))
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id:path}/native-positions/{digest}")
    async def api_native_positions(run_id: str, digest: str) -> Response:
        """One evaluated mesh's vertices and topology hash, for a deforming object's later frames. Immutable."""
        from .native_animation import mesh_positions
        try:
            data = await run_in_threadpool(mesh_positions, _run_dir(run_id), digest)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type="application/octet-stream",
                        headers={"Cache-Control": _IMMUTABLE_CACHE})

    @app.get("/api/runs/{run_id:path}/native-mesh/{digest}")
    async def api_native_mesh(run_id: str, digest: str, request: Request) -> Response:
        """One evaluated mesh. Immutable: the digest IS the content."""
        from .native_animation import mesh_blob
        run_dir = _run_dir(run_id)
        try:
            data, encoding = await run_in_threadpool(lambda: _gzip_level_1(request, mesh_blob(run_dir, digest)))
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type="application/octet-stream",
                        headers={"Cache-Control": _IMMUTABLE_CACHE, **encoding})

    @app.websocket("/api/runs/{run_id:path}/live")
    async def api_live(websocket: WebSocket, run_id: str) -> None:
        """Interactive Cycles frames for the viewport, from live viewer state.

        The one non-read-only-looking route here, and it still writes nothing:
        it holds a resident Blender child that renders the camera and joint
        pose the browser is currently showing. See `viewer/live.py` for the
        wire format and the coalescing rule, and
        `render/interactive_session.py` for why this is fast enough to be worth
        having.

        A run that cannot be resolved is reported *on the socket* rather than
        as an HTTP error, because the client's contract on failure is to keep
        the existing three.js viewport and say why — not to fall back to a
        material approximation that would look like the generated result and
        not be it.
        """
        run_dir = runs_mod.resolve_run(resolved_runs, run_id)
        if run_dir is None:
            await websocket.accept()
            await websocket.send_text(json.dumps(
                {"type": "error", "fatal": True,
                 "message": f"unknown run {run_id}"}))
            await websocket.close()
            return
        await live_mod.serve(websocket, run_dir)

    # Built in their own interpreter and served as stored JSON (`_payload_pass`): parsing a long run's transcript in
    # this process held the interpreter lock against everything else it serves, and a film's playback ran dry.
    @app.get("/api/runs/{run_id:path}/trajectory")
    async def api_trajectory(run_id: str) -> Response:
        return await _payload("trajectory", run_id)

    @app.get("/api/runs/{run_id:path}/agent_timeline")
    async def api_agent_timeline(run_id: str) -> Response:
        return await _payload("agent_timeline", run_id)

    async def _payload(kind: str, run_id: str) -> Response:
        try:
            data = await run_in_threadpool(cached_payload, kind, _run_dir(run_id))
        except PayloadTimeout as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=data, media_type="application/json")


    def _bundle_plan(run_dir: Path, derived: bool) -> list:
        key = _urdf_cache_key(run_dir, derived)
        if key is not None:
            cached = bundle_cache.get(key)
            if cached is not None:
                return cached
        plan = mesh_bundle_mod.bundle_plan(run_dir, _urdf(run_dir, derived))
        if key is not None:
            if len(bundle_cache) >= 16:
                bundle_cache.clear()
            bundle_cache[key] = plan
        return plan

    @app.get("/api/runs/{run_id:path}/mesh-bundle")
    async def api_mesh_bundle(run_id: str, derived: bool = False) -> dict:
        """The chunking of this run's meshes, so the viewport can fetch them
        in a handful of requests instead of one per visual."""
        run_dir = _run_dir(run_id)

        def plan() -> dict:
            return mesh_bundle_mod.manifest(_bundle_plan(run_dir, derived))

        return await run_in_threadpool(plan)

    @app.get("/api/runs/{run_id:path}/mesh-bundle/{index}")
    async def api_mesh_bundle_chunk(run_id: str, index: int, request: Request,
                                    derived: bool = False) -> Response:
        run_dir = _run_dir(run_id)

        def pack() -> tuple[bytes, dict[str, str]] | None:
            data = mesh_bundle_mod.chunk_tar(_bundle_plan(run_dir, derived), index)
            return None if data is None else _gzip_level_1(request, data)

        packed = await run_in_threadpool(pack)
        if packed is None:
            raise HTTPException(status_code=404,
                                detail=f"no mesh bundle chunk {index}")
        # Immutable under `?rev=<ir hash>`, which the client appends: the
        # entries are content-addressed meshes of one IR revision, so a URL
        # that includes the revision can never mean different bytes.
        data, encoding = packed
        return Response(content=data, media_type="application/x-tar",
                        headers={"Cache-Control": _IMMUTABLE_CACHE, **encoding})

    @app.get("/api/runs/{run_id:path}/files/{file_path:path}")
    async def api_file(run_id: str, file_path: str, derived: bool = False):
        run_dir = _run_dir(run_id)
        # The three.js loader is hardwired to `{baseFileUrl}/model.urdf` and
        # resolves every <mesh filename> against the same base. Serving the
        # generated URDF under that name lets `files/` be the one base URL, so
        # `assets/meshes/<digest>.obj` resolves without touching the loader.
        if file_path == _MODEL_URDF_NAME:
            # `?derived=1` selects the baked-atlas variant. The loader appends
            # a fixed `model.urdf` to one base URL, so a query parameter is the
            # only place the choice can travel without teaching it a second
            # path.
            document, flat_status = await run_in_threadpool(_urdf_with_flat, run_dir, derived)
            return PlainTextResponse(
                document, media_type="application/xml",
                headers={"Cache-Control": "no-cache", "X-Dycraft-Flat-Colors": flat_status})
        if file_path.startswith(runs_mod.DERIVED_URL_PREFIX):
            # Derived assets live in a cache OUTSIDE the run directory, but the
            # three.js loader resolves every `<mesh filename>` against this one
            # base — so they are addressed under a reserved prefix here, the
            # same trick `model.urdf` above uses.
            #
            # `__derived__/<ir_id>/<file>`: a scene draws several objects, each
            # with its own cache entry, and their manifests number files per
            # object — so the id segment is what disambiguates two `000_*.glb`.
            rest = file_path[len(runs_mod.DERIVED_URL_PREFIX):]
            ir_id, _, name = rest.partition("/")
            # **The cache is shared across every run on disk, so the id is
            # checked against what THIS run draws.** Without that, any run's URL
            # could name any other run's baked object.
            manifests = derived_mod.manifests_for(run_dir)
            if not name or ir_id not in manifests:
                raise HTTPException(status_code=404, detail=file_path)
            from ..render.viewer_bake import asset_path

            target = asset_path(ir_id, name)
            if target is None:
                raise HTTPException(status_code=404, detail=file_path)
            # Immutable: the cache key is the IR hash plus the bake version, so
            # these bytes can never change under this URL.
            return FileResponse(
                target, headers={"Cache-Control": "public, max-age=31536000, immutable"})
        target = runs_mod.resolve_file(run_dir, file_path)
        if target is None:
            raise HTTPException(status_code=404, detail=file_path)
        # A content-addressed asset is cached for a year; everything else is
        # revalidated. The scene runs are what forced the distinction: one
        # kitchen is 454 part meshes and 33 MB of OBJ, and on `no-cache` the
        # browser re-fetched and re-parsed all of it every time the sidebar
        # came back to that run.
        return FileResponse(target, headers={"Cache-Control": (
            _IMMUTABLE_CACHE if _is_content_addressed(file_path)
            else "no-cache")})

    @app.get("/api/runs/{run_id:path}/text/{file_path:path}")
    async def api_text(run_id: str, file_path: str) -> dict:
        target = runs_mod.resolve_file(_run_dir(run_id), file_path)
        if target is None:
            raise HTTPException(status_code=404, detail=file_path)
        if target.suffix.lower() not in _TEXT_SUFFIXES:
            raise HTTPException(status_code=415, detail=target.suffix)
        size = target.stat().st_size
        truncated = size > _MAX_TEXT_BYTES

        def read() -> str:
            with open(target, "r", errors="replace") as fh:
                return fh.read(_MAX_TEXT_BYTES)

        return {"path": file_path, "size": size, "truncated": truncated,
                "text": await run_in_threadpool(read)}

    # Registered last on purpose — see the note above the urdf route.
    @app.get("/api/runs/{run_id:path}")
    async def api_run(run_id: str) -> dict:
        return await run_in_threadpool(
            runs_mod.run_detail, _run_dir(run_id), run_id)

    _install_frontend(app, dist_dir)
    return app


def _install_frontend(app: FastAPI, dist_dir: Path | None) -> None:
    dist = (dist_dir or Path(__file__).resolve().parent / "web" / "dist")
    if not dist.exists():
        @app.get("/", include_in_schema=False)
        async def frontend_missing() -> dict:
            return {"message": (
                "Frontend build not found. Run: npm --prefix viewer/web ci && "
                "npm --prefix viewer/web run build")}
        return

    root = dist.resolve()

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(root / "index.html",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/") or full_path in {
                "health", "docs", "redoc", "openapi.json"}:
            raise HTTPException(status_code=404)
        requested = Path(full_path)
        if requested.is_absolute() or ".." in requested.parts:
            raise HTTPException(status_code=404)
        target = (root / requested).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404) from exc
        if target.is_file():
            # Vite fingerprints everything under assets/, so it can be cached
            # forever; index.html and the rest must not be.
            immutable = target.parts[-2:-1] == ("assets",)
            return FileResponse(target, headers={"Cache-Control": (
                "public, max-age=31536000, immutable" if immutable
                else "no-cache")})
        if requested.parts[:1] == ("assets",):
            raise HTTPException(status_code=404)
        return FileResponse(root / "index.html",
                            headers={"Cache-Control": "no-cache"})
