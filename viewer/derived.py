"""Derived viewer assets: when to bake one, and what to tell the browser.

The viewer's answer to `Material(shader=...)`. A Cycles node graph cannot be
evaluated in a browser, so `render/viewer_bake.py` samples it onto a packed
atlas and exports a GLB; this module decides which runs get one, runs the bake
off the request thread, and reports progress so the viewport can show the
existing preview until the real thing is ready.

**Read-only with respect to the run.** The bake reads `ir.json` and the staged
meshes and writes only into the cache under `render.viewer_bake.cache_root()`.
No IR key is added, so no object id moves; the staged assets, the canonical
URDF, the stored `renders/ir.urdf`, and the IK solver export are untouched.

**A run resolves to a LIST of bakes, not one.** An object run has a single IR
and a single entry; a scene run has one entry per distinct `object_ref` its
instances name, because a SceneIR draws its geometry out of the per-object
catalog rather than out of `ir["parts"]`. Everything downstream — the cache,
the URL namespace, the status payload — is keyed by the *object's* ir_id for
exactly that reason, and two scenes that share an object share its bake.

**Animation runs are included**: the bake happens on the REST shape of the
staged scene's parts.

There is one place this deliberately does not match the canonical render, and
it is a choice rather than a defect. A shader driving from **object space**
(`tc.outputs["Object"]`, `Generated`, `Position`) is evaluated by Cycles at the
*deformed* position, so its pattern swims through the surface as the part
bends; a baked map is pinned to the surface instead. Pinning is the intended
behavior — paint travels with the cloth it is on — so the viewer is knowingly
the more correct picture here. Measured divergence: 0.58-1.11/255.

One refusal remains, deliberate and stated rather than silent:

  * **Runs with no `Material(shader=...)`.** Not a failure: a scalar or
    textured material is already drawn correctly by `materials.ts`, so there is
    nothing to derive and a bake would only spend GPU reproducing it. This is
    also what keeps every existing plain-color and textured deformation
    animation on exactly the path it uses today.
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from . import live as live_mod

#: How many bakes run at once, process-wide. On a GPU, one: a bake saturates
#: the device, and two in parallel would make both slower without making either
#: finish sooner, so a scene's objects queue rather than storm it. On a CPU,
#: two: Cycles samples on every core, but a bake's set-up and its GLB export
#: run on one, and a second bake overlaps them: a scene's two largest objects
#: took 87.8 s one after the other and 59.9 s together on a busy 16-core
#: machine. Which device a bake used is only known once one has run, so the
#: first runs alone.
CPU_BAKES = 2
_SLOTS = threading.Condition()
_slots = {"running": 0, "limit": 1}


def _take_slot() -> None:
    with _SLOTS:
        while _slots["running"] >= _slots["limit"]:
            _SLOTS.wait()
        _slots["running"] += 1


def _give_slot(device: str | None) -> None:
    with _SLOTS:
        _slots["running"] -= 1
        if device == "CPU":
            _slots["limit"] = CPU_BAKES
        _SLOTS.notify_all()

#: `ir_id -> state`, and the lock that guards it. In-process only: the manifest
#: on disk is the durable record, and this exists so a browser polling every
#: second gets "building" rather than a second bake. Keyed by the object's id
#: rather than by the run, so two scenes sharing an object also share its
#: in-flight bake.
_STATE: dict[str, dict] = {}
_STATE_LOCK = threading.Lock()


class DerivedUnavailable(RuntimeError):
    """This run will not get a derived asset, with the reason."""


@dataclass(frozen=True)
class DerivedEntry:
    """One object to bake, and where its meshes resolve.

    `ir_id` is the cache key AND the URL segment — see
    `viewer/runs.derived_url_prefix`. `asset_base_dir` is the run directory for
    both layers, because a catalog object's `params["mesh"]` is stored
    **run-relative** (`catalog/assets/meshes/<digest>.obj`) rather than relative
    to `catalog/`. Passing `run_dir / catalog` here — which is what
    `live.resolve_target` hands back for a scene, for its own reasons — would
    double the prefix and resolve nothing.
    """

    ir_id: str
    ir: dict
    asset_base_dir: Path
    #: For messages only: the object's name, or its ref when it has none.
    label: str


#: What to say when a run has no `Material(shader=...)` anywhere. Not an
#: error and not a failure: every other material kind already draws correctly
#: in the viewport, so a bake would spend 15-95 s of GPU per object to
#: reproduce an appearance three.js can compute itself.
_NO_SHADER = (
    "nothing in this run is authored with Material(shader=...), and every "
    "other material — a scalar color, a staged texture — already draws "
    "correctly in the viewport, so there is nothing to derive")


def _has_shader(ir: dict) -> bool:
    """Whether any part of this object is shader-authored.

    **The whole reason a derived asset exists.** A scalar or textured material
    is already drawn correctly by `materials.ts`, so baking one is pure cost —
    and on the scene layer it is the cost multiplied by the catalog, which is
    how opening a PATINA-era scene would have started six pointless bakes.
    """
    for part in ir.get("parts") or ():
        if not isinstance(part, dict):
            continue
        material = part.get("material")
        if isinstance(material, dict) and material.get("shader"):
            return True
    return False


def _entry_for(ir: dict, run_dir: Path, fallback: str) -> DerivedEntry | None:
    """One entry, or None when the IR carries no id to key a cache on."""
    ir_id = str(ir.get("id") or "")
    if not ir_id:
        return None
    return DerivedEntry(ir_id=ir_id, ir=ir, asset_base_dir=run_dir,
                        label=str(ir.get("name") or fallback))


def resolve(run_dir: Path) -> list[DerivedEntry]:
    """Every object this run needs baked, in a stable order.

    Reuses `live.resolve_target` so the layer handling has exactly one
    implementation. Raises rather than returning an empty list, so a caller
    never has to distinguish "nothing to do" from "nothing to show".
    """
    from . import animation as anim_mod
    from . import scene as scene_mod

    run_dir = Path(run_dir)

    # **An animation run resolves as its staged scene.** `run_urdf` already
    # converts one that way before generating geometry, so doing the same here
    # means both halves agree about which catalog objects exist — and the
    # per-catalog-object substitution needs no animation-specific branch.
    # `live.resolve_target` refuses the layer (the live *stream* genuinely
    # cannot carry actor motion), so the conversion happens before it is asked.
    ir_path = run_dir / "ir.json"
    animation = False
    if ir_path.is_file():
        import json as _json

        try:
            kind = scene_mod.ir_kind(_json.loads(ir_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            kind = None
        animation = kind == "animation"

    if animation:
        scene_ir = anim_mod.load_scene_ir(run_dir)
        if scene_ir is None:
            raise DerivedUnavailable(
                f"animation run has no {anim_mod.SCENE_JSON}, so there is no "
                f"scene to resolve its objects against")
        catalog = scene_mod.load_catalog(run_dir)
        return _scene_entries(scene_ir, catalog, run_dir)

    try:
        target = live_mod.resolve_target(run_dir)
    except live_mod.LiveUnavailable as exc:
        raise DerivedUnavailable(str(exc)) from exc

    if target.layer == "object":
        entry = _entry_for(target.ir, run_dir, "object")
        if entry is None:
            raise DerivedUnavailable(
                "the IR has no id, so there is no cache key — geometry is only "
                "finalized on commit")
        if not _has_shader(target.ir):
            raise DerivedUnavailable(_NO_SHADER)
        return [entry]

    if target.layer != "scene":
        raise DerivedUnavailable(
            f"derived viewer assets do not cover a {target.layer} run, which "
            f"keeps the preview viewport")

    return _scene_entries(target.ir, target.catalog or {}, run_dir)


def _scene_entries(scene_ir: dict, catalog, run_dir: Path) -> list[DerivedEntry]:
    """One entry per distinct shader-authored `object_ref` the scene instances.

    **Distinct object_refs, not instances.** Two chairs of the same object draw
    the same derived meshes from the same cache entry, so baking per instance
    would spend the whole cost again for a byte-identical answer. Insertion
    order is the scene's own instance order, which keeps the status payload and
    the bake queue reproducible.
    """
    entries: dict[str, DerivedEntry] = {}
    saw_shader = False
    for inst in scene_ir.get("instances") or ():
        if not isinstance(inst, dict):
            continue
        ref = inst.get("object_ref")
        obj_ir = catalog.get(ref)
        if not isinstance(obj_ir, dict):
            continue
        # Only shader-bearing objects. A scene commonly mixes them with
        # scalar and textured ones, and those keep their staged meshes and
        # their URDF materials — already the right picture for them.
        if not _has_shader(obj_ir):
            continue
        saw_shader = True
        entry = _entry_for(obj_ir, run_dir, str(ref))
        if entry is not None and entry.ir_id not in entries:
            entries[entry.ir_id] = entry
    if not entries:
        raise DerivedUnavailable(
            "no shader-authored instance in this scene resolves to a catalog "
            "object with an id, so there is nothing to key a derived asset on"
            if saw_shader else _NO_SHADER)
    return list(entries.values())


def status(run_dir: Path, *, start: bool = True) -> dict:
    """What the browser needs to know, and kick off any bake that is missing.

    `start=False` makes this a pure query, which is what the tests and the
    URDF route want: neither should be able to trigger a 90-second bake as a
    side effect of asking a question.

    Aggregate over the run's objects. `ready` means **every** one of them has
    a manifest: the URDF is fetched once, and switching to it mid-bake would
    draw the objects that have not landed yet as gray with nothing saying so.
    `partial` is the settled version of that — some baked, the rest are not
    coming — where drawing the mixture *is* the best answer and the browser is
    told which part of it is still the preview. `objects` / `objects_ready`
    carry the progress either way.
    """
    from ..render import viewer_bake as VB

    try:
        entries = resolve(run_dir)
    except DerivedUnavailable as exc:
        return {"status": "unavailable", "reason": str(exc)}

    done: dict[str, dict] = {}
    pending: list[DerivedEntry] = []
    for entry in entries:
        manifest = VB.read_manifest(entry.ir_id)
        if manifest is None:
            pending.append(entry)
        else:
            done[entry.ir_id] = manifest

    progress = {"objects": len(entries), "objects_ready": len(done)}

    if not pending:
        return {**_ready(done), **progress}

    building = 0
    failures: list[str] = []
    launch: list[DerivedEntry] = []
    with _STATE_LOCK:
        for entry in pending:
            state = _STATE.get(entry.ir_id)
            status_now = (state or {}).get("status")
            if status_now == "building":
                building += 1
                continue
            if not start:
                # **A failed bake is reported as an error, not as "absent".**
                # `absent` is a state the browser waits on, and a failure never
                # changes on its own — returning it here is what would poll a
                # dead bake every two seconds forever.
                if status_now == "error":
                    failures.append(
                        f"{entry.label}: "
                        f"{(state or {}).get('reason') or 'bake failed'}")
                continue
            # Claimed whether this entry has never run or previously failed:
            # `start=True` is the browser's first poll for this mount, and a
            # retry is what it is asking for. Nothing is reported as failed
            # while it is being attempted again. Marking it here, under the
            # lock, is what stops a concurrent poll claiming it twice.
            _STATE[entry.ir_id] = {"status": "building"}
            launch.append(entry)
            building += 1

    # **Started outside `_STATE_LOCK`.** `_build` takes the same lock to record
    # its outcome, and the lock is not reentrant — so spawning under it would
    # deadlock any caller that runs the body synchronously instead of on a
    # thread, which is how a test gets a deterministic bake.
    for entry in launch:
        threading.Thread(
            target=_build, args=(entry,),
            name=f"dycraft-bake-{entry.ir_id[:8]}", daemon=True).start()

    if building:
        return {"status": "building", **progress}
    if failures:
        # **`partial` when some objects made it and the rest are not coming.**
        # A scene of six objects where one bake fails still has five correct
        # ones, and `substituted_ir` already leaves the failed object on its
        # staged mesh — so drawing the mixture shows five real materials
        # instead of six gray placeholders. `error` stays reserved for the case
        # where there is nothing better than the preview.
        if not done:
            return {"status": "error", "reason": failures[0], **progress}
        ready = _ready(done)
        # `_ready` names an `ir_id` when exactly one manifest is present; here
        # that would be "the one object that worked", which is not what the
        # field means anywhere else.
        ready.pop("ir_id", None)
        return {**ready, "status": "partial", "reason": failures[0], **progress}
    return {"status": "absent", **progress,
            **({"ir_id": entries[0].ir_id} if len(entries) == 1 else {})}


def manifests_for(run_dir: Path) -> dict[str, dict]:
    """`{ir_id: manifest}` for every complete bake this run can draw.

    Never builds, and never raises: a run with no derived assets — or one whose
    layer is refused — is an empty mapping, which every caller already has to
    handle because a bake can be evicted between the status poll and the fetch.

    Doubles as the **authorization set** for the asset route: an `ir_id` that is
    not a key here is not one this run draws, and must not be served out of the
    shared cache just because some other run baked it.
    """
    from ..render import viewer_bake as VB

    try:
        entries = resolve(run_dir)
    except DerivedUnavailable:
        return {}
    out: dict[str, dict] = {}
    for entry in entries:
        manifest = VB.read_manifest(entry.ir_id)
        if manifest is not None:
            out[entry.ir_id] = manifest
    return out


def _ready(manifests: dict[str, dict]) -> dict:
    """The ready payload, summed over however many objects the run has."""
    parts = [e for m in manifests.values() for e in (m.get("parts") or ())]
    versions = {m.get("version") for m in manifests.values()}
    return {
        "status": "ready",
        # One id when there is one object, so an object run's payload is what
        # it always was. A scene has several and no single one identifies it.
        **({"ir_id": next(iter(manifests))} if len(manifests) == 1 else {}),
        "version": versions.pop() if len(versions) == 1 else None,
        "parts": len(parts),
        "bytes": sum(int(e.get("bytes") or 0) for e in parts),
        "warnings": [w for m in manifests.values()
                     for w in (m.get("warnings") or [])],
    }


def _build(entry: DerivedEntry) -> None:
    from ..render import RenderRequest
    from ..render import viewer_bake as VB

    try:
        device = None
        _take_slot()
        try:
            # Re-check once a slot is free: two browsers on the same run both
            # got "building", and the second must not re-bake what the first
            # finished while it waited.
            existing = VB.read_manifest(entry.ir_id)
            if existing is None:
                # A temp dir, not somewhere under the run: `_prepare` mkdirs
                # `out_dir`, and this module's contract is that a run directory
                # is read-only. Nothing is written into it.
                with tempfile.TemporaryDirectory(prefix="dycraft-derived-") as tmp:
                    request = RenderRequest(
                        # **Always the object layer.** A scene's entry is one of
                        # its catalog objects, which IS an ObjectIR — so the
                        # bake never needs to learn about instances, and
                        # `pack_bake_job` stays the single object-layer path.
                        layer="object", ir=entry.ir,
                        catalog=None, target="all",
                        out_dir=Path(tmp) / "renders", step=0,
                        asset_base_dir=entry.asset_base_dir,
                        backend="blender")
                    existing = VB.build(request)
                    device = existing.get("device")
        finally:
            _give_slot(device)
        # The manifest on disk is the durable record and `status` reads it
        # straight back, so this only stops a concurrent poll re-queueing the
        # same object.
        with _STATE_LOCK:
            _STATE[entry.ir_id] = {"status": "ready"}
    except BaseException as exc:      # noqa: BLE001 — a failed bake is a
        # message, not a dead viewer: the browser keeps the preview viewport
        # and shows why, which is the whole contract on failure.
        with _STATE_LOCK:
            _STATE[entry.ir_id] = {"status": "error",
                                   "reason": f"{type(exc).__name__}: {exc}"}


#: The default view's flat colors (`render/viewer_flat.py`): `ir_id -> state` while a pass runs or after one failed,
#: under its own lock. They do not take a bake slot: the pass builds no meshes and takes seconds, and it must not
#: queue behind a bake of minutes.
_FLAT_STATE: dict[str, dict] = {}
_FLAT_LOCK = threading.Lock()
#: Bumped whenever a pass finishes, so a URDF cached before it is generated again (`viewer/app.py`).
_flat_generation = [0]


def flat_generation() -> int:
    return _flat_generation[0]


def flat_colors(run_dir: Path) -> tuple[str, dict[str, dict]]:
    """`(status, {ir_id: parts})`: the flat colors of this run's shader parts, and whether more are coming.

    `status` is `"ready"` when every shader object's colors are read, `"pending"` while a pass runs for the rest
    (started here, once per object), `"error"` when it failed (not retried until the server restarts, so a missing
    Blender does not start one per request), and `"none"` for a run with no shader parts. The mapping holds what is
    read so far; an object missing from it draws its IR color.
    """
    from ..render import viewer_flat as VF

    try:
        entries = resolve(run_dir)
    except DerivedUnavailable:
        return "none", {}
    found: dict[str, dict] = {}
    missing: list[DerivedEntry] = []
    for entry in entries:
        parts = VF.read(entry.ir_id)
        if parts is None:
            missing.append(entry)
        else:
            found[entry.ir_id] = parts
    if not missing:
        return "ready", found
    launch: list[DerivedEntry] = []
    failed = False
    with _FLAT_LOCK:
        for entry in missing:
            state = (_FLAT_STATE.get(entry.ir_id) or {}).get("status")
            if state == "error":
                failed = True
            elif state != "building":
                _FLAT_STATE[entry.ir_id] = {"status": "building"}
                launch.append(entry)
        building = any((_FLAT_STATE.get(e.ir_id) or {}).get("status") == "building" for e in missing)
    if launch:
        threading.Thread(target=_flat_build, args=(Path(run_dir), launch), name="dycraft-flat", daemon=True).start()
    if building:
        return "pending", found
    return ("error" if failed else "ready"), found


def _flat_pass(run_dir: Path, entries: list[DerivedEntry]) -> set[str]:
    """`viewer_flat.compute` for these entries in a fresh interpreter (`viewer/_flat_pass.py`); the ids it read."""
    import json
    import os
    import subprocess
    import sys

    from .. import _oom
    from ..render import RenderUnavailable
    from ..render import viewer_flat as VF

    package = __name__.split(".")[0]
    # The folder this package was imported from, unresolved, so a package reached through a symlink keeps its name.
    root = str(Path(sys.modules[package].__file__).parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
    try:
        proc = subprocess.run(
            _oom.tiered([sys.executable, "-m", f"{package}.viewer._flat_pass", str(run_dir),
                         *(e.ir_id for e in entries)], _oom.WORKER),
            capture_output=True, text=True, timeout=VF.TIMEOUT_S + 60, env=env)
    except subprocess.TimeoutExpired:
        raise RenderUnavailable(f"reading shader colors timed out after {VF.TIMEOUT_S + 60:.0f} s") from None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
        raise RenderUnavailable("\n  ".join(tail) or f"the shader color pass exited {proc.returncode}")
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def _flat_build(run_dir: Path, entries: list[DerivedEntry]) -> None:
    try:
        read = _flat_pass(run_dir, entries)
        outcome = {e.ir_id: {"status": "ready"} if e.ir_id in read else
                   {"status": "error", "reason": "the pass did not report this object"} for e in entries}
    except BaseException as exc:      # noqa: BLE001 -- a failed pass leaves the view gray, with the reason logged
        print(f"[viewer] shader colors not read: {type(exc).__name__}: {exc}", flush=True)
        outcome = {e.ir_id: {"status": "error", "reason": f"{type(exc).__name__}: {exc}"} for e in entries}
    with _FLAT_LOCK:
        _FLAT_STATE.update(outcome)
        _flat_generation[0] += 1
