"""The scene layer: one `scene.py` that authors the objects it places.

The scene layer arranges a catalog it is given; the object layer builds one
object and cannot place it. This layer is both, so what these tests protect is
the composition itself — that several objects can be built in one execution,
that a definition placed twice stays one definition, and that the ownership
checks hold once several objects exist at once.

The CAD-backed tests go through `object_worker` rather than calling
`build_ir_with_layer` directly: routing a scene build to the CAD interpreter
is itself part of the contract, since `scene.py` now constructs live build123d
solids and the daemon has no build123d.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.sdk import ObjectHandle
from dycraft.sdk._errors import (BuildError, DanglingReference,
                                 InvalidInstance, InvalidScene)
from dycraft.sdk.object_builder import (Object, _reset_object_singleton,
                                        take_finalized_objects)
from dycraft.sdk.scene_builder import Scene, _reset_scene_singleton


# ---------------------------------------------------------------------------
# Builder-level: many objects in one execution
# ---------------------------------------------------------------------------


@pytest.fixture
def scene_sdk():
    """Reset both builders the way `reset_scene_layer_state` does."""
    _reset_object_singleton(limit=None)
    _reset_scene_singleton()
    take_finalized_objects()
    yield
    _reset_object_singleton(limit=1)
    _reset_scene_singleton()
    take_finalized_objects()


def _trivial(name: str) -> Object:
    obj = Object(name)
    obj.declare_root(obj.make_box("body", 1.0, 1.0, 1.0))
    return obj


def test_the_object_layer_still_refuses_a_second_object():
    """The scene layer lifts the guard; the object layer must still have it.

    `object.py` finishes by assigning one `ir`, so a second `Object(...)` there
    is a mistake with a confusing downstream symptom — the second builder's
    parts simply never reach the IR. Parameterizing the limit is only safe if
    the default is still 1.
    """
    _reset_object_singleton(limit=1)
    try:
        Object("first")
        with pytest.raises(BuildError, match="only one Object"):
            Object("second")
    finally:
        _reset_object_singleton(limit=1)


def test_a_scene_execution_builds_many_objects(scene_sdk):
    objs = [_trivial(f"o{i}") for i in range(4)]
    assert len({o._owner_id for o in objs}) == 4


def test_owner_ids_are_not_reused_when_builders_are_dropped(scene_sdk):
    """The check that `id(self)` could not make.

    A factory function returns `obj.finalize()` and drops the builder, so a
    `scene.py` building cabinets in a loop allocates and frees Objects in a
    tight cycle — precisely the pattern under which CPython hands the next
    allocation the address the last one just freed. Every cross-object check in
    `object_builder` compares `_owner_id`, so a repeat does not raise: it
    accepts one object's PartRef inside another and builds a wrong joint tree.
    """
    seen = set()
    for i in range(200):
        obj = Object(f"t{i}")
        seen.add(obj._owner_id)
        del obj
    assert len(seen) == 200


def test_a_ref_from_one_object_is_still_refused_by_another(scene_sdk):
    a, b = Object("a"), Object("b")
    part_of_a = a.make_box("body", 1.0, 1.0, 1.0)
    with pytest.raises(DanglingReference):
        b.declare_root(part_of_a)


def test_finalized_objects_drain_once_and_only_once(scene_sdk):
    _trivial("one").finalize()
    _trivial("two").finalize()
    drained = take_finalized_objects()
    assert len(drained) == 2
    assert take_finalized_objects() == {}, "a second drain must not repeat them"


def test_one_definition_finalized_twice_is_one_catalog_entry(scene_sdk):
    """Idempotent `finalize` must not double-register."""
    obj = _trivial("once")
    first, second = obj.finalize(), obj.finalize()
    assert first is second
    assert len(take_finalized_objects()) == 1


# ---------------------------------------------------------------------------
# Scene composition: what `place` accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["builder", "ir_dict", "handle"])
def test_place_accepts_every_spelling_of_a_definition(scene_sdk, spelling):
    """A scene author has an `Object`, not a catalog handle.

    Requiring `ObjectHandle(make_counter(...).finalize())` at each call site
    would be ceremony over a distinction the author has no reason to make, so
    all three normalize to the same instance.
    """
    obj = _trivial("thing")
    passed = {"builder": obj,
              "ir_dict": obj.finalize(),
              "handle": ObjectHandle(obj.finalize())}[spelling]

    scene = Scene("s")
    scene.set_room(5.0, 5.0, 3.0)
    scene.place(passed, name="i", pose=((0.0, 0.0, 0.0), (0, 0, 0)))
    ir = scene.finalize()

    assert [i["name"] for i in ir["instances"]] == ["i"]
    assert ir["instances"][0]["object_ref"] == obj.finalize()["id"]


@pytest.mark.parametrize("bad,match", [
    (42, "expected ObjectHandle"),
    ("counter", "expected ObjectHandle"),
    ({"not": "an ir"}, "not a finalized ObjectIR"),
])
def test_place_still_refuses_what_is_not_a_definition(scene_sdk, bad, match):
    scene = Scene("s")
    scene.set_room(5.0, 5.0, 3.0)
    with pytest.raises(InvalidInstance, match=match):
        scene.place(bad, name="i", pose=((0.0, 0.0, 0.0), (0, 0, 0)))


# ---------------------------------------------------------------------------
# The whole build, through the CAD worker
# ---------------------------------------------------------------------------

pytestmark_cad = pytest.mark.skipif(
    not W.is_available(),
    reason="needs a build123d interpreter (DYCRAFT_B123D_PYTHON / .b123d_python)")


#: Two definitions from one shared dimension, one of them placed twice. This is
#: the case the layer exists for: `CABINET_D` sets both the cabinet's depth and
#: the counter's, so the interface between them is one number in one file
#: rather than two briefs that have to agree.
_KITCHEN = '''
CABINET_D, CABINET_H = 0.6, 0.9

def make_cabinet(width):
    obj = Object("cabinet")
    body = obj.add_part("carcass", Box(width, CABINET_D, CABINET_H),
                        material=Material(color=(120, 90, 60)))
    door = obj.add_part("door", Box(width, 0.02, CABINET_H * 0.9),
                        material=Material(color=(200, 200, 205)))
    obj.declare_root(body)
    obj.attach_revolute("hinge", parent=body, child=door,
                        parent_anchor=obj.anchor(body, "front_left"),
                        child_anchor=obj.anchor(door, "back_left"),
                        axis=(0, 0, 1), limits=(0, 100), default=0)
    return obj.finalize()

def make_counter(width):
    obj = Object("counter")
    obj.declare_root(obj.add_part("top", Box(width, CABINET_D, 0.04),
                                  material=Material(color=(60, 60, 65))))
    return obj.finalize()

cabinet = make_cabinet(0.6)
scene = Scene("kitchen")
scene.set_room(4.0, 3.0, 2.6)
scene.place(cabinet, name="cabinet_left", pose=((-0.4, 0.0, 0.0), (0, 0, 0)))
scene.place(cabinet, name="cabinet_right", pose=((0.4, 0.0, 0.0), (0, 0, 0)))
scene.place(make_counter(1.8), name="counter",
            pose=((0.0, 0.0, CABINET_H), (0, 0, 0)))
ir = scene.finalize()
'''


@pytest.fixture(scope="module")
def kitchen(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("scene_kitchen")
    out = W.run_object_source(_KITCHEN, run_dir=run_dir, layer="scene")
    assert out["ok"], (out.get("build_error") or out.get("sandbox_violation")
                       or out)
    return out, run_dir


@pytestmark_cad
def test_a_scene_build_returns_a_scene_and_its_catalog(kitchen):
    out, _ = kitchen
    ir = out["ir"]
    assert ir["schema"].startswith("dycraft.scene")
    assert len(out["object_catalog"]) == 2, "one cabinet, one counter"
    refs = {i["object_ref"] for i in ir["instances"]}
    assert refs <= set(out["object_catalog"]), (
        "every instance's definition must be in the catalog the build returned "
        "— nothing else carries it across the worker boundary")


@pytestmark_cad
def test_one_definition_placed_twice_stays_one_definition(kitchen):
    """The reuse that separate object slots could not express.

    Two instances of one design must share a catalog entry rather than
    duplicating its geometry, and must still be separately addressable.
    """
    out, _ = kitchen
    by_name = {i["name"]: i["object_ref"] for i in out["ir"]["instances"]}
    assert by_name["cabinet_left"] == by_name["cabinet_right"]
    assert by_name["counter"] != by_name["cabinet_left"]
    assert len({i["name"] for i in out["ir"]["instances"]}) == 3


@pytestmark_cad
def test_the_built_objects_carry_real_geometry_and_articulation(kitchen):
    out, run_dir = kitchen
    cabinet = next(o for o in out["object_catalog"].values()
                   if o["name"] == "cabinet")
    assert [j["name"] for j in cabinet["joints"]] == ["hinge"]
    for part in cabinet["parts"]:
        mesh = run_dir / part["params"]["mesh"]
        brep = run_dir / part["params"]["brep"]
        assert mesh.is_file() and brep.is_file(), (
            "a scene build stages its parts into its own run dir, so ObjectIR "
            f"paths resolve against the run root: {part['name']}")


@pytestmark_cad
def test_a_shared_dimension_reaches_both_sides_of_the_interface(kitchen):
    """`CABINET_D` is one number; the cabinet and the counter both use it."""
    out, _ = kitchen
    depths = {}
    for obj in out["object_catalog"].values():
        root = next(p for p in obj["parts"] if p["name"] == obj["root"])
        lo, hi = root["params"]["aabb_min"], root["params"]["aabb_max"]
        depths[obj["name"]] = round(hi[1] - lo[1], 6)
    assert depths["cabinet"] == depths["counter"] == 0.6


@pytestmark_cad
def test_editing_one_object_changes_the_scene_hash(tmp_path):
    """Why the scene verifier cache needs no catalog fingerprint.

    The scene layer keys its verifier cache on a catalog fingerprint because
    its objects are inputs: the same `scene.py` can be compiled against a
    catalog that changed underneath it. Here the objects are outputs, so a
    geometry edit changes an ObjectIR hash, which changes the instance's
    `object_ref`, which changes the SceneIR's own hash. If that chain ever
    breaks, a stale verdict survives a real geometry change.
    """
    def build(width: str) -> dict:
        src = _KITCHEN.replace("cabinet = make_cabinet(0.6)",
                               f"cabinet = make_cabinet({width})")
        out = W.run_object_source(src, run_dir=tmp_path / f"w{width}",
                                  layer="scene")
        assert out["ok"], out.get("build_error")
        return out

    before, after = build("0.6"), build("0.7")
    assert before["ir"]["id"] != after["ir"]["id"]
    assert ({i["object_ref"] for i in before["ir"]["instances"]}
            != {i["object_ref"] for i in after["ir"]["instances"]})


@pytestmark_cad
def test_a_failed_build_contributes_no_objects_to_the_next_one(tmp_path):
    """A half-finished build must not spill its objects into the next catalog.

    The failing source finalizes a real object and *then* dies, so its
    definition is sitting in the registry when the build unwinds. The next
    build's catalog must contain only its own.
    """
    failing = '''
obj = Object("orphan")
obj.declare_root(obj.make_box("body", 1.0, 1.0, 1.0))
ir = obj.finalize()
raise ValueError("author's own bug, after finalize")
'''
    bad = W.run_object_source(failing, run_dir=tmp_path / "bad", layer="scene")
    assert not bad["ok"]

    good = W.run_object_source(_KITCHEN, run_dir=tmp_path / "good",
                               layer="scene")
    assert good["ok"], good.get("build_error")
    names = {o["name"] for o in good["object_catalog"].values()}
    assert names == {"cabinet", "counter"}, (
        f"the failed build's object leaked into this catalog: {names}")


@pytestmark_cad
def test_the_object_layer_is_unchanged_by_the_new_route(tmp_path):
    """The same worker now serves two layers; `object.py` must not notice."""
    src = '''
obj = Object("widget")
obj.declare_root(obj.add_part("body", Box(0.2, 0.2, 0.2)))
ir = obj.finalize()
'''
    out = W.run_object_source(src, run_dir=tmp_path / "obj")
    assert out["ok"], out.get("build_error")
    assert out["ir"]["schema"].startswith("dycraft.object")
    assert "object_catalog" not in out, (
        "only a layer that authors its own objects drains a catalog")
    assert (tmp_path / "obj" / "object.py").is_file()


# ---------------------------------------------------------------------------
# SceneCatalog: the catalog is a function of the build
# ---------------------------------------------------------------------------


class _FakeCtx:
    """The two attributes `SceneCatalog` reads."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._current_build_cache = None

    def built(self, catalog: dict, source_hash: str = "h"):
        self._current_build_cache = {
            "source_hash": source_hash,
            "result": {"ok": True, "object_catalog": catalog},
        }


def _bound_payload(ctx, snapshot_dir):
    """Built from a run dir, then bound — the daemon's own order.

    `server._build_layer` constructs the payload before any ToolContext
    exists, so anything that reads live build state is joined afterwards.
    Constructing it the same way here is what keeps this from testing a
    convenience the production path never uses.
    """
    from dycraft.agent.layers.scene_context import SceneCatalog

    payload = SceneCatalog(ctx.run_dir, snapshot_dir)
    payload.bind(ctx)
    return payload


def _catalog_of(*objs) -> dict:
    return {o["id"]: o for o in objs}


@pytest.fixture
def two_definitions(scene_sdk):
    return _trivial("alpha").finalize(), _trivial("beta").finalize()


def test_the_catalog_follows_the_current_build(tmp_path, two_definitions):
    """No adopt step, so there is no adopt step to forget.

    A mutable payload plus a copy-in call has to be called on every path that
    produces a build — compile, probe, view, snapshot, and each of their
    cache-hit branches. Missing one does not raise; it verifies the previous
    revision's geometry and passes. Reading through to the build cache is what
    makes that failure unrepresentable.
    """
    alpha, beta = two_definitions
    ctx = _FakeCtx(tmp_path)
    payload = _bound_payload(ctx, tmp_path / "catalog")

    assert payload.catalog == {}, "no build yet means no objects"
    assert payload.object_handles == {}

    ctx.built(_catalog_of(alpha))
    assert set(payload.name_to_hash) == {"alpha"}

    ctx.built(_catalog_of(beta), source_hash="h2")
    assert set(payload.name_to_hash) == {"beta"}, (
        "the previous build's objects must not survive into this one")


def test_a_failed_build_exposes_no_catalog(tmp_path, two_definitions):
    alpha, _ = two_definitions
    ctx = _FakeCtx(tmp_path)
    payload = _bound_payload(ctx, tmp_path / "catalog")
    ctx.built(_catalog_of(alpha))
    assert payload.catalog

    ctx._current_build_cache = {"source_hash": "h2", "result": {"ok": False}}
    assert payload.catalog == {}, (
        "a build that did not produce a scene has no objects to offer")


def test_definitions_sharing_a_name_stay_separately_addressable(tmp_path,
                                                                scene_sdk):
    """Names are advisory in a scene build; the catalog is hash-keyed.

    One factory parameterized into a wide and a narrow cabinet legitimately
    produces two definitions called "cabinet". Neither may shadow the other,
    because `view` and `probe` address definitions by name.
    """
    a = Object("cabinet")
    a.declare_root(a.make_box("body", 1.0, 1.0, 1.0))
    b = Object("cabinet")
    b.declare_root(b.make_box("body", 2.0, 1.0, 1.0))
    wide, narrow = a.finalize(), b.finalize()
    assert wide["id"] != narrow["id"]

    ctx = _FakeCtx(tmp_path)
    ctx.built(_catalog_of(wide, narrow))
    payload = _bound_payload(ctx, tmp_path / "catalog")

    assert len(payload.name_to_hash) == 2
    assert set(payload.name_to_hash.values()) == {wide["id"], narrow["id"]}
    assert len(payload.object_handles) == 2


def test_refresh_snapshot_writes_the_definitions_and_a_manifest(tmp_path,
                                                               two_definitions):
    import json

    from dycraft.agent.layers.scene_context import CATALOG_MANIFEST

    alpha, beta = two_definitions
    ctx = _FakeCtx(tmp_path)
    ctx.built(_catalog_of(alpha, beta))
    out = tmp_path / "catalog"
    payload = _bound_payload(ctx, out)

    payload.refresh_snapshot({"instances": [{"object_ref": alpha["id"]}]})

    manifest = json.loads((out / CATALOG_MANIFEST).read_text())
    assert set(manifest["objects"]) == {"alpha", "beta"}
    assert manifest["placed"] == [alpha["id"]], (
        "the manifest distinguishes what the program built from what the "
        "scene placed — an author inspecting a candidate before placing it "
        "must not have it disappear")
    assert (out / f"{alpha['id']}.json").is_file()
    assert (out / f"{beta['id']}.json").is_file()


def test_refresh_snapshot_drops_definitions_a_rebuild_removed(tmp_path,
                                                              two_definitions):
    alpha, beta = two_definitions
    ctx = _FakeCtx(tmp_path)
    out = tmp_path / "catalog"
    payload = _bound_payload(ctx, out)

    ctx.built(_catalog_of(alpha, beta))
    payload.refresh_snapshot({"instances": []})
    assert (out / f"{beta['id']}.json").is_file()

    ctx.built(_catalog_of(alpha), source_hash="h2")
    payload.refresh_snapshot({"instances": []})
    assert not (out / f"{beta['id']}.json").exists(), (
        "a definition the author deleted must not linger in the run dir")


# ---------------------------------------------------------------------------
# load_object: building another file's program inside this build
# ---------------------------------------------------------------------------

#: What a delegated object run leaves on disk: a top-level program that builds
#: one object and assigns `ir`. Loaded verbatim — the point of the loader is
#: that the existing object contract needs no rewrite to be composable.
_DELEGATE = '''
obj = Object("fridge")
body = obj.add_part("body", Box(0.7, 0.7, {height}),
                    material=Material(color=(210, 210, 215)))
door = obj.add_part("door", Box(0.7, 0.03, {height} * 0.9))
obj.declare_root(body)
obj.attach_revolute("hinge", parent=body, child=door,
                    parent_anchor=obj.anchor(body, "front_left"),
                    child_anchor=obj.anchor(door, "back_left"),
                    axis=(0, 0, 1), limits=(0, 120), default=0)
ir = obj.finalize()
'''

_SCENE_WITH_DELEGATE = '''
fridge = load_object("objects/fridge/object.py")

def make_counter(width):
    obj = Object("counter")
    obj.declare_root(obj.add_part("top", Box(width, 0.6, 0.04)))
    return obj.finalize()

scene = Scene("kitchen")
scene.set_room(4.0, 3.0, 2.6)
scene.place(fridge, name="fridge", pose=((-1.0, 0.0, 0.0), (0, 0, 0)))
scene.place(make_counter(1.6), name="counter", pose=((0.5, 0.0, 0.9), (0, 0, 0)))
ir = scene.finalize()
'''


def _with_delegate(run_dir, height="1.8", scene=_SCENE_WITH_DELEGATE):
    """Lay out a run dir the way a completed delegation leaves it."""
    dele = run_dir / "objects" / "fridge"
    dele.mkdir(parents=True, exist_ok=True)
    (dele / "object.py").write_text(_DELEGATE.format(height=height))
    return W.run_object_source(scene, run_dir=run_dir, layer="scene")


def _part_height(obj_ir, part_name="body"):
    part = next(p for p in obj_ir["parts"] if p["name"] == part_name)
    return round(part["params"]["aabb_max"][2] - part["params"]["aabb_min"][2], 4)


def _named(catalog, name):
    return next(o for o in catalog.values() if o["name"] == name)


@pytestmark_cad
def test_a_delegated_object_program_loads_unchanged(tmp_path):
    """The existing `object.py` contract composes without being rewritten.

    An object agent is told to write a complete top-level program, not a
    factory. If integrating one required rewriting it, every delegated result
    would need an editing pass before it could be placed.
    """
    out = _with_delegate(tmp_path)
    assert out["ok"], out.get("build_error") or out.get("sandbox_violation")
    catalog = out["object_catalog"]
    assert sorted(o["name"] for o in catalog.values()) == ["counter", "fridge"]
    fridge = _named(catalog, "fridge")
    assert [j["name"] for j in fridge["joints"]] == ["hinge"], (
        "the delegate's articulation must survive loading")
    assert _part_height(fridge) == 1.8


@pytestmark_cad
def test_only_an_object_with_its_own_program_cites_a_source(tmp_path):
    """`source` is a claim about the object, so only an object with one makes it.

    A loaded `object.py` is one object's recipe and is recorded as such. A
    definition built inline by a function in `scene.py` has no program of its
    own: naming the whole file would describe every object in it as sharing one
    recipe and, because `source` is inside the content hash, re-id all of them
    on any edit to the file. The bundle records the entry source instead.
    """
    out = _with_delegate(tmp_path)
    catalog = out["object_catalog"]
    fridge = _named(catalog, "fridge")
    counter = _named(catalog, "counter")

    delegate_text = (tmp_path / "objects" / "fridge" / "object.py").read_text()
    from dycraft.agent.layers._scene_source import source_digest
    assert source_digest(delegate_text) in fridge["source"], (
        "the fridge's staged recipe must be the delegate's own program")
    assert "source" not in counter, (
        "an inline definition has no program of its own to cite")


@pytestmark_cad
def test_the_build_reports_every_file_it_loaded(tmp_path):
    out = _with_delegate(tmp_path)
    assert set(out["source_deps"]) == {"objects/fridge/object.py"}


@pytestmark_cad
def test_editing_only_the_loaded_program_changes_the_scene(tmp_path):
    """The invalidation the entry-source hash cannot express.

    `scene.py` is byte-identical across these two builds; only the delegate
    changed. If that did not propagate, an author editing a loaded object would
    keep being shown the geometry from before the edit.
    """
    first = _with_delegate(tmp_path, height="1.8")
    second = _with_delegate(tmp_path, height="2.0")
    assert _part_height(_named(first["object_catalog"], "fridge")) == 1.8
    assert _part_height(_named(second["object_catalog"], "fridge")) == 2.0
    assert first["ir"]["id"] != second["ir"]["id"]


@pytestmark_cad
@pytest.mark.parametrize("bad_scene,reason", [
    ('ir = load_object("../escape/object.py")', "inside the run directory"),
    ('ir = load_object("objects/absent/object.py")', "no such file"),
])
def test_load_object_refuses_paths_it_should(tmp_path, bad_scene, reason):
    out = W.run_object_source(bad_scene, run_dir=tmp_path, layer="scene")
    assert not out["ok"]
    blob = str(out.get("build_error") or out.get("sandbox_violation"))
    assert reason in blob, blob


@pytestmark_cad
def test_a_loaded_program_that_assigns_no_ir_is_named(tmp_path):
    dele = tmp_path / "objects" / "silent"
    dele.mkdir(parents=True)
    (dele / "object.py").write_text('obj = Object("silent")\n')
    out = W.run_object_source('ir = load_object("objects/silent/object.py")',
                              run_dir=tmp_path, layer="scene")
    assert not out["ok"]
    assert "defined no `ir`" in str(out.get("build_error"))


@pytestmark_cad
def test_a_circular_load_is_reported_not_recursed(tmp_path):
    """A cycle would otherwise exhaust the stack and name none of the files."""
    d = tmp_path / "objects" / "loop"
    d.mkdir(parents=True)
    (d / "object.py").write_text('ir = load_object("objects/loop/object.py")\n')
    out = W.run_object_source('ir = load_object("objects/loop/object.py")',
                              run_dir=tmp_path, layer="scene")
    assert not out["ok"]
    assert "circular load" in str(out.get("build_error"))


@pytestmark_cad
def test_a_loaded_program_obeys_the_same_sandbox(tmp_path):
    """Loading is not a way around the AST allowlist."""
    d = tmp_path / "objects" / "sneaky"
    d.mkdir(parents=True)
    (d / "object.py").write_text("import os\nir = {}\n")
    out = W.run_object_source('ir = load_object("objects/sneaky/object.py")',
                              run_dir=tmp_path, layer="scene")
    assert not out["ok"]
    assert "import" in str(out.get("build_error")
                           or out.get("sandbox_violation")).lower()


# ---------------------------------------------------------------------------
# Dependency-aware build cache
# ---------------------------------------------------------------------------


def _scene_ctx(run_dir):
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers.layer import scene_layer_config
    return T.ToolContext(run_dir=run_dir, layer_cfg=scene_layer_config())


@pytestmark_cad
def test_an_unchanged_build_is_served_from_cache(tmp_path):
    from dycraft.agent.layers import tools as T

    _with_delegate(tmp_path)          # lays out the delegate + writes scene.py
    ctx = _scene_ctx(tmp_path)
    first = T.build_current_ir(ctx)
    assert first["ok"], first.get("error_summary")
    again = T.build_current_ir(ctx)
    assert again is first, "unchanged source and deps must not rebuild"


@pytestmark_cad
def test_touching_a_loaded_program_invalidates_the_cache(tmp_path):
    """The cache-hit branch, not just the hash chain.

    `build_current_ir` returns the cached result whenever the entry source
    hash matches. `scene.py` does not change when a loaded program does, so
    without the dependency check this returns the stale build forever — and
    every probe, snapshot and compile downstream reads it.
    """
    from dycraft.agent.layers import tools as T

    _with_delegate(tmp_path, height="1.8")
    ctx = _scene_ctx(tmp_path)
    first = T.build_current_ir(ctx)
    assert _part_height(_named(first["object_catalog"], "fridge")) == 1.8

    delegate = tmp_path / "objects" / "fridge" / "object.py"
    delegate.write_text(_DELEGATE.format(height="2.0"))

    second = T.build_current_ir(ctx)
    assert second is not first
    assert _part_height(_named(second["object_catalog"], "fridge")) == 2.0


#: A delegate that builds nothing: `ir` is never assigned, so the load fails
#: inside the exec and the whole build fails with it. The entry `scene.py` is
#: byte-identical across the repair, which is the point.
_BROKEN_DELEGATE = """
obj = Object("fridge")
obj.declare_root(obj.add_part("body", Box(0.7, 0.7, 1.8)))
# No `ir = obj.finalize()` -- the loaded program produces nothing.
"""


@pytestmark_cad
def test_repairing_a_loaded_program_invalidates_a_cached_FAILURE(tmp_path):
    """The failure path of the dependency cache.

    A failed build must carry its `source_deps` too: `_cached_build_if_fresh`
    iterates `result.get("source_deps") or {}`, so an empty one means "nothing
    to re-check" rather than "nothing was recorded". Without them the cached
    failure would answer every later probe, snapshot and compile of an
    unchanged `scene.py`, and fixing the loaded `object.py` would not change
    the entry hash.
    """
    from dycraft.agent.layers import tools as T

    dele = tmp_path / "objects" / "fridge"
    dele.mkdir(parents=True, exist_ok=True)
    (dele / "object.py").write_text(_BROKEN_DELEGATE)
    (tmp_path / "scene.py").write_text(_SCENE_WITH_DELEGATE)
    ctx = _scene_ctx(tmp_path)

    failed = T.build_current_ir(ctx)
    assert not failed["ok"], "the delegate assigns no `ir`; this must fail"
    assert failed.get("source_deps"), (
        "a failed build must still name the files it read — they are exactly "
        "what the author is about to edit")
    assert "objects/fridge/object.py" in failed["source_deps"]

    # Repair ONLY the loaded program. `scene.py` is untouched, so its hash is
    # unchanged and the cache is reached on its entry-source check alone.
    (dele / "object.py").write_text(_DELEGATE.format(height="1.8"))
    after = T.build_current_ir(ctx)
    assert after is not failed, "the cached failure survived its own repair"
    assert after["ok"], after.get("error_summary")


@pytestmark_cad
def test_a_failed_build_still_contributes_no_objects(tmp_path):
    """The half that must NOT change: `object_catalog` stays success-only.

    Draining the dependency ledger on every path is not a license to drain
    the definitions too — a rejected build contributes no objects, and the
    two are drained separately for exactly this reason.
    """
    from dycraft.agent.layers import tools as T

    dele = tmp_path / "objects" / "fridge"
    dele.mkdir(parents=True, exist_ok=True)
    (dele / "object.py").write_text(_BROKEN_DELEGATE)
    (tmp_path / "scene.py").write_text(_SCENE_WITH_DELEGATE)

    failed = T.build_current_ir(_scene_ctx(tmp_path))
    assert not failed["ok"]
    assert not failed.get("object_catalog")


@pytestmark_cad
def test_deleting_a_loaded_program_invalidates_the_cache(tmp_path):
    from dycraft.agent.layers import tools as T

    _with_delegate(tmp_path)
    ctx = _scene_ctx(tmp_path)
    assert T.build_current_ir(ctx)["ok"]
    (tmp_path / "objects" / "fridge" / "object.py").unlink()
    after = T.build_current_ir(ctx)
    assert not after["ok"], "a build cannot still succeed on a file that is gone"


# ---------------------------------------------------------------------------
# Inspection: snapshots and the verifier policy
# ---------------------------------------------------------------------------

_SNAPSHOT_SCENE = '''
def make_cabinet(width):
    obj = Object("cabinet")
    body = obj.add_part("carcass", Box(width, 0.6, 0.9),
                        material=Material(color=(120, 90, 60)))
    door = obj.add_part("door", Box(width, 0.02, 0.8))
    obj.declare_root(body)
    obj.attach_revolute("hinge", parent=body, child=door,
                        parent_anchor=obj.anchor(body, "front_left"),
                        child_anchor=obj.anchor(door, "back_left"),
                        axis=(0, 0, 1), limits=(0, 100), default=0)
    return obj.finalize()

cab = make_cabinet(0.6)
scene = Scene("kitchen")
scene.set_room(4.0, 3.0, 2.6)
scene.place(cab, name="cabinet_left", pose=(({left}, 0.0, 0.0), (0, 0, 0)))
scene.place(cab, name="cabinet_right", pose=((0.4, 0.0, 0.0), (0, 0, 0)))
ir = scene.finalize()
'''


def _snapshot_ctx(run_dir, left="-0.4"):
    """A bound scene-layer context over a built scene, as the daemon assembles it."""
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers.scene_context import SceneCatalog

    (run_dir / "scene.py").write_text(_SNAPSHOT_SCENE.format(left=left))
    ctx = _scene_ctx(run_dir)
    payload = SceneCatalog(run_dir)
    payload.bind(ctx)
    ctx.layer_payload = payload
    assert T.build_current_ir(ctx)["ok"]
    return ctx


@pytestmark_cad
@pytest.mark.parametrize("label,kwargs", [
    ("whole scene", {"views": [{"on": "all", "azimuth": 45, "elevation": 30}]}),
    ("one instance", {"views": [{"on": "cabinet_left", "azimuth": 45,
                                 "elevation": 30}]}),
    ("explicit camera", {"views": [{"eye": [2, 2, 2], "look_at": [0, 0, 0.5]}]}),
    ("posed joint", {"views": [{"on": "cabinet_left", "azimuth": 45,
                                "elevation": 30}],
                     "poses": {"cabinet_left.hinge": 90}}),
    ("isolated instance", {"views": [{"on": "all", "azimuth": 45,
                                      "elevation": 30}],
                           "only": ["cabinet_right"]}),
])
def test_every_snapshot_form_works_on_a_scene(tmp_path, label, kwargs):
    """`on="all"`, `on="<instance>"`, and explicit cameras all frame a scene.

    Resolved through the object kernel, the named form would ask a SceneIR for
    `parts`, which it has none of, and answer `unknown part name
    'cabinet_left'. Parts: []` about an instance placed right there; `on="all"`
    would walk into `KeyError: 'joints'`. With explicit eye/look-at cameras
    still working, that would read as a modeling mistake rather than a broken
    tool.
    """
    from dycraft.agent.layers import snapshot as S

    ctx = _snapshot_ctx(tmp_path)
    out = S.run_snapshot(ctx, **kwargs)
    assert out.get("ok"), f"{label}: {out.get('error')}"
    paths = out.get("paths") or []
    assert paths, f"{label}: reported ok but produced no image"
    assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths), (
        f"{label}: reported image paths that are not on disk")


@pytestmark_cad
def test_an_unknown_target_names_the_entities_a_scene_has(tmp_path):
    """The error has to point at instances, not at the parts a scene lacks."""
    from dycraft.agent.layers import snapshot as S

    ctx = _snapshot_ctx(tmp_path)
    out = S.run_snapshot(
        ctx, views=[{"on": "nope", "azimuth": 45, "elevation": 30}])
    assert not out["ok"]
    assert "unknown instance or group name" in out["error"]
    assert "cabinet_left" in out["error"] and "cabinet_right" in out["error"]


@pytestmark_cad
def test_a_scene_that_fails_a_check_can_still_be_looked_at(tmp_path):
    """Inspection must not be gated on acceptance.

    Two instances placed on top of each other fail
    `NoCrossInstancePenetration`. That is exactly the moment an author needs a
    picture; a preview that ran the verifier suite first would leave a
    buildable scene with a real collision unviewable through the one tool that
    would show it.
    """
    from dycraft.agent.layers import compile as C
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers import snapshot as S

    ctx = _snapshot_ctx(tmp_path, left="0.4")     # exactly on the right one
    source = (tmp_path / "scene.py").read_text()
    report = C.compile_with_layer(source, layer_cfg=ctx.layer_cfg, ctx=ctx,
                                  prebuilt=T.build_current_ir(ctx))
    failed = [r["name"] for r in report["verifier_reports"]
              if r["verdict"] == "fail"]
    assert failed, "the fixture must actually fail a check"

    out = S.run_snapshot(
        ctx, views=[{"on": "all", "azimuth": 45, "elevation": 30}])
    assert out.get("ok"), (
        f"a buildable scene failing {failed} must still render: "
        f"{out.get('error')}")


def test_the_scene_suite_covers_both_levels_and_drops_the_judge():
    """No automatic suite commissions a visual critique, and the scene suite
    runs both levels of geometric checks.

    An expensive critique on every compile rebuilds from images what the
    author holds as measurements; `review()` is how an agent asks for one. The
    geometric checks are a different thing and stay — and in a scene build
    that has to mean BOTH suites, because this layer authors its objects as
    well as placing them.
    """
    from dycraft.agent.layers.layer import scene_layer_config
    from dycraft.verify import (OBJECT_VERIFIERS, INSTANCE_VERIFIERS,
                                SCENE_VERIFIERS)

    scene = [n for n, _ in SCENE_VERIFIERS]
    assert "SceneVlmJudge" not in scene
    assert "SceneVlmJudge" not in [n for n, _ in INSTANCE_VERIFIERS], (
        "the scene layer's automatic judge is gone; review() replaced it")

    missing = [n for n, _ in OBJECT_VERIFIERS if n not in scene]
    assert missing == [], (
        f"object checks that would not run on authored geometry: {missing}")
    for n, _ in INSTANCE_VERIFIERS:
        assert n in scene
    assert [n for n, _ in scene_layer_config().verifiers] == scene
    assert scene.index("NoSelfPenetration") < scene.index(
        "NoCrossInstancePenetration"), (
        "an object's own defect should be reported before the same geometry "
        "resurfaces as a collision between instances")


def test_every_automatic_suite_checks_geometry():
    """Geometry is checked automatically; a visual critique is bought
    deliberately with `review()`."""
    from dycraft.agent.layers.layer import (object_layer_config,
                                                  scene_layer_config)

    for label, cfg in {"object": object_layer_config(),
                       "scene": scene_layer_config()}.items():
        assert {n for n, _ in (cfg.verifiers or ())}, (
            f"{label} lost its geometry checks entirely")
        assert cfg.supports_review, (
            f"{label} has no automatic judge and no review() either — that is "
            "not the policy, that is no visual check at all")


def test_the_daemon_builds_a_scene_layer_without_a_catalog():
    """A scene run has no catalog to be handed; the scene layer requires one."""
    from dycraft.agent.layers import server
    from pathlib import Path
    import tempfile

    assert "scene" in server.LAYER_NAMES
    with tempfile.TemporaryDirectory() as td:
        cfg, payload = server._build_layer("scene", Path(td))
        assert cfg.name == "scene"
        assert payload.catalog == {}, "nothing is built yet"
        assert callable(payload.bind)


# ---------------------------------------------------------------------------
# Review criteria and the scene reviewer
# ---------------------------------------------------------------------------


def test_scene_checks_reach_the_ir_and_are_validated(scene_sdk):
    scene = Scene("k")
    scene.set_room(4.0, 3.0, 2.6)
    scene.place(_trivial("t"), name="i", pose=((0.0, 0.0, 0.0), (0, 0, 0)))
    scene.add_check("the counter run reads as continuous along the wall")
    scene.add_check("  the sink   is centred under the window ")

    with pytest.raises(InvalidScene, match="duplicate claim"):
        scene.add_check("the sink is centred under the window")
    with pytest.raises(InvalidScene, match="claim is empty"):
        scene.add_check("   ")
    with pytest.raises(InvalidScene, match="must be a string"):
        scene.add_check(42)

    ir = scene.finalize()
    assert [c["id"] for c in ir["checks"]] == ["c01", "c02"]
    assert ir["checks"][1]["claim"] == "the sink is centred under the window", (
        "whitespace is normalized so two spellings of one claim collide")


def test_a_scene_with_no_checks_omits_the_key(scene_sdk):
    """The IR is content-hashed whole; an empty list would re-id every scene."""
    scene = Scene("k")
    scene.set_room(4.0, 3.0, 2.6)
    scene.place(_trivial("t"), name="i", pose=((0.0, 0.0, 0.0), (0, 0, 0)))
    assert "checks" not in scene.finalize()


def test_the_check_limit_is_enforced(scene_sdk):
    scene = Scene("k")
    scene.set_room(4.0, 3.0, 2.6)
    for i in range(Scene.MAX_CHECKS):
        scene.add_check(f"claim number {i}")
    with pytest.raises(InvalidScene, match="at most"):
        scene.add_check("one too many")


def test_review_is_offered_by_the_layers_that_declare_claims():
    """Gated on a capability flag, not a layer name.

    The scene layer dropped the automatic visual judge on the argument that
    the author should choose when a look is worth paying for. That argument
    only holds if it can actually buy one.
    """
    from dycraft.agent.layers.layer import (object_layer_config,
                                                  scene_layer_config)
    assert object_layer_config().supports_review
    assert scene_layer_config().supports_review


@pytestmark_cad
def test_review_asks_a_scene_for_scene_checks(tmp_path):
    """The refusal has to name the builder the author actually has."""
    from dycraft.agent.layers import review as RV

    ctx = _snapshot_ctx(tmp_path)          # a scene declaring no checks
    out = RV.run_review(ctx)
    assert not out["ok"]
    assert "this scene declares no checks" in out["error"]
    assert "scene.add_check" in out["error"]


@pytestmark_cad
@pytest.mark.parametrize("on,expect_ok", [("all", True), ("cabinet_left", True),
                                          ("nope", False)])
def test_the_reviewers_own_camera_works_on_a_scene(tmp_path, on, expect_ok,
                                                   capsys):
    """`_review_render` frames and renders a SceneIR in scene mode.

    Framed through the object kernel and rendered with no catalog, the reviewer
    of a scene would be handed empty frames and no way to tell why. The catalog
    argument is what selects scene mode.
    """
    from dycraft.agent.layers import _review_render as RR
    from dycraft.agent.layers import tools as T

    ctx = _snapshot_ctx(tmp_path)
    build = T.build_current_ir(ctx)
    T._post_successful_compile(ctx, build["ir"], is_finish=True)
    ir_path = tmp_path / "review_ir.json"
    ir_path.write_text(json.dumps(build["ir"]))
    out_dir = tmp_path / "extra"

    rc = RR.main(["_review_render", "--ir", str(ir_path),
                  "--out-dir", str(out_dir), "--on", on,
                  "--azimuth", "45", "--elevation", "30",
                  "--asset-base", str(tmp_path / "catalog"),
                  "--catalog", str(tmp_path / "catalog")])

    if not expect_ok:
        assert rc == 1
        assert "Instances:" in capsys.readouterr().out, (
            "an unknown name must name the entities a scene has")
        return
    assert rc == 0
    pngs = sorted(out_dir.glob("extra_*.png"))
    assert pngs and all(p.stat().st_size > 0 for p in pngs)


def _spec(key, kind, claims, entity, label=None, submission=False):
    return {"key": key, "kind": kind, "noun": kind, "entity": entity,
            "label": label or key, "images": [f"{key}.png"],
            "submission": submission,
            "checks": [{"id": f"c{i + 1:02d}", "claim": c}
                       for i, c in enumerate(claims)],
            "render_cmd": "CMD"}


def test_the_review_brief_speaks_of_instances_for_a_scene():
    from dycraft.agent.layers.review import _review_prompt

    scene = _review_prompt([_spec("scene", "scene", ["the cabinets sit flush"],
                                  "instance")], user_prompt="a kitchen")
    # An object-layer review: its one object IS the submission, so the request
    # is a question about it. See `asks_prompt_match`.
    obj = _review_prompt([_spec("object", "object", ["the lid seats"], "part",
                                submission=True)],
                         user_prompt="a mug")

    assert "--on <instance|all>" in scene
    assert "the scene was built for" in scene
    assert "--on <part|all>" in obj
    assert "the object was built for" in obj


def test_a_multi_target_brief_separates_the_models_and_prefixes_ids():
    """One reviewer, many models — so nothing may be ambiguous between them.

    Both namespaces start at `c01`, so an unprefixed verdict for `c01` in a
    two-target review answers neither claim.
    """
    from dycraft.agent.layers.review import _review_prompt

    brief = _review_prompt(
        [_spec("scene", "scene", ["the run reads as continuous"], "instance"),
         _spec("espresso", "object", ["the portafilter seats square"], "part",
               label="espresso_machine")],
        user_prompt="a cafe")

    assert "scene:c01" in brief and "espresso:c01" in brief
    assert "2 separate models" in brief
    assert "different models, not" in brief, (
        "the reviewer must not read two models as two views of one")
    # prompt_match is the scene's question; asking it of the espresso machine
    # would have no sensible verdict.
    assert brief.count("the scene was built for") == 1
    assert "the object was built for" not in brief


def test_prompt_match_is_not_asked_when_no_scene_is_reviewed():
    from dycraft.agent.layers.review import _review_prompt

    brief = _review_prompt(
        [_spec("espresso", "object", ["the portafilter seats square"], "part"),
         _spec("grinder", "object", ["the hopper meets the collar"], "part")],
        user_prompt="a cafe")
    assert "prompt_match" not in brief, (
        "the request describes the scene; two appliances are not an answer "
        "to it")


def test_a_question_that_was_not_asked_gets_no_verdict_at_all():
    """The parser has to agree with the prompt about what was asked.

    If `run_review` decided whether to PARSE a `prompt_match` from
    `bool(user_prompt)` while `_review_prompt` decided whether to ASK using
    the scene rule above, a multi-target review with no scene would never pose
    the question and still record
    `{"verdict": "cant_tell", "reason": "no verdict returned"}` for it — a
    fabricated non-answer.
    """
    from dycraft.agent.layers.review import asks_prompt_match

    two_objects = [_spec("espresso", "object", ["x"], "part"),
                   _spec("grinder", "object", ["y"], "part")]
    with_scene = [_spec("scene", "scene", ["x"], "instance"),
                  _spec("espresso", "object", ["y"], "part")]

    assert asks_prompt_match(two_objects, "a cafe") is False
    assert asks_prompt_match(with_scene, "a cafe") is True
    assert asks_prompt_match(with_scene, "") is False
    # An OBJECT RUN's one object is what its brief describes, so it is asked.
    assert asks_prompt_match(
        [_spec("espresso", "object", ["x"], "part", submission=True)],
        "an espresso machine") is True
    # A lone DEFINITION inside a scene run is not: deciding on arity alone would
    # ask an espresso machine whether it answers a cafe's brief — see
    # `test_a_lone_definition_is_not_asked_to_answer_the_scene_s_brief`.
    assert asks_prompt_match([_spec("espresso", "object", ["x"], "part")],
                             "a cafe") is False


def test_a_dead_reviewer_does_not_read_as_an_abstention():
    """`cant_tell` from a timed-out sub-agent and `cant_tell` from a reviewer
    that looked and could not tell are different events.

    The first says nothing about the geometry and means re-run the review; the
    second is a real judgment. Sharing one string would give a dead sub-agent
    an abstention's label while `outcome` carried the truth unread.
    """
    from dycraft.agent.layers.review import _parse_review, _parse_prompt_match, _silence_reason

    ids = ("scene:c01", "scene:c02")
    answered = _parse_review(
        '{"results": [{"id": "scene:c01", "verdict": "met", "view": "front", '
        '"reason": "it is there"}]}',
        ids, silence_reason=_silence_reason("completed"))
    assert answered["scene:c01"]["verdict"] == "met"
    assert answered["scene:c02"]["reason"] == "no verdict returned"

    died = _parse_review("", ids, silence_reason=_silence_reason("timeout"))
    assert all(v["verdict"] == "cant_tell" for v in died.values())
    assert all("timed out" in v["reason"] for v in died.values())
    assert died["scene:c02"]["reason"] != answered["scene:c02"]["reason"], (
        "a dead reviewer and a skipped claim must not produce the same text")

    assert "timed out" in _parse_prompt_match(
        "", silence_reason=_silence_reason("timeout"))["reason"]
    assert _parse_prompt_match(
        '{"results": [], "prompt_match": {"verdict": "met", "reason": "yes"}}',
        silence_reason=_silence_reason("timeout")) == {
            "verdict": "met", "reason": "yes"}, (
        "a reviewer that DID answer must be reported as answering, whatever "
        "the outcome field says")


# ---------------------------------------------------------------------------
# The finish bundle
# ---------------------------------------------------------------------------


@pytestmark_cad
def test_assets_are_copied_at_finish_and_not_before(tmp_path):
    """Iteration must not pay to duplicate the run's geometry on every edit."""
    from dycraft.agent.layers import tools as T

    ctx = _snapshot_ctx(tmp_path)
    ir = T.build_current_ir(ctx)["ir"]
    catalog_dir = tmp_path / "catalog"

    T._post_successful_compile(ctx, ir, is_finish=False)
    assert not (catalog_dir / "assets").exists(), (
        "mid-run the assets are already resolvable under the run's own root")
    assert list(catalog_dir.glob("*.json")), "the definitions are written though"

    T._post_successful_compile(ctx, ir, is_finish=True)
    assert (catalog_dir / "assets").is_dir()


@pytestmark_cad
def test_the_finished_bundle_survives_being_moved(tmp_path):
    """The acceptance case: a bundle that outlives the run that produced it.

    Every ObjectIR path is relative and every referenced file is copied in
    beside it, so `catalog/` resolves wherever it lands. The original run is
    deleted here rather than merely left behind, because a bundle that only
    works while its run still exists is not one.
    """
    import shutil

    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers.scene_context import (
        referenced_object_assets)

    run = tmp_path / "run"
    run.mkdir()
    ctx = _snapshot_ctx(run)
    ir = T.build_current_ir(ctx)["ir"]
    T._post_successful_compile(ctx, ir, is_finish=True)

    moved = tmp_path / "elsewhere"
    shutil.copytree(run / "catalog", moved)
    shutil.rmtree(run)

    catalog = {}
    for path in moved.glob("*.json"):
        if path.name in ("manifest.json", "provenance.json"):
            continue
        obj = json.loads(path.read_text())
        catalog[obj["id"]] = obj
    assert catalog

    missing = [str(rel) for obj in catalog.values()
               for rel in referenced_object_assets(obj)
               if not (moved / rel).is_file()]
    assert not missing, f"the moved bundle cannot resolve: {missing}"


@pytestmark_cad
def test_the_bundle_records_the_sources_it_was_built_from(tmp_path):
    """A bundle that cannot say what produced it cannot be regenerated."""
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers._scene_source import source_digest
    from dycraft.agent.layers.scene_context import PROVENANCE_FILE

    _with_delegate(tmp_path)               # writes scene.py + the delegate
    ctx = _scene_ctx(tmp_path)
    from dycraft.agent.layers.scene_context import SceneCatalog
    payload = SceneCatalog(tmp_path)
    payload.bind(ctx)
    ctx.layer_payload = payload
    build = T.build_current_ir(ctx)
    assert build["ok"], build.get("error_summary")
    T._post_successful_compile(ctx, build["ir"], is_finish=True)

    prov = json.loads((tmp_path / "catalog" / PROVENANCE_FILE).read_text())
    assert prov["entry_source"]["digest"] == source_digest(
        (tmp_path / "scene.py").read_text())
    assert prov["loaded_sources"] == {
        "objects/fridge/object.py": source_digest(
            (tmp_path / "objects" / "fridge" / "object.py").read_text())}
    assert set(prov["objects"]) == {"fridge", "counter"}
    assert prov["placed"] == sorted(
        {i["object_ref"] for i in build["ir"]["instances"]})


@pytestmark_cad
def test_the_bundle_carries_the_recipes_that_built_it(tmp_path):
    """`source` is a referenced asset, so the programs travel with the geometry."""
    from dycraft.agent.layers import tools as T

    _with_delegate(tmp_path)
    ctx = _scene_ctx(tmp_path)
    from dycraft.agent.layers.scene_context import SceneCatalog
    payload = SceneCatalog(tmp_path)
    payload.bind(ctx)
    ctx.layer_payload = payload
    build = T.build_current_ir(ctx)
    T._post_successful_compile(ctx, build["ir"], is_finish=True)

    recipes = sorted((tmp_path / "catalog" / "assets" / "src").glob("*.py"))
    texts = {p.read_text() for p in recipes}
    assert any("load_object(" in t for t in texts), (
        "scene.py must travel even though no object cites it as its recipe")
    assert any('Object("fridge")' in t for t in texts), "the delegate travelled"


# ---------------------------------------------------------------------------
# review targets
# ---------------------------------------------------------------------------

_CLAIMED_SCENE = '''
def make_cabinet(width):
    obj = Object("cabinet")
    obj.declare_root(obj.add_part("carcass", Box(width, 0.6, 0.9)))
    obj.add_check("the carcass reads as one continuous box")
    return obj.finalize()

def make_lamp():
    obj = Object("lamp")
    obj.declare_root(obj.add_part("post", Box(0.1, 0.1, 1.4)))
    return obj.finalize()

scene = Scene("k")
scene.set_room(4.0, 3.0, 2.6)
scene.add_check("the cabinets read as one continuous run")
scene.place(make_cabinet(0.6), name="cab_left", pose=((-0.3, 0, 0), (0, 0, 0)))
scene.place(make_cabinet(0.6), name="cab_right", pose=((0.3, 0, 0), (0, 0, 0)))
scene.place(make_lamp(), name="lamp", pose=((1.5, 0, 0), (0, 0, 0)))
ir = scene.finalize()
'''


def _claimed_ctx(run_dir):
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers.scene_context import SceneCatalog

    (run_dir / "scene.py").write_text(_CLAIMED_SCENE)
    ctx = _scene_ctx(run_dir)
    payload = SceneCatalog(run_dir)
    payload.bind(ctx)
    ctx.layer_payload = payload
    assert T.build_current_ir(ctx)["ok"]
    return ctx


@pytestmark_cad
def test_review_defaults_to_the_scene(tmp_path):
    from dycraft.agent.layers.review import _review_targets

    specs, err = _review_targets(_claimed_ctx(tmp_path), None)
    assert err is None
    assert [s["key"] for s in specs] == ["scene"]
    assert specs[0]["catalog"], "a scene renders through its catalog"


@pytestmark_cad
def test_review_can_target_a_definition(tmp_path):
    """A definition, not an instance — one geometry placed twice is one model."""
    from dycraft.agent.layers.review import _review_targets

    ctx = _claimed_ctx(tmp_path)
    specs, err = _review_targets(ctx, ["cabinet"])
    assert err is None
    assert [s["key"] for s in specs] == ["cabinet"]
    assert specs[0]["kind"] == "object"
    assert specs[0]["ir"]["name"] == "cabinet"

    _, err = _review_targets(ctx, ["cab_left"])
    assert err is not None and "unknown review target" in err, (
        "an instance name is a placement, and placement claims are scene claims")


@pytestmark_cad
def test_review_takes_several_targets_at_once(tmp_path):
    """One reviewer, so the startup cost is paid once for the whole pass."""
    from dycraft.agent.layers.review import _review_targets

    specs, err = _review_targets(_claimed_ctx(tmp_path), ["scene", "cabinet"])
    assert err is None
    assert [s["key"] for s in specs] == ["scene", "cabinet"]


@pytestmark_cad
def test_review_refuses_a_definition_with_no_claims(tmp_path):
    """Nothing to judge is a mistake worth naming, not an empty review."""
    from dycraft.agent.layers.review import _review_targets

    _, err = _review_targets(_claimed_ctx(tmp_path), ["lamp"])
    assert err is not None
    assert "declares no checks" in err and "obj.add_check" in err


@pytestmark_cad
def test_review_caps_how_many_models_one_pass_may_judge(tmp_path):
    from dycraft.agent.layers.review import REVIEW_MAX_TARGETS, _review_targets

    ctx = _claimed_ctx(tmp_path)
    _, err = _review_targets(ctx, ["scene"] * 1 + ["cabinet"] * 1)
    assert err is None
    _, err = _review_targets(ctx, [f"t{i}" for i in range(REVIEW_MAX_TARGETS + 1)])
    assert err is not None and "at most" in err


# ---------------------------------------------------------------------------
# Per-part compile cache
# ---------------------------------------------------------------------------


@pytestmark_cad
def test_recompiling_identical_geometry_reuses_the_compiled_part(tmp_path):
    """Editing one object must not re-pay for the others.

    `object.py` compiles one object, so nothing recurs and no cache was
    needed. A scene-layer `scene.py` compiles every object it authors on every
    build, and the eleven objects an author did not touch reconstruct
    byte-identically each time. Measured on a moderately detailed part,
    `compile_entry` is a bit under half of the per-part cost.
    """
    from dycraft.agent.cad import object_worker as W

    src = '''
def make(name, w):
    obj = Object(name)
    body = fillet(Box(w, 0.5, 0.35).edges().filter_by(Axis.Z), 0.05)
    obj.declare_root(obj.add_part("body", body))
    return obj.finalize()

scene = Scene("s")
scene.set_room(20.0, 6.0, 3.0)
for i, w in enumerate([%s]):
    scene.place(make("o%%d" %% i, w), name="i%%d" %% i,
                pose=((-6.0 + i * 2.0, 0.0, 0.0), (0, 0, 0)))
ir = scene.finalize()
'''
    widths = [1.0, 1.05, 1.10, 1.15]
    base = src % ", ".join(f"{w:.3f}" for w in widths)
    edited = src % ", ".join(f"{w + (0.007 if i == 2 else 0):.3f}"
                             for i, w in enumerate(widths))

    first = W.run_object_source(base, run_dir=tmp_path / "a", layer="scene")
    assert first["ok"], first.get("build_error")
    second = W.run_object_source(edited, run_dir=tmp_path / "b",
                                 layer="scene")
    assert second["ok"], second.get("build_error")

    # Correctness is the claim under test, not speed: the untouched
    # definitions must come back with byte-identical geometry.
    #
    # Geometry, not `id`. Every object in a scene build cites `scene.py` as
    # its `source`, and `source` is inside the content hash — so editing any
    # line re-ids every definition in the file, including the ones whose shape
    # did not move. The staged B-rep and mesh digests are what actually say
    # whether geometry changed, and they are what the cache turns on.
    def geometry(out):
        return {o["name"]: [(p["name"], p["params"]["brep"], p["params"]["mesh"])
                            for p in o["parts"]]
                for o in out["object_catalog"].values()}

    a, b = geometry(first), geometry(second)
    for untouched in ("o0", "o1", "o3"):
        assert a[untouched] == b[untouched], (
            f"{untouched} was not edited; its geometry must be identical")
    assert a["o2"] != b["o2"], "the edited definition must change"


def test_the_compile_cache_is_keyed_on_the_whole_entry():
    """Every field, not the ones that look relevant.

    Keying on `brep_bytes` plus the tessellation parameters is not enough:
    texture state reaches the entry separately and phases the UV corners per
    part, so an untextured part would collide with a textured one off the same
    box and be served its bytes. Hashing the input rather than a
    summary of it is what makes the next field added here safe by default.
    """
    from dycraft.agent.cad._solid_compile import _compile_cache_key

    entry = {"name": "p", "brep_bytes": b"SURFACE", "tolerance": 0.001,
             "angular_tolerance": 0.2, "planar_max_edge": None,
             "verts": [0] * 9, "faces": [0] * 3}
    assert _compile_cache_key(entry) == _compile_cache_key(dict(entry))

    for field, value in (("brep_bytes", b"OTHER"), ("tolerance", 0.002),
                         ("planar_max_edge", 0.05), ("name", "q"),
                         ("verts", [1] * 9), ("faces", [1] * 3)):
        other = dict(entry, **{field: value})
        assert _compile_cache_key(other) != _compile_cache_key(entry), (
            f"{field} changes the compiled part and must change the key")

    # A field this key has never heard of must still change it.
    assert _compile_cache_key(dict(entry, texture_scale_m=0.55)) \
        != _compile_cache_key(entry)

    assert _compile_cache_key({"name": "p"}) is None, (
        "a part with no B-rep has nothing to key on")


def test_a_cache_hit_cannot_be_mutated_by_its_caller():
    """`object_builder` reads `warnings` off the result and may edit it."""
    from dycraft.agent.cad import _solid_compile as SC

    entry = {"name": "p", "brep_bytes": b"S", "tolerance": 0.001,
             "verts": [], "faces": []}
    key = SC._compile_cache_key(entry)
    SC._COMPILE_CACHE[key] = {"warnings": [], "obj_bytes": b"x"}
    try:
        first = SC.compile_entry(entry)
        first["warnings"].append("mutated")
        assert SC.compile_entry(entry)["warnings"] == [], (
            "one caller's edit must not reach the next hit")
    finally:
        SC.clear_compile_cache()


# ---------------------------------------------------------------------------
# Asynchronous object delegation
# ---------------------------------------------------------------------------


class _FakeChild:
    """A child process that finishes when the test says so.

    Real children are CLI agent runs measured in tens of minutes; what needs
    testing here is the bookkeeping around them.
    """

    def __init__(self, run_dir, *, commit=True, returncode=0):
        self.run_dir = run_dir
        self._commit = commit
        self._returncode = returncode
        self._release = __import__("threading").Event()

    def wait(self):
        self._release.wait(timeout=10)
        return self._returncode

    def terminate(self):
        self._release.set()

    def finish(self):
        """Write what a committed object run leaves behind, then exit."""
        if self._commit:
            (self.run_dir / "object.py").write_text("ir = None\n")
            (self.run_dir / "ir.json").write_text('{"id": "abc", "name": "x"}')
            (self.run_dir / "meta.json").write_text(json.dumps({
                "outcome": "committed",
                "deliberate_deviations": [
                    {"what": "no hinge barrels", "why": "concealed",
                     "severity": "low"}]}))
        else:
            (self.run_dir / "meta.json").write_text(
                json.dumps({"outcome": "escalated_repair_budget"}))
        self._release.set()


def _dispatch_ctx(tmp_path):
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers.layer import scene_layer_config
    return T.ToolContext(run_dir=tmp_path, layer_cfg=scene_layer_config())


def _submit(ctx, name, children, **kw):
    from dycraft.agent.layers import _object_dispatch as D

    def spawn(argv, sub_run_dir):
        child = _FakeChild(sub_run_dir, **kw)
        children[name] = (child, argv)
        return child

    return D.submit_object(ctx, name=name, brief=f"build a {name}",
                           _spawn=spawn)


def test_submit_object_returns_before_the_child_finishes(tmp_path):
    """The whole point: the author keeps working while a child builds."""
    from dycraft.agent.layers import _object_dispatch as D

    ctx = _dispatch_ctx(tmp_path)
    children = {}
    out = _submit(ctx, "espresso", children)
    assert out["ok"] and out["status"] == "running"

    # Nothing has landed, and asking does not block. A poll that finds
    # nothing is an answer, not a failed op.
    poll = D.await_any(ctx, timeout=0)
    assert poll["ok"] and poll["outcome"] == "nothing_ready"
    assert [r["name"] for r in poll["pending"]] == ["espresso"]
    # A wait the author asked for that did not deliver still fails: that is
    # the one case where the outcome says something about a stuck child.
    waited = D.await_any(ctx, timeout=0.2)
    assert not waited["ok"] and waited["outcome"] == "timeout"

    children["espresso"][0].finish()
    landed = D.await_any(ctx, timeout=10)
    assert landed["ok"] and landed["status"] == "committed"
    assert landed["source_path"] == "objects/espresso/object.py", (
        "the path must be the one load_object takes")


def test_a_landed_child_hands_back_its_deviations(tmp_path):
    """They reach an agent that can act on them, because the parent owns the
    placement too."""
    from dycraft.agent.layers import _object_dispatch as D

    ctx = _dispatch_ctx(tmp_path)
    children = {}
    _submit(ctx, "hood", children)
    children["hood"][0].finish()
    out = D.await_any(ctx, timeout=10)
    assert [d["what"] for d in out["deviations"]] == ["no hinge barrels"]


def test_a_failed_child_says_so_and_suggests_what_to_do(tmp_path):
    from dycraft.agent.layers import _object_dispatch as D

    ctx = _dispatch_ctx(tmp_path)
    children = {}
    _submit(ctx, "grinder", children, commit=False, returncode=1)
    children["grinder"][0].finish()
    out = D.await_any(ctx, timeout=10)
    assert not out["ok"] and out["status"] == "failed"
    assert out["source_path"] is None
    assert "author the object yourself" in out["next"]


def test_await_any_returns_each_landing_once(tmp_path):
    """Two children, two answers — the second call must not repeat the first."""
    from dycraft.agent.layers import _object_dispatch as D

    ctx = _dispatch_ctx(tmp_path)
    children = {}
    _submit(ctx, "a", children)
    _submit(ctx, "b", children)
    children["a"][0].finish()
    children["b"][0].finish()

    first = D.await_any(ctx, timeout=10)
    second = D.await_any(ctx, timeout=10)
    assert {first["name"], second["name"]} == {"a", "b"}
    third = D.await_any(ctx, timeout=0)
    assert not third["ok"] and third["outcome"] == "all_reported"


def test_concurrency_is_bounded(tmp_path):
    from dycraft.agent.layers import _object_dispatch as D

    ctx = _dispatch_ctx(tmp_path)
    children = {}
    for i in range(D.DEFAULT_MAX_CONCURRENT):
        assert _submit(ctx, f"o{i}", children)["ok"]
    over = _submit(ctx, "one_too_many", children)
    assert not over["ok"] and over["outcome"] == "at_concurrency_limit"
    for child, _ in children.values():
        child.terminate()


def test_readiness_reaches_the_agent_through_an_unrelated_op(tmp_path):
    """The notification channel.

    A CLI agent cannot be interrupted, so a child landing has to arrive
    through whatever the author does next.
    """
    from dycraft.agent.layers import _object_dispatch as D

    ctx = _dispatch_ctx(tmp_path)
    children = {}
    _submit(ctx, "espresso", children)
    rows = D.readiness(ctx)
    assert rows and rows[0]["status"] == "running"

    children["espresso"][0].finish()
    D.await_any(ctx, timeout=10)
    assert D.readiness(ctx) is None, (
        "a landing already handed to the author stops being news")


def test_a_brief_carries_the_constraints_that_make_delegation_safe(tmp_path):
    """A child's brief carries the request, the room and the neighboring geometry.

    With only a slot name, size and palette hint, a child designs against a box
    and the mismatch surfaces on integration. A scene author
    holds the whole design, so it can say what the object has to meet.
    """
    from dycraft.agent.layers._object_dispatch import compose_brief

    text = compose_brief(
        name="espresso", brief="A prosumer espresso machine.",
        scene_request="a 12-room luxury restaurant",
        size=(0.4, 0.5, 0.42),
        interfaces=["sits on a 0.9 m counter",
                    "its drip tray must clear the counter lip by 20 mm"],
        materials=["brushed stainless, matching the range hood"],
        states=["portafilter seated", "portafilter removed"])

    assert "12-room luxury restaurant" in text
    assert "Build the object, not the scene." in text
    assert "(0.4, 0.5, 0.42)" in text
    assert "drip tray must clear" in text
    assert "portafilter removed" in text

    bare = compose_brief(name="stool", brief="A three-legged stool.")
    assert bare.strip() == "A three-legged stool.", (
        "a field that constrains nothing is noise in a brief")


def test_only_the_scene_layer_may_delegate():
    from dycraft.agent.layers.layer import (object_layer_config,
                                                  scene_layer_config)
    assert scene_layer_config().supports_dispatch
    assert not object_layer_config().supports_dispatch, (
        "an object run has nothing to delegate")


def test_finish_refuses_while_an_object_is_still_building(tmp_path):
    """A commit mid-dispatch ships a bundle missing work in flight.

    It would also orphan the child, which keeps running against a run
    directory nobody is reading any more.
    """
    from dycraft.agent.layers import tools as T

    ctx = _dispatch_ctx(tmp_path)
    children = {}
    _submit(ctx, "espresso", children)

    out = T.run_compile_and_persist(ctx, is_finish=True)
    assert not out["ok"]
    assert out["build_error"]["evidence"]["reason"] == "dispatch_in_flight"
    assert "espresso" in out["build_error"]["summary"]
    repairs = " ".join(out["build_error"]["suggested_repairs"])
    assert "await_any" in repairs and "cancel_dispatch" in repairs

    # Canceling is the explicit way to commit without it.
    from dycraft.agent.layers import _object_dispatch as D
    D.cancel_dispatch(ctx, "d_001")
    after = T.run_compile_and_persist(ctx, is_finish=True)
    assert after["build_error"]["evidence"].get("reason") != "dispatch_in_flight"


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------


def test_the_staged_harness_matches_what_the_daemon_registers():
    """A stub the daemon never registered is drift nobody sees until a call.

    The staged `harness.py` is assembled from per-layer stub files while the
    ops are registered from `LayerConfig` flags, so the two can disagree
    silently — an agent calls a documented tool and gets an unknown-op error,
    or a live op goes unmentioned and unused.
    """
    import inspect
    import re
    import tempfile

    from dycraft.agent import LAYERS, stage_harness_client
    from dycraft.agent.layers.layer import scene_layer_config
    from dycraft.agent.layers.session import Session

    assert "scene" in LAYERS
    with tempfile.TemporaryDirectory() as td:
        staged = stage_harness_client(Path(td), "scene").read_text()
    stubs = set(re.findall(r"^def (\w+)\(", staged, re.M))

    cfg = scene_layer_config()
    registered = set(re.findall(r'@self\.op\("([^"]+)"',
                                inspect.getsource(Session)))
    gate = {"review": cfg.supports_review,
            "submit_object": cfg.supports_dispatch,
            "await_any": cfg.supports_dispatch,
            "dispatch_status": cfg.supports_dispatch,
            "cancel_dispatch": cfg.supports_dispatch,
            "snapshot": cfg.supports_snapshot}
    live = {op for op in registered if gate.get(op, True)}

    assert not (live - stubs), f"live ops with no stub: {sorted(live - stubs)}"
    assert not {s for s in stubs if s in registered and s not in live}, (
        "stubs for ops this layer does not register")
    assert "view" not in stubs


def test_the_composed_prompt_and_footer_are_scene_specific():
    from dycraft.agent.agent_runner import compose_footer, compose_system_prompt

    prompt = compose_system_prompt(layer="scene")
    assert prompt.startswith("# Scene agent\n")
    assert "submit_object" in prompt
    # The claim, not a phrasing: this layer's inspection surface is `snapshot`
    # and `review`, and the prompt must not offer `view`, which it does not
    # register. Asserted as an absence plus the positive statement, because the
    # prompts state what is available rather than what is not.
    assert "h.view(" not in prompt, "the scene prompt offers an unregistered view"
    assert "`snapshot` and `review`" in prompt, (
        "the scene prompt no longer names its inspection surface")

    footer = compose_footer("scene", "a machine shop", Path("/tmp/run"))
    assert "the objects in it as well as their arrangement" in footer
    assert "objects/" in footer, "the delegation directory must be announced"


@pytestmark_cad
def test_a_scene_session_runs_the_whole_loop(tmp_path):
    """Boot the Session the daemon builds, and drive every stage of a run."""
    from dycraft.agent.layers import server as S
    from dycraft.agent.layers.session import Session, SessionConfig

    (tmp_path / "scene.py").write_text('''
def make_bench(width):
    obj = Object("bench")
    obj.declare_root(obj.add_part("top", Box(width, 0.7, 0.05)))
    return obj.finalize()

def make_crate(size):
    obj = Object("crate")
    obj.declare_root(obj.add_part("body", Box(size, size, size * 0.7)))
    return obj.finalize()

scene = Scene("workshop")
scene.set_room(8.0, 6.0, 3.0)
scene.place(make_bench(2.0), name="bench_a", pose=((-1.0, 1.5, 0.0), (0, 0, 0)))
crate = make_crate(0.5)
scene.place(crate, name="crate_1", pose=((2.0, -1.0, 0.0), (0, 0, 0)))
scene.place(crate, name="crate_2", pose=((2.7, -1.0, 0.0), (0, 0, 0)))
ir = scene.finalize()
''')
    cfg, payload = S._build_layer("scene", tmp_path)
    session = Session(SessionConfig(
        run_dir=tmp_path, layer=cfg, layer_payload=payload,
        user_prompt="a small machine shop", initial_meta=None))

    with session:
        h = session.handlers
        assert "view" not in h
        report = h["compile"]()
        assert report["ok"], [r for r in report["verifier_reports"]
                              if r["verdict"] == "fail"]

        # the scene probe surface reaches both levels
        names = h["probe"](expression="sorted(i['name'] for i in ir['instances'])")
        assert names["ok"] and "bench_a" in names["repr"]
        defs = h["probe"](expression="sorted(objects.keys())")
        assert defs["ok"] and defs["repr"] == "['bench', 'crate']", (
            "the catalog the build produced must be probeable by name")

        shot = h["snapshot"](views=[{"on": "all", "azimuth": 45,
                                     "elevation": 30}])
        assert shot["ok"] and shot["paths"]

        assert h["finish"]()["ok"]
        assert (tmp_path / "ir.json").is_file()
        assert (tmp_path / "catalog" / "assets").is_dir(), "bundle staged"
        # `final_render` must not hand a SceneIR to the object URDF exporter.
        rendered = h["final_render"]()
        assert rendered["ok"], rendered


def test_the_import_refusal_does_not_enumerate_layers():
    """The message must not name the layers it applies to.

    The prompts say nothing about the sandbox on purpose, so this refusal is
    where an agent that assumed ordinary Python learns the edge — it has to
    name what replaces the import. Describing the surface by what the source
    *does*, rather than listing the layers it applies to, keeps it correct for
    every layer and stable against the next one.
    """
    from dycraft.agent.cad import sandbox as sb
    from dycraft.sdk._errors import SandboxViolation

    with pytest.raises(SandboxViolation) as ei:
        sb.check_ast("import numpy as np\n", mode="exec")
    repair = ei.value.to_dict()["suggested_repairs"][0]

    for handle in ("bd", "math", "np"):
        assert f"`{handle}`" in repair, (
            f"the refusal must name what replaces the import: {repair}")
    for layer in ("object", "scene", "animation", "plain"):
        assert layer not in repair, (
            f"the refusal names the {layer!r} layer; a list of layers goes "
            f"stale with the next layer: {repair}")


def test_the_geometry_handles_are_exactly_where_the_refusal_says():
    """The refusal claims geometry-authoring source gets these; check it.

    A message that names handles the source does not have is worse than a
    vague one — it sends the agent looking for a name that is not there.
    """
    from dycraft.agent.layers.layer import (object_layer_config,
                                                  scene_layer_config)

    class _Payload:
        object_handles: dict = {}
        catalog: dict = {}

    class _Ctx:
        run_dir = None
        step = 0
        layer_payload = _Payload()

    handles = {"bd", "math", "np", "OCP"}
    for cfg in (object_layer_config(), scene_layer_config()):
        overlay = set(cfg.build_overlay_factory(_Ctx()))
        assert handles <= overlay, (
            f"{cfg.name} authors geometry and must have {handles - overlay}")


def test_collision_advice_omits_placeholder_mechanics_the_layer_lacks(tmp_path):
    """A scene run has no slots, so slot advice sends it hunting.

    `NoCrossInstancePenetration` explains that unbuilt pairs are masked and
    that layout allowances are stripped once a slot lands. Both would describe
    slot staging, which this layer does not have: it authors every object it
    places, so an author reading them would look for a slot system that does
    not exist.
    """
    from dycraft.agent.layers import compile as C
    from dycraft.agent.layers import tools as T

    src = '''
def box(n, s):
    o = Object(n)
    o.declare_root(o.add_part("b", Box(s, s, s)))
    return o.finalize()
scene = Scene("s"); scene.set_room(6.0, 5.0, 3.0)
scene.place(box("a", 0.6), name="a1", pose=((0, 0, 0), (0, 0, 0)))
scene.place(box("b", 0.6), name="b1", pose=((0.2, 0, 0), (0, 0, 0)))
ir = scene.finalize()
'''
    from dycraft.agent.layers.scene_context import SceneCatalog
    (tmp_path / "scene.py").write_text(src)
    ctx = _scene_ctx(tmp_path)
    payload = SceneCatalog(tmp_path)
    payload.bind(ctx)
    ctx.layer_payload = payload
    build = T.build_current_ir(ctx)
    report = C.compile_with_layer(src, layer_cfg=ctx.layer_cfg, ctx=ctx,
                                  prebuilt=build)

    collision = next(r for r in report["verifier_reports"]
                     if r["name"] == "NoCrossInstancePenetration")
    assert collision["verdict"] == "fail", "the fixture must actually collide"
    advice = " ".join(collision["suggested_repairs"]).lower()
    assert advice, "a failure must still say what to do about it"
    for dead in ("unbuilt slot", "placeholder", "orchestrator"):
        assert dead not in advice, (
            f"advice mentions {dead!r}, which does not exist here: {advice}")


def test_a_lone_definition_is_not_asked_to_answer_the_scene_s_brief():
    """At the level an author sees it: a review of one definition in a scene
    run is not asked whether it answers the scene's brief.

    Asked of a chair's `backrest`, the reviewer would answer `prompt_match:
    not_met — "the visible object is only a backrest"` against a scene that
    answers its brief. It would be right about the image it was shown; the
    question should not be put to it.
    """
    from dycraft.agent.layers.review import _review_prompt

    brief = _review_prompt(
        [_spec("backrest", "object", ["it has ventilation slots"], "part")],
        user_prompt="a Unitree G1 unpacking a flat-pack office chair")
    assert "prompt_match" not in brief
    assert "was built for" not in brief


def test_an_object_run_is_still_asked_about_its_own_object():
    """The case the arity rule existed to serve, kept."""
    from dycraft.agent.layers.review import _review_prompt

    brief = _review_prompt(
        [_spec("chair", "object", ["four legs reach the floor"], "part",
               submission=True)],
        user_prompt="a sturdy dining chair")
    assert "prompt_match" in brief
    assert "the object was built for" in brief


def test_the_layers_mark_their_submission_spec():
    """`submission` has to be set where the spec is built, because the object
    layer's object and a scene run's definition are otherwise identical."""
    import inspect
    from dycraft.agent.layers import review as RV
    src = inspect.getsource(RV._review_targets)
    # The scene spec and the object-layer spec carry it; the definition spec
    # inside the scene-building loop does not.
    assert src.count('"submission": True') == 2, (
        "exactly the object-layer object and the scene should be submissions")
def test_the_scene_namespace_binds_both_layers_material_vocabulary():
    """`@lighting` and `@shader` must both be reachable from `scene.py`.

    The scene overlay is the object overlay plus the scene half, and the two
    halves own different parts of the appearance vocabulary: `shader` comes
    from the object side, `lighting` only from the scene side. Without
    `lighting`, `scene.set_lighting` — the *only* lighting surface there is
    (`Scene.set_lighting`: "There is no add_light, set_sky, set_environment or
    set_auto_lighting") — would be unreachable and every scene would render
    unlit. The AST sandbox rejects `import`, so absent from this dict means
    absent.
    """
    from dycraft.agent.layers.layer import _scene_build_overlay
    from dycraft.sdk._lighting_block import lighting

    class _Ctx:
        run_dir = None
        layer_payload = type("P", (), {"object_handles": {},
                                       "asset_base_dir": None})()

    overlay = _scene_build_overlay(_Ctx())
    for name in ("shader", "lighting", "Material", "Emission"):
        assert name in overlay, f"{name} is unreachable in a scene scene.py"
    assert overlay["lighting"] is lighting, (
        "the scene overlay must bind the SDK's own @lighting decorator")


def test_every_authoring_layer_binds_its_own_direction_producer():
    """`Direction` must be reachable wherever assertions are authored.

    Two documented capabilities need it. `assert_flush`: `_sugar` requires
    `axis_world` to be a DirectionRef whose source is `"world"` — it refuses a
    bare tuple and refuses a joint-axis or anchor-outward direction — so
    without `Direction` the whole sugar is unreachable. The angle family: with
    no world direction to compare against, an agent can relate two anchor
    normals to each other but cannot constrain either against a world axis.

    Sibling of `test_the_scene_namespace_binds_both_layers_material_vocabulary`,
    for the shared half rather than the scene half.
    """
    from dycraft.agent.layers.layer import (_object_build_overlay,
                                                  _scene_build_overlay)

    class _Ctx:
        run_dir = None
        step = 0
        layer_payload = type("P", (), {"object_handles": {}, "catalog": {},
                                       "asset_base_dir": None})()

    for factory in (_object_build_overlay, _scene_build_overlay):
        overlay = factory(_Ctx())
        assert "Direction" in overlay, (
            f"{factory.__name__} authors assertions but cannot name a world "
            f"axis; assert_flush and the angle family are unreachable")


def test_an_ops_catalog_is_the_build_it_made_not_a_concurrent_ops(tmp_path):
    """A compile verifying build A reads A's catalog even when a snapshot
    rebuilds edited source (B) in the middle. Read through the run's shared
    build cache, it would get B's, and an instance of A would resolve its
    `object_ref` against B and miss (`KeyError: '<hash>'`). Each op reads the
    build it produced or reused."""
    import threading
    from types import SimpleNamespace
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers.scene_context import SceneCatalog

    ctx = SimpleNamespace(run_dir=tmp_path, _current_build_cache=None)
    payload = SceneCatalog(tmp_path)
    payload.bind(ctx)

    def raw(ref):
        return {"ok": True, "ir": {"instances": [{"object_ref": ref}]},
                "build_error": None, "sandbox_violation": None,
                "object_catalog": {ref: {"name": ref}}}

    built, swapped, seen = threading.Event(), threading.Event(), {}

    def compile_op():
        with T.build_pin_scope():
            T._cache_build(ctx, "source-A", raw("hash-A"))
            built.set()
            swapped.wait(5)
            seen["compile"] = sorted(payload.catalog)

    def snapshot_op():
        built.wait(5)
        with T.build_pin_scope():
            T._cache_build(ctx, "source-B", raw("hash-B"))
            seen["snapshot"] = sorted(payload.catalog)
        swapped.set()

    threads = [threading.Thread(target=compile_op), threading.Thread(target=snapshot_op)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert seen == {"compile": ["hash-A"], "snapshot": ["hash-B"]}
    # Outside any op, the run's latest build, as before.
    assert sorted(payload.catalog) == ["hash-B"]
    # And an op that ended leaves nothing pinned on its pooled thread.
    with T.build_pin_scope():
        assert T.pinned_build(ctx) is None


def test_a_pin_never_answers_for_another_context(tmp_path):
    """A context made after another was pinned and dropped reads no pin, even
    at the dropped one's address. Pins are keyed by `id(ctx)`, and CPython
    hands a new object of the same size the address the last one freed, so a
    bare id key would give the new context the earlier one's build."""
    from dycraft.agent.layers import tools as T

    stale = {"ok": True, "object_catalog": {"h": {"name": "stale"}}}
    with T.build_pin_scope():
        for _ in range(20):
            old = _FakeCtx(tmp_path)
            T._pin(old, stale)
            del old
            assert T.pinned_build(_FakeCtx(tmp_path)) is None


def test_every_op_runs_inside_the_sessions_op_scope():
    """The pin is scoped by `AgentSession.op`: entering an op clears whatever an
    earlier op left on a pooled thread."""
    from contextlib import contextmanager
    from dycraft.agent.agent_session import AgentSession

    entered = []

    class Probe(AgentSession):
        @contextmanager
        def op_scope(self):
            entered.append("in")
            yield
            entered.append("out")

    session = Probe.__new__(Probe)
    session.handlers = {}
    session.tick = lambda: 1
    session.trace = lambda *a, **k: None
    session.op("noop")(lambda: {"ok": True})
    assert session.handlers["noop"]() == {"ok": True}
    assert entered == ["in", "out"]
