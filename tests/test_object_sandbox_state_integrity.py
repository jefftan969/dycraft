"""What sandboxed code may *write*.

The previous pass closed the escape axis: no modules, no filesystem, no process,
no network, no unrestricted reflection. It left the integrity axis open, and the
reason is structural rather than an oversight — it tried to protect mutable
injected state with an AST rule about names, and mutation needs no distinctive
syntax:

    objects.clear()            # no subscript, no attribute write, no banned node
    h = objects; h['x'] = 1    # an alias; the AST sees a local name
    def f(t): t['x'] = 1
    f(objects)                 # a parameter; the AST sees a local name

Without the read-only view, on the real runtime paths:

  * `objects` is the live handle dict for a scene run — the same instance every
    exec — so `objects.clear()` would empty a run's catalog permanently, with
    no error reported anywhere.
  * `ir` in a probe is the dict held in `ctx._current_build_cache`, so
    `probe("ir['parts'].clear()")` would corrupt it. The silent variant is
    worse: `probe("ir['parts'][0]['material'].update(...)")` then `compile()`
    would return `ok: True` with every verifier passing and an `ir_hash`
    computed *before* the mutation — the content-addressed identity broken with
    no signal.

`sdk/_readonly.py` closes this: the sandbox never receives the mutable original.
These tests pin the invariant rather than individual reproductions, because the
point is that aliases, parameters, nested values, and mutating methods close *by
construction*.

Invariants cited by name:
  I5 — sandboxed code cannot mutate host state that outlives its execution
  I6 — sandboxed code cannot consume unbounded wall-clock in the daemon
  I7 — every failure is classified accurately and the daemon survives it
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from dycraft.agent.cad import _timelimit
from dycraft.agent.layers.compile import build_ir_with_layer
from dycraft.agent.layers.layer import object_layer_config
from dycraft.agent.layers.tools import ToolContext, run_probe, run_compile_and_persist
from dycraft.sdk._errors import SandboxViolation
from dycraft.sdk._readonly import freeze

_OBJECT_SRC = (
    "obj = Object('t')\n"
    "a = obj.make_box('a', 0.1, 0.1, 0.1)\n"
    "b = obj.make_box('b', 0.1, 0.1, 0.1)\n"
    "obj.declare_root(a)\n"
    "obj.attach_fixed('j', parent=a, child=b,\n"
    "                 parent_anchor=obj.anchor(a, 'right'),\n"
    "                 child_anchor=obj.anchor(b, 'left'))\n"
    "ir = obj.finalize()\n"
)

@pytest.fixture
def object_ctx(tmp_path):
    ctx = ToolContext(run_dir=tmp_path, layer_cfg=object_layer_config())
    ctx.source_path().write_text(_OBJECT_SRC)
    return ctx


# ---------------------------------------------------------------------------
# I5 — the invariant, at the level of the mechanism
# ---------------------------------------------------------------------------

def test_a_frozen_view_refuses_every_mutating_method():
    """Enumerated rather than sampled: the guarantee is that the *set* is
    closed, and a sampled test would pass while `popitem` stayed open."""
    v = freeze({"a": [1, 2], "b": {"c": 3}}, "state")
    for call in ("clear", "update", "setdefault", "pop", "popitem"):
        with pytest.raises(SandboxViolation):
            getattr(v, call)()
    for call in ("clear", "append", "extend", "insert", "remove", "sort",
                 "reverse", "pop"):
        with pytest.raises(SandboxViolation):
            getattr(v["a"], call)()


def test_freezing_reaches_arbitrary_depth():
    """A view that froze only the top level would leave every nested dict and
    list of an IR writable — which is where an IR's data lives."""
    real = {"parts": [{"material": {"color": [1, 2, 3]}}]}
    v = freeze(real, "ir")
    with pytest.raises(SandboxViolation):
        v["parts"][0]["material"]["color"].clear()
    with pytest.raises(SandboxViolation):
        v["parts"][0]["material"].update({"color": [9, 9, 9]})
    assert real["parts"][0]["material"]["color"] == [1, 2, 3]


def test_an_alias_is_the_same_frozen_view():
    """The AST rule could not see this; the runtime does not need to."""
    v = freeze({"a": 1}, "state")
    alias = v
    with pytest.raises(SandboxViolation):
        alias.clear()

    def via_parameter(t):
        t["x"] = 1
    with pytest.raises(SandboxViolation):
        via_parameter(v)


def test_reads_are_unimpaired():
    real = {"parts": [{"name": "a"}], "id": "abc", "n": 2}
    v = freeze(real, "ir")
    assert v["id"] == "abc"
    assert len(v) == 3 and "parts" in v
    assert v.get("id") == "abc" and v.get("missing") is None
    assert sorted(v.keys()) == ["id", "n", "parts"]
    assert len(list(v.items())) == 3 and len(list(v.values())) == 3
    assert v["parts"][0]["name"] == "a"
    assert [p["name"] for p in v["parts"]] == ["a"]
    assert v == real            # compares equal to the dict it views
    assert repr(v) == repr(real)


def test_freezing_leaves_the_original_writable():
    """The harness must keep mutating and serializing its own data — the build
    cache, ir.json, the verifier suite all operate on the real dict."""
    import json
    real = {"parts": [{"name": "a"}]}
    freeze(real, "ir")
    real["parts"].append({"name": "b"})          # harness side: still fine
    assert len(real["parts"]) == 2
    assert json.loads(json.dumps(real))["parts"][1]["name"] == "b"


def test_strings_are_not_wrapped():
    """`str` is a Sequence; wrapping it would break every f-string and name."""
    v = freeze({"name": "chair"}, "ir")
    assert v["name"].upper() == "CHAIR"
    assert v["name"][1:3] == "ha"


# ---------------------------------------------------------------------------
# I5 — object probe, on the real path
# ---------------------------------------------------------------------------

_PROBE_ATTACKS = (
    "ir.clear()",
    "ir['parts'].clear()",
    "ir['joints'].clear()",
    "ir['parts'].append(1)",
    "ir.update({'parts': []})",
    "ir.setdefault('x', 1)",
    "ir.pop('parts')",
    "ir['parts'][0].update({'name': 'hacked'})",
    "ir['parts'][0]['material'].update({'color': [255, 0, 0]})",
    "ir['parts'][0]['params'].clear()",
)


@pytest.mark.parametrize("attack", _PROBE_ATTACKS)
def test_no_probe_expression_can_mutate_the_cached_ir(object_ctx, attack):
    """I5. `eval` mode prevents none of these — none is an assignment."""
    assert run_probe(object_ctx, "len(ir['parts'])")["repr"] == "2"
    out = run_probe(object_ctx, attack)
    assert not out["ok"], f"{attack} was accepted"
    assert out["sandbox_violation"]["code"] == "SandboxViolation"
    assert run_probe(object_ctx, "len(ir['parts'])")["repr"] == "2"


def test_a_probe_cannot_contaminate_a_later_compile(object_ctx):
    """I5, end to end — the silent case.

    A mutating probe that got through would leave the compile and every
    verifier passing, with an ir_hash computed before the mutation.
    """
    clean = run_compile_and_persist(object_ctx, is_finish=False)
    assert clean["ok"]
    baseline_hash = clean["ir_hash"]

    denied = run_probe(object_ctx,
                       "ir['parts'][0]['material'].update({'color': [255, 0, 0]})")
    assert not denied["ok"]

    after = run_compile_and_persist(object_ctx, is_finish=False)
    assert after["ok"]
    assert after["ir_hash"] == baseline_hash
    cached = object_ctx._current_build_cache["result"]["ir"]
    assert cached["parts"][0]["material"]["color"] != [255, 0, 0]


def test_every_object_probe_helper_still_works_against_a_frozen_ir(object_ctx):
    """The compatibility half. A frozen `ir` that broke the probe library would
    have traded one bug for a worse one."""
    run_probe(object_ctx, "1")
    for expr in ("part_shape(ir,'a')", "object_source()", "list_parts(ir)",
                 "list_joints(ir)", "joint_table(ir)", "local_bbox(ir,'a')",
                 "frame_of(ir,'a')", "global_bbox(ir)", "centroid(ir)",
                 "part_color(ir,'a')", "penetration_pairs(ir)",
                 "anchor_position(ir,'a','top')", "outward(ir,'a','top')",
                 "ir_snapshot(ir)"):
        assert run_probe(object_ctx, expr)["ok"], expr


# ---------------------------------------------------------------------------
# The scene layer's probe surface (`objects`, `catalog`)
# ---------------------------------------------------------------------------

@pytest.fixture
def scene_ctx(tmp_path, monkeypatch):
    """A scene-layer ToolContext whose catalog holds two built objects.

    The build itself is stubbed: what is under test is the probe overlay
    (`_scene_layer_probe_overlay` on top of `_scene_probe_overlay`) and the probe
    evaluation path, not the CAD worker.
    """
    from dycraft.agent.layers import tools as T
    from dycraft.agent.layers.layer import scene_layer_config
    from dycraft.agent.layers.scene_context import SceneCatalog
    from dycraft.sdk import ObjectHandle
    from dycraft.sdk.object_builder import Object, _reset_object_singleton

    catalog = {}
    for name in ("chair", "dining_table"):
        _reset_object_singleton(limit=None)
        obj = Object(name)
        obj.declare_root(obj.make_box("body", 0.5, 0.5, 0.5))
        ir = obj.finalize()
        catalog[ir["id"]] = ir
    _reset_object_singleton(limit=1)

    payload = SceneCatalog(tmp_path / "run")
    payload._handles = {ir["name"]: ObjectHandle(ir) for ir in catalog.values()}
    monkeypatch.setattr(SceneCatalog, "_derive", lambda self: None)
    monkeypatch.setattr(SceneCatalog, "catalog",
                        property(lambda self: catalog))
    ctx = ToolContext(run_dir=tmp_path / "run", layer_cfg=scene_layer_config(),
                      layer_payload=payload)
    scene_ir = {"id": "s", "schema": "dycraft.scene.v2", "instances": [],
                "groups": [], "assertions": []}
    monkeypatch.setattr(T, "build_current_ir",
                        lambda c: {"ok": True, "ir": scene_ir})
    return ctx, payload


_OBJECTS_ATTACKS = (
    "objects.clear()",
    "objects.update({'ghost': None})",
    "objects.pop('chair', None)",
    "objects.setdefault('ghost', None)",
    "h = objects\nh['pwned'] = 1",
    "h = objects\nh.clear()",
    "def f(t):\n    t['x'] = 1\nf(objects)",
    "for h in [objects]:\n    h.clear()",
    "[o.clear() for o in [objects]]",
    "catalog.clear()",
    "catalog.popitem()",
)


@pytest.mark.parametrize("attack", _OBJECTS_ATTACKS)
def test_scene_probes_cannot_mutate_objects_or_catalog(scene_ctx, attack):
    """`objects` and `catalog` are frozen views of the run's built catalog; a
    probe that could mutate them would change what later probes and verifiers
    see."""
    ctx, payload = scene_ctx
    out = run_probe(ctx, attack)
    assert not out["ok"], f"{attack!r} was accepted"
    assert sorted(payload.object_handles) == ["chair", "dining_table"]
    assert len(payload.catalog) == 2


@pytest.mark.parametrize("src", [
    "d = dict(objects)\nd['mine'] = 1",              # a copy the probe owns
    "vs = list(objects.values())\nvs.clear()",
    "names = sorted(objects.keys())\nnames.append('x')",
    "picked = {n: objects[n] for n in objects}\npicked.clear()",
    "cols = objects['chair'].colors\ncols.clear()",  # property builds a fresh dict
])
def test_copies_of_injected_state_stay_writable(scene_ctx, src):
    """Reading `objects` into a container of your own and editing *that* is
    ordinary use, not an attack; refusing it would be a regression."""
    ctx, payload = scene_ctx
    out = run_probe(ctx, src)
    assert out["ok"], out
    assert sorted(payload.object_handles) == ["chair", "dining_table"]
    assert len(payload.object_handles["chair"].colors) > 0


def test_object_handle_shape_tables_are_frozen():
    """`ObjectHandle.part_shape` returns `ir['parts'][i]['params']['shape']`
    itself, so an unwrapped return let a scene source edit the catalog IR
    through `objects['x'].part_shape('p').clear()`."""
    from dycraft.sdk.scene_builder import ObjectHandle
    ir = {"schema": "dycraft.object.v1", "id": "x", "name": "o", "root": "p",
          "joints": [],
          "parts": [{"id": 0, "name": "p", "primitive": "solid",
                     "params": {"shape": {"volume": 1.0}},
                     "material": {"color": [1, 2, 3]}}]}
    h = ObjectHandle(ir)
    with pytest.raises(SandboxViolation):
        h.part_shape("p").clear()
    with pytest.raises(SandboxViolation):
        h.part_shapes["p"].update({"volume": 0.0})
    assert ir["parts"][0]["params"]["shape"] == {"volume": 1.0}


# ---------------------------------------------------------------------------
# I6 — execution bounds inside the daemon
# ---------------------------------------------------------------------------

def test_a_runaway_probe_expression_is_bounded(object_ctx, monkeypatch):
    """I6. A probe evaluates in the daemon too."""
    assert run_probe(object_ctx, "len(ir['parts'])")["ok"]   # warm the cache
    monkeypatch.setenv(_timelimit.LIMIT_ENV_VAR, "2")
    t0 = time.time()
    out = run_probe(object_ctx, "sum(1 for _ in range(10**10))")
    assert time.time() - t0 < 30
    assert not out["ok"] and "did not finish" in out["error"]
    # The real elapsed time rides beside the limit: they differ whenever one
    # native call outruns the limit, and "346 s against 300 s" is the lesson
    # that "did not finish within 300s" hid.
    assert out["limit_s"] == 2.0 and out["elapsed_s"] >= 2.0
    # And the session keeps working afterwards.
    monkeypatch.delenv(_timelimit.LIMIT_ENV_VAR)
    assert run_probe(object_ctx, "len(ir['parts'])")["ok"]


def test_the_limit_works_off_the_main_thread():
    """The reason this is not `signal.setitimer`.

    `agent/server.py` services handlers from a ThreadPoolExecutor, and signal
    handlers only run on the main thread — a signal-based limit would pass its
    own tests and bound nothing in production.
    """
    import threading
    result = {}

    def worker():
        try:
            with _timelimit.time_limit(1.0):
                while True:
                    pass
        except _timelimit.SandboxTimeout:
            result["bounded"] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=20)
    assert result.get("bounded"), "no bound on a non-main thread"


def test_the_limit_does_not_leak_into_later_work():
    """An async exception queued for a thread that has already left the block
    would surface at an arbitrary later point, inside harness code."""
    try:
        with _timelimit.time_limit(0.3):
            time.sleep(1.0)      # C-level: the exception queues, undelivered
    except _timelimit.SandboxTimeout:
        pass
    total = 0
    for i in range(2_000_000):   # would blow up on a stale pending exception
        total += i
    assert total > 0


def test_normal_work_is_not_interrupted(object_ctx):
    assert run_probe(object_ctx, "len(ir['parts'])")["ok"]
    out = run_compile_and_persist(object_ctx, is_finish=False)
    assert out["ok"], out


# ---------------------------------------------------------------------------
# I7 — deliberate failure handling
# ---------------------------------------------------------------------------

def test_system_exit_is_a_policy_breach_not_a_typo(monkeypatch, tmp_path):
    """I7. `SourceRuntimeError: object.py raised SystemExit` would call an
    attempt to end the host a mistake in the program."""
    from dycraft.agent.cad import sandbox as sb

    class _Exits:
        def __call__(self, *a, **k):
            raise SystemExit(3)

    real = sb.build_globals
    monkeypatch.setattr(
        sb, "build_globals",
        lambda overlay=None: {**real(overlay=overlay), "Object": _Exits()})

    class _Ctx:
        run_dir = tmp_path
        step = 0

    out = build_ir_with_layer("obj = Object('t')\nir = None\n",
                             layer_cfg=object_layer_config(), ctx=_Ctx())
    assert out["build_error"] is None
    assert out["sandbox_violation"]["evidence"]["node_type"] == "SystemExit"


def test_keyboard_interrupt_is_not_absorbed(monkeypatch, tmp_path):
    """I7. Swallowing it would make Ctrl-C during an in-process compile do nothing."""
    from dycraft.agent.cad import sandbox as sb

    class _Interrupts:
        def __call__(self, *a, **k):
            raise KeyboardInterrupt

    real = sb.build_globals
    monkeypatch.setattr(
        sb, "build_globals",
        lambda overlay=None: {**real(overlay=overlay), "Object": _Interrupts()})

    class _Ctx:
        run_dir = tmp_path
        step = 0

    with pytest.raises(KeyboardInterrupt):
        build_ir_with_layer("obj = Object('t')\nir = None\n",
                            layer_cfg=object_layer_config(), ctx=_Ctx())


def test_the_classifier_survives_a_hostile_exception(tmp_path):
    """I7. The error path must not itself raise: an exception whose `__str__`
    fails would otherwise escape as an unhandled error from the tool."""
    from dycraft.agent.cad import sandbox as sb

    class _Nasty(Exception):
        def __str__(self):
            raise RuntimeError("no message for you")

    class _Raises:
        def __call__(self, *a, **k):
            raise _Nasty()

    real = sb.build_globals
    sb.build_globals = (
        lambda overlay=None: {**real(overlay=overlay), "Object": _Raises()})
    try:
        class _Ctx:
            run_dir = tmp_path
            step = 0

        out = build_ir_with_layer("obj = Object('t')\nir = None\n",
                                 layer_cfg=object_layer_config(), ctx=_Ctx())
    finally:
        sb.build_globals = real
    assert out["build_error"]["code"] == "SourceRuntimeError"
    assert "unavailable" in out["build_error"]["summary"]
