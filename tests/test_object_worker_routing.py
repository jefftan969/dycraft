"""Where an `object.py` executes, and what happens when it cannot.

The object layer's build runs in the frozen CAD environment. These tests pin
three properties of that split:

  * **object builds in the worker.** `build_current_ir` and
    `run_compile_and_persist` share one routing helper, and the helper is the
    only place that decides, so probe, snapshot, compile and finish can never
    disagree about which interpreter ran the source.
  * **there is no fallback.** Without a CAD interpreter the run refuses to
    start and a build returns a worker failure. It must never quietly produce
    primitive-only geometry, which would commit an object that looks fine and
    is not the one that was authored.
  * **verifiers still run here.** Only the build runs in the worker; everything
    downstream of the IR stays in the main process.

The end-to-end cases need the real CAD interpreter and are marked expensive.
The routing and failure-mode cases stub the worker and run everywhere.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.agent.layers import tools as T
from dycraft.agent.layers.compile import build_ir_with_layer
from dycraft.agent.layers.layer import object_layer_config

_needs_cad = pytest.mark.skipif(
    not W.is_available(), reason="no build123d interpreter")


LIVE_SRC = (
    'obj = Object("routed")\n'
    'b = obj.add_part("b", Box(0.2, 0.2, 0.05))\n'
    'obj.declare_root(b)\n'
    'ir = obj.finalize()\n'
)

BOX_SRC = (
    'obj = Object("boxes")\n'
    'base = obj.make_box("base", width=0.4, length=0.3, height=0.04)\n'
    'post = obj.make_box("post", width=0.04, length=0.04, height=0.5)\n'
    'obj.declare_root(base)\n'
    'obj.attach_fixed("j", parent=base, child=post,\n'
    '                 parent_anchor=obj.anchor(base, "top"),\n'
    '                 child_anchor=obj.anchor(post, "bottom"))\n'
    'ir = obj.finalize()\n'
)


def _ctx(layer_cfg, tmp_path=None):
    d = Path(tmp_path or tempfile.mkdtemp())
    (d / "compile_reports").mkdir(parents=True, exist_ok=True)
    return T.ToolContext(run_dir=d, layer_cfg=layer_cfg)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@_needs_cad
@pytest.mark.expensive
def test_object_layer_builds_in_the_worker(monkeypatch):
    seen = {}
    orig = W.run_object_source

    def spy(source, **kw):
        seen["run_dir"] = kw.get("run_dir")
        return orig(source, **kw)

    monkeypatch.setattr(W, "run_object_source", spy)
    ctx = _ctx(object_layer_config())
    out = T._build_ir_routed(ctx, LIVE_SRC)
    assert out["ok"], out
    assert seen["run_dir"] == ctx.run_dir
    assert out["ir"]["parts"][0]["primitive"] == "solid"


@_needs_cad
@pytest.mark.expensive
def test_the_worker_interpreter_is_the_pinned_build123d():
    """The whole point of the process split is the frozen version."""
    from dycraft.agent.cad import _solid_compile
    interp = W.b123d_python()
    assert interp, "no CAD interpreter resolved"
    import subprocess
    from dycraft.agent.cad._cad_env import b123d_child_env
    out = subprocess.run(
        [interp, "-c", "import build123d; print(build123d.__version__)"],
        capture_output=True, text=True, env=b123d_child_env(), timeout=180)
    assert out.stdout.strip() == _solid_compile.REQUIRED_BUILD123D, out


# ---------------------------------------------------------------------------
# No fallback
# ---------------------------------------------------------------------------

def test_no_cad_interpreter_is_a_build_failure_not_a_fallback(monkeypatch):
    """The failure mode that must never exist: a run that quietly degrades.

    The object layer has no in-process path, so an absent interpreter has
    exactly one correct answer — a failure the agent can read. Falling back to
    coarser geometry would commit an object that looks fine and is not what was
    asked for.
    """
    monkeypatch.setenv("DYCRAFT_B123D_PYTHON", "")
    ctx = _ctx(object_layer_config())
    out = T._build_ir_routed(ctx, LIVE_SRC)
    assert out["ok"] is False
    assert out["ir"] is None
    assert out["error_kind"] == "worker_unavailable"
    assert out["build_error"]["suggested_repairs"], "must tell the agent what to do"


def _object_session(run_dir):
    from dycraft.agent.layers.session import LayerSession, SessionConfig
    return LayerSession(SessionConfig(
        run_dir=run_dir, layer=object_layer_config(), layer_payload=None,
        user_prompt="x", initial_meta={}))


def _film_session(run_dir):
    from dycraft.agent.layers.film import FilmSession
    return FilmSession(run_dir, user_prompt="x", initial_meta={})


@pytest.mark.parametrize("build", [_object_session, _film_session], ids=["object", "animation"])
def test_the_session_refuses_to_start_without_a_cad_interpreter(monkeypatch, build):
    """Fail before the agent is handed a prompt telling it to author CAD. The
    film agent's scene source is build123d too, so its session refuses alike."""
    monkeypatch.setenv("DYCRAFT_B123D_PYTHON", "")
    d = Path(tempfile.mkdtemp())
    (d / "compile_reports").mkdir()
    with pytest.raises(RuntimeError, match="no build123d interpreter"):
        build(d)


def test_the_repair_table_answers_every_declared_failure_kind():
    """The two halves of one vocabulary, checked against each other.

    A kind with no entry reaches the agent with an empty `suggested_repairs`,
    which reads as "nothing to be done" rather than as a gap.

    `FAILURE_KINDS` is the producer's declaration, so this compares the answers
    to the questions rather than to another copy of the answers.
    """
    assert set(T._WORKER_REPAIRS) == set(W.FAILURE_KINDS), (
        "every worker failure kind needs a repair, and every repair needs a "
        "kind that something can actually produce")


@pytest.mark.parametrize("kind", W.FAILURE_KINDS)
def test_every_worker_error_kind_reaches_the_agent_with_repairs(kind, monkeypatch):
    """A geometry fault and an environment fault demand opposite responses.

    Without the distinction an agent answers a broken CAD environment by
    rewriting perfectly good geometry, forever.
    """
    monkeypatch.setattr(W, "run_object_source", lambda *a, **k: {
        "ok": False, "ir": None, "sandbox_violation": None,
        "build_error": {"error": "X", "summary": "s", "evidence": {},
                        "refs": [], "suggested_repairs": []},
        "error_kind": kind, "message": "s",
    })
    out = T._build_ir_routed(_ctx(object_layer_config()), LIVE_SRC)
    assert out["ok"] is False
    assert out["build_error"]["suggested_repairs"], f"{kind} carries no repair"


# ---------------------------------------------------------------------------
# Compatibility + the pieces that must NOT move
# ---------------------------------------------------------------------------

@_needs_cad
@pytest.mark.expensive
def test_a_box_only_source_routes_to_the_same_ir_it_builds_in_process():
    """The worker must be a faithful execution environment, not a variant.

    A box-only source touches no CAD machinery, so both paths must produce
    identical IR — same content hash. A divergence here is silent,
    repository-wide content-hash churn.
    """
    routed = T._build_ir_routed(_ctx(object_layer_config()), BOX_SRC)

    class _Ctx:
        def __init__(self, d):
            self.run_dir = Path(d)
            self.step = 0

    inproc = build_ir_with_layer(BOX_SRC, layer_cfg=object_layer_config(),
                                 ctx=_Ctx(tempfile.mkdtemp()))
    assert routed["ok"] and inproc["ok"]
    assert routed["ir"]["id"] == inproc["ir"]["id"]
    assert routed["ir"] == inproc["ir"]


@_needs_cad
@pytest.mark.expensive
def test_verifiers_still_run_in_this_process_after_the_build_moves():
    ctx = _ctx(object_layer_config())
    (ctx.run_dir / "object.py").write_text(LIVE_SRC)
    payload = T.run_compile_and_persist(ctx, is_finish=False)
    assert payload["ok"], payload
    names = {r["name"] for r in payload["verifier_reports"]}
    assert "NoSelfPenetration" in names and "SupportConnected" in names


@_needs_cad
@pytest.mark.expensive
def test_compile_then_probe_pays_for_one_worker_launch(monkeypatch):
    """The CAD subprocess costs seconds; compile and probe share one build."""
    calls = []
    orig = W.run_object_source

    def spy(source, **kw):
        calls.append(1)
        return orig(source, **kw)

    monkeypatch.setattr(W, "run_object_source", spy)
    ctx = _ctx(object_layer_config())
    (ctx.run_dir / "object.py").write_text(LIVE_SRC)
    T.run_compile_and_persist(ctx, is_finish=False)
    T.run_probe(ctx, "list_parts(ir)")
    T.run_compile_and_persist(ctx, is_finish=False)
    assert len(calls) == 1, f"expected one worker launch, got {len(calls)}"


@_needs_cad
@pytest.mark.expensive
def test_editing_the_source_invalidates_the_shared_build():
    ctx = _ctx(object_layer_config())
    (ctx.run_dir / "object.py").write_text(LIVE_SRC)
    first = T.run_compile_and_persist(ctx, is_finish=False)["ir_hash"]
    (ctx.run_dir / "object.py").write_text(
        LIVE_SRC.replace("0.2, 0.2, 0.05", "0.3, 0.2, 0.05"))
    second = T.run_compile_and_persist(ctx, is_finish=False)["ir_hash"]
    assert first and second and first != second


def test_an_environment_failure_is_not_cached_against_the_source():
    """A dead worker is a property of the machine, not of `object.py`.

    The build cache answers "the source has not changed, so neither has the
    build". True of a geometry error; false of a crash, a timeout or a missing
    interpreter — and caching those made one transient fault answer every
    later probe, view and compile of an unchanged source, so the author's only
    escape was an edit that changed nothing but the hash.
    """
    from dycraft.agent.layers import tools as T
    from dycraft.agent.cad import object_worker as OW

    for kind in ("worker_crash", "timeout", "worker_unavailable",
                 "worker_internal", "build123d_version_mismatch",
                 "ocp_missing"):
        assert kind in OW.FAILURE_KINDS, (
            f"{kind!r} is classified as an environment failure but is not a "
            f"registered worker kind")
        assert T._is_environment_failure({"ok": False, "error_kind": kind})

    # A geometry error IS a function of the source and stays cacheable.
    for kind in ("invalid_shape", "empty_shape", "tessellation_failure", None):
        assert not T._is_environment_failure({"ok": False, "error_kind": kind})
    assert not T._is_environment_failure({"ok": True, "error_kind": "timeout"})
