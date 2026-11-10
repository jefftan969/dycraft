"""Warm CAD workers must be indistinguishable from fresh ones.

`object_worker` stopped spawning an interpreter per build (see
`_worker_pool`'s module docstring for the measurement that motivated it). The
saving is real — ~1.4 s of `import build123d` per build — but it is only
allowed if reuse cannot change an answer, so that is what this file pins.

The load-bearing claim is *equivalence*: for the same source, a warm child and
a single-use child produce the same IR, down to the content hash. Everything
else here supports that claim by attacking the ways reuse could break it —
state carried between builds, a fault leaving a child in an unreasonable state,
and the failure vocabulary changing shape depending on which path ran.

Isolation across builds is not new machinery. `compile.build_ir_with_layer`
already resets the SDK singleton, the mesh-asset session and the live-Shape
compiler and hands the exec fresh globals on *every* call — that is how scene
builds and probes have always worked inside the long-lived daemon. These tests
hold the object layer to the same standard now that it reuses a process too.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from dycraft.agent.cad import _worker_pool
from dycraft.agent.cad import object_worker as W

pytestmark = pytest.mark.skipif(
    not W.is_available(),
    reason="needs a build123d interpreter (DYCRAFT_B123D_PYTHON / .b123d_python)")


def _src(name: str, x: float = 0.1) -> str:
    return (f'obj = Object("{name}")\n'
            f'b = obj.add_part("body", Box({x}, 0.2, 0.3))\n'
            'obj.declare_root(b)\n'
            'ir = obj.finalize()\n')


_ARTICULATED = (
    'obj = Object("art")\n'
    'a = obj.make_box("a", width=0.4, length=0.3, height=0.04)\n'
    'b = obj.make_box("b", width=0.04, length=0.04, height=0.5)\n'
    'obj.declare_root(a)\n'
    'obj.attach_fixed("j", parent=a, child=b,\n'
    '                 parent_anchor=obj.anchor(a, "top"),\n'
    '                 child_anchor=obj.anchor(b, "bottom"))\n'
    'ir = obj.finalize()\n'
)


@pytest.fixture(autouse=True)
def _clean_pool(request, monkeypatch):
    """Normalize the pool env for every test; tear the pool down for the few
    tests whose claim depends on its state.

    **The env half is unconditional.** Forcing the three pool variables to
    their defaults makes this file test pooling even when the ambient
    configuration has disabled it — otherwise running the suite with
    `DYCRAFT_OBJECT_WORKER_PERSISTENT=0` turns the equivalence tests below into
    a comparison of two identical one-shot runs, which passes while proving
    nothing.

    **The teardown half is opt-in, via `@pytest.mark.cold_pool`.** A fresh
    `import build123d` costs ~1.5 s, so only the tests whose claim depends on
    pool state pay it — the warm-vs-cold equivalence pairs, the ones reading `POOL._live` /
    `POOL._idle`, and the cap / idle-TTL / non-pooled paths. The rest assert
    things a warm child cannot affect (the stdio keys are present, a sandbox
    violation does not poison the next build, a snippet's stdout round-trips),
    and inheriting a warm child is not merely harmless for them but closer to
    production, where a child *is* reused across compiles.

    The marker is therefore load-bearing, not decoration: if a new test starts
    reading pool accounting or comparing against a cold start, it must carry
    `cold_pool` or it will silently measure whatever the previous test left
    checked in. `tests/conftest.py` registers it and says so.
    """
    monkeypatch.delenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", raising=False)
    monkeypatch.delenv("DYCRAFT_CAD_POOL_MAX_CHILDREN", raising=False)
    monkeypatch.delenv("DYCRAFT_CAD_POOL_IDLE_TTL_S", raising=False)
    cold = request.node.get_closest_marker("cold_pool") is not None
    if cold:
        W.shutdown_pool()
    yield
    if cold:
        W.shutdown_pool()


def _run(source: str, **kw) -> dict:
    return W.run_object_source(source, run_dir=Path(tempfile.mkdtemp()), **kw)


def _one_shot(source: str, monkeypatch, **kw) -> dict:
    monkeypatch.setenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", "0")
    W.shutdown_pool()
    return _run(source, **kw)


# ---------------------------------------------------------------------------
# Equivalence — the property that permits reuse at all
# ---------------------------------------------------------------------------

#: One source, not two. `_ARTICULATED` strictly dominates a single box — two
#: parts, a joint, anchors, a composed transform — so the box case bought a
#: second cold interpreter (~1.5 s) to re-check a subset of what this already
#: covers. The claim is unchanged: warm and cold produce the identical IR.
@pytest.mark.parametrize("source", [_ARTICULATED])
@pytest.mark.cold_pool
def test_a_warm_child_and_a_single_use_child_agree_exactly(source, monkeypatch):
    """Not merely equivalent — identical, including the content hash.

    A content-addressed IR means any divergence at all shows up here, so this
    does not have to guess which field reuse might perturb.
    """
    warm = _run(source)
    cold = _one_shot(source, monkeypatch)
    assert warm["ok"] and cold["ok"], (warm.get("message"), cold.get("message"))
    assert warm["ir"]["id"] == cold["ir"]["id"]
    assert warm["ir"] == cold["ir"]


@pytest.mark.cold_pool
def test_a_reused_child_matches_a_cold_one_build_for_build(monkeypatch):
    """A *sequence* on one child, against the same sequence on fresh children.

    Reuse could plausibly hold for one build and drift on the third — state
    accumulates, it does not usually appear at once. Running several different
    sources through a single warm child and comparing each answer to its
    cold-start counterpart is what catches that.
    """
    # Three builds: drift appears on the *third* build rather than the first,
    # and the same source re-run after unrelated work must still yield the
    # same id.
    sources = [_src("s1"), _ARTICULATED, _src("s1")]

    warm = [_run(s) for s in sources]
    cold = [_one_shot(s, monkeypatch) for s in sources]

    for i, (w, c) in enumerate(zip(warm, cold)):
        assert w["ok"] and c["ok"], f"build {i}: {w.get('message')} / {c.get('message')}"
        assert w["ir"] == c["ir"], f"build {i} diverged after reuse"

    # The first and last source are the same text, so the same IR must come
    # back after an unrelated build ran in between on that child.
    assert warm[0]["ir"]["id"] == warm[-1]["ir"]["id"]


def _pool_key() -> str:
    """The pool's identity for the authoring child, as the pool computes it."""
    return f"{W.b123d_python()}\0{W._CHILD_PATH}"


@pytest.mark.cold_pool
def test_only_one_child_served_the_whole_sequence():
    """The saving is real: several builds, one interpreter.

    Without this the equivalence tests above would still pass on a pool that
    silently spawned per build, which is the thing being removed.
    """
    for _ in range(4):
        assert _run(_src("reuse"))["ok"]
    idle = _worker_pool.POOL._idle.get(_pool_key(), [])
    assert len(idle) == 1, f"expected one warm child, found {len(idle)}"
    assert _worker_pool.POOL._live.get(_pool_key()) == 1


# ---------------------------------------------------------------------------
# State cannot survive a build
# ---------------------------------------------------------------------------

def test_the_one_object_guard_is_re_armed_not_disabled(monkeypatch):
    """Reset must restore the guard, not remove it.

    A reset that simply cleared the flag and never re-armed it would make every
    equivalence test above pass while quietly allowing a second `Object` in one
    program — so the guard is checked on a *reused* child specifically.
    """
    assert _run(_src("first"))["ok"]

    two = _run('obj = Object("a")\nother = Object("b")\n')
    assert two["ok"] is False
    assert "only one Object" in json.dumps(two["build_error"])

    # And the child is still good for ordinary work afterwards.
    assert _run(_src("third"))["ok"]


def test_a_failed_build_does_not_poison_the_next_one():
    """The realistic reuse hazard: repair loops build failures back to back."""
    bad = _run('obj = Object("x")\nc = no_such_name\n')
    assert bad["ok"] is False

    good = _run(_src("after_failure"))
    assert good["ok"], good.get("build_error") or good.get("message")


def test_a_sandbox_violation_does_not_poison_the_next_one():
    violation = _run("import build123d\n")
    assert violation["sandbox_violation"] is not None

    good = _run(_src("after_violation"))
    assert good["ok"], good.get("build_error") or good.get("message")


# ---------------------------------------------------------------------------
# A child that faults is never reused
# ---------------------------------------------------------------------------

@pytest.mark.cold_pool
def test_a_timeout_retires_the_child_and_the_next_build_still_works():
    """A killed child must not be counted as still live.

    The leak this guards against is silent: recycle that forgets to decrement
    the live count walks the pool into its cap and every later build quietly
    falls back to a single-use spawn — correct output, none of the saving.
    """
    assert _run(_src("warm"))["ok"]

    out = _run("obj = Object('x')\nwhile True:\n    pass\n", timeout=2)
    assert out["ok"] is False
    assert out["error_kind"] == "timeout"
    assert not _worker_pool.POOL._idle.get(_pool_key()), "a timed-out child was kept"
    assert _worker_pool.POOL._live.get(_pool_key(), 0) == 0, "live count leaked"

    assert _run(_src("after_timeout"))["ok"]


@pytest.mark.cold_pool
def test_a_child_that_dies_while_idle_is_replaced_silently():
    """Death between builds is not the next build's failure."""
    assert _run(_src("warm"))["ok"]

    (idle,) = _worker_pool.POOL._idle[_pool_key()]
    idle.proc.kill()
    idle.proc.wait(timeout=10)

    out = _run(_src("after_idle_death"))
    assert out["ok"], out.get("message")
    assert _worker_pool.POOL._live.get(_pool_key()) == 1


@pytest.mark.cold_pool
def test_a_child_that_dies_mid_build_is_reported_as_a_crash():
    """A child dying mid-build is rare, and the classification still has to
    be right when one does: a dead child is a
    `worker_crash` carrying its exit status, not a modeling error the agent
    would try to repair.
    """
    child = _worker_pool.POOL.acquire(W.b123d_python(), W._CHILD_PATH)
    child.ensure_ready(120.0)
    child.proc.kill()
    child.proc.wait(timeout=10)

    with pytest.raises(_worker_pool.WorkerProtocolError):
        child.run_job({"source_path": "/dev/null", "run_dir": "/tmp",
                       "result": "/tmp/none.json"}, deadline=_deadline(10))
    _worker_pool.POOL.recycle(child)

    assert _run(_src("after_crash"))["ok"]


def _deadline(seconds: float) -> float:
    import time
    return time.monotonic() + seconds


# ---------------------------------------------------------------------------
# The failure vocabulary does not depend on which path ran
# ---------------------------------------------------------------------------

@pytest.mark.cold_pool
def test_a_missing_interpreter_is_unavailable_on_both_paths(monkeypatch):
    warm = _run(_src("x"), python="/nonexistent/python")
    cold = _one_shot(_src("x"), monkeypatch, python="/nonexistent/python")
    assert warm["error_kind"] == cold["error_kind"] == "worker_unavailable"


#: One case per `slot`, since each pays a cold interpreter: the `declare_root`
#: case covers a `build_error` classified identically warm and cold, and the
#: syntax-error case the *other* slot, `sandbox_violation`, with no geometry.
@pytest.mark.parametrize("src,slot,needle", [
    ("obj = Object('x'\n", "sandbox_violation", "syntax error"),
    ("obj = Object('x')\nobj.add_part('b', Box(.1,.1,.1))\nir = obj.finalize()\n",
     "build_error", "declare_root"),
])
@pytest.mark.cold_pool
def test_failures_classify_identically_on_both_paths(src, slot, needle,
                                                     monkeypatch):
    warm = _run(src)
    cold = _one_shot(src, monkeypatch)
    assert warm["ok"] is cold["ok"] is False
    assert warm[slot] is not None and cold[slot] is not None
    assert needle in json.dumps(warm[slot])
    assert warm[slot] == cold[slot]


@pytest.mark.cold_pool
def test_both_paths_always_carry_the_stdio_keys(monkeypatch):
    """Callers read these unconditionally (`tools._build_ir_routed` pops them),
    so the warm and the one-shot path both carry them."""
    for out in (_run(_src("stdio")), _one_shot(_src("stdio"), monkeypatch)):
        assert isinstance(out["worker_stdout"], str)
        assert isinstance(out["worker_stderr"], str)


# ---------------------------------------------------------------------------
# Spool slicing
# ---------------------------------------------------------------------------

def test_a_span_reads_exactly_its_own_bytes(tmp_path):
    """Each build gets the range it wrote — not the file, and not its
    predecessor's output."""
    spool = tmp_path / "out.log"
    spool.write_bytes(b"AAAAfirstBBBBsecond")
    assert _worker_pool._read_span(spool, [4, 9]) == "first"
    assert _worker_pool._read_span(spool, [13, 19]) == "second"


@pytest.mark.parametrize("span", [None, [], [5], [9, 4], [4, 4], "nope"])
def test_a_malformed_span_reads_as_empty_not_as_an_error(tmp_path, span):
    """A child that died before reporting leaves no usable span, and that must
    not turn a crash report into an exception inside the reporter."""
    spool = tmp_path / "out.log"
    spool.write_bytes(b"content")
    assert _worker_pool._read_span(spool, span) == ""


# ---------------------------------------------------------------------------
# The adjudicator is pooled too
# ---------------------------------------------------------------------------
#
# `NoSelfPenetration` sends SAT survivors to `_adjudicate_runner.py`, which used
# to pay its own CAD kernel start on every compile — a second one, on top of the
# authoring child's. It is warmed by the same pool, and has to satisfy the same
# equivalence claim.

_ADJ_A = {"primitive": "box",
          "params": {"width": 0.1, "length": 0.1, "height": 0.1}}


def _se3(dx: float = 0.0):
    """A dycraft SE3: ((tx,ty,tz), (qw,qx,qy,qz)), unrotated."""
    return [[dx, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]


_IDENTITY = _se3(0.0)


def _adj_queries():
    return [
        {"tag": "overlapping", "a": _ADJ_A, "a_se3": _IDENTITY,
         "b": _ADJ_A, "b_se3": _se3(0.02)},
        {"tag": "clear", "a": _ADJ_A, "a_se3": _IDENTITY,
         "b": _ADJ_A, "b_se3": _se3(0.5)},
    ]


def _adj_invoke():
    from dycraft.verify import _exact_adjudicator as EA
    return EA._invoke(_adj_queries(), tol=1e-9, vol_eps=1e-12)


@pytest.mark.cold_pool
def test_the_adjudicator_agrees_warm_and_cold(monkeypatch):
    """Same verdicts either way — the property that permits reuse."""
    warm = _adj_invoke()
    assert warm is not None, "the adjudicator could not run at all"

    monkeypatch.setenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", "0")
    W.shutdown_pool()
    cold = _adj_invoke()
    assert cold is not None
    assert warm == cold


def test_the_adjudicator_still_distinguishes_the_two_verdicts():
    """Guards against a warm child that returns something uniform and wrong.

    Equivalence alone would be satisfied by a child that answered every pair
    identically on both paths, so the fixture's two pairs are asserted to land
    on *different* verdicts.
    """
    out = _adj_invoke()
    assert out["overlapping"]["verdict"] == "PENETRATING"
    assert out["clear"]["verdict"] == "CLEAR"


def test_a_reused_adjudicator_rereads_a_rewritten_brep(tmp_path):
    """The one way reuse could silently corrupt a verdict.

    The runner memoizes shapes on their spec, and a spec names a B-rep by path.
    Compiles rewrite those paths, so a cache that outlived a batch would answer
    a later compile with earlier geometry — a wrong collision verdict with
    nothing to indicate it. `_adjudicate` builds the cache per batch; this holds
    it to that by adjudicating the same path twice with different contents.
    """
    from dycraft.verify import _exact_adjudicator as EA

    brep = tmp_path / "part.brep"

    def _stage(size: float) -> None:
        src = (f'obj = Object("b")\n'
               f'p = obj.add_part("p", Box({size}, {size}, {size}))\n'
               'obj.declare_root(p)\nir = obj.finalize()\n')
        run_dir = Path(tempfile.mkdtemp())
        out = W.run_object_source(src, run_dir=run_dir)
        assert out["ok"], out.get("build_error") or out.get("message")
        staged = run_dir / out["ir"]["parts"][0]["params"]["brep"]
        brep.write_bytes(staged.read_bytes())

    def _gap() -> float:
        spec = {"brep": str(brep)}
        got = EA._invoke([{"tag": "t", "a": spec, "a_se3": _IDENTITY,
                           "b": spec, "b_se3": _se3(0.30)}],
                         tol=1e-9, vol_eps=1e-12)
        assert got is not None
        return got["t"]

    # A 0.1 m cube pair 0.30 m apart is clear; a 0.4 m cube pair at the same
    # separation overlaps. Same path both times, different bytes.
    _stage(0.1)
    small = _gap()
    _stage(0.4)
    large = _gap()

    assert small["verdict"] == "CLEAR", small
    assert large["verdict"] == "PENETRATING", (
        "the warm adjudicator answered from a stale B-rep cache")


@pytest.mark.cold_pool
def test_the_adjudicator_and_the_author_get_separate_children():
    """Two scripts under one interpreter are not interchangeable."""
    assert _run(_src("author"))["ok"]
    assert _adj_invoke() is not None
    keys = list(_worker_pool.POOL._idle)
    assert len(keys) == 2, f"expected one child per script, got {keys}"
    for bucket in _worker_pool.POOL._idle.values():
        assert len(bucket) == 1


# ---------------------------------------------------------------------------
# Operational properties
# ---------------------------------------------------------------------------

@pytest.mark.cold_pool
def test_concurrent_builds_do_not_share_a_child():
    """`agent/server.py` dispatches handlers on a 32-thread pool.

    A child serves one build at a time, so the hazard is two threads writing
    jobs into the same stdin and reading each other's acks. That does not fail
    loudly — it returns *someone's* valid IR — so each thread here builds a
    differently-sized box and is checked against its own expected extents.
    """
    import threading

    results: dict[int, dict] = {}

    def work(i: int) -> None:
        src = (f'obj = Object("o{i}")\n'
               f'b = obj.add_part("b", Box(0.{i + 1}, 0.2, 0.3))\n'
               'obj.declare_root(b)\nir = obj.finalize()\n')
        results[i] = W.run_object_source(src, run_dir=Path(tempfile.mkdtemp()))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)

    assert len(results) == 8
    for i, out in results.items():
        assert out["ok"], out.get("build_error") or out.get("message")
        # Box(w, l, h) is centered, so the +X half-extent identifies the source.
        assert out["ir"]["parts"][0]["params"]["aabb_max"][0] == pytest.approx(
            float(f"0.{i + 1}") / 2), f"thread {i} got another thread's build"
    assert len({out["ir"]["id"] for out in results.values()}) == 8

    live = sum(_worker_pool.POOL._live.values())
    idle = sum(len(b) for b in _worker_pool.POOL._idle.values())
    assert live == idle, f"live ({live}) and idle ({idle}) accounting diverged"


@pytest.mark.cold_pool
def test_a_warm_child_does_not_outlive_a_killed_parent():
    """Orphaned CAD interpreters would accumulate invisibly.

    `atexit` covers an orderly exit, and cannot cover `SIGKILL` — which is how a
    wedged daemon actually dies. The child's stdin is the backstop: the pipe
    closes with the parent, the child reads EOF and exits on its own. Tested
    against `SIGKILL` specifically, because that is the case no handler runs for.
    """
    import subprocess
    import sys
    import textwrap
    import time

    repo = Path(__file__).resolve().parents[1]
    program = textwrap.dedent(f"""
        import sys, tempfile, importlib.util as u
        from pathlib import Path
        sp = u.spec_from_file_location(
            'dycraft', {str(repo / '__init__.py')!r},
            submodule_search_locations=[{str(repo)!r}])
        m = u.module_from_spec(sp); sys.modules['dycraft'] = m
        sp.loader.exec_module(m)
        from dycraft.agent.cad import object_worker as W, _worker_pool
        src = ('obj = Object("w")\\n'
               'b = obj.add_part("b", Box(.1,.1,.1))\\n'
               'obj.declare_root(b)\\nir = obj.finalize()\\n')
        assert W.run_object_source(src, run_dir=Path(tempfile.mkdtemp()))["ok"]
        pids = [c.proc.pid for bucket in _worker_pool.POOL._idle.values()
                for c in bucket]
        print(pids[0], flush=True)
        sys.stdin.read()
    """)
    parent = subprocess.Popen([sys.executable, "-c", program],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True)
    try:
        child_pid = int(parent.stdout.readline().strip())
    except ValueError:  # pragma: no cover — the helper failed to warm a child
        parent.kill()
        pytest.fail("the helper process never reported a warm child")

    parent.kill()
    parent.wait(timeout=30)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except OSError:
            return  # gone, as required
        time.sleep(0.2)
    pytest.fail(f"warm child {child_pid} outlived its killed parent")


@pytest.mark.cold_pool
def test_an_idle_child_is_retired_and_its_memory_released():
    """Warmth costs ~500 MB per child, so it is not held indefinitely.

    An agent compiles in bursts and then thinks for minutes; a child kept warm
    across that pause holds a gigabyte for nothing. The reaper is what bounds
    that, and a reaper that silently failed to decrement the live count would
    wedge the pool at its cap instead — so both are asserted.
    """
    assert _run(_src("warm"))["ok"]
    assert _worker_pool.POOL._live.get(_pool_key()) == 1

    # ttl=0 is "disabled", so drive the sweep with an explicit tiny TTL rather
    # than by waiting out the real one.
    assert _worker_pool.POOL.reap_idle(ttl=0.0001) == 1
    assert not _worker_pool.POOL._idle.get(_pool_key())
    assert _worker_pool.POOL._live.get(_pool_key(), 0) == 0

    # And the pool still works afterwards, from cold.
    assert _run(_src("after_reap"))["ok"]
    assert _worker_pool.POOL._live.get(_pool_key()) == 1


@pytest.mark.cold_pool
def test_reaping_is_off_when_the_ttl_is_zero():
    """`0` means "keep children forever" — a long-lived batch job may want it."""
    assert _run(_src("warm"))["ok"]
    assert _worker_pool.POOL.reap_idle(ttl=0) == 0
    assert len(_worker_pool.POOL._idle[_pool_key()]) == 1


@pytest.mark.cold_pool
def test_the_cap_falls_back_instead_of_failing(monkeypatch):
    """Past the cap, a build runs one-shot rather than blocking or erroring.

    This is the property that lets the cap stay small: overflow costs the old
    1.4 s spawn, not a failure and not a queue.
    """
    monkeypatch.setenv("DYCRAFT_CAD_POOL_MAX_CHILDREN", "1")
    W.shutdown_pool()

    first = _worker_pool.POOL.acquire(W.b123d_python(), W._CHILD_PATH)
    assert first is not None
    try:
        assert _worker_pool.POOL.acquire(
            W.b123d_python(), W._CHILD_PATH) is None, "the cap was not honoured"
        # A build still succeeds while the only pooled slot is checked out.
        assert _run(_src("overflow"))["ok"]
    finally:
        _worker_pool.POOL.recycle(first)


# ---------------------------------------------------------------------------
# Snippet jobs — the other thing the warm child is for
# ---------------------------------------------------------------------------
#
# `run_cad_snippet` lets anything that needs the CAD environment rather than
# the object layer — `_solid_compile`'s phases, `BRepTools`, the
# `_b123d_namespace` guard table — run on the pool's shared child instead of
# launching `<b123d-python> -c ...` and re-paying `import build123d` (~1.5 s)
# per call. These tests pin the contract that makes that safe.


_SNIPPET = ('import json, build123d as bd\n'
            'print(json.dumps({"v": round(bd.Box(0.4, 0.2, 0.1).volume, 9)}))\n')


def test_a_snippet_returns_its_stdout_as_the_result():
    """The body's stdout *is* the answer, so it comes back verbatim."""
    out = W.run_cad_snippet(_SNIPPET)
    assert out["ok"], out["message"]
    assert json.loads(out["stdout"].strip().splitlines()[-1]) == {"v": 0.008}


def test_a_snippet_sees_dycraft_already_bound():
    """The child bootstraps `dycraft` once at startup, so a body must not — and
    does not need to. A second `spec_from_file_location` would bind a fresh copy
    of the package while live submodules still referenced the first."""
    out = W.run_cad_snippet(
        "import sys\n"
        "from dycraft.agent.cad import _solid_compile as sc\n"
        "print(sys.modules['dycraft'].__file__)\n")
    assert out["ok"], out["message"]
    assert out["stdout"].strip().endswith("__init__.py")


def test_a_raising_body_is_reported_without_costing_the_pool_a_child():
    """A body that raises is *data*, not a worker fault.

    It matters twice: the caller
    still learns the body failed and why, and — because the child is not
    recycled — the next snippet is still warm. A body treated as a fault would
    make every negative-control test pay a fresh 1.5 s import for the *next*
    one.
    """
    bad = W.run_cad_snippet("raise ValueError('boom')\n")
    assert bad["ok"] is False
    assert bad["error_kind"] == "snippet_error"
    assert "ValueError: boom" in bad["message"]
    assert "ValueError: boom" in (bad["traceback"] or "")

    ok = W.run_cad_snippet(_SNIPPET)
    assert ok["ok"], "the raising body retired the child"


def test_two_snippets_do_not_share_a_namespace():
    """A fresh globals dict per body. Bodies share the interpreter, not names —
    otherwise one test's leftover binding silently satisfies another's lookup."""
    first = W.run_cad_snippet("_leaked = 1\nprint('set')\n")
    assert first["ok"], first["message"]
    second = W.run_cad_snippet("print('_leaked' in dir())\n")
    assert second["ok"], second["message"]
    assert second["stdout"].strip() == "False"


def test_a_snippet_result_is_not_truncated():
    """Build stdio is capped at 8 kB because it is a diagnostic note. A
    snippet's stdout is the value the caller parses, so capping it would corrupt
    that value from the *head* — quietly, and only for large results."""
    n = 40_000
    out = W.run_cad_snippet(f"print('x' * {n})\nprint('TAIL')\n")
    assert out["ok"], out["message"]
    lines = out["stdout"].strip().splitlines()
    assert len(lines[0]) == n, f"first line truncated to {len(lines[0])}"
    assert lines[-1] == "TAIL"

@pytest.mark.cold_pool
def test_the_snippet_path_falls_back_when_pooling_is_off(monkeypatch):
    """With the pool disabled the one-shot path answers, in the same shape."""
    monkeypatch.setenv("DYCRAFT_OBJECT_WORKER_PERSISTENT", "0")
    out = W.run_cad_snippet(_SNIPPET)
    assert out["ok"], out["message"]
    assert json.loads(out["stdout"].strip().splitlines()[-1]) == {"v": 0.008}


@pytest.mark.cold_pool
def test_a_snippet_and_a_build_share_one_child():
    """The saving only materializes if both job kinds reuse the same warm
    interpreter — a snippet that spawned its own child would just move the
    import cost rather than remove it."""
    W.run_cad_snippet(_SNIPPET)          # warm one up
    before = sum(_worker_pool.POOL._live.values())
    assert _run(_src("mixed"))["ok"]
    assert W.run_cad_snippet(_SNIPPET)["ok"]
    assert sum(_worker_pool.POOL._live.values()) == before, \
        "a snippet or build spawned a new child"


def test_a_snippet_times_out_without_wedging_the_pool():
    """A runaway body is killed and its child retired, and the pool recovers."""
    out = W.run_cad_snippet("while True:\n    pass\n", timeout=1)
    assert out["ok"] is False
    assert out["error_kind"] == "timeout"
    assert W.run_cad_snippet(_SNIPPET)["ok"], "the pool did not recover"
