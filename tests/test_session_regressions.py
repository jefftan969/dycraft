"""The object session's verifier and build caches, `probe`, and run metadata."""

from __future__ import annotations


from dycraft.agent.layers.compile import compile_with_layer, build_ir_with_layer
from dycraft.agent.layers.layer import object_layer_config
from dycraft.agent.layers.tools import ToolContext, build_current_ir, _invalidate_build_cache, run_probe
_CLEAN_OBJECT_SRC = (
    "obj = Object(\"thing\")\n"
    "body = obj.make_box(name=\"body\", width=0.5, length=0.4, height=0.3)\n"
    "obj.declare_root(body)\n"
    "ir = obj.finalize()\n"
)

_ALT_OBJECT_SRC = (
    "obj = Object(\"thing\")\n"
    "body = obj.make_box(name=\"body\", width=0.8, length=0.6, height=0.4)\n"
    "obj.declare_root(body)\n"
    "ir = obj.finalize()\n"
)


# ---------------------------------------------------------------------------
# Bug 1: verifier-suite cache must invalidate on ir_hash change.
# ---------------------------------------------------------------------------

def test_verifier_cache_invalidates_on_ir_hash_change(tmp_path):
    """The verifier cache key includes the ir_hash: without it every compile
    returns the same stale verdict, and an agent sees "same hash, same fail"
    across edits that materially change the geometry.

    Invariant: compiling two materially different sources back-to-back in
    one ToolContext must produce two distinct verifier reports — not the
    same cached report under both ir_hashes.
    """
    cfg = object_layer_config()
    ctx = ToolContext(run_dir=tmp_path, layer_cfg=cfg)

    res_a = compile_with_layer(_CLEAN_OBJECT_SRC, layer_cfg=cfg, ctx=ctx,
                                verifier_cache=ctx.verifier_cache)
    res_b = compile_with_layer(_ALT_OBJECT_SRC, layer_cfg=cfg, ctx=ctx,
                                verifier_cache=ctx.verifier_cache)

    a_hash = res_a["ir_hash"]
    b_hash = res_b["ir_hash"]
    assert a_hash != b_hash, "Materially different sources must yield different ir_hashes."

    # The NoSelfPenetration report's `ir_hash` must match the IR that was
    # actually fed in, not a stale cached one.
    rep_a = next(r for r in res_a["verifier_reports"]
                 if r["name"] == "NoSelfPenetration")
    rep_b = next(r for r in res_b["verifier_reports"]
                 if r["name"] == "NoSelfPenetration")
    assert rep_a["ir_hash"] == a_hash
    assert rep_b["ir_hash"] == b_hash
    assert rep_a["ir_hash"] != rep_b["ir_hash"], \
        "Each compile must run its verifier on the actual current IR, not a cached one."


def test_verifier_cache_reuses_when_ir_unchanged(tmp_path):
    """Counterpart to the above: when the IR genuinely doesn't change,
    the cache SHOULD hit (so we don't pay verifier cost twice). Same
    source compiled twice must produce identical reports and the cache
    must contain exactly one entry per verifier."""
    cfg = object_layer_config()
    ctx = ToolContext(run_dir=tmp_path, layer_cfg=cfg)

    res_a = compile_with_layer(_CLEAN_OBJECT_SRC, layer_cfg=cfg, ctx=ctx,
                                verifier_cache=ctx.verifier_cache)
    cache_after_a = len(ctx.verifier_cache)
    res_b = compile_with_layer(_CLEAN_OBJECT_SRC, layer_cfg=cfg, ctx=ctx,
                                verifier_cache=ctx.verifier_cache)
    cache_after_b = len(ctx.verifier_cache)

    assert res_a["ir_hash"] == res_b["ir_hash"]
    assert cache_after_a == cache_after_b, \
        "Identical IR must not re-populate the verifier cache."


# ---------------------------------------------------------------------------
# probe and snapshot build from the CURRENT source, not from
# last_successful_ir.
# ---------------------------------------------------------------------------

def test_build_current_ir_reflects_latest_write(tmp_path):
    """An agent edits the source, then probes: the probe answers from the
    current source, rebuilding from the freshly-edited file, not from the
    last-compiled IR.
    """
    cfg = object_layer_config()
    ctx = ToolContext(run_dir=tmp_path, layer_cfg=cfg)
    src = tmp_path / "object.py"

    src.write_text(_CLEAN_OBJECT_SRC)
    _invalidate_build_cache(ctx)
    res1 = build_current_ir(ctx)
    assert res1["ok"] is True
    hash1 = res1["ir"]["id"]

    # Edit the source.
    src.write_text(_ALT_OBJECT_SRC)
    _invalidate_build_cache(ctx)
    res2 = build_current_ir(ctx)
    assert res2["ok"] is True
    hash2 = res2["ir"]["id"]

    assert hash1 != hash2, \
        "probe/snapshot must reflect the latest source on each call, not a cached pre-edit IR."


def test_build_current_ir_caches_when_source_unchanged(tmp_path):
    """Two consecutive probes without an edit must reuse the cached IR
    (same object identity for the returned ir dict). Performance matters
    when probe is called repeatedly between edits."""
    cfg = object_layer_config()
    ctx = ToolContext(run_dir=tmp_path, layer_cfg=cfg)
    (tmp_path / "object.py").write_text(_CLEAN_OBJECT_SRC)

    r1 = build_current_ir(ctx)
    r2 = build_current_ir(ctx)
    assert r1["ir"] is r2["ir"], "cached build must reuse the IR object, not rebuild."


def test_build_current_ir_returns_structured_error_on_missing_source(tmp_path):
    """When object.py doesn't exist yet, the probe/snapshot helper returns a
    well-formed error payload (not a raised exception)."""
    cfg = object_layer_config()
    ctx = ToolContext(run_dir=tmp_path, layer_cfg=cfg)
    res = build_current_ir(ctx)
    assert res["ok"] is False
    assert res["error_summary"] and "does not exist" in res["error_summary"]


def test_probe_model_works_without_prior_compile(tmp_path):
    """Object-layer probe must succeed on the first call after the source is
    written, with no compile in between."""
    cfg = object_layer_config()
    ctx = ToolContext(run_dir=tmp_path, layer_cfg=cfg)
    (tmp_path / "object.py").write_text(_CLEAN_OBJECT_SRC)
    payload = run_probe(ctx, "list_parts(ir)")
    assert payload["ok"] is True, payload
    assert "body" in payload["repr"]


# ---------------------------------------------------------------------------
# meta.json finalization
# ---------------------------------------------------------------------------

def test_plain_meta_extras_does_not_collide_with_finalize_metas_outcome(tmp_path):
    """An `outcome` key from `PlainContext.meta_extras()` would be spliced in
    as `**merged` alongside `finalize_meta`'s own positional `outcome` —
    `TypeError: TraceMeta.finalize() got multiple values for argument
    'outcome'`.

    It would fail quietly: `finalize_meta` runs from an `atexit` hook, so
    Python prints "Exception ignored" and still exits 0, leaving a run that
    succeeded at `outcome: null, finished_at: null`.

    Asserted by outcome rather than by inspecting `meta_extras()`, so any
    future layer hook that reintroduces a reserved key fails here too.
    """
    import json

    from dycraft.agent.plain.session import PlainContext, PlainSession

    (tmp_path / "meta.json").write_text(json.dumps(
        {"run_id": "r", "outcome": None, "started_at": "x",
         "finished_at": None}))
    ctx = PlainContext(run_dir=tmp_path)
    session = PlainSession(ctx)
    ctx.committed, ctx.result_text = True, "OK"

    session.finalize_meta("committed")          # must not raise TypeError

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["outcome"] == "committed", meta
    assert meta["finished_at"], meta
    assert meta["result_chars"] == 2, meta


def test_a_daemon_that_never_bound_is_recorded_in_meta_not_only_in_a_dotfile(
        tmp_path):
    """A daemon that dies before binding (a `NameError` in its `main()`, say)
    leaves a run dir with a passing preflight canary and no `trace.jsonl`.
    Unless `meta.json` records the failure, it says `outcome: null,
    finished_at: null`, nothing distinguishes the run from one still in
    progress, and the only record is `.daemon.log`, a dotfile.

    Also pins the rank interaction: a stale daemon's `finalize("interrupted")`
    must not overwrite the more specific verdict, while a real `committed`
    still must.
    """
    import json

    from dycraft.agent.agent_session import TraceMeta, record_launch_failure

    (tmp_path / "meta.json").write_text(json.dumps(
        {"run_id": "r", "outcome": None, "started_at": "x",
         "finished_at": None}))
    (tmp_path / ".daemon.log").write_text(
        "Traceback (most recent call last):\n"
        "NameError: name 'vlm_model' is not defined\n")

    record_launch_failure(
        tmp_path, reason="harness_daemon_exited_early",
        detail="dycraft.agent.layers.server exited rc=1 before binding",
        log_path=tmp_path / ".daemon.log")

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["outcome"] == "daemon_failed", meta
    assert meta["finished_at"], meta
    assert meta["launch_failure"]["reason"] == "harness_daemon_exited_early"
    assert meta["launch_failure"]["log"] == ".daemon.log"
    assert "vlm_model" in meta["launch_failure"]["log_tail"]

    TraceMeta(tmp_path).finalize("interrupted")
    assert json.loads(
        (tmp_path / "meta.json").read_text())["outcome"] == "daemon_failed"
    TraceMeta(tmp_path).finalize("committed")
    assert json.loads(
        (tmp_path / "meta.json").read_text())["outcome"] == "committed"


def test_record_launch_failure_never_raises_on_a_bare_run_dir(tmp_path):
    """Every caller is on its way to `sys.exit`, so failing to *describe* a
    failure must not replace it with a different one."""
    from dycraft.agent.agent_session import record_launch_failure

    record_launch_failure(tmp_path, reason="x", detail="y")          # no meta
    (tmp_path / "meta.json").write_text("{ not json")
    record_launch_failure(tmp_path, reason="x", detail="y",
                          log_path=tmp_path / "missing.log")
