"""Viewer: the per-agent lane timeline (`viewer/agent_timeline.py`).

Two kinds of test here, and the second kind is the point.

The hermetic ones build a synthetic bundle in `tmp_path` and pin the behavior
that is easy to get wrong and invisible when it is: op segments from the trace,
zero-duration bookkeeping, an op whose result was never written.

The corpus ones run over whatever real bundles are present in `runs/` and assert
the invariants that keep the action inventory honest as the harness and the
external CLI's tool surface change:

  * nothing classifies as `aside.unknown`
  * every duration and total is sane

Both skip cleanly when `runs/` is absent, so the suite stays hermetic for anyone
who does not have the run corpus checked out.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dycraft.viewer import agent_timeline as at
from dycraft.viewer.agent_timeline import build, classify_bash, classify_tool

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _iso(offset_s: float) -> str:
    return (T0 + timedelta(seconds=offset_s)).isoformat()


def _epoch(offset_s: float) -> float:
    return (T0 + timedelta(seconds=offset_s)).timestamp()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _op(op: str, step: int, start_s: float, elapsed_ms: float,
        ok: bool = True, args: dict | None = None) -> list[dict]:
    return [
        {"ev": "op_call", "op": op, "step": step, "ts": _iso(start_s),
         "args": args or {}},
        {"ev": "op_result", "op": op, "step": step,
         "ts": _iso(start_s + elapsed_ms / 1000.0),
         "elapsed_ms": elapsed_ms, "ok": ok},
    ]


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    """A minimal object run: a 100 s compile, a deviation, a probe, a commit, and
    a 20 s review."""
    run_dir = tmp_path / "obj_run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": "obj_run", "layer": "object", "outcome": "committed",
        "started_at": _iso(-5), "finished_at": _iso(200),
    }))
    _write_jsonl(run_dir / "trace.jsonl", [
        {"ev": "daemon_start", "layer": "object", "step": 0, "ts": _iso(0)},
        *_op("compile", 1, 10, 100_000),
        {"ev": "compile_attempt", "step": 1, "ts": _iso(110), "attempt": 1,
         "ok": True, "build_error": None, "is_finish": False,
         "elapsed_ms": 100_000,
         "verifier_summary": [{"name": "NoSelfPenetration", "verdict": "pass"},
                              {"name": "MinPartDimension", "verdict": "fail"}]},
        *_op("record_deviation", 2, 120, 0,
             args={"severity": "low", "what": "x", "why": "y"}),
        *_op("probe", 3, 130, 40, args={"expression": "anchor_position(ir, 'a')"}),
        {"ev": "commit", "step": 4, "ts": _iso(150), "ir_hash": "deadbeef"},
        *_op("review", 5, 160, 20_000),
    ])
    return run_dir


def _by_action(timeline: dict) -> dict[str, dict]:
    """First segment per action."""
    out: dict[str, dict] = {}
    for event in timeline["events"]:
        out.setdefault(event["action"], event)
    return out


def _all_by_action(timeline: dict, action: str) -> list[dict]:
    return [e for e in timeline["events"] if e["action"] == action]


# ---------------------------------------------------------------------------
# op segments
# ---------------------------------------------------------------------------


def test_an_op_is_one_segment_of_its_own_elapsed_time(bundle: Path) -> None:
    """A harness op is drawn as one bar of its `op_result.elapsed_ms`, in its
    category: a compile is Build, and a review, which grades the work through a
    sub-agent, is Judge."""
    timeline = build(bundle, now=_epoch(300))
    compiles = _all_by_action(timeline, "op:compile")
    assert len(compiles) == 1
    assert compiles[0]["category"] == "build"
    assert compiles[0]["duration_ms"] == pytest.approx(100_000)
    review = _by_action(timeline)["op:review"]
    assert review["category"] == "judge"
    assert review["duration_ms"] == pytest.approx(20_000)
    assert timeline["totals"]["judge_ms"] == pytest.approx(20_000)


def test_an_ops_arguments_stay_out_of_the_timeline(tmp_path: Path) -> None:
    """The browser fetches the timeline whenever a run is selected, so its size
    has to stay bounded on a long run. An op's arguments can be megabytes (a
    probe's code or data, an edit's source), and the label already summarizes
    them, so only the label carries them."""
    run_dir = tmp_path / "long_run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": "long_run", "layer": "animation", "outcome": "committed",
        "started_at": _iso(-5), "finished_at": _iso(200)}))
    payload = "x" * 2_000_000
    _write_jsonl(run_dir / "trace.jsonl", [
        {"ev": "daemon_start", "layer": "animation", "step": 0, "ts": _iso(0)},
        *_op("probe", 1, 10, 40, args={"expression": "len(data)", "data": payload}),
    ])
    timeline = build(run_dir, now=_epoch(300))
    probe = _by_action(timeline)["op:probe"]
    assert probe["label"] == "probe len"
    assert "args" not in probe["detail"]
    assert len(json.dumps(timeline)) < 100_000


# ---------------------------------------------------------------------------
# statuses and bookkeeping
# ---------------------------------------------------------------------------


def test_bookkeeping_actions_carry_no_duration(bundle: Path) -> None:
    deviation = _by_action(build(bundle, now=_epoch(300)))["op:record_deviation"]
    assert deviation["zero_duration"] is True
    assert deviation["duration_ms"] is None
    assert deviation["finished_at"] is None
    assert deviation["category"] == "deliver"


def test_compile_detail_carries_the_failed_verifiers(bundle: Path) -> None:
    detail = _by_action(build(bundle, now=_epoch(300)))["op:compile"]["detail"]
    assert detail["compile"]["n_verifiers"] == 2
    assert detail["compile"]["failed_verifiers"] == ["MinPartDimension"]


@pytest.mark.parametrize("run_finished,expected", [(True, "unknown"),
                                                   (False, "running")])
def test_an_op_with_no_result_is_unknown_only_once_the_run_ends(
    tmp_path: Path, run_finished: bool, expected: str,
) -> None:
    """A bar runs to `now` only while the run itself is going.

    A killed daemon leaves an `op_call` with no `op_result`. While the run is
    live that op plausibly still is too; once the run has finished, nobody ever
    recorded a result and the duration is simply unknown.
    """
    run_dir = tmp_path / f"killed_{run_finished}"
    run_dir.mkdir()
    meta = {"run_id": "killed", "layer": "object"}
    if run_finished:
        meta["outcome"] = "escalated_gave_up"
        meta["finished_at"] = _iso(200)
    (run_dir / "meta.json").write_text(json.dumps(meta))
    _write_jsonl(run_dir / "trace.jsonl", [
        {"ev": "op_call", "op": "compile", "step": 1, "ts": _iso(10),
         "args": {}},
    ])
    event = _by_action(build(run_dir, now=_epoch(300)))["op:compile"]
    assert event["status"] == expected
    assert event["finished_at"] is None


def test_steps_with_no_recorded_result_are_one_note_in_plain_words() -> None:
    """A delivered run's transcript can show a dozen kinds of step the harness log has no result for. One note says
    so for all of them, naming the first three, and in words a reader of the viewer uses: no file names."""
    assert at._unrecorded_note(["plan"]) == (
        "1 kind of step (plan) has no recorded result, so its times come from the agent's transcript and whether it "
        "worked is unknown.")
    assert "(plan and build)" in at._unrecorded_note(["plan", "build"])
    assert "(plan, build, and probe)" in at._unrecorded_note(["plan", "build", "probe"])
    many = at._unrecorded_note([f"op{i}" for i in range(15)])
    assert many.startswith("15 kinds of step (op0, op1, op2, and 12 more) have no recorded result")
    assert "trace.jsonl" not in many and "`" not in many


def test_a_failed_op_is_marked_failed(tmp_path: Path) -> None:
    run_dir = tmp_path / "failing"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({
        "layer": "object", "outcome": "escalated_repair_cap",
        "finished_at": _iso(200)}))
    _write_jsonl(run_dir / "trace.jsonl", _op("compile", 1, 10, 500, ok=False))
    assert _by_action(build(run_dir, now=_epoch(300)))["op:compile"]["status"] \
        == "failed"


def test_lifecycle_events_become_axis_marks(bundle: Path) -> None:
    marks = {m["kind"]: m for m in build(bundle, now=_epoch(300))["marks"]}
    assert "daemon_start" in marks
    assert marks["commit"]["failure"] is False
    assert marks["commit"]["detail"]["ir_hash"] == "deadbeef"


def test_categories_are_offered_per_layer_and_in_order(bundle: Path) -> None:
    keys = [c["key"] for c in build(bundle, now=_epoch(300))["categories"]]
    # `solve` is animation-only, so an object run must not offer an empty one.
    assert "solve" not in keys
    assert keys == [k for k in at.CATEGORY_ORDER if k in keys]
    # Both model categories are offered on an object run: every run has an
    # LLM, and the object layer has `review`.
    assert keys[:2] == ["model", "judge"]


def test_an_animation_run_gets_the_solve_lane(tmp_path: Path) -> None:
    run_dir = tmp_path / "anim"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({
        "layer": "animation", "outcome": "committed", "finished_at": _iso(200)}))
    _write_jsonl(run_dir / "trace.jsonl", _op("ik", 1, 10, 120_000))
    timeline = build(run_dir, now=_epoch(300))
    assert "solve" in [c["key"] for c in timeline["categories"]]
    assert _by_action(timeline)["op:ik"]["category"] == "solve"


def test_a_bundle_with_nothing_in_it_still_builds(tmp_path: Path) -> None:
    run_dir = tmp_path / "bare"
    run_dir.mkdir()
    timeline = build(run_dir, now=_epoch(300))
    assert timeline["events"] == []
    assert timeline["totals"]["all_ms"] == 0
    assert timeline["notes"], "an empty timeline must say why it is empty"


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command,expected", [
    # The `.terminate` idiom wins over the bare `test` it is written as.
    ('test -s "$D/context_summary.md" && touch "$D/.terminate"',
     "deliver.terminate"),
    # Reading its own grade is inspection, not housekeeping.
    ('cd "$D" && python3 -c "import json; print(json.load(open(\'compile_reports/latest.json\')))"',
     "inspect.report"),
    # Computing geometry out of band is authoring.
    ('cd "$D" && "$DYCRAFT_PYTHON" -c "from build123d import Torus; print(Torus(1,2))"',
     "author.experiment"),
    # Running a script it wrote itself, where the imports are in the file.
    ('cd "$D" && timeout 900 "$DYCRAFT_PYTHON" scratch_exec.py 2>&1 | tail -60',
     "author.experiment"),
    # An import a report reader would never need.
    ('cd "$D" && "$DYCRAFT_PYTHON" - <<\'PY\'\nimport scratch_clearance as sc\nPY',
     "author.experiment"),
    # Writing a helper script through the shell rather than through `Write`.
    ("cd \"$D\" && cat > ik_probe.py <<'PYEOF'\nimport numpy\nPYEOF",
     "author.aux"),
    ("cd \"$D\" && cat > object.py <<'PYEOF'\npass\nPYEOF", "author.source"),
    ("cd \"$D\" && sed -i 's/a/b/' object.py", "author.source"),
    # `cd` is a preamble, never the intent.
    ('cd "$DYCRAFT_RUN_DIR" && ls -la', "inspect.search"),
    ('cd "$D" && grep -rn "foo" agent/', "inspect.search"),
    # Blocking on a background job.
    ("sleep 100; cat /tmp/x/tasks/babc.output", "aside.wait"),
    ('cd "$D" && until [ -f done ]; do sleep 3; done', "aside.wait"),
    ('cd "$D" && mkdir -p _scratch && echo ok', "aside.plumbing"),
])
def test_bash_classification(command: str, expected: str) -> None:
    assert classify_bash(command) == expected


@pytest.mark.parametrize("name,payload,expected", [
    ("Read", {"file_path": "/x/object.py"}, "inspect.read_source"),
    ("Read", {"file_path": "/x/renders/1_front.png"}, "look.read_image"),
    ("Read", {"file_path": "/skills/loft.md"}, "inspect.read_reference"),
    ("Read", {"file_path": "/tmp/tasks/b1.output"}, "aside.wait"),
    ("Edit", {"file_path": "/x/animation.py"}, "author.source"),
    ("Write", {"file_path": "/x/context_summary.md"}, "author.summary"),
    ("Write", {"file_path": "/x/solids.py"}, "author.aux"),
    ("Skill", {"skill": "save_context"}, "author.summary"),
    ("Skill", {"skill": "build123d_authoring"}, "inspect.read_reference"),
    ("TaskOutput", {}, "aside.wait"),
    ("Monitor", {}, "aside.wait"),
    ("ToolSearch", {}, "aside.tool_search"),
    ("Grep", {}, "inspect.search"),
    # A tool nobody has seen yet must be visible, not silently absorbed.
    ("SomeFutureTool", {}, "aside.unknown"),
])
def test_tool_classification(name: str, payload: dict, expected: str) -> None:
    assert classify_tool(name, payload) == expected


def test_every_action_key_has_a_category() -> None:
    """Nothing may be classified into a key the category map does not know."""
    assert set(at.ZERO_DURATION_ACTIONS) <= set(at.ACTION_CATEGORY)
    for category in at.ACTION_CATEGORY.values():
        assert category in at.CATEGORY_ORDER
    for key in at.CATEGORY_ORDER:
        assert key in at.CATEGORY_LABEL and key in at.CATEGORY_BLURB
    # Every layer may only offer categories that exist.
    for categories in at.LAYER_CATEGORIES.values():
        assert set(categories) <= set(at.CATEGORY_ORDER)


def test_category_order_is_pinned() -> None:
    """The legend's order is a shared contract with the CSS and the panel, and
    `web/scripts/check-agent-timeline.mjs` pins those two ends."""
    assert at.CATEGORY_ORDER == ("model", "judge", "author", "build",
                                 "inspect", "look", "solve", "deliver",
                                 "aside")


# ---------------------------------------------------------------------------
# corpus invariants
# ---------------------------------------------------------------------------


#: Layers the viewer does not lay out; runs of them in a development `runs/`
#: are not part of the contract.
_REMOVED_LAYERS = {"orchestrator", "project"}


def _corpus() -> list[Path]:
    runs = Path(__file__).resolve().parents[1] / "runs"
    if not runs.is_dir():
        return []
    found = []
    for run in sorted({p.parent for p in runs.rglob("trace.jsonl")}):
        try:
            layer = json.loads((run / "meta.json").read_text()).get("layer")
        except (OSError, ValueError):
            layer = None
        if layer not in _REMOVED_LAYERS:
            found.append(run)
    return found


CORPUS = _corpus()


@pytest.mark.skipif(not CORPUS, reason="no run corpus checked out")
def test_corpus_classifies_completely() -> None:
    """Nothing in `runs/` may fall through to `aside.unknown`.

    The tool surface belongs to an external CLI and grows without warning, so
    a new tool should surface as a test failure naming the command, not as a
    silent reclassification into the last lane.
    """
    stragglers: dict[str, str] = {}
    for run_dir in CORPUS:
        for event in build(run_dir)["events"]:
            if event["action"] == "aside.unknown":
                stragglers.setdefault(event["label"][:100], str(run_dir))
    assert not stragglers, (
        f"{len(stragglers)} unclassified action(s); add them to "
        f"agent_timeline's classifier: {stragglers}")


@pytest.mark.skipif(not CORPUS, reason="no run corpus checked out")
def test_corpus_durations_are_sane() -> None:
    for run_dir in CORPUS:
        timeline = build(run_dir)
        for event in timeline["events"]:
            if event["zero_duration"]:
                assert event["duration_ms"] is None
            duration = event["duration_ms"]
            if duration is not None:
                assert duration >= 0, f"{run_dir}: {event['action']} {duration}"
            assert event["category"] in at.CATEGORY_ORDER
            assert event["status"] in ("ok", "failed", "running", "unknown")
            # Every segment carries its own visible label — that is the
            # identity channel, with color reinforcing it.
            assert (event["label"] or "").strip(), f"{run_dir}: {event['action']}"
            if event["started_at"] and event["finished_at"]:
                assert event["finished_at"] >= event["started_at"], (
                    f"{run_dir}: {event['action']} ends before it begins")
        totals = timeline["totals"]
        assert totals["all_ms"] == pytest.approx(
            totals["action_ms"] + totals["judge_ms"] + totals["model_ms"],
            rel=1e-6)


# ---------------------------------------------------------------------------
# the endpoint
# ---------------------------------------------------------------------------


def test_api_serves_the_agent_timeline(bundle: Path) -> None:
    """The route is registered before the greedy `/api/runs/{id:path}` catch-all.

    Registration order is match order in FastAPI, so a suffixed route added
    after the bare one would be swallowed as part of the run id — the failure
    mode `app.py`'s own comment warns about. This asserts it did not happen.
    """
    fastapi = pytest.importorskip("fastapi", reason="viewer needs fastapi")
    assert fastapi
    from fastapi.testclient import TestClient

    from dycraft.viewer.app import create_app

    client = TestClient(create_app(runs_dir=bundle.parent))
    response = client.get(f"/api/runs/{bundle.name}/agent_timeline")
    assert response.status_code == 200
    payload = response.json()
    assert payload["layer"] == "object"
    assert [c["key"] for c in payload["categories"]]
    assert payload["totals"]["judge_ms"] > 0
    assert any(e["action"] == "op:compile" for e in payload["events"])

    missing = client.get("/api/runs/nope/agent_timeline")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# the api backend
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# model phases: thinking vs composing a tool call
# ---------------------------------------------------------------------------


def test_model_phases_split_thinking_from_composing() -> None:
    """The split is measured from per-block record timestamps, not apportioned.

    The CLI writes one record per content block, so the gap before a record is
    how long the model spent producing that block. A turn that thinks for 168 s
    and then spends 35 s emitting a `Write object.py` call is two very
    different kinds of work, and bundling them would hide about a third of an
    object run's model time. On an authoring turn, that stretch *is* the
    authoring.
    """
    phases = at.model_phases(_iso(0), [
        {"at": _iso(168), "kinds": ["thinking"]},
        {"at": _iso(169), "kinds": ["text"]},
        {"at": _iso(204), "kinds": ["tool_use"]},
    ])
    assert [(p["kind"], p["duration_ms"]) for p in phases] == [
        ("think", 168_000.0), ("text", 1_000.0), ("write", 35_000.0)]
    # Offsets are relative to the turn's start, so the panel can place the
    # hatch inside the segment without re-deriving anything.
    assert [p["offset_ms"] for p in phases] == [0.0, 168_000.0, 169_000.0]


def test_model_phases_name_a_mixed_record_by_its_tool_call() -> None:
    """Several blocks in one record cannot be split further, so the record is
    named by the most specific thing in it — composing a call is the claim
    worth making."""
    phases = at.model_phases(_iso(0), [
        {"at": _iso(10), "kinds": ["thinking", "tool_use"]},
    ])
    assert [p["kind"] for p in phases] == ["write"]


def test_model_phases_are_empty_without_records() -> None:
    """The `api` path writes one record per response, so it has no internal
    boundaries to read and the segment stays undivided."""
    assert at.model_phases(_iso(0), []) == []
    assert at.model_phases(None, [{"at": _iso(1), "kinds": ["thinking"]}]) == []


def test_model_phases_ignore_non_monotonic_records() -> None:
    phases = at.model_phases(_iso(10), [
        {"at": _iso(5), "kinds": ["thinking"]},       # before the turn began
        {"at": _iso(20), "kinds": ["tool_use"]},
    ])
    assert [(p["kind"], p["duration_ms"]) for p in phases] == [
        ("write", 10_000.0)]


def test_model_event_carries_the_split(tmp_path: Path) -> None:
    """End to end: the label leads with the split and the detail carries the
    phases the panel hatches."""
    run_dir = tmp_path / "phased"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({
        "layer": "object", "outcome": "committed", "finished_at": _iso(400)}))
    _write_jsonl(run_dir / "trace.jsonl", [])
    _write_jsonl(run_dir / "agent_transcript.jsonl", [
        {"type": "user", "timestamp": _iso(0),
         "message": {"content": "go"}},
        {"type": "assistant", "timestamp": _iso(120),
         "message": {"id": "m1", "model": "claude-opus-5",
                     "usage": {"input_tokens": 1, "output_tokens": 2},
                     "content": [{"type": "thinking", "thinking": "hmm"}]}},
        {"type": "assistant", "timestamp": _iso(150),
         "message": {"id": "m1", "model": "claude-opus-5",
                     "usage": {"input_tokens": 1, "output_tokens": 2},
                     "content": [{"type": "tool_use", "id": "t1",
                                  "name": "Write",
                                  "input": {"file_path": "object.py"}}]}},
        {"type": "user", "timestamp": _iso(151), "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
    ])
    timeline = build(run_dir, now=_epoch(500))
    model = _by_action(timeline)["model.turn"]
    assert model["category"] == "model"
    assert model["detail"]["think_ms"] == pytest.approx(120_000)
    assert model["detail"]["write_ms"] == pytest.approx(30_000)
    assert "2m thinking" in model["label"]      # _short rolls over at 90 s
    assert "30s writing" in model["label"]
    kinds = [p["kind"] for p in model["detail"]["phases"]]
    assert kinds == ["think", "write"]
    # And the phases stay inside the segment they are drawn in.
    for phase in model["detail"]["phases"]:
        assert phase["offset_ms"] + phase["duration_ms"] <= model["duration_ms"] + 1


@pytest.mark.skipif(not CORPUS, reason="no run corpus checked out")
def test_corpus_phases_stay_inside_their_segment() -> None:
    """A hatch is placed as a fraction of its segment, so a phase running past
    the end would paint outside the bar."""
    for run_dir in CORPUS:
        for event in build(run_dir)["events"]:
            phases = event["detail"].get("phases") or []
            if not phases:
                continue
            duration = event["duration_ms"] or 0
            for phase in phases:
                assert phase["offset_ms"] >= 0
                assert phase["duration_ms"] >= 0
                assert phase["offset_ms"] + phase["duration_ms"] <= duration + 1, (
                    f"{run_dir}: {event['action']} phase overruns its segment")


def test_not_ok_ops_split_into_errors_findings_and_rejections():
    """Errors, findings, and rejections are counted apart.

    A check that runs as asked and reports a defect in the film is a finding,
    not a failure; summed into one counter with errors, the run that checks
    most thoroughly would score worst on the only aggregate an operator reads
    at a glance.
    """
    import dycraft.agent.agent_session as S

    def klass(**record):
        return S._failure_class(record)

    # Declared by the handler wins.
    assert klass(failure_class="rejection", error="anything") == "rejection"
    assert klass(failure_class="check_finding") == "check_finding"
    # Inferred for traces already on disk.
    assert klass(error="A project edit is pending; poll it before submitting "
                       "another operation.") == "rejection"
    assert klass(error="Project is committed; start a refinement run.") == "rejection"
    assert klass(error="every declared state needs reviewed storyboard evidence") == "rejection"
    assert klass(verifier_summary=[{"name": "NoPenetration", "verdict": "fail"}]) == "check_finding"
    assert klass(error="ValueError: unpacked dependency outside workspace") == "error"
    assert klass() == "error"


def test_an_op_result_carries_its_own_call_step_and_a_raise_still_files_one(tmp_path):
    """Each `op_result` carries its own call's step, and a raising op still files one.

    Taking whatever step the run had reached when the handler returned would
    file overlapping ops under one number. Every record also names its
    process, because the counter restarts with each daemon."""
    import os
    from types import SimpleNamespace
    import dycraft.agent.agent_session as S

    session = S.AgentSession(SimpleNamespace(run_dir=tmp_path, step=0))

    @session.op("slow")
    def _slow() -> dict:
        session.tick()                    # another op starts meanwhile
        return {"ok": True}

    @session.op("broken")
    def _broken() -> dict:
        raise ValueError("no such shot")

    session.handlers["slow"]()
    with pytest.raises(ValueError):
        session.handlers["broken"]()
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [(r["ev"], r["step"]) for r in records] == [
        ("op_call", 1), ("op_result", 1), ("op_call", 3), ("op_result", 3)]
    # Classified from the text, not stamped: a raised refusal is a rejection.
    assert records[-1]["ok"] is False and "failure_class" not in records[-1]
    assert S._failure_class(records[-1]) == "error"
    assert records[-1]["error_type"] == "ValueError" and "no such shot" in records[-1]["error"]
    assert {r["epoch"] for r in records} == {os.getpid()}


def test_failed_batch_jobs_are_counted_under_their_own_names(tmp_path):
    """A batch op returns its handle `ok` and reports the outcome later in
    `job_result`, so its failures are counted from `job_result`. An
    interactive op writes a `job_result` too, but its `op_result` already
    counted it."""
    import dycraft.agent.agent_session as S

    records = [
        {"ev": "op_result", "op": "compile", "ok": True},
        {"ev": "job_result", "status": "completed",
         "result": {"ok": False, "failure_class": "check_finding"}},
        {"ev": "job_result", "status": "failed",
         "result": {"ok": False, "error": "unknown shot, clip or state to invalidate: x"}},
        {"ev": "job_result", "status": "completed",
         "result": {"ok": False, "error": "A project edit is pending; poll it first."}},
        {"ev": "job_result", "status": "cancelled", "result": {"ok": False, "error": "job cancelled"}},
        {"ev": "job_result", "status": "completed", "result": {"ok": True}},
        {"ev": "job_result", "status": "failed", "interactive": True, "result": {"ok": False}},
        {"ev": "compile_attempt", "attempt": 11, "ok": True},
        {"ev": "compile_attempt", "attempt": 13, "ok": False},
    ]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    shape = S._run_shape(tmp_path, None)
    assert shape["op_failures"] == 0
    assert (shape["jobs"], shape["job_errors"], shape["job_check_findings"],
            shape["job_rejections"], shape["jobs_cancelled"]) == (5, 1, 1, 1, 1)
    # A gap in the attempt ledger shows as the two disagreeing.
    assert shape["compile_attempts"] == 2 and shape["compile_attempt_max"] == 13


def test_op_detection_requires_a_real_op_name() -> None:
    """`h.<name>(` is a shape, not a harness call.

    A shell heredoc is full of receivers called `h`: a packaging script that
    hashes its files with `h = hashlib.sha256()` and `h.update(b)` must not
    read as harness op `update`, which does not exist and would reach the
    timeline as `aside.unknown`.
    """
    from dycraft.viewer.agent_timeline import ACTION_CATEGORY
    from dycraft.viewer.trajectory import _HARNESS_OP_NAMES

    declared = {key[3:] for key in ACTION_CATEGORY if key.startswith("op:")}
    assert declared <= _HARNESS_OP_NAMES, (
        "an op the timeline can classify must also be detectable: "
        f"{sorted(declared - _HARNESS_OP_NAMES)}")
    assert "update" not in _HARNESS_OP_NAMES, "hashlib's update is not an op"
    assert {"render", "inspect", "compile", "storyboard"} <= _HARNESS_OP_NAMES
