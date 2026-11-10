"""`viewer/_trace.py`: one scan of a trace serves every reader, and gives what decoding every line would."""
from __future__ import annotations

import json
import os

import pytest

from dycraft.viewer import _trace


def _line(record: dict) -> str:
    # The writer's own layout: sorted keys, default separators.
    return json.dumps(record, sort_keys=True) + "\n"


def _records():
    big = "x" * (_trace.LONG + 10)
    return [
        {"ev": "daemon_start", "ts": "2026-01-01T00:00:00"},
        {"ev": "op_call", "op": "edit", "step": 1, "args": {"code": "short"}},
        {"ev": "job_result", "step": 1, "value": big},
        {"ev": "op_result", "op": "edit", "step": 1, "elapsed_ms": 3.0},
        # A long call whose arguments hold `ev` keys of their own.
        {"ev": "op_call", "op": "probe", "step": 2,
         "args": {"expression": "scene.objects " + big, "nested": {"ev": "op_result"}}},
        {"ev": "op_call", "op": "edit", "step": 3, "args": {"code": big}},
        {"ev": "compile_attempt", "step": 3, "detail": big},
    ]


def _naive(path, wanted, args):
    """Every line decoded, then filtered and projected as `events` promises."""
    out = []
    for raw in path.read_bytes().splitlines():
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if record.get("ev") not in wanted:
            continue
        if record["ev"] == "op_call":
            keys = args.get(record.get("op"))
            record["args"] = ({k: (v[:_trace.ARG_CHARS] if isinstance(v, str) else v)
                               for k, v in record["args"].items() if k in keys} if keys else None)
        out.append(record)
    return out


@pytest.fixture
def trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(_line(r) for r in _records()) + "{not json\n")
    return path


def test_each_reader_gets_what_decoding_every_line_gives(trace):
    readers = [({"op_call"}, {}),
               ({"op_call", "op_result", "compile_attempt", "daemon_start"}, {"probe": ("expression",)}),
               ({"op_call"}, {"edit": ("code",), "probe": ("expression", "nested")}),
               ({"job_result"}, {})]
    for wanted, args in readers:
        _trace.register(wanted, args)
    for wanted, args in readers:
        assert list(_trace.events(trace, wanted, args=args)) == _naive(trace, wanted, args)


def test_an_op_s_arguments_keep_only_the_fields_named_and_short(trace):
    _trace.register({"op_call"}, {"probe": ("expression",)})
    probe = next(r for r in _trace.events(trace, {"op_call"}, args={"probe": ("expression",)})
                 if r["op"] == "probe")
    assert set(probe["args"]) == {"expression"}
    assert probe["args"]["expression"].startswith("scene.objects ")
    assert len(probe["args"]["expression"]) == _trace.ARG_CHARS


def test_a_file_is_scanned_once_for_every_reader(trace, monkeypatch):
    _trace.register({"op_call", "op_result"}, {"probe": ("expression",)})
    scans = []
    real = _trace._scan
    monkeypatch.setattr(_trace, "_scan", lambda *a: scans.append(a[1:3]) or real(*a))
    list(_trace.events(trace, {"op_call"}))
    list(_trace.events(trace, {"op_result"}, args={"probe": ("expression",)}))
    assert [s for s in scans if s[0] != s[1]] == [(0, trace.stat().st_size)]


def test_a_growing_trace_is_read_on_and_a_half_written_line_is_not_kept(trace):
    wanted = {"op_call", "op_result"}
    _trace.register(wanted)
    before = list(_trace.events(trace, wanted))
    record = {"ev": "op_result", "op": "probe", "step": 2, "elapsed_ms": 9.0}
    line = _line(record)
    with open(trace, "a") as fh:
        fh.write(line[:-1])                   # the newline not yet written
    assert list(_trace.events(trace, wanted)) == before + [record]
    with open(trace, "a") as fh:
        fh.write("\n")
    assert list(_trace.events(trace, wanted)) == before + [record]
    assert list(_trace.events(trace, wanted)) == _naive(trace, wanted, {})


def test_a_replaced_trace_is_read_again(trace):
    wanted = {"op_call"}
    _trace.register(wanted)
    list(_trace.events(trace, wanted))
    replacement = trace.with_name("new.jsonl")
    replacement.write_text(_line({"ev": "op_call", "op": "view", "step": 1, "args": {}}))
    os.replace(replacement, trace)
    assert list(_trace.events(trace, wanted)) == [{"ev": "op_call", "op": "view", "step": 1, "args": None}]


def test_a_missing_or_empty_trace_has_no_records(tmp_path):
    assert list(_trace.events(tmp_path / "absent.jsonl", {"op_call"})) == []
    (tmp_path / "empty.jsonl").write_text("")
    assert list(_trace.events(tmp_path / "empty.jsonl", {"op_call"})) == []
