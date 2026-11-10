"""`h.ik` on the animation layer: the agent's own cuRobo code in a persistent `ik/`.

The code runs in `<run>/ik/`, which outlives the call, rather than once per
frame inside an export, so a script is not limited to the robot and world
dycraft builds for it. It exports only when asked (`frame`/`frames`), into
`ik/exports/`. No GPU here:
the child is replaced, and what is pinned is what it is handed.
"""
from __future__ import annotations

import importlib
import json
import threading
from types import SimpleNamespace

import pytest

from dycraft.animation.blender import solvers

# `dycraft.animation.ik` re-exports the function `ik`, which shadows the module
# of the same name on attribute access.
ik_mod = importlib.import_module("dycraft.animation.ik.ik")


def _service(tmp_path, calls):
    run = tmp_path / "run"
    run.mkdir()
    native = run / "native"
    native.mkdir()

    def worker(workspace, out, request, cancel):
        calls.setdefault("worker", []).append(request)

    return SimpleNamespace(root=run, native=native, worker=worker,
                           version_of=lambda ws: "abcdef0123456789",
                           ctx=SimpleNamespace(scene_ir={}, catalog={}))


@pytest.fixture
def fake_child(monkeypatch):
    seen = {}

    def fake_ik(code, **kw):
        seen.update(code=code, **kw)
        return {"ok": True, "result": {"n": 1}}

    monkeypatch.setattr(ik_mod, "ik", fake_ik)
    # The plumbing under test runs only where IK is available.
    monkeypatch.setattr(ik_mod, "availability", lambda: {"available": True, "reason": None, "detail": ""})
    return seen


def test_no_frame_means_no_export_and_no_blender_pass(tmp_path, fake_child):
    calls = {}
    svc = _service(tmp_path, calls)
    out = solvers.consult(svc, "ik", {"code": "result['n'] = 1"}, tmp_path, tmp_path / "job",
                          threading.Event())
    assert "worker" not in calls
    assert out["exports"] == {} and out["result"] == {"n": 1}
    assert fake_child["ik_dir"] == svc.root / "ik" and (svc.root / "ik").is_dir()
    assert fake_child["require_scene"] is False
    assert json.loads(fake_child["extra_env"]["DYCRAFT_IK_EXPORTS"]) == {}


def test_requested_frames_defaults_to_none():
    assert solvers.requested_frames({}) == []
    assert solvers.requested_frames({"frame": 0}) == [0]
    with pytest.raises(ValueError):
        solvers.requested_frames({"frames": list(range(17))})


def test_a_script_under_the_run_dir_is_read_and_recorded(tmp_path, fake_child):
    svc = _service(tmp_path, {})
    (svc.root / "ik").mkdir()
    (svc.root / "ik" / "plan.py").write_text("result['ok'] = 1\n")
    out = solvers.consult(svc, "ik", {"script": "ik/plan.py", "timeout_s": 1800},
                          tmp_path, tmp_path / "job", threading.Event())
    assert fake_child["code"] == "result['ok'] = 1\n"
    assert fake_child["timeout_s"] == 1800
    assert out["script"] == "ik/plan.py"


@pytest.mark.parametrize("args", [{}, {"code": "x=1", "script": "ik/a.py"},
                                  {"script": "../outside.py"}, {"script": "ik/missing.py"}])
def test_code_and_script_are_exclusive_and_scripts_stay_in_the_run(tmp_path, fake_child, args):
    svc = _service(tmp_path, {})
    (tmp_path / "outside.py").write_text("x = 1\n")
    with pytest.raises(ValueError):
        solvers.consult(svc, "ik", args, tmp_path, tmp_path / "job", threading.Event())


def test_an_export_is_reused_once_on_disk(tmp_path, fake_child, monkeypatch):
    calls = {}
    svc = _service(tmp_path, calls)
    (svc.native / "baseline.json").write_text("{}")
    exported = []

    def fake_export(scene_ir, catalog, base, baseline, evaluation, frame, dest, **kw):
        dest.mkdir(parents=True)
        (dest / "scene_meta.json").write_text("{}")
        exported.append(frame)

    monkeypatch.setattr(solvers, "export_state", fake_export)
    for _ in range(2):
        out = solvers.consult(svc, "ik", {"code": "x=1", "frames": [3, 7]}, tmp_path,
                              tmp_path / "job", threading.Event())
    assert exported == [3, 7] and len(calls["worker"]) == 1
    assert set(out["exports"]) == {"3", "7"}
    assert out["exports"]["3"].startswith(str(svc.root / "ik" / "exports"))
    assert json.loads(fake_child["extra_env"]["DYCRAFT_IK_EXPORTS"]) == out["exports"]


def test_a_preloaded_urdf_robot_is_named_to_the_script(tmp_path, fake_child):
    svc = _service(tmp_path, {})
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "import.json").write_text(json.dumps(
        {"source_urdf": "/robots/g1.urdf", "package_root": "/robots"}))
    (svc.root / "meta.json").write_text(json.dumps(
        {"preloaded_assets": {"g1": {"source": str(asset)}, "cafe": {"source": str(tmp_path)}}}))
    out = solvers.consult(svc, "ik", {"code": "x=1"}, tmp_path, tmp_path / "job",
                          threading.Event())
    assert out["robots"] == {"g1": {"urdf": "/robots/g1.urdf", "package_root": "/robots"}}
    assert json.loads(fake_child["extra_env"]["DYCRAFT_IK_ROBOTS"]) == out["robots"]


