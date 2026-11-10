"""Two guarantees added with the job-scoped resident, both cheap to check.

* Cycles renders on the GPU when the host has one, because an explicit
  ``compute_backend="CPU"`` is refused rather than quietly path-traced on the
  CPU beside an idle card.
* One Blender process serves a job's read-write subcalls, while a read-only
  request — which is every call `motion_reference` makes — still forks its own
  worker, because a resident binds its workspace once at spawn.
"""
from types import SimpleNamespace

import pytest

from dycraft.animation.blender.render_config import _cycles_device
from dycraft.tests._animation_fixtures import service


def _bpy(*backends):
    """A `bpy` whose Cycles preferences enumerate `backends`, in that order."""
    devices = [SimpleNamespace(type=b, name=f"{b} Device", use=False) for b in backends]
    prefs = SimpleNamespace(compute_device_type="NONE", devices=devices,
                            get_devices=lambda: None)
    addon = SimpleNamespace(preferences=prefs)
    addons = {"cycles": addon}
    return SimpleNamespace(context=SimpleNamespace(
        preferences=SimpleNamespace(addons=SimpleNamespace(get=addons.get))))


def _scene():
    return SimpleNamespace(cycles=SimpleNamespace(device=None))


def test_explicit_cpu_is_refused_when_a_gpu_is_available():
    """An agent asking for CPU beside an idle GPU is refused."""
    scene = _scene()
    with pytest.raises(ValueError) as error:
        _cycles_device(_bpy("OPTIX"), scene, "CPU")
    message = str(error.value)
    assert "OPTIX" in message and "AUTO" in message
    assert scene.cycles.device is None, "a refused request must not configure the scene"


def test_explicit_cpu_is_allowed_when_no_gpu_backend_exists():
    scene = _scene()
    assert _cycles_device(_bpy(), scene, "CPU") == ("CPU", [])
    assert scene.cycles.device == "CPU"



def test_the_operator_escape_allows_cpu_even_beside_a_usable_gpu(monkeypatch):
    """The refusal is for the agent, not for the launcher.

    A batch that runs several arms on one box may deliberately keep one off the
    card; "a GPU exists" is not "a GPU is free". The knob is an environment
    variable the harness reads, never an argument the agent can pass.
    """
    monkeypatch.setenv("DYCRAFT_ALLOW_CPU_CYCLES", "1")
    scene = _scene()
    assert _cycles_device(_bpy("OPTIX"), scene, "CPU") == ("CPU", [])
    assert scene.cycles.device == "CPU"

def test_explicit_cpu_is_allowed_without_the_cycles_addon():
    bpy = SimpleNamespace(context=SimpleNamespace(
        preferences=SimpleNamespace(addons=SimpleNamespace(get=lambda _n: None))))
    scene = _scene()
    assert _cycles_device(bpy, scene, "CPU") == ("CPU", [])
    assert scene.cycles.device == "CPU"


def test_the_cpu_probe_restores_the_device_type_it_borrowed():
    """`get_devices` needs `compute_device_type` set; the probe must not keep it."""
    bpy = _bpy()
    prefs = bpy.context.preferences.addons.get("cycles").preferences
    prefs.compute_device_type = "NONE"
    _cycles_device(bpy, _scene(), "CPU")
    assert prefs.compute_device_type == "NONE"


def test_auto_still_selects_a_gpu_backend():
    """Guard the path the refusal points callers at."""
    scene = _scene()
    backend, devices = _cycles_device(_bpy("CUDA"), scene, "AUTO")
    assert (backend, devices) == ("CUDA", ["CUDA Device"])
    assert scene.cycles.device == "GPU"


def test_a_read_only_request_forks_and_a_read_write_one_reuses(tmp_path, monkeypatch):
    """`motion_reference` passes `workspace_readonly=True` for every call."""
    svc = service(tmp_path)
    workspace = tmp_path / "native" / "jobs" / "j" / "workspace"
    workspace.mkdir(parents=True)

    forked = []
    monkeypatch.setattr("dycraft.animation.blender.runtime.run_worker",
                        lambda *a, **k: forked.append(a[0]) or {"ok": True})
    served = []
    monkeypatch.setattr(svc, "resident_for",
                        lambda ws: SimpleNamespace(
                            request=lambda *a, **k: served.append(ws) or {"ok": True}))

    svc.worker(workspace, tmp_path / "out_a", {"op": "probe", "code": "1"}, None,
               workspace_readonly=True)
    assert forked == [workspace] and served == []

    svc.worker(workspace, tmp_path / "out_b", {"op": "edit", "code": "1"}, None)
    assert served == [workspace], "a read-write subcall must reuse the job's process"
    assert len(forked) == 1

    # A probe off the live project is read-only even unasked:
    # a batch probe's snapshot is hardlinked, so a write through it would land
    # in the live project.
    svc.worker(workspace, tmp_path / "out_c", {"op": "probe", "code": "1"}, None)
    assert len(forked) == 2 and served == [workspace]


def test_a_job_closes_every_process_it_started_not_just_its_own_workspace(tmp_path):
    """`initialize` binds `<job>/initial` and renames it away; a repair `build`
    runs against an earlier job's tree. Closing by the job's own workspace
    would leak both, and closing everything would kill a concurrent job."""
    svc = service(tmp_path)
    borrowed = tmp_path / "someone_elses_tree"
    own = tmp_path / "own"
    borrowed.mkdir(), own.mkdir()
    closed = []
    monkey = {str(borrowed.resolve()): SimpleNamespace(close=lambda: closed.append("borrowed")),
              str(own.resolve()): SimpleNamespace(close=lambda: closed.append("own"))}
    svc._residents.update(monkey)
    svc._owned.keys = list(monkey)          # both started by this job
    svc.close_job_residents()
    assert sorted(closed) == ["borrowed", "own"]
    assert svc._residents == {} and svc._owned.keys == []


def test_close_job_residents_leaves_another_thread_s_process_alone(tmp_path):
    """Non-mutating ops run concurrently, so ownership is per thread."""
    svc = service(tmp_path)
    other = tmp_path / "other_job"
    other.mkdir()
    closed = []
    svc._residents[str(other.resolve())] = SimpleNamespace(
        close=lambda: closed.append("other"))
    # This thread started nothing.
    svc.close_job_residents()
    assert closed == [] and str(other.resolve()) in svc._residents


def test_closing_a_job_s_resident_is_idempotent_and_scoped(tmp_path):
    svc = service(tmp_path)
    a, b = tmp_path / "ws_a", tmp_path / "ws_b"
    a.mkdir(), b.mkdir()
    closed = []
    svc._residents = {str(a.resolve()): SimpleNamespace(close=lambda: closed.append("a")),
                      str(b.resolve()): SimpleNamespace(close=lambda: closed.append("b"))}
    svc.close_residents(a)
    assert closed == ["a"] and set(svc._residents) == {str(b.resolve())}
    svc.close_residents(a)                      # already gone
    assert closed == ["a"]
    svc.close_residents()                       # the rest
    assert closed == ["a", "b"] and svc._residents == {}
