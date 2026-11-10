"""Where the two workspace audits run, and why they are two.

`audit_dependencies` refuses a file dependency outside the workspace and
`audit_bakes` refuses an unbaked physics cache. Both run where a state is
checked (`evaluate`, a render's fused check, a replay), not on save: during
`build` the `update` merge holds a reference to the donor blend until the file
has been saved and reopened, and failing a save on an unbaked cache would break
authoring cloth in one call and baking it in the next.
"""
from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest


def _worker(monkeypatch, tmp_path, *, blend_paths=(), objects=(),
            libraries=(), images=(), weak=()):
    """Load the real worker module against a fake `bpy`.

    `libraries`/`images` are the *holders* a real dependency hangs off, and
    `weak` is a list of filepaths carried as `library_weak_reference`
    provenance. The audit distinguishes them, so a fake that reports paths
    with no holder at all cannot exercise it.
    """
    from dycraft.animation.blender.runtime import REPO
    bpy = SimpleNamespace(
        utils=SimpleNamespace(blend_paths=lambda **kw: list(blend_paths)),
        path=SimpleNamespace(abspath=lambda p: p),
        data=SimpleNamespace(
            objects=list(objects),
            libraries=[SimpleNamespace(filepath=p) for p in libraries],
            images=[SimpleNamespace(filepath=p, packed_file=None) for p in images],
            meshes=[SimpleNamespace(library_weak_reference=SimpleNamespace(filepath=p))
                    for p in weak]),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    monkeypatch.setitem(sys.modules, "_blender_child", SimpleNamespace())
    monkeypatch.setenv("DYCRAFT_BLENDER_WORKSPACE_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("audit_worker", REPO / "animation/blender/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cloth(name="cloth", baked=False):
    modifier = SimpleNamespace(type="CLOTH", name="Cloth",
                               point_cache=SimpleNamespace(is_baked=baked))
    return SimpleNamespace(name=name, modifiers=[modifier])


def test_a_dependency_inside_the_workspace_is_accepted(monkeypatch, tmp_path):
    inside = tmp_path / "texture.png"
    inside.write_bytes(b"x")
    worker = _worker(monkeypatch, tmp_path, blend_paths=[str(inside)])
    worker.audit_dependencies()          # must not raise


def test_a_dependency_in_another_jobs_directory_is_rejected(monkeypatch, tmp_path):
    """The exact r2 failure: a link into a previous job's donor directory."""
    other = tmp_path.parent / "jobs" / "e7151de4bd66469d" / "donor" / "animation.blend"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")
    worker = _worker(monkeypatch, tmp_path, blend_paths=[str(other)])
    with pytest.raises(ValueError, match="Unpacked dependency outside workspace"):
        worker.audit_dependencies()


def test_appended_provenance_is_not_a_dependency(monkeypatch, tmp_path):
    """`blend_paths()` reports more than dependencies.

    Appending with `link=False` leaves a `library_weak_reference` on every
    datablock it copied, recording the donor it came from. The data is fully
    local -- the donor can be deleted and the blend opens identically -- but
    `blend_paths()` reports those notes alongside real external references, so
    counting them would refuse the check after a `build` that succeeded,
    naming a job directory the author never chose.

    In Blender 4.5 the notes **survive save, `pack_all()`,
    `make_paths_relative()`, and reopen**, and Python cannot clear them
    (`LibraryWeakReference` is read-only), so no repair the author could make
    would satisfy such a refusal.
    """
    donor = tmp_path.parent / "jobs" / "e7151de4bd66469d" / "donor" / "animation.blend"
    # `tmp_path.parent` is shared with the sibling test above, which builds the
    # same path to model the same r2 failure from the other side.
    donor.parent.mkdir(parents=True, exist_ok=True)
    donor.write_bytes(b"x")
    worker = _worker(monkeypatch, tmp_path, blend_paths=[str(donor)], weak=[str(donor)])
    worker.audit_dependencies()          # must not raise


def test_a_real_library_link_outside_the_workspace_is_still_rejected(monkeypatch, tmp_path):
    """The other half: narrowing the audit must not blind it.

    A linked library really is a dependency -- the data lives in the other file
    -- so the packaged project would open broken elsewhere.
    """
    other = tmp_path.parent / "assets" / "rig.blend"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"x")
    worker = _worker(monkeypatch, tmp_path, blend_paths=[str(other)],
                     libraries=[str(other)])
    with pytest.raises(ValueError, match="linked library"):
        worker.audit_dependencies()


def test_an_unpacked_image_outside_the_workspace_is_rejected(monkeypatch, tmp_path):
    outside = tmp_path.parent / "texture.png"
    outside.write_bytes(b"x")
    worker = _worker(monkeypatch, tmp_path, blend_paths=[str(outside)],
                     images=[str(outside)])
    with pytest.raises(ValueError, match="image"):
        worker.audit_dependencies()


def test_a_missing_dependency_is_rejected(monkeypatch, tmp_path):
    worker = _worker(monkeypatch, tmp_path, blend_paths=[str(tmp_path / "gone.png")])
    with pytest.raises(ValueError, match="Unpacked dependency outside workspace or missing"):
        worker.audit_dependencies()


def test_saving_does_not_require_baked_physics(monkeypatch, tmp_path):
    """The dependency audit does not check bakes.

    Authoring cloth in one `h.blender` call and baking it in the next must not
    fail the dependency audit; only the bake audit refuses an unbaked cache.
    """
    worker = _worker(monkeypatch, tmp_path, objects=[_cloth(baked=False)])
    worker.audit_dependencies()          # must not raise


def test_checking_does_require_baked_physics(monkeypatch, tmp_path):
    worker = _worker(monkeypatch, tmp_path, objects=[_cloth(baked=False)])
    with pytest.raises(ValueError, match="Bake cloth/Cloth before checking"):
        worker.audit_bakes()
    worker_ok = _worker(monkeypatch, tmp_path, objects=[_cloth(baked=True)])
    worker_ok.audit_bakes()              # must not raise


# ---------------------------------------------------------------------------
# `scene['dycraft_states']`: a dict literal or a JSON string
# ---------------------------------------------------------------------------

class _IDPropertyGroup:
    """What Blender stores a dict literal assigned to a scene property as."""

    def __init__(self, value):
        self._value = value

    def to_dict(self):
        return dict(self._value)


def _states_worker(monkeypatch, tmp_path, value):
    # Inside Blender the worker imports its sibling as top-level `verify`.
    from dycraft.animation.blender import verify
    monkeypatch.setitem(sys.modules, "verify", verify)
    worker = _worker(monkeypatch, tmp_path)
    scene = {} if value is None else {"dycraft_states": value}
    worker.bpy.context = SimpleNamespace(scene=scene)
    return worker


_BINDINGS = {"start": {"clip": "approach", "frame": 0},
             "settled": {"clip": "settle", "frame": 3}}
_INFO = {"clips": {"approach": {"frames": [1, 9]}, "settle": {"frames": [21, 29]}},
         "frame_start": 0, "n_frames": 40}


@pytest.mark.parametrize("stored", [
    pytest.param(__import__("json").dumps(_BINDINGS), id="json-string"),
    pytest.param(_IDPropertyGroup(_BINDINGS), id="dict-literal"),
])
def test_state_bindings_read_both_spellings_the_same(monkeypatch, tmp_path, stored):
    """The prompt shows a dict literal, which Blender keeps as an
    IDPropertyGroup; a JSON string means the same bindings."""
    worker = _states_worker(monkeypatch, tmp_path, stored)
    assert worker.scene_json("dycraft_states", {}) == _BINDINGS
    states = [{"id": "start"}, {"id": "settled"}]
    assert worker.state_clips(states) == {"start": "approach", "settled": "settle"}
    assert worker.resolve_states(_INFO, states) == {"start": 1, "settled": 24}


def test_absent_states_are_no_bindings(monkeypatch, tmp_path):
    assert _states_worker(monkeypatch, tmp_path, None).scene_json("dycraft_states", {}) == {}
    assert _states_worker(monkeypatch, tmp_path, "").scene_json("dycraft_states", {}) == {}


@pytest.mark.parametrize("stored, says", [
    ("{'start': ", "is a string that is not JSON"),
    ("[1, 2]", "must map state ids"),
    (7, "must map state ids"),
])
def test_a_malformed_states_property_is_a_rejection_naming_it(monkeypatch, tmp_path,
                                                              stored, says):
    """A ValueError, which the harness replies with as a rejection, naming the
    property -- not a TypeError, which would read as a harness fault."""
    worker = _states_worker(monkeypatch, tmp_path, stored)
    with pytest.raises(ValueError, match=says) as err:
        worker.resolve_states(_INFO, [{"id": "start"}])
    assert "scene['dycraft_states']" in str(err.value)


def test_an_unbound_state_is_told_the_dict_form(monkeypatch, tmp_path):
    worker = _states_worker(monkeypatch, tmp_path, _IDPropertyGroup({}))
    with pytest.raises(ValueError, match="has no frame and no binding") as err:
        worker.resolve_states(_INFO, [{"id": "start"}])
    assert "scene['dycraft_states'] = {'start': {'clip': ..., 'frame': ...}" in str(err.value)
