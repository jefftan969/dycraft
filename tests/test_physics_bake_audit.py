"""`audit_bakes` over all five solvers, and how the Blender binary is found.

Two subjects, one file, because they are the same feature seen from two ends:
fluid is the solver that made an official Blender binary necessary (the `bpy`
wheel's Mantaflow bindings abort on any bake), and it is also the one solver
with no `is_baked` flag to audit, so it has to be read off its own on-disk
cache. Testing the audit without testing the resolution would leave the
capability silently absent on a host where nothing looks broken.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

#: These read the bubblewrap argv `runtime.command` builds, whatever this machine's own sandbox is.
pytestmark = pytest.mark.usefixtures("namespaces_sandbox")

REPO = Path(__file__).resolve().parents[1]


def _load_worker(monkeypatch, workspace):
    """The real worker module with `bpy` stubbed, as `test_animation_render_config`."""
    monkeypatch.setenv("DYCRAFT_BLENDER_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils",
                        SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    monkeypatch.setitem(sys.modules, "_blender_child", SimpleNamespace())
    spec = importlib.util.spec_from_file_location(
        "bake_audit_worker", REPO / "animation/blender/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache(baked):
    return SimpleNamespace(is_baked=baked)


def _obj(name, modifiers=(), particle_systems=()):
    return SimpleNamespace(name=name, modifiers=list(modifiers),
                           particle_systems=list(particle_systems))


def _bind(worker, objects=(), rigidbody_world=None):
    worker.bpy = SimpleNamespace(
        data=SimpleNamespace(objects=list(objects)),
        context=SimpleNamespace(
            scene=SimpleNamespace(rigidbody_world=rigidbody_world)),
        path=SimpleNamespace(abspath=lambda raw: raw))
    return worker


# ---------------------------------------------------------------- point caches

@pytest.mark.parametrize("kind", ["CLOTH", "SOFT_BODY"])
def test_an_unbaked_modifier_cache_is_refused_and_named(tmp_path, monkeypatch, kind):
    worker = _load_worker(monkeypatch, tmp_path)
    modifier = SimpleNamespace(type=kind, name="sim", point_cache=_cache(False))
    _bind(worker, [_obj("sheet", [modifier])])
    with pytest.raises(ValueError) as caught:
        worker.audit_bakes()
    # Named, because a scene can hold many and "bake something" is not a repair.
    assert "sheet/sim" in str(caught.value)
    modifier.point_cache = _cache(True)
    worker.audit_bakes()


def test_every_unbaked_cache_is_reported_at_once(tmp_path, monkeypatch):
    """One pass, not one failure at a time: an author fixing four solvers
    should not have to compile four times to discover the other three."""
    worker = _load_worker(monkeypatch, tmp_path)
    objects = [
        _obj("a", [SimpleNamespace(type="CLOTH", name="c", point_cache=_cache(False))]),
        _obj("b", [SimpleNamespace(type="SOFT_BODY", name="s", point_cache=_cache(False))]),
        _obj("c", particle_systems=[SimpleNamespace(
            name="p", point_cache=_cache(False),
            settings=SimpleNamespace(type="EMITTER"))]),
    ]
    _bind(worker, objects, rigidbody_world=SimpleNamespace(point_cache=_cache(False)))
    message = str(pytest.raises(ValueError, worker.audit_bakes).value)
    for named in ("a/c", "b/s", "c/p", "scene/rigidbody_world"):
        assert named in message


def test_the_rigid_body_cache_belongs_to_the_scene_not_an_object(tmp_path, monkeypatch):
    """One cache for every rigid body there is, which is why baking any of
    them bakes all of them and why this is not reported per object."""
    worker = _load_worker(monkeypatch, tmp_path)
    world = SimpleNamespace(point_cache=_cache(False))
    _bind(worker, [_obj("crate"), _obj("floor")], rigidbody_world=world)
    assert "scene/rigidbody_world" in str(
        pytest.raises(ValueError, worker.audit_bakes).value)
    world.point_cache = _cache(True)
    worker.audit_bakes()


def test_a_scene_with_no_rigid_body_world_is_not_asked_about_one(tmp_path, monkeypatch):
    worker = _load_worker(monkeypatch, tmp_path)
    _bind(worker, [_obj("crate")], rigidbody_world=None)
    worker.audit_bakes()


# -------------------------------------------------------------------- particles

def test_a_particle_cache_lives_on_the_system_not_the_modifier(tmp_path, monkeypatch):
    """The `PARTICLE_SYSTEM` modifier carries no `point_cache`; the system
    does. Auditing the modifier list alone silently passes every emitter."""
    worker = _load_worker(monkeypatch, tmp_path)
    system = SimpleNamespace(name="spray", point_cache=_cache(False),
                             settings=SimpleNamespace(type="EMITTER"))
    modifier = SimpleNamespace(type="PARTICLE_SYSTEM", name="ParticleSystem")
    _bind(worker, [_obj("nozzle", [modifier], [system])])
    assert "nozzle/spray" in str(pytest.raises(ValueError, worker.audit_bakes).value)


def test_static_hair_is_exempt_but_hair_with_dynamics_is_not(tmp_path, monkeypatch):
    """Static hair is generated, not solved: its cache stays unbaked forever,
    so refusing on it would make any haired character uncompilable."""
    worker = _load_worker(monkeypatch, tmp_path)
    settings = SimpleNamespace(type="HAIR", use_hair_dynamics=False)
    system = SimpleNamespace(name="fur", point_cache=_cache(False), settings=settings)
    _bind(worker, [_obj("cat", particle_systems=[system])])
    worker.audit_bakes()

    settings.use_hair_dynamics = True
    assert "cat/fur" in str(pytest.raises(ValueError, worker.audit_bakes).value)


# ------------------------------------------------------------------------ fluid

def _domain(name, cache_directory):
    settings = SimpleNamespace(cache_directory=cache_directory)
    modifier = SimpleNamespace(type="FLUID", name="Fluid", fluid_type="DOMAIN",
                               domain_settings=settings)
    return _obj(name, [modifier])


def _bake(directory, frames=2):
    """Write what Mantaflow writes, which is the only evidence a bake happened."""
    for sub in ("config", "data", "mesh"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
        for frame in range(frames):
            (directory / sub / f"{sub}_{frame:04d}.uni").write_bytes(b"x")


def test_a_baked_domain_inside_the_workspace_passes(tmp_path, monkeypatch):
    worker = _load_worker(monkeypatch, tmp_path)
    cache = tmp_path / "cache_fluid_4b85f272"
    _bake(cache)
    _bind(worker, [_domain("Domain", str(cache))])
    worker.audit_bakes()


def test_a_liquid_domain_s_own_particles_are_audited_through_its_cache(tmp_path, monkeypatch):
    """A liquid domain carries a `FLIP` particle system whose point cache never
    reads baked; its data is in the domain cache. Auditing that point cache
    would refuse a baked liquid as `LiquidDomain/Liquid`."""
    worker = _load_worker(monkeypatch, tmp_path)
    cache = tmp_path / "cache_fluid_4b85f272"
    _bake(cache)
    domain = _domain("LiquidDomain", str(cache))
    domain.particle_systems = [SimpleNamespace(
        name="Liquid", point_cache=_cache(False), settings=SimpleNamespace(type="FLIP"))]
    _bind(worker, [domain])
    worker.audit_bakes()


def test_an_unbaked_domain_is_refused_by_its_empty_cache(tmp_path, monkeypatch):
    """Mantaflow exposes no `is_baked` (checked on 4.5.13: the modifier has no
    `point_cache` and `domain_settings` has no such flag), and it does not
    create the directory until it bakes. Absence is the signal."""
    worker = _load_worker(monkeypatch, tmp_path)
    _bind(worker, [_domain("Domain", str(tmp_path / "cache_fluid_never"))])
    assert "holds no baked data" in str(
        pytest.raises(ValueError, worker.audit_bakes).value)


def test_a_domain_caching_outside_the_workspace_is_refused(tmp_path, monkeypatch):
    """The failure this exists for. Where the cache lands is decided when the
    domain is created: add one to an unsaved blend and Blender pins it to its
    own temp dir, which is the sandbox's tmpfs. It bakes and reads back inside
    the one call, then is gone -- so without this the run fails later, opaquely.
    """
    worker = _load_worker(monkeypatch, tmp_path / "workspace")
    (tmp_path / "workspace").mkdir()
    outside = tmp_path / "blender_NHO3yv" / "cache_fluid_4b85f272"
    _bake(outside)
    _bind(worker, [_domain("Domain", str(outside))])
    message = str(pytest.raises(ValueError, worker.audit_bakes).value)
    assert "outside the workspace" in message
    assert "will not survive this job" in message


def test_a_domain_with_no_cache_directory_at_all_is_refused(tmp_path, monkeypatch):
    worker = _load_worker(monkeypatch, tmp_path)
    _bind(worker, [_domain("Domain", "")])
    assert "no cache directory" in str(
        pytest.raises(ValueError, worker.audit_bakes).value)


# ------------------------------------------------- finding the Blender binary

def _env(monkeypatch, tmp_path):
    from dycraft.render import _blender_env as module
    monkeypatch.setattr(module, "_BINARY_POINTER_FILE", tmp_path / ".blender_binary")
    monkeypatch.setattr(module, "_BINARY_CONVENTIONAL", (tmp_path / "roots",))
    monkeypatch.delenv(module._BINARY_ENV_VAR, raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    return module


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_the_binary_is_resolved_in_documented_precedence(tmp_path, monkeypatch):
    """env > pointer file > conventional roots > PATH. Each source is shadowed
    by the one above it, which is what lets an operator override a host's
    default without editing anything that is checked in."""
    module = _env(monkeypatch, tmp_path)
    # Introduced lowest-precedence first, so each assertion sees only the
    # sources below it. Creating them all up front makes the test pass for the
    # wrong reason -- a conventional root shadows PATH the moment it exists.
    on_path = _executable(tmp_path / "path" / "blender")
    monkeypatch.setattr(module.shutil, "which", lambda name: str(on_path))
    assert module.blender_binary() == str(on_path), "PATH is the backstop"

    rooted = _executable(tmp_path / "roots" / "blender-4.5.13" / "blender")
    assert module.blender_binary() == str(rooted), "a conventional root beats PATH"

    pointed = _executable(tmp_path / "pointed" / "blender")
    (tmp_path / ".blender_binary").write_text(f"{pointed}\n")
    assert module.blender_binary() == str(pointed), "the pointer file beats a root"

    explicit = _executable(tmp_path / "explicit" / "blender")
    monkeypatch.setenv(module._BINARY_ENV_VAR, str(explicit))
    assert module.blender_binary() == str(explicit), "the env var beats everything"


def test_an_empty_env_var_forces_the_wheel(tmp_path, monkeypatch):
    """The documented escape hatch, and the only way to ask for the `bpy`
    module path on a host that has a usable binary."""
    module = _env(monkeypatch, tmp_path)
    _executable(tmp_path / "pointed" / "blender")
    (tmp_path / ".blender_binary").write_text(str(tmp_path / "pointed" / "blender"))
    assert module.blender_binary() is not None
    monkeypatch.setenv(module._BINARY_ENV_VAR, "")
    assert module.blender_binary() is None


def test_an_unusable_candidate_is_not_returned(tmp_path, monkeypatch):
    """A path that does not exist, or exists without the execute bit, is not a
    Blender. Returning it would trade a clear "no binary" for an exec failure
    inside a sandboxed worker."""
    module = _env(monkeypatch, tmp_path)
    monkeypatch.setenv(module._BINARY_ENV_VAR, str(tmp_path / "nope" / "blender"))
    assert module.blender_binary() is None

    plain = tmp_path / "plain" / "blender"
    plain.parent.mkdir(parents=True)
    plain.write_text("not executable")
    plain.chmod(0o644)
    monkeypatch.setenv(module._BINARY_ENV_VAR, str(plain))
    assert module.blender_binary() is None


def test_a_directory_is_accepted_as_an_extracted_tarball(tmp_path, monkeypatch):
    """`.blender_binary` pointing at an unpacked blender.org directory is the
    shape an installer would write, so it resolves to the binary inside."""
    module = _env(monkeypatch, tmp_path)
    _executable(tmp_path / "blender-4.5.13-linux-x64" / "blender")
    monkeypatch.setenv(module._BINARY_ENV_VAR,
                       str(tmp_path / "blender-4.5.13-linux-x64"))
    assert module.blender_binary() == str(
        (tmp_path / "blender-4.5.13-linux-x64" / "blender").resolve())


# ------------------------------------------- the five solvers, really sandboxed
#
# Everything above runs against a stubbed `bpy`, which proves the audit's logic
# and nothing about whether Blender can actually bake. These run a real worker
# under bubblewrap, twice per solver, because the question that matters is not
# "did the bake return FINISHED" but "is the result still there in the next
# call" -- the workspace is copied per op and the blend is reopened every time.
#
# Fluid is the reason the suite exists: the `bpy` PyPI wheel is built
# `WITH_PYTHON_MODULE=ON` and its Mantaflow bindings abort on any bake
# (`'LevelsetGrid' object has no attribute 'setConst'`), so these skip unless
# `render._blender_env.blender_binary()` finds an official build.

_BUILD = {
    "rigid_body": """
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 5))
bpy.ops.rigidbody.object_add()
bpy.ops.mesh.primitive_plane_add(size=20)
bpy.ops.rigidbody.object_add(type='PASSIVE')
cache = bpy.context.scene.rigidbody_world.point_cache
with bpy.context.temp_override(scene=bpy.context.scene, point_cache=cache):
    bpy.ops.ptcache.bake(bake=True)
state = {'baked': cache.is_baked}
""",
    "cloth": """
bpy.ops.mesh.primitive_grid_add(size=2, location=(0, 0, 3))
mod = bpy.context.object.modifiers.new('Cloth', 'CLOTH')
with bpy.context.temp_override(scene=bpy.context.scene, point_cache=mod.point_cache):
    bpy.ops.ptcache.bake(bake=True)
state = {'baked': mod.point_cache.is_baked}
""",
    "soft_body": """
bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 3))
mod = bpy.context.object.modifiers.new('Softbody', 'SOFT_BODY')
with bpy.context.temp_override(scene=bpy.context.scene, point_cache=mod.point_cache):
    bpy.ops.ptcache.bake(bake=True)
state = {'baked': mod.point_cache.is_baked}
""",
    "particles": """
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 4))
obj = bpy.context.object
obj.modifiers.new('ParticleSystem', 'PARTICLE_SYSTEM')
psys = obj.particle_systems[0]
with bpy.context.temp_override(scene=bpy.context.scene, point_cache=psys.point_cache):
    bpy.ops.ptcache.bake(bake=True)
state = {'baked': psys.point_cache.is_baked}
""",
    # `bake_all()` acts on the ACTIVE object's domain, so the domain has to be
    # made active after the flow object is added -- otherwise Blender reports
    # "Bake failed: invalid domain" and it reads as a broken build.
    "fluid": """
bpy.ops.mesh.primitive_cube_add(size=4, location=(0, 0, 2))
dom = bpy.context.object
dom.name = 'Domain'
bpy.ops.object.modifier_add(type='FLUID')
dom.modifiers['Fluid'].fluid_type = 'DOMAIN'
ds = dom.modifiers['Fluid'].domain_settings
ds.domain_type = 'LIQUID'
ds.resolution_max = 16
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 3))
flow = bpy.context.object
bpy.ops.object.modifier_add(type='FLUID')
flow.modifiers['Fluid'].fluid_type = 'FLOW'
flow.modifiers['Fluid'].flow_settings.flow_type = 'LIQUID'
bpy.ops.object.select_all(action='DESELECT')
dom.select_set(True)
bpy.context.view_layer.objects.active = dom
ds = dom.modifiers['Fluid'].domain_settings
result_bake = bpy.ops.fluid.bake_all()
import os
cache = bpy.path.abspath(ds.cache_directory)
state = {'bake': list(result_bake), 'raw': ds.cache_directory,
         'data_files': len(os.listdir(os.path.join(cache, 'data')))}
""",
}

#: How the second call re-reads what the first one baked, without rebuilding it.
_REREAD = {
    "rigid_body": "state = {'baked': bpy.context.scene.rigidbody_world.point_cache.is_baked}",
    "cloth": "state = {'baked': bpy.data.objects['Grid'].modifiers['Cloth'].point_cache.is_baked}",
    "soft_body": "state = {'baked': bpy.data.objects['Icosphere'].modifiers['Softbody'].point_cache.is_baked}",
    "particles": "state = {'baked': bpy.data.objects['Plane'].particle_systems[0].point_cache.is_baked}",
    "fluid": """
import os
ds = [m.domain_settings for o in bpy.data.objects for m in o.modifiers
      if m.type == 'FLUID' and m.fluid_type == 'DOMAIN'][0]
cache = bpy.path.abspath(ds.cache_directory)
state = {'raw': ds.cache_directory,
         'data_files': len(os.listdir(os.path.join(cache, 'data')))}
""",
}


def _report(code):
    """Wrap solver code so its `state` dict lands in a file the test can read."""
    return (f"{code}\n"
            "import json, pathlib\n"
            "pathlib.Path('/workspace/probe_state.json').write_text(json.dumps(state))\n")


@pytest.mark.expensive
@pytest.mark.parametrize("solver", sorted(_BUILD))
def test_each_solver_bakes_and_survives_into_the_next_call(tmp_path, solver):
    import json

    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.render._blender_env import blender_binary
    from dycraft.tests._scene_fixtures import _stack_scene

    if blender_binary() is None:
        pytest.skip("needs an official Blender build; the bpy wheel cannot bake fluid")

    scene, catalog = _stack_scene()
    (tmp_path / "catalog").mkdir()
    ctx = SimpleNamespace(run_dir=tmp_path, scene_ir=scene, catalog=catalog)
    inputs, workspace = tmp_path / "inputs", tmp_path / "work"
    workspace.mkdir()
    prepare_scene(ctx, inputs)
    readonly = [inputs, tmp_path / "catalog"]
    run_worker(workspace, tmp_path / "init",
               {"op": "initialize", "spec": str(inputs / "scene_spec.json")},
               readonly=readonly)

    run_worker(workspace, tmp_path / "build",
               {"op": "edit", "code": _report(_BUILD[solver])}, readonly=readonly)
    built = json.loads((workspace / "probe_state.json").read_text())

    run_worker(workspace, tmp_path / "reread",
               {"op": "edit", "code": _report(_REREAD[solver])}, readonly=readonly)
    survived = json.loads((workspace / "probe_state.json").read_text())

    if solver == "fluid":
        assert built["bake"] == ["FINISHED"], built
        # Relative to the blend, therefore inside the workspace, therefore
        # carried forward by the per-op copy. An absolute path here is the bug
        # `audit_bakes` refuses.
        assert built["raw"].startswith("//"), built["raw"]
        assert not built["raw"].startswith("//.."), built["raw"]
        assert built["data_files"] > 0 and survived["data_files"] == built["data_files"]
    else:
        assert built["baked"] is True, built
        assert survived["baked"] is True, survived


# ------------------------------------------- which branch `command()` launches

def _command(monkeypatch, tmp_path, binary, *, serve=False):
    import sys

    from dycraft.animation.blender import runtime
    monkeypatch.setattr(runtime, "blender_binary", lambda: binary)
    monkeypatch.setattr(runtime, "blender_python", lambda: sys.executable)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: sys.executable)
    return runtime.command(tmp_path / "work", tmp_path / "out", tmp_path / "job",
                           serve=serve)


def test_a_binary_is_launched_as_blender_and_the_wheel_as_an_interpreter(
        tmp_path, monkeypatch):
    """The two branches are not interchangeable invocations of one program.

    A real Blender needs `--background --factory-startup`, and
    `--python-use-system-env` to honor the `PYTHONPATH=/tools` that
    `worker_env` sets and Blender otherwise ignores, and
    `--python-exit-code 1` so a raising script fails the process instead of
    exiting 0. The wheel needs none of that and takes the script as argv[1].
    """
    import sys

    blender = tmp_path / "blender"
    blender.write_text("#!/bin/sh\n")
    blender.chmod(0o755)

    under_binary = _command(monkeypatch, tmp_path, str(blender))
    assert under_binary[-7:] == [
        str(blender), "--background", "--factory-startup",
        "--python-use-system-env", "--python-exit-code", "1",
        "--python", "/tools/worker.py"][-7:]
    assert "--background" in under_binary and "--factory-startup" in under_binary

    under_wheel = _command(monkeypatch, tmp_path, None)
    assert under_wheel[-3:] == [sys.executable, "-B", "/tools/worker.py"]
    assert "--background" not in under_wheel


def test_serve_reaches_the_script_through_a_separator_only_under_a_binary(
        tmp_path, monkeypatch):
    """`worker.py` looks for `--serve` in `sys.argv`. Blender only leaves
    arguments there if they follow `--`; passing it bare would have Blender
    itself try to parse it and fail. The wheel takes it directly."""
    blender = tmp_path / "blender"
    blender.write_text("#!/bin/sh\n")
    blender.chmod(0o755)

    assert _command(monkeypatch, tmp_path, str(blender), serve=True)[-2:] == ["--", "--serve"]
    assert _command(monkeypatch, tmp_path, None, serve=True)[-1] == "--serve"
    assert "--" not in _command(monkeypatch, tmp_path, None, serve=True)[-2:-1]
