"""Native bundle integrity, frame semantics, and isolated bpy integration."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dycraft.animation.blender.bundle import make_manifest, validate_manifest
from dycraft.animation.blender.verify import (frame_indices, load_frame,
                                              out_of_room, penetration_pairs, report)
from dycraft.animation.blender.continuity import frame_runs
from dycraft.render._blender_env import blender_python

#: `verify()` imports `bpy`; with no bpy it delegates to `_offline_verify`,
#: which re-runs itself under the Blender interpreter and raises outright when
#: none is configured. So every test that reaches `verify()` needs one — the
#: same condition the `pytest.skip("Blender interpreter not configured")` calls
#: further down guard, expressed as a marker because these have no other setup
#: to hang it off.
_needs_blender = pytest.mark.skipif(
    blender_python() is None,
    reason="needs a configured Blender interpreter (native verify path)")


def test_bundle_detects_asset_changes_and_path_escape(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "evaluation").mkdir(parents=True)
    (bundle / "animation.blend").write_bytes(b"blend")
    (bundle / "evaluation/index.json").write_text("{}")
    info = {"fps": 29.97, "n_frames": 60, "frame_start": 7, "blender_version": "test"}
    manifest = make_manifest(bundle, scene_ref="scene", evaluation=info)
    validate_manifest(tmp_path, manifest)
    (bundle / "animation.blend").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        validate_manifest(tmp_path, manifest)


def test_fragment_preserves_fractional_fps_and_inclusive_endpoints():
    """Output frame indices are the interval unit; seconds is the named alternate.

    `frame_indices(info, 4, 4)` means frame 4, not "4.0 seconds in", because
    every other index on this surface — snapshot frames, plan state frames, ik
    frames, camera_path frames — is a zero-based output index, and the author
    should not have to divide by the frame rate for one of them. Both meanings
    are reachable; only one of them is the default.
    """
    info = {"n_frames": 101, "fps": 25.0}
    assert frame_indices(info, 4, 4) == [4]
    assert frame_indices(info, 0, 2) == [0, 1, 2]
    assert frame_indices(info) == list(range(101))
    assert frame_indices(info, seconds=[4, 4]) == [100]
    assert frame_indices(info, seconds=[1.0, 1.08]) == [25, 26, 27]
    # A float is a seconds value that lost its keyword. Refuse it rather than
    # silently reading 0.0833 as frame 0.
    with pytest.raises(ValueError):
        frame_indices(info, 2 / 24, 2 / 24)
    with pytest.raises(ValueError):
        frame_indices(info, True, 3)
    with pytest.raises(ValueError):  # one end is not an interval
        frame_indices(info, 4)
    with pytest.raises(ValueError):  # ordered
        frame_indices(info, 5, 4)
    with pytest.raises(ValueError):  # inside the timeline
        frame_indices(info, 0, 101)
    with pytest.raises(ValueError):  # never both spellings
        frame_indices(info, 0, 2, seconds=[0, 1])
    with pytest.raises(ValueError):
        frame_indices(info, seconds=[float("nan"), 2])
    with pytest.raises(ValueError):
        frame_indices(info, seconds=[4, 5])


@pytest.mark.expensive
def test_blender_import_edit_evaluate_isolated(tmp_path):
    from dycraft.tests._scene_fixtures import _stack_scene
    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.render._blender_env import blender_python
    if blender_python() is None:
        pytest.skip("Blender interpreter not configured")
    scene, catalog = _stack_scene()
    (tmp_path / "catalog").mkdir()
    ctx = SimpleNamespace(run_dir=tmp_path, scene_ir=scene, catalog=catalog)
    inputs, workspace = tmp_path / "inputs", tmp_path / "work"
    workspace.mkdir()
    prepare_scene(ctx, inputs)
    readonly = [inputs, tmp_path / "catalog"]
    run_worker(workspace, tmp_path / "init", {"op": "initialize", "spec": str(inputs / "scene_spec.json")}, readonly=readonly)
    assert (workspace / "animation.blend").is_file()
    before = run_worker(workspace, tmp_path / "before", {"op": "evaluate"}, readonly=readonly)
    run_worker(workspace, tmp_path / "edit", {"op": "edit", "code": '''
s = bpy.context.scene
s.frame_start = 7
s.frame_end = 9
s.render.fps = 30
s.render.fps_base = 1.001
obj = bpy.data.objects['part/box/body']
obj.shape_key_add(name='Basis')
key = obj.shape_key_add(name='Bend')
key.data[0].co.z += .1
key.value = 0
key.keyframe_insert('value', frame=7)
key.value = 1
key.keyframe_insert('value', frame=9)
result['objects'] = len(bpy.data.objects)
'''}, readonly=readonly)
    info = run_worker(workspace, tmp_path / "after", {"op": "evaluate"}, readonly=readonly)
    assert info["frame_start"] == 7
    assert info["fps"] == pytest.approx(30 / 1.001)
    _, first = load_frame(tmp_path / "after/evaluation", 0)
    _, last = load_frame(tmp_path / "after/evaluation", 2)
    _, original = load_frame(tmp_path / "before/evaluation", 0)
    assert np.allclose(first["part/box/body"][0], original["part/box/body"][0])
    a, b = first["part/box/body"][0], last["part/box/body"][0]
    assert a.shape != b.shape or not np.allclose(a, b)

    # Seeking a fragment must preserve evaluated shape keys and fractional FPS.
    seek = run_worker(workspace, tmp_path / "seek", {
        "op": "evaluate", "frames": [1, 2], "seek_to_start": True}, readonly=readonly)
    sequential = run_worker(workspace, tmp_path / "sequential", {
        "op": "evaluate", "frames": [1, 2]}, readonly=readonly)
    # No history in this scene, so both jump to their first frame; the
    # flag does not choose.
    assert seek["evaluation_start_frame"] == sequential["evaluation_start_frame"] == 1
    assert seek["frames"] == sequential["frames"] == [1, 2]
    from dycraft.animation.blender.evaluation import read_record
    for i in (1, 2):
        a = read_record(tmp_path / "seek/evaluation", i)
        b = read_record(tmp_path / "sequential/evaluation", i)
        assert a["signature"] == b["signature"]
        assert a == b, "seeking to a frame must evaluate exactly what walking to it does"

    # Native Blender edits exercise the relaxed rules, not just synthetic records.
    from dycraft.animation.blender.verify import verify
    run_worker(workspace, tmp_path / "relax_edit", {"op": "edit", "code": """
obj = bpy.data.objects['part/box/body']
obj.name = 'cut_piece'
obj.hide_render = True
bpy.data.objects['link/box__body'].location.z += .2
bpy.data.objects.remove(bpy.data.objects['part/table/body'], do_unlink=True)
"""}, readonly=readonly)
    run_worker(workspace, tmp_path / "relaxed", {"op": "evaluate"}, readonly=readonly)
    baseline = json.loads((tmp_path / "init/baseline.json").read_text())
    reports = verify(tmp_path / "relaxed/evaluation", baseline, scene, catalog)
    assert all(r['verdict'] != 'fail' for r in reports), reports
    record, _ = load_frame(tmp_path / "relaxed/evaluation", 0)
    assert next(o for o in record['objects'] if o['name'] == 'cut_piece')['hidden']


def test_jobs_cancel_and_recover_without_polling_steps(tmp_path):
    import threading
    import time
    from dycraft.animation.blender.runtime import Jobs
    from dycraft.animation.blender.bundle import write_json
    entered = threading.Event()
    jobs = Jobs(tmp_path)
    def blocked(directory, cancel):
        entered.set()
        cancel.wait(2)
        if cancel.is_set():
            raise RuntimeError("cancelled")
        return {"ok": True}
    receipt = jobs.submit("test", blocked, step=7, version="v")
    assert entered.wait(1)
    jobs.cancel(receipt["job_id"])
    jobs.close()
    assert jobs.status(receipt["job_id"])["status"] == "cancelled"
    assert Jobs(tmp_path).status(receipt["job_id"])["step"] == 7
    write_json(tmp_path / "unfinished/job.json", {"job_id": "unfinished", "status": "running"})
    assert Jobs(tmp_path).status("unfinished")["status"] == "interrupted"


def test_the_scene_spec_carries_which_instance_replaced_a_shell_surface(tmp_path):
    """The animation verifier resolves `room.floor` to a modeled floor through
    this mapping; it had no way to, so events on it went unchecked."""
    from dycraft.tests._scene_fixtures import _hinged_object
    from dycraft.sdk import Scene, ObjectHandle
    from dycraft.sdk.scene_builder import _reset_scene_singleton
    from dycraft.animation.blender.import_scene import prepare_scene
    _reset_scene_singleton()
    obj = _hinged_object()
    s = Scene("replaced_floor")
    s.set_room(width=4, depth=4, height=3)
    slab = s.place(ObjectHandle(obj), name="slab", pose=((0, 0, 0), (0, 0, 0)))
    s.replace_room_surface("floor", slab)
    scene = s.finalize()
    spec = prepare_scene(SimpleNamespace(run_dir=tmp_path, scene_ir=scene,
                                         catalog={obj["id"]: obj}), tmp_path / "inputs")
    assert spec["surface_replacements"] == {"floor": "slab"}


@pytest.mark.expensive
@pytest.mark.parametrize("angle", [60, 270, -270, 630])
def test_native_joint_and_solver_default_use_evaluated_configuration(tmp_path, angle):
    from dycraft.tests._scene_fixtures import _hinged_object
    from dycraft.sdk import Scene, ObjectHandle
    from dycraft.sdk.scene_builder import _reset_scene_singleton
    from dycraft.animation.blender.import_scene import prepare_scene
    from dycraft.animation.blender.runtime import run_worker
    from dycraft.animation.blender.solvers import export_state
    _reset_scene_singleton()
    import math
    from dycraft.sdk.ir import content_hash
    obj = _hinged_object()
    obj["joints"][0]["limits"] = [math.radians(min(0, angle)), math.radians(max(360, angle))]
    obj["id"] = content_hash({k:v for k,v in obj.items() if k != "id"})
    s = Scene("joint_test")
    s.set_room(width=4, depth=4, height=3)
    s.place(ObjectHandle(obj), name="cabinet", pose=((0,0,0), (0,0,0)))
    scene, catalog = s.finalize(), {obj["id"]: obj}
    (tmp_path / "catalog").mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inputs = tmp_path / "inputs"
    prepare_scene(SimpleNamespace(run_dir=tmp_path, scene_ir=scene, catalog=catalog), inputs)
    readonly = [inputs, tmp_path / "catalog"]
    run_worker(workspace, tmp_path / "init", {"op": "initialize", "spec": str(inputs / "scene_spec.json")}, readonly=readonly)
    run_worker(workspace, tmp_path / "edit", {"op": "edit", "code": '''
key_joint('cabinet__hinge', 0, 0)
key_joint('cabinet.hinge', ANGLE, 1)
bpy.context.scene.frame_end = 1
'''.replace('ANGLE', repr(math.radians(angle)))}, readonly=readonly)
    run_worker(workspace, tmp_path / "state", {"op": "evaluate"}, readonly=readonly)
    a, _ = load_frame(tmp_path / "state/evaluation", 0)
    b, _ = load_frame(tmp_path / "state/evaluation", 1)
    assert not np.allclose(a["transforms"]["part/cabinet/door"], b["transforms"]["part/cabinet/door"])
    seek = run_worker(workspace, tmp_path / "seek_joint", {
        "op": "evaluate", "frames": [1], "seek_to_start": True}, readonly=readonly)
    seek_record, _ = load_frame(tmp_path / "seek_joint/evaluation", 1)
    assert seek["evaluation_start_frame"] == 1
    assert seek_record["signature"] == b["signature"]
    baseline = json.loads((tmp_path / "init/baseline.json").read_text())
    meta = export_state(scene, catalog, tmp_path / "catalog", baseline, tmp_path / "state/evaluation", 1, tmp_path / "export")
    assert meta["joints"]["cabinet"]["hinge"]["default"] == pytest.approx(angle, abs=1e-4)

    from dycraft.animation.blender.verify import joint_position
    joint = baseline["joints"]["cabinet__hinge"]
    assert b["joint_positions"][joint["object"]] == pytest.approx(math.radians(angle))
    assert joint_position(b, joint) == pytest.approx(math.radians(angle))


def _continuity_sequence(positions, *, rotations=None, deformation=None, hidden=(), topology_change=(), fps=30):
    from dycraft.animation.blender.continuity import MotionContinuity
    from scipy.spatial.transform import Rotation
    check = MotionContinuity(fps)
    vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1]])
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    for i, x in enumerate(positions):
        mat = np.eye(4)
        mat[0, 3] = x
        if rotations is not None:
            mat[:3, :3] = Rotation.from_euler('z', rotations[i], degrees=True).as_matrix()
        local = vertices.copy()
        if deformation is not None:
            local[3, 2] += deformation[i]
        obj = {'name': 'moving', 'matrix': mat.tolist(), 'hidden': i in hidden}
        triangles = tris[::-1].copy() if i in topology_change else tris
        check.add(i, {'objects': [obj]}, {'moving': (local @ mat[:3, :3].T + mat[:3, 3], triangles)})
    return check.report()


def test_fast_smooth_motion_is_not_reported_at_all():
    """There is no speed channel; constant velocity is not a finding.

    A speed bound tests displacement against an absolute threshold (such as
    `max(.5 m/s, 2 x body diagonal)` or `pi rad/s`), so a body in steady motion
    is flagged for as long as it moves, and measuring relative to a kinematic
    parent does not help: the fault is a threshold calibrated below ordinary
    motion, not the frame of reference. The spike channel is scale-free by
    construction and is what localizes a teleport.
    """
    for positions in (np.arange(12) * 2.,              # fast, constant velocity
                      np.arange(8) * .02,              # slow, constant velocity
                      [0, .01, .02, .03, .02, .01, 0]):  # slow, smooth reversal
        report = _continuity_sequence(positions)
        assert report['verdict'] == 'pass', positions
        assert 'fast' not in report['evidence']
        assert 'speed_warning_count' not in report['evidence']
        assert 'spikes' not in report['evidence']

    # Steady rotation is the same: no channel, no row, whatever the rate.
    rotating = _continuity_sequence(np.zeros(8), rotations=np.arange(8) * 10)
    assert rotating['verdict'] == 'pass'
    assert 'fast' not in rotating['evidence']


@pytest.mark.parametrize('channel', ['translation', 'rotation', 'deformation'])
def test_continuity_reports_teleport_rotation_and_deformation_spikes(channel):
    motion = np.array([0.] * 5 + [1.] * 6)
    kwargs = {}
    if channel == 'rotation':
        kwargs['rotations'] = motion * 90
    if channel == 'deformation':
        kwargs['deformation'] = motion
    report = _continuity_sequence(motion if channel == 'translation' else motion * 0, **kwargs)
    assert report['verdict'] == 'uncertain'
    warning = next(w for w in report['evidence']['spikes'] if w['channel'] == channel)
    assert warning['frames'] == [[4, 5]]
    assert warning['window'] == pytest.approx([4/30, 5/30], abs=5e-5)
    assert report['evidence']['diagnostic_only']


def test_continuity_reports_jitter_but_skips_visibility_and_topology_boundaries():
    report = _continuity_sequence([0, .03, 0, .03, 0, .03, 0])
    assert any(w['reason'] == 'repeated direction reversal' for w in report['evidence']['spikes'])
    jump = [0.] * 5 + [1.] * 6
    assert _continuity_sequence(jump, hidden=[4, 5])['verdict'] == 'pass'
    report = _continuity_sequence(np.zeros(11), deformation=jump, topology_change=range(5, 11))
    assert report['verdict'] == 'pass'
    assert report['evidence']['topology_changes'] == 1


def _native_verifier_fixture(tmp_path, *, hidden=False, deleted=False, events=(), moving=False):
    from dycraft.tests._scene_fixtures import _stack_scene
    from dycraft.animation.blender.verify import verify
    scene, catalog = _stack_scene()
    mat = np.eye(4)
    baseline = {'identities': {'box/body': {'object': 'part/box/body', 'link': 'link/box__body',
                                           'local_matrix': mat.tolist(), 'world_matrix': mat.tolist()}},
                'joints': {}, 'fixed_edges': []}
    frames = 11 if moving else 1
    (tmp_path / 'index.json').write_text(json.dumps({'fps': 30., 'n_frames': frames,
                                                   'scene_ref': scene['id'], 'declarations': {'events': list(events)}}))
    for i in range(frames):
        mat[0, 3] = 2. + (1. if moving and i >= 5 else 0.)
        objects = [] if deleted else [{'name': 'renamed_part', 'identity': 'box/body', 'key': 'o0',
                                      'matrix': mat.tolist(), 'hidden': hidden}]
        record = {'objects': objects, 'transforms': {}, 'frame': i}
        (tmp_path / f'{i:06d}.json').write_text(json.dumps(record))
        np.savez(tmp_path / f'{i:06d}.npz', o0_v=np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]]),
                 o0_t=np.array([[0, 1, 2]]))
    return verify, baseline, scene, catalog


@_needs_blender
@pytest.mark.parametrize('hidden,deleted', [(False, False), (True, False), (False, True)])
def test_native_accepts_changed_initial_pose_renamed_hidden_or_deleted_parts(tmp_path, hidden, deleted):
    verify, baseline, scene, catalog = _native_verifier_fixture(tmp_path, hidden=hidden, deleted=deleted)
    reports = verify(tmp_path, baseline, scene, catalog)
    # One frame of a scene with no joints, events, or weighed bodies: what has
    # nothing to check says so, and nothing fails.
    assert {r['name']: r['verdict'] for r in reports} == {
        'AnimationWellFormed': 'pass', 'NoPenetration': 'pass',
        'JointLimits': 'not_applicable', 'AnatomicalJointLimits': 'not_applicable',
        'EventsSatisfied': 'not_applicable', 'MotionContinuity': 'not_applicable',
        'SupportMargin': 'not_applicable', 'ContactSlip': 'not_applicable'}


@_needs_blender
def test_missing_event_target_is_reported_after_authorized_deletion(tmp_path):
    event = {'name': 'required contact', 'predicate': 'touch', 'a': 'box/body', 'b': 'table/body', 'window': [0, 0]}
    verify, baseline, scene, catalog = _native_verifier_fixture(tmp_path, deleted=True, events=[event])
    report = next(r for r in verify(tmp_path, baseline, scene, catalog) if r['name'] == 'EventsSatisfied')
    assert report['verdict'] == 'fail'
    # The deleted target is named once per distinct reference set, not once
    # per frame that could not find it.
    assert report['evidence']['events'][0]['missing_refs'] == [['box/body', 'table/body']]
    assert report['evidence']['events'][0]['unsatisfied'] == [[0, 0]]


@_needs_blender
def test_fragment_continuity_has_only_internal_transitions_and_is_nonblocking(tmp_path):
    from dycraft.agent.layers.session import _slim_compile_payload_for_llm
    verify, baseline, scene, catalog = _native_verifier_fixture(tmp_path, moving=True)
    reports = verify(tmp_path, baseline, scene, catalog)
    assert all(r['verdict'] != 'fail' for r in reports)
    slim = _slim_compile_payload_for_llm({'verifier_reports': reports})
    assert any(r['name'] == 'MotionContinuity' and r['verdict'] == 'uncertain' for r in slim['verifier_reports'])
    fragment = verify(tmp_path, baseline, scene, catalog, start=5, end=10)
    assert next(r for r in fragment if r['name'] == 'MotionContinuity')['verdict'] == 'pass'


@_needs_blender
def test_relaxed_visual_rules_do_not_remove_mechanism_checks(tmp_path):
    verify, baseline, scene, catalog = _native_verifier_fixture(tmp_path, deleted=True)
    baseline['fixed_edges'] = [{'parent': 'parent', 'child': 'child', 'matrix': np.eye(4).tolist()}]
    assert next(r for r in verify(tmp_path, baseline, scene, catalog) if r['name'] == 'JointLimits')['verdict'] == 'fail'


def _joint_record(degrees, limits, hint=None):
    import math
    from scipy.spatial.transform import Rotation
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler('z', degrees, degrees=True).as_matrix()
    record = {'frame': 0, 'objects': [], 'transforms': {'origin': np.eye(4).tolist(), 'joint': m.tolist()}}
    if hint is not None:
        record['joint_positions'] = {'joint': math.radians(hint)}
    joint = {'origin': 'origin', 'object': 'joint', 'axis': [0,0,1],
             'type': 'revolute', 'limits': list(map(math.radians, limits))}
    return record, joint


@pytest.mark.parametrize('angle,limits,hint', [(270,(0,360),None),(-270,(-360,0),None),
    (630,(0,720),630),(270,(0,360),0),(180,(0,360),None),(-180,(-360,0),None)])
def test_joint_position_preserves_turns(angle, limits, hint):
    import math
    from dycraft.animation.blender.verify import joint_position
    record, joint = _joint_record(angle, limits, hint)
    assert joint_position(record, joint) == pytest.approx(math.radians(angle))


def test_joint_position_rejects_ambiguity_and_wrong_axis():
    from dycraft.animation.blender.verify import joint_position
    record, joint = _joint_record(270, (0,720))
    with pytest.raises(ValueError, match='ambiguous'):
        joint_position(record, joint)
    record, joint = _joint_record(270, (0,360), 270)
    record['transforms']['joint'][0][3] = .1
    with pytest.raises(ValueError, match='axis/frame'):
        joint_position(record, joint)


def test_joint_turn_count_changes_replay_signature():
    from dycraft.animation.blender.evaluation import frame_signature
    a, _ = _joint_record(0, (0,720), 0)
    b = {**a, 'joint_positions': {'joint': 2*np.pi}}
    assert frame_signature(a, {}) != frame_signature(b, {})


def test_a_signed_zero_signs_like_a_zero(tmp_path):
    """A jump keeps a stored -0.0 where stepping writes +0.0, which alone would
    refuse a delivery render's replay. The record a frame is stored as, and so
    what replay compares, must not depend on it."""
    from dycraft.animation.blender.evaluation import (
        check_replay_frame, frame_signature, read_record, write_record)
    walked, _ = _joint_record(0, (0, 720), 0)
    jumped = {**walked, 'joint_positions': {'joint': -0.0},
              'transforms': {name: [[-0.0 if v == 0 else v for v in row] for row in matrix]
                             for name, matrix in walked['transforms'].items()}}
    assert np.signbit(jumped['joint_positions']['joint'])
    assert frame_signature(jumped, {}) == frame_signature(walked, {})
    write_record(tmp_path, {**jumped, 'signature': frame_signature(jumped, {})})
    stored = read_record(tmp_path, 0)
    assert not np.signbit(stored['joint_positions']['joint'])
    check_replay_frame(tmp_path, 0, walked, {})

    # A real difference is refused, naming the field that differed and no bake.
    turned = {**walked, 'joint_positions': {'joint': .5}}
    with pytest.raises(ValueError) as refused:
        check_replay_frame(tmp_path, 0, turned, {})
    assert "joints: ['joint']" in str(refused.value)
    assert 'bake' not in str(refused.value) and 'transforms' not in str(refused.value)
    with pytest.raises(ValueError, match='joint carry simulation caches: bake them'):
        check_replay_frame(tmp_path, 0, turned, {}, simulated={'joint'})


@pytest.mark.expensive
def test_joint_events_limits_and_allowances_share_unwrapped_coordinate(tmp_path):
    import math
    from dycraft.animation.blender.verify import verify, joint_position
    from dycraft.verify._inherited_allowances import InheritedAllowances
    record, joint = _joint_record(270, (0,360), 270)
    info = {'fps': 1, 'n_frames': 1, 'scene_ref': 'scene', 'declarations': {'events': [
        {'name': 'turn', 'predicate': 'joint_at', 'joint': 'robot.hinge',
         'value': math.radians(270), 'window': [0,0]}]}}
    (tmp_path/'index.json').write_text(json.dumps(info))
    (tmp_path/'000000.json').write_text(json.dumps(record))
    np.savez(tmp_path/'000000.npz')
    scene = {'id': 'scene', 'instances': [{'name':'robot', 'object_ref':'robot'}]}
    obj = {'joints': [{'name':'hinge','type':'revolute','default':0}],
           'collision_allowances': [{'a':'body','b':'door','joint':'hinge','q_range':[260,280]}]}
    baseline = {'identities':{},'joints':{'robot__hinge':joint}}
    reports = {r['name']:r for r in verify(tmp_path, baseline, scene, {'robot':obj})}
    assert reports['JointLimits']['verdict'] == 'pass'
    assert reports['EventsSatisfied']['verdict'] == 'pass'
    poses = {'robot.hinge': joint_position(record,joint)}
    assert InheritedAllowances(scene, {'robot':obj}).allows('robot','body','door',poses)
    # Same orientation outside the allowed revolution range must NOT be wrapped in.
    record['joint_positions']['joint'] = math.radians(630)
    (tmp_path/'000000.json').write_text(json.dumps(record))
    reports = {r['name']:r for r in verify(tmp_path, baseline, scene, {'robot':obj})}
    assert reports['JointLimits']['verdict'] == 'fail'
    assert reports['EventsSatisfied']['verdict'] == 'fail'
    assert not InheritedAllowances(scene, {'robot':obj}).allows(
        'robot','body','door',{'robot.hinge':joint_position(record,joint)})


@pytest.mark.expensive
def test_seek_preserves_preroll_for_physics_and_nested_simulation_nodes():
    import subprocess
    from dycraft.render._blender_env import blender_python
    interpreter = blender_python()
    if not interpreter:
        pytest.skip("Blender interpreter unavailable")
    script = r"""
import bpy
import importlib.util
spec = importlib.util.spec_from_file_location('seek_worker', WORKER_PATH)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
info = {'frame_start': s.frame_start}
steps = []
def record(scene, *_):
    steps.append(scene.frame_current)
# Left registered, a Python handler hangs the bpy module's exit.
bpy.app.handlers.frame_change_pre.append(record)
try:
    def visited(frames):
        steps.clear()
        assert list(worker.visit_frames(frames, info)) == sorted(set(frames))
        return list(steps)
    JUMP, WALK = [1, 31, 32], list(range(1, 33))
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    assert visited([31, 30, 31]) == JUMP
    assert visited([0, 1]) == [1, 2]
    for kind in ('CLOTH','SOFT_BODY','FLUID','DYNAMIC_PAINT','PARTICLE_SYSTEM'):
        modifier = obj.modifiers.new('stateful', kind)
        assert visited([30, 31]) == WALK, kind
        obj.modifiers.remove(modifier)
    bpy.ops.rigidbody.world_add()
    assert visited([30, 31]) == WALK
    bpy.ops.rigidbody.world_remove()
    outer = bpy.data.node_groups.new('outer', 'GeometryNodeTree')
    inner = bpy.data.node_groups.new('inner', 'GeometryNodeTree')
    outer.nodes.new('GeometryNodeGroup').node_tree = inner
    modifier = obj.modifiers.new('nodes','NODES'); modifier.node_group = outer
    assert visited([30, 31]) == JUMP
    inner.nodes.new('GeometryNodeSimulationOutput')
    assert visited([30, 31]) == WALK
finally:
    bpy.app.handlers.frame_change_pre.remove(record)
"""
    script = script.replace('WORKER_PATH', repr(str(Path(__file__).resolve().parents[1] / 'animation/blender/worker.py')))
    result = subprocess.run([interpreter, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def _interval(a, b, start, end, *, bound, directional=None, kind="surface_crossing"):
    entry = {"a": a, "b": b, "kind": kind, "start_frame": start, "end_frame": end,
             "representative_frame": start, "separation_upper_bound_m": bound,
             "separation_direction": [1.0, 0.0, 0.0]}
    if directional is not None:
        entry["directional_separation_m"] = directional
    return entry


def test_frame_runs_compresses_dense_indices_and_keeps_the_gaps():
    assert frame_runs([]) == []
    assert frame_runs([4]) == [[4, 4]]
    assert frame_runs(range(0, 5)) == [[0, 4]]
    assert frame_runs([0, 1, 2, 7, 8, 11]) == [[0, 2], [7, 8], [11, 11]]


def test_penetration_pairs_report_the_measured_distance_and_its_direction():
    """The box bound seeds the search and does not go out; the winning
    direction does, because moving `b` along it by `penetration_m` is what the
    bisection verified.

    The bound is an axis-aligned *bounding-box* overlap, which on slender
    diagonal parts ran to hundreds of times the measured distance and turned a
    grazing touch into a reported deep overlap.
    """
    pairs = penetration_pairs([_interval("a", "b", 3, 3, bound=.151, directional=.017)])
    span = pairs[0]["spans"][0]
    assert span["separating_shift_m"] == pytest.approx(.017)
    assert pairs[0]["worst_separating_shift_m"] == pytest.approx(.017)
    assert span["separating_direction"] == [1., 0., 0.]
    assert .151 not in [v for v in span.values() if isinstance(v, float)]
    # A search that came apart along no candidate direction reports the
    # overlap without inventing a size for it — and names no direction,
    # because none of them separated.
    stuck = penetration_pairs([_interval("a", "b", 3, 3, bound=.151)])
    assert stuck[0]["spans"][0]["penetration_unresolved"] is True
    assert "separating_shift_m" not in stuck[0]["spans"][0]
    assert "separating_direction" not in stuck[0]["spans"][0]
    assert "worst_separating_shift_m" not in stuck[0]


def test_penetration_pairs_group_by_pair_and_sort_by_time_in_contact():
    pairs = penetration_pairs([
        _interval("a", "b", 10, 10, bound=.9, directional=.5),
        _interval("c", "d", 0, 3, bound=.4, directional=.1),
        _interval("a", "b", 20, 22, bound=.6, directional=.2),
    ])
    assert [(p["a"], p["b"], p["frame_count"]) for p in pairs] == [
        ("a", "b", 4), ("c", "d", 4)]
    assert [s["frames"] for s in pairs[0]["spans"]] == [[10, 10], [20, 22]]
    assert pairs[0]["worst_separating_shift_m"] == pytest.approx(.5)


def test_out_of_room_names_each_object_once_with_how_far_it_reached():
    """The box that decided it stays internal; the overshoot is the finding."""
    rows = out_of_room([
        {"frame": 1, "object": "cart", "reason": "outside room",
         "aabb": [[0., 0., 0.], [1., 1., 1.]]},
        {"frame": 2, "object": "cart", "reason": "outside room",
         "aabb": [[-1., 0., 0.], [1., 3., 1.]]},
    ], [[0., 0., 0.], [2., 2., 2.]])
    assert len(rows) == 1
    assert rows[0]["frames"] == [[1, 2]] and rows[0]["frame_count"] == 2
    assert rows[0]["outside_by_m"] == pytest.approx(1.)
    assert "aabb" not in rows[0]


def test_report_can_keep_the_verdict_without_shipping_the_failure_list():
    """`NoPenetration` carries the same items, merged, in its own evidence."""
    kept = report("X", [{"frame": 1}], detail=[])
    assert kept["verdict"] == "fail" and kept["evidence"]["failures"] == [{"frame": 1}]
    dropped = report("X", [{"frame": 1}], list_failures=False, detail=[])
    assert dropped["verdict"] == "fail" and "failures" not in dropped["evidence"]
    assert dropped["evidence"]["failure_count"] == 1
