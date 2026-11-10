"""Every verifier, measured in chunks, must report what one process reports.

Runs the real `verify.measure` -- collision, distance, and support queries
included -- in the Blender interpreter, over a synthetic 40-frame evaluation
built to put each frame-spanning question across a chunk boundary:

* an articulated robot whose foot creeps from frame 5, so `ContactSlip`,
  `SupportMargin`, and a `planted` event all anchor before the boundary;
* a `supported` event whose body leaves its support;
* an `any` touch that holds only inside a later chunk;
* a hand passing through a crate across the boundary, so one overlap span is
  measured in pieces;
* a spike next to a boundary for `MotionContinuity`.

The film is checked whole, in two chunks, and in four, exactly as
`check_in_parallel` plans them, and the reports must be equal apart from the
per-process counters.
"""
import json
import subprocess

import pytest

from dycraft.render._blender_env import blender_python

SCRIPT = r'''
import bpy, json, sys
from pathlib import Path
import numpy as np
from dycraft.animation.blender.evaluation import FORMAT, frame_signature, write_mesh, write_record
from dycraft.animation.blender.verify import aggregate, fill_advisories, measure
from dycraft.animation.blender.chunking import merge_rows, plan_chunks

def merge_reports(bundles, fill=None):
    rows = merge_rows(bundles)
    if fill is not None:
        fill(rows)
    return aggregate(rows)

out = Path(sys.argv[1])
N, FPS = 40, 24.0
CUBE = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                 [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)

def box(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])], dtype=np.float32)

SHAPES = {'floor': box([-2, -2, -.1], [2, 2, 0]),
          'robot/foot': box([0, 0, 0], [.2, .1, .05]),
          'robot/body': box([0, 0, .05], [.2, .1, .5]),
          'robot/hand': box([0, 0, 0], [.05, .05, .05]),
          'robot/arm': box([0, 0, 0], [.05, .05, .3]),
          'crate/body': box([1, 0, 0], [1.3, .3, .3]),
          'ball/body': box([1.1, .1, 0], [1.2, .2, .1]),
          # A weighed instance that rests until frame 30: it moves in the last
          # chunk only, so every other chunk lacks its advisory rows.
          'lever/body': box([-1, -1, 0], [-.8, -.9, .1])}

def place(name, f):
    if name == 'robot/foot':
        return [.0005 * max(f - 5, 0), 0, .03 if f < 5 else 0]
    if name == 'robot/body':
        return [.0005 * max(f - 5, 0) + .01 * f, 0, .03 if f < 5 else 0]
    if name == 'robot/hand':
        return [.6 + .02 * f, .1, .1]
    if name == 'robot/arm':
        return [.001 * f + (.5 if f == 21 else 0), .5, 1]
    if name == 'ball/body':
        return [0, 0, .3005 if 25 <= f <= 30 else .4]
    if name == 'lever/body':
        return [.004 * max(f - 30, 0), 0, 0]
    return [0, 0, 0]

known, tables = set(), set()
for f in range(N):
    objects, transforms, local = [], {}, {}
    for i, (name, vertices) in enumerate(SHAPES.items()):
        matrix = np.eye(4)
        matrix[:3, 3] = place(name, f)
        matrix = matrix.tolist()
        local[name] = (vertices, CUBE)
        objects.append({'key': f'o{i}', 'name': name, 'room': name == 'floor',
                        'identity': None if name == 'floor' else name, 'hidden': False,
                        'color': [.5, .5, .5, 1], 'matrix': matrix,
                        'mesh': write_mesh(out, vertices, CUBE, known)})
        transforms[name] = matrix
    record = {'frame': f, 'objects': objects, 'transforms': transforms, 'joint_positions': {}}
    record['signature'] = frame_signature(record, local)
    write_record(out, record, tables)

events = [{'name': 'stand', 'predicate': 'planted', 'a': 'robot/foot', 'window': [5, 35],
           'tolerance': .005},
          {'name': 'balanced', 'predicate': 'supported', 'a': 'robot', 'window': [0, 39]},
          {'name': 'rests', 'predicate': 'touch', 'mode': 'any', 'a': 'ball/body',
           'b': 'crate/body', 'window': [0, 39]},
          {'name': 'clear', 'predicate': 'separated', 'a': 'robot/hand', 'b': 'crate/body',
           'distance': .01, 'window': [0, 39]}]
info = {'format': FORMAT, 'complete': True, 'fps': FPS, 'n_frames': N, 'frame_start': 1,
        'frames': list(range(N)), 'scene_ref': 'scene', 'clips': {},
        'declarations': {'units': 'frames', 'events': events}}
(out / 'index.json').write_text(json.dumps(info))

def part(name, lo, hi, volume):
    return {'name': name, 'params': {'shape': {'volume': volume,
                                               'center': [(a + b) / 2 for a, b in zip(lo, hi)]},
                                     'aabb_min': lo, 'aabb_max': hi}}
catalog = {'robot': {'joints': [{'name': 'ankle', 'type': 'revolute', 'default': 0}],
                     'parts': [part('foot', [0, 0, 0], [.2, .1, .05], .001),
                               part('body', [0, 0, .05], [.2, .1, .5], .01),
                               part('hand', [0, 0, 0], [.05, .05, .05], .0001),
                               part('arm', [0, 0, 0], [.05, .05, .3], .0007)]},
           'crate': {'parts': [part('body', [1, 0, 0], [1.3, .3, .3], .027)]},
           'ball': {'parts': [part('body', [1.1, .1, 0], [1.2, .2, .1], .001)]},
           'lever': {'joints': [{'name': 'pin', 'type': 'fixed'}],
                     'parts': [part('body', [-1, -1, 0], [-.8, -.9, .1], .002)]}}
scene = {'id': 'scene', 'instances': [{'name': n, 'object_ref': n} for n in catalog]}
baseline = {'identities': {}, 'joints': {}, 'room_objects': ['floor']}

def check(frames=None, owned=None):
    return measure(out, baseline, scene, catalog, frames=frames, owned=owned)

result = {'single': aggregate(check())}
fill = lambda rows: fill_advisories(rows, out, baseline, scene, catalog)
for workers in (2, 4):
    plan = plan_chunks(range(N), workers)
    result[f'chunks{workers}'] = [[lo, hi] for lo, hi, _ in plan]
    chunks = [check(frames, [lo, hi]) for lo, hi, frames in plan]
    try:
        merge_reports(chunks)
        result[f'unfilled{workers}'] = 'accepted'
    except ValueError as error:
        result[f'unfilled{workers}'] = str(error)
    result[f'merged{workers}'] = merge_reports(chunks, fill=fill)
# Overlap diagnostics come back from the pair store on a second check.
from dycraft.animation.blender import collision as C, verify as V
store = str(out.parent / 'pairs.sqlite')
first = aggregate(measure(out, baseline, scene, catalog, pair_store=store))
real = C.directional_separation, V.deepest_vertex_m
def refuse(*args, **kwargs):
    raise AssertionError('recomputed a stored diagnostic')
C.directional_separation = V.deepest_vertex_m = refuse
try:
    result['stored_diagnostics'] = [first, aggregate(measure(out, baseline, scene, catalog, pair_store=store))]
finally:
    C.directional_separation, V.deepest_vertex_m = real
# Declared liquids: the crate, here, as an instance name.
named = {**baseline, 'identities': {'crate/body': {'link': '-'}, 'robot/hand': {'link': '-'}}}
for label, liquids in (('liquid', ['crate']), ('bad_liquid', ['vat'])):
    info['declarations']['liquids'] = liquids
    (out / 'index.json').write_text(json.dumps(info))
    result[label] = aggregate(measure(out, named, scene, catalog))
Path(sys.argv[2]).write_text(json.dumps(result))
'''


@pytest.mark.skipif(blender_python() is None, reason="needs a configured Blender interpreter")
def test_every_verifier_reports_the_same_chunked_as_whole(tmp_path, pkgroot, sibling_test_module):
    _coal_env = sibling_test_module("test_blender_collision")._coal_env
    evaluation, output = tmp_path / "evaluation", tmp_path / "result.json"
    evaluation.mkdir()
    script = tmp_path / "rows.py"
    script.write_text("import sys\nsys.path.insert(0, " + repr(str(pkgroot)) + ")\n" + SCRIPT)
    done = subprocess.run([blender_python(), str(script), str(evaluation), str(output)],
                          capture_output=True, text=True, timeout=300, env=_coal_env())
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(output.read_text())
    assert result["chunks2"] == [[0, 19], [20, 39]]
    assert result["chunks4"] == [[0, 9], [10, 19], [20, 29], [30, 39]]

    def comparable(reports):
        return [{**r, "evidence": {k: v for k, v in r["evidence"].items() if k != "stats"}}
                for r in reports]
    single = comparable(result["single"])
    assert comparable(result["merged2"]) == single
    assert comparable(result["merged4"]) == single

    # The fixture exercises what it claims to, so the equality means something.
    reports = {r["name"]: r for r in result["single"]}
    support, slip = reports["SupportMargin"], reports["ContactSlip"]
    assert support["evidence"]["considered"] == ["lever", "robot"]
    assert support["evidence"]["measured"] == ["lever", "robot"]
    # The late mover is reported over the whole film, and a merge without the
    # chunks' missing rows is refused rather than read as a gap.
    assert "advisory rows are missing" in result["unfilled2"]
    assert "advisory rows are missing" in result["unfilled4"]
    foot = {row["part"]: row for row in slip["evidence"]["parts"]}["robot/foot"]
    assert foot["max_slip_m"] == pytest.approx(.0005 * 34, abs=1e-6) and foot["bearing_intervals"] == 1
    assert foot["max_clearance_m"] == pytest.approx(.03, abs=1e-6)
    events = {row["name"]: row for row in reports["EventsSatisfied"]["evidence"]["events"]}
    # Anchored at frame 5, where the foot landed: slip passes 5 mm after frame 15.
    assert events["stand"]["ok"] is False and events["stand"]["unsatisfied"] == [[16, 35]]
    assert events["rests"]["ok"] is True and events["rests"]["satisfied_at"] == pytest.approx(25 / 24, abs=1e-4)
    # The gap the verdict turns on -- an `any` touch, so its closest approach.
    assert events["rests"]["worst_gap_m"] == pytest.approx(.0005, abs=1e-5)
    assert 25 <= events["rests"]["worst_gap_frame"] <= 30
    assert events["clear"]["worst_gap_m"] == 0.0 and "worst_margin_m" in events["balanced"]
    assert events["balanced"]["ok"] is False
    assert events["clear"]["ok"] is False
    pair, = reports["NoPenetration"]["evidence"]["pairs"]
    span, = pair["spans"]
    assert (pair["a"], pair["b"]) == ("robot/hand", "crate/body") or (pair["b"], pair["a"]) == ("robot/hand", "crate/body")
    assert span["frames"][0] < 20 < span["frames"][1] and "separating_shift_m" in span
    first, again = (comparable(r) for r in result["stored_diagnostics"])
    assert first == again == single, "stored diagnostics must be the measured ones"
    spikes = reports["MotionContinuity"]["evidence"]["spikes"]
    assert any(row["object"] == "robot/arm" and row["frames"][0][0] <= 21 for row in spikes)


@pytest.mark.skipif(blender_python() is None, reason="needs a configured Blender interpreter")
def test_a_declared_liquid_is_exempt_from_penetration_and_kept_in_events(tmp_path, pkgroot, sibling_test_module):
    """Liquids are declared, and exempt from penetration checks. The fixture's hand passes through the crate; declared
    liquid, the crate is never a candidate pair, the report names it, and its
    events read as before."""
    _coal_env = sibling_test_module("test_blender_collision")._coal_env
    evaluation, output = tmp_path / "evaluation", tmp_path / "result.json"
    evaluation.mkdir()
    script = tmp_path / "rows.py"
    script.write_text("import sys\nsys.path.insert(0, " + repr(str(pkgroot)) + ")\n" + SCRIPT)
    done = subprocess.run([blender_python(), str(script), str(evaluation), str(output)],
                          capture_output=True, text=True, timeout=300, env=_coal_env())
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(output.read_text())
    solid = {r["name"]: r for r in result["single"]}
    liquid = {r["name"]: r for r in result["liquid"]}
    assert solid["NoPenetration"]["verdict"] == "fail"
    penetration = liquid["NoPenetration"]
    assert penetration["verdict"] == "pass" and penetration["evidence"]["liquids_exempt"] == ["crate"]
    assert "pairs" not in penetration["evidence"]
    events = lambda reports: {row["name"]: row["ok"] for row in reports["EventsSatisfied"]["evidence"]["events"]}
    assert events(liquid) == events(solid), "a liquid stays in events"
    # A liquid that names nothing is reported, and the object stays a solid.
    bad = {r["name"]: r for r in result["bad_liquid"]}["NoPenetration"]
    assert bad["verdict"] == "fail" and bad["evidence"]["invalid_liquids"][0]["reason"].startswith("liquid 'vat'")
    assert any("liquid declaration" in repair for repair in bad["suggested_repairs"])
