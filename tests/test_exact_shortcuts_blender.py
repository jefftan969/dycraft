"""Shortcuts in the checks must answer exactly as the slow way does.

Each test runs a shortcut beside the computation it replaces, on a scene built
to put the answers on a knife's edge.

* `collision.BearingResolver` against `bearing_map` below (the plain loop) with
  `gap_predicates`: the advisory checks' bearing tests. Rigid and skinned
  parts drift in and out of the 1 cm band over a finely meshed ground, turn,
  pass over a raised step, and two rigid parts move together.
* `collision._contains`' vectorized witness filters against a per-component
  loop, on many-component meshes such as a furred character's.
"""
import json
import subprocess

import pytest

from dycraft.render._blender_env import blender_python

SCRIPT = r'''
import bpy, json, math, sys
import numpy as np
from dycraft.animation.blender import collision
from dycraft.animation.blender.collision import BearingResolver, GeometryCache, gap_predicates
from dycraft.animation.blender.verify import ground_candidates_many, ground_index


def bearing_map(bodies, prefixes, body, gaps, contact_m=.01, index=None):
    """Reference for `BearingResolver`: `{body index: identity it bears on, or None}`.

    For each part of `prefixes`, the first candidate in `ground_candidates_many`
    order that bears, asked in rounds through `gaps` (`collision.gap_predicates`):
    every part's first candidate, then the next one for parts still not bearing.
    """
    index = index if index is not None else ground_index(bodies)
    parts = [i for i, entry in enumerate(bodies) if entry[0].split("/")[0] + "/" in prefixes]
    bears = dict.fromkeys(parts)
    waiting = [(i, grounds) for i, grounds in
               zip(parts, ground_candidates_many(bodies, parts, contact_m, index)) if grounds]
    depth = 0
    while waiting:
        asked = [(i, grounds[depth]) for i, grounds in waiting]
        held = gaps([(body(i), body(g)) for i, g in asked], contact_m)
        for (i, g), bearing in zip(asked, held):
            if bearing:
                bears[i] = bodies[g][0]
        depth += 1
        waiting = [(i, grounds) for i, grounds in waiting
                   if bears[i] is None and depth < len(grounds)]
    return bears


rng = np.random.default_rng(3)
CUBE = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                 [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)

def box(size):
    sx, sy, sz = size
    return np.array([[x, y, z] for x in (0, sx) for y in (0, sy) for z in (0, sz)], dtype=float)

def grid(n, size, z=0.):
    xs = np.linspace(-size, size, n)
    v = np.array([[x, y, z] for x in xs for y in xs], dtype=float)
    t = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            t += [[a, b, d], [a, d, c]]
    return v, np.array(t, dtype=np.int32)

def rot(z_angle, x_angle):
    cz, sz, cx, sx = math.cos(z_angle), math.sin(z_angle), math.cos(x_angle), math.sin(x_angle)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ rx

floor_v, floor_t = grid(40, 2.)
step_v, step_t = grid(6, .3, z=.08)
step_v[:, 0] += 1.
cache = GeometryCache()

def frame_bodies(f):
    """`bodies` entries and placed bodies, as `measure` builds them, for frame `f`."""
    out = []
    def add(name, vertices, triangles, matrix):
        placed = cache.placed(vertices, triangles, matrix)
        world = vertices @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
        out.append(((name, vertices, triangles, world.min(axis=0), world.max(axis=0), matrix, True), placed))
    add('ground/floor', floor_v, floor_t, np.eye(4))
    add('ground/step', step_v, step_t, np.eye(4))
    for k in range(6):
        # Feet drifting through the band: heights cross 1 cm, and over the step.
        m = np.eye(4)
        m[:3, :3] = rot(.01 * f + k, .02 * math.sin(.3 * f + k))
        m[:3, 3] = [.7 + .004 * f + .05 * k, .1 * k - .2, .004 + .012 * (1 + math.sin(.13 * f + k)) / 2]
        add(f'walker/foot{k}', box([.1, .06, .03]), CUBE, m)
    # A skinned part: the same triangles, new vertices every frame.
    soft = box([.2, .2, .05]) + rng.normal(0, .002, (8, 3))
    soft[:, 2] += .006 + .006 * math.sin(.21 * f)
    soft[:, 0] += .003 * f - .3
    add('walker/coat', soft, CUBE, np.eye(4))
    # Two rigid parts moving together, as a newspaper in a dog's jaw: the
    # jaw swings, the paper rides on it a few millimetres up.
    hold = np.eye(4); hold[:3, :3] = rot(.2 * f, .1 * math.sin(.2 * f)); hold[:3, 3] = [.5, -.5, .3 + .05 * math.sin(.1 * f)]
    add('walker/jaw', box([.1, .1, .02]), CUBE, hold)
    held = hold.copy(); held[:3, 3] += hold[:3, :3] @ np.array([.01, .01, .02 + .006 + .006 * math.sin(.4 * f)])
    add('news/paper', box([.08, .08, .01]), CUBE, held)
    return out

resolver = BearingResolver()
mismatches, tested = [], 0
for f in range(60):
    entries = frame_bodies(f)
    bodies = [e for e, _ in entries]
    placed = [p for _, p in entries]
    body = lambda i: placed[i]
    gindex = ground_index(bodies)
    prefixes = {'walker/', 'news/'}
    parts = [i for i, e in enumerate(bodies) if e[0].split('/')[0] + '/' in prefixes]
    expected = bearing_map(bodies, prefixes, body, gap_predicates, .01, gindex)
    got = resolver.resolve(parts, ground_candidates_many(bodies, parts, .01, gindex), bodies, body, .01, f)
    tested += len(parts)
    if got != expected:
        mismatches.append([f, {bodies[i][0]: (expected[i], got[i]) for i in parts if expected[i] != got[i]}])
print("RESULT " + json.dumps({"mismatches": mismatches[:5], "tested": tested,
                              "counts": resolver.counts}))
'''


@pytest.mark.skipif(blender_python() is None, reason="needs a configured Blender interpreter")
def test_the_resolver_answers_every_bearing_test_as_the_exact_one_does(tmp_path, pkgroot, sibling_test_module):
    _coal_env = sibling_test_module("test_blender_collision")._coal_env
    script = tmp_path / "resolver.py"
    script.write_text("import sys\nsys.path.insert(0, " + repr(str(pkgroot)) + ")\n" + SCRIPT)
    done = subprocess.run([blender_python(), str(script)], capture_output=True, text=True,
                          timeout=600, env=_coal_env())
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    line = next(l for l in done.stdout.splitlines() if l.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    assert result["mismatches"] == [], result
    counts = result["counts"]
    # The scene exercises every path, so the equality means something.
    assert counts["carried"] > 0 and counts["measured"] > 0 and counts["exact"] > 0, counts


CONTAINS_SCRIPT = r'''
import bpy, json, sys, time
import numpy as np
from dycraft.animation.blender import collision
from dycraft.animation.blender.collision import Body, PlacedBody, TOLERANCE

def reference(a, b, enclosed_only=False):
    """`_contains` as it was: every component tested one at a time."""
    for source, target in ((a, b), (b, a)):
        multi = len(source.components) > 1
        for component, (indices, closed) in enumerate(source.components):
            face = source.vertices[source.triangles[indices[0]]]
            if enclosed_only and (np.any(face < target.lo) or np.any(face > target.hi)):
                continue
            lo, hi = source.component_bounds[component]
            pad = 4*TOLERANCE if closed else 0.
            if multi and (np.any(hi + pad < target.lo) or np.any(lo - pad > target.hi)):
                continue
            if enclosed_only:
                if np.any(lo < target.lo) or np.any(hi > target.hi):
                    continue
            centroid = face.mean(axis=0)
            if target.inside(face[0]):
                return True
            inside, distance = target.inside_at(centroid)
            if inside:
                return True
            if closed and distance <= 5*TOLERANCE:
                normal = np.cross(face[1]-face[0], face[2]-face[0])
                length = np.linalg.norm(normal)
                if length:
                    for sign in (-1, 1):
                        point = centroid + sign*normal/length*(4*TOLERANCE)
                        if target.inside(point) and source.inside(point):
                            return True
    return False

CUBE = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                 [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
def cube(lo, size):
    return np.array([[x, y, z] for x in (lo[0], lo[0] + size) for y in (lo[1], lo[1] + size)
                     for z in (lo[2], lo[2] + size)], dtype=float)

rng = np.random.default_rng(11)
# "Fur": 300 open cards (one triangle each) and 40 closed cubes, scattered
# across and around a closed target, some exactly on its faces.
verts, tris = [], []
for k in range(300):
    base = rng.uniform(-.1, 1.1, 3)
    if k % 25 == 0:
        base[2] = 1.0                                  # on the target's top face
    tri = base + rng.normal(0, .01, (3, 3))
    tris.append([len(verts), len(verts) + 1, len(verts) + 2]); verts.extend(tri)
for k in range(40):
    lo = rng.uniform(-.2, 1.1, 3)
    tris.extend((CUBE + len(verts)).tolist()); verts.extend(cube(lo, rng.uniform(.01, .08)))
fur = Body(np.array(verts), np.array(tris, dtype=np.int32))
shell = Body(cube([0, 0, 0], 1.), CUBE)
assert len(fur.components) == 340
results, t_new, t_old = [], 0., 0.
for trial in range(60):
    m = np.eye(4)
    angle = rng.uniform(0, 6.3)
    m[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    m[:3, 3] = rng.uniform(-.6, .6, 3)
    a, b = PlacedBody(fur, m), PlacedBody(shell, np.eye(4))
    for enclosed in (False, True):
        t = time.process_time(); old = reference(a, b, enclosed); t_old += time.process_time() - t
        a, b = PlacedBody(fur, m), PlacedBody(shell, np.eye(4))
        t = time.process_time(); new = collision._contains(a, b, enclosed_only=enclosed); t_new += time.process_time() - t
        results.append([old, new])

# The witness filter's box test against its dense form: on faces, at the
# reach and either side of it, NaN in a box or a point, and chunked.
from dycraft.animation.blender.collision import _within_boxes
def dense(points, lo, hi, reach):
    g = np.maximum(np.maximum(lo[None] - points[:, None], points[:, None] - hi[None]), 0.)
    return np.sqrt((g ** 2).sum(axis=2)).min(axis=1) <= reach
within = []
for trial in range(301):
    lo = rng.uniform(-1, 1, (int(rng.integers(1, 60)), 3))
    hi = lo + rng.uniform(0, .3, lo.shape)
    points = rng.uniform(-1.5, 1.5, (40000 if trial == 300 else int(rng.integers(1, 400)), 3))
    for reach in (0., 6*TOLERANCE):
        near = points.copy()
        k = rng.integers(0, len(lo), len(near) // 2)
        step = reach * rng.choice([0., 1., 1 - 1e-9, 1 + 1e-9], len(k)) + rng.choice([0., 1e-9], len(k))
        near[:len(k)] = hi[k] + step[:, None] * np.eye(3)[rng.integers(0, 3, len(k))]
        if trial % 50 == 1:
            near[0, 1] = np.nan
        if trial % 50 == 2:
            lo, hi = lo.copy(), hi.copy()
            lo[0, 0] = np.nan
        got, want = _within_boxes(near, lo, hi, reach), dense(near, lo, hi, reach)
        within.append([bool(np.array_equal(got, want)), int(want.sum()), int(len(want))])
# Clustered points, as fur is against a forest: most boxes are dropped, and
# the ones kept include boxes the cluster's extreme points reach exactly.
for trial in range(200):
    points = rng.uniform(0, .1, (int(rng.integers(1, 300)), 3))
    points[0] = .1
    for reach in (0., 6*TOLERANCE):
        far = rng.uniform(-3, 3, (40, 3))
        gaps = reach * rng.choice([0., .5, 1., 1 - 1e-9, 1 + 1e-9, 2., 3.], 20) + rng.choice([0., 1e-9, 2e-9], 20)
        axis = rng.integers(0, 3, 20)
        lo = np.full((20, 3), -1.)
        lo[np.arange(20), axis] = .1 + gaps
        hi = lo + 2.
        lo, hi = np.concatenate([far, lo]), np.concatenate([far + .05, hi])
        got, want = _within_boxes(points, lo, hi, reach), dense(points, lo, hi, reach)
        within.append([bool(np.array_equal(got, want)), int(want.sum()), int(len(want))])
print("RESULT " + json.dumps({"results": results, "old_s": t_old, "new_s": t_new, "within": within}))
'''


@pytest.mark.skipif(blender_python() is None, reason="needs a configured Blender interpreter")
def test_containment_tests_only_the_witnesses_that_can_be_inside_and_agrees(tmp_path, pkgroot, sibling_test_module):
    """The vectorized filters give exactly the answer of testing one witness
    per connected component, one at a time, for every component, which on fur
    is thousands."""
    _coal_env = sibling_test_module("test_blender_collision")._coal_env
    script = tmp_path / "contains.py"
    script.write_text("import sys\nsys.path.insert(0, " + repr(str(pkgroot)) + ")\n" + CONTAINS_SCRIPT)
    done = subprocess.run([blender_python(), str(script)], capture_output=True, text=True,
                          timeout=600, env=_coal_env())
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    line = next(l for l in done.stdout.splitlines() if l.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])
    pairs = result["results"]
    assert all(old == new for old, new in pairs), pairs
    assert any(old for old, _ in pairs) and not all(old for old, _ in pairs), "both answers occur"
    # The witness filter's box test: the dense answer, with both answers occurring.
    assert all(same for same, _, _ in result["within"])
    assert sum(hits for _, hits, _ in result["within"]) and sum(n - hits for _, hits, n in result["within"])
