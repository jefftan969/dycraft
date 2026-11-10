"""Shortcuts in the check that must change no answer, on the host.

Each is a cheaper way to the same value: the support hull's chain on Python
floats, a standing instance's support margin reused from its last frame, a
frame record decoded a whole array at a time (or, for an advisory fill, without
the transforms it never reads), and the span diagnostics reading a frame once
for every span that visits it. The Blender-side ones are in
`test_exact_shortcuts_blender.py`.
"""
import types

import numpy as np


def _hull_on_numpy_scalars(points):
    """`verify._hull_vertices` as it was: the chain on numpy scalars."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    index = np.arange(len(pts))
    if len(pts) > 32:
        quad = pts[[np.argmin(pts[:, 0]), np.argmin(pts[:, 1]),
                    np.argmax(pts[:, 0]), np.argmax(pts[:, 1])]]
        inside = np.ones(len(pts), dtype=bool)
        for a, b in zip(quad, np.roll(quad, -1, axis=0)):
            inside &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) > 1e-12
        if inside.any():
            index = np.flatnonzero(~inside)
    order = index[np.lexsort((pts[index, 1], pts[index, 0]))]
    unique = order[np.concatenate(([True], (np.diff(pts[order], axis=0) != 0).any(axis=1)))]
    if len(unique) < 3:
        return None

    def half(indices):
        chain = []
        for i in indices:
            while len(chain) >= 2:
                a, b, c = pts[chain[-2]], pts[chain[-1]], pts[i]
                if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0:
                    break
                chain.pop()
            chain.append(i)
        return chain

    lower, upper = half(unique), half(unique[::-1])
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=int) if len(hull) >= 3 else None


def test_the_support_hull_is_the_same_hull_on_python_floats():
    """Rings of soles, grids with duplicates, and collinear runs, lines, clusters."""
    from dycraft.animation.blender.verify_support import _hull_vertices

    rng = np.random.default_rng(0)
    ring = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for trial in range(1500):
        n = int(rng.integers(1, 300))
        kind = trial % 5
        if kind == 0:
            points = rng.normal(size=(n, 2))
        elif kind == 1:
            points = np.concatenate([np.c_[np.cos(ring), np.sin(ring)] * .03 + rng.uniform(-.5, .5, 2)
                                     for _ in range(int(rng.integers(1, 8)))])
        elif kind == 2:
            points = np.round(rng.uniform(-1, 1, (n, 2)), 1)
        elif kind == 3:
            points = np.c_[np.linspace(0, 1, n), np.linspace(0, 2, n)]
        else:
            points = np.repeat(rng.normal(size=(max(1, n // 7), 2)), 7, axis=0)
        expected, got = _hull_on_numpy_scalars(points), _hull_vertices(points)
        assert (expected is None) == (got is None), trial
        if expected is not None:
            assert got.dtype == expected.dtype and np.array_equal(got, expected), trial


def _feet(right_x=.3):
    """Two feet of `bot` on the floor and the floor, as `measure`'s body entries."""
    cube = np.array([[x, y, z] for x in (0, .1) for y in (0, .1) for z in (0, .1)], dtype=float)
    entries, records = [], []
    for k, (identity, x) in enumerate((("bot/left", 0.), ("bot/right", right_x), ("floor/slab", -1.))):
        matrix = np.eye(4)
        matrix[0, 3] = x
        verts = cube + matrix[:3, 3]
        entries.append((identity, verts, np.zeros((0, 3), dtype=int), verts.min(axis=0),
                        verts.max(axis=0), matrix.tolist(), False))
        records.append({"mesh": f"{k:064x}"})
    masses = {name: (1.0, np.array([.05, .05, .05]), np.zeros(3), np.full(3, .1))
              for name in ("bot/left", "bot/right")}
    return entries, records, masses


def test_a_standing_instance_reuses_its_support_margin_and_nothing_else_does(monkeypatch):
    from dycraft.animation.blender import verify_support

    measured = verify_support._supported_margin
    calls = []
    monkeypatch.setattr(verify_support, "_supported_margin", lambda *a, **k: calls.append(1) or measured(*a, **k))
    held, bearing = {}, {0: "floor/slab", 1: "floor/slab"}

    def margin(entries, records, masses, bearing):
        return verify_support._supported_margin_held(held, "bot", masses, entries, records, None, .01,
                                             None, bearing)

    first = margin(*_feet()[:2], _feet()[2], bearing)
    assert first[0] is not None and len(calls) == 1
    # The same inputs in new arrays: the last frame's result, not a new one.
    assert margin(*_feet()[:2], _feet()[2], bearing) is first and len(calls) == 1
    # A foot moved, or a foot lifted: measured again, and as a fresh measurement says.
    for entries, records, masses, bears in ((*_feet(.35), bearing), (*_feet(), {0: "floor/slab", 1: None})):
        got = margin(entries, records, masses, bears)
        assert got == measured("bot", masses, entries, records, None, None, .01, None, bears)
    assert len(calls) == 3
    # A bearing footprint without a mesh key has no key here either: never reused.
    entries, records, masses = _feet()
    records[1] = {}
    margin(entries, records, masses, bearing)
    margin(entries, records, masses, bearing)
    assert len(calls) == 5


def test_a_decoded_record_is_the_per_element_conversion_in_lists_of_its_own(tmp_path):
    from dycraft.animation.blender.evaluation import frame_signature, read_record, write_record

    def f4(matrix):
        return [[float(np.float32(v)) for v in row] for row in matrix]

    rng = np.random.default_rng(3)
    for placed_apart in (False, True):      # body matrices stored, or read from the transforms
        transforms = {name: rng.normal(size=(4, 4)).tolist() for name in ("a", "b", "c")}
        objects = [{"name": name, "identity": f"i/{name}", "mesh": f"{k:064x}",
                    "matrix": (np.asarray(transforms[name]) + (placed_apart and name == "b")).tolist(),
                    "color": rng.random(4).tolist(), "hidden": name == "c"}
                   for k, name in enumerate(transforms)]
        record = {"frame": 0, "objects": objects, "transforms": transforms,
                  "joint_positions": {"j": .25, "k": -1.5}}
        write_record(tmp_path, {**record, "signature": frame_signature(record, {})})
        stored = read_record(tmp_path, 0)
        assert stored["transforms"] == {name: f4(m) for name, m in transforms.items()}
        assert stored["joint_positions"] == {"j": .25, "k": -1.5}
        for obj, source in zip(stored["objects"], objects):
            assert obj["matrix"] == f4(source["matrix"])
            assert obj["color"] == [float(np.float32(v)) for v in source["color"]]
            assert obj["hidden"] is source["hidden"]
            assert all(type(v) is float for row in obj["matrix"] for v in row)
            # A body's matrix is not its transform's list: editing one leaves the other.
            obj["matrix"][0][0] = 99.
            assert stored["transforms"][obj["name"]][0][0] != 99.


def test_span_diagnostics_read_each_frame_once_and_measure_its_own_bodies(monkeypatch, tmp_path):
    """Two spans of `x` share frames 1, 3, and 4 among them; each frame is read once."""
    from dycraft.animation.blender import verify_penetration

    loads = []

    def load_frame(directory, frame):
        loads.append(frame)
        objects = [{"name": name, "identity": name, "matrix": np.eye(4).tolist()} for name in "xyz"]
        return {"frame": frame, "objects": objects}, {name: (frame, name) for name in "xyz"}

    cache = types.SimpleNamespace(get=lambda frame, name: types.SimpleNamespace(
        frame=frame, name=name, local=types.SimpleNamespace(key=name.encode()), matrix=np.eye(4) + frame),
        next_frame=lambda: None)
    monkeypatch.setattr(verify_penetration, "load_frame", load_frame)
    monkeypatch.setattr(verify_penetration, "deepest_vertex_m", lambda a, b: 10. * a.frame + "xyz".index(b.name))
    bounds = {"y": [1, 3, 2, 5, 4], "z": [5, 4, 3, 2, 1]}
    hits = [{"frame": frame, "a": "x", "b": other, "kind": "containment",
             "bound": float(bounds[other][frame]), "direction": [0., 0., 1.]}
            for frame in range(5) for other in "yz"]
    shifts, depths = [], []
    verify_penetration._measure_diagnostics(hits, shifts, depths, tmp_path, cache,
                                collision_tools=(lambda a, b, direction, bound: {"shift_m": a.frame + .5},))
    assert sorted(loads) == [0, 1, 2, 3, 4]
    assert [(row["b"], row["frame"], row["shift_m"]) for row in shifts] == [("y", 3, 3.5), ("z", 0, .5)]
    assert [(row["b"], row["frame"], row["depth"]) for row in depths] == [
        ("y", 3, 31.), ("y", 4, 41.), ("y", 1, 11.), ("z", 0, 2.), ("z", 1, 12.), ("z", 2, 22.)]


def test_a_fill_reads_the_record_without_transforms_and_a_changed_table_is_read_again(tmp_path):
    """The advisory fill places bodies only. Its record is the whole one less
    `transforms` and `joint_positions`, left out so a reader that needed them
    fails. A cached table is the file's: rewritten, it is read and checked anew."""
    import json
    import pytest
    from dycraft.animation.blender.evaluation import frame_signature, read_record, write_record

    transforms = {name: np.eye(4).tolist() for name in ("a", "b", "bone")}
    objects = [{"name": name, "identity": f"i/{name}", "mesh": f"{k:064x}",
                "matrix": transforms[name]} for k, name in enumerate(("a", "b"))]
    record = {"frame": 0, "objects": objects, "transforms": transforms, "joint_positions": {"j": .5},
              "note": "residual"}
    write_record(tmp_path, {**record, "signature": frame_signature(record, {})})
    whole, bodies_only = read_record(tmp_path, 0), read_record(tmp_path, 0, transforms=False)
    assert list(whole) == ["frame", "objects", "note", "transforms", "joint_positions", "signature"]
    assert bodies_only == {k: v for k, v in whole.items() if k not in ("transforms", "joint_positions")}
    assert list(bodies_only) == ["frame", "objects", "note", "signature"]

    table = next((tmp_path / "tables").glob("*.json"))
    altered = json.loads(table.read_text())
    altered["bodies"][0][1] = "i/other"
    table.write_text(json.dumps(altered) + " ")
    with pytest.raises(ValueError, match="does not match its table"):
        read_record(tmp_path, 0)


def test_a_standing_body_retakes_its_steps_and_continuity_reports_the_same():
    """Against the same class recomputing every frame (`_still` cleared)."""
    from dycraft.animation.blender.continuity import MotionContinuity

    rng = np.random.default_rng(7)
    cube = rng.normal(size=(40, 3))
    other = rng.normal(size=(30, 3))
    tris = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)

    def placed(x, angle=0.):
        m = np.eye(4)
        m[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        m[0, 3] = x
        return m

    def frame_of(f):
        poses = {
            "still": (placed(1.), "a" * 64, cube, False),
            "mover": (placed(.05 * f, .03 * f), "b" * 64, cube, False),
            "stopper": (placed(.1 * min(f, 10) + (.2 if f >= 30 else 0.)), "c" * 64, cube, False),
            "blinker": (placed(2.), "d" * 64, cube, f % 9 == 4),
            "morph": (placed(3.), ("e" if f < 20 else "f") * 64, cube if f < 20 else other, False),
            "spiker": (placed(4. + (1. if f == 17 else 0.)), "1" * 64, cube, False),
            "keyless": (placed(5.), None, cube, False),
        }
        objects, geometry = [], {}
        for name, (m, mesh, verts, hidden) in poses.items():
            obj = {"name": name, "matrix": m.tolist(), "hidden": hidden}
            if mesh:
                obj["mesh"] = mesh
            objects.append(obj)
            geometry[name] = (verts @ m[:3, :3].T + m[:3, 3], tris)
        return {"frame": f, "objects": objects}, geometry

    held, fresh = MotionContinuity(24.), MotionContinuity(24.)
    for f in range(40):
        record, geometry = frame_of(f)
        held.add(f, record, geometry)
        fresh.add(f, record, geometry)
        fresh._still.clear()
    assert any(entry[2] is not None for entry in held._still.values()), "the memo was used"
    assert dict(held.steps) == dict(fresh.steps)
    assert held.topology_changes == fresh.topology_changes
    assert held.last_vectors.keys() == fresh.last_vectors.keys()
    for key, (frame, vector) in fresh.last_vectors.items():
        assert held.last_vectors[key][0] == frame and np.array_equal(held.last_vectors[key][1], vector)
    assert held.displacements.keys() == fresh.displacements.keys()
    for key, value in fresh.displacements.items():
        assert all(np.array_equal(a, b) for a, b in zip(held.displacements[key], value))


def test_mechanism_checks_batched_answer_as_one_at_a_time():
    """`allclose(solve(left, right), target)` per triple: near the tolerance, with
    infinities and NaN, and with a singular matrix (`None`, or raised)."""
    import pytest
    from dycraft.animation.blender.verify import _mechanism_held

    rng = np.random.default_rng(5)

    def pose(angle):
        m = np.eye(4)
        m[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        m[:3, 3] = rng.normal(size=3)
        return m

    def one_at_a_time(lefts, rights, targets):
        held = []
        for left, right, target in zip(lefts, rights, targets):
            try:
                actual = np.linalg.solve(np.asarray(left), np.asarray(right))
            except np.linalg.LinAlgError:
                held.append(None)
                continue
            held.append(bool(np.allclose(actual, target, atol=1e-5)))
        return held

    for trial in range(200):
        k = int(rng.integers(1, 120))
        lefts = [pose(rng.uniform(0, 6)) for _ in range(k)]
        targets = [pose(rng.uniform(0, 6)) for _ in range(k)]
        rights = [left @ target + rng.choice([0, 1e-6, 9.99e-6, 1.001e-5, 1e-3]) * rng.normal(size=(4, 4))
                  for left, target in zip(lefts, targets)]
        if trial % 20 == 3:
            rights[0][0, 0] = targets[0][0, 0] = np.inf
        if trial % 20 == 5:
            rights[0][1, 1] = np.nan
        if trial % 20 == 7:
            lefts[-1] = np.zeros((4, 4))
        lists = [[m.tolist() for m in ms] for ms in (lefts, rights, targets)]
        assert _mechanism_held(*lists) == one_at_a_time(*lists), trial
        if trial % 20 == 7:
            with pytest.raises(np.linalg.LinAlgError):
                _mechanism_held(*lists, singular=False)


def test_the_mesh_writer_writes_what_a_direct_write_does_and_reports_a_failure(tmp_path):
    import pytest
    from dycraft.animation.blender.evaluation import MeshWriter, _read_asset, write_mesh

    rng = np.random.default_rng(2)
    meshes = [(rng.normal(size=(50 + k, 3)).astype(np.float32),
               rng.integers(0, 50, size=(20, 3)).astype(np.int32)) for k in range(30)]
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    writer, known = MeshWriter(limit=3), set()
    keys = [write_mesh(tmp_path / "a", v, t, known, writer=writer) for v, t in meshes]
    writer.flush()
    direct = [write_mesh(tmp_path / "b", v, t, set()) for v, t in meshes]
    assert keys == direct
    for key, (v, t) in zip(keys, meshes):
        assert ((tmp_path / "a/meshes" / f"{key}.npz").read_bytes()
                == (tmp_path / "b/meshes" / f"{key}.npz").read_bytes())
        got_v, got_t = _read_asset(tmp_path / "a/meshes" / f"{key}.npz", key)
        assert np.array_equal(got_v, v) and np.array_equal(got_t, t)
    writer.close()
    failing = MeshWriter()
    failing.submit(tmp_path / "missing/dir/x.npz", *meshes[0])
    with pytest.raises(OSError):
        failing.flush()
    failing.close()
