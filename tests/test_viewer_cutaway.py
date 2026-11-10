"""The walls and ceilings the free camera may cut away (`viewer/cutaway.py`), on a room built here from boxes: the
classifier sees only geometry, so the names below are for the reader, never for it."""
from __future__ import annotations

import numpy as np

from dycraft.viewer.cutaway import classify

# A box's 12 triangles, outward: corner k has x from bit 0, y from bit 1, z from bit 2.
_FACES = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4), (2, 6, 3), (3, 6, 7),
          (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]


def _box(lo, hi, base=0):
    v = np.array([[(hi if k & 1 else lo)[0], (hi if k & 2 else lo)[1], (hi if k & 4 else lo)[2]] for k in range(8)],
                 dtype=float)
    return v, np.array(_FACES) + base


def _mesh(*boxes):
    vs, ts = [], []
    for lo, hi in boxes:
        v, t = _box(lo, hi, 8 * len(vs))
        vs.append(v); ts.append(t)
    return np.concatenate(vs), np.concatenate(ts)


def _room(*, through_doorway=True):
    """A 6 x 6 m room, walls 10 cm thick and 2.6 m tall, and a character walking from its middle to the east."""
    static = {
        "floor": _mesh(((-3, -3, -0.1), (3, 3, 0))),
        "south_wall": _mesh(((-3, -3.1, 0), (3, -3, 2.6))),
        "north_wall": _mesh(((-3, 3, 0), (3, 3.1, 2.6))),
        # the east wall has a doorway a meter wide, under a lintel
        "east_wall": _mesh(((3, -3, 0), (3.1, -0.5, 2.6)), ((3, 0.5, 0), (3.1, 3, 2.6)),
                           ((3, -0.5, 2.1), (3.1, 0.5, 2.6))),
        "west_wall": _mesh(((-3.1, -3, 0), (-3, 3, 2.6))),
        "ceiling": _mesh(((-3, -3, 2.6), (3, 3, 2.7))),
        # a window in the west wall, off the floor
        "window": _mesh(((-3.08, -0.5, 1.0), (-3.02, 0.5, 1.8))),
        # a door in the south wall: the leaf stands on the floor, its glass is off it
        "door_leaf": _mesh(((-1.0, -3.08, 0.01), (0.0, -3.02, 2.0))),
        "door_glass": _mesh(((-0.8, -3.07, 1.0), (-0.2, -3.03, 1.8))),
        # shelving along the north wall: a back panel and five shelves
        "shelving": _mesh(((0, 2.85, 0), (1.5, 2.9, 2.0)),
                          *[((0, 2.5, z), (1.5, 2.9, z + 0.03)) for z in (0.0, 0.5, 1.0, 1.5, 1.97)]),
        # the next building's roof, beside the action rather than over it
        "neighbor_roof": _mesh(((8, 8, 3.0), (12, 12, 3.1))),
    }
    names = list(static) + ["character"]
    meshes = list(static.values()) + [_mesh(((-0.25, -0.25, 0), (0.25, 0.25, 1.7)))]
    nf = 60
    mats = np.zeros((nf, len(names), 3, 4), dtype=np.float32)
    mats[:, :, :, :3] = np.eye(3)
    end = 4.0 if through_doorway else 2.0
    mats[:, -1, 0, 3] = np.linspace(0.0, end, nf)
    header = {"n_frames": nf, "meshes": [str(k) for k in range(len(names))],
              "objects": [{"name": n, "mesh": k, "hidden": False} for k, n in enumerate(names)]}
    return header, mats, lambda k: meshes[k]


def _by_kind(result):
    walls = sorted(u["objects"][0] for u in result["units"] if u["kind"] == "wall")
    ceilings = sorted(u["objects"][0] for u in result["units"] if u["kind"] == "ceiling")
    mounted = {a["objects"][0]: result["units"][a["units"][0]]["objects"][0] for a in result["attach"]}
    kept = {k["objects"][0]: k["why"] for k in result["kept"]}
    return walls, ceilings, mounted, kept


def test_the_walls_and_ceiling_of_a_room_are_found_from_geometry_alone():
    walls, ceilings, mounted, kept = _by_kind(classify(*_room()))
    assert walls == ["north_wall", "south_wall", "west_wall"]
    assert ceilings == ["ceiling"]
    assert "shelving" not in walls, "shelving is furniture, not a wall"
    assert "neighbor_roof" not in ceilings, "a roof beside the action is not this room's ceiling"


def test_the_wall_whose_doorway_the_action_passes_through_is_never_cut():
    walls, _, _, kept = _by_kind(classify(*_room()))
    assert "east_wall" not in walls and "through" in kept["east_wall"]
    walls, _, _, kept = _by_kind(classify(*_room(through_doorway=False)))
    assert "east_wall" in walls, "with nobody passing through it, it is a wall like the others"


def test_a_window_goes_with_its_wall_and_a_door_never_goes():
    _, _, mounted, _ = _by_kind(classify(*_room()))
    assert mounted == {"window": "west_wall"}
    assert "door_leaf" not in mounted, "a door stands on the floor"
    assert "door_glass" not in mounted, "a door's glass stays with its door"


def test_a_scene_with_no_room_cuts_nothing_away():
    header, mats, load = _room()
    keep = [i for i, o in enumerate(header["objects"]) if o["name"] in ("floor", "character", "neighbor_roof")]
    header = {**header, "objects": [header["objects"][i] for i in keep]}
    result = classify(header, mats[:, keep], load)
    assert result["units"] == [] and result["attach"] == []


def _bundle(tmp_path, monkeypatch, name="film"):
    """A run the cutaway can key on: a native manifest, and the viewer asset cache in a temp folder."""
    import json
    from dycraft.animation.blender import SCHEMA
    monkeypatch.setenv("DYCRAFT_VIEWER_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    run = tmp_path / name
    run.mkdir()
    (run / "ir.json").write_text(json.dumps({"schema": SCHEMA, "id": f"{name}-id"}))
    return run


def test_a_cutaway_that_times_out_cuts_nothing_and_never_runs_in_the_server(tmp_path, monkeypatch):
    """Working it out in the server after the child ran out of time would hold the server's interpreter lock for as
    long again, which is what running it apart is for: the film gets no cutaway, and this server does not retry."""
    import subprocess
    from dycraft.viewer import cutaway as C
    run = _bundle(tmp_path, monkeypatch)
    calls = []

    def times_out(run_dir):
        calls.append("apart")
        raise subprocess.TimeoutExpired("child", C.TIMEOUT_S)

    monkeypatch.setattr(C, "_TIMED_OUT", set())          # this test's, so what it remembers goes with it
    monkeypatch.setattr(C, "_work_out_apart", times_out)
    monkeypatch.setattr(C, "_work_out", lambda *a: calls.append("here") or {})
    for _ in range(2):
        result = C.cutaway(run)
        assert result["units"] == [] and result["animation"] == "film-id"
    assert calls == ["apart"], calls


def test_a_child_that_cannot_start_falls_back_to_the_server(tmp_path, monkeypatch):
    from dycraft.viewer import cutaway as C
    run = _bundle(tmp_path, monkeypatch, "other")

    def cannot_start(run_dir):
        raise OSError("no interpreter")

    monkeypatch.setattr(C, "_work_out_apart", cannot_start)
    monkeypatch.setattr(C, "_work_out", lambda run_dir, ir: {"version": C.VERSION, "animation": ir["id"],
                                                             "action": None, "units": [{"kind": "wall"}],
                                                             "attach": [], "kept": []})
    assert C.cutaway(run)["units"] == [{"kind": "wall"}]
