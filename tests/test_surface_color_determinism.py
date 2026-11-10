"""A frame's flat surface color must not depend on the worker process's history.

A compile and a final render must sign the same frame identically, so:

  * the color walk identifies nodes by their identity in the graph, not by
    `id()` of bpy's per-access Python wrappers, which are freed and reused;
  * the resident Blender resets the per-material color cache before every
    request, since an author edits materials in place between them;
  * a replay mismatch names what changed, colors included, not only transforms.
"""
import importlib.util
import io
import json
import pickle
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dycraft.animation.blender.evaluation import (check_replay_frame, frame_signature,
                                                  read_local, write_mesh)
from dycraft.animation.blender.runtime import REPO
from dycraft.render._blender_env import blender_python


def _worker(monkeypatch, tmp_path, bpy=None):
    monkeypatch.setitem(sys.modules, "bpy", bpy or SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mathutils",
                        SimpleNamespace(Euler=None, Matrix=None, Vector=None))
    monkeypatch.setitem(sys.modules, "_blender_child", SimpleNamespace())
    monkeypatch.setenv("DYCRAFT_BLENDER_WORKSPACE_ROOT", str(tmp_path))
    # The worker imports the modules beside it by name; drop any a previous test cached, so they see this bpy.
    for sibling in ("surface_colors", "frame_export", "animation_diff"):
        monkeypatch.delitem(sys.modules, sibling, raising=False)
    spec = importlib.util.spec_from_file_location(
        "color_worker", REPO / "animation/blender/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.reset_surface_color_cache()
    return module


class _Wrapper:
    """One Python object standing for whichever node was last accessed: the
    worst case of bpy recycling a freed wrapper's address."""

    def bind(self, node):
        self.type, self.inputs, self._ptr = node["type"], node["inputs"], node["ptr"]
        return self

    def as_pointer(self):
        return self._ptr


class _Socket:
    def __init__(self, wrapper, upstream=None, value=(0, 0, 0, 1)):
        self.type, self.default_value = "RGBA", list(value)
        self._wrapper, self._upstream = wrapper, upstream

    @property
    def is_linked(self):
        return self._upstream is not None

    @property
    def links(self):
        if self._upstream is None:
            return []
        return [SimpleNamespace(from_node=self._wrapper.bind(self._upstream))]


def test_the_walk_tells_nodes_apart_when_their_wrappers_share_an_id(monkeypatch, tmp_path):
    worker = _worker(monkeypatch, tmp_path)
    w = _Wrapper()
    red = {"type": "MIX", "ptr": 1, "inputs": [_Socket(w, value=(1, 0, 0, 1))]}
    blue = {"type": "MIX", "ptr": 2, "inputs": [_Socket(w, value=(0, 0, 1, 1))]}
    mix = {"type": "MIX", "ptr": 3,
           "inputs": [_Socket(w, upstream=red), _Socket(w, upstream=blue)]}
    colors = sys.modules["surface_colors"]._constant_colors(_Socket(w, upstream=mix))
    assert sorted(colors) == [(0, 0, 1), (1, 0, 0)]


def _frame(stream):
    data = pickle.dumps(stream, protocol=4)
    return struct.pack("!Q", len(data)) + data


def test_the_resident_loop_resets_colours_before_every_request(monkeypatch, tmp_path):
    bpy = SimpleNamespace(
        context=SimpleNamespace(preferences=SimpleNamespace(filepaths=SimpleNamespace())),
        data=SimpleNamespace(objects=[]))
    worker = _worker(monkeypatch, tmp_path, bpy)
    monkeypatch.setenv("DYCRAFT_BLENDER_OUTPUT_ROOT", str(tmp_path))   # serve sets it per request
    seen = []

    def run_job(job, reopen=True):
        seen.append(dict(sys.modules["surface_colors"]._SURFACE_COLOR_CACHE))
        sys.modules["surface_colors"]._SURFACE_COLOR_CACHE["walls"] = (0.28, 0.26, 0.27, 1.0)

    monkeypatch.setattr(worker, "run_job", run_job)
    request = {"job": {"op": "evaluate"}, "output": str(tmp_path / "out")}
    replies = io.BytesIO()
    worker.serve(io.BytesIO(_frame(request) * 2), replies)
    assert seen == [{}, {}]


def test_a_replay_refusal_names_the_field_that_changed(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    t = np.array([[0, 1, 2]], dtype=np.int32)
    key = write_mesh(tmp_path, v, t, set())
    eye = np.eye(4).tolist()
    record = {"frame": 0, "transforms": {"walls": eye, "pouch": eye},
              "objects": [{"name": "walls", "mesh": key, "matrix": eye, "color": [0.28, 0.26, 0.27, 1]},
                          {"name": "pouch", "mesh": key, "matrix": eye}]}
    local = read_local(tmp_path, record)
    record["signature"] = frame_signature(record, local)
    (tmp_path / "000000.json").write_text(json.dumps(record))
    check_replay_frame(tmp_path, 0, record, local)
    walls = {**record["objects"][0], "color": [0.136, 0.094, 0.066, 1]}
    changed = {**record, "objects": [walls, record["objects"][1]]}
    with pytest.raises(ValueError) as refused:
        check_replay_frame(tmp_path, 0, changed, local)
    message = str(refused.value)
    assert "changed color: ['walls [0.28, 0.26, 0.27] -> [0.136, 0.094, 0.066]']" in message
    assert "pouch" not in message and "bake" not in message


_REAL = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
os.environ["DYCRAFT_BLENDER_WORKSPACE_ROOT"] = sys.argv[2]
import bpy, worker
m = bpy.data.materials.new("bricks"); m.use_nodes = True
nt = m.node_tree; bsdf = nt.nodes["Principled BSDF"]
prev = None
for k in range(12):
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    bricks = nt.nodes.new("ShaderNodeTexBrick")
    bricks.inputs["Color1"].default_value = (0.05 * k, 0.1, 0.2, 1)
    nt.links.new(bricks.outputs["Color"], mix.inputs["A"])
    if prev is not None:
        nt.links.new(prev.outputs["Result"], mix.inputs["B"])
    else:
        mix.inputs["B"].default_value = (0.9, 0.1, 0.1, 1)
    prev = mix
nt.links.new(prev.outputs["Result"], bsdf.inputs["Base Color"])
walks = set()
for _ in range(20):
    junk = [n.inputs for n in nt.nodes for _ in range(3)]
    walks.add(tuple(map(tuple, sys.modules["surface_colors"]._constant_colors(bsdf.inputs["Base Color"]))))
print("DISTINCT", len(walks))
"""


def test_a_real_shader_graph_walks_the_same_every_time(tmp_path):
    py = blender_python()
    if py is None:
        pytest.skip("needs the dycraft-blender interpreter")
    script = tmp_path / "walk.py"
    script.write_text(_REAL)
    out = subprocess.run([str(py), str(script), str(REPO / "animation/blender"), str(tmp_path)],
                         capture_output=True, text=True, timeout=300)
    assert "DISTINCT 1" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]
