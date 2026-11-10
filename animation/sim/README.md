# Scene Export and Child Runner (`animation/sim`)

Two pieces ship here, and `h.ik` (the cuRobo consultation op) uses both.

## `scene_export.py`

`export_sim_scene(scene_ir, catalog, out_dir, *, free=(), asset_base_dir)` writes a
world a kinematics or physics library can load, and returns its meta dict:

| Path | Contents |
|---|---|
| `world.urdf` | Every static instance, one robot, fixed base |
| `bodies/<instance>.urdf` | One per instance named in `free`, each its own robot with no weld |
| `meshes/` | The geometry, with paths relative to the URDF that references them |
| `scene_meta.json` | Rest poses, world AABBs, joint limits **and defaults**, room dimensions |

It composes `render/urdf_export.py` without changing it. Two things a consumer must
know:

- **URDF `q = 0` is not the rest pose.** A joint whose authored `default` is non-zero
  loads at a configuration the scene never committed; apply the defaults from
  `scene_meta.json` (as `animation/ik/dycraft_ik.py::default_q` does for cuRobo).
- **Collisions are the meshes.** A mesh part's box collision proxy, which suits
  rendering, is replaced by the mesh its visual names. How a library treats concave
  meshes is its own choice.

The room is not in the URDF: `scene_meta.json` carries its dimensions, so a
consumer builds the floor and walls natively.

`h.ik` calls this through `animation/blender/solvers.py::export_state`, which
exports the evaluated Blender frame.

## `_sim_runner.py`

A stdlib-only child that runs agent-authored code under another interpreter and
records its result:

    <foreign-python> _sim_runner.py --code CODE.py --result RESULT.json \
        [--codename NAME.py] [--checkpoint PARTIAL.json]

The child's working directory is the exported world. The code gets two injected
globals: `result`, a dict whose contents come back as the reply, and
`checkpoint()`, which snapshots `result` to disk so a process that never returns
still reports what it had. `animation/ik/ik.py` runs it by path.
