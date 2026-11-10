"""The animation layer: one author for source geometry, native Blender
animation and film delivery.

`session` is the layer's service (`AnimationSession`), which the launcher runs
through agent/layers/layer.py; `planning`, `delivery`, `dispatch`, `visual`, and
`context` are its parts, and `review_render` and `stage_scene` run as
`python -m`. `blender` owns native authoring, versioned asynchronous jobs,
evaluated geometry verification, accepted bundles, and rendering.

`ik` (cuRobo) returns reference solver data, not committed motion. `sim` holds
the URDF scene export and the child runner `ik` uses; blender/solvers.py adapts
the selected evaluated Blender frame to that export. A missing IK environment
returns error_kind="unavailable". Native Blender simulation may also be authored/baked.

This package has no eager re-exports; importing it does not load bpy or solvers.
"""
