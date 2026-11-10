"""dycraft.animation.sim — the URDF scene export, and the child runner `h.ik` uses.

  * `scene_export`  — SceneIR + catalog -> a self-contained URDF world plus one
                      URDF per free body, and `scene_meta.json`, the ground
                      truth a loaded scene is checked against. `h.ik`'s solver
                      export (`animation/blender/solvers.py`) builds on it.
  * `_sim_runner`   — a stdlib-only child that runs agent code under another
                      interpreter and records its result; `h.ik` executes it
                      by path (`animation/ik/ik.py`).
"""

from .scene_export import (
    SIM_DIR,
    SCENE_META,
    SimExportError,
    export_sim_scene,
)

__all__ = ["SIM_DIR", "SCENE_META", "SimExportError", "export_sim_scene"]
