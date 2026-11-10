"""The animation layer's daemon session: the film agent's context and its op host.

The film agent authors a Blender project; its operations are the animation service's
(`animation.session.register`), not the object and scene layers' compile/probe/snapshot. It therefore has
no `LayerConfig`: the session below hosts those operations, carries the few fields the service reads, and
writes the layer's `meta.json` fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..agent_session import AgentSession
# At module scope, so the daemon holds it before the worktree it ran from can
# be deleted under its shutdown hook (`meta_extras` runs at shutdown).
from ...animation.delivery import best_film


@dataclass
class FilmContext:
    """What the animation service and the tools it calls read from `session.ctx`.

    `run_dir` is how `review_backend`, `run_model`, `run_reasoning_effort` and
    `_child_argv` find the run's `meta.json`.
    """

    run_dir: Path
    user_prompt: str = ""
    step: int = 0
    committed: bool = False
    last_successful_ir: dict | None = None
    n_compile_attempts: int = 0
    last_compile_payload: dict | None = None

    def meta_extras(self) -> dict:
        """`meta.json`'s animation fields. `best_film` is what an uncommitted run
        leaves behind, and `None` once it commits, since `ir.json` names the film
        then. Never raises: a fault here would cost the whole `meta.json` write."""
        meta = {"animation_authoring": "blender", "best_film": None}
        if not self.committed:
            try:
                meta["best_film"] = best_film(self.run_dir)
            except Exception:                                    # noqa: BLE001
                pass
        return meta


class FilmSession(AgentSession[FilmContext]):
    """The animation layer's session: the animation service's operations on the generic session."""

    def __init__(self, run_dir: Path, *, user_prompt: str = "", initial_meta: dict | None = None):
        ctx = FilmContext(run_dir=run_dir, user_prompt=user_prompt)
        super().__init__(ctx, user_prompt=user_prompt, initial_meta_fields=initial_meta)
        from ...animation.session import register
        register(self)
        self._require_cad_worker()

    def op_scope(self):
        """Pin each op to the build it produced or reused (`tools.pinned_build`):
        the film's `design/scene.py` builds through the scene layer's machinery."""
        from .tools import build_pin_scope
        return build_pin_scope()

    def _require_cad_worker(self) -> None:
        """Refuse to start without a CAD interpreter: the film's scene source is
        build123d and runs in the frozen CAD environment."""
        from .session import require_cad_worker
        require_cad_worker(self)
