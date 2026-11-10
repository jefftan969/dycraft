"""PlainSession — the one-op session behind the `plain` layer.

Mirrors `LayerSession` in shape (an
`AgentSession` subclass that registers its ops in `self.handlers` for the
generic daemon to dispatch), but carries no SDK state. The sole op is
`submit_result`, the free-text return channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..agent_session import AgentSession, utcnow_iso

RESULT_FILENAME = "result.json"


@dataclass
class PlainContext:
    """Per-run context for the plain layer. Satisfies the AgentContext
    protocol (`run_dir`, `step`, `committed`) plus
    the result accumulator and a `meta_extras()` hook for meta.json."""

    run_dir: Path
    step: int = 0
    committed: bool = False
    result_text: str | None = None

    def meta_extras(self) -> dict:
        """Layer-specific meta.json fields. Keeps a copy of the returned
        text's length for post-hoc inspection.

        `outcome` must not be in here. It is `finalize_meta`'s own positional
        argument, and `AgentSession.finalize_meta` splices these extras in as
        `**merged`, so the key would raise `TypeError` inside the daemon's
        `atexit` hook and the run would record no outcome. The daemon passes
        the outcome itself (`server.py`).
        """
        out: dict = {"layer": "plain"}
        if self.committed and self.result_text is not None:
            out["result_chars"] = len(self.result_text)
        return out


class PlainSession(AgentSession[PlainContext]):
    """Registers the single `submit_result` op and nothing else."""

    def __init__(self, ctx: PlainContext, **kwargs):
        super().__init__(ctx, **kwargs)
        self._register_ops()

    def _register_ops(self) -> None:
        ctx = self.ctx

        @self.op("submit_result")
        def _submit_result(*, text: str) -> dict:
            if not isinstance(text, str):
                return {"ok": False,
                        "error": f"text must be a string, got {type(text).__name__}"}
            ctx.result_text = text
            # Persist the answer where `run_oneoff` reads it back from.
            try:
                (ctx.run_dir / RESULT_FILENAME).write_text(json.dumps(
                    {"text": text, "submitted_at": utcnow_iso()},
                    indent=2,
                ))
            except OSError as e:
                return {"ok": False,
                        "error": f"failed to write {RESULT_FILENAME}: {e}"}
            ctx.committed = True
            # The answer at the run's top level, for a run laid out for a
            # reader; a one-off's sub-run is not, and is left as it is.
            from ...run_layout import publish
            publish(ctx.run_dir)
            # Drop the self-kill sentinel so the launcher's watcher SIGTERMs
            # the CLI promptly — a one-off should end the instant it answers,
            # not idle to the step cap. Best-effort: if this fails the run
            # still terminates at the step cap.
            try:
                (ctx.run_dir / ".terminate").touch()
            except OSError:
                pass
            return {"ok": True, "result_chars": len(text)}


__all__ = ["PlainContext", "PlainSession", "RESULT_FILENAME"]
