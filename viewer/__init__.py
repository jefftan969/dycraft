"""dycraft run viewer — FastAPI server + React SPA over `runs/`.

Nothing here is imported by the harness or the build path; the viewer is a
read-only lens over finished run bundles. It has its own dependencies
(`fastapi`, `uvicorn`) that the agent runtime does not need.

Launch:

    python -m dycraft.viewer --port 8770

The frontend is vendored from Articraft's viewer (Apache-2.0) — see the
top-level NOTICE.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazy re-export so importing `dycraft.viewer` costs no fastapi import."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)
