"""Plain harness daemon — binds a PlainSession and serves it.

Thin per-agent wrapper around the generic `agent.server.serve()`, the plain
analog of `layers/server.py`. Started by the launcher before it
starts the agent CLI:

    python -m dycraft.agent.plain.server serve <run_dir>

No catalog, no SDK state — just the single `submit_result` op.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..agent_runner import user_prompt_from_env
from ..server import serve
from .session import PlainContext, PlainSession


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "serve":
        print("usage: python -m dycraft.agent.plain.server serve <run_dir>",
              file=sys.stderr)
        return 2
    run_dir = Path(argv[2]).resolve()

    ctx = PlainContext(run_dir=run_dir)
    session = PlainSession(
        ctx,
        user_prompt=user_prompt_from_env(),
        # meta.json was already written by the launcher (cli.py::write_run_meta);
        # finalize_meta at shutdown refreshes outcome / finished_at.
        initial_meta_fields={"layer": "plain"},
    )

    with session:
        serve(session, run_dir, start_fields={"layer": "plain"})
    return 0


if __name__ == "__main__":
    # Inherited from `dycraft.cli` when it spawns this daemon; raised here
    # too for a daemon started any other way. See `_oom`.
    from ... import _oom
    _oom.raise_self(_oom.SESSION)
    raise SystemExit(main(sys.argv))
