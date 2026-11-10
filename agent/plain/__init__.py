"""dycraft.agent.plain — minimal "one-off text prompt → text result" layer.

The object and scene layers carry heavy SDK semantics (catalog,
compile, verifier suite, frozen IR commit). This layer carries none of that:
it exists purely so a CLI sub-agent can be handed an arbitrary text prompt and
hand back an arbitrary text answer, while still riding the *exact* launch path
(daemon + harness + PTY-wrapped CLI + sandbox + staged home + no-progress
watchdog) that the real dispatches use.

The single harness op is `submit_result(text)` — the free-text return channel
that the SDK layers lack. Calling it persists `<run_dir>/result.json`, marks
the run committed, and drops the `.terminate` sentinel so the CLI self-kills
promptly instead of rambling to the step cap.

Its consumer is `agent/oneoff.py::run_oneoff`, the blocking primitive behind
`review()` and the scene and motion rewriters.
"""

from __future__ import annotations

from .session import PlainContext, PlainSession

__all__ = ["PlainContext", "PlainSession"]
