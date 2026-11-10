"""dycraft.agent — the agent layers' harnesses.

Public API:

    from dycraft.agent import LAYERS, LayerSpec, stage_harness_client

Module layout:

    agent/
      agent_session.py    — generic AgentSession + AgentContext + TraceMeta
      agent_runner.py     — run-id, ensure_run_dir,
                            compose_system_prompt, compose_footer
      oneoff.py           — one-off CLI sub-agent calls (review, the rewriters)
      scene_rewrite.py    — the per-brief scene entry, written before the run starts
      motion_rewrite.py   — the per-brief motion entry, likewise
      transcript_capture.py — the agent CLI's session transcript, into the run bundle
      usage.py            — token totals read from that transcript
      tool_io.py          — round_floats, atomic_write_text, confine_to_run_dir
      _harness_core.py    — stdlib-only transport core for $RUN_DIR/harness.py
      server.py           — generic daemon (framing, watchdog, accept loop)
      layers/             — the object, scene, and animation layers' harness
        session.py        — LayerSession: the object and scene layers' daemon ops
        film.py           — FilmSession: the animation layer's daemon ops
        server.py         — the layers' daemon main (builds the session)
        layer.py          — LayerConfig per layer
        tools.py          — op implementations (compile, probe, final render)
        snapshot.py       — the `snapshot` op
        review.py         — the `review` op
        compile.py        — IR build + verifier suite driver
        _harness_tools*.py — the client stubs staged into $RUN_DIR/harness.py
        scene_staging.py  — frozen SceneIR + catalog staging
      cad/                — CAD worker, solid compile, build123d namespace, sandbox
      assets/             — glTF/URDF importers, asset library, --preload-asset
      plain/              — one-off text-task layer (review, rewriters)

Adding a new agent layer: register it in `LAYERS` below (server module +
tools-stub path) — that is the single source of truth the launcher reads
from when spawning the harness daemon and assembling `harness.py`.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import _home
from .layers import materialize_catalog


_AGENT_DIR = Path(__file__).resolve().parent
# The checkout this package was imported from. `.resolve()` is what makes it
# right under the ablation: a cell imports `dycraft` through a pkgroot symlink
# into a git worktree, so this lands in the worktree rather than in whatever
# tree happens to be installed editable.
_REPO_ROOT = _AGENT_DIR.parent
_HARNESS_CORE_SRC = _AGENT_DIR / "_harness_core.py"


@dataclass(frozen=True)
class LayerSpec:
    """Per-agent-layer registration: where the daemon lives + which tool
    stubs to append to the staged `harness.py`."""

    name: str
    server_module: str  # e.g. "dycraft.agent.layers.server"
    tools_src: Path     # path to <subpkg>/_harness_tools.py
    #: Further stub files concatenated after `tools_src`, in order, for a
    #: layer whose surface is the shared one PLUS something — cheaper than a
    #: forked copy of the whole file, and **two staged clients that drift
    #: apart is a bug nobody sees until an agent calls a tool the daemon never
    #: registered.** Object appends `review`, scene appends its dispatch
    #: stubs.
    extra_tools_src: tuple = ()


LAYERS: dict[str, LayerSpec] = {
    # Object appends its own `review`. See `_harness_tools_object.py`.
    "object": LayerSpec(
        name="object",
        server_module="dycraft.agent.layers.server",
        tools_src=_AGENT_DIR / "layers" / "_harness_tools.py",
        extra_tools_src=(_AGENT_DIR / "layers" / "_harness_tools_object.py",),
    ),
    # Scene is the shared surface plus a targetable `review` and the object
    # dispatch calls. See `_harness_tools_scene.py`.
    "scene": LayerSpec(
        name="scene",
        server_module="dycraft.agent.layers.server",
        tools_src=_AGENT_DIR / "layers" / "_harness_tools.py",
        extra_tools_src=(_AGENT_DIR / "layers" / "_harness_tools_scene.py",),
    ),
    "animation": LayerSpec(
        name="animation",
        server_module="dycraft.agent.layers.server",
        tools_src=_AGENT_DIR / "layers" / "_harness_tools_animation.py",
    ),
    "plain": LayerSpec(
        name="plain",
        server_module="dycraft.agent.plain.server",
        tools_src=_AGENT_DIR / "plain" / "_harness_tools.py",
    ),
}


def stage_harness_client(run_dir: Path, layer: str) -> Path:
    """Assemble the self-contained harness client into `$run_dir/harness.py`.

    Concatenates the agent-agnostic transport core (`_harness_core.py`) with
    the layer's tool stubs (`<subpkg>/_harness_tools.py`) so the staged
    module exposes exactly the tools the running agent's prompt documents.
    Stdlib-only — a staged run dir can be replayed/debugged without the
    dycraft tree on path.
    """
    spec = LAYERS.get(layer)
    if spec is None:
        raise SystemExit(
            f"stage_harness_client: unknown layer {layer!r} "
            f"(known: {sorted(LAYERS)})"
        )
    # The revision's own Skill trees, COPIED (not linked) beside the stubs.
    # Staged here rather than from `cli.py` so nested runs, `run_oneoff`
    # reviewers and dispatched children get one too — the launcher-only site
    # missed exactly the sub-runs `review()` depends on.
    stage_skills(run_dir)
    stage_references(run_dir)
    if layer in _IK_LAYERS and _ik_available():
        stage_ik_references(run_dir)
    dst = run_dir / "harness.py"
    parts = [_HARNESS_CORE_SRC.read_text(), spec.tools_src.read_text()]
    parts.extend(Path(extra).read_text() for extra in spec.extra_tools_src)
    dst.write_text("\n".join(parts))
    return dst


# The Skill trees an agent may be pointed at. `.claude/skills` is what the claude
# CLI reads and `.agents/skills` what codex reads; the repo keeps them mirrored.
# Both are tracked, so both are part of the revision under test.
_SKILL_TOOL_DIRS = (".claude", ".agents")


def stage_skills(run_dir: Path) -> list:
    """Copy this checkout's Skill trees into `run_dir`, beside `harness.py`.

    The object and scene prompts point the agent at
    `$DYCRAFT_RUN_DIR/.claude/skills/<name>/SKILL.md`, and the run dir is also
    the cwd (`backends` chdirs there so the sandbox's writable root and the cwd
    tree agree), so a relative `.claude/skills/...` resolves to the same place.
    Without a tree there, the agent walks back to whatever checkout it can
    see: the run dir's own path names the checkout (`--runs-dir` defaults to
    `<repo>/runs`), so it reconstructs the repo root from `$DYCRAFT_RUN_DIR`
    and reads the Skills of the **main working tree**, not the revision's own
    copy. A run of an older revision would then read the current copy of the
    Skills, which are tracked and do change between commits.

    Copied, not symlinked. The source resolves through `__file__`, so for a run
    of another revision it can be a temporary git worktree that is deleted
    afterwards, and a symlink would leave every archived bundle pointing at
    nothing. A copy also freezes the text the agent actually saw into the
    bundle, which is otherwise unrecorded, and keeps the checkout out of reach
    of a run that is granted write access to `run_dir`.

    A Skill about something this checkout lacks (`agent_runner.OPTIONAL_SKILLS`,
    such as `mocap_reference` without the motion-capture library) is left out.

    Returns the staged destinations. A revision that carries no Skill tree is
    reported, not fatal.
    """
    from .agent_runner import OPTIONAL, OPTIONAL_SKILLS
    absent = {name for name, need in OPTIONAL_SKILLS.items() if not OPTIONAL[need]()}
    staged = []
    for tool_dir in _SKILL_TOOL_DIRS:
        src = _REPO_ROOT / tool_dir / "skills"
        if not src.is_dir():
            continue
        dst = run_dir / tool_dir / "skills"
        # `skills` rather than its parent: `.claude/` also collects CLI-owned
        # state during a run, and re-staging into a replayed run dir must not
        # take that with it.
        shutil.rmtree(dst, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, ignore=lambda d, names: absent.intersection(names) if Path(d) == src else ())
        staged.append(dst)
    if not staged:
        print(f"[dycraft] no Skill tree under {_REPO_ROOT} — the prompt points "
              f"the agent at .claude/skills, where it will now find nothing",
              file=sys.stderr)
    return staged


#: Read-only reference libraries a run can browse from its own directory, as
#: `<run>/references/<name>` -> `<repo>/assets/<dir>`. Linked, not copied: the
#: CMU library is 3.3 GB. The Blender sandbox sees the mocap tree at
#: `animation.blender.runtime.MOCAP_MOUNT`; this link is what the agent's own
#: shell sees, because a sandbox-only mount reads as "not installed" to an agent
#: looking from its shell. `assets` is the imported-asset library the agent
#: retrieves its characters and robots from (`assets/asset_library.py`;
#: `load_asset` stages what it picks).
_REFERENCE_LIBRARIES = {"mocap": "mocap", "assets": "imported"}


#: The kinematics solver's docs and examples, copied into an animation run as
#: `<run>/references/ik/`, because a run has no way to resolve a path into the
#: DyCraft checkout. Copied rather than linked for the reason Skills are: the
#: bundle then records the text the agent saw. `h.ik(script=...)` runs anything
#: under the run dir, so the examples run from here as they are.
_IK_REFERENCES = ("README.md", "CUROBO.md", "curobo_docs", "examples")
_IK_LAYERS = ("animation",)


def _ik_available() -> bool:
    """Whether this run has `h.ik`: its docs are staged only then, as the
    prompt describes `h.ik` only then."""
    from ..animation.ik.ik import is_available
    return is_available()


def stage_ik_references(run_dir: Path) -> Path | None:
    """Copy `animation/ik/`'s docs and examples to `run_dir/references/ik/`."""
    src = _REPO_ROOT / "animation" / "ik"
    if not (src / "README.md").is_file():
        return None
    dst = run_dir / "references" / "ik"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for name in _IK_REFERENCES:
        item = src / name
        if item.is_dir():
            shutil.copytree(item, dst / name, ignore=shutil.ignore_patterns("__pycache__"))
        elif item.is_file():
            shutil.copy2(item, dst / name)
    return dst


def stage_references(run_dir: Path) -> list:
    """Link the reference libraries under `assets/` into `run_dir/references/`.

    `assets/` is user state, so it sits under `_home.home()`: the source
    checkout when running from one, else the launch directory.
    """
    from ..animation.blender.runtime import mocap_library
    from .assets.asset_library import LIBRARY_ENV
    linked = []
    for name, sub in _REFERENCE_LIBRARIES.items():
        env = os.environ.get(LIBRARY_ENV) if name == "assets" else None
        src = (Path(env) if env else _home.state_path("assets", sub)).resolve()
        if not src.is_dir() or (name == "mocap" and mocap_library() is None):
            continue
        dst = run_dir / "references" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and dst.resolve() == src:
                linked.append(dst)
                continue
            dst.unlink()
        dst.symlink_to(src, target_is_directory=True)
        linked.append(dst)
    return linked


__all__ = [
    "LayerSpec",
    "LAYERS",
    "stage_harness_client",
    "stage_skills",
    "stage_references",
    "stage_ik_references",
    "materialize_catalog",
]
