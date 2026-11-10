"""The `build123d-authoring` Skill must stay true to the runtime.

A Skill is a document, and documents drift. The prompt and
`_b123d_namespace.CURATED` say what `object.py` may call, and a disagreement
costs an agent a compile to discover. The Skill is a third surface saying the
same, so it gets the checks the prompt gets:

  * every identifier it uses must resolve in the real namespace;
  * every complete recipe must run — under the AST allowlist, in the pinned CAD
    interpreter, through the same `validate_shape` gate `add_part` applies;
  * the generated namespace page must equal a fresh generation;
  * the names deliberately kept out must stay out of the prose;
  * both CLI copies must be identical, and the prompt must point at it.

Recipes are marked by a leading `# validated` comment in their fenced block and
must leave a name `part`. Unmarked blocks are fragments: they are still parsed
and still checked for unknown identifiers, but not executed.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util as _u
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dycraft.agent.cad import object_worker as W
from dycraft.agent.cad import sandbox as SB
from dycraft.agent.cad import _cad_env as SA

REPO = Path(__file__).resolve().parents[1]
SKILL_DIRS = (REPO / ".claude" / "skills" / "build123d_authoring",
              REPO / ".agents" / "skills" / "build123d_authoring")
SKILL = SKILL_DIRS[0]
REFERENCE_PAGE = SKILL / "references" / "namespace_reference.md"

_spec = _u.spec_from_file_location(
    "_b123d_ns_skill", REPO / "agent" / "cad" / "_b123d_namespace.py")
NS = _u.module_from_spec(_spec)
_spec.loader.exec_module(NS)

_needs_cad = pytest.mark.skipif(
    not W.is_available(), reason="needs the pinned build123d interpreter")

# Names a recipe may use that are not build123d: the SDK surface `object.py`
# gets, and the restricted builtins.
_SDK_NAMES = {"Object", "Material", "obj", "ir"}

# Stand-ins in illustrative fragments — "`Pos(x, y, z) * shape`". Listed
# explicitly rather than waved through by a lowercase rule, so a fragment that
# reaches for a real build123d name the namespace does not bind still fails
# here instead of reading as a placeholder.
#: Names an illustrative fragment may reference without binding them, because
#: they stand for something the author supplies. A fragment is not executed —
#: `# validated` blocks are, and they may not use these.
_PLACEHOLDERS = {
    "a", "b", "body", "shape", "sketch", "wall",
    "x", "y", "z", "rx", "ry", "rz",
    "radius", "count", "x_spacing", "y_spacing", "x_count", "y_count",
    # The author's own solid and cutter list, in the loop that shows how a
    # subtraction can annihilate a part without raising.
    "top", "cutters", "c",
    # A half-thickness constant, in the one-sided-extrude alternative to
    # `both=True`.
    "HALF_T",
}
# `math` and `np` are the real modules, not curated tuples, so their
# contents are not enumerated here — a page writing `math.tau` or `np.linspace`
# uses the module handle, which is what `_ALLOWED_FREE` needs to contain.
_ALLOWED_FREE = (set(NS.CURATED) | {"bd", "math", "np"}
                 | _SDK_NAMES | set(SB._RESTRICTED_BUILTINS)
                 | set(dir(builtins)))
_ALLOWED_IN_FRAGMENTS = _ALLOWED_FREE | _PLACEHOLDERS


def _pages() -> list[Path]:
    return sorted([SKILL / "SKILL.md"] + list((SKILL / "references").glob("*.md")))


def _blocks(path: Path) -> list[tuple[str, bool]]:
    """(source, is_recipe) for every ```python fence in `path`."""
    out = []
    for m in re.finditer(r"```python\n(.*?)```", path.read_text(), re.DOTALL):
        src = m.group(1)
        out.append((src, src.lstrip().startswith("# validated")))
    return out


def _free_names(src: str) -> set[str]:
    """Names loaded but never bound in `src` — what it expects from the namespace."""
    tree = ast.parse(src)
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - bound


def _all_blocks():
    for page in _pages():
        for i, (src, is_recipe) in enumerate(_blocks(page)):
            yield page, i, src, is_recipe


# ---------------------------------------------------------------------------
# Structure and loading
# ---------------------------------------------------------------------------

def test_the_skill_exists_with_frontmatter():
    text = (SKILL / "SKILL.md").read_text()
    assert text.startswith("---\n"), "SKILL.md needs YAML frontmatter to be discoverable"
    front = text.split("---", 2)[1]
    assert re.search(r"^name:\s*build123d-authoring\s*$", front, re.M), front
    assert re.search(r"^description:\s*\S", front, re.M), "a description drives discovery"


def test_both_cli_copies_are_identical():
    """Claude reads `.claude/skills`, Codex reads `.agents/skills`. One guide."""
    claude, agents = SKILL_DIRS
    for path in _pages():
        mirror = agents / path.relative_to(claude)
        assert mirror.exists(), f"{mirror} is missing"
        assert mirror.read_text() == path.read_text(), (
            f"{mirror} has drifted from {path}")
    extra = {p.relative_to(agents) for p in agents.rglob("*.md")} - {
        p.relative_to(claude) for p in _pages()}
    assert not extra, f"{agents} has files the canonical copy does not: {extra}"


def test_the_cli_prompt_tells_the_agent_to_load_the_skill():
    """Frontmatter alone leaves loading to chance; the prompt makes it an
    instruction. Both, deliberately."""
    cli = (REPO / "agent" / "prompts" / "object.md").read_text()
    assert "build123d-authoring" in cli
    assert "build123d_authoring/SKILL.md" in cli, (
        "give the path too — discovery by name is not guaranteed on every CLI")
    assert re.search(r"[Ll]oad it before", cli) or re.search(
        r"[Ll]oad the `build123d-authoring` Skill", cli)


# ---------------------------------------------------------------------------
# Namespace synchronization
# ---------------------------------------------------------------------------

_BLOCKS = list(_all_blocks())
_BLOCK_IDS = [f"{page.name}-{i}" for page, i, _, _ in _BLOCKS]


def test_the_skill_has_code_for_the_block_checks_to_read():
    """The two checks below run once per ```python block, so a Skill whose
    fences they no longer match would pass both having checked nothing."""
    assert _BLOCKS, "no ```python block found in the Skill"


@pytest.mark.parametrize("page, i, src, is_recipe", _BLOCKS, ids=_BLOCK_IDS)
def test_skill_code_uses_only_names_the_namespace_binds(page, i, src, is_recipe):
    """The failure this prevents: a Skill that teaches a name the runtime does
    not bind, which the agent can only discover as a NameError it cannot fix."""
    # A recipe has to run, so every name must resolve. A fragment may use a
    # declared placeholder and nothing else.
    allowed = _ALLOWED_FREE if is_recipe else _ALLOWED_IN_FRAGMENTS
    unknown = sorted(n for n in _free_names(src) if n not in allowed)
    assert not unknown, f"{page.name} block {i} uses unbound names: {unknown}"


@pytest.mark.parametrize("page, i, src, is_recipe", _BLOCKS, ids=_BLOCK_IDS)
def test_skill_code_passes_the_sandbox(page, i, src, is_recipe):
    """Teaching syntax the allowlist rejects would be worse than teaching
    nothing — the agent would copy it and lose a compile to `with` or `try`."""
    SB.check_ast(src)


# Kept out of the namespace on purpose. The Skill may name them only to rule
# them out, so this checks the *code*, where naming one means offering it.
_EXCLUDED = ("BuildPart", "BuildSketch", "BuildLine",           # builder mode
             "MM", "CM", "IN", "FT",                            # unit constants
             "Text", "FontStyle",                               # fonts
             "import_step", "import_stl", "export_step",        # I/O
             "export_stl", "export_gltf", "export_brep",
             "Mesher", "ExportSVG", "ExportDXF", "export_to_pcbway",
             "RigidJoint", "RevoluteJoint", "LinearJoint",      # b123d joints
             "CylindricalJoint", "BallJoint",
             "show_object", "show", "show_topology", "pack")    # viewers/tools


@pytest.mark.parametrize("name", _EXCLUDED)
def test_excluded_names_are_absent_from_the_namespace_and_from_skill_code(name):
    assert name not in NS.CURATED, f"{name} was re-added to the namespace"
    for page, i, src, _ in _all_blocks():
        assert name not in _free_names(src), (
            f"{page.name} block {i} calls {name}, which is not in scope")


def test_the_skill_does_not_restate_what_the_prompt_owns():
    """`object.md` owns the sandbox, the timeout, read-only state, and the
    error-code definitions. Two documents stating the same rule is how they
    come to state it differently."""
    for page in _pages():
        # A line that names `object.md` is deferring to it, not restating
        # it — SKILL.md's opening paragraph lists these topics precisely to say
        # they are covered there.
        lines = [l for l in page.read_text().splitlines()
                 if "object.md" not in l]
        for phrase in ("AST allowlist", "sandbox_violation", "verifier_reports",
                       "300 s by default", "read-only", "SourceTimeout"):
            assert not any(phrase in l for l in lines), (
                f"{page.name} restates {phrase!r}; that belongs to object.md")


# ---------------------------------------------------------------------------
# The generated page, and the recipes — both need the pinned interpreter
# ---------------------------------------------------------------------------

def _run_in_cad(prog: str, *args: str) -> str:
    proc = subprocess.run([SA.b123d_python(), "-c", prog, *args],
                          capture_output=True, text=True, timeout=600,
                          env=SA.b123d_child_env())
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    return proc.stdout


_BOOTSTRAP = f"""
import importlib.util as u, sys
_repo = {str(REPO)!r}
_s = u.spec_from_file_location("dycraft", _repo + "/__init__.py",
                               submodule_search_locations=[_repo])
_m = u.module_from_spec(_s); sys.modules["dycraft"] = _m; _s.loader.exec_module(_m)
from dycraft.agent.cad import _b123d_namespace as NS
from dycraft.agent.cad import sandbox as SB
from dycraft.agent.cad import _solid_compile as SC
from dycraft.agent.cad import _namespace_doc as ND
"""


@_needs_cad
def test_the_namespace_reference_equals_a_fresh_generation():
    """Regenerate and diff. A namespace edit that skips the regeneration fails
    here, so the page cannot quietly describe a namespace that no longer
    exists — which is the whole reason it is generated rather than written."""
    fresh = _run_in_cad(_BOOTSTRAP + """
import importlib.metadata as md
sys.stdout.write(ND.render(NS.CURATED, md.version("build123d")))
""")
    committed = REFERENCE_PAGE.read_text()
    assert fresh == committed, (
        "namespace_reference.md is stale. Regenerate it:\n"
        "  cd agent/cad && LD_LIBRARY_PATH=<cad-env>/lib "
        "PYTHONPATH=. <cad-env>/bin/python _namespace_doc.py > "
        "../../.claude/skills/build123d_authoring/references/"
        "namespace_reference.md\n"
        "then copy it to .agents/skills/build123d_authoring/references/.")


def _recipes() -> list[tuple[str, str]]:
    out = []
    for page, i, src, is_recipe in _all_blocks():
        if is_recipe:
            out.append((f"{page.name}:{i}", src))
    return out


#: How many `# validated` recipes the skill must carry.
#:
#: Four, because the skill documents only failures that raise nothing AND
#: that the compiler accepts; a failure the kernel or `add_part` already
#: reports needs no recipe.
#:
#: The number is not the point of the assertion. It is a guard on the guard:
#: if the `# validated` convention ever changes, the executable test below
#: would pass by testing nothing, and this is what notices. Set it to the
#: count the skill actually has, and move it when the skill moves.
_MIN_VALIDATED_RECIPES = 5


def test_there_are_recipes_to_validate():
    """A guard on the guard: if the marker convention changes, the executable
    test below would pass by testing nothing."""
    assert len(_recipes()) >= _MIN_VALIDATED_RECIPES, (
        f"only {len(_recipes())} validated recipes found")


@_needs_cad
def test_every_validated_recipe_builds_a_registrable_solid(tmp_path):
    """The claim each `# validated` block makes: this runs, in this sandbox, in
    this namespace, and produces something `add_part` accepts.

    Run as one subprocess over all recipes — the CAD interpreter costs seconds
    to start and the recipes cost milliseconds each.
    """
    recipes = _recipes()
    payload = tmp_path / "recipes.json"
    payload.write_text(json.dumps(recipes))
    out = _run_in_cad(_BOOTSTRAP + """
import json
recipes = json.load(open(sys.argv[1]))
bad = []
for label, src in recipes:
    try:
        SB.check_ast(src)
    except BaseException as e:
        bad.append((label, "sandbox", f"{type(e).__name__}: {e}")); continue
    g = SB.build_globals(overlay=NS.build_namespace(reserved={"Object", "Material"}))
    try:
        exec(compile(src, "recipe.py", "exec"), g)
    except BaseException as e:
        bad.append((label, "exec", f"{type(e).__name__}: {e}")); continue
    shape = g.get("part")
    if shape is None:
        bad.append((label, "no-part", "recipe left no name `part`")); continue
    try:
        SC.validate_shape(shape, name=label)
    except BaseException as e:
        bad.append((label, "add_part", str(e))); continue
    print(f"OK {label} solids={len(shape.solids())} vol={shape.volume:.6g}")
print("FAILURES " + json.dumps(bad))
""", str(payload))
    failures = json.loads(out.rsplit("FAILURES ", 1)[1])
    assert not failures, "\n".join(f"{a}: [{b}] {c}" for a, b, c in failures)
    assert out.count("OK ") == len(recipes)
