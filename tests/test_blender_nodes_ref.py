"""The checked-in shader-node reference, and that it matches the real Blender.

Two tests on purpose. The cheap one runs everywhere and catches a missing or
corrupted artifact plus the three version traps the skill quotes; the expensive
one spawns `bpy` and is the only thing that can catch a Blender upgrade moving
the vocabulary underneath us.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dycraft.render import blender_nodes as BN


# ---------------------------------------------------------------------------
# Cheap: the artifact itself
# ---------------------------------------------------------------------------

def test_the_reference_ships_and_is_readable_without_bpy():
    """The CAD interpreter reads this file and has no Blender.

    `sdk/_shader` validates `nodes.new(...)` against it at author time, so the
    import must not pull `bpy` in even transitively.
    """
    ref = BN.load()
    assert ref, "render/blender_nodes.json is missing or unparseable"
    assert BN.blender_version().startswith("4.5"), BN.blender_version()
    types = BN.node_types()
    assert len(types) > 50, f"only {len(types)} node types recorded"
    for expected in ("ShaderNodeBsdfPrincipled", "ShaderNodeTexNoise",
                     "ShaderNodeTexVoronoi", "ShaderNodeTexWave",
                     "ShaderNodeValToRGB", "ShaderNodeBump",
                     "ShaderNodeTexCoord", "ShaderNodeMapping"):
        assert expected in types, expected


def test_socket_order_is_preserved_because_names_are_not_unique():
    """`ShaderNodeMix` names four socket pairs identically.

    `inputs` is a list and the index is the contract: `mix.inputs["A"]` resolves
    to the FLOAT variant at index 2 and is silently wrong under
    `data_type='RGBA'`, which wants index 6. A name-keyed dict would erase the
    one fact a caller needs, so this pins the shape as well as the content.
    """
    mix = BN.sockets("ShaderNodeMix")
    assert mix["inputs"] == ["Factor", "Factor", "A", "B", "A", "B",
                             "A", "B", "A", "B"], mix["inputs"]
    assert mix["outputs"].count("Result") == 4
    assert BN.enums("ShaderNodeMix")["data_type"] == [
        "FLOAT", "VECTOR", "RGBA", "ROTATION"]


def test_the_reference_records_the_version_traps_the_skill_quotes():
    """Three facts the agent-facing guidance depends on being true here."""
    # Removed in Blender 4.1, folded into the Noise Texture node. An agent
    # working from older tutorials reaches for it.
    assert "ShaderNodeTexMusgrave" not in BN.node_types()
    # ...and what replaced it carries the merged controls.
    noise = BN.sockets("ShaderNodeTexNoise")["inputs"]
    for socket in ("Scale", "Detail", "Roughness", "Lacunarity", "Distortion"):
        assert socket in noise, socket
    assert BN.enums("ShaderNodeTexNoise")["noise_dimensions"] == [
        "1D", "2D", "3D", "4D"]
    # Principled's post-4.0 socket names, which the harness itself binds.
    principled = BN.sockets("ShaderNodeBsdfPrincipled")["inputs"]
    for socket in ("Base Color", "Metallic", "Roughness", "Alpha", "Normal",
                   "Emission Color", "Emission Strength"):
        assert socket in principled, socket


def test_an_unknown_node_gets_a_nearest_match_rather_than_silence():
    """The repair hint `sdk/_shader` puts in front of the agent."""
    assert "ShaderNodeTexNoise" in BN.suggest("ShaderNodeTexNoize")
    assert "ShaderNodeTexNoise" in BN.suggest("shadernodetexnoise")
    # The retired node should point at something real, not nothing.
    assert BN.suggest("ShaderNodeTexMusgrave")


def test_a_missing_artifact_degrades_instead_of_raising(monkeypatch, tmp_path):
    """Losing the file must not make an object unbuildable.

    The author-time check is a convenience that moves a render-time failure
    earlier. If the artifact goes missing the check has to fall silent, not take
    the compile down.
    """
    monkeypatch.setattr(BN, "NODES_JSON", tmp_path / "absent.json")
    monkeypatch.setattr(BN, "_CACHE", None)
    assert BN.load() == {}
    assert BN.node_types() == frozenset()
    assert BN.sockets("ShaderNodeTexNoise") == {"inputs": [], "outputs": []}
    assert BN.suggest("anything") == []


# ---------------------------------------------------------------------------
# Expensive: the artifact vs. the installed bpy
# ---------------------------------------------------------------------------

_REGEN = r'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("_bn", %(module)r)
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)
print("RESULT" + json.dumps(bn.generate()))
'''


@pytest.mark.expensive
def test_the_checked_in_reference_matches_the_installed_blender(tmp_path):
    """A Blender upgrade cannot move the node vocabulary silently.

    Regenerate with `$(cat .blender_python) render/blender_nodes.py` and commit
    the result; the diff is the upgrade's effect on what an agent may author.
    """
    from dycraft.render import _blender_env as E

    interp = E.blender_python()
    if not interp:
        pytest.skip("no blender interpreter (run ./setup_blender.sh)")

    module = Path(BN.__file__).resolve()
    script = tmp_path / "regen.py"
    script.write_text(_REGEN % {"module": str(module)})
    out = subprocess.run([interp, str(script)], capture_output=True, text=True,
                         timeout=600, env=E.blender_child_env())
    assert "RESULT" in out.stdout, f"regen failed:\n{out.stdout}\n{out.stderr}"
    fresh = json.loads(out.stdout.split("RESULT", 1)[1].splitlines()[0])

    stored = json.loads(BN.NODES_JSON.read_text(encoding="utf-8"))
    assert fresh["blender_version"] == stored["blender_version"], (
        f"Blender moved {stored['blender_version']} -> "
        f"{fresh['blender_version']}; regenerate render/blender_nodes.json")

    fresh_nodes, stored_nodes = fresh["nodes"], stored["nodes"]
    added = sorted(set(fresh_nodes) - set(stored_nodes))
    gone = sorted(set(stored_nodes) - set(fresh_nodes))
    assert not added and not gone, (
        f"node types added={added} removed={gone}; regenerate the reference")
    drifted = [k for k in sorted(stored_nodes)
               if fresh_nodes[k] != stored_nodes[k]]
    assert not drifted, f"sockets/enums drifted on {drifted[:5]}"
