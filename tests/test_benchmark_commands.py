"""The benchmark commands' launch settings, without launching an agent."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from dycraft.benchmark import bare as A
from dycraft.benchmark import briefs as B
from dycraft.benchmark import stage as S
from dycraft.benchmark import report as R
from dycraft.benchmark import run as D


def _args(**kw):
    base = dict(agent="codex", model="gpt-6-astra", effort="medium", cli_arg=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_dycraft_arm_is_one_animation_run(tmp_path):
    cmd = D.command(tmp_path, tmp_path / "p.md", _args(), 2.0)
    assert cmd[1:3] == ["-m", "dycraft.cli"]
    pairs = dict(zip(cmd[3::2], cmd[4::2]))
    assert pairs["--layer"] == "animation"
    assert pairs["--backend"] == "codex"
    assert pairs["--on-rewrite-failure"] == "abort"
    assert pairs["--advisory-hours"] == "2"
    assert pairs["--max-hours"] == f"{2 + 20 / 60:g}"
    assert pairs["--reasoning-effort"] == "medium"
    assert pairs["--run-id"] == "dycraft_codex"


def test_the_launcher_stops_at_the_budget_and_the_runner_backs_it_up():
    """The launcher's --max-hours is the budget plus 20 minutes; the runner's own stop comes 15 minutes after it,
    plus the rewriters, which run before the launcher's clock starts; with no budget, after the launcher's default."""
    from dycraft.cli.cli import DEFAULT_MAX_HOURS
    assert D.max_hours(1) == 1 + 20 / 60 and D.max_hours(None) is None and D.max_hours(0) is None
    assert D.backstop_seconds(1) == 3600 + 20 * 60 + 15 * 60 + 1800
    assert D.backstop_seconds(0) == DEFAULT_MAX_HOURS * 3600 + 15 * 60 + 1800
    assert "--max-hours" not in D.command(Path("x"), Path("p.md"), _args(), None)


def test_each_brief_takes_its_sets_budget_unless_one_is_given():
    """Short cuts get 8 hours, medium cuts and long briefs 24, a brief of your own 8; --budget-hours overrides
    every brief, and 0 means no budget."""
    briefs = B.load()
    assert D.budget_for(briefs["B01S"], "short", None) == 8
    assert D.budget_for(briefs["B13"], "short", None) == 8          # a set's full brief takes the set's budget
    assert D.budget_for(briefs["B01M"], "medium", None) == 24
    assert D.budget_for(briefs["B01"], "long", None) == 24
    assert D.budget_for(briefs["B01S"], None, None) == 8 and D.budget_for(briefs["B13"], None, None) == 24
    assert D.budget_for(B.adhoc("a dog fetches a stick"), None, None) == 8
    assert D.budget_for(briefs["B01"], "long", 3) == 3 and D.budget_for(briefs["B01"], "long", 0) is None


def test_without_a_budget_each_brief_runs_with_its_sets_budget_end_to_end(tmp_path, capsys, monkeypatch):
    """The per-brief default reaches everything that carries it: the paragraph the agent reads, the advisory
    countdown, the launcher's hard limit and the runner's own backstop."""
    assert D.main(["--briefs", "medium", "--agent", "claude", "--out", str(tmp_path / "m"), "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    medium = next(line for line in lines if "b01m_" in line)
    whole = next(line for line in lines if "b17_" in line)         # a full brief in the medium set: 24 too
    for line in (medium, whole):
        assert "--advisory-hours 24" in line and "--max-hours 24.3333" in line
    assert "budget of 24 hours" in (tmp_path / "m" / "b01m_g1_scrambles_eggs" / "prompt_dycraft_claude.md").read_text()
    assert D.main(["--briefs", "short", "--agent", "claude", "--out", str(tmp_path / "s"), "--dry-run"]) == 0
    short = next(line for line in capsys.readouterr().out.splitlines() if "b17_" in line)
    assert "--advisory-hours 8" in short and "--max-hours 8.33333" in short
    assert D.main(["--briefs", "B17", "--agent", "claude", "--out", str(tmp_path / "z"), "--budget-hours", "0",
                   "--dry-run"]) == 0
    none = capsys.readouterr().out
    assert "--advisory-hours 0" in none and "--max-hours" not in none
    assert "budget of" not in (tmp_path / "z" / "b17_marble_course_runs" / "prompt_dycraft_claude.md").read_text()

    waited = []

    class Launch:
        def __init__(self, cmd, **kw):
            self.cmd = cmd

        def wait(self, timeout=None):
            waited.append((self.cmd, timeout))
            return 0
    monkeypatch.setattr(D.subprocess, "Popen", Launch)
    assert D.main(["--briefs", "B01M", "--agent", "claude", "--out", str(tmp_path / "r")]) == 0
    (cmd, timeout), = waited
    assert "--max-hours" in cmd and timeout == D.backstop_seconds(24.0)


@pytest.mark.parametrize("spec, ids", [("short", ["B13", "B14", "B16"]), ("medium", ["B13", "B14", "B15", "B16"])])
def test_the_dycraft_arm_takes_the_library_variants_in_the_short_and_medium_sets(tmp_path, capsys, spec, ids):
    """The DyCraft command opts in to the asset library: where a set has no cut, its prompt is the imported-character
    variant of the brief, not the brief as written."""
    assert D.main(["--briefs", spec, "--agent", "claude", "--out", str(tmp_path), "--dry-run"]) == 0
    variants, written = B.load(imported_characters=True), B.load()
    for bid in ids:
        prompt = (tmp_path / variants[bid].slug / "prompt_dycraft_claude.md").read_text()
        assert prompt.startswith(variants[bid].text) and variants[bid].text != written[bid].text, bid


def test_a_negative_budget_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "blender_prefix", lambda: tmp_path)
    with pytest.raises(SystemExit, match="--budget-hours"):
        D.main(["--briefs", "B17", "--agent", "codex", "--out", str(tmp_path), "--budget-hours", "-1", "--dry-run"])
    with pytest.raises(SystemExit, match="--budget-hours"):
        A.main(["--briefs", "B17", "--agent", "codex", "--out", str(tmp_path), "--budget-hours", "-1", "--dry-run"])


def test_a_dry_run_writes_the_prompts_and_runs_nothing(tmp_path, capsys):
    assert D.main(["--briefs", "B01S,B13", "--agent", "claude", "--out", str(tmp_path),
                   "--budget-hours", "3", "--dry-run"]) == 0
    p = (tmp_path / "b01s_g1_scrambles_eggs" / "prompt_dycraft_claude.md").read_text()
    assert "about 60–75 seconds" in p and "budget of 3 hours" in p
    assert "dycraft.cli" in capsys.readouterr().out
    assert not (tmp_path / "b01s_g1_scrambles_eggs" / "dycraft_claude").exists()


def test_a_medium_set_tells_its_full_briefs_the_medium_length(tmp_path):
    D.main(["--briefs", "medium", "--agent", "codex", "--out", str(tmp_path), "--dry-run"])
    assert "about 120–150 seconds" in (
        tmp_path / "b13_dog_fetches_newspaper" / "prompt_dycraft_codex.md").read_text()


def test_codex_bare_runs_with_the_papers_flags():
    cmd = A.agent_command("codex", A.Path("/opt/codex"), "gpt-6-astra", "medium")
    assert cmd[1] == "exec" and cmd[-1] == "-"
    joined = " ".join(cmd)
    for flag in ("--disable multi_agent", "--disable skill_search", "--disable plugins",
                 "--ignore-user-config", 'web_search="disabled"', 'model_reasoning_effort="medium"'):
        assert flag in joined
    # Approvals as in DyCraft's own Codex launch, not the paper's bypass.
    assert "--approve-for-me" in cmd and "--dangerously-bypass-approvals-and-sandbox" not in cmd


def test_claude_bare_runs_headless_without_sub_agents_or_web():
    cmd = A.agent_command("claude", A.Path("/opt/claude"), "claude-opus-5-5", "high")
    assert "-p" in cmd and "--disallowedTools" in cmd
    i = cmd.index("--disallowedTools")
    assert {"Agent", "Task", "WebSearch", "WebFetch"} <= set(cmd[i + 1:i + 5])


def test_the_bare_sandbox_writes_only_the_workspace(tmp_path):
    for n in ("workspace", "empty", "inputs/g1"):
        (tmp_path / n).mkdir(parents=True)
    cmd = A.sandbox(tmp_path, tmp_path / "home", tmp_path / "blender", [], g1=True)
    binds = [cmd[i + 2] for i, c in enumerate(cmd) if c == "--bind"]
    assert sorted(binds) == ["/home/agent", "/workspace"]
    assert "/assets/g1" in cmd and "--clearenv" in cmd and "--unshare-pid" in cmd
    assert "/assets/g1" not in A.sandbox(tmp_path, tmp_path / "home", tmp_path / "blender", [],
                                         g1=False)


def test_a_missing_g1_description_names_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "G1_SOURCE", tmp_path / "g1_description")
    with pytest.raises(SystemExit, match="not in git"):
        A.copy_g1(tmp_path / "dest")
    (tmp_path / "g1_description").mkdir()
    (tmp_path / "g1_description" / B.G1_URDF).write_text(
        '<robot><link><visual><geometry><mesh filename="meshes/a.STL"/></geometry></visual>'
        '</link></robot>')
    with pytest.raises(SystemExit, match="incomplete"):
        A.copy_g1(tmp_path / "dest")
    (tmp_path / "g1_description" / "meshes").mkdir()
    (tmp_path / "g1_description" / "meshes" / "a.STL").write_text("solid a")
    A.copy_g1(tmp_path / "dest")
    assert (tmp_path / "dest" / "meshes" / "a.STL").is_file()


def test_the_bare_command_refuses_imported_characters():
    with pytest.raises(SystemExit):
        A.main(["--briefs", "B13", "--agent", "codex", "--out", "x", "--imported-characters"])


def test_the_report_puts_both_arms_side_by_side(tmp_path):
    for arm, extra in (("dycraft_codex", {"outcome": "committed", "unmet": 1,
                                          "film": str(tmp_path / "f.mp4")}),
                       ("bare_codex", {"blend": "x.blend", "hard_stop": True, "staged": True,
                                       "film_source": "harness", "seconds": 13.0})):
        d = tmp_path / "b13_dog_fetches_newspaper" / arm / "benchmark"
        d.mkdir(parents=True)
        (d / "record.json").write_text(json.dumps({
            "slug": "b13_dog_fetches_newspaper", "arm": arm, "model": "m", "wall_seconds": 7200,
            **extra}))
    rows = R.collect(tmp_path)
    table = R.text_table(rows)
    assert "committed (1 unmet)" in table and "delivered, hard stop" in table
    assert "harness" in table and "13.0" in table and "mean over scored briefs" not in table
    assert R.main([str(tmp_path)]) == 0
    page = (tmp_path / "report.html").read_text()
    assert "<video" in page and "no film" in page


def test_the_report_shows_each_arms_scores_and_their_means(tmp_path):
    """`benchmark.score` writes each arm's rates beside its record; the report shows them, and each arm's mean over
    the briefs it was scored on, as the paper's Table 2 does."""
    for brief, rates in (("b13_dog_fetches_newspaper", (0.25, 0.5)), ("b17_marble_course_runs", (0.75, 0.0))):
        d = tmp_path / brief / "bare_codex" / "benchmark"
        d.mkdir(parents=True)
        (d / "record.json").write_text(json.dumps({"slug": brief, "arm": "bare_codex", "model": "m",
                                                   "blend": "x.blend"}))
        (d / "score.json").write_text(json.dumps({"penetration_rate": rates[0], "floating_rate": rates[1]}))
    table = R.text_table(R.collect(tmp_path))
    assert "25.0" in table and "75.0" in table
    assert "50.0 (2 briefs)" in table and "25.0 (2 briefs)" in table      # means: penetration, floating


def test_a_bare_budget_is_told_and_sets_the_hard_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "blender_prefix", lambda: tmp_path)
    assert A.main(["--briefs", "B13", "--agent", "codex", "--out", str(tmp_path),
                   "--budget-hours", "2", "--film-seconds", "default", "--dry-run"]) == 0
    root = tmp_path / "b13_dog_fetches_newspaper" / "bare_codex"
    prompt = (root / "workspace" / "prompt.md").read_text()
    assert "budget of 2 hours" in prompt and "about 60–75 seconds" in prompt
    config = json.loads((root / "control" / "config.json").read_text())
    assert config["hours"] == pytest.approx(2 + 20 / 60) and config["paper_prompt"] is False


def test_without_flags_the_bare_arm_keeps_the_papers_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "blender_prefix", lambda: tmp_path)
    A.main(["--briefs", "B17", "--agent", "codex", "--out", str(tmp_path), "--dry-run"])
    root = tmp_path / "b17_marble_course_runs" / "bare_codex"
    assert (root / "workspace" / "prompt.md").read_text() == B.bare_prompt(B.load()["B17"])
    config = json.loads((root / "control" / "config.json").read_text())
    assert config["hours"] == A.DEFAULT_HOURS and config["paper_prompt"] is True
    assert config["blender"] == B.PAPER_BLENDER
    assert config["effort"] == "xhigh"  # DyCraft's default, so both arms run alike


@pytest.mark.parametrize("spec", ["short", "medium"])
def test_the_bare_command_gives_every_set_member_a_prompt_without_the_library(tmp_path, monkeypatch, spec):
    """Through the bare command itself: every brief of the short and medium sets gets the bare form of the cut or the
    brief as written, never an imported-character variant, and no prompt tells the agent to use an asset library."""
    monkeypatch.setattr(A, "blender_prefix", lambda: tmp_path)
    monkeypatch.setattr(A, "copy_g1", lambda dest: None)
    assert A.main(["--briefs", spec, "--agent", "codex", "--out", str(tmp_path / "out"), "--dry-run"]) == 0
    chosen, _ = B.select(spec)
    assert len(chosen) == 24
    for brief in chosen:
        prompt = (tmp_path / "out" / brief.slug / "bare_codex" / "workspace" / "prompt.md").read_text()
        assert prompt == B.bare_prompt(brief) and "asset library" not in prompt, brief.id


def test_the_bare_prompt_names_the_installed_blender(tmp_path, monkeypatch):
    """The environment sentence names the Blender the sandbox has; only on the paper's
    version is the prompt the paper's."""
    monkeypatch.setattr(A, "blender_prefix", lambda: tmp_path)
    monkeypatch.setattr(A, "installed_blender", lambda prefix: "4.5.15")
    A.main(["--briefs", "B17", "--agent", "codex", "--out", str(tmp_path), "--dry-run"])
    root = tmp_path / "b17_marble_course_runs" / "bare_codex"
    prompt = (root / "workspace" / "prompt.md").read_text()
    assert "python has Blender 4.5.15 bpy available" in prompt and "4.5.13" not in prompt
    assert prompt == B.bare_prompt(B.load()["B17"], blender="4.5.15")
    config = json.loads((root / "control" / "config.json").read_text())
    assert config["blender"] == "4.5.15" and config["paper_prompt"] is False


def test_the_installed_blender_is_read_from_bpy(tmp_path, monkeypatch):
    from dycraft.render import _blender_env
    monkeypatch.setattr(_blender_env, "probe_blender_version", lambda interp: "4.5.15 LTS")
    assert A.installed_blender(tmp_path) == "4.5.15"
    monkeypatch.setattr(_blender_env, "probe_blender_version", lambda interp: None)
    assert A.installed_blender(tmp_path) == B.PAPER_BLENDER


@pytest.mark.parametrize("prefix", [[], ["/usr/bin/unshare", "-Ur"]])
def test_bwrap_makes_the_user_namespace_where_unshare_may_not(tmp_path, monkeypatch, prefix):
    """Stock Ubuntu 24.04 lets only bwrap create user namespaces; the network stays shared."""
    monkeypatch.setattr(A, "userns_prefix", lambda: prefix)
    cmd = A.sandbox(tmp_path, tmp_path / "home", tmp_path / "blender", [], g1=False)
    assert ("--unshare-user" in cmd) == (not prefix)
    assert cmd[:len(prefix)] == prefix and "--unshare-net" not in cmd


def test_both_arms_take_a_prompt_of_your_own(tmp_path, monkeypatch):
    text = "a humanoid robot is initially lying on the ground, but gets up into a standing position"
    assert D.main(["--prompt", text, "--agent", "claude", "--out", str(tmp_path), "--dry-run"]) == 0
    slug = B.adhoc(text).slug
    assert text in (tmp_path / slug / "prompt_dycraft_claude.md").read_text()
    monkeypatch.setattr(A, "blender_prefix", lambda: tmp_path)
    A.main(["--prompt", text, "--agent", "codex", "--out", str(tmp_path), "--dry-run"])
    assert (tmp_path / slug / "bare_codex" / "workspace" / "prompt.md").read_text() == B.bare_prompt(B.adhoc(text))


def test_npms_codex_launcher_is_replaced_by_the_native_binary_it_ships(tmp_path):
    package = tmp_path / "node_modules" / "@openai" / "codex"
    launcher = package / "bin" / "codex.js"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env node\n")
    native = package / "node_modules" / "@openai" / "codex-linux-x64" / "vendor" / "x86_64-unknown-linux-musl" / "bin" / "codex"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"\x7fELF")
    native.chmod(0o755)
    assert A.codex_native(launcher) == native.resolve()
    assert A.codex_native(native) is None
    assert A.agent_binary("codex", str(launcher)) == native.resolve()


def test_claude_may_bypass_permissions_as_the_sandboxs_root(tmp_path):
    for n in ("workspace", "empty"):
        (tmp_path / n).mkdir(parents=True)
    cmd = A.sandbox(tmp_path, tmp_path / "home", tmp_path / "blender", [], g1=False)
    env = {cmd[i + 1]: cmd[i + 2] for i, c in enumerate(cmd) if c == "--setenv"}
    assert env["IS_SANDBOX"] == "1"


_TINY_BLEND = r'''
import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start, scene.frame_end, scene.render.fps = 1, 2, 24
bpy.ops.mesh.primitive_cube_add(size=0.5)
cube = bpy.context.object
cube.location = (0, 0, 0.25); cube.keyframe_insert("location", frame=1)
cube.location = (0.2, 0, 0.25); cube.keyframe_insert("location", frame=2)
bpy.ops.object.camera_add(location=(3, -3, 2), rotation=(1.1, 0, 0.78))
scene.camera = bpy.context.object
bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1])
'''


#: What a DyCraft run's blend carries that a bare agent's does not.
_DECLARED = _TINY_BLEND.replace("bpy.ops.wm.save_as_mainfile", """import json
scene["dycraft_scene_ref"] = "scene-1"
scene["dycraft_declarations"] = json.dumps({
    "events": [{"name": "push", "start": 0, "end": 1}],
    "collision_allowances": [{"a": "Cube", "b": "floor"}], "units": "frames"})
bpy.ops.wm.save_as_mainfile""")


def _tiny_blend(tmp_path, blend=None, source=_TINY_BLEND):
    from dycraft.render._blender_env import blender_python
    py = blender_python()
    if not py:
        pytest.skip("no Blender interpreter configured")
    blend = blend or tmp_path / "agent" / "final.blend"
    blend.parent.mkdir(parents=True)
    script = tmp_path / "make.py"
    script.write_text(source)
    import subprocess
    subprocess.run([py, "-B", str(script), str(blend)], check=True, capture_output=True, timeout=300)
    return blend


def test_a_bare_bundle_carries_the_agents_film_or_one_rendered_for_it(tmp_path):
    from dycraft.animation.blender.bundle import validate_manifest
    blend = _tiny_blend(tmp_path)
    own = tmp_path / "agent" / "final.mp4"
    own.write_bytes(b"the agent's film")
    out = tmp_path / "with_film"
    record = S.stage(blend, out, scene_ref="t:with", film=own, render_missing_film=True)
    ir = json.loads((out / "ir.json").read_text())
    assert record["film"]["source"] == "agent" and ir["film"] == "bundle/film/clip.mp4"
    assert (out / ir["film"]).read_bytes() == b"the agent's film"
    assert not (out / "measurements.json").exists()  # staged for the viewer, not measured
    validate_manifest(out, ir)
    out = tmp_path / "rendered"
    record = S.stage(blend, out, scene_ref="t:none", render_missing_film=True)
    ir = json.loads((out / "ir.json").read_text())
    assert record["film"]["source"] == "harness", record["film"]
    assert record["film"]["frames"] == 2 and record["film"]["settings"]["width"] == 960
    assert ir["film"] == "bundle/film/clip.mp4" and (out / ir["film"]).stat().st_size > 0
    validate_manifest(out, ir)


def test_a_dycraft_runs_evaluation_is_rebuilt_from_its_blend_in_place(tmp_path, capsys):
    from dycraft.animation.blender.bundle import validate_manifest
    run = tmp_path / "b99_film"
    _tiny_blend(tmp_path, run / "bundle" / "animation.blend", _DECLARED)
    (run / "bundle" / "evaluation").mkdir()
    (run / "bundle" / "evaluation" / "stale.json").write_text("{}")
    (run / "bundle" / "design_state").mkdir()
    (run / "bundle" / "design_state" / "baseline.json").write_text('{"kept": true}')
    (run / "film.mp4").write_bytes(b"the run's film")
    assert S.main([str(run)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["manifest"] == "written" and printed["meta_written"] and printed["frames"] == 2
    index = json.loads((run / "bundle" / "evaluation" / "index.json").read_text())
    assert index["declarations"]["collision_allowances"] == [{"a": "Cube", "b": "floor"}]
    assert not (run / "bundle" / "evaluation" / "stale.json").exists()
    assert (run / "bundle" / "design_state" / "baseline.json").read_text() == '{"kept": true}'
    ir = json.loads((run / "ir.json").read_text())
    assert ir["scene_ref"] == "scene-1" and ir["events"] == [{"name": "push", "start": 0, "end": 1}]
    assert ir["film"] == "bundle/film/clip.mp4" and (run / ir["film"]).read_bytes() == b"the run's film"
    validate_manifest(run, ir)
    assert json.loads((run / "meta.json").read_text()) == {"run_id": "b99_film", "layer": "animation",
                                                         "outcome": "delivered"}
    # A run that travels with its manifest and meta.json keeps what it recorded.
    (run / "ir.json").write_text(json.dumps({**ir, "outcome": "committed"}))
    (run / "meta.json").write_text('{"run_id": "recorded"}')
    shutil.rmtree(run / "bundle" / "evaluation")
    record = S.rebuild(run)
    ir = json.loads((run / "ir.json").read_text())
    assert record["manifest"] == "updated" and not record["meta_written"]
    assert ir["outcome"] == "committed" and ir["scene_ref"] == "scene-1"
    validate_manifest(run, ir)
    assert (run / "meta.json").read_text() == '{"run_id": "recorded"}'


#: A declared blend with a texture saved beside it, which the dependency audit must find inside the bundle.
_TEXTURED = _DECLARED + """
img = bpy.data.images.new("tex", 4, 4)
img.filepath_raw = "//tex.png"
img.file_format = "PNG"
img.save()
material = bpy.data.materials.new("textured")
material.use_nodes = True
material.node_tree.nodes.new("ShaderNodeTexImage").image = img
cube.data.materials.append(material)
bpy.ops.wm.save_mainfile()
"""


def _declared_run(tmp_path, name="b99_film", source=_DECLARED):
    run = tmp_path / name
    _tiny_blend(tmp_path, run / "bundle" / "animation.blend", source)
    (run / "bundle" / "design_state").mkdir()
    (run / "bundle" / "design_state" / "baseline.json").write_text("{}")
    return run


def _snapshot(run):
    index = run / "bundle" / "evaluation" / "index.json"
    return index.read_bytes(), index.stat().st_mtime_ns, (run / "ir.json").read_bytes()


def test_a_film_is_prepared_for_viewing_beside_it_and_never_in_it(tmp_path):
    """`prepare` writes what `rebuild` would into `dest`, and only there: a film installed from the published set links
    into the dataset mirror, which must stay as it was downloaded. The manifest covers both folders, each file
    hashed where it is."""
    from dycraft.animation.blender.bundle import file_hash
    film = _declared_run(tmp_path, "B99_film")
    (film / "film.mp4").write_bytes(b"the film")
    (film / "run.json").write_text('{"brief": "a brief"}')
    before = sorted((p.relative_to(film).as_posix(), p.stat().st_mtime_ns) for p in film.rglob("*"))
    dest = tmp_path / "prepared"
    record = S.prepare(film, dest)
    assert sorted((p.relative_to(film).as_posix(), p.stat().st_mtime_ns) for p in film.rglob("*")) == before
    assert record["frames"] == 2 and record["film"] == "film.mp4"
    ir = json.loads((dest / "ir.json").read_text())
    assert ir["film"] == "film.mp4" and ir["layer"] == "animation" and ir["scene_ref"] == "scene-1"
    assert sorted(p.name for p in dest.iterdir()) == ["bundle", "ir.json"], "scratch files were left behind"
    for rel, digest in ir["files"].items():
        where = dest if rel.startswith("bundle/evaluation/") else film
        assert file_hash(where / rel) == digest, rel
    assert "bundle/animation.blend" in ir["files"] and ir["evaluation"] in ir["files"]


def test_a_rebuild_that_cannot_finish_leaves_the_run_as_it_was(tmp_path):
    run = _declared_run(tmp_path)
    S.rebuild(run)
    ir = json.loads((run / "ir.json").read_text())
    before = _snapshot(run)
    # Refused: the blend is not the film the manifest describes.
    (run / "ir.json").write_text(json.dumps({**ir, "n_frames": 3}))
    before = _snapshot(run)
    with pytest.raises(ValueError, match="n_frames"):
        S.rebuild(run)
    assert _snapshot(run) == before
    # Fails after evaluating: a symlink in the bundle makes the manifest impossible.
    (run / "ir.json").write_text(json.dumps(ir))
    (run / "bundle" / "link.json").symlink_to(run / "bundle" / "design_state" / "baseline.json")
    before = _snapshot(run)
    with pytest.raises(ValueError, match="symlink"):
        S.rebuild(run)
    assert _snapshot(run) == before
    (run / "bundle" / "link.json").unlink()
    # A recorded manifest without an `evaluation` entry, as the paper's B01 and B06 films have, gets one.
    (run / "ir.json").write_text(json.dumps({k: v for k, v in ir.items() if k != "evaluation"}))
    S.rebuild(run)
    rebuilt = json.loads((run / "ir.json").read_text())
    assert rebuilt["evaluation"] == "bundle/evaluation/index.json"
    from dycraft.animation.blender.bundle import validate_manifest
    validate_manifest(run, rebuilt)
    # A run without a design baseline (a bare agent's staged arm) is not relabelled as a DyCraft run.
    (run / "bundle" / "design_state" / "baseline.json").unlink()
    before = _snapshot(run)
    with pytest.raises(ValueError, match="not a DyCraft run"):
        S.rebuild(run)
    assert _snapshot(run) == before
    assert not list(run.glob(".rebuild_*"))



def test_a_killed_rebuild_is_undone_and_the_film_may_name_the_clip_itself(tmp_path):
    run = _declared_run(tmp_path)
    (run / "film.mp4").write_bytes(b"the run's film")
    S.rebuild(run)
    clip = run / "bundle" / "film" / "clip.mp4"
    S.rebuild(run, film=clip)
    assert clip.read_bytes() == b"the run's film"
    # Killed between its two renames: the previous evaluation sits in the leftover folder and the bundle has none.
    # The next rebuild puts it back before anything else, so even refused, it leaves the run as it was.
    index = (run / "bundle" / "evaluation" / "index.json").read_bytes()
    stale = run / ".rebuild_killed"
    stale.mkdir()
    (run / "bundle" / "evaluation").rename(stale / "previous_evaluation")
    ir = json.loads((run / "ir.json").read_text())
    (run / "ir.json").write_text(json.dumps({**ir, "n_frames": 3}))
    with pytest.raises(ValueError, match="n_frames"):
        S.rebuild(run)
    assert (run / "bundle" / "evaluation" / "index.json").read_bytes() == index
    assert not list(run.glob(".rebuild_*"))

def test_a_relative_run_dir_finds_the_files_beside_its_blend(tmp_path, monkeypatch):
    run = _declared_run(tmp_path, "textured", _TEXTURED)
    assert (run / "bundle" / "tex.png").is_file()
    monkeypatch.chdir(tmp_path)
    record = S.rebuild(Path("textured"))
    assert record["frames"] == 2
    assert "bundle/tex.png" in json.loads((run / "ir.json").read_text())["files"]
