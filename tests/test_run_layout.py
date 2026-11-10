"""`run_layout`: finding a run's folders, writing its top level, and reopening a delivered run for a revision.

`reopen_delivery` is what lets `--resume --prompt` revise a delivered film: it must move every part of the delivery
that would block a second `finish()` or a second viewer staging, keep each earlier delivery, and leave the run
untouched when it cannot finish the move.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dycraft import run_layout as L


def _w(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _animation_run(root: Path, outcome="committed") -> Path:
    _w(root / "meta.json", json.dumps({"layer": "unified_animation", "outcome": outcome}))
    _w(root / "ir.json", json.dumps({"outcome": outcome, "film": "bundle/film/clip.mp4"}))
    _w(root / "bundle" / "animation.blend")
    _w(root / "bundle" / "film" / "clip.mp4")
    _w(root / "finish_result.json", "{}")
    _w(root / "scene.json", "{}")
    _w(root / "catalog" / "manifest.json", "{}")
    _w(root / "native" / "work" / "animation.blend")
    _w(root / "design" / "scene.py")
    return root


def test_the_internal_and_reader_folders_of_both_layouts(tmp_path):
    flat, reader = tmp_path / "flat", tmp_path / "reader"
    _w(flat / "meta.json", "{}")
    _w(reader / "src" / "meta.json", "{}")
    assert L.internal_dir(flat) == flat
    assert L.internal_dir(reader) == reader / "src"
    assert L.reader_dir(reader / "src") == reader and L.reader_dir(reader) == reader
    assert L.reader_dir(flat) == flat


def test_a_film_delivery_moves_whole_and_the_work_stays(tmp_path):
    """ir.json and bundle/ block a second finish(); scene.json and catalog/ would block staging a changed scene."""
    run = _animation_run(tmp_path / "run")
    got = L.reopen_delivery(run)
    assert got["prior"] == "prior_deliveries/1" and got["outcome"] == "committed"
    prior = run / "prior_deliveries" / "1"
    for rel in ("ir.json", "bundle/animation.blend", "bundle/film/clip.mp4", "finish_result.json", "scene.json",
                "catalog/manifest.json", "meta.json"):
        assert (prior / rel).is_file(), rel
    for rel in ("ir.json", "bundle", "finish_result.json", "scene.json", "catalog"):
        assert not (run / rel).exists(), rel
    assert (run / "meta.json").is_file(), "meta.json is copied, not moved"
    assert (run / "native" / "work" / "animation.blend").is_file() and (run / "design" / "scene.py").is_file()
    assert "prior_deliveries/1/" in got["note"] and "outcome: committed" in got["note"]


def test_each_revision_keeps_the_earlier_deliveries(tmp_path):
    run = _animation_run(tmp_path / "run")
    assert L.reopen_delivery(run)["prior"] == "prior_deliveries/1"
    _animation_run(run, outcome="delivered_with_exceptions")
    second = L.reopen_delivery(run)
    assert second["prior"] == "prior_deliveries/2" and second["outcome"] == "delivered_with_exceptions"
    assert (run / "prior_deliveries" / "1" / "ir.json").is_file()


@pytest.mark.parametrize("layer", ["object", "scene"])
def test_an_object_or_scene_delivery_moves_and_its_assets_stay(tmp_path, layer):
    run = tmp_path / "run"
    _w(run / "meta.json", json.dumps({"layer": layer, "outcome": "committed"}))
    _w(run / "ir.json", "{}")
    _w(run / "final_render" / "0_front.png")
    _w(run / "renders" / "ir.urdf")
    _w(run / "assets" / "meshes" / "a.obj")
    got = L.reopen_delivery(run)
    prior = run / got["prior"]
    assert (prior / "ir.json").is_file() and (prior / "final_render" / "0_front.png").is_file()
    assert (prior / "renders" / "ir.urdf").is_file() and not (run / "ir.json").exists()
    assert (run / "assets" / "meshes" / "a.obj").is_file(), "content-addressed assets are shared, not moved"
    assert f"the {layer} can change" in got["note"]


def test_a_plain_answer_moves(tmp_path):
    run = tmp_path / "run"
    _w(run / "meta.json", json.dumps({"layer": "plain"}))
    _w(run / "result.json", '{"text": "hi"}')
    got = L.reopen_delivery(run)
    assert (run / got["prior"] / "result.json").is_file() and got["note"].endswith("prior_deliveries/1/result.json.")


def test_a_run_that_has_not_delivered_is_left_alone(tmp_path):
    run = tmp_path / "run"
    _w(run / "meta.json", json.dumps({"layer": "animation"}))
    _w(run / "native" / "work" / "animation.blend")
    assert L.reopen_delivery(run) is None
    assert not (run / "prior_deliveries").exists()


def test_a_move_that_fails_part_way_leaves_the_run_as_it_was(tmp_path, monkeypatch):
    run = _animation_run(tmp_path / "run")
    before = sorted(str(p.relative_to(run)) for p in run.rglob("*"))
    real = Path.rename

    def flaky(self, target):
        if self.name == "finish_result.json":
            raise OSError("disk full")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(RuntimeError, match="could not move the delivery"):
        L.reopen_delivery(run)
    monkeypatch.setattr(Path, "rename", real)
    assert sorted(str(p.relative_to(run)) for p in run.rglob("*")) == before


# ---------------------------------------------------------------------------------------------------- the top level


@pytest.fixture
def no_blender(monkeypatch):
    """The top-level blend as a plain hardlink: these tests' blends are not real Blender files."""
    monkeypatch.setattr("dycraft.render._blender_env.blender_python", lambda: None)


def _film(top: Path, outcome="committed", unmet=()) -> Path:
    inner = top / "src"
    _w(inner / "meta.json", json.dumps({"layer": "unified_animation", "outcome": outcome, "runtime": "claude",
                                        "model": "claude-opus-5-5", "reasoning_effort": "high",
                                        "started_at": "2026-11-09T10:02:00+00:00",
                                        "finished_at": "2026-11-09T14:41:00+00:00",
                                        "prompt": "A clock strikes four.\n\nMore of the brief."}))
    _w(inner / "ir.json", json.dumps({"outcome": outcome, "film": "bundle/film/clip.mp4",
                                      "blend": "bundle/animation.blend", "n_frames": 768, "fps": 24.0,
                                      "duration": 32.0, "blender_version": "4.5.13 LTS",
                                      "render": {"width": 1920, "height": 1080, "resolution_percentage": 100},
                                      "criteria": {"hammer": {"verdict": "pass", "evidence": "four | blows"}},
                                      "unmet": list(unmet)}))
    _w(inner / "bundle" / "animation.blend", "blend")
    _w(inner / "bundle" / "film" / "clip.mp4", "film")
    _w(inner / "bundle" / "notes" / "EDITING.md", "edit")
    _w(inner / "bundle" / "notes" / "inspection" / "review.md", "look")
    return inner


def test_new_runs_go_in_src_and_a_flat_run_stays_flat(tmp_path):
    inner = L.new_run(tmp_path, "r1")
    assert inner == (tmp_path / "r1" / "src").resolve() and inner.is_dir()
    assert list((tmp_path / "r1").iterdir()) == [tmp_path / "r1" / "src"], "nothing else is made before a commit"
    _w(tmp_path / "old" / "meta.json", "{}")
    assert L.new_run(tmp_path, "old") == (tmp_path / "old").resolve()
    assert L.top_of(inner) == inner.parent, "the top is known before meta.json exists"
    assert L.top_of(tmp_path / "old") == tmp_path / "old"


def test_a_delivered_film_is_at_the_top_without_taking_space(tmp_path, no_blender):
    top = tmp_path / "runs" / "r1"
    inner = _film(top)
    got = L.publish(inner)
    assert "errors" not in got
    assert os.path.samefile(top / "film.mp4", inner / "bundle" / "film" / "clip.mp4")
    assert os.path.samefile(top / "animation.blend", inner / "bundle" / "animation.blend")
    assert (top / "notes" / "EDITING.md").read_text() == "edit"
    assert (top / "notes" / "inspection" / "review.md").is_file()
    readme = (top / "README.md").read_text()
    assert readme.startswith("# A clock strikes four\n")
    assert "The film was delivered. It is `film.mp4` (32.0 s, 768 frames at 24 fps, 1920×1080), and " \
           "`animation.blend` is the Blender 4.5 project that renders it." in readme
    assert "| hammer | pass | four \\| blows |" in readme
    assert "Nothing: every frame, state, and check is clean." in readme
    assert "[notes/EDITING.md](notes/EDITING.md)" in readme
    assert f"--resume {top}" in readme
    assert sorted(p.name for p in top.iterdir()) == ["README.md", "animation.blend", "film.mp4", "notes", "src"]


def test_publishing_again_leaves_nothing_behind(tmp_path, no_blender):
    top = tmp_path / "r1"
    inner = _film(top)
    L.publish(inner)
    L.publish(inner)
    assert not [p.name for p in top.iterdir() if p.name.startswith(".")], "a temporary was left at the top"


def test_unmet_criteria_say_delivered_with_exceptions(tmp_path, no_blender):
    inner = _film(tmp_path / "r1", outcome="delivered_with_exceptions", unmet=["the bell is never struck"])
    L.publish(inner)
    readme = (tmp_path / "r1" / "README.md").read_text()
    assert "The film was delivered with exceptions." in readme and "- the bell is never struck" in readme


def test_a_delivery_set_aside_leaves_the_top(tmp_path, no_blender):
    """`--resume --prompt` reopens a delivered film: what was at the top is no longer the run's, and goes."""
    top = tmp_path / "r1"
    inner = _film(top)
    L.publish(inner)
    L.reopen_delivery(inner)
    L.publish(inner)
    assert sorted(p.name for p in top.iterdir()) == ["README.md", "src"]
    assert "was delivered" not in (top / "README.md").read_text()
    assert (inner / "prior_deliveries" / "1" / "bundle" / "film" / "clip.mp4").read_text() == "film"


@pytest.mark.parametrize("layer, urdf_name", [("object", "object.urdf"), ("scene", "scene.urdf")])
def test_an_object_or_scene_has_its_views_urdf_and_meshes_at_the_top(tmp_path, layer, urdf_name):
    top = tmp_path / "r1"
    inner = top / "src"
    _w(inner / "meta.json", json.dumps({"layer": layer, "outcome": "committed", "prompt": "A stool."}))
    _w(inner / "ir.json", json.dumps({"id": "x", "parts": []}))
    _w(inner / "final_render" / "0_front.png", "png")
    _w(inner / "final_render" / "1_top.png", "png")
    _w(inner / "final_render" / "ir.urdf", '<robot><mesh filename="assets/meshes/a.obj"/></robot>')
    _w(inner / "final_render" / "assets" / "meshes" / "a.obj", "v 0 0 0")
    assert "errors" not in L.publish(inner)
    assert sorted(p.name for p in (top / "views").iterdir()) == ["0_front.png", "1_top.png"]
    urdf = (top / urdf_name).read_text()
    assert 'filename="meshes/meshes/a.obj"' in urdf and (top / "meshes" / "meshes" / "a.obj").is_file()
    readme = (top / "README.md").read_text()
    assert (f"The {layer} was delivered. It is `{urdf_name}`, with the meshes it names in `meshes/`, and `views/` "
            "holds 2 renders of it.") in readme
    assert f"--preload-asset NAME={top}" in readme


def test_a_plain_run_has_its_answer_at_the_top(tmp_path):
    inner = tmp_path / "r1" / "src"
    _w(inner / "meta.json", json.dumps({"layer": "plain", "prompt": "Say hi."}))
    _w(inner / "result.json", json.dumps({"text": "hi", "submitted_at": "x"}))
    L.publish(inner)
    assert (tmp_path / "r1" / "answer.md").read_text() == "hi\n"
    assert "The answer is in `answer.md`." in (tmp_path / "r1" / "README.md").read_text()


@pytest.mark.parametrize("meta, says", [
    ({"launch_failure": {"reason": "environment", "message": "FATAL: claude is not logged in",
                         "fix": ["claude login"]}}, ["The run did not start: claude is not logged in. To fix it:",
                                                      "- Claude login"]),
    ({"layer": "object", "outcome": "interrupted", "finished_at": "2026-11-09T12:00:00+00:00",
      "claude_session_id": "s-1", "time_limit": {"hours": 2.0, "reached": True}}, [
          "The run stopped at its 2-hour time limit without delivering an object.",
          "Continue it with more time: `python -m dycraft.cli --resume"]),
    ({"layer": "object", "runtime": "claude", "claude_session_id": "s-1",
      "agent_stop": {"reason": "claude_rate_limit", "detail": "You've hit your session limit"}},
     ["The run stopped without delivering an object. Claude Code stopped with \"You've hit your session limit\".",
      "Continue it once the limit resets"]),
    ({"layer": "scene", "runtime": "codex2", "outcome": "interrupted",
      "agent_stop": {"reason": "codex_never_started", "detail": "codex opened no session in 30 s"}},
     ["Its agent never started a session, so there is nothing to resume: fix what stopped it, above, then run the "
      "same command again."]),
    ({"layer": "object", "started_at": "2026-11-09T10:00:00+00:00"},
     ["The run is still going. The object's URDF, meshes, and views appear here when the agent delivers them.",
      "Follow it in the viewer"]),
], ids=["refused", "time-limit", "stopped", "never-started", "running"])
def test_a_run_that_has_not_delivered_says_why(tmp_path, meta, says):
    inner = tmp_path / "r1" / "src"
    _w(inner / "meta.json", json.dumps({"prompt": "A stool.", **meta}))
    L.publish(inner)
    readme = (tmp_path / "r1" / "README.md").read_text()
    for line in says:
        assert line in readme, readme
    if "never started" in " ".join(says):
        assert "--resume" not in readme
    assert sorted(p.name for p in (tmp_path / "r1").iterdir()) == ["README.md", "src"]


def test_a_flat_run_and_a_sub_run_are_left_alone(tmp_path):
    flat = tmp_path / "flat"
    _w(flat / "meta.json", json.dumps({"layer": "plain"}))
    _w(flat / "result.json", json.dumps({"text": "hi"}))
    sub = tmp_path / "r1" / "src" / "sub_runs" / "scene_rewrite"
    _w(sub / "meta.json", json.dumps({"layer": "plain"}))
    assert L.publish(flat) == {} and L.publish(sub) == {}
    assert sorted(p.name for p in flat.iterdir()) == ["meta.json", "result.json"]
    assert sorted(p.name for p in sub.iterdir()) == ["meta.json"]


def test_a_blend_that_names_files_beside_it_is_copied_with_its_paths_into_the_bundle(tmp_path):
    """Blender's own Save Copy, so the top-level blend opens as the sealed one does; a blend with no such file stays
    a hardlink."""
    from dycraft.render._blender_env import blender_child_env, blender_python
    interp = blender_python()
    if not interp:
        pytest.skip("no Blender interpreter")
    import subprocess
    inner = _film(tmp_path / "r1")
    bundle = inner / "bundle"
    (bundle / "animation.blend").unlink()
    _w(bundle / "tex.png", "")
    make = ("import bpy, sys\n"
            "img = bpy.data.images.new('t', 4, 4)\nimg.use_fake_user = True\n"
            "img.filepath_raw = sys.argv[-1]\nimg.file_format = 'PNG'\nimg.save()\n"
            "img.filepath = '//tex.png'\n"
            "bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-2])\n")
    subprocess.run([interp, "-c", make, str(bundle / "animation.blend"), str(bundle / "tex.png")], check=True,
                   env=blender_child_env(), capture_output=True)
    assert "errors" not in L.publish(inner)
    top_blend = tmp_path / "r1" / "animation.blend"
    assert not os.path.samefile(top_blend, bundle / "animation.blend")
    read = ("import bpy, sys\nbpy.ops.wm.open_mainfile(filepath=sys.argv[-1])\n"
            "print('PATHS', [bpy.path.abspath(i.filepath) for i in bpy.data.images if i.filepath])\n")
    out = subprocess.run([interp, "-c", read, str(top_blend)], env=blender_child_env(), capture_output=True,
                         text=True).stdout
    paths = eval(out.split("PATHS ", 1)[1].splitlines()[0])
    assert [Path(p).resolve() for p in paths] == [(bundle / "tex.png").resolve()]


def test_unmet_is_said_in_sentences_and_leaves_criteria_to_their_table(tmp_path, no_blender):
    """The kinds `finish()` records, from the paper's films: a failing check with its frames, storyboard states, frames
    no check covered, a design repair, and a criterion, which the criteria table already shows."""
    unmet = [{"kind": "check", "verifier": "NoPenetration", "frames": [[9, 323], [328, 2221]]},
             {"kind": "check", "verifier": "EventsSatisfied", "frames": [[1, 2], [5, 5], [8, 9], [12, 12], [20, 30],
                                                                         [40, 41], [50, 52]]},
             {"kind": "storyboard", "states": ["actor_mark", "camera_lock"]},
             {"kind": "unchecked", "frames": [[100, 120]]},
             {"kind": "design", "verifier": "SupportConnected", "repairs": ["seat the lamp on the bench"]},
             {"kind": "criterion", "id": "hammer", "evidence": "late"}]
    inner = _film(tmp_path / "r1", outcome="delivered_with_exceptions", unmet=unmet)
    L.publish(inner)
    readme = (tmp_path / "r1" / "README.md").read_text()
    section = readme.split("## Unmet\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert section == [
        "- `NoPenetration` fails on frames 9–323 and 328–2221.",
        "- `EventsSatisfied` fails on frames 1–2, 5, 8–9, 12, 20–30, and 2 more ranges (5 frames).",
        "- Storyboard states not accepted: `actor_mark` and `camera_lock`.",
        "- Frames no check covered: 100–120.",
        "- `SupportConnected` asks for a design repair: seat the lamp on the bench.",
    ]


def test_a_run_without_a_sandbox_says_so_under_its_title(tmp_path):
    from dycraft import run_readme
    inner = tmp_path / "r1" / "src"
    _w(inner / "meta.json", json.dumps({"layer": "object", "runtime": "codex", "prompt": "A stool.",
                                        "unsandboxed": {"agent": "Codex", "reason": "bwrap: loopback: no"}}))
    readme = run_readme.render(inner, set())
    assert ("> **Ran without a sandbox** (`--allow-unsandboxed`): Codex's sandbox could not start on this machine "
            "(bwrap: loopback: no), so the agent's commands could read and write anywhere the user who launched it "
            "could.") in readme
    _w(inner / "meta.json", json.dumps({"layer": "object", "runtime": "codex", "prompt": "A stool."}))
    assert "without a sandbox" not in run_readme.render(inner, set())
