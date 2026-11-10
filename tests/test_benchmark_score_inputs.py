"""What a path names for the scorer: a DyCraft run's delivered evaluation, its blend exported and scored with its
joint tree, or any other blend exported with its own records; and a run over several inputs going on past one that
cannot be scored."""
import json

import pytest

from dycraft.benchmark.score import __main__ as M
from dycraft.benchmark.score.inputs import resolve


def _bundle(root, *, evaluation=None, baseline="design_state/baseline.json"):
    """A bundle folder under `root`: a blend, a design baseline unless `baseline` is None, and an evaluation index
    whose `complete` is `evaluation` unless that is None."""
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "animation.blend").write_bytes(b"BLENDER")
    if baseline:
        (bundle / baseline).parent.mkdir(parents=True, exist_ok=True)
        (bundle / baseline).write_text("{}")
    if evaluation is not None:
        (bundle / "evaluation").mkdir()
        (bundle / "evaluation" / "index.json").write_text(json.dumps({"complete": evaluation}))
    return bundle


def test_a_dycraft_films_blend_is_scored_as_its_bundle(tmp_path):
    # As the films ship: blend and baseline, no evaluation. Given its folder, its bundle or its blend, the film is
    # exported and scored with its joint tree (kind "dycraft"), never as a plain `.blend`.
    bundle = _bundle(tmp_path / "films" / "B23_spring_cart_runs_course")
    for given in (bundle.parent, bundle, bundle / "animation.blend"):
        inp = resolve(given)
        assert inp.kind == "dycraft" and inp.task == "B23", given
        assert inp.evaluation_dir is None and inp.blend == bundle / "animation.blend"
        assert inp.baseline == bundle / "design_state" / "baseline.json"


def test_a_complete_evaluation_is_read_and_an_incomplete_one_exported(tmp_path):
    done = _bundle(tmp_path / "b05_done", evaluation=True)
    assert resolve(done / "animation.blend").evaluation_dir == done / "evaluation"
    partial = _bundle(tmp_path / "b05_partial", evaluation=False)
    inp = resolve(partial.parent)
    assert inp.kind == "dycraft" and inp.evaluation_dir is None
    assert inp.notes == ["no complete evaluation: exported from animation.blend"]


def test_any_other_blend_is_exported_with_its_own_records(tmp_path):
    staged = _bundle(tmp_path / "B14_crow" / "bare_codex", baseline=None)
    inp = resolve(staged / "animation.blend")
    assert inp.kind == "blend" and inp.task == "B14" and inp.evaluation_dir is None
    other = tmp_path / "b02_film.blend"
    other.write_bytes(b"BLENDER")
    assert resolve(other).kind == "blend" and resolve(other).task == "B02"


def test_the_innermost_brief_in_a_path_wins_and_task_overrides_it(tmp_path):
    bundle = _bundle(tmp_path / "rerun_b22" / "B14_crow_opens_puzzle_feeder" / "dycraft_codex")
    assert resolve(bundle.parent).task == "B14"
    assert resolve(bundle.parent, "B05").task == "B05"


def test_an_arm_is_found_from_its_folder_its_bundle_or_its_blend(tmp_path):
    arm = tmp_path / "out" / "B23_spring" / "dycraft_codex"
    bundle = _bundle(arm, evaluation=True)
    (arm / "benchmark").mkdir()
    for given in (arm, bundle, bundle / "animation.blend"):
        assert resolve(given).arm == arm, given


def _arm(out, brief, name, *, scorable=True):
    arm = out / brief / name
    (arm / "benchmark").mkdir(parents=True)
    (arm / "benchmark" / "record.json").write_text("{}")
    if scorable:
        _bundle(arm, evaluation=True)
    return arm


def test_a_benchmark_directory_skips_arms_it_cannot_score(tmp_path, capsys):
    out = tmp_path / "out"
    _arm(out, "B23_spring", "dycraft_codex")
    unscorable = _arm(out, "B23_spring", "bare_codex", scorable=False)
    inputs = M.expand([out], None)
    assert [inp.arm.name for inp in inputs] == ["dycraft_codex"]
    assert f"skipped {unscorable}" in capsys.readouterr().err
    with pytest.raises(SystemExit):                     # named on its own, it is an error
        M.expand([unscorable], None)


def test_one_film_that_cannot_be_scored_does_not_stop_the_others(tmp_path, monkeypatch):
    out = tmp_path / "out"
    good, bad = _arm(out, "B05_good", "dycraft_codex"), _arm(out, "B06_bad", "dycraft_codex")
    half = _arm(out, "B07_half", "dycraft_codex")
    monkeypatch.setattr(M, "blender_python", lambda explicit=None: "blender-python")
    monkeypatch.setattr(M, "coal_python", lambda explicit=None: "coal-python")

    def penetration(inp, where, settings):
        if inp.arm == bad:
            raise SystemExit("_export_child.py failed (exit 1)")
        return {"primary": {"dynamic_penetration_frame_rate": 0.1, "evaluated_frames": 10, "dynamic_invalid_frames": 1,
                            "cross_object_dynamic_frame_rate": 0.1, "self_penetration_dynamic_frame_rate": 0.0}}

    def floating(inp, where, settings):
        if inp.arm in (bad, half):
            raise RuntimeError("support graph")
        return {"primary": {"floating_rate": 0.0, "floating_body_frames": 0, "animated_body_frames": 10}}

    monkeypatch.setattr(M.penetration, "run", penetration)
    monkeypatch.setattr(M.floating, "run", floating)
    argv = [str(bad), str(half), str(good), "--out", str(tmp_path / "scores"), "--json", str(tmp_path / "rows.json")]
    assert M.main(argv) == 1
    rows = json.loads((tmp_path / "rows.json").read_text())
    assert [("error" in row) for row in rows] == [True, True, False]
    failed = json.loads((bad / "benchmark" / "score.json").read_text())
    assert "penetration: SystemExit: _export_child.py failed" in failed["error"]
    assert "floating: RuntimeError: support graph" in failed["error"] and "Traceback" in failed["traceback"]
    # A metric that was scored is kept when the other fails.
    halfway = json.loads((half / "benchmark" / "score.json").read_text())
    assert halfway["penetration_rate"] == 0.1 and "floating_rate" not in halfway
    assert halfway["error"].startswith("floating: RuntimeError")
    assert json.loads((good / "benchmark" / "score.json").read_text())["penetration_rate"] == 0.1
    assert "failed" in M.table(rows).splitlines()[2] and "10.0" in M.table(rows).splitlines()[2]
