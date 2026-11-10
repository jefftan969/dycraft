"""`--prompt` as text, or as a file of any type (`cli/prompt_files.py`).

A text file is the prompt; any other file (a PDF, an image, a model) is copied into the run's `inputs/` and the
agent's task points at it; something that looks like a path but names no file is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from dycraft.cli import cli, prompt_files

PDF = b"%PDF-1.7\n\x00\x01binary"


@pytest.mark.parametrize("name, body", [("brief.txt", "A wooden stool."), ("brief.md", "# Stool\nThree legs."),
                                        ("brief.json", '{"object": "stool"}'), ("brief.yaml", "object: stool"),
                                        ("brief", "no extension at all")])
def test_a_text_file_of_any_kind_is_the_prompt(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body + "\n")
    text, brief, note, error = prompt_files.read(str(path))
    assert (text, brief, error) == (body, None, None)
    assert note.startswith(f"[cli] loaded the prompt from {path} (")


def test_text_that_is_not_utf8_is_read_with_a_warning(tmp_path):
    path = tmp_path / "brief.txt"
    path.write_bytes("Caf\xe9 stool".encode("latin-1"))
    text, brief, note, _ = prompt_files.read(str(path))
    assert text == "Caf� stool" and brief is None and "not valid UTF-8" in note


@pytest.mark.parametrize("name, kind", [("brief.pdf", "PDF"), ("sketch.png", "image"), ("chair.glb", "3D model"),
                                        ("thing.bin", "file")])
def test_any_other_file_is_a_brief_the_agent_opens(tmp_path, name, kind):
    path = tmp_path / name
    path.write_bytes(PDF)
    text, brief, note, error = prompt_files.read(str(path))
    assert text is None and error is None and brief.kind == kind and brief.size == len(PDF)
    assert brief.task(brief.staged_name()) == (
        f"Your brief is the file inputs/{name} ({kind}, {len(PDF)} bytes), in your run directory. Read it before you "
        "start; it is the whole task.")
    assert note.endswith(f"staged as inputs/{name} for the agent to open")


@pytest.mark.parametrize("arg", ["briefs/stool.txt", "stool.pdf", "./x"])
def test_a_path_with_no_file_is_refused(tmp_path, monkeypatch, arg):
    monkeypatch.chdir(tmp_path)
    *_, error = prompt_files.read(arg)
    assert error == (f"--prompt {arg}: no such file (relative to {tmp_path}). Pass the brief's text in quotes, or the "
                     "path to an existing file.")


@pytest.mark.parametrize("arg", ["stool", "a wooden stool.pdf with three legs"])
def test_a_word_or_a_sentence_stays_the_prompt(arg):
    assert prompt_files.read(arg) == (arg, None, None, None)


def test_a_brief_is_staged_beside_an_earlier_one_of_the_same_name(tmp_path):
    first, second = tmp_path / "a" / "brief.pdf", tmp_path / "b" / "brief.pdf"
    for path, body in ((first, PDF), (second, PDF + b"2")):
        path.parent.mkdir()
        path.write_bytes(body)
    run_dir = tmp_path / "run"
    assert prompt_files.stage(prompt_files.read(str(first))[1], run_dir) == "inputs/brief.pdf"
    assert prompt_files.stage(prompt_files.read(str(first))[1], run_dir) == "inputs/brief.pdf"     # the same file
    assert prompt_files.stage(prompt_files.read(str(second))[1], run_dir) == "inputs/brief-2.pdf"
    assert (run_dir / "inputs" / "brief-2.pdf").read_bytes() == PDF + b"2"


def _settled(argv):
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    cli._settle_args(parser, args)
    return args


def test_a_launch_reads_a_text_brief_and_points_at_a_file_brief(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    (tmp_path / "brief.txt").write_text("A wooden stool.")
    (tmp_path / "brief.pdf").write_bytes(PDF)
    args = _settled(["--backend", "claude", "--layer", "object", "--prompt", str(tmp_path / "brief.txt")])
    assert args.prompt == "A wooden stool." and args.prompt_file is None
    args = _settled(["--backend", "claude", "--layer", "object", "--prompt", str(tmp_path / "brief.pdf")])
    assert args.prompt.startswith("Your brief is the file inputs/brief.pdf (PDF, ") and args.prompt_file.size == len(PDF)


def test_a_launch_with_a_missing_brief_file_stops_at_the_parser(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _settled(["--backend", "claude", "--layer", "object", "--prompt", "briefs/stool.txt"])
    assert "--prompt briefs/stool.txt: no such file" in capsys.readouterr().err


def test_the_rewriter_is_pointed_at_the_staged_brief_by_its_full_path(tmp_path, monkeypatch):
    """A rewriter works from its own sub-run, so `inputs/brief.pdf` would not find it."""
    from dycraft.agent import scene_rewrite
    monkeypatch.setenv("DYCRAFT_RENDER_BACKEND", "open3d")
    (tmp_path / "brief.pdf").write_bytes(PDF)
    args = _settled(["--backend", "claude", "--layer", "animation", "--prompt", str(tmp_path / "brief.pdf"),
                     "--no-motion-brief"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = args.prompt_staged = prompt_files.stage(args.prompt_file, run_dir)
    args.rewriter_prompt = args.prompt_file.task(str(run_dir / staged), in_run_dir=False)
    seen = {}
    monkeypatch.setattr(scene_rewrite, "rewrite", lambda prompt, **kw: seen.update(prompt=prompt) or "A workshop.")
    cli._stage_scene_entry(cli.build_parser(), args, run_dir)
    assert seen["prompt"] == (f"Your brief is the file {run_dir}/inputs/brief.pdf (PDF, {len(PDF)} bytes). Read it "
                              "before you start; it is the whole task.")
    assert (run_dir / "inputs" / "brief.pdf").read_bytes() == PDF
    assert cli._launch_meta(args, cli.BACKENDS["claude"](), {})["prompt_source"] == {
        "path": str((tmp_path / "brief.pdf").resolve()), "type": "PDF", "bytes": len(PDF),
        "staged": "inputs/brief.pdf"}
