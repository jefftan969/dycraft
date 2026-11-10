"""The run's ordered record of what each operation did to the project.

It answers two questions after a run finishes. "What source ran?" — for an
agent that keeps its logic in helper files, the recorded `code` is a pointer,
not a program, and source reaches `animation_src/` through the agent's own
shell rather than any harness operation, so only the journal versions it. "What
changed in the project?" — a successful edit renames its workspace into place,
so the pre-state is gone, and `version` hashes a tree that is mostly agent
scratch.

These pin the journal that answers both, and the boundaries of what it claims.
"""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import time

import pytest

from dycraft.animation.blender.journal import (Journal, MAX_BLOB_BYTES,
                                               MAX_INLINE_ARG, describe)
from dycraft.animation.blender.scene_digest import SCHEMA, diff


def test_blobs_are_content_addressed_and_stored_once(tmp_path):
    journal = Journal(tmp_path)
    first = journal.store_blob("import bpy\n")
    again = journal.store_blob("import bpy\n")
    other = journal.store_blob("import bpy  # changed\n")
    assert first == again != other
    assert journal.read_blob(first) == b"import bpy\n"
    assert len(list(journal.blobs.rglob("*"))) - len(list(journal.blobs.glob("*"))) == 2


def test_source_capture_keeps_python_and_only_notes_bulk_data(tmp_path):
    """Motion caches (`.json.gz`) can be many times the size of the Python.

    Storing those per call would cost hundreds of megabytes to answer a
    question nobody asks. Noting their size and mtime still lets a replay say "the cache
    differed" rather than diverge silently, which is why they appear at all.
    """
    src = tmp_path / "animation_src"
    (src / "human_motion").mkdir(parents=True)
    (src / "gait.py").write_text("STRIDE = 0.62\n")
    (src / "human_motion" / "solve.py").write_text("def solve(): ...\n")
    (src / "cache.json.gz").write_bytes(b"\x1f\x8b" + b"\0" * 4096)
    (src / "huge.json").write_bytes(b"[" + b"0," * MAX_BLOB_BYTES + b"0]")

    journal = Journal(tmp_path / "journal")
    manifest = journal.read_manifest(journal.capture_sources([src]))
    files = manifest["files"]

    assert files["animation_src/gait.py"]["stored"] is True
    assert files["animation_src/human_motion/solve.py"]["stored"] is True
    assert journal.read_blob(files["animation_src/gait.py"]["sha"]) == b"STRIDE = 0.62\n"
    # Recorded, but by size and mtime, decided from the stat without ever
    # reading them: hashing content here meant 519 MB of reads per operation on
    # one of the two runs, 156 GB across it.
    for name in ("animation_src/cache.json.gz", "animation_src/huge.json"):
        assert files[name]["stored"] is False
        assert files[name]["bytes"] and files[name]["mtime_ns"]
        assert "sha" not in files[name]


def test_bulk_files_are_never_read(tmp_path, monkeypatch):
    """The cost rule, pinned: a source root full of data must stay cheap."""
    src = tmp_path / "animation_src"
    src.mkdir()
    (src / "small.py").write_text("ok = 1\n")
    (src / "dump.json").write_bytes(b"x" * (MAX_BLOB_BYTES + 1))
    (src / "cache.bin").write_bytes(b"y" * 4096)

    read = []
    original = pathlib.Path.read_bytes
    monkeypatch.setattr(pathlib.Path, "read_bytes",
                        lambda self: (read.append(self.name), original(self))[1])
    Journal(tmp_path / "journal").capture_sources([src])
    assert read == ["small.py"], read


def test_the_manifest_hash_is_the_identity_of_the_tree(tmp_path):
    src = tmp_path / "animation_src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    journal = Journal(tmp_path / "journal")

    first = journal.capture_sources([src])
    assert journal.capture_sources([src]) == first, "unchanged tree, same identity"
    (src / "a.py").write_text("x = 2\n")
    edited = journal.capture_sources([src])
    assert edited != first
    # Both versions survive: this is what makes "which helper did call 57 run"
    # answerable after the agent has rewritten it fifteen more times.
    older = journal.read_manifest(first)["files"]["animation_src/a.py"]["sha"]
    newer = journal.read_manifest(edited)["files"]["animation_src/a.py"]["sha"]
    assert journal.read_blob(older) == b"x = 1\n"
    assert journal.read_blob(newer) == b"x = 2\n"


def test_source_capture_ignores_symlinks(tmp_path):
    """The edit path already refuses them; following one would read outside the run."""
    src = tmp_path / "animation_src"
    src.mkdir()
    (src / "real.py").write_text("ok = True\n")
    secret = tmp_path / "outside.py"
    secret.write_text("TOKEN = 'do not capture'\n")
    (src / "link.py").symlink_to(secret)

    journal = Journal(tmp_path / "journal")
    files = journal.read_manifest(journal.capture_sources([src]))["files"]
    assert "animation_src/real.py" in files
    assert "animation_src/link.py" not in files


def test_missing_source_roots_are_not_an_error(tmp_path):
    """A source root may not exist yet: a fresh run has no helpers and may have
    no `design/`."""
    journal = Journal(tmp_path / "journal")
    assert journal.read_manifest(journal.capture_sources([tmp_path / "nope"])) == {"files": {}}


def test_large_arguments_become_blobs_and_small_ones_stay_readable(tmp_path):
    """An `h.blender` snippet can be a few hundred bytes or tens of kilobytes."""
    journal = Journal(tmp_path)
    short, long = "s.frame_set(3)", "x = 1\n" * MAX_INLINE_ARG
    out = describe({"code": long, "name": short, "stage": "animation",
                    "invalidate_shots": None, "_version": "secret"}, journal)
    assert out["name"] == short and out["stage"] == "animation"
    assert out["code"]["bytes"] == len(long)
    assert journal.read_blob(out["code"]["blob"]).decode() == long
    assert "_version" not in out, "internal arguments are not the agent's call"
    assert "invalidate_shots" not in out, "None carries nothing"


def test_records_are_ordered_and_survive_a_daemon_restart(tmp_path):
    journal = Journal(tmp_path)
    for _ in range(3):
        journal.record(seq=journal.next_seq(), operation="blender")
    # `Jobs` reloads its records the same way; a restarted daemon must continue
    # the run's numbering rather than overwrite it.
    resumed = Journal(tmp_path)
    resumed.record(seq=resumed.next_seq(), operation="compile")
    assert [r["seq"] for r in resumed.records()] == [1, 2, 3, 4]
    assert resumed.records()[-1]["operation"] == "compile"


def test_a_record_never_raises_into_the_operation(tmp_path):
    """The journal is diagnostic. A logging failure must not fail an edit."""
    journal = Journal(tmp_path)
    journal.path.parent.chmod(0o500)
    try:
        journal.record(seq=1, operation="blender")     # unwritable directory
    finally:
        journal.path.parent.chmod(0o700)


def test_the_newest_digest_wins_because_build_writes_two(tmp_path):
    """`build` imports a donor and then merges it; the merge is what publishes."""
    journal = Journal(tmp_path / "journal")
    job = tmp_path / "job"
    (job / "import").mkdir(parents=True)
    (job / "update").mkdir(parents=True)
    (job / "import" / "digest.json").write_text(json.dumps({"schema": SCHEMA, "which": "donor"}))
    time.sleep(0.01)
    (job / "update" / "digest.json").write_text(json.dumps({"schema": SCHEMA, "which": "merged"}))

    assert journal.store_digest(4, job)
    assert journal.read_digest(4)["which"] == "merged"
    assert gzip.decompress((journal.digests / "000004.json.gz").read_bytes())


def test_an_operation_that_writes_no_digest_reports_none(tmp_path):
    """`plan` and `accept_storyboard` publish without touching the blend."""
    journal = Journal(tmp_path / "journal")
    empty = tmp_path / "job"
    empty.mkdir()
    assert journal.store_digest(1, empty) is None
    assert journal.read_digest(1) is None


def test_latest_digest_carries_the_scene_forward_across_such_operations(tmp_path):
    journal = Journal(tmp_path / "journal")
    job = tmp_path / "job"
    job.mkdir()
    (job / "digest.json").write_text(json.dumps({"schema": SCHEMA, "totals": {"objects": 7}}))
    journal.store_digest(2, job)
    # Sequence 3 was a `plan`: it published a version but changed no geometry.
    seq, digest = journal.latest_digest(before=4)
    assert seq == 2 and digest["totals"]["objects"] == 7


def test_the_journal_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("DYCRAFT_ANIMATION_JOURNAL", "0")
    journal = Journal(tmp_path)
    assert journal.next_seq() is None
    assert journal.store_blob("x") is None
    assert journal.capture_sources([tmp_path]) is None
    journal.record(seq=None, operation="blender")
    assert not journal.path.exists()


# -- the digest's own claims ------------------------------------------------

def _digest(objects=None, curves=None, **scene):
    return {"schema": SCHEMA,
            "scene": {"frame_start": 0, "frame_end": 100, **scene},
            "objects": objects or {}, "curves": curves or {},
            "counts": {"objects": len(objects or {})},
            "totals": {"objects": len(objects or {})}}


def test_an_operation_that_changed_nothing_is_distinguishable_from_a_small_change():
    """The distinction that makes bisecting a regression cheap.

    Many operations in a run change no curve at all, and the digest diff is
    what tells them apart from the ones that do.
    """
    before = _digest({"base": {"matrix": "aa"}})
    assert diff(before, _digest({"base": {"matrix": "aa"}})) is None
    changed = diff(before, _digest({"base": {"matrix": "bb"}}))
    assert changed["objects"]["modified"] == ["base"]
    assert not changed["objects"]["added"] and not changed["objects"]["removed"]


def test_diff_names_added_and_removed_curves():
    before = _digest(curves={"A|location[0]": {"n": 2}})
    after = _digest(curves={"A|location[1]": {"n": 5}})
    changed = diff(before, after)
    assert changed["curves"] == {"added": ["A|location[1]"],
                                 "removed": ["A|location[0]"], "modified": []}


def test_a_retimed_scene_shows_up_even_with_identical_objects():
    changed = diff(_digest(), _digest(frame_end=240))
    assert changed["scene"]["frame_end"] == [100, 240]


def test_digests_of_different_schemas_refuse_to_compare():
    """Otherwise a schema change reads as scene churn across the whole run."""
    before = _digest()
    after = {**_digest(), "schema": SCHEMA + 1}
    with pytest.raises(ValueError, match="schema"):
        diff(before, after)


def test_diff_is_none_when_either_side_is_missing():
    assert diff(None, _digest()) is None
    assert diff(_digest(), None) is None
