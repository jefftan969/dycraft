"""The sizing tool: the brief file's tables are its output, and it sizes a brief of one's own the same way."""

import pytest

from dycraft.benchmark import briefs as B
from dycraft.benchmark import sizing as S


def test_the_brief_files_sizing_tables_are_the_tools_output():
    """The tables a reader sees must be what the tool computes, or the two drift apart."""
    assert S.write_tables(B.BRIEFS_FILE) == B.BRIEFS_FILE.read_text()


def test_every_shipped_brief_has_beats_and_nothing_else_does():
    assert set(S.BEATS) == set(B.load())


def test_one_brief_is_sized_from_its_beats():
    beats = S.BEATS["B17"]
    assert S.mmss(S.plan_seconds(beats)) == "0:29"
    assert S.mmss(S.predicted_film(beats)) == "0:37"
    mid, lo, hi = S.runtime(S.predicted_film(beats))
    assert (round(mid), round(lo), round(hi)) == (5, 4, 7)


def test_a_brief_of_your_own_is_sized_from_a_beats_file(tmp_path):
    f = tmp_path / "beats.txt"
    f.write_text("# my brief\n2-3 establish\n\n4 walk to the door\n1.5 open it\n")
    beats = S.read_beats(f)
    assert [b[1] for b in beats] == ["establish", "walk to the door", "open it"]
    assert S.plan_seconds(beats) == 8.0


def test_a_malformed_beat_line_is_refused_with_its_line_number(tmp_path):
    f = tmp_path / "beats.txt"
    f.write_text("2-3 establish\nwalk to the door\n")
    with pytest.raises(SystemExit, match="beats.txt:2"):
        S.read_beats(f)
