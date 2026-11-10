"""The benchmark's brief sets, and the exact text each arm is given.

The bare arm's prompt is pinned to the paper: every one of the 24 final bare runs recorded the
sha256 of the prompt it was handed (`control/config.json`, `prompt_sha256`), and composing the same
brief here must reproduce those bytes. The guidance file is pinned the same way.
"""
import hashlib

import pytest

from dycraft.benchmark import briefs as B

#: `prompt_sha256` recorded by each of the paper's 24 final bare runs.
PAPER_BARE_PROMPT_SHA256 = {
    "B01": "a4339c7892f2b65d0e399217c2866f8183c9d37390106844a432f27cafc0b96f",
    "B02": "7e44827e1a9ba90b7820e5f53863ec983c5414c99752c65eaf4323a786e5caf6",
    "B03": "d26891450217479898847d575575e9ae3902c3ca94d0002aa588b764943bae48",
    "B04": "5c735a2b688dd3183944f5aff2cf5b45c3f6eba85f1572258a212c91176486f9",
    "B05": "a71ff8aa341f5a6e1138069711b7fee0eaaceaa4dc7b73fdcf4cd532b29b4dc6",
    "B06": "62c5bf482ee151f8708a344aa882fb5d82d52522a066a352820fe901201162df",
    "B07": "22a8543d9e9b18848ee4de8be6e08f445f937302b69f2e869d59d9b35246f2bb",
    "B08": "a7b32aaa9ce3f8d2180d74848131add7e6f156f7cd990a06d6e226402c6f0b67",
    "B09": "bc2ade8efaa922fa8fad664407c9829a271197a6f8dfca86fce6aac034be3e56",
    "B10": "760d1ed667ac538a378f35ae454bd88d58f128b16ba3c669f396c03ff7a0f424",
    "B11": "9087e018aafb9936ba8ea81d5372514d95c4e8be4c194648b046516ec691abe8",
    "B12": "892de798993befbcf3ac6bc93155a452b788d3bf389a8a834575b4fcd710ae98",
    "B13": "50390c4f6eb5cdad9e096e6d17bf21ec05cbe811d24128c2684748c591159eee",
    "B14": "8e823cedb8efe3be44abb0eec5090749548fd2f5efa561375e1ddb58c6f58273",
    "B15": "ff645b0771165ac98b93ac5eea782214a63498cbe2f04724a61944a57d40cf78",
    "B16": "a1b0a689de19d6326050c6a78ab5cf5fb7e0fa4f4e0864b49ccda970465f22f9",
    "B17": "bb58adf9b875d11093f79b883fdff4be0d3d57f4ab7d902cd3896f9770e133ab",
    "B18": "41d81af8c135e6aaf408d6a3993384c111c626f56c285af8a7d17b6d6b3e917f",
    "B19": "ae8adff8f4ffd7ccec9d4a4eace74b6fe534c3bfdf297adfcdfbefabc42afae2",
    "B20": "078b928886d38ee9b16344dd36d874bd43001947beca177c0dac91155fbe0f0e",
    "B21": "30c27d2c986c6398c69d4e05e525f9784ad1a582259d2dd08e0891f083f05d6d",
    "B22": "042c72e1a0ae49551f36e608e38b7381213240b7c58e117513bb2b3e7fc77e0d",
    "B23": "1776d5091ec27a6d64bf632c0a0fa44eb49ba389df7100c6b37190a10df54a85",
    "B24": "d2a089f4203b62fb6602a2056a3821cfae352c42a84e78bea22b859905c90a33",
}
PAPER_GUIDANCE_SHA256 = "6862668b55ee490686f3d2fb1f910077dfd1b72ccf38e413480233338ff5ab6d"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_bare_guidance_is_the_one_the_paper_used():
    assert _sha(B.BARE_GUIDANCE.read_text()) == PAPER_GUIDANCE_SHA256


@pytest.mark.parametrize("bid", sorted(PAPER_BARE_PROMPT_SHA256))
def test_the_bare_prompt_reproduces_the_papers_bytes(bid):
    assert _sha(B.bare_prompt(B.load()[bid])) == PAPER_BARE_PROMPT_SHA256[bid]


def test_every_brief_parses_with_its_staged_id():
    briefs = B.load()
    assert len(briefs) == 47                        # 24 long, 13 short, 10 medium
    assert sum(b.horizon == "short" for b in briefs.values()) == 13
    assert sum(b.horizon == "medium" for b in briefs.values()) == 10
    assert briefs["B17"].staged_id == "marble_course_runs"
    assert briefs["B01S"].slug == "b01s_g1_scrambles_eggs"
    assert all(b.text and not b.text.endswith("---") for b in briefs.values())


def _imported(brief):
    return brief.text == B.load(imported_characters=True)[brief.id].text != B.load()[brief.id].text


def test_the_sets_take_each_cut_where_there_is_one():
    """A set takes the cut; else, for the DyCraft arm in the short and medium sets, the variant that casts its
    characters from the library; else the brief as written. The bare arm has no library, so never the variant."""
    short, horizon = B.select("short", library=True)
    assert horizon == "short" and len(short) == 24
    assert [b.id for b in short[:12]] == [f"B{n:02d}S" for n in range(1, 13)]
    assert [b.id for b in short[12:16]] == ["B13", "B14", "B15S", "B16"]
    assert [_imported(b) for b in short[12:16]] == [True, True, False, True]
    assert [b.id for b in short[16:]] == [f"B{n:02d}" for n in range(17, 25)]
    medium, _ = B.select("medium", library=True)
    ids = [b.id for b in medium]
    assert ids[4] == "B05" and ids[6] == "B07" and ids[0] == "B01M" and ids[11] == "B12M"
    assert ids[12:] == [f"B{n:02d}" for n in range(13, 25)]
    assert [_imported(b) for b in medium[12:16]] == [True] * 4
    long_set = B.select("long", library=True)[0]
    assert [b.id for b in long_set] == [f"B{n:02d}" for n in range(1, 25)] and not any(map(_imported, long_set))
    for spec in ("short", "medium", "long"):
        assert not any(_imported(b) for b in B.select(spec)[0])      # without the library: the default
    assert [b.id for b in B.select("short")[0][12:16]] == ["B13", "B14", "B15S", "B16"]


def test_the_bare_form_drops_a_cuts_library_casting_and_keeps_its_cast():
    """The bare arm has no asset library: no bare prompt tells it to take anything from one, while the cut's own
    description of the cast stays; the long briefs have no such sentence, so their paper prompts are untouched."""
    briefs = B.load()
    cast = [b for b in briefs.values() if B.BARE_CASTING.search(b.text)]
    assert sorted(b.id for b in cast) == ["B08M", "B08S", "B09M", "B09S", "B10M", "B10S", "B11M", "B11S",
                                          "B12M", "B12S", "B15S"]
    assert all(len(B.BARE_CASTING.findall(b.text)) == 1 for b in cast)
    assert not any("asset library" in B.bare_task(b) for b in briefs.values())
    assert "The dog is a German shepherd and the trainer a woman in sportswear." in B.bare_task(briefs["B15S"])


def test_ids_are_selected_as_given_and_unknown_ones_refused():
    chosen, horizon = B.select("b01s, B13")
    assert [b.id for b in chosen] == ["B01S", "B13"] and horizon is None
    with pytest.raises(SystemExit, match="unknown id"):
        B.select("B99")
    with pytest.raises(SystemExit, match="twice"):
        B.select("B13,B13")


def test_every_g1_brief_carries_the_one_sentence_the_bare_form_swaps():
    for b in B.load().values():
        assert b.uses_g1 == (b.number <= 8), b.id
        task = B.bare_task(b)
        if b.uses_g1:
            assert B.BARE_G1_SENTENCE in task and B.G1_SENTENCE not in task
            assert B.BARE_ENV_G1 in task
        else:
            assert "/assets/g1" not in task


def test_the_dycraft_prompt_is_the_brief_then_length_then_budget():
    b = B.load()["B17"]
    text = B.dycraft_prompt(b, film_seconds=(60, 75), budget_hours=8)
    brief, length, budget = text.split("\n\n")
    assert brief == b.text
    assert length.startswith("Expect a film of no more than about 60–75 seconds")
    assert budget.startswith("Work to a wall-clock budget of 8 hours from the start")
    assert B.dycraft_prompt(b, film_seconds=None, budget_hours=None) == b.text
    assert "1 hour from" in B.budget_note(1)


def test_film_seconds_parsing():
    assert B.parse_film_seconds("60-75") == (60.0, 75.0)
    assert B.parse_film_seconds("90") == (90.0, 90.0)
    assert B.parse_film_seconds("none") is None
    with pytest.raises(SystemExit):
        B.parse_film_seconds("75-60")


def test_imported_characters_replace_only_the_five_character_briefs():
    plain, imported = B.load(), B.load(imported_characters=True)
    changed = sorted(k for k in plain if plain[k].text != imported[k].text)
    assert changed == ["B10", "B13", "B14", "B15", "B16"]


def test_the_bare_arm_can_be_told_the_dycraft_arms_length_and_budget():
    b = B.load()["B01S"]
    plain = B.bare_prompt(b)
    told = B.bare_prompt(b, film_seconds=(60, 75), budget_hours=2)
    assert plain == B.bare_prompt(b, film_seconds=None, budget_hours=None)
    task = told.split("\n\n---\n\n", 1)[0]
    prefix, brief, length, budget, env = task.split("\n\n")
    assert prefix == B.BARE_PREFIX and B.BARE_G1_SENTENCE in brief
    assert length == B.length_note((60, 75)) and budget == B.budget_note(2)
    assert env.strip() == B.BARE_ENV_G1
    assert told.endswith(B.BARE_GUIDANCE.read_text())


def test_a_prompt_of_your_own_gets_the_benchmarks_bare_wrapper():
    b = B.adhoc("A cat jumps onto a table, then naps.")
    assert b.slug == "p_a_cat_jumps_onto_a_table" and not b.uses_g1
    task = B.bare_task(b)
    assert task.startswith(B.BARE_PREFIX + "\n\nA cat jumps onto a table, then naps.\n\n")
    assert task.rstrip().endswith(B.BARE_ENV_NO_ASSETS)


def test_a_prompt_given_the_g1_gets_the_papers_g1_sentence_and_mount():
    b = B.adhoc("a humanoid robot gets up", g1=True)
    task = B.bare_task(b)
    assert b.uses_g1
    assert f"a humanoid robot gets up {B.BARE_G1_SENTENCE}" in task
    assert task.rstrip().endswith(B.BARE_ENV_G1)


def test_a_command_takes_briefs_or_one_prompt(tmp_path):
    from types import SimpleNamespace as NS
    for bad in (NS(briefs=None, prompt=None), NS(briefs="B01", prompt="x"), NS(briefs="B01", prompt=None, g1=True)):
        with pytest.raises(SystemExit):
            B.choose(bad)
    md = tmp_path / "brief.md"
    md.write_text("a lamp nods twice\n")
    (brief,), horizon = B.choose(NS(briefs=None, prompt=str(md), g1=False))
    assert brief.text == "a lamp nods twice" and horizon is None


def test_the_brief_the_bare_arm_reads_is_the_one_its_task_carries():
    """`bare_brief` is the text inside the bare task (and what the bare run records), with the casting dropped and the
    G1 sentence pointed at /assets/g1."""
    briefs = B.load()
    for bid in ("B15S", "B01", "B17"):
        text = B.bare_brief(briefs[bid])
        assert text in B.bare_task(briefs[bid]) and "asset library" not in text
    assert B.BARE_G1_SENTENCE in B.bare_brief(briefs["B01"])
    assert B.bare_brief(briefs["B17"]) == briefs["B17"].text
