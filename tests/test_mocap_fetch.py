"""`python -m dycraft.install mocap [--dog]`: the motion-capture libraries, fetched into `assets/mocap/`.

No test reaches the network: the downloads are replaced with small archives built here, and a real request fails.
The layout written is the one `runtime.mocap_library`, the agent's mount, and the environment check read.
"""
import hashlib
import io
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from dycraft.animation.mocap import _fetch as fetch

PREFIX = f"cmu-mocap-{fetch.CMU_COMMIT}/"
CLIPS = {"data/001/01_01.bvh": b"HIERARCHY\nROOT Hips\n", "data/008/08_01.bvh": b"HIERARCHY\nROOT Hips walk\n"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(fetch.urllib.request, "urlopen", refuse)


def _blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _digest(clips):
    return hashlib.sha256("\n".join(sorted(f"{p} {_blob(d)}" for p, d in clips.items())).encode()).hexdigest()


def _tarball(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _serve(monkeypatch, payloads, calls=None):
    def _open(url):
        if calls is not None:
            calls.append(url)
        return io.BytesIO(payloads[url])
    monkeypatch.setattr(fetch, "_open", _open)


@pytest.fixture
def cmu(monkeypatch):
    """A two-clip CMU mirror, with an extra file kept and one skipped."""
    members = {PREFIX + p: d for p, d in CLIPS.items()}
    members[PREFIX + "READMEFIRST.txt"] = b"free for use"
    members[PREFIX + "cmu-mocap-index-spreadsheet.xls"] = b"skipped"
    monkeypatch.setattr(fetch, "CMU_CLIPS", len(CLIPS))
    monkeypatch.setattr(fetch, "CMU_DIGEST", _digest(CLIPS))
    monkeypatch.setattr(fetch, "CMU_BYTES", 1)
    return members


def test_the_cmu_library_installs_where_the_runtime_looks(tmp_path, monkeypatch, cmu):
    from dycraft.animation.blender import runtime
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)})
    assert fetch.run(out=lambda *a: None) == 0
    lib = tmp_path / "assets" / "mocap"
    assert (lib / "cmu/data/008/08_01.bvh").read_bytes() == CLIPS["data/008/08_01.bvh"]
    assert (lib / "cmu/READMEFIRST.txt").is_file() and not list(lib.rglob("*.xls"))
    assert not fetch._partial(lib, "cmu").exists() and not list(lib.glob(".*partial"))
    assert runtime.mocap_library() == lib.resolve()


def test_a_verified_library_is_not_fetched_again(tmp_path, monkeypatch, cmu):
    calls = []
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)}, calls)
    assert fetch.run(dest=tmp_path, out=lambda *a: None) == 0
    assert fetch.run(dest=tmp_path, out=lambda *a: None) == 0
    assert calls == [fetch.CMU_URL]


def _shipped(lib, name):
    """Each file `name` ships for agents, as written beside `lib`'s clips: (installed bytes, shipped bytes)."""
    return [((lib / f).read_bytes(), (fetch.LIBRARIES / name / f).read_bytes()) for f in fetch.SHIPPED[name]]


def test_the_cmu_library_gets_its_notes_for_agents_beside_its_clips(tmp_path, monkeypatch, cmu):
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)})
    assert fetch.run(dest=tmp_path, out=lambda *a: None) == 0
    assert all(got == want for got, want in _shipped(tmp_path / "cmu", "cmu"))
    assert [l.split("\t")[0] for l in (tmp_path / "cmu" / "index.txt").read_text().splitlines()] == ["01_01", "08_01"]


def _bvh(frames, step):
    return f"HIERARCHY\nROOT Hips\n{{\n}}\nMOTION\nFrames: {frames}\nFrame Time: {step}\n0 0 0\n".encode()


def test_cmus_index_is_written_in_the_libraries_common_format(tmp_path):
    """One line per clip, `<clip> TAB <start>-<end>s TAB <label> TAB <notes>` as in the dog index, so one grep
    covers every library: the clip's span from its header, CMU's label, and its subject's description. A clip
    CMU's index leaves out is still listed; an index line with no clip is not."""
    (tmp_path / "data" / "008").mkdir(parents=True)
    (tmp_path / "data" / "117").mkdir()
    (tmp_path / "data" / "008" / "08_01.bvh").write_bytes(_bvh(276, 0.008333))
    (tmp_path / "data" / "008" / "08_02.bvh").write_bytes(b"HIERARCHY\n")
    (tmp_path / "data" / "117" / "117_01.bvh").write_bytes(_bvh(1000, 0.008333))
    (tmp_path / fetch.CMU_INDEX).write_text("CMU index\n\nSubject #8 (walking, with \u201cstyle\u201d)\n\n"
                                            "08_01\twalk\n08_02\tturn \n08_09\tnot fetched\n", encoding="utf-8")
    assert fetch._cmu_index(tmp_path).splitlines() == [
        "08_01\t   0.0-   2.3s\twalk\tsubject 8: walking, with \u201cstyle\u201d",
        "08_02\t\tturn\tsubject 8: walking, with \u201cstyle\u201d",
        "117_01\t   0.0-   8.3s\t\tsubject 117: not in CMU's index",
    ]


def test_a_verified_library_gets_this_releases_notes_without_a_download(tmp_path, monkeypatch, cmu):
    """Notes from an older install are rewritten by the next `install mocap`; the clips are not fetched again."""
    calls = []
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)}, calls)
    assert fetch.run(dest=tmp_path, out=lambda *a: None) == 0
    (tmp_path / "cmu" / "NOTES.md").write_text("an older release's notes")
    (tmp_path / "cmu" / "JOINT_MAPS.md").unlink()
    assert fetch.run(dest=tmp_path, out=lambda *a: None) == 0
    assert calls == [fetch.CMU_URL]
    assert all(got == want for got, want in _shipped(tmp_path / "cmu", "cmu"))


def test_a_verified_library_it_cannot_write_to_still_installs(tmp_path, monkeypatch, cmu):
    """A read-only copy keeps its clips usable: the missing notes are reported, and the install still succeeds."""
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)})
    assert fetch.run(dest=tmp_path, out=lambda *a: None) == 0
    lib = tmp_path / "cmu"
    (lib / "NOTES.md").unlink()
    lib.chmod(0o555)
    said = []
    try:
        assert fetch.run(dest=tmp_path, out=said.append) == 0
    finally:
        lib.chmod(0o755)
    assert any("notes for agents could not be written" in s and str(lib) in s for s in said), said
    assert not (lib / "NOTES.md").exists()


def test_a_download_that_does_not_match_the_pinned_commit_installs_nothing(tmp_path, monkeypatch, cmu):
    cmu[PREFIX + "data/008/08_01.bvh"] = b"altered"
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)})
    said = []
    assert fetch.run(dest=tmp_path, out=said.append) == 1
    assert any("does not match" in s for s in said)
    assert not (tmp_path / "cmu").exists() and not fetch._partial(tmp_path, "cmu").exists()


def test_a_failed_fetch_leaves_no_library_for_the_agent(tmp_path, monkeypatch, cmu):
    """A fetch that fails, or is refused for disk space, leaves nothing at `assets/mocap/`, so the runtime finds
    no library and the agent is shown no mocap skill or prompt."""
    from dycraft.animation.blender import runtime
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    lib = tmp_path / "assets" / "mocap"

    def broken(url):
        raise OSError("connection reset")
    monkeypatch.setattr(fetch, "_open", broken)
    said = []
    assert fetch.run(out=said.append) == 1 and any("connection reset" in s for s in said)
    assert not lib.exists() and runtime.mocap_library() is None
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    assert fetch.run(out=said.append) == 1 and any("GB free" in s for s in said)
    assert not lib.exists() and runtime.mocap_library() is None


def test_nothing_appears_under_assets_mocap_until_a_library_verifies(tmp_path, monkeypatch, cmu, dog):
    """A fetch killed mid-download (a closed terminal, a job's time limit) runs no cleanup. So the partial copy is
    assembled beside `assets/mocap/`, and the directory appears only with a verified library; a stale partial left
    by such a kill is cleared before the next fetch measures free space."""
    from dycraft.animation.blender import runtime
    monkeypatch.setenv("DYCRAFT_HOME", str(tmp_path))
    lib, seen = tmp_path / "assets" / "mocap", []

    def watched(url):
        seen.append((url, lib.exists(), (lib / "mann_dog").exists(), runtime.mocap_library()))
        return io.BytesIO(dog[url])
    monkeypatch.setattr(fetch, "_open", watched)
    stale = fetch._partial(lib, "cmu") / "data" / "001"
    stale.mkdir(parents=True)
    (stale / "01_99.bvh").write_bytes(b"left by a killed fetch")
    assert fetch.run(dog=True, accept=True, out=lambda *a: None) == 0
    assert seen[0] == (fetch.CMU_URL, False, False, None)
    assert seen[1][0] == fetch.MANN_URL and seen[1][2] is False
    assert not list(tmp_path.rglob("01_99.bvh")) and not list(tmp_path.rglob("*.partial"))
    assert (lib / "cmu" / ".verified").is_file() and (lib / "mann_dog" / ".verified").is_file()


def test_a_dropped_connection_fails_without_a_traceback(tmp_path, monkeypatch, cmu):
    """codeload streams the tarball chunked, so a connection that drops ends in `http.client.IncompleteRead`."""
    def dropped(url):
        raise fetch.http.client.IncompleteRead(b"partial")
    monkeypatch.setattr(fetch, "_open", dropped)
    said = []
    assert fetch.run(dest=tmp_path / "lib", out=said.append) == 1 and any("install mocap failed" in s for s in said)
    assert not (tmp_path / "lib").exists()


def test_a_library_this_command_did_not_install_is_never_replaced(tmp_path, monkeypatch, cmu):
    """Only a library carrying this command's `.verified` stamp is replaced. Anything else at `cmu/` (files placed
    by hand, a link to an unverified copy) stops the fetch before the download, and stays as it was."""
    calls = []
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)}, calls)
    lib, said = tmp_path / "lib", []
    own = lib / "cmu" / "my_own_clips" / "extra.bvh"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"mine")
    assert fetch.run(dest=lib, out=said.append) == 1
    assert any(str(lib / "cmu") in s and "move or remove it, then re-run" in s for s in said), said
    assert calls == [] and own.read_bytes() == b"mine"
    # A link to a copy without the stamp: left alone, before the download.
    other = tmp_path / "other"
    (lib / "cmu").rename(other)
    (lib / "cmu").symlink_to(other)
    assert fetch.run(dest=lib, out=said.append) == 1 and calls == [] and (other / "my_own_clips").is_dir()
    # A link to a verified copy counts as installed.
    (other / ".verified").write_text(fetch.CMU_DIGEST + "\n")
    assert fetch.run(dest=lib, out=said.append) == 0 and calls == []
    # A library an earlier run installed (stamped, but outdated) is replaced.
    (lib / "cmu").unlink()
    (lib / "cmu").mkdir()
    (lib / "cmu" / ".verified").write_text("an earlier commit\n")
    assert fetch.run(dest=lib, out=said.append) == 0 and calls == [fetch.CMU_URL]
    assert (lib / "cmu" / "data" / "008" / "08_01.bvh").is_file() and (lib / "cmu" / ".verified").read_text().strip() == fetch.CMU_DIGEST


def test_no_archive_member_is_written_outside_the_library(tmp_path, monkeypatch, cmu):
    cmu[PREFIX + "../escape.bvh"] = b"x"
    cmu[PREFIX + "data/../../escape2.bvh"] = b"x"
    cmu["elsewhere/data/001/01_09.bvh"] = b"x"
    _serve(monkeypatch, {fetch.CMU_URL: _tarball(cmu)})
    assert fetch.run(dest=tmp_path / "lib", out=lambda *a: None) == 0
    assert sorted(p.name for p in tmp_path.rglob("*.bvh")) == ["01_01.bvh", "08_01.bvh"]


@pytest.fixture
def dog(monkeypatch, cmu):
    archive = _zip({"D1_001_KAN01_001.bvh": b"dog stand", "D1_ex01_KAN01_001.bvh": b"dog sit", "notes.txt": b"x",
                    "../escape.bvh": b"x", "sub/dir.bvh": b"x"})
    monkeypatch.setattr(fetch, "MANN_CLIPS", 2)
    monkeypatch.setattr(fetch, "MANN_SHA256", hashlib.sha256(archive).hexdigest())
    return {fetch.CMU_URL: _tarball(cmu), fetch.MANN_URL: archive}


def test_the_dog_captures_are_not_downloaded_unless_their_terms_are_accepted(tmp_path, monkeypatch, dog):
    calls = []
    _serve(monkeypatch, dog, calls)
    said = []
    # Asked for and refused: the people are installed, and the command says so and fails.
    assert fetch.run(dog=True, dest=tmp_path, ask=lambda: "no", out=said.append) == 1
    refused = "Agents now see the CMU library. The dog captures are not installed: their terms were not accepted."
    assert any("CC BY-NC 4.0" in s for s in said) and refused in said
    assert fetch.MANN_URL not in calls and not (tmp_path / "mann_dog").exists() and (tmp_path / "cmu").is_dir()
    # Not a terminal and no flag: refused rather than assumed. Ctrl-D at the prompt is a refusal, not a traceback.
    monkeypatch.setattr(fetch.sys, "stdin", SimpleNamespace(isatty=lambda: False))   # also under `pytest -s`
    assert fetch.fetch_dog(tmp_path, out=lambda *a: None) is False

    def eof():
        raise EOFError
    assert fetch.run(dog=True, dest=tmp_path, ask=eof, out=lambda *a: None) == 1
    assert fetch.MANN_URL not in calls


def test_dog_captures_placed_by_hand_are_never_replaced(tmp_path, monkeypatch, dog):
    calls = []
    _serve(monkeypatch, dog, calls)
    own = tmp_path / "mann_dog" / "data" / "my_dog.bvh"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"mine")
    said = []
    assert fetch.run(dog=True, accept=True, dest=tmp_path, out=said.append) == 1
    assert any(str(tmp_path / "mann_dog") in s and "move or remove it, then re-run" in s for s in said), said
    assert fetch.MANN_URL not in calls and own.read_bytes() == b"mine"


def test_accepted_dog_captures_install_with_their_notes_and_span_index(tmp_path, monkeypatch, dog):
    _serve(monkeypatch, dog)
    assert fetch.run(dog=True, dest=tmp_path, ask=lambda: "accept", out=lambda *a: None) == 0
    lib = tmp_path / "mann_dog"
    assert sorted(p.name for p in (lib / "data").iterdir()) == ["D1_001_KAN01_001.bvh", "D1_ex01_KAN01_001.bvh"]
    assert not list(tmp_path.parent.glob("escape*")) and not (lib / "data" / "sub").exists()
    assert (lib / "index.txt").read_bytes() == fetch.DOG_INDEX.read_bytes()
    assert all(got == want for got, want in _shipped(lib, "mann_dog"))


def test_the_closing_line_says_whether_agents_see_the_dog_captures(tmp_path, monkeypatch, dog):
    """The dog captures are optional: the last line says whether this install has them, and how to add them."""
    _serve(monkeypatch, dog)
    said = []
    assert fetch.run(dest=tmp_path, out=said.append) == 0
    assert said[-1] == ("Agents now see the CMU library. The dog captures are optional and not installed; "
                        "add them with --dog.")
    assert fetch.run(dog=True, dest=tmp_path, ask=lambda: "accept", out=said.append) == 0
    assert said[-1] == "Agents now see the CMU library and the dog captures."
    assert fetch.run(dest=tmp_path, out=said.append) == 0          # installed by the earlier --dog run
    assert said[-1] == "Agents now see the CMU library and the dog captures."


def test_a_dog_archive_with_the_wrong_hash_installs_nothing(tmp_path, monkeypatch, dog):
    dog[fetch.MANN_URL] = dog[fetch.MANN_URL] + b"tampered"
    _serve(monkeypatch, dog)
    said = []
    assert fetch.run(dog=True, accept=True, dest=tmp_path, out=said.append) == 1
    assert any("sha256" in s for s in said) and not (tmp_path / "mann_dog").exists()


def test_the_installer_offers_the_mocap_step(monkeypatch):
    from dycraft import install
    seen = {}
    monkeypatch.setattr(fetch, "run", lambda **kw: seen.update(kw) or 0)
    assert install.main(["mocap", "--dog", "--accept-cc-by-nc"]) == 0
    assert seen == {"dog": True, "accept": True}
