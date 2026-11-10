"""Download the motion-capture libraries the retarget helpers read: `python -m dycraft.install mocap [--dog]`.

- **People:** the CMU Graphics Lab Motion Capture Database in its BVH conversion, from a GitHub mirror pinned to
  one commit. Every file is checked against that commit's git blob hashes.
- **Dogs (with `--dog`):** the MANN captures, from their authors' archive, checked by sha256, after the user accepts
  their CC BY-NC 4.0 terms.

Both go under `assets/mocap/` in dycraft's state directory, where `runtime.mocap_library` looks: `cmu/data/<SSS>/*.bvh`
and `mann_dog/data/*.bvh`. Beside each library's clips go its files for agents, in the layout the `mocap_reference`
skill describes: `NOTES.md`, `JOINT_MAPS.md`, and `index.txt`. Every run rewrites them in a verified library, so the
notes match the installed code. A library is written to a partial directory beside `assets/mocap/`
and moved into place only once it verifies, so a fetch that fails or is killed leaves nothing half-installed and no
`assets/mocap/` (the runtime takes any such directory as a library). A verified library is not fetched again; one
this command did not install is never replaced. Standard library only.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from ... import _home

CMU_COMMIT = "09a07f54f3bbb58797325f009282d0b2048a2871"
CMU_URL = f"https://codeload.github.com/una-dinosauria/cmu-mocap/tar.gz/{CMU_COMMIT}"
CMU_CLIPS = 2548
CMU_BYTES = 3_229_831_313
#: sha256 of the sorted "<path> <git blob sha1>" lines of every BVH at CMU_COMMIT.
CMU_DIGEST = "4768c3a7ae9c98ab91a9d2456a29e60ebde87d84684370966beb49866ee6ef56"
CMU_CLIP = re.compile(r"data/\d{3}/\d+_\d+\.bvh")
#: CMU's own index: a heading per subject, `Subject #<n> (<description>)`, then one `<SS>_<CC>\t<label>` line per clip.
CMU_INDEX = "cmu-mocap-index-text.txt"
#: Kept beside the clips: the conversion's own notes and CMU's index.
CMU_EXTRA = ("READMEFIRST.txt", "README.md", CMU_INDEX)

MANN_URL = "https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip"
MANN_SHA256 = "0ecd909b673cfca5e20c3f568376dcf27d7da3bd153957c2b034d95af366132a"
MANN_CLIPS = 51
#: A clip is a plain BVH file name with no path (the archive itself is pinned by sha256).
MANN_CLIP = re.compile(r"[A-Za-z0-9_]+\.bvh")
#: Each library's files for agents, shipped in `libraries/<name>/` and copied beside its clips.
LIBRARIES = Path(__file__).resolve().parent / "libraries"
SHIPPED = {"cmu": ("NOTES.md", "JOINT_MAPS.md"), "mann_dog": ("NOTES.md", "JOINT_MAPS.md", "index.txt")}
#: dycraft's labels of the captures' spans, which `dog_retarget.inspect_bvh` reads beside the clips.
DOG_INDEX = LIBRARIES / "mann_dog" / "index.txt"

CMU_TERMS = """\
CMU Graphics Lab Motion Capture Database (BVH conversion by B. Hahne), ~1 GB download, 3.2 GB on disk.
  Free for research and for use in products; the data itself may not be resold, even converted.
  If you publish results obtained with it, CMU asks you to acknowledge: "The data used in this project was
  obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217."
"""
MANN_TERMS = """\
MANN dog captures: Zhang, Starke, Komura, and Saito, "Mode-Adaptive Neural Networks for Quadruped Motion
Control" (SIGGRAPH 2018), 33 MB. Licensed CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/):
attribution required, non-commercial use only. Films and other work made with them carry the same terms.
"""


class FetchError(RuntimeError):
    pass


def root() -> Path:
    """Where the libraries go: `assets/mocap` in dycraft's state directory (`$DYCRAFT_HOME` when set)."""
    return _home.state_path("assets", "mocap")


def _open(url):
    """The response for `url`; tests replace this, so no test reaches the network."""
    req = urllib.request.Request(url, headers={"User-Agent": "dycraft-install"})
    return urllib.request.urlopen(req, timeout=60)


def _stamp(directory: Path) -> str:
    try:
        return (directory / ".verified").read_text().strip()
    except OSError:
        return ""


def _replace(partial: Path, final: Path) -> None:
    if final.exists():
        if final.is_symlink() or not _stamp(final):           # `_check_ours` stops before any download
            raise FetchError(f"{final} was not installed by this command; nothing was replaced")
        shutil.rmtree(final)
    partial.rename(final)


def _check_ours(final: Path) -> None:
    """Stop, before any download, if `final` holds anything this command did not install
    (no `.verified` stamp), or is a link to an unverified or outdated copy: replacing
    either would delete the user's files. A link to a verified copy counts as installed."""
    if final.is_symlink() or (final.exists() and not _stamp(final)):
        raise FetchError(f"{final} already exists and was not installed by this command, so it is left as it is; "
                         "move or remove it, then re-run")


def _partial(dest: Path, name: str) -> Path:
    """Where a library is assembled: beside `dest`, on its filesystem, so the final move is
    atomic and nothing appears under `dest` until a library verifies."""
    real = dest.resolve()
    return real.parent / f".{real.name}-{name}.partial"


def _bvh_seconds(path: Path) -> float | None:
    """A BVH clip's length from its header's `Frames:` and `Frame Time:` lines, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(16384).decode("latin-1")
    except OSError:
        return None
    frames = re.search(r"^Frames:\s*(\d+)", head, re.M)
    step = re.search(r"^Frame Time:\s*([0-9.eE+-]+)", head, re.M)
    return int(frames.group(1)) * float(step.group(1)) if frames and step else None


def _span(path: Path) -> str:
    seconds = _bvh_seconds(path)
    return "" if seconds is None else f"{0:6.1f}-{seconds:6.1f}s"


def _cmu_index(lib: Path) -> str:
    """CMU's index in the libraries' common format, `<clip>\t<start>-<end>s\t<label>\t<notes>`: one line per clip in
    `lib`, with its whole span, CMU's label, and its subject's description. Clips CMU's index leaves out are listed
    with no label."""
    rows, listed, subject = [], set(), ""
    try:
        text = (lib / CMU_INDEX).read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        heading = re.match(r"^Subject #(\d+) \((.*)\)\s*$", line)
        if heading:
            subject = f"subject {int(heading.group(1))}: {heading.group(2).strip()}"
            continue
        m = re.match(r"^(\d+)_(\d+)\t(.*?)\s*$", line)
        clip = m and f"{m.group(1)}_{m.group(2)}"
        bvh = m and lib / "data" / f"{int(m.group(1)):03d}" / f"{clip}.bvh"
        if m and clip not in listed and bvh.is_file():
            rows.append(f"{clip}\t{_span(bvh)}\t{m.group(3)}\t{subject}")
            listed.add(clip)
    for bvh in sorted((lib / "data").glob("*/*.bvh")):
        if bvh.stem not in listed:
            rows.append(f"{bvh.stem}\t{_span(bvh)}\t\tsubject {int(bvh.parent.name)}: not in CMU's index")
    return "".join(r + "\n" for r in rows)


def _write_files(directory: Path, name: str, out=print) -> bool:
    """Copy library `name`'s shipped files into `directory`, and for CMU write its `index.txt`. Where they can't be
    written (a read-only copy, or a link to one), say so and go on: the clips stay usable, without the notes."""
    try:
        for f in SHIPPED[name]:
            shutil.copyfile(LIBRARIES / name / f, directory / f)
        if name == "cmu":
            (directory / "index.txt").write_text(_cmu_index(directory), encoding="utf-8")
    except OSError as e:
        out(f"note: the {name} library's notes for agents could not be written into {directory} ({e})")
        return False
    return True


def _free(path: Path) -> int:
    """Free bytes where `path` is or would be created, without creating it."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def fetch_cmu(dest: Path, out=print) -> None:
    final, partial = dest / "cmu", _partial(dest, "cmu")
    if _stamp(final) == CMU_DIGEST:
        _write_files(final, "cmu", out=out)
        out(f"CMU library already fetched and verified: {final}")
        return
    _check_ours(final)
    if partial.exists():                                        # left by a fetch that was killed
        shutil.rmtree(partial)
    free = _free(partial.parent)
    if free < CMU_BYTES * 1.1:
        raise FetchError(f"{dest} has {free / 1e9:.1f} GB free; the CMU library needs {CMU_BYTES * 1.1 / 1e9:.1f} GB")
    partial.mkdir(parents=True)
    prefix = f"cmu-mocap-{CMU_COMMIT}/"
    hashes = []
    out(f"downloading {CMU_URL}")
    try:
        with _open(CMU_URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.startswith(prefix):
                    continue
                rel = m.name[len(prefix):]
                if not (CMU_CLIP.fullmatch(rel) or rel in CMU_EXTRA):
                    continue
                target = partial / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha1(b"blob %d\0" % m.size)
                with tar.extractfile(m) as src, open(target, "wb") as dst:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        h.update(chunk)
                        dst.write(chunk)
                if rel.endswith(".bvh"):
                    hashes.append(f"{rel} {h.hexdigest()}")
                    if len(hashes) % 250 == 0:
                        out(f"  {len(hashes)} / {CMU_CLIPS} clips")
        digest = hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()
        if len(hashes) != CMU_CLIPS or digest != CMU_DIGEST:
            raise FetchError(f"the CMU download does not match commit {CMU_COMMIT[:8]}: {len(hashes)} clips "
                             f"(expected {CMU_CLIPS}), digest {digest[:12]} (expected {CMU_DIGEST[:12]}); "
                             "nothing was installed")
        _write_files(partial, "cmu", out=out)
        (partial / ".verified").write_text(CMU_DIGEST + "\n")
        dest.mkdir(parents=True, exist_ok=True)
        _replace(partial, final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    out(f"CMU library: {CMU_CLIPS} clips in {final}")


def _accepted(accept: bool, ask) -> bool:
    if accept:
        return True
    if ask is None:
        if not sys.stdin.isatty():
            return False
        ask = lambda: input('Type "accept" to accept these terms and download the dog captures: ')
    try:
        return ask().strip().lower() == "accept"
    except EOFError:                                            # Ctrl-D at the prompt
        return False


def fetch_dog(dest: Path, accept: bool = False, ask=None, out=print) -> bool:
    """Fetch the MANN captures once their terms are accepted; False when they were not."""
    final, partial = dest / "mann_dog", _partial(dest, "mann_dog")
    if _stamp(final) == MANN_SHA256:
        _write_files(final, "mann_dog", out=out)
        out(f"Dog captures already fetched and verified: {final}")
        return True
    _check_ours(final)
    out(MANN_TERMS)
    if not _accepted(accept, ask):
        out("Dog captures not downloaded: their terms were not accepted (pass --accept-cc-by-nc to accept "
            "without being asked).")
        return False
    out(f"downloading {MANN_URL}")
    with _open(MANN_URL) as resp:
        blob = resp.read()
    got = hashlib.sha256(blob).hexdigest()
    if got != MANN_SHA256:
        raise FetchError(f"the dog archive's sha256 is {got[:12]}, expected {MANN_SHA256[:12]}; nothing was installed")
    if partial.exists():                                        # left by a fetch that was killed
        shutil.rmtree(partial)
    (partial / "data").mkdir(parents=True)
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = [n for n in z.namelist() if MANN_CLIP.fullmatch(n)]
            if len(names) != MANN_CLIPS:
                raise FetchError(f"the dog archive holds {len(names)} clips, expected {MANN_CLIPS}; nothing was installed")
            for n in names:
                (partial / "data" / n).write_bytes(z.read(n))
        _write_files(partial, "mann_dog", out=out)
        (partial / ".verified").write_text(MANN_SHA256 + "\n")
        dest.mkdir(parents=True, exist_ok=True)
        _replace(partial, final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    out(f"Dog captures: {MANN_CLIPS} clips in {final}")
    return True


def run(dog: bool = False, accept: bool = False, dest: Path | None = None, ask=None, out=print) -> int:
    """The `install mocap` step: the CMU library, then the dog captures if asked. 0 when
    everything asked for is installed; 1 otherwise, including dog terms not accepted."""
    dest = Path(dest) if dest else root()
    out(CMU_TERMS)
    dogs = True
    try:
        fetch_cmu(dest, out=out)
        if dog:
            dogs = fetch_dog(dest, accept=accept, ask=ask, out=out)
    except (FetchError, OSError, http.client.HTTPException, tarfile.TarError, zipfile.BadZipFile) as e:
        out(f"install mocap failed: {e}")
        return 1
    finally:
        try:                                                    # an empty assets/mocap/ would read as a library
            dest.rmdir()
        except OSError:
            pass
    if not dogs:
        out("Agents now see the CMU library. The dog captures are not installed: their terms were not accepted.")
        return 1
    if dog or _stamp(dest / "mann_dog") == MANN_SHA256:            # this run's, or an earlier --dog run's
        out("Agents now see the CMU library and the dog captures.")
    else:
        out("Agents now see the CMU library. The dog captures are optional and not installed; add them with --dog.")
    return 0
