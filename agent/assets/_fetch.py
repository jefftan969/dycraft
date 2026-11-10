"""Download the reviewed asset library (the dycraft-assets dataset) or the paper's films (the dycraft-films dataset)
from Hugging Face.

    python -m dycraft.install library [--names A,B | --category person] [--revision TAG] [--prune]
    python -m dycraft.install films [--brief B05,B13] [--revision TAG]

The asset dataset is mirrored under `assets/.dataset/` in DyCraft's state directory ($DYCRAFT_HOME when set), the
films dataset under `.datasets/dycraft-films/`. Each library bundle is linked into `assets/imported/<name>`, where the library is read from, beside any bundle you imported
yourself; the G1's URDF description is linked to `assets/robots/g1_description`, where the bare benchmark arm looks.
The library's `catalog.json` lists the reviewed characters of its version with their caveats and credits. Running
`library` again fetches only what is new or changed (the Hub client keeps each file's hash), so a later, larger
library installs in place; a bundle the new catalog no longer lists is reported, and unlinked with --prune. Each run
then writes the library's `INDEX.md` and `index.json` (`asset_library.write_index`); the index is what tells agents
the library exists, since the animation prompt shows its asset-library section only with it. Films
are linked the same way to `films/<Bnn_slug>/` under the state directory, named as in the dataset, since the scorer
reads the brief id from the name. --revision pins a dataset tag or commit.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ... import _home

ASSETS_URL = "https://huggingface.co/datasets/jefftan969/dycraft-assets"   # the library, at the dataset's root
FILMS_URL = "https://huggingface.co/datasets/jefftan969/dycraft-films"
ASSETS_REPO, FILMS_REPO = (u.split("/datasets/", 1)[1] for u in (ASSETS_URL, FILMS_URL))
REPO_TYPE = "dataset"
CATALOG = "catalog.json"
G1_DESCRIPTION = "robots/g1_description"


class FetchError(RuntimeError):
    pass


def _hub():
    """The Hugging Face client, imported here so only these two steps need it; tests replace this."""
    try:
        import huggingface_hub
    except ImportError as e:
        raise FetchError("this step needs huggingface_hub: pip install huggingface_hub") from e
    return huggingface_hub


def mirror(repo: str = ASSETS_REPO) -> Path:
    if repo == ASSETS_REPO:
        return _home.state_path("assets", ".dataset")
    return _home.state_path(".datasets", repo.split("/")[-1])


def _download(repo: str, patterns: list, revision: str | None) -> Path:
    dest = mirror(repo)
    dest.mkdir(parents=True, exist_ok=True)
    hub = _hub()
    try:
        hub.snapshot_download(repo_id=repo, repo_type=REPO_TYPE, allow_patterns=patterns, local_dir=str(dest),
                              revision=revision)
    except Exception as e:      # the Hub client's errors (not found, no access, offline) raise a dozen types
        status = getattr(getattr(e, "response", None), "status_code", None)
        why = ("the dataset is private or your token cannot read it: log in with `hf auth login` or set HF_TOKEN"
               if status in (401, 403) else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        raise FetchError(f"could not download from https://huggingface.co/datasets/{repo} ({why})") from e
    return dest


def _link(target: Path, link: Path) -> str:
    """Point `link` at `target`; a real directory of the same name (your own import) is left alone."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return "kept"
        link.unlink()
    elif link.exists():
        return "skipped: a bundle of that name is already there and was not installed by this command"
    link.symlink_to(target, target_is_directory=True)
    return "linked"


def select(catalog: dict, names: list | None, category: str | None) -> list:
    entries = catalog.get("assets") or {}
    if names:
        missing = [n for n in names if n not in entries]
        if missing:
            raise FetchError(f"not in library {catalog.get('library_version')}: {', '.join(missing)}")
        return list(names)
    return [n for n, e in entries.items() if not category or e.get("category") == category]


def run_library(names: list | None = None, category: str | None = None, revision: str | None = None,
                prune: bool = False, out=None) -> int:
    out = out or sys.stdout
    try:
        dest = _download(ASSETS_REPO, [CATALOG], revision)
        catalog = json.loads((dest / CATALOG).read_text())
        chosen = select(catalog, names, category)
        _download(ASSETS_REPO, [f"bundles/{n}/**" for n in chosen] + [f"{G1_DESCRIPTION}/**"], revision)
    except FetchError as e:
        print(f"[library] {e}", file=out)
        return 1
    lib = _home.state_path("assets", "imported")
    for where in (lib, lib.parent):
        if where.is_symlink() and not where.exists():
            # A link to something gone: linking into it cannot work, and what it pointed at may be a library of
            # yours, on a disk not mounted now. Said, not replaced.
            print(f"[library] {where} is a broken link to {os.readlink(where)}: restore what it points at, or remove "
                  f"the link (`rm {where}`), then run this again", file=out)
            return 1
    states = {n: _link(dest / "bundles" / n, lib / n) for n in chosen}
    _link(dest / G1_DESCRIPTION, _home.state_path("assets", "robots", "g1_description"))
    (lib / "catalog.json").unlink(missing_ok=True)
    (lib / "catalog.json").symlink_to(dest / CATALOG)
    listed = set(catalog.get("assets") or {})
    stale = sorted(p.name for p in lib.iterdir() if p.is_symlink() and p.name != "catalog.json"
                   and str(os.readlink(p)).startswith(str(dest)) and p.name not in listed)
    for n in stale:
        if prune:
            (lib / n).unlink()
    skipped = {n: s for n, s in states.items() if s.startswith("skipped")}
    print(f"[library] version {catalog.get('library_version')}: {len(chosen)} characters in {lib}"
          f" ({sum(s == 'linked' for s in states.values())} new)", file=out)
    for n, s in skipped.items():
        print(f"[library]   {n}: {s}", file=out)
    if stale:
        print(f"[library] no longer in this version{' (unlinked)' if prune else ' (keep with no flag; --prune unlinks)'}: "
              f"{', '.join(stale)}", file=out)
    # The index is what tells agents the library exists: the prompt's asset-library section is shown only with it.
    # It lists every bundle now in the library, your own imports included, after this version's links and --prune.
    from .asset_library import INDEX_MD, write_index
    try:
        rows = write_index(lib)             # a bundle of your own that cannot be read is left out, and named
    except OSError as e:
        print(f"[library] could not write {lib / INDEX_MD} ({e}), so agents are not told about the library; run "
              f"`python -m dycraft.install library` again once it can be written", file=out)
        return 1
    print(f"[library] {lib / INDEX_MD} lists {len(rows)} characters for agents", file=out)
    return 0


def run_films(briefs: list | None = None, revision: str | None = None, out=None) -> int:
    out = out or sys.stdout
    patterns = [f"films/{b.upper()}_*/**" for b in briefs] if briefs else ["films/**"]
    try:
        dest = _download(FILMS_REPO, patterns, revision)
    except FetchError as e:
        print(f"[films] {e}", file=out)
        return 1
    have = sorted(p.name for p in (dest / "films").iterdir() if p.is_dir()) if (dest / "films").is_dir() else []
    got = [n for n in have if not briefs or any(n.upper().startswith(b.upper() + "_") for b in briefs)]
    missing = [b for b in briefs or [] if not any(n.upper().startswith(b.upper() + "_") for n in got)]
    films = _home.state_path("films")
    for n in got:
        _link(dest / "films" / n, films / n)
    if missing:
        print(f"[films] not in the dataset: {', '.join(missing)}", file=out)
        return 1
    # Scoring reads a film's blend as it is installed, and the viewer prepares an installed film itself on first open.
    print(f"[films] {len(got)} films in {films}. Score one with `python -m dycraft.benchmark.score {films}/<film>`, "
          f"or play them in the viewer with `python -m dycraft.viewer --runs-dir {films}`.", file=out)
    return 0
