"""The imported-asset library a scene or animation run retrieves from (`assets/imported/`).

A run is given the library, read-only at `<run>/references/assets/`, with
`INDEX.md` there describing every bundle it may use, and `load_asset("<bundle>")`
stages the one it picks on demand (`preload_asset.stage_library_asset`). An
asset passed with `--preload-asset` wins on a name clash.

What the index says comes from two places:

* the bundle itself — `import.json` and `ir.json` give the size, the skeleton,
  the bone count, the clips and, for a URDF robot, its joints; and
* `catalogs/asset_library.json`, a tracked, hand-written file with what no
  bundle records: what the thing *is* (category, a one-line description,
  search words) and whether it is offered at all. A bundle the file does not
  list is offered with its facts and no description, so a newly imported one
  is never invisible; one marked `hidden` (a superseded variant, a debugging
  import) is left out of the index and refused by `load_asset`, with the
  reason, so an agent cannot wander into it by name.

Thumbnails are rendered offline by `python -m dycraft.agent.assets.asset_library
--thumbnails` and written into each bundle as `thumb.png`; the index points at
them so an agent can look before it chooses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ... import _home

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Overrides the library location (tests; a checkout without `assets/`).
LIBRARY_ENV = "DYCRAFT_ASSET_LIBRARY"
CURATION_PATH = _REPO_ROOT / "catalogs" / "asset_library.json"
INDEX_MD = "INDEX.md"
INDEX_JSON = "index.json"
THUMB = "thumb.png"
CREDIT_FILES = ("ROCKETBOX_CREDITS.md", "SKETCHFAB_CREDITS.md", "OBJAVERSE_CREDITS.md")


class AssetLibraryError(ValueError):
    pass


def library_root() -> Path | None:
    """The library directory, or None when there is none.

    `$DYCRAFT_ASSET_LIBRARY`, else `assets/imported` under the user-state root
    (`_home`): the source tree, or the launch directory of an installed copy --
    the same directory `stage_references` links into a run."""
    env = os.environ.get(LIBRARY_ENV)
    root = Path(env) if env else _home.state_path("assets", "imported")
    root = root.resolve()
    return root if root.is_dir() else None


def installed_index(root: Path | None = None) -> Path | None:
    """The library's `INDEX.md` (in `library_root()`, or `root`), or None. Agents are told about the library only
    when it exists: the prompt's asset-library section and the environment check's `optional.library` both read it
    from here."""
    root = library_root() if root is None else root
    index = root / INDEX_MD if root is not None else None
    return index if index is not None and index.is_file() else None


def load_curation(path: Path = CURATION_PATH, lib: Path | None = None) -> dict:
    """`{bundle: entry}`: the hand-written catalog, with the installed library's `catalog.json` over it.

    `install library` puts the catalog of the library version it fetched beside the bundles; its category,
    description, tags, and caveats win, so a later library version needs no code release. What only the hand-written catalog
    says (`hidden`) still holds. `{}` when neither exists."""
    out = {}
    try:
        out = {n: dict(e) for n, e in (json.loads(Path(path).read_text()).get("assets") or {}).items()}
    except (OSError, ValueError):
        pass
    lib = library_root() if lib is None else lib
    try:
        installed = json.loads((lib / "catalog.json").read_text()).get("assets") or {} if lib else {}
    except (OSError, ValueError):
        installed = {}
    for name, entry in installed.items():
        out.setdefault(name, {}).update({k: entry[k] for k in ("category", "description", "tags", "caveats") if entry.get(k)})
    return out


def bundles(lib: Path) -> list:
    """Every directory under `lib` that holds a loadable bundle."""
    return sorted(p for p in lib.iterdir()
                  if p.is_dir() and (p / "ir.json").is_file())


def _credits(lib: Path) -> dict:
    """`{bundle: credit}` from the credits files that ship with the library.

    Rocketbox's file lists the local FBX each avatar came from, which says
    nothing to a reader, so those get the collection's own credit; the other
    files carry title, author, license, and URL per line, kept minus any local
    path and extent note."""
    out = {}
    for name in CREDIT_FILES:
        try:
            text = (lib / name).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.lstrip().startswith("|") and len(cells) >= 5 and (lib / cells[0]).is_dir():
                # table row: bundle | title | author | license | source
                out.setdefault(cells[0], f'"{cells[1]}" by {cells[2]}, {cells[3]}, {cells[4]}')
                continue
            for b in re.findall(r"`([A-Za-z0-9_.-]+)`", line):
                if not (lib / b).is_dir() or b in out:
                    continue
                if name.startswith("ROCKETBOX"):
                    out[b] = "Microsoft Rocketbox avatar, MIT license"
                    continue
                c = re.sub(r"`[^`]*`\s*(<-|:)?\s*", "", line)
                c = re.sub(r"\(extent[^)]*\)", "", c)
                c = re.sub(r"(^|\s)/\S+", r"\1", c)
                out[b] = re.sub(r"\s+", " ", c.strip("-| ")).strip(" ,;")
    return out


def bundle_facts(bundle: Path) -> dict:
    """What the bundle itself says: kind, size, skeleton, clips, joints."""
    from .preload_asset import _describe
    ir = json.loads((bundle / "ir.json").read_text())
    try:
        imp = json.loads((bundle / "import.json").read_text())
    except (OSError, ValueError):
        imp = {}
    desc = _describe(ir)
    rig = ir.get("native_rig") or {}
    facts = {"name": bundle.name, "object_name": ir.get("name"),
             "size_m": desc.get("size")}
    if rig:
        facts["kind"] = "character rig"
        facts["bones"] = len(rig.get("bones") or [])
        facts["skeleton"] = (rig.get("anatomy") or {}).get("skeleton", "unknown")
        facts["clips"] = [a if isinstance(a, str) else (a.get("clip") or a.get("name")) for a in rig.get("actions") or []]
        facts["shape_keys"] = len(rig.get("shape_keys") or [])
        if rig.get("blend"):
            # For an agent working in Blender directly (a bare arm): append the
            # armature and mesh from this file. A DyCraft arm uses load_asset.
            facts["blend"] = f"{bundle.name}/{rig['blend']}"
    elif imp.get("source_urdf"):
        facts["kind"] = "robot (URDF)"
        facts["joints"] = desc.get("n_joints")
        facts["urdf"] = Path(imp["source_urdf"]).name
    else:
        facts["kind"] = "articulated parts"
        facts["joints"] = desc.get("n_joints")
    if (bundle / THUMB).is_file():
        facts["thumbnail"] = f"{bundle.name}/{THUMB}"
    return facts


def build_index(lib: Path, curation: dict | None = None) -> list:
    """One row per offered bundle: its facts merged with its catalog entry. A bundle whose `ir.json` cannot be read
    is left out, and named on stderr, so one broken import of your own does not hide the rest of the library."""
    curation = load_curation() if curation is None else curation
    credits = _credits(lib)
    rows = []
    for b in bundles(lib):
        entry = curation.get(b.name) or {}
        if entry.get("hidden"):
            continue
        try:
            row = bundle_facts(b)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"asset library: {b.name} left out of the index, its ir.json cannot be read "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            continue
        for k in ("category", "description", "tags", "caveats"):
            if entry.get(k):
                row[k] = entry[k]
        if b.name in credits:
            row["credit"] = credits[b.name]
        rows.append(row)
    return rows


def _size(s) -> str:
    return "x".join(f"{v:.2f}" for v in s) + " m" if s else "?"


def render_index_md(rows: list) -> str:
    lines = [
        "# Asset library",
        "",
        "Characters and robots you may place with `load_asset(\"<name>\")`, which stages the",
        "bundle into your run the first time you load it. Pick by what the brief describes;",
        "`grep -i` this file for a species, a role, or a kind of robot, and open a bundle's",
        "`thumb.png` to see it. `size` is the bounding box in the asset's rest pose",
        "(x = side to side, y = front to back, z = height); a character's x includes its",
        "arms in a T-pose. Each bundle's `ir.json` (and `import.json`) holds the rest:",
        "bone names, clip names, joint limits.",
        "",
    ]
    order = ["person", "animal", "robot", "creature", "other"]
    by_cat: dict = {}
    for r in rows:
        by_cat.setdefault(r.get("category") or "other", []).append(r)
    for cat in sorted(by_cat, key=lambda c: (order.index(c) if c in order else len(order), c)):
        lines += [f"## {cat}", ""]
        for r in sorted(by_cat[cat], key=lambda r: r["name"]):
            head = f"- `{r['name']}` — {r.get('description') or '(no description)'}"
            lines.append(head)
            bits = [r["kind"], f"size {_size(r.get('size_m'))}"]
            if r.get("bones"):
                sk = r.get("skeleton")
                bits.append(f"{r['bones']} bones" + (f" ({sk} skeleton)" if sk and sk != "unknown" else ""))
            if r.get("joints"):
                bits.append(f"{r['joints']} joints")
            if r.get("clips"):
                bits.append("clips: " + ", ".join(r["clips"][:6]) + (" …" if len(r["clips"]) > 6 else ""))
            if r.get("tags"):
                bits.append("tags: " + ", ".join(r["tags"]))
            if r.get("thumbnail"):
                bits.append(f"thumbnail `{r['thumbnail']}`")
            if r.get("blend"):
                bits.append(f"blend `{r['blend']}`")
            lines.append("  " + "; ".join(bits))
            if r.get("caveats"):
                lines.append("  caveats: " + "; ".join(r["caveats"]))
            if r.get("credit"):
                lines.append(f"  credit: {r['credit']}")
        lines.append("")
    return "\n".join(lines)


def library_listing(lib: Path | None = None) -> str:
    """One line per offered asset (name, category, description, size): what a
    brief rewriter needs to know about the characters the author will pick.
    Read from the generated `index.json` when present; `""` with no library."""
    lib = library_root() if lib is None else lib
    if lib is None:
        return ""
    try:
        rows = json.loads((lib / INDEX_JSON).read_text())["assets"]
    except (OSError, ValueError, KeyError):
        rows = build_index(lib)
    return "\n".join(f"- `{r['name']}` ({r.get('category') or r['kind']}): "
                     f"{r.get('description') or r.get('object_name')}; size {_size(r.get('size_m'))}"
                     for r in sorted(rows, key=lambda r: (r.get("category") or "", r["name"])))


def write_index(lib: Path, curation: dict | None = None) -> list:
    """Write `index.json` and `INDEX.md` into `lib`. Each goes to a sibling temporary file first and replaces the old
    one in a single rename, so a run launched meanwhile reads the old index or the new one, never a partial one."""
    rows = build_index(lib, curation)
    for name, text in ((INDEX_JSON, json.dumps({"schema": "dycraft.asset_library.v1", "assets": rows}, indent=1)),
                       (INDEX_MD, render_index_md(rows))):
        tmp = lib / f".{name}.{os.getpid()}.tmp"
        tmp.write_text(text)
        os.replace(tmp, lib / name)
    return rows


def resolve_library_asset(name: str, lib: Path | None = None,
                          curation: dict | None = None) -> Path:
    """The bundle directory `load_asset(name)` may stage, or `AssetLibraryError`."""
    lib = library_root() if lib is None else lib
    if lib is None:
        raise AssetLibraryError("this checkout has no asset library (assets/imported); install it with "
                                "`python -m dycraft.install library`")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or "") or name.startswith("."):
        raise AssetLibraryError(f"{name!r} is not a bundle name")
    bundle = lib / name
    if not (bundle / "ir.json").is_file():
        raise AssetLibraryError(f"no asset named {name!r} in the library")
    entry = (load_curation() if curation is None else curation).get(name) or {}
    if entry.get("hidden"):
        raise AssetLibraryError(f"{name!r} is not offered: {entry['hidden']}")
    return bundle


# ---------------------------------------------------------------- thumbnails

_RIG_THUMB = r'''
import bpy, sys, math
from mathutils import Vector
blend, out, n = sys.argv[-3], sys.argv[-2], int(sys.argv[-1])
bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False)
s = bpy.context.scene
meshes = [o for o in s.objects if o.type == "MESH"]
for o in s.objects:
    if o.type == "ARMATURE" and o.animation_data: o.animation_data.action = None
dg = bpy.context.evaluated_depsgraph_get()
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
c = (lo + hi) / 2; r = max((hi - lo).length / 2, 0.05)
cd = bpy.data.cameras.new("c"); cd.lens = 50; cam = bpy.data.objects.new("c", cd); s.collection.objects.link(cam); s.camera = cam
s.render.engine = "BLENDER_WORKBENCH"; sh = s.display.shading; sh.light = "STUDIO"; sh.color_type = "TEXTURE"
sh.show_shadows = True; sh.show_cavity = True
s.render.resolution_x = s.render.resolution_y = 384
s.render.film_transparent = False
if s.world is None: s.world = bpy.data.worlds.new("w")
s.world.color = (0.62, 0.64, 0.66)
for i, (az, el) in enumerate(((-35, 15), (145, 15))[:n]):
    d = Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), math.tan(math.radians(el)))).normalized()
    cam.location = c + d * r * 2.9; cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
    s.render.filepath = f"{out}_{i}.png"; bpy.ops.render.render(write_still=True)
'''


def _blender_python() -> str | None:
    try:
        lines = _home.state_path(".blender_python").read_text().splitlines()
    except OSError:
        return None
    lines = [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]
    return lines[-1] if lines else None


def _hstack(parts, out: Path) -> None:
    args = ["ffmpeg", "-v", "error", "-y"]
    for p in parts:
        args += ["-i", str(p)]
    args += ["-filter_complex", f"hstack=inputs={len(parts)}"] if len(parts) > 1 else []
    subprocess.run(args + [str(out)], check=True)


def render_thumbnail(bundle: Path, tmp: Path) -> Path:
    """Write `<bundle>/thumb.png`: two views of a rig (Blender), or the
    renderer's views of a URDF robot."""
    ir = json.loads((bundle / "ir.json").read_text())
    tmp.mkdir(parents=True, exist_ok=True)
    out = bundle / THUMB
    rig = ir.get("native_rig") or {}
    if rig.get("blend"):
        py = _blender_python()
        if not py:
            raise AssetLibraryError("no .blender_python; cannot render a rig thumbnail")
        script = tmp / "rig_thumb.py"
        script.write_text(_RIG_THUMB)
        stem = tmp / bundle.name
        subprocess.run([py, str(script), str(bundle / rig["blend"]), str(stem), "2"],
                       check=True, capture_output=True)
        _hstack([f"{stem}_0.png", f"{stem}_1.png"], out)
        return out
    from ...render import RenderRequest, render_request
    res = render_request(RenderRequest(layer="object", ir=ir, catalog=None, target="all",
                                       out_dir=tmp / bundle.name, step=0, asset_base_dir=bundle))
    paths = [Path(p) for p in res.ordered_paths()]
    # The renderer's six orthographic views; front and left show any robot's
    # face and profile between them, whichever way it was imported facing.
    pick = [p for v in ("front", "left") for p in paths if p.stem.endswith("_" + v)] or paths[:2]
    _hstack(pick, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dycraft.agent.assets.asset_library",
                                 description="Rebuild the asset library's INDEX.md / index.json.")
    ap.add_argument("--library", help="library dir (default: assets/imported or $%s)" % LIBRARY_ENV)
    ap.add_argument("--thumbnails", action="store_true", help="(re)render every bundle's thumb.png")
    ap.add_argument("--only", nargs="*", help="limit --thumbnails to these bundles")
    ap.add_argument("--missing", action="store_true", help="render only bundles without a thumb.png")
    args = ap.parse_args(argv)
    lib = Path(args.library).resolve() if args.library else library_root()
    if lib is None:
        print("no asset library", file=sys.stderr)
        return 2
    if args.thumbnails:
        import tempfile
        cur = load_curation()
        with tempfile.TemporaryDirectory() as td:
            for b in bundles(lib):
                if args.only and b.name not in args.only:
                    continue
                if (cur.get(b.name) or {}).get("hidden"):
                    continue
                if args.missing and (b / THUMB).is_file():
                    continue
                try:
                    render_thumbnail(b, Path(td))
                    print("thumb", b.name)
                except Exception as e:                              # noqa: BLE001
                    print(f"thumb FAILED {b.name}: {type(e).__name__}: {e}", file=sys.stderr)
    rows = write_index(lib)
    print(f"{len(rows)} assets indexed in {lib / INDEX_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
