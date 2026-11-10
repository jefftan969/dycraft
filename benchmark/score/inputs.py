"""What a path names, and which geometry represents it.

* **A DyCraft run, arm or bundle** (a bundle with `design_state/baseline.json` or `baseline.json`): its
  `evaluation/` is the delivered animation and is read as it is; the baseline gives the joint tree, and
  `animation.blend` the materials and armature rigs.
* **Anything else with an `animation.blend`, or a `.blend` file** (a bare agent's staged arm, another method's
  film): the blend is exported into the `dycraft.evaluation` format by `_export_child.py`, which also records
  visibility, Blender parenting and material classes. A bare arm's staged `bundle/evaluation/` (`stage.py`) is not
  used: it lacks those records and keeps captions and camera overlays, which are not physical geometry.

A DyCraft bundle with no complete evaluation (a run that did not finish, or a film shipped as its blend) is exported
from its blend the same way and scored as a delivered evaluation is: with its joint tree alone, without the export's
own records (Blender parenting, visibility, and the classes it derives from fluid and particle modifiers); its liquid
and effect materials are read from the blend, as for a delivered evaluation. Its `animation.blend` given directly
names the bundle.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Input:
    path: Path
    label: str
    blend: Path | None = None
    evaluation_dir: Path | None = None    # a delivered evaluation; None: export `blend`
    baseline: Path | None = None
    task: str | None = None
    arm: Path | None = None               # a benchmark arm folder: its benchmark/score.json gets the summary
    notes: list = field(default_factory=list)

    @property
    def kind(self) -> str:
        return "dycraft" if self.baseline else "blend"


def task_id_of(*labels) -> str | None:
    """B01..B24 from a path component such as `b14_crow_opens_puzzle_feeder` or `B01S_...` (short cuts map to
    their brief). Of several components, the innermost that names a brief wins: `rerun_b22/B14_crow/dycraft_codex`
    is B14."""
    for label in reversed(labels):
        if not label:
            continue
        m = re.search(r"(?:^|[^a-z0-9])b(\d{2})s?(?:[^0-9]|$)", str(label).lower())
        if m:
            return f"B{m.group(1)}"
    return None


def _complete(index: Path) -> bool:
    try:
        return bool(json.loads(index.read_text()).get("complete"))
    except (OSError, ValueError):
        return False


def _label(path: Path) -> str:
    parts = [p for p in path.parts if p not in ("bundle", "src", "/")]
    if path.suffix == ".blend":
        parts[-1] = path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(parts[-2:]))


def _baseline(bundle: Path) -> Path | None:
    return next((b for b in (bundle / "design_state" / "baseline.json", bundle / "baseline.json") if b.is_file()),
                None)


def resolve(path, task: str | None = None) -> Input:
    from ...run_layout import internal_dir
    path = Path(path).expanduser().resolve()
    # A run laid out for a reader: its folder, or the copy of its blend at the top, is scored from its records in
    # `src/`, as an older run's folder is.
    if path.is_file() and path.name == "animation.blend" and internal_dir(path.parent) != path.parent:
        path = path.parent
    if path.is_dir():
        path = internal_dir(path)
    if path.is_file() and path.suffix == ".blend":
        # A DyCraft film's own blend names its bundle: scored as a `.blend`, it would lose its joint tree and take the
        # export's Blender parenting instead, which exempts pairs the paper measured (a tire against its welded rim).
        if path.name == "animation.blend" and _baseline(path.parent) is not None:
            return resolve(path.parent, task)
        # Four components: a staged arm's blend sits at `<brief>/<arm>/bundle/animation.blend`.
        return Input(path, _label(path), blend=path, task=task or task_id_of(*path.parts[-4:]))
    if not path.is_dir():
        raise FileNotFoundError(f"not a run, bundle, or .blend: {path}")
    bundle = path / "bundle" if (path / "bundle").is_dir() else path
    # The run or arm folder, also when the path names its bundle (or the bundle's blend).
    home = bundle.parent if bundle.name == "bundle" else bundle
    arm = home if (home / "benchmark").is_dir() else None
    blend = bundle / "animation.blend"
    baseline = _baseline(bundle)
    inp = Input(path, _label(path), blend=blend if blend.is_file() else None, baseline=baseline,
                task=task or task_id_of(*path.parts[-3:]), arm=arm)
    if baseline is not None and _complete(bundle / "evaluation" / "index.json"):
        inp.evaluation_dir = bundle / "evaluation"
    elif inp.blend is None:
        raise FileNotFoundError(f"{path}: no complete evaluation/ and no animation.blend to export")
    elif baseline is not None:
        inp.notes.append("no complete evaluation: exported from animation.blend")
    return inp


def benchmark_arms(out: Path) -> list[Path]:
    """Every arm a benchmark output directory holds (`<out>/<brief>/<arm>/benchmark/record.json`)."""
    return [record.parents[1] for record in sorted(Path(out).glob("*/*/benchmark/record.json"))]
