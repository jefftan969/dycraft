"""Current generated design, read from the atomically published workspace."""
import json
from pathlib import Path
from types import SimpleNamespace


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def workspace_context(workspace):
    design = Path(workspace) / "design_state"
    return SimpleNamespace(
        run_dir=design, asset_base_dir=design / "source",
        scene_ir=read_json(design / "scene.json", {}),
        catalog=read_json(design / "catalog.json", {}),
    )
