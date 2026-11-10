"""Prepare the frozen scene using the shared renderer and URDF frame math."""
from pathlib import Path

from ...render import RenderRequest
from ...render._world import _prepare, plan_outputs
from ...render.blender_renderer import _frame_spec
from ...render.urdf_export import scene_ir_to_urdf
from .bundle import write_json


def prepare_scene(ctx, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    request = RenderRequest(layer="scene", ir=ctx.scene_ir, catalog=ctx.catalog,
                            target="all", out_dir=destination / "views", step=0,
                            asset_base_dir=getattr(ctx, "asset_base_dir", None)
                            or Path(ctx.run_dir) / "catalog")
    plan = _prepare(request)
    outputs = plan_outputs(request, plan, config_version="native-import-v1")
    spec = _frame_spec(plan, outputs, destination, 0, request)
    if spec is None:
        raise ValueError("scene import unexpectedly has no geometry specification")
    # Render previews can approximate broken imports; authoritative animation cannot.
    for part in spec["parts"]:
        part.pop("fallback", None)
        if part["kind"] == "asset" and not Path(part["obj"]).is_file():
            raise ValueError(f"missing scene mesh: {part['obj']}")
    _attach_native_rigs(spec, ctx, request.asset_base_dir)
    spec["urdf"] = scene_ir_to_urdf(ctx.scene_ir, ctx.catalog)
    # Each joint starts where its instance was placed: the instance's
    # `initial_joints` over the definition's `default`, in IR units. The parts
    # are exported at that configuration, so a channel started at `default`
    # left the difference baked into the part and every keyed value landed
    # offset by it.
    spec["joints"] = {}
    for inst in ctx.scene_ir["instances"]:
        initial = inst.get("initial_joints") or {}
        for j in ctx.catalog[inst["object_ref"]]["joints"]:
            spec["joints"][f"{inst['name']}__{j['name']}"] = (
                {**j, "default": float(initial[j["name"]])} if j["name"] in initial else j)
    spec["scene_ref"] = ctx.scene_ir["id"]
    # Which instance replaced which shell surface (`replace_room_surface`), so
    # an event's `room.floor` can still name the floor once it is modeled.
    spec["surface_replacements"] = dict(
        (ctx.scene_ir.get("room") or {}).get("surface_replacements") or {})
    # Anatomical ranges for any preloaded asset that declares them, keyed the
    # same `<instance>__<joint>` way the joints above are. These narrow a
    # vendor's hardware limits to what still reads as an unbroken body, and the
    # animation suite checks them per frame beside `JointLimits`. In IR units,
    # because that is what `joint_position` returns and what `limits` already
    # carries — a unit conversion here would be a second place to get it wrong.
    spec["anatomical_limits"] = _anatomical_spec(ctx)
    write_json(destination / "scene_spec.json", spec)
    return spec


def _attach_native_rigs(spec, ctx, asset_base_dir) -> None:
    """Point each rigged part at the frozen blend that replaces it.

    An imported character (`agent/assets/import_gltf.py`) is one proxy
    part in the IR; `worker.initialize` builds its armature and skinned mesh
    from this blend instead of from the proxy OBJ. Resolved here, against the
    same base the part meshes are, so the worker needs no second path rule.
    """
    by_name = {part["name"]: part for part in spec["parts"]}
    for inst in ctx.scene_ir.get("instances") or ():
        rig = (ctx.catalog.get(inst.get("object_ref")) or {}).get("native_rig")
        if not rig:
            continue
        part = by_name.get(f"{inst['name']}/{rig['part']}")
        if part is None:
            raise ValueError(f"instance {inst['name']!r}: rigged part "
                             f"{rig['part']!r} is not in the scene spec")
        blend = Path(asset_base_dir) / rig["blend"]
        if not blend.is_file():
            raise ValueError(f"missing character rig: {blend}")
        part["native_rig"] = {"blend": str(blend), "digest": rig["digest"],
                              "collection": rig["collection"],
                              "armature": rig["armature"],
                              "mesh": rig["mesh"]}


def _anatomical_spec(ctx) -> dict:
    """`{"<instance>__<joint>": [lo, hi]}` for every narrowed joint placed.

    Resolved from the source tree the scene was built from, which is where
    `--preload-asset` stages an animation run's assets. Absent overlays
    are the ordinary case and yield `{}`, so nothing downstream has to branch.
    """
    from ...agent.assets.preload_asset import anatomical_limits
    root = getattr(ctx, "asset_base_dir", None) or getattr(ctx, "run_dir", None)
    if root is None:
        return {}
    overlays = anatomical_limits(Path(root))
    if not overlays:
        return {}
    out = {}
    for inst in ctx.scene_ir.get("instances") or ():
        overlay = overlays.get(inst.get("object_ref"))
        for jname, rule in ((overlay or {}).get("joints") or {}).items():
            out[f"{inst['name']}__{jname}"] = list(rule["limits"])
    return out
