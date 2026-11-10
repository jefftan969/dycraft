"""Material rule: which meshes are liquid or visual effects, not solid bodies.

Runs inside Blender (needs `bpy` data blocks). A penetration between a solid and a water surface, a pour stream or a
foam or wake decal is not a solid-body failure, but such meshes are often ordinary closed meshes of fixed topology.
Their *materials* tell them apart:

* **liquid** — every material slot has a Principled IOR in [1.30, 1.36]
  (water 1.333; solid glass is 1.45–1.52, plastics ~1.49) or drives the
  material output's Volume socket (absorption / scatter);
* **effect** — every material slot has its Principled Alpha driven by a
  texture or other node (procedural transparency: foam, wash, wake, decal), or
  contains no surface BSDF other than Transparent / Emission / Holdout.

Solid transparent parts stay solid: glass bottles and panes have IOR ~1.47 and constant alpha; a glazing that mixes
Principled with a Transparent BSDF still has a Principled surface. Only material data is read, never object names,
so the rule is the same for every method. Known trade-off: ice (IOR 1.31) classifies as liquid.
"""
from __future__ import annotations

RULE_VERSION = "fluid-effect-material-v1"
LIQUID_IOR = (1.30, 1.36)
_SURFACE_ONLY_EFFECT = {"ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeHoldout"}
_SURFACE_BSDF_PREFIX = ("ShaderNodeBsdf", "ShaderNodeSubsurface", "ShaderNodeEmission", "ShaderNodeHoldout")


def material_class(material):
    """'liquid' | 'effect' | None for one material, with the reason."""
    if material is None or not material.use_nodes or material.node_tree is None:
        return None, None
    nodes = list(material.node_tree.nodes)
    outputs = [n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    if any(o.inputs["Volume"].is_linked for o in outputs if "Volume" in o.inputs):
        return "liquid", "volume shader on material output"
    for n in nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled" and "IOR" in n.inputs:
            ior = n.inputs["IOR"]
            if not ior.is_linked and LIQUID_IOR[0] <= float(ior.default_value) <= LIQUID_IOR[1]:
                return "liquid", f"principled IOR {float(ior.default_value):.3f}"
    for n in nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled" and "Alpha" in n.inputs and n.inputs["Alpha"].is_linked:
            return "effect", "principled alpha driven by a node (procedural transparency)"
    surfaces = {n.bl_idname for n in nodes if n.bl_idname.startswith(_SURFACE_BSDF_PREFIX)}
    if surfaces and surfaces <= _SURFACE_ONLY_EFFECT:
        return "effect", "only transparent / emission / holdout shaders"
    return None, None


def object_class(obj):
    """Class for an object only if *every* material slot agrees (else solid)."""
    materials = [slot.material for slot in obj.material_slots if slot.material is not None]
    if not materials:
        return None, None
    verdicts = [material_class(m) for m in materials]
    classes = {c for c, _ in verdicts}
    if len(classes) == 1 and None not in classes:
        kind = classes.pop()
        return ("fluid" if kind == "liquid" else "effect"), "; ".join(sorted({r for _, r in verdicts}))
    return None, None
