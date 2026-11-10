"""Body classes for the floating rate: which bodies are scored, and how they connect.

| class   | scored | edges                                                            |
|---------|--------|------------------------------------------------------------------|
| solid   | yes    | all (passive, mounted, structural, bonded, pivot)                |
| carrier | yes    | contact in any direction (deforming: an animal's coat, a hand, a cord) |
| agent   | no     | all, and a support source (self-propelled: a flying bird)        |
| exempt  | no     | all, and a support source (buoyant: a boat, a float)             |
| fluid   | no     | delivered structure only                                         |
| text    | no     | delivered structure only                                         |

Fluid and text bodies keep their delivered structural links (joints, welds, parenting, with contact), so a
structural part misnamed as fluid does not cut a support chain; they never give contact support (water does not hold
up a boat; the boat, being exempt, holds itself up and carries its cargo). Classes are automatic seeds from material
classes recorded by the export, name tokens, mesh deformation, and the task manifest (`tasks.json`). Standard library
only: it also runs inside the geometry child, which skips the contact pairs of bodies that cannot take part.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SOLID, CARRIER, AGENT, FLUID, TEXT, EXEMPT = "solid", "carrier", "agent", "fluid", "text", "exempt"
ALL_CLASSES = (SOLID, CARRIER, AGENT, FLUID, TEXT, EXEMPT)
#: bodies that take part in contact-based support (the rest: structural links only)
CONTACT_CLASSES = (SOLID, CARRIER, AGENT, EXEMPT)
#: unscored bodies that hold themselves up and so support what they carry or what rests on them
SELF_SUPPORTING_CLASSES = (AGENT, EXEMPT)
#: bodies the rate scores
SCORED_CLASSES = (SOLID, CARRIER)

TASKS = Path(__file__).resolve().parent / "tasks.json"

_FLUID_TOKENS = {
    "liquid", "liquids", "fluid", "fluids", "pour", "pouring", "stream", "streams", "splash", "splashes",
    "foam", "froth", "steam", "smoke", "spray", "droplet", "droplets", "drip", "drips", "mist", "vapor",
    "vapour", "puddle", "wake", "wakes", "backwash", "ripple", "ripples", "drink",
    "coffee", "espresso", "milk", "juice",
    "wine", "beer", "yolk", "yolks", "albumen", "batter", "slosh", "meniscus", "particles", "particle"}
_WATER_TOKENS = {"water", "waters"}
#: ambiguous effect words ('car wash', 'wash basin'): fluid only next to a water word, or on a deforming or
#: scale-changing mesh ('Propeller_Wash', 'bow wash')
_WEAK_FLUID_TOKENS = {"wash", "washes"}
_WATER_CONTEXT_TOKENS = {"propeller", "prop", "boat", "ship", "hull", "bow", "stern", "water", "wave", "waves",
                         "jet", "spray", "surf", "wake", "river", "canal", "sea", "lake"}
_CONTAINER_TOKENS = {
    "tank", "bucket", "glass", "glasses", "jug", "pitcher", "bottle", "bottles", "cup", "cups", "valve",
    "pipe", "pipes", "wheel", "tower", "reservoir", "basin", "tray", "can", "pail", "kettle", "channel",
    "trough", "funnel", "wall", "walls", "bank", "banks", "tap", "faucet", "nozzle", "spout", "gauge",
    "heater", "boiler", "cooler", "dispenser", "fountain", "hose", "sink", "shell", "lid", "frame",
    "mount", "bracket", "housing", "pump", "gate", "sluice", "wheelhouse", "mill", "paddle", "float"}
_TEXT_TOKENS = {"title", "subtitle", "caption", "captions", "text", "lettering", "credits", "watermark"}


def task_entry(task_id: str | None) -> dict | None:
    """The benchmark brief's entry in `tasks.json` (self-propelled and buoyant bodies, extra fluid words)."""
    if task_id is None:
        return None
    return json.loads(TASKS.read_text()).get("tasks", {}).get(task_id)


def tokens(*labels) -> set:
    out = set()
    for label in labels:
        if label:
            out.update(t for t in re.split(r"[^a-z0-9]+", str(label).lower()) if t)
    return out


def components(label) -> list:
    """Name path components: 'Water cutter / drain_apron' -> ['water cutter', 'drain_apron']."""
    if not label:
        return []
    return [c.strip().lower() for c in re.split(r"[/|•:]", str(label)) if c.strip()]


def seed_classes(names, identities, *, sidecar_classes, deforming, scale_changing, task_entry=None):
    """Automatic classes, with a reason per non-solid body.

    Fluid and text names are matched on the *last* name component (the part name), or on a whole component that is
    just a fluid word ('Water / reservoir'), so a group label such as 'Water cutter / drain_apron' does not turn its
    structural parts into fluid. `task_entry` lists name tokens of self-propelled (-> agent) and buoyant (-> exempt)
    bodies, and extra fluid tokens.
    """
    task_entry = task_entry or {}
    agent_tokens = {t.lower() for t in task_entry.get("self_propelled_airborne", [])}
    buoyant_tokens = {t.lower() for t in task_entry.get("buoyant", [])}
    task_fluid = {t.lower() for t in task_entry.get("fluid_tokens", [])}
    fluid_words = _FLUID_TOKENS | _WATER_TOKENS | task_fluid
    classes, reasons = [], {}
    for b, (name, ident) in enumerate(zip(names, identities)):
        toks = tokens(name, ident)
        comps = components(name) + components(ident)
        last = tokens(components(name)[-1] if components(name) else name) | \
            (tokens(components(ident)[-1]) if components(ident) else set())
        whole_fluid = [c for c in comps if tokens(c) and tokens(c) <= fluid_words and len(tokens(c)) == 1]
        side = (sidecar_classes or {}).get(name)
        cls, why = SOLID, None
        if side in ("fluid", "particles"):
            cls, why = FLUID, f"export sidecar class {side}"
        elif toks & agent_tokens:
            cls, why = AGENT, "task manifest (self-propelled): " + ",".join(sorted(toks & agent_tokens))
        elif toks & buoyant_tokens:
            cls, why = EXEMPT, "task manifest (buoyant): " + ",".join(sorted(toks & buoyant_tokens))
        elif last & _TEXT_TOKENS:
            cls, why = TEXT, "name token " + ",".join(sorted(last & _TEXT_TOKENS))
        elif whole_fluid:
            cls, why = FLUID, f"name component '{whole_fluid[0]}'"
        elif last & _FLUID_TOKENS and ((deforming[b] or scale_changing[b]) or not last & _CONTAINER_TOKENS):
            cls, why = FLUID, "name token " + ",".join(sorted(last & _FLUID_TOKENS))
        elif last & _WEAK_FLUID_TOKENS and not last & _CONTAINER_TOKENS and (
                deforming[b] or scale_changing[b] or toks & _WATER_CONTEXT_TOKENS):
            cls, why = FLUID, "name token " + ",".join(sorted(last & _WEAK_FLUID_TOKENS)) + " (water effect)"
        elif last & (_WATER_TOKENS | task_fluid) and ((deforming[b] or scale_changing[b])
                                                      or not last & _CONTAINER_TOKENS):
            cls, why = FLUID, "name token " + ",".join(sorted(last & (_WATER_TOKENS | task_fluid)))
        elif deforming[b]:
            cls, why = CARRIER, "mesh changes over time (deforming solid)"
        elif scale_changing[b]:
            cls, why = CARRIER, "matrix scale changes over time (non-rigid)"
        classes.append(cls)
        if why:
            reasons[name] = why
    return classes, reasons
