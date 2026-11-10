"""The object suite, run over the definitions a scene places.

A scene's `scene.py` authors its objects itself, so no object run has checked
them; `scene_verify.SCENE_VERIFIERS` runs these first. Without them a door that
swept through its own carcass would compile clean.

The worst of it was `ObjectRelationalAssertions`. `obj.assert_distance(...)`
records an assertion in the ObjectIR and the object suite is what evaluates it;
without that verifier the call succeeded, the assertion rode along in the IR,
and nothing ever read it. An author following the object contract — numbers in
assertions, looks in checks — got silence for half of it.

Each object check is lifted to a scene-level verifier that loops the placed
definitions, rather than folded into one composite. The name an author sees on
a failure is then the same name the object prompt teaches, and its repairs are
the object-layer repairs, which is the whole point of running them.
"""

from __future__ import annotations

from ..sdk import _primitives as _prim
from .object_verify import (NO_SELF_PENETRATION, OBJECT_VERIFIERS,
                            SUPPORT_CONNECTED)
from .report import PASSING, VerifierReport

#: Object checks that do NOT transfer to a scene build.
#:
#: `SupportConnected` asks whether an object's parts reach the ground through
#: contact. For an object being authored alone that is the right question. For
#: one authored to be placed, it is the wrong one twice over: a wall cabinet or
#: a hanging lamp supports nothing by itself and would fail, and the scene suite
#: already asks the question that matters — whether the *instance* is supported
#: where it was placed. It also collides by name with the scene-level check.
_SKIP = frozenset({SUPPORT_CONNECTED})


#: Object checks that do not transfer to an *imported* definition.
#:
#: `NoSelfPenetration` sweeps each joint and adjudicates the survivors, and it
#: asks for the B-rep backend explicitly because its repair advice is written
#: in exact intersection volumes. An imported part has no B-rep by
#: construction, so every candidate pair comes back `not_adjudicable` and is
#: kept — the Panda fails on 15 pairs it cannot judge. Re-adjudicating those
#: on the mesh backend shows the overlaps are real, not an artifact: a URDF's
#: collision meshes routinely interpenetrate at the joints, and `import_urdf`
#: declares nothing about it.
#:
#: The check is therefore unpassable *and* unfixable here. Its declared repair
#: is `obj.allow_collision(...)`, which needs a builder that does not exist
#: for an imported asset, and editing the IR to add the allowance would break
#: the content hash that is its identity.
#:
#: Skipping it is policy rather than convenience: an imported asset must not
#: have to pass the swept object suite first, and
#: `animation/build.py::compile_design` omits this one check for the
#: animation layer too, so the two layers agree.
#:
#: One check, not the suite. A large imported humanoid run through
#: `OBJECT_VERIFIERS` directly fails `NoSelfPenetration`, is `uncertain` on
#: `JointOriginNearBothBodies` (advisory, it does not block), fails
#: `SupportConnected`, which this layer's suite leaves out via `_SKIP`, and
#: passes the rest. Exempting the whole object suite for it would suppress
#: checks that pass and one that has something to say.
_SKIP_FOR_IMPORTED = frozenset({NO_SELF_PENETRATION})


def _is_imported(obj_ir: dict) -> bool:
    """True for a definition that came from a converter, not a builder.

    Every part, not any: a mixed definition cannot be produced today, but if
    one ever is, the authored half is exactly the half that can be repaired,
    so the check should keep running on it. Reading the primitive tag rather
    than the absent `source` because that is the fact the exemption is about
    — no B-rep to adjudicate — and because objects authored inline in a
    `scene.py` carry no `source` either.
    """
    parts = obj_ir.get("parts") or ()
    return bool(parts) and all(
        p.get("primitive") == _prim.PRIMITIVE_IMPORTED for p in parts)


def _placed_definitions(scene_ir: dict, catalog: dict) -> list:
    """The definitions this scene actually places, in a stable order.

    Placed rather than built: an author may finalize a definition to look at it
    before deciding where it goes, and a candidate under inspection should not
    fail the compile of a scene that does not contain it. What ships is what is
    placed.

    Every placed definition is returned, imported ones included. Which checks
    an imported definition is excused from is `_SKIP_FOR_IMPORTED`'s decision,
    made per check in `per_object`, not a decision this function makes for the
    whole suite.
    """
    refs = {inst.get("object_ref") for inst in (scene_ir.get("instances") or ())}
    return [(h, catalog[h]) for h in sorted(refs) if h in catalog]


def _instances_of(scene_ir: dict, object_ref: str) -> list:
    return sorted(inst["name"] for inst in (scene_ir.get("instances") or ())
                  if inst.get("object_ref") == object_ref)


def _worst(verdicts) -> str:
    """The scene's verdict over its definitions'; `not_applicable` only if every one was."""
    for level in ("fail", "uncertain", "pass"):
        if level in verdicts:
            return level
    return "not_applicable"


def per_object(name: str, fn, *, skip_imported: bool = False):
    """Lift one object verifier to a scene-level one over placed definitions.

    `skip_imported` exempts converter-supplied definitions — see
    `_SKIP_FOR_IMPORTED`. The exemption is reported, not silent: the skipped
    definitions are named in `evidence.imported_definitions_skipped`, so a
    passing report says which geometry it did not cover. An exemption nobody
    can see is indistinguishable from a check that never ran.
    """

    def run(ir: dict, ctx=None) -> VerifierReport:
        payload = getattr(ctx, "layer_payload", None)
        catalog = getattr(payload, "catalog", None) or {}
        offenders: list = []
        verdicts: list = []
        repairs: list = []
        skipped: list = []
        for object_ref, obj_ir in _placed_definitions(ir, catalog):
            if skip_imported and _is_imported(obj_ir):
                skipped.append(object_ref)
                continue
            report = fn(obj_ir, ctx)
            verdicts.append(report.verdict)
            if report.verdict in PASSING:
                continue
            offenders.append({
                "object": obj_ir.get("name"),
                "object_ref": object_ref,
                # Which instances carry the defect. An author reads the scene
                # in instance names, and one bad definition can be placed
                # eight times — naming only the definition would leave them
                # hunting for what to look at.
                "instances": _instances_of(ir, object_ref),
                "verdict": report.verdict,
                "evidence": report.evidence,
            })
            for repair in report.suggested_repairs:
                if repair not in repairs:
                    repairs.append(repair)

        evidence: dict = {"definitions_checked": len(verdicts)}
        if offenders:
            evidence["offenders"] = offenders
        # Reported rather than silent: "3 definitions checked" in a scene that
        # places four is a number the author should be able to account for, and
        # a skipped definition is a fact about the run, not an implementation
        # detail of this verifier. Named, with its instances, so the report
        # says which geometry this check did not cover.
        if skipped:
            evidence["imported_definitions_skipped"] = [
                {"object_ref": h,
                 "object": (catalog.get(h) or {}).get("name"),
                 "instances": _instances_of(ir, h)}
                for h in sorted(skipped)]
        return VerifierReport(name=name, verdict=_worst(verdicts),
                              ir_hash=ir.get("id", ""), evidence=evidence,
                              suggested_repairs=repairs)

    run.__name__ = f"per_object_{getattr(fn, '__name__', 'check')}"
    run.__doc__ = (
        f"{name}, evaluated on every object definition the scene places"
        + (", except imported ones." if skip_imported else "."))
    return run


#: The object checks that run inside a scene build.
PER_OBJECT_VERIFIERS = tuple(
    (name, per_object(name, fn, skip_imported=name in _SKIP_FOR_IMPORTED))
    for name, fn in OBJECT_VERIFIERS if name not in _SKIP)


__all__ = ["PER_OBJECT_VERIFIERS", "per_object"]
