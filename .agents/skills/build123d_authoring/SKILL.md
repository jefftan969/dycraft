---
name: build123d-authoring
description: Build123d 0.11.1 placement, boolean, loft, and fillet pitfalls, plus the DyCraft geometry namespace. Use before substantial geometry or when a valid shape differs from the intended result.
---

# build123d authoring

This skill covers traps in the pinned build123d version, especially valid shapes that differ from the intended geometry and failures reported after the operation that caused them. Use it to choose the relevant construction or diagnostic; it does not prescribe a modeling style.

Examples use meter-valued dimensions for dycraft. The namespace reference applies to `object.py`; raw build123d scripts may use additional APIs.

Read the page relevant to the operation or failure:

| Page | Use for |
|---|---|
| [Boolean and loft failures](references/silent_failures.md) | Disconnected solids, invalid unions, compound cuts, loft overshoot and section orientation, and which measurements reveal them. |
| [Placement traps](references/placement_traps.md) | Transform composition, patterns on non-XY planes, winding-dependent extrusion direction. |
| [Operation choices](references/robust_construction.md) | Analytic surfaces, failed sweeps, thin-part fillets, non-uniform scaling, and articulated surface partitions. |
| [Namespace reference](references/namespace_reference.md) | Names and call signatures available in DyCraft. |
