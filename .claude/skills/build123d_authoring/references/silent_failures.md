# Boolean and loft failures

A valid B-rep need not have the intended shape. Check the invariant relevant to the operation: solid count for a fuse, penetration for a through-cut, bounds and sections for a loft.

## Fuses and coincident surfaces

`a + b` can contain multiple disconnected solids when the operands do not intersect. If one continuous body is intended, inspect `len(part.solids())`. A Compound is also legitimate when multiple solids are intended.

In DyCraft, `expect_solids=1` checks that single-body intent at registration. Inspect connections after cuts too: a cutter crossing the full width can split a body. Preserve connecting webs or cheeks where the design requires them.

Face-touching solids can fuse successfully. For features intended to merge, a small overlap can make the boolean less sensitive to coincidence; choose it relative to the feature and intended surface, not a universal length. Preserve designed clearance between distinct moving bodies.

Coincident curved faces do not always fail. If a boolean fails or produces implausible volume near a coincident/tangent interface, adjust the local overlap or rebuild that interface and recheck it.

Valid operands can produce an invalid union. Isolate intermediate unions to locate the failing interface; repairs have included changing union order, deepening a shallow connection, and retaining separately registered fixed regions when fusion remained invalid.

A union that returns less volume than one of its operands has failed, even when the result is valid. Where the operands meet at a coincident section, a slight radius change or offset can resolve it; the right offset depends on the case.

## Cuts that miss, stop short, or return empty

A nonintersecting cutter leaves the operand unchanged. A shallow cutter can reduce volume while leaving a membrane. For a through-hole, extend the cutter beyond both wall faces and inspect its exit or a section.

`Compound(children=...)` groups solids without fusing overlaps. Subtraction involving overlapping compounds has silently left residual material. Fuse the overlapping members before the boolean, or cut each member before compounding; verify the intended removal with residual intersection volume or a bore/section inspection.

Repeated booleans can return an empty shape without raising; an error may then appear on the next operation. For a suspect cut sequence, check immediately:

```python
for c in cutters:
    top = top - c
    if not top.solids():
        raise ValueError("cut annihilated the part")
```

Also compare the volume change with the cutter's expected intersection. `is_valid` alone does not establish nonemptiness or correct removal. Cleaning, reordering, or combining cutters may help a particular failure; none guarantees a repair.

A cut that works in isolation can still empty the part after earlier booleans. When the line that raises looks innocuous, locate the first bad intermediate result before changing that line's operands.

## Loft section placement, order, and orientation

`loft` uses sections in the supplied order. Repeated or out-of-sequence stations can fold or overshoot while producing a valid solid. Check bounds against the station positions and inspect the surface.

Even correctly ordered smooth lofts can overshoot sparse stations or introduce unwanted concavity. Use true elliptical sections where intended, controlled intermediate stations, or `ruled=True` where appropriate. Independently interpolated inner and outer lofts can cross between stations and sever a thin shell; derive them from a common master and inspect intermediate sections.

Plane/translation composition can put the whole stack on the wrong axis; see [placement traps](placement_traps.md). Keep section winding and orientation consistent unless a twist is intended. A rotation of +90° about X sends sketch +Y to world +Z; −90° sends it to −Z. Symmetric profiles can hide that inversion.

Similar volume, face count, or bounds do not rule out a twisted or inverted profile. Use a diagnostic view or section for the intended contour. A loft's sign or size alone does not identify which construction step was wrong.

### Choose evidence for the suspected failure

- **Missing or excess material:** compare volume with a rough expected value and inspect the change across the suspect operation. A volume decrease alone cannot show that a hole goes all the way through.
- **Disconnected material or unintended cavities:** inspect `len(part.solids())` before registration. After registration, `part_shape(ir, name)` exposes `qc.n_shells`: one outside boundary plus one per sealed cavity for a connected solid. An open bin can have one shell; a closed hollow body has two. Compare with the intended topology.
- **Hollow versus solid construction:** use the same probe's `fill_ratio`, volume divided by bounding-box volume. A shelled carcass can be around 0.1; a solid box is near 1. Interpret it relative to the design, not as a target value.
- **Bore placement:** the `cylinders` inventory marks cylindrical inner faces with `concave: true`. An absent expected bore merits inspecting the cutter placement; a nonanalytic hole may not appear in that inventory.
- **Loft contour:** inspect a view or section that reveals profile orientation and the surface between stations. Twisting an otherwise similar stack can leave volume and bounds nearly unchanged.

In a raw CAD experiment, inspect the shape directly; the `part_shape` fact table is available through the DyCraft probe after building `object.py`. A full verifier pass is not required to query it.
