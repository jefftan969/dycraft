# Operation choices

## Preserve analytic geometry where useful

Use an extrusion for a constant section and a revolve for a surface of revolution when they describe the intended form. These retain analytic faces that the SDK can report as cylinders and bores. A spline loft may use more triangles and lose that analytic classification. Use lofts for varying sections and freeform surfaces.

That classification matters when recovering a shaft or bore axis for a semantic anchor: a visually cylindrical spline surface need not produce an entry in `part_shape`'s cylinder inventory. Choose based on the intended surface and the measurements you will need, rather than avoiding an operation merely because it is more elaborate.

## Sweep sections and path transitions

For a continuous curved member, prefer one sweep along a connected path, such as `Wire([Line(...), Bezier(...), Line(...)])`. Orient the section normal along the starting path tangent and make adjacent segments tangent where a smooth transition is intended. Sharp joins and misoriented sections have caused collapsed or invalid sweeps. For a failing closed sweep, inspect the closing seam; splitting the path into open arcs has repaired some cases, but recheck the assembled result.

Separately swept segments with overlapping tangent ends can fuse into a valid solid yet leave mesh boundary edges at the joins. If that occurs, sweep the full branch together. Inspect tessellation on slender sweeps and tune mesh settings when needed.

## Select fillets after the relevant boolean

An edge created by a union can only be filleted after that union. Rounding the operands first changes the surfaces that meet and may prevent the intended blend. Select edges by their geometry or location rather than an index that changes after booleans:

```python
# validated
bracket = Box(0.08, 0.05, 0.010) + Pos(0, 0, 0.025) * Box(0.08, 0.010, 0.050)
seam = [edge for edge in bracket.edges().filter_by(Axis.X)
        if abs(edge.center().Z - 0.005) < 1e-6
        and abs(abs(edge.center().Y) - 0.005) < 1e-6]
assert len(seam) == 2
part = fillet(seam, radius=0.004)
```

This is a rule for the target seam, not a requirement to postpone every fillet. An isolated detail can be easier to finish before combining it with the rest.

On thin plates, an all-edge fillet reaching half the thickness can fail. Reduce the radius to leave a planar band, or round only edges along the thickness direction when the intended feature is a rounded outline.

`max_fillet` searches at the shape's coordinate scale. Its default tolerance of 0.1 is too coarse for centimeter-sized parts authored in meters. Values such as `tolerance=1e-5, max_iterations=40` give a search at that scale; choose them for the dimensions involved. A search result is still subject to the selected edges and topology. If many smaller radii fail at the same seam, inspect the edge selection and split faces ([placement traps](placement_traps.md)) rather than assuming the radius is still too large.

## Choose boolean order around non-uniform scaling

Non-uniform `scale(shape, by=(a, b, c))` can convert analytic geometry to B-splines, making some booleans less reliable. If a cut fails on a stretched surface, try cutting the analytic master first and scaling the result. This also stretches the cut: use it when that produces the intended feature.

For features defined in the final shape's coordinates, construct cutters from that surface's positions and local frames; a loft through sections along an ellipsoid is one option. Booleans after scaling can succeed. Check cutter placement and intersection before attributing failure to the surface type. Use a common explicit `about=(x,y,z)` when scaling multiple operands together; the default center is each operand's own location.

## Partition one master surface into articulated regions

Partitioning a master solid gives moving regions a common exterior at rest. Independently constructed regions can also match if derived from the same surface definition.

- **Plane:** `split(master, bisect_by=Plane(...), keep=Keep.BOTH)` gives complementary flat interfaces. Rotation that tilts the interface can expose gaps or cause interference.
- **Mask solid:** a spherical interface is invariant under rotation about its center. Put the joint origin there, and check the surrounding geometry and any nonspherical portion of the mask through the intended travel; the sphere alone does not guarantee a concealed seam.

Use intersection and subtraction with the same mask for complementary regions without clearance:

```python
# validated
master = loft([Pos(0, 0, z) * Ellipse(rx, ry) for z, rx, ry in [
    (0.00, 0.060, 0.045), (0.13, 0.076, 0.056),
    (0.26, 0.054, 0.041), (0.34, 0.037, 0.033)]])

# The box extends the spherical mask beyond the top of the master.
mask = (Pos(0, 0, 0.30) * Sphere(0.05)) + (Pos(0, 0, 0.42) * Box(0.4, 0.4, 0.2))
assert len(mask.solids()) == 1

head = master & mask
body = master - mask
assert len(head.solids()) == 1
assert len(body.solids()) == 1
assert abs(head.volume + body.volume - master.volume) < 1e-9

part = body
```

The solid counts and volume sum catch gross boolean errors; equal volume alone does not prove matching surfaces or a gap-free assembly.

Rigid joints do not deform the skin. Inspect seams and interference at rest, limits, and relevant intermediate poses. Choose clearance and joint geometry for the required travel; a continuous rest surface can still collide in motion. Record any remaining representation limit after checking those states.
