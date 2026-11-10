# Placement traps

## Parenthesize plane placement before world translation

`Pos(...) * Plane.XZ * sketch` groups as `(Pos(...) * Plane.XZ) * sketch`. In build123d 0.11.1, that translates the Plane along its local axes. It differs from translating an already oriented shape:

| Expression | Center for Pos(0, −0.110, 0.0435) and a centered sketch |
|---|---|
| `Pos(...) * Plane.XZ * sketch` | (0, −0.0435, −0.1100) |
| `Pos(...) * (Plane.XZ * sketch)` | (0, −0.1100, +0.0435) |

```python
# validated
sections = []
for y in [-0.06, 0.0, 0.06]:
    sections.append(Pos(0, y, 0.02) * (Plane.XZ * Circle(0.02)))
part = loft(sections)
```

This differs from `Plane.XZ * loc * cutter` below: there the intent is to rotate the entire XY pattern into XZ. For a loft, compare the placed section centers with the intended stations before diagnosing the loft itself; the wrong composition can put a valid stack along the wrong axis.

## Patterns start in XY

`GridLocations` and `PolarLocations` return locations in XY. For a pattern in XZ, transform the locations and cutters together:

```python
# validated
panel = Box(0.12, 0.002, 0.08)
slots = [Plane.XZ * loc * Box(0.09, 0.004, 0.01)
         for loc in GridLocations(0, 0.010, 1, 5)]
part = panel - slots
```

Here `Plane.XZ * loc` maps local Y spacing to world Z. Without it, most cutters would sit outside the thin panel and remove nothing.

## Extrusion direction and centering

| Plane | Positive normal |
|---|---|
| `Plane.XY` | +Z |
| `Plane.XZ` | −Y |
| `Plane.YZ` | +X |

These are the plane normals; polygon winding can reverse the resulting face normal. Without `dir=`, extrusion follows that face normal, so a clockwise XZ polygon can extrude toward +Y. When direction matters, pass it explicitly, e.g. `extrude(Plane.XZ * profile, amount=t, dir=(0, -1, 0))`, and check the resulting bounds. `both=True` extrudes the full amount in each direction, giving twice that total thickness.

Some profiles retain split topology at the sketch plane after a two-sided extrusion; a simple rectangular extrusion need not.

When edges crossing the sketch plane are split in two, reducing the fillet radius again and again does not help, and a geometrically identical one-sided extrusion can blend. Inspect the edge topology before trying another smaller radius.

If a later blend fails near that plane, compare a one-sided extrusion translated to the center:

```python
# validated
HALF_T = 0.004
sketch = Rectangle(0.08, 0.05)
part = Pos(0, HALF_T, 0) * extrude(Plane.XZ * sketch, 2 * HALF_T)
assert abs(part.bounding_box().min.Y + HALF_T) < 1e-6
assert abs(part.bounding_box().max.Y - HALF_T) < 1e-6
```

The centering offset is **+HALF_T** because the extrusion extends toward −Y.
