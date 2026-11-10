# Namespace reference

**Generated — do not edit.** Produced by `agent/cad/_namespace_doc.py` from the live namespace under build123d 0.11.1, and checked by `tests/test_object_namespace_reference.py`.

`object.py` binds **122** build123d names, also reachable through `bd`. This page describes DyCraft's injected namespace, not the full API available to raw build123d scripts. `math`, `np` (NumPy), and `OCP` are module handles; scalar math names such as `sin`, `cos`, `radians`, and `sqrt` are also available bare.

Call signatures preserve all parameters and defaults; type annotations and enum display brackets are omitted. A bare class name or `(...)` means its signature was unavailable to introspection.

## Classes and shape constructors

| name | signature |
|---|---|
| `Airfoil` | `Airfoil(airfoil_code, n_points=50, finite_te=False, mode=Mode.ADD)` |
| `Axis` | `Axis(*args, **kwargs)` |
| `BSpline` | `BSpline(control_points, knots, degree, weights=None, periodic=False, mode=Mode.ADD)` |
| `Bezier` | `Bezier(*cntl_pnts, weights=None, mode=Mode.ADD)` |
| `BlendCurve` | `BlendCurve(curve0, curve1, continuity=ContinuityLevel.C2, end_points=None, tangent_scalars=None, mode=Mode.ADD)` |
| `Box` | `Box(length, width, height, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `CenterArc` | `CenterArc(center, radius, start_angle, arc_size, mode=Mode.ADD)` |
| `Circle` | `Circle(radius, arc_size=360.0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `Color` | `Color(*args, **kwargs)` |
| `Compound` | `Compound(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)` |
| `Cone` | `Cone(bottom_radius, top_radius, height, arc_size=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `ConstrainedArcs` | `ConstrainedArcs(*args, **kwargs)` |
| `ConstrainedLines` | `ConstrainedLines(*args, **kwargs)` |
| `ConvexPolyhedron` | `ConvexPolyhedron(points, rotation=(0, 0, 0), align=Align.NONE, mode=Mode.ADD)` |
| `CounterBoreHole` | `CounterBoreHole(radius, counter_bore_radius, counter_bore_depth, depth=None, mode=Mode.SUBTRACT)` |
| `CounterSinkHole` | `CounterSinkHole(radius, counter_sink_radius, depth=None, counter_sink_angle=82, mode=Mode.SUBTRACT)` |
| `Curve` | `Curve(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)` |
| `Cylinder` | `Cylinder(radius, height, arc_size=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `DoubleTangentArc` | `DoubleTangentArc(pnt, tangent, other, keep=Keep.TOP, mode=Mode.ADD)` |
| `DraftAngleError` | `DraftAngleError(message, face=None, problematic_shape=None)` |
| `Edge` | `Edge(obj=None, label='', color=None, parent=None)` |
| `Ellipse` | `Ellipse(x_radius, y_radius, rotation=0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `EllipticalCenterArc` | `EllipticalCenterArc(center, x_radius, y_radius, start_angle=0.0, end_angle=None, *, arc_size=90.0, rotation=0.0, angular_direction=None, mode=Mode.ADD)` |
| `EllipticalStartArc` | `EllipticalStartArc(start_pnt, start_tangent, x_radius, y_radius, arc_size, *, start_angle=None, major_axis_dir=None, mode=Mode.ADD)` |
| `Face` | `Face(*args, **kwargs)` |
| `FilletPolyline` | `FilletPolyline(*pts, radius, close=False, mode=Mode.ADD)` |
| `GridLocations` | `GridLocations(x_spacing, y_spacing, x_count, y_count, align=(Align.CENTER, Align.CENTER))` |
| `Helix` | `Helix(pitch, height, radius, center=(0, 0, 0), direction=(0, 0, 1), cone_angle=0, lefthand=False, mode=Mode.ADD)` |
| `HexLocations` | `HexLocations(radius, x_count, y_count, major_radius=False, align=(Align.CENTER, Align.CENTER))` |
| `Hole` | `Hole(radius, depth=None, mode=Mode.SUBTRACT)` |
| `HyperbolicCenterArc` | `HyperbolicCenterArc(center, x_radius, y_radius, start_angle=0.0, end_angle=None, *, arc_size=90.0, rotation=0.0, angular_direction=None, mode=Mode.ADD)` |
| `IntersectingLine` | `IntersectingLine(start, direction, other, mode=Mode.ADD)` |
| `JernArc` | `JernArc(start, tangent, radius, arc_size, mode=Mode.ADD)` |
| `Line` | `Line(*pts, mode=Mode.ADD)` |
| `Location` | `Location(*args, **kwargs)` |
| `Locations` | `Locations(*pts)` |
| `Matrix` | `Matrix(*args, **kwargs)` |
| `ParabolicCenterArc` | `ParabolicCenterArc(vertex, focal_length, start_angle=0.0, end_angle=None, *, arc_size=90.0, rotation=0.0, angular_direction=None, mode=Mode.ADD)` |
| `Part` | `Part(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)` |
| `Plane` | `Plane(*args, **kwargs)` |
| `PolarLine` | `PolarLine(start, length, angle=None, direction=None, length_mode=LengthMode.DIAGONAL, mode=Mode.ADD)` |
| `PolarLocations` | `PolarLocations(radius, count, start_angle=0.0, angular_range=360.0, rotate=True, endpoint=False)` |
| `Polygon` | `Polygon(*pts, rotation=0, align=(Align.NONE, Align.NONE), mode=Mode.ADD)` |
| `Polyline` | `Polyline(*pts, close=False, mode=Mode.ADD)` |
| `Pos` | `Pos(*args, **kwargs)` |
| `RadiusArc` | `RadiusArc(start_point, end_point, radius, short_sagitta=True, mode=Mode.ADD)` |
| `Rectangle` | `Rectangle(width, height, rotation=0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `RectangleRounded` | `RectangleRounded(width, height, radius, rotation=0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `RegularPolygon` | `RegularPolygon(radius, side_count, major_radius=True, rotation=0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `Rot` | `Rotation(*args, **kwargs)` |
| `Rotation` | `Rotation(*args, **kwargs)` |
| `SagittaArc` | `SagittaArc(start_point, end_point, sagitta, mode=Mode.ADD)` |
| `Shell` | `Shell(obj=None, label='', color=None, parent=None)` |
| `Sketch` | `Sketch(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)` |
| `SlotArc` | `SlotArc(arc, height, rotation=0, mode=Mode.ADD)` |
| `SlotCenterPoint` | `SlotCenterPoint(center, point, height, rotation=0, mode=Mode.ADD)` |
| `SlotCenterToCenter` | `SlotCenterToCenter(center_separation, height, rotation=0, mode=Mode.ADD)` |
| `SlotOverall` | `SlotOverall(width, height, rotation=0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `Solid` | `Solid(obj=None, label='', color=None, material='', joints=None, parent=None)` |
| `Sphere` | `Sphere(radius, arc_size1=-90, arc_size2=90, arc_size3=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `Spline` | `Spline(*pts, tangents=None, tangent_scalars=None, periodic=False, mode=Mode.ADD)` |
| `Standard_ConstructionError` | `Standard_ConstructionError` |
| `Standard_Failure` | `Standard_Failure` |
| `StdFail_NotDone` | `StdFail_NotDone` |
| `TangentArc` | `TangentArc(*pts, tangent, tangent_from_first=True, mode=Mode.ADD)` |
| `ThreePointArc` | `ThreePointArc(*pts, mode=Mode.ADD)` |
| `Torus` | `Torus(major_radius, minor_radius, minor_start_angle=0, minor_end_angle=360, major_angle=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `Trapezoid` | `Trapezoid(width, height, left_side_angle, right_side_angle=None, rotation=0, align=(Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `Triangle` | `Triangle(*, a=None, b=None, c=None, A=None, B=None, C=None, align=None, rotation=0, mode=Mode.ADD)` |
| `Vector` | `Vector(*args, **kwargs)` |
| `Vertex` | `Vertex(*args, **kwargs)` |
| `Wedge` | `Wedge(xsize, ysize, zsize, xmin, zmin, xmax, zmax, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.ADD)` |
| `Wire` | `Wire(*args, **kwargs)` |

## Operations

| name | signature |
|---|---|
| `add` | `add(objects, rotation=None, clean=True, mode=Mode.ADD)` |
| `chamfer` | `chamfer(objects, length, length2=None, angle=None, reference=None)` |
| `draft` | `draft(faces, neutral_plane, angle)` |
| `edges_to_wires` | `edges_to_wires(edges, tol=1e-06)` |
| `extrude` | `extrude(to_extrude=None, amount=None, dir=None, until=None, target=None, both=False, taper=0.0, clean=True, mode=Mode.ADD)` |
| `fillet` | `fillet(objects, radius)` |
| `full_round` | `full_round(edge, invert=False, voronoi_point_count=100, mode=Mode.REPLACE)` |
| `loft` | `loft(sections=None, ruled=False, clean=True, mode=Mode.ADD)` |
| `make_brake_formed` | `make_brake_formed(thickness, station_widths, line=None, side=Side.LEFT, kind=Kind.ARC, clean=True, mode=Mode.ADD)` |
| `make_face` | `make_face(edges=None, mode=Mode.ADD)` |
| `make_hull` | `make_hull(edges=None, mode=Mode.ADD)` |
| `mirror` | `mirror(objects=None, about=Plane((0, 0, 0), (1, 0, 0), (0, -1, 0)), mode=Mode.ADD)` |
| `new_edges` | `new_edges(*objects, combined)` |
| `offset` | `offset(objects=None, amount=0, openings=None, kind=Kind.ARC, side=Side.BOTH, closed=True, min_edge_length=None, mode=Mode.REPLACE)` |
| `polar` | `polar(length, angle)` |
| `project` | `project(objects=None, workplane=None, target=None, mode=Mode.ADD)` |
| `project_workplane` | `project_workplane(origin, x_dir, projection_dir, distance)` |
| `revolve` | `revolve(profiles=None, axis=Axis((0, 0, 0), (0, 0, 1)), revolution_arc=360.0, clean=True, mode=Mode.ADD)` |
| `scale` | `scale(objects=None, by=1, about=None, mode=Mode.REPLACE)` |
| `section` | `section(obj=None, section_by=Plane((0, 0, 0), (1, 0, 0), (0, -1, 0)), height=0.0, clean=True, mode=Mode.PRIVATE)` |
| `split` | `split(objects=None, bisect_by=Plane((0, 0, 0), (1, 0, 0), (0, -1, 0)), keep=Keep.TOP, mode=Mode.REPLACE)` |
| `sweep` | `sweep(sections=None, path=None, multisection=False, is_frenet=False, transition=Transition.TRANSFORMED, normal=None, binormal=None, clean=True, mode=Mode.ADD)` |
| `thicken` | `thicken(to_thicken=None, amount=None, normal_override=None, both=False, clean=True, mode=Mode.ADD)` |
| `topo_distance_to` | `topo_distance_to(other)` |
| `topo_explore_common_vertex` | `topo_explore_common_vertex(edge1, edge2)` |
| `topo_explore_connected_edges` | `topo_explore_connected_edges(edge, parent=None, continuity=ContinuityLevel.C0)` |
| `trace` | `trace(lines=None, line_width=1, mode=Mode.ADD)` |

## Enums

| name | members |
|---|---|
| `Align` | `MIN, CENTER, MAX, NONE` |
| `AngularDirection` | `CLOCKWISE, COUNTER_CLOCKWISE` |
| `CenterOf` | `GEOMETRY, MASS, BOUNDING_BOX` |
| `ContinuityLevel` | `C0, C1, C2` |
| `Extrinsic` | `XYZ, XZY, YZX, YXZ, ZXY, ZYX, XYX, XZX, YZY, YXY, ZXZ, ZYZ` |
| `FrameMethod` | `FRENET, CORRECTED` |
| `GeomType` | `PLANE, CYLINDER, CONE, SPHERE, TORUS, BEZIER, BSPLINE, REVOLUTION, EXTRUSION, OFFSET, LINE, CIRCLE, ELLIPSE, HYPERBOLA, PARABOLA, OTHER` |
| `Intrinsic` | `XYZ, XZY, YZX, YXZ, ZXY, ZYX, XYX, XZX, YZY, YXY, ZXZ, ZYZ` |
| `Keep` | `ALL, BOTTOM, BOTH, INSIDE, OUTSIDE, TOP` |
| `Kind` | `ARC, INTERSECTION, TANGENT` |
| `LengthMode` | `DIAGONAL, HORIZONTAL, VERTICAL` |
| `MeshType` | `OTHER, MODEL, SUPPORT, SOLIDSUPPORT` |
| `Mode` | `ADD, SUBTRACT, INTERSECT, REPLACE, PRIVATE` |
| `PositionMode` | `LENGTH, PARAMETER` |
| `Sagitta` | `SHORT, LONG, BOTH` |
| `Select` | `ALL, LAST, NEW` |
| `Side` | `LEFT, RIGHT, BOTH` |
| `SortBy` | `LENGTH, RADIUS, AREA, VOLUME, DISTANCE` |
| `Tangency` | `UNQUALIFIED, ENCLOSING, ENCLOSED, OUTSIDE` |
| `Transition` | `RIGHT, ROUND, TRANSFORMED` |
| `Until` | `NEXT, LAST, PREVIOUS, FIRST` |

## Values

| name | definition |
|---|---|
| `RotationLike` | `Rotation \| tuple[float, float, float]` |

## Class factory methods

Call factories on the class, e.g. `Face.make_surface(...)`. These complement the constructors above: `Face.make_surface*` and `Face.make_gordon_surface` construct surfaces, `Shell.make_loft` and `Solid.make_loft` loft sections, and `Solid.thicken` offsets a face into a wall. Check these before implementing the operation through `OCP`.

Return types vary: factories produce topology objects such as `Face`, `Shell`, or `Solid`; algebra operations can return `Curve`, `Sketch`, or `Part`. An empty `Solid & x` can return `None`, while `Part & x` returns an empty `Compound`. Before continuing a one-solid construction, check `shape is not None and len(shape.solids()) == 1`.

| name | signature |
|---|---|
| `Airfoil.parse_naca4` | `(value)` |
| `Color.categorical_set` | `(color_count, starting_hue=0.0, alpha=1.0)` |
| `Compound.extrude` | `(obj, direction)` |
| `Compound.make_triad` | `(axes_scale)` |
| `Edge.extrude` | `(obj, direction)` |
| `Edge.make_bezier` | `(*cntl_pnts, weights=None)` |
| `Edge.make_bspline` | `(control_points, knots, degree, weights=None, periodic=False)` |
| `Edge.make_circle` | `(radius, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), start_angle=360.0, end_angle=360, angular_direction=AngularDirection.COUNTER_CLOCKWISE)` |
| `Edge.make_constrained_arcs` | `(*args, sagitta=Sagitta.SHORT, **kwargs)` |
| `Edge.make_constrained_lines` | `(*args, **kwargs)` |
| `Edge.make_ellipse` | `(x_radius, y_radius, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), start_angle=360.0, end_angle=360.0, angular_direction=AngularDirection.COUNTER_CLOCKWISE)` |
| `Edge.make_helix` | `(pitch, height, radius, center=(0, 0, 0), normal=(0, 0, 1), angle=0.0, lefthand=False)` |
| `Edge.make_hyperbola` | `(x_radius, y_radius, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), start_angle=360.0, end_angle=360.0, angular_direction=AngularDirection.COUNTER_CLOCKWISE)` |
| `Edge.make_line` | `(point1, point2)` |
| `Edge.make_mid_way` | `(first, second, middle=0.5)` |
| `Edge.make_parabola` | `(focal_length, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), start_angle=0.0, end_angle=90.0, angular_direction=AngularDirection.COUNTER_CLOCKWISE)` |
| `Edge.make_spline` | `(points, tangents=None, periodic=False, parameters=None, scale=True, tol=1e-06)` |
| `Edge.make_spline_approx` | `(points, tol=0.001, smoothing=None, min_deg=1, max_deg=6)` |
| `Edge.make_tangent_arc` | `(start, tangent, end)` |
| `Edge.make_three_point_arc` | `(point1, point2, point3)` |
| `Face.extrude` | `(obj, direction)` |
| `Face.make_bezier_surface` | `(points, weights=None)` |
| `Face.make_gordon_surface` | `(profiles, guides, tolerance=0.0003)` |
| `Face.make_plane` | `(plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)))` |
| `Face.make_rect` | `(width, height, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)))` |
| `Face.make_surface` | `(exterior, surface_points=None, interior_wires=None)` |
| `Face.make_surface_from_array_of_points` | `(points, tol=0.01, smoothing=None, min_deg=1, max_deg=3)` |
| `Face.make_surface_from_curves` | `(*args, **kwargs)` |
| `Face.make_surface_patch` | `(edge_face_constraints=None, edge_constraints=None, point_constraints=None)` |
| `Face.revolve` | `(profile, angle, axis)` |
| `Face.sew_faces` | `(faces)` |
| `Face.sweep` | `(profile, path, transition=Transition.TRANSFORMED)` |
| `Plane.get_topods_face_normal` | `(face)` |
| `Shell.extrude` | `(obj, direction)` |
| `Shell.make_loft` | `(objs, ruled=False)` |
| `Shell.revolve` | `(profile, angle, axis)` |
| `Shell.sweep` | `(profile, path, transition=Transition.TRANSFORMED)` |
| `Solid.extrude` | `(obj, direction)` |
| `Solid.extrude_linear_with_rotation` | `(section, center, normal, angle, inner_wires=None)` |
| `Solid.extrude_taper` | `(profile, direction, taper, flip_inner=True)` |
| `Solid.extrude_until` | `(profile, target, direction, until=Until.NEXT)` |
| `Solid.from_bounding_box` | `(bbox)` |
| `Solid.make_box` | `(length, width, height, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)))` |
| `Solid.make_cone` | `(base_radius, top_radius, height, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), angle=360)` |
| `Solid.make_cylinder` | `(radius, height, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), angle=360)` |
| `Solid.make_loft` | `(objs, ruled=False)` |
| `Solid.make_sphere` | `(radius, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), angle1=-90, angle2=90, angle3=360)` |
| `Solid.make_torus` | `(major_radius, minor_radius, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), start_angle=0, end_angle=360, major_angle=360)` |
| `Solid.make_wedge` | `(delta_x, delta_y, delta_z, min_x, min_z, max_x, max_z, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)))` |
| `Solid.revolve` | `(section, angle, axis, inner_wires=None)` |
| `Solid.sweep` | `(section, path, inner_wires=None, make_solid=True, is_frenet=False, mode=None, transition=Transition.TRANSFORMED)` |
| `Solid.sweep_multi` | `(profiles, path, make_solid=True, is_frenet=False, binormal=None)` |
| `Solid.thicken` | `(surface, depth, normal_override=None)` |
| `Vertex.extrude` | `(obj, direction)` |
| `Wire.extrude` | `(obj, direction)` |
| `Wire.make_circle` | `(radius, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)))` |
| `Wire.make_convex_hull` | `(edges, tolerance=0.001)` |
| `Wire.make_ellipse` | `(x_radius, y_radius, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)), start_angle=360.0, end_angle=360.0, angular_direction=AngularDirection.COUNTER_CLOCKWISE, closed=True)` |
| `Wire.make_polygon` | `(vertices, close=True)` |
| `Wire.make_rect` | `(width, height, plane=Plane((0, 0, 0), (1, 0, 0), (0, 0, 1)))` |

## The kernel, as `OCP`

For lower-level control, pass `shape.wrapped` to OCCT and wrap the `TopoDS_*` result with the matching build123d topology class, such as `Face`, `Shell`, or `Solid`. Examples include variable-radius fillets through `OCP.BRepFilletAPI.BRepFilletAPI_MakeFillet` and sewing through `OCP.BRepBuilderAPI.BRepBuilderAPI_Sewing`. For algorithms that expose `IsDone()`, check it before retrieving the result.

Wrap a bare `Geom_Surface` with `Face(OCP.BRepBuilderAPI.BRepBuilderAPI_MakeFace(surface, tol).Face())`. For point grids, start with `Face.make_surface_from_array_of_points`; its `tol` is in model units, so the default `0.01` can be too coarse for thin sheets authored in meters. If 3-D fitting distorts X/Y enough to disrupt prism booleans or volume, the `OCP.GeomAPI.GeomAPI_PointsToBSplineSurface` **height-grid overload** accepts a `TColStd_Array2OfReal` Z grid plus `X0, dX, Y0, dY`, preserving affine X/Y parameterization.

## Not available, and why

build123d's documentation shows all of these. They are not in scope here.

| what | why |
|---|---|
| BuildPart, BuildSketch, BuildLine | builder mode needs `with`, which the sandbox rejects — use algebra mode |
| edges(), faces(), solids(), vertices(), wires(), and their singular forms | these read the *active builder*; each raises `requires a Builder context to be in scope`. Use the methods on a shape: `part.edges()` |
| MM, CM, M, IN, FT | these constants assume millimeters (`M` is 1000); DyCraft interprets geometry coordinates as meters |
| Text, FontStyle, available_fonts | depends on installed fonts rather than on the geometry asked for |
| RigidJoint, RevoluteJoint, LinearJoint, CylindricalJoint, BallJoint | build123d's own joint system; DyCraft owns anchors and joints |
| import_step, import_stl, export_step, export_stl, export_gltf, export_brep, Mesher, ExportSVG, ExportDXF, export_to_pcbway | file and network I/O; `add_part` is the only supported way geometry leaves `object.py` — the harness stages the .brep and the mesh |
| DimensionLine, ExtensionLine, TechnicalDrawing, Draft, Arrow, ArrowHead | dimensioned 2D drawings, which this pipeline does not produce — and they render text from a system font |
| ArcArcTangentArc, ArcArcTangentLine, PointArcTangentArc, PointArcTangentLine | deprecated in build123d 0.11.1; use `ConstrainedArcs` / `ConstrainedLines`, both of which are bound |
| BoundBox, OrientedBoundBox, bounding_box | already reachable as methods: `part.bounding_box()`, `part.oriented_bounding_box()`, `Solid.from_bounding_box(...)` |
| new_edges is available; Select.LAST and Select.NEW are not usable | `Select.LAST`/`Select.NEW` need a builder context to diff against — `new_edges(a, b, combined=a + b)` is the algebra-mode equivalent |

## Algebra-mode arguments

- `mode=` controls combination with an active builder; in algebra mode combine returned shapes with `+`, `-`, or `&`.
- `clean=` also applies in algebra mode; it controls topology cleanup.
- Supply operands explicitly where a `None` default would otherwise take input from an active builder.
