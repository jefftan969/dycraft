# Animation source reference

`design/scene.py` and the files it loads are written with this SDK, the ObjectIR/SceneIR source API. They describe the geometry and articulated objects that `h.build()` imports into Blender.

## Coordinate convention

Author lengths in **meters** with **+Z up**, and each object with its natural front on **+Y**; scene placement supplies its world orientation.

| Direction | Axis |
|---|---|
| front / back | +Y / −Y |
| left / right | +X / −X |
| top / bottom | +Z / −Z |

Seen from +Y looking toward −Y, +X is on the viewer's left; anchor names and the brief's directions use this convention.

Source rotations (intrinsic Euler XYZ) and revolute `limits`, `default`, `.at(...)`, and `initial_joints` values are **degrees**; prismatic values are meters. Blender joint channels and `joint_at` events are radians.

Object anchors are part-local; scene anchor offsets compose in the resolved anchor's frame, including its rotation. An object's frame is its root part's authored frame: not recentered, and independent of the bbox `center` and `bottom` anchors.

The room frame is the world frame: its AABB is `((-w/2, -d/2, 0), (w/2, d/2, h))`, with the floor at `z = 0`, centered in X and Y.

## The authoring contract

`Object`, `Scene`, `Material`, `Emission`, `shader`, `lighting`, `Direction`, and `load_object` are injected alongside the geometry namespace. Build definitions with local functions or load object programs from other files.

### build123d vocabulary

**Algebra mode only:** `BuildPart`, `BuildSketch`, and `BuildLine` are unavailable. Build shapes as values, place them with `Pos(...) * shape` and combine with `+`, `-`, and `&`. `bd` is **not** the build123d module but the curated names available bare; `math`, `np`, and `OCP` are modules. The `build123d_authoring` skill's `references/namespace_reference.md` lists names and signatures.

### Object build API

- `Object(name)` — independent builder; refs from one object cannot be used in another.
- `obj.add_part(name, shape, *, material=None, tolerance=1e-3, angular_tolerance=0.1, planar_max_edge=None, expect_solids=None) -> PartRef`. Snapshots a finished `Solid`, `Part`, or `Compound` of solids (not faces, wires, sketches, or empty shapes); later Shape edits do not change it. `expect_solids=1` catches a fuse or cut that split the body; otherwise several solids may form one rigid part.
- `obj.add_anchor(part, name, *, translation=(0,0,0), rotation=(0,0,0)) -> AnchorRef` declares a mechanical frame in part-local coordinates.
- `obj.anchor(part, anchor_name, *, translation=None, rotation=None) -> AnchorRef` references a named or built-in anchor; `.offset_by(translation=..., rotation=...)` adds further offsets.
- `obj.declare_root(part)` — the single root; every other part needs one incoming joint.
- `obj.attach_fixed(name, parent, child, parent_anchor, child_anchor)`, `obj.attach_revolute(name, parent, child, axis, limits, default, parent_anchor, child_anchor)`, and `obj.attach_prismatic(...)`, with the revolute arguments, each `-> JointRef`.
- `obj.allow_collision(a, b, *, reason, joint=None, q_range=None)` exempts intended overlap between two of this object's parts, always or, with `joint`, over `q_range` (default: its limits); the animated `NoPenetration` check applies it at each frame's joint value. Do not mix blanket and scoped exemptions for a pair; it cannot exempt cross-instance collisions.
- `obj.add_check(claim)` — visual claim (§Assertions, references, and visual checks).
- `obj.finalize() -> dict` — seals the builder and returns the reusable definition.

Part arguments take a part name or `PartRef`, and `allow_collision(joint=...)` a joint name or `JointRef`. Two surfaces require refs: assertion entities and directions, and `attach_*`'s `parent_anchor`/`child_anchor`, which must be `AnchorRef`s from `obj.anchor(part, name)`.

At zero joint displacement the parent and child anchor frames coincide; `default` sets the rest configuration. Each anchor belongs to its own part's frame. The joint axis is expressed in the **parent anchor's frame**, pivoting at its origin: vectors, signed names (`"z"`, `"-y"`) and `Direction.world(...)` are all read in that frame, not in world.

Built-in anchors are the 27 AABB landmarks: `center`, six faces, twelve edges, and eight corners, named in top/bottom → front/back → left/right order (`top_front_right`, not `right_front`). They may lie in empty space and keep the part's orientation; a face name does not rotate +Z outward.

Names match `^[A-Za-z_][A-Za-z0-9_]*$`, up to 64 characters; part and joint names have separate namespaces. Custom anchors are unique on their part and cannot shadow built-ins. Give definition variants distinct names.

### Scene build API

- `scene = Scene(name)` — one per program.
- `scene.set_room(width, depth, height, floor_material=None, wall_material=None, ceiling_material=None, enclosure="interior") -> RoomRef`. Call once, before placement, anchors, groups, or assertions; `scene.room` returns the ref. The materials describe the **automatic shell** (the ceiling defaults to `wall_material`). `enclosure` is `"interior"` (floor, four walls, ceiling), `"open_top"` (no ceiling) or `"open_air"` (floor only). Unreplaced shell surfaces are real objects in the blend: every camera sees them and they light the scene.
- `scene.replace_room_surface(surface, instance=None)` removes shell surface `floor`, `ceiling`, `wall_xmin`, `wall_xmax`, `wall_ymin`, or `wall_ymax`, naming an already placed instance as its replacement, or none (§Room architecture).
- `scene.place(definition, *, name, pose, anchor=None, initial_joints=None) -> InstanceRef` takes an `Object` builder (finalizing it) or a finalized dict. `pose=((tx,ty,tz),(rx,ry,rz))` targets the selected anchor, by default the bbox-bottom-center in the configuration `initial_joints` selects, so `tz=0` rests an upright object on the floor. Tilting can move geometry below that anchor.
- `scene.attach(definition, *, name, parent, parent_anchor, child_anchor, follow=True, initial_joints=None) -> InstanceRef` brings `child_anchor` onto `parent_anchor`, each a name or `AnchorRef` resolved on `parent` and on `definition` (for an offset child anchor, `scene.anchor(definition, ...)`). Only `follow=True` with a part-scoped `parent_anchor` (`"<part>.<anchor>"`) parents the child to that part in Blender, so it rides the part's joints and motion. Any other attachment (`follow=False`, a whole-object anchor such as `"top"`, the room, a group) is parented to the world and stays put when the parent moves.
- `scene.anchor(entity, anchor_name, *, translation=None, rotation=None) -> AnchorRef` takes a room, instance, or group ref, **or a definition** in either form `place` accepts (a builder is finalized, so add every part, joint, and anchor first). Names are landmarks, `"<part>.<landmark>"`, or `"<part>.<declared_name>"`; a declared frame carries its rotation, so it positions and aims.
- `scene.create_group(*, name, members, root=None) -> GroupRef`. Members must already form a connected attachment tree; group anchors use their union AABB.
- `scene.attach_group(group, *, parent, parent_anchor, child_anchor)` re-anchors a group to an instance, group, or room; anchors resolve on `parent` and on `group`.
- `scene.add_check(claim)` — visual claim about the placed scene.
- `scene.finalize() -> dict`.

**Per-instance initial state.** `initial_joints={"door_hinge": 40}` sets one instance's joint values, within `limits`, without changing the shared definition; unnamed joints keep `default`. Placement, source assertions, support, and containment use it, and the instance's Blender joint channels start there, so a keyed value is the joint's value. Name instances as stable animation targets, such as `lathe_bay_2`.

### Room architecture

Modeled floors, walls, and ceilings replace shell surfaces through `replace_room_surface`:

- The replacement may straddle its named boundary by its own thickness. The room AABB is unchanged, and any other instance that leaves it is an error.
- `None` removes a surface outright (an open top, a missing wall). **It does not cut a doorway or window:** an opening is a hole in the authored geometry that replaces the surface.
- A modeled wall in front of an **unreplaced** shell wall leaves two layers; the shell keeps blocking light and seals any opening in the object.
- Room anchors remain whether a surface is present, replaced, or removed.

### Composing source

`load_object("objects/<name>/object.py")` runs an object program in a fresh restricted namespace and returns its finalized definition; the file builds an Object and assigns `ir = obj.finalize()`. Paths, including nested loads, are relative to `design/`. A loaded file does not see `scene.py`'s variables, so share values through local factories.

## Asset library

The asset library at `references/assets/` in your run directory holds ready-made people, animals, and robots. `INDEX.md` there lists each by category with a description, search tags, size, skeleton, and clips; `<name>/thumb.png` shows it.

`load_asset(name) -> ObjectIR` stages the named bundle into the build and returns its frozen definition, to place like any other: `scene.place(load_asset("g1_with_dex1_v3"), name="robot", pose=((0, 0, 0), (0, 0, 0)))`. An asset passed at launch with `--preload-asset` loads by the name given there.

An asset's description is its bundle's `references/assets/<name>/ir.json`; once a build has loaded it, `native/work/design_state/source/preloaded/manifest.json` also lists its `object_name`, `size`, `root`, `parts`, and joints with type, parent, child, and range (`limits_deg` revolute, `limits_m` prismatic). It cannot be edited or rebuilt: place it, anchor to it, and key its joints (`joint/<instance>__<joint>`) in Blender. Source and animated checks judge it like authored objects.

The manifest's `anatomical_limits.narrowed` ranges are tighter than the hardware range and exclude poses that read as a broken body. `place` accepts the hardware range; an advisory `AnatomicalJointLimits` report flags animated frames outside the narrowed one.

An imported character, whose IR carries `native_rig`, is one part with no joints, placed in its bind pose. In Blender its skinned body is `part/<instance>/body` and its armature `rig/<instance>/body`, both on the link `link/<instance>__body`. Its `native_rig` lists the bones, with rest heads and tails in the asset's own frame (meters, z up, feet on z = 0, facing +Y), and the asset's clips as Actions named `<object>/<clip>`, with their frame ranges. Animate it by posing the armature's bones, assigning those Actions with a slot from the armature's `animation_data.action_suitable_slots`, and adding constraints such as IK; a clip that plays one of those Actions needs every other Action in it keyed from the same first frame. Move it by its link, never the body or the armature; `key_joint` does not apply. Where the manifest's `native_rig.anatomy` lists what it checks, the advisory `AnatomicalJointLimits` flags frames in which an elbow, knee, shoulder, hip, forearm, wrist, ankle, or the head leaves a human range, and names the joint and the limit it passed. `native_rig.anatomy.bend_axes` gives each elbow's, knee's, and finger joint's flexing axis in its child bone's own frame, and `hands.palm_normal` where each palm faces; they differ between rig families (a 3ds Max Biped's fingers curl about -Z, a Mixamo's about +X), so read them rather than assume one.

## Materials and lighting

A material takes one of two forms, each optionally with `alpha=` and `emission=`:

```python
Material(color=(r, g, b), metallic=0.0, roughness=0.9)
Material(shader=surface)
```

`color` is RGB integers 0–255, required without a shader; `metallic` and `roughness` lie in [0, 1]. A shader material does not also set `color`, `metallic`, or `roughness`: the graph owns those sockets. `alpha` (in (0, 1]) and `emission` apply outside the graph. The room's shell arguments take the same forms.

A top-level `@shader` function takes `(nodes, links, bsdf)` or `(nodes, links, bsdf, mat)`: the node-tree collections, a Principled BSDF already wired to the output, and optionally the Material datablock. The body travels alone: literals, local values, and plain builtins are available; module constants, helpers, `math`, `radians`, `np`, and `bpy` are not. Attribute assignment is allowed inside it. Compilation checks the signature, free names, and literal node classes against Blender 4.5.13. Socket names, properties, and enum values are checked when Blender runs the body during the `h.build()` import, and a body that raises there fails the build, naming the material and the shader.

Cycles and Eevee execute shader graphs; Workbench does not. Object-coordinate inputs are the part's local meters; shading bumps do not change collision geometry. `alpha < 1` selects transparency; refraction and transmission belong in the shader.

`Emission(color, strength)` uses RGB integers and **W/m² of emitting surface** (power P over area A: P/A), covers the whole part, and lights the surroundings under Cycles.

### Lighting

`scene.set_lighting(rig)` takes a top-level `@lighting` function `def rig(L)`. Its self-contained body has `L`, plain builtins, and `radians`, `sin`, `pi`, and the other bare math helpers; no module constants, `math`, or `bpy`.

| Interface | Contract |
|---|---|
| `L.new_light(name, type="POINT")` | A light object (`POINT`, `SPOT`, `SUN`, `AREA`); set its transform and `.data` in meters, radians, and native energy. AREA emits along local −Z; `L.aim(light, target_xyz)` points −Z at a world point. |
| `L.world_nodes`, `L.world_links`, `L.world_output` | The World starts with only an output node; an unconnected `Surface` input is black. |
| `L.view` | View transform, look, exposure, white balance; set `view_transform` before `look`, whose options depend on it. |
| `L.display`, `L.render`, `L.cycles` | Display, film, and light-transport settings. |
| `L.image(relative_path)` | Loads an existing image by a path relative to `design/`; absolute paths and `..` are rejected. |

The block runs in Blender, after geometry and materials, at the first `h.build()` import, and again with `refresh={'lighting': True}`, which brings back its lights, World, and `L.view`, `L.display`, `L.render`, and `L.cycles` settings. Harness renders override engine, resolution, samples, and denoising. A block that raises fails `h.build()` with a `LightingError` naming it.

## Assertions, references, and visual checks

Object assertions measure a definition at its `default` configuration; scene assertions measure placed entities and the room at each instance's `initial_joints`, then `default`. A ref's `.at({joint_name: q, ...})` selects another configuration for that assertion only; keys outside the ref's ancestor chain are ignored, and scene keys may be qualified as `"<instance>.<joint>"`.

Direction sources: `joint.axis()` for a moving joint, `.outward()` on a built-in landmark anchor, and `Direction.world((x,y,z))` / `world_x()` / `world_y()` / `world_z()`. `outward()` exists only for built-in **non-center** landmarks; a declared anchor's authored frame already carries its orientation.

Assertion `axis=` arguments use world components, unlike joint-constructor axes. AABB assertions measure world-aligned boxes at the selected configuration.

Both builders provide:

| Assertion | Semantics |
|---|---|
| `assert_distance(*, a, b, axis=None, min, max, name)` | Euclidean anchor distance, or signed projection of b−a onto the world axis. |
| `assert_collinear(*anchors, axis, tol, name)` | Two or more anchors within perpendicular distance tol of the line through the first along axis. |
| `assert_overlap(*, a, b, axis, min, max=None, name)` | Signed AABB overlap (negative is a gap); omitted max is unbounded. |
| `assert_angle(*, a, b, min_deg, max_deg, name)` | Unsigned angle between DirectionRefs, 0–180 degrees. |
| `assert_contact(*, a, b, tol, name)` | Anchor distance at most tol; not surface contact. |
| `assert_within(*, inner, outer, axes=("x","y","z"), margin=0.0, name)` | Opposing face-anchor positions along the world axes; positive margin permits overhang. |
| `assert_aabb_overlap(*, a, b, axes, min_overlap, name)` | AABB overlap at least min_overlap on each axis. |
| `assert_parallel(*, a, b, tol_deg=1.0, name)` | DirectionRefs within tol_deg of parallel; `assert_antiparallel` and `assert_perpendicular` take the same arguments. |
| `assert_flush(*, face_a, face_b, axis_world, tol_pos=0.001, tol_deg=1.0, opposing=True, name)` | Anchor separation projected onto literal world DirectionRef axis_world; outward directions oppose, or are parallel if opposing=False. Supply the intended face normal. |

An object's `assert_overlap`, `assert_within`, and `assert_aabb_overlap` take `PartRef`s; the scene's take instances, groups, or the room. Objects also provide `obj.assert_axis_world_aligned(*, joint, world_axis, tol_deg=1.0, name)` for unsigned joint-axis alignment. A rotated entity's face anchors are not its world-AABB extremes, and no anchor or AABB relation establishes contact between real surfaces.

`obj.add_check(claim)` and `scene.add_check(claim)` register short, falsifiable visual claims, at most 16 unique claims each; numeric tolerances belong in assertions. `h.review` reads them.

## Mesh resolution

`add_part` tessellates for rendering and deformation: `tolerance=1e-3` is relative chord deflection and `angular_tolerance=0.1` is radians; smaller values refine curves. Parts target 20,000 triangles and are coarsened automatically above that. `planar_max_edge` (meters) subdivides flat faces that must bend; the achieved, possibly coarser spacing is `planar_max_edge_achieved` in the `geometry_notes` of `native/work/design_state/build.json`.

## File skeleton

One complete program: shader and scalar materials, a reused definition, architecture replacing a shell surface, an attachment, and the lighting.

```python
@shader
def stone(nodes, links, bsdf):
    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 14.0
    links.new(grain.outputs["Fac"], bsdf.inputs["Roughness"])
    bsdf.inputs["Base Color"].default_value = (0.68, 0.66, 0.62, 1.0)

STONE = Material(shader=stone)
BRASS = Material(color=(181, 140, 66), metallic=1.0, roughness=0.35)

def make_block(name, w, d, h, material):
    obj = Object(name)
    obj.declare_root(obj.add_part("body", Box(w, d, h), material=material,
                                  expect_solids=1))
    return obj.finalize()

scene = Scene("display")
scene.set_room(4, 3, 2.8)

# Slab top at z = 0; unreplaced, the shell floor would coincide with it.
floor = scene.place(make_block("floor_slab", 4.0, 3.0, 0.08, STONE),
                    name="floor_slab", pose=((0, 0, -0.08), (0, 0, 0)))
scene.replace_room_surface("floor", floor)

plinth = make_block("plinth", 0.4, 0.4, 0.6, STONE)   # one definition, two instances
left = scene.place(plinth, name="plinth_left", pose=((-0.7, 0, 0), (0, 0, 0)))
scene.place(plinth, name="plinth_right", pose=((0.7, 0, 0), (0, 0, 0)))

cap = Object("cap")
cap.declare_root(cap.add_part("body", Box(0.3, 0.3, 0.05), material=BRASS))
scene.attach(cap, name="cap_left", parent=left,      # part-scoped: rides the plinth
             parent_anchor=scene.anchor(left, "body.top"),
             child_anchor=scene.anchor(cap, "bottom"))

@lighting
def rig(L):
    key = L.new_light("key", type="AREA")
    key.location = (1.4, 2.0, 2.3)
    key.data.energy = 180.0
    L.aim(key, (0.0, 0.0, 0.7))
    bg = L.world_nodes.new("ShaderNodeBackground")
    L.world_links.new(bg.outputs["Background"], L.world_output.inputs["Surface"])
    L.view.view_transform = "AgX"     # before any L.view.look
    L.view.exposure = -0.3

scene.set_lighting(rig)
ir = scene.finalize()
```

## Source-style constraints

`scene.py` and files loaded through `load_object` run under restricted Python:

- No `import`, `with`, `async`, or `global`.
- No `exec`, `eval`, `compile`, `__import__`, `open`, `globals`, `locals`, `vars`, `dir`, `getattr`, `setattr`, `delattr`, `hasattr`, `type`, or `object`, and no underscore-prefixed attributes.
- Your own lists and dicts are mutable; outside `@shader` and `@lighting` bodies, writes through attributes or call results and mutation of injected objects are refused.
- `print()` output is discarded; return evidence through the IR and build reports.
- A build of `scene.py`, every object it loads included, has 900 s; one that runs out fails with a timeout naming the file.
