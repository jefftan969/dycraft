# A minimal articulated object, both parts built as live
# build123d Shapes in this one file.
#
# Run with:
#   python -m dycraft.agent.layers.fixed_source \
#       --source tests/fixtures/m1_live_shape_articulated.py \
#       --run-dir runs/m1_live_shape_articulated \
#       --poses 'closed:door_hinge=0;open:door_hinge=-75'
#
# Geometry, part registration, anchors, the joint, and finalization all happen in
# ONE execution of ONE program. Nothing is authored twice, nothing is fetched
# back by name.
#
# Both shapes are deliberately non-boxy, so a renderer that silently fell back
# to bounding boxes would draw two plain slabs and the difference would be
# unmissable: the carcass is shelled (87% of its box is air) and the door
# carries integral hinge knuckles, two through-bores and a recessed pull.
#
# Axis convention: +X left, +Y front, +Z top.
# The cabinet therefore opens toward +Y and the hinge runs along Z.

WOOD = Material(color=(150, 111, 74), metallic=0.0, roughness=0.75)
STEEL = Material(color=(160, 163, 168), metallic=0.9, roughness=0.3)

DEPTH = 0.40
WALL = 0.011
HINGE_Z = (-0.22, 0.22)


def build_carcass():
    """A box shelled out from the front, leaving an 11 mm back wall."""
    outer = Box(0.50, DEPTH, 0.60)
    # Full-depth inner box offset forward, so it breaks through the front face.
    return outer - Pos(0, WALL, 0) * Box(0.50 - 2 * WALL, DEPTH, 0.60 - 2 * WALL)


def build_door():
    """A panel with fused hinge knuckles, 6 mm bores and a recessed pull."""
    panel = Box(0.49, 0.018, 0.58)
    for z in HINGE_Z:
        # The knuckle is part of the same rigid body — fusing it here means one
        # part rather than a panel plus hardware held on by fixed joints.
        panel += Pos(0.245, 0, z) * Rotation(90, 0, 0) * Cylinder(0.011, 0.05)
        panel -= Pos(0.245, 0, z) * Rotation(90, 0, 0) * Cylinder(0.003, 0.06)
    # A recessed finger pull on the far side from the hinge.
    panel -= Pos(-0.18, -0.006, 0) * Box(0.10, 0.02, 0.03)
    return fillet(panel.edges().filter_by(Axis.Y).group_by(Axis.X)[0], 0.006)


obj = Object("m1_cabinet")

carcass = obj.add_part("carcass", build_carcass(), material=WOOD)
door = obj.add_part("door", build_door(), material=STEEL)

obj.declare_root(carcass)

# The door's hinge edge meets the carcass's front-left vertical edge. Anchors
# are the existing 27 AABB landmarks and the joint uses the existing free axis
# vector — Milestone 1 changes neither.
obj.attach_revolute(
    "door_hinge",
    parent=carcass,
    child=door,
    axis=(0.0, 0.0, 1.0),
    limits=(-110.0, 0.0),
    default=0.0,
    parent_anchor=obj.anchor(carcass, "front_left"),
    child_anchor=obj.anchor(door, "back_left"),
)

ir = obj.finalize()
