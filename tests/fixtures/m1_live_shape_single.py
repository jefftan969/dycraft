# One rigid part, built as a live build123d Shape.
#
# Run with:
#   python -m dycraft.agent.layers.fixed_source \
#       --source tests/fixtures/m1_live_shape_single.py \
#       --run-dir runs/m1_live_shape_single
#
# Everything here happens in ONE execution of ONE file: build123d constructs
# the geometry, `add_part` registers it, and `finalize` compiles it into BREP,
# a render mesh, an AABB, shape facts, and an ObjectIR.
#
# The shape is chosen so a silent renderer fallback to the bounding box would
# be obvious rather than plausible: it is shelled (open at the front), has a
# filleted vertical edge, and carries a through-bore. Its bounding box is ~4x
# its actual volume, so a fallback would draw a solid brick where the real part
# is a thin-walled channel.

steel = Material(color=(150, 150, 160), metallic=0.85, roughness=0.35)


def build_bracket():
    """A shelled channel with a filleted front edge and a cross bore."""
    # Outer block, hollowed from the front (+Y): the inner box runs the full
    # depth so it breaks through, leaving an 11 mm back wall.
    shell = Box(0.20, 0.12, 0.16) - Pos(0, 0.006, 0) * Box(0.18, 0.12, 0.14)
    # An 8 mm bore straight through the left/right walls, near the top.
    shell -= Pos(0.09, 0, 0.05) * Rotation(90, 0, 0) * Cylinder(0.004, 0.30)
    # Round the leftmost pair of vertical edges — a fillet is impossible to
    # fake with a primitive, so it is also a fallback tell.
    return fillet(shell.edges().filter_by(Axis.Y).group_by(Axis.X)[0], 0.004)


obj = Object("m1_bracket")
body = obj.add_part("body", build_bracket(), material=steel)
obj.declare_root(body)

ir = obj.finalize()
