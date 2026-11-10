"""dycraft.sdk — object-level + scene-level build and query SDK (v1).

Public API:

    from dycraft.sdk import Object, Scene, ObjectHandle, Material
    from dycraft.sdk import object_query, scene_query

Object layer example (inside `object.py`, where the curated build123d names are
injected as globals):

    obj = Object("microwave")
    steel = Material(color=(190, 190, 195), metallic=0.6, roughness=0.4)

    carcass = Box(0.50, 0.40, 0.30) - Pos(0, -0.02, 0) * Box(0.46, 0.38, 0.26)
    body = obj.add_part("body", carcass, material=steel)

    obj.declare_root(body)
    ir = obj.finalize()

`add_part` takes a finished build123d Shape and freezes it as one rigid part;
build every operation that defines the part before the call.

Scene layer example (inside `scene.py`, which builds its own objects):

    table = Object("dining_table")
    top = table.add_part("top", Box(1.2, 0.8, 0.75),
                         material=Material(color=(140, 100, 60)))
    table.declare_root(top)

    scene = Scene("kitchen")
    scene.set_room(width=4.0, depth=5.0, height=2.6,
                   floor_material=Material(color=(170, 130, 90)),
                   wall_material=Material(color=(235, 230, 220)))
    scene.place(table, name="table", pose=((0, 0, 0), (0, 0, 0)))
    ir = scene.finalize()
"""

from ._material import (Emission, Mapping, Material, Shader, Texture,
                        TextureMap)
from ._lighting_block import LightingBlock, lighting
from .object_builder import Object, PartRef, JointRef
from .scene_builder import (
    Scene,
    ObjectHandle,
    RoomRef,
    InstanceRef,
    GroupRef,
    AnchorRef,
    AssertionRef,
)
from ._refs import DirectionRef, Direction
from ._errors import (
    DycraftError,
    SandboxViolation,
    BuildError,
    SourceRuntimeError,
    GeometryKernelError,
    # object-layer subclasses
    InvalidPrimitive,
    InvalidJoint,
    DanglingReference,
    RootNotDeclared,
    NotConnectedToRoot,
    # scene-layer subclasses
    InvalidScene,
    InvalidRoom,
    RoomNotDeclared,
    NoInstances,
    InvalidInstance,
    InvalidGroup,
    InvalidAnchor,
    InvalidAssertion,
    # scene-agent catalog-validation subclasses
    InvalidCatalog,
    DuplicateObjectName,
    # orchestrator-layer subclasses
)
from . import _object_kernel as object_query
from . import _scene_kernel as scene_query

__all__ = [
    # value types
    "Material",
    "Emission",
    "Shader",
    "LightingBlock",
    "lighting",
    "Texture",
    "TextureMap",
    "Mapping",
    # object layer
    "Object",
    "PartRef",
    "JointRef",
    "object_query",
    # scene layer
    "Scene",
    "ObjectHandle",
    "RoomRef",
    "InstanceRef",
    "GroupRef",
    "AnchorRef",
    "AssertionRef",
    "DirectionRef",
    "Direction",
    "scene_query",
    # errors
    "DycraftError",
    "SandboxViolation",
    "BuildError",
    "SourceRuntimeError",
    "GeometryKernelError",
    "InvalidPrimitive",
    "InvalidJoint",
    "DanglingReference",
    "RootNotDeclared",
    "NotConnectedToRoot",
    "InvalidScene",
    "InvalidRoom",
    "RoomNotDeclared",
    "NoInstances",
    "InvalidInstance",
    "InvalidGroup",
    "InvalidAnchor",
    "InvalidAssertion",
    "InvalidCatalog",
    "DuplicateObjectName",
]
