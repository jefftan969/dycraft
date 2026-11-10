"""Material value type for the DyCraft SDK.

A `Material` bundles the surface appearance of a part / floor / wall: an
RGB base color plus PBR scalars (metallic, roughness, alpha), optionally a
`Texture` — the PBR maps the surface samples, keyed by kind — and optionally
a `Mapping` saying how the surface was parameterized. It is the single place
where surface-appearance fields are validated and serialized;
`Object.add_part`, `Object.make_box`, and `Scene.set_room` all route through
it.

**`to_ir()` omits `texture` when there is none, and that is load-bearing.**
`sdk/ir.py::content_hash` hashes the whole canonical dict, so emitting the
key unconditionally — even as `null` — would change the id of every existing
object, and with it every catalog ref, render-cache key and verifier-cache
entry.

Map kinds are open-ended data rather than schema. `base_color`, `roughness`,
`metallic`, `normal`, `height`, and `emission` are the kinds the renderers know
how to sample; the importers are the only producers today, and they supply
`base_color` alone. Every consumer samples whichever kinds it knows: Cycles
binds all six, Open3D binds base color, roughness, metallic, and emission, and
the browser viewer binds base color and roughness. A kind a consumer does not
know is still carried everywhere and simply not sampled, which is the right
degradation — adding one is therefore a data change, not a schema change (see
`texture_maps_from_ir`, which enumerates no kinds on purpose). See `Texture`
for why base color is an ordinary kind rather than a distinguished field.

Future extensions (deferred):
    - additional PBR fields (clearcoat, transmission)
    - `normal` on any backend but Cycles. Filament has `normal_img` and derives
      tangents from UVs unaided (measured), so this is a deferral rather than a
      limit — but binding it there also means gating on UV provenance, because
      `_world._triplanar_uvs` picks a projection axis per TRIANGLE and hands a
      projected surface a discontinuous tangent basis. On Cycles it IS bound,
      guarded on the mesh actually having a UV layer.
    - a spatial `metallic` map. The SCALAR is reachable — `Material(...,
      metallic=...)` — so a clean homogeneous metal is already expressible;
      what stays deferred is the spatially MIXED surface (rust, chipped paint
      over steel), which no scalar represents correctly.

The procedural surface is `Material.shader`: agent-authored Blender shader-node
code, carried past the CAD compile and executed in the render child
(`sdk/_shader`). It is Cycles-only and says so; every other consumer draws the
scalars. Emission is `Material.emission`.

An image-backed material cannot be authored. The importers are the only
producers of a `Texture`, and everything downstream of this module is in place
(the map-kind vocabulary, the per-corner UV bake and its per-part tile phase in
`_solid_compile.part_uv_phase`, and all three renderers' sampling), so a
producer that returns a `Texture` plus a `Mapping` needs no changes below it.

The class is a frozen value object (eq + hash by value) so it can be
reused freely across parts (e.g. `wood = Material(...)`, then
`obj.add_part(..., material=wood)` for many parts).
"""

from __future__ import annotations

import builtins
import math
from dataclasses import dataclass, field

from ._errors import InvalidPrimitive


# ---------------------------------------------------------------------------
# Defaults (preserved from the pre-Material flat-gray baseline)
# ---------------------------------------------------------------------------

DEFAULT_PART_COLOR = (176, 176, 176)
DEFAULT_FLOOR_COLOR = (153, 153, 153)
DEFAULT_WALL_COLOR = (199, 199, 199)

# PBR scalar defaults. roughness=0.9 + metallic=0.0 reproduces the previous
# Lambertian-looking appearance under Filament's defaultLit shader.
DEFAULT_METALLIC = 0.0
DEFAULT_ROUGHNESS = 0.9
DEFAULT_ALPHA = 1.0

#: The texture map kind that pairs with `Material.emission`. Named here rather
#: than spelled as a literal at each site, because three modules have to agree
#: on it: this validator, and both renderers.
EMISSION = "emission"

#: Tile size handed to the UV baker for a material whose surface is a `shader`
#: rather than an image.
#:
#: **A shader still needs UVs, and it gets none for free.** `_solid_compile`
#: bakes per-corner coordinates only when `add_part` passes a tile size, and
#: `add_part` takes that from `Mapping.scale_m` — which a shader material has
#: none of, because there is no image to lay out. Without this a
#: `ShaderNodeTexCoord`'s `UV` output samples a layer that was never written and
#: every UV-driven graph renders one flat color.
#:
#: 1.0 so the coordinates ARE meters: a `Scale` of 1 in the graph is one feature
#: per meter, which is a unit an author can reason about from the dimensions
#: they just typed. It is deliberately NOT recorded as a `Mapping` in the IR —
#: `Mapping` sizes images, there are none, and `Material` refuses the pair.
SHADER_UV_SCALE_M = 1.0

_COLOR_MIN = 0
_COLOR_MAX = 255


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextureMap:
    """One staged image, and how to read it.

    Fields
    ------
    image : str
        Run-relative staged path, ``assets/textures/<digest><ext>``. Relative
        so a run directory stays movable, and under ``assets/`` because that is
        the only subtree `scene_context.referenced_object_assets` carries into
        a catalog.
    digest : str
        blake2b-128 of the staged bytes — the same digest that names the file.
    channel : str | None
        Which channel of the image holds this map, for a PACKED source. glTF
        packs occlusion/roughness/metallic into one image as R/G/B, and the
        assets in this repo use it (`Michelle.glb` and `readyplayer.me.glb`
        both ship a `metalRough` texture). Splitting such an image would mean
        re-encoding it, which puts the installed Pillow version into the digest
        and therefore into the id of every object that used it — exactly what
        staging bytes verbatim exists to prevent. So a packed source is staged
        once and referenced twice, the two entries differing only here.
    strength : float | None
        The map's own scalar weight, for the kinds that have one — a normal
        map's `normalScale`, an occlusion map's strength. Per map rather than
        per material because it describes this image's contribution, not the
        surface's parameterization.

    Both optional fields are omitted from the IR when None, so a plain map
    serializes to exactly the two keys it always did.
    """

    image: str
    digest: str
    channel: str | None = None
    strength: float | None = None

    def __post_init__(self):
        _check_asset_path(self.image, "texture_map.image")
        _check_digest(self.digest, "texture_map.digest")
        if self.channel is not None and (
                not isinstance(self.channel, str) or not self.channel):
            raise InvalidPrimitive(
                f"texture map channel must be a non-empty string, got "
                f"{self.channel!r}",
                evidence={"param": "texture_map.channel",
                          "value": repr(self.channel)},
            )
        if self.strength is not None:
            _check_strength(self.strength, "texture_map.strength")
            object.__setattr__(self, "strength", float(self.strength))

    def to_ir(self) -> dict:
        d = {"image": self.image, "digest": self.digest}
        if self.channel is not None:
            d["channel"] = self.channel
        if self.strength is not None:
            d["strength"] = self.strength
        return d


@dataclass(frozen=True)
class Texture:
    """The PBR maps a material samples, keyed by kind.

    Kinds are ``"base_color"``, ``"roughness"``, ``"metallic"``, ``"normal"``
    — whatever a producer supplies; an importer supplies the first. A dict
    rather than one
    field per kind so that adding a kind is a data change, not a schema change:
    every layer between here and the renderers moves maps by iterating this,
    and only the renderers know what a given kind MEANS. A kind they do not
    know is carried everywhere and simply not sampled, which is the right
    degradation — the material still renders, on its scalars.

    **Base color is a map like any other**, in one uniform dict, so no
    consumer needs a predicate to tell a map from a non-map structurally.

    **A map replaces the scalar it corresponds to; it does not tint it.**
    `color` is the fallback for `base_color`, `roughness` for a roughness map,
    `metallic` for a metallic map — and each is what a consumer that cannot
    sample the image draws. Both renderers MULTIPLY a bound map by the
    corresponding factor (Filament does `albedo_img * base_color` and
    `roughness_img * base_roughness`; three does
    `roughnessFactor *= texelRoughness.g`), so honouring "replaces" means
    binding the IDENTITY value — white, or 1.0 — beside the map rather than
    ignoring the scalar. Filament multiplies vertex color in as well, so both
    multiplicands go to identity together or the contract is only half true.
    See `render/open3d_renderer.py::_pbr_material`.

    Frozen, like `Material`, so a Material carrying one stays hashable and
    reusable across parts. `map_items` is a sorted tuple of pairs for exactly
    that reason; `.maps` re-presents it as a dict.
    """

    #: Sorted tuple of `(kind, TextureMap)`, so the value stays hashable and
    #: its canonical JSON is order-independent. Read `.maps`.
    map_items: tuple = ()

    def __post_init__(self):
        items = self.map_items
        if isinstance(items, dict):
            items = tuple(items.items())
        norm = []
        seen: set = set()
        for kind, m in items:
            if not isinstance(kind, str) or not kind:
                raise InvalidPrimitive(
                    f"texture map kind must be a non-empty string, got {kind!r}",
                    evidence={"param": "texture.maps", "value": repr(kind)},
                )
            if not isinstance(m, TextureMap):
                raise InvalidPrimitive(
                    f"texture map {kind!r} must be a TextureMap, got "
                    f"{type(m).__name__}",
                    evidence={"param": f"texture.maps[{kind!r}]",
                              "constraint": "type:TextureMap"},
                )
            if kind in seen:
                # Ambiguous rather than merely redundant: `.maps` and `to_ir`
                # would keep one of the two with nothing recording which, and
                # `sorted` would fall through to comparing the TextureMaps
                # themselves — a bare TypeError from a frozen dataclass with no
                # ordering. A packed image shared by two kinds is two DIFFERENT
                # kinds naming one file (see `TextureMap.channel`), never one
                # kind twice.
                raise InvalidPrimitive(
                    f"texture map kind {kind!r} was given more than once; a "
                    f"kind names exactly one image",
                    evidence={"param": "texture.maps", "value": kind,
                              "constraint": "one map per kind"},
                    suggested_repairs=[
                        "drop the duplicate, or pass a dict so the kinds are "
                        "unique by construction",
                    ],
                )
            seen.add(kind)
            norm.append((kind, m))
        if not norm:
            raise InvalidPrimitive(
                "a Texture with no maps is not a texture; pass None instead",
                evidence={"param": "texture", "constraint": "at least one map"},
                suggested_repairs=[
                    "drop the texture and let the material's scalars describe "
                    "the surface",
                ],
            )
        object.__setattr__(
            self, "map_items", tuple(sorted(norm, key=lambda kv: kv[0])))

    @property
    def maps(self) -> dict:
        """The maps by kind."""
        return dict(self.map_items)

    def to_ir(self) -> dict:
        """``{kind: {"image", "digest", ...}}`` — one entry per map, no more."""
        return {kind: m.to_ir() for kind, m in self.map_items}


@dataclass(frozen=True)
class Mapping:
    """How a surface is parameterized, when something had to decide.

    Fields
    ------
    scale_m : float
        Real-world edge length of one tile, in meters. Cedar planks at 0.6
        read as one plank across a 0.6 m table top; at 0.25, as planks.

    **This is authoring and staging metadata, not a renderer input for a staged
    asset.** UVs live in the prepared mesh: a CAD solid's are baked from its
    B-rep faces at this tile size when it is staged, and an imported asset's
    were authored into it. By the time a renderer sees either, the coordinates
    are on the mesh and this value has already done its work — which is why it
    is a sibling of `texture` rather than a field inside it, and why a material
    may legitimately carry maps with no `mapping` at all. That case is the
    imported asset, and it needs no special handling anywhere precisely because
    the absence is structural rather than a flag.

    The one place it still reaches draw time is geometry that has no staged
    mesh to have baked into — an analytic box part, and the room's floor and
    walls, which are synthesized per camera from the room bounds. There it is
    the projection's tile size, passed to it directly.

    **Never use the presence or absence of `mapping` as the renderer's
    UV-provenance discriminator.** The mesh is the authority: existing mesh UVs
    always win — `open3d_renderer._apply_texture_uvs` returns early on
    `mesh.has_triangle_uvs()` — and `scale_m` is consulted only where
    projection is actually required, because nothing else can parameterize the
    surface. Branching on `mapping` instead would re-project a staged asset
    that already carries authored coordinates, discarding the seam placement
    and atlas registration its author chose. That failure is silent: a
    re-projected surface is still plausibly textured, so it looks wrong only to
    someone who knows what it should have looked like.
    """

    scale_m: float

    def __post_init__(self):
        _check_scale(self.scale_m, "mapping.scale_m")
        object.__setattr__(self, "scale_m", float(self.scale_m))

    def to_ir(self) -> dict:
        return {"scale_m": self.scale_m}


@dataclass(frozen=True)
class Emission:
    """Light this surface gives off, on top of the light it reflects.

    Fields
    ------
    color : (r, g, b)
        sRGB bytes, the same convention as `Material.color`. The *hue* of the
        emitted light. Defaults to white, because an author who writes
        `Emission(strength=...)` without a color means white. This does not
        soften the `strength=0` refusal below: that one answers *whether a
        surface is a light*, which is structural, while a color is a property
        of a light that already exists.
    strength : float
        Radiant exitance in W/m² of surface. A backlit screen is single digits;
        a bare bulb's glass, tens to hundreds. Zero is refused — a material that
        emits nothing should have `emission=None`, so that "is this surface a
        light source" is answered structurally rather than by comparing a float
        to zero.

        It carries a default only because `color` has one and a dataclass
        cannot put a defaulted field first without reordering the documented
        positional signature. **1.0 W/m² is dim on purpose** — it is a legal
        emitter and a visible one is orders of magnitude brighter, so state
        `strength` whenever you mean the surface to read as lit.

    **This is a PART property, and that is load-bearing.** `Material` hangs off
    `part["material"]`; there is no object-level material in the IR. So a lamp
    emits from whichever parts its author chose to make emissive — its bulb,
    or the inner face of its shade — and getting that right is a decomposition
    decision, not a color decision. An author who paints a whole lamp emissive
    gets a glowing lump of plastic.

    **The two renderers differ here in a way that matters.** Cycles treats an
    emissive surface as a real light source: it illuminates its surroundings,
    casts light into the room, and shows up in reflections. Filament only makes
    the surface *look* bright — a rasterizer does not gather light from
    geometry. So a scene lit entirely by emissive parts is correct under the
    Blender backend and nearly black under the Open3D one. That asymmetry is
    documented rather than papered over, for the same reason the unsampled map
    kinds are: hiding it turns a capability gap into a mystery.
    """

    color: tuple[int, int, int] = (255, 255, 255)
    strength: float = 1.0

    def __post_init__(self):
        _check_color(self.color, "emission.color")
        _check_emission_strength(self.strength, "emission.strength")
        object.__setattr__(
            self, "color",
            (int(self.color[0]), int(self.color[1]), int(self.color[2])),
        )
        object.__setattr__(self, "strength", float(self.strength))

    def to_ir(self) -> dict:
        return {"color": [self.color[0], self.color[1], self.color[2]],
                "strength": self.strength}


# ---------------------------------------------------------------------------
# Shader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shader:
    """A procedural surface, as the Blender shader-node code that builds it.

    Fields
    ------
    name : str
        The authored function's name. The Blender child `exec`s `body` and then
        calls this, so the two travel together or neither means anything.
    body : str
        The `def <name>(nodes, links, bsdf): ...` source, verbatim out of the
        authored `object.py`. Ordinary Blender shader-node Python — see
        `sdk/_shader.py::shader`, which is the only thing that builds one.

    **The source is carried inline rather than staged as an asset**, and that is
    a considered choice rather than the lazy one. `stage_solid` already stages
    the whole `object.py` as every part's `source`, so the text is on disk as a
    content-addressed `.py` file either way and a second copy would add no
    provenance. What it would add is a path: three places that resolve material
    assets (`scene_context.referenced_object_assets`,
    `viewer.scene._rebase_material`, `stage_catalog_object`) would each have to
    learn about it, and each fails by LOSING the material rather than degrading
    it. Inline has no path to go stale.

    Hashing is unaffected either way: `body` enters `content_hash` directly here
    and would enter it through a digest-named filename if staged. The body is a
    substring of a file the agent wrote, so it is byte-stable across a run's
    compiles, which is why this needs no cache.
    """

    name: str
    body: str
    #: The parameter names the body declares, so the render child can call a
    #: three-argument body and a four-argument (material-aware) body correctly
    #: without inspecting the function it just `exec`d. Empty means the
    #: three-argument `(nodes, links, bsdf)` form, which is how an IR without
    #: `params` reads back. A tuple, not a list:
    #: `Shader` is a frozen dataclass that materials are deduplicated by, so
    #: an unhashable field would silently make every shader material unique.
    params: tuple = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise InvalidPrimitive(
                f"shader.name must be a Python identifier, got "
                f"{self.name!r}",
                evidence={"param": "shader.name", "value": repr(self.name),
                          "constraint": "identifier"},
            )
        if not isinstance(self.body, str) or not self.body.strip():
            raise InvalidPrimitive(
                "shader.body must be non-empty source text",
                evidence={"param": "shader.body",
                          "constraint": "non-empty str"},
            )
        # Compiled here, in the interpreter that has no bpy, so a syntax error
        # is an author-time compile report rather than a render-child traceback
        # minutes later. It only proves the text PARSES — a wrong socket name
        # still needs Blender to find out.
        try:
            builtins.compile(self.body, f"<shader:{self.name}>", "exec")
        except SyntaxError as e:
            raise InvalidPrimitive(
                f"shader {self.name!r} does not parse: {e.msg}",
                evidence={"param": "shader.body", "detail": e.msg,
                          "lineno": e.lineno or 0},
                suggested_repairs=["fix the syntax error in the shader body"],
            ) from None

    def to_ir(self) -> dict:
        d = {"name": self.name, "body": self.body}
        if self.params:
            d["params"] = list(self.params)
        return d


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    """Surface appearance: PBR scalars, image maps, or a procedural shader.

    Fields
    ------
    color : tuple[int, int, int] | None
        Base RGB color; each channel an int in [0, 255]. Required unless the
        material carries a `shader`, which computes the surface itself.
    metallic : float | None
        Cook-Torrance metallic factor in [0, 1]. 0 = dielectric (wood,
        plastic, fabric), 1 = pure metal (steel, aluminum). Default 0.0.
    roughness : float | None
        Microfacet roughness in [0, 1]. 0 = mirror, 1 = fully diffuse.
        Default 0.9 (matte).
    alpha : float | None
        Opacity in (0, 1]. 1.0 = fully opaque; < 1.0 selects Filament's
        `defaultLitTransparency` shader. Default 1.0. Alpha = 0 is
        rejected. Animation-time absence is authored on an instance with the
        boolean ``exists`` channel; static invisible parts are unsupported.

        The three scalars above default to `None` in the signature and are
        filled from `DEFAULT_METALLIC` / `DEFAULT_ROUGHNESS` / `DEFAULT_ALPHA`
        by `__post_init__` **only when there is no `shader`** — see `to_ir`.
    texture : Texture | None
        The PBR maps this surface samples, keyed by kind. Default None.

        **The scalars above are the fallbacks for the maps below.** `color`
        falls back for `base_color`, `roughness` for a roughness map,
        `metallic` for a metallic map. Where a map is present it REPLACES its
        scalar rather than tinting it — see `Texture` for why honouring that
        means binding an identity factor rather than ignoring the scalar — and
        where it is absent the scalar is what every consumer draws. So a
        scalar should still be authored as a plausible approximation of the
        image it stands in for; an importer fills it with the atlas's own
        mean.

        `alpha` is the one scalar with no corresponding map, deliberately.
        Opacity in a base-color image's alpha channel is not read.

    mapping : Mapping | None
        How the surface was parameterized, when something had to decide.
        Authoring and staging metadata — see `Mapping`. Legitimately absent on
        a material whose maps are sampled through UVs the asset already had.

    shader : Shader | None
        A procedural surface as Blender shader-node code. Default None.

        **Where present it OWNS the appearance sockets** — base color,
        roughness, metallic, the shading normal.

        **A shader must not be authored alongside `color` / `metallic` /
        `roughness`** — see `SHADER_OWNED_FIELDS`. `check_material` carries the
        refusal for that at the authoring boundary, currently deferred and
        documented there. The intended authored forms are
        `Material(shader=s)` plus, where the surface needs them, `alpha=` and
        `emission=`: those two are applied by the harness outside the graph,
        so they are the only properties a shader genuinely cannot express.
        The scalar fields stay on this dataclass for READING — a stored IR
        that carries both still parses, round-trips byte-identically, and keeps
        its content hash — and for the renderer fallback described below,
        which is supplied at the packing boundary rather than authored.

        Mutually exclusive with `texture` for the same reason, and that one is
        enforced in `__post_init__` because no stored IR pairs them.

        Cycles-only, and that asymmetry is deliberate rather than temporary
        neglect — there is no Filament or three.js equivalent for an arbitrary
        node graph. Every other consumer draws the scalars, which
        `render/_world._material_to_render` supplies from the module defaults —
        which is the ONLY way a shader material gets them, and is the reason
        the author need not: a shader material degrades to a
        plain surface everywhere else instead of failing, without a second
        authored description of the surface to disagree with the graph. That substitution lives at
        the packing boundary and not in `material_from_ir`, which keeps the
        absence so an IR survives a round trip unchanged.
    """

    color: "tuple[int, int, int] | None" = None
    metallic: "float | None" = None
    roughness: "float | None" = None
    alpha: "float | None" = None
    texture: "Texture | None" = None
    mapping: "Mapping | None" = None
    emission: "Emission | None" = None
    shader: "Shader | None" = None

    def __post_init__(self):
        if self.shader is not None and not isinstance(self.shader, Shader):
            raise InvalidPrimitive(
                f"material.shader must be a Shader or None, got "
                f"{type(self.shader).__name__}",
                evidence={"param": "material.shader",
                          "constraint": "type:Shader|None"},
                suggested_repairs=[
                    "build one by decorating a function with @shader",
                ],
            )

        if self.shader is None:
            # **The scalar material.** `color` is required and the other three are
            # required-with-defaults, so `to_ir()` emits all four and no stored
            # object's id moves. That equality is what
            # `test_shader_material.py` guards.
            if self.color is None:
                raise InvalidPrimitive(
                    "a material needs either a color or a shader: with no "
                    "shader to compute the surface, `color` is what every "
                    "consumer draws",
                    evidence={"param": "material.color",
                              "constraint": "required without a shader"},
                    suggested_repairs=[
                        "pass color=(r, g, b), or attach a @shader function",
                    ],
                )
            for _field, _default in (("metallic", DEFAULT_METALLIC),
                                     ("roughness", DEFAULT_ROUGHNESS),
                                     ("alpha", DEFAULT_ALPHA)):
                if getattr(self, _field) is None:
                    object.__setattr__(self, _field, _default)

        # Each scalar is checked only where it is present. Under a shader they
        # are all optional — the shader computes the surface, so asking the
        # author for a color that duplicates it invites drift nothing can see.
        # `_material_to_render` substitutes the module defaults when packing
        # for a renderer, so a backend always has a fallback to draw — while
        # the IR keeps saying only what the author said.
        if self.color is not None:
            _check_color(self.color, "material.color")
        if self.metallic is not None:
            _check_unit(self.metallic, "material.metallic")
        if self.roughness is not None:
            _check_unit(self.roughness, "material.roughness")
        if self.alpha is not None:
            _check_alpha(self.alpha, "material.alpha")
        if self.texture is not None:
            check_texture(self.texture, "material.texture")
        if self.mapping is not None and not isinstance(self.mapping, Mapping):
            raise InvalidPrimitive(
                f"material.mapping must be a Mapping or None, got "
                f"{type(self.mapping).__name__}",
                evidence={"param": "material.mapping",
                          "constraint": "type:Mapping|None"},
            )
        if self.emission is not None and not isinstance(self.emission, Emission):
            raise InvalidPrimitive(
                f"material.emission must be an Emission or None, got "
                f"{type(self.emission).__name__}",
                evidence={"param": "material.emission",
                          "constraint": "type:Emission|None"},
            )
        if self.shader is not None and self.texture is not None:
            # Both describe the same sockets and nothing arbitrates. A shader
            # is handed `nodes`/`links`/`bsdf` and no asset paths, so it cannot
            # sample a staged image anyway — allowing the pair would only make
            # "who owns Base Color" a question with no answer.
            raise InvalidPrimitive(
                "material carries both a shader and texture maps; a shader "
                "computes the surface itself and cannot sample staged images",
                evidence={"param": "material.shader",
                          "constraint": "mutually exclusive with texture"},
                suggested_repairs=[
                    "drop the texture and let the shader build the surface, "
                    "or drop the shader and bind the maps",
                ],
            )
        if (self.emission is None
                and self.texture is not None
                and EMISSION in self.texture.maps):
            # An emission MAP with no emission scalar. Refused for the same
            # reason `mapping` without `texture` is: the map has nothing to
            # modulate and no fallback for a consumer that cannot sample it, so
            # the author believes a surface glows when for half the pipeline it
            # does not.
            raise InvalidPrimitive(
                "material.texture carries an 'emission' map but the material "
                "has no emission; the map has no strength to scale and no "
                "fallback for a consumer that cannot sample it",
                evidence={"param": "material.emission",
                          "constraint": "required by texture['emission']"},
                suggested_repairs=[
                    "set emission=Emission(color=..., strength=...), or drop "
                    "the 'emission' map",
                ],
            )
        if self.mapping is not None and self.texture is None:
            # A tile size with nothing to tile. Rejected rather than ignored
            # because it means the author believes a surface is textured when
            # it is not, and every symptom of that is a missing image.
            raise InvalidPrimitive(
                "material.mapping describes how to lay out texture maps, but "
                "this material has none",
                evidence={"param": "material.mapping",
                          "constraint": "requires texture"},
                suggested_repairs=[
                    "drop the mapping, or bind the maps it was meant to size",
                ],
            )
        # Normalize color to a tuple of ints so identity / hash is stable
        # even if a list was passed. Absent scalars stay absent: under a shader
        # `None` means "the shader decides", and coercing it to a number would
        # put a value in the IR the author never wrote.
        if self.color is not None:
            object.__setattr__(
                self, "color",
                (int(self.color[0]), int(self.color[1]), int(self.color[2])),
            )
        for _field in ("metallic", "roughness", "alpha"):
            _value = getattr(self, _field)
            if _value is not None:
                object.__setattr__(self, _field, float(_value))

    def to_ir(self) -> dict:
        """Canonical JSON-safe dict form for embedding in object / scene IR.

        `texture`, `mapping`, `emission`, and `shader` are OMITTED when absent
        rather than written as null, and the same rule runs one level down: a
        map kind with no map is simply not a key. See the module docstring —
        the IR is content-hashed whole, so an unconditional key would re-id
        every object that has ever been built.

        **The four scalars follow that rule too, and the effect is conditional
        on `shader`.** A material with no shader has them filled by
        `__post_init__`, so it emits all four exactly as it always did — that is
        what keeps stored runs byte-identical. A shader material emits
        only what its author wrote, so a shader-only surface serializes to a
        single `shader` key and says nothing it does not mean.
        """
        d: dict = {}
        if self.color is not None:
            d["color"] = [self.color[0], self.color[1], self.color[2]]
        for key in ("metallic", "roughness", "alpha"):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        if self.texture is not None:
            d["texture"] = self.texture.to_ir()
        if self.mapping is not None:
            d["mapping"] = self.mapping.to_ir()
        if self.emission is not None:
            d["emission"] = self.emission.to_ir()
        if self.shader is not None:
            d["shader"] = self.shader.to_ir()
        return d


# ---------------------------------------------------------------------------
# Validators (used by Material.__post_init__ and by callers that need to
# raise their own error subclass).
# ---------------------------------------------------------------------------

def _check_color(color, label: str, *, error_cls=InvalidPrimitive, refs=()):
    if not (isinstance(color, (tuple, list)) and len(color) == 3):
        raise error_cls(
            f"{label} must be a 3-tuple (r, g, b), got {color!r}",
            refs=list(refs),
            evidence={"param": label, "value": repr(color),
                      "constraint": "(r, g, b) ints in [0, 255]"},
            suggested_repairs=["pass color as (r, g, b) ints in [0, 255]"],
        )
    for i, c in enumerate(color):
        if isinstance(c, bool) or not isinstance(c, int):
            raise error_cls(
                f"{label}[{i}] must be an int in [0, 255], got "
                f"{type(c).__name__}",
                refs=list(refs),
                evidence={"param": f"{label}[{i}]", "value": repr(c),
                          "constraint": "int in [0, 255]"},
                suggested_repairs=[
                    "use ints in [0, 255] for each channel "
                    "(e.g. (200, 100, 50))"
                ],
            )
        if not (_COLOR_MIN <= c <= _COLOR_MAX):
            raise error_cls(
                f"{label}[{i}] must be in [0, 255], got {c}",
                refs=list(refs),
                evidence={"param": f"{label}[{i}]", "value": c,
                          "constraint": "int in [0, 255]"},
                suggested_repairs=["clamp each channel to [0, 255]"],
            )


def _check_unit(x, label: str, *, error_cls=InvalidPrimitive, refs=()):
    """Validate a unit scalar in [0, 1]."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise error_cls(
            f"{label} must be a number, got {type(x).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": repr(x),
                      "constraint": "type:number"},
        )
    if not math.isfinite(float(x)):
        raise error_cls(
            f"{label} must be finite, got {x}",
            refs=list(refs),
            evidence={"param": label, "value": x, "constraint": "finite"},
        )
    if not (0.0 <= float(x) <= 1.0):
        raise error_cls(
            f"{label} must be in [0, 1], got {x}",
            refs=list(refs),
            evidence={"param": label, "value": x,
                      "constraint": "float in [0, 1]"},
            suggested_repairs=[f"clamp {label} to [0, 1]"],
        )


def _check_alpha(x, label: str, *, error_cls=InvalidPrimitive, refs=()):
    """Alpha excludes 0 (fully invisible parts are rejected — they would
    be a footgun and almost always indicate a bug at the call site)."""
    _check_unit(x, label, error_cls=error_cls, refs=refs)
    if float(x) <= 0.0:
        raise error_cls(
            f"{label} must be > 0, got {x}",
            refs=list(refs),
            evidence={"param": label, "value": x,
                      "constraint": "float in (0, 1]"},
            suggested_repairs=[
                "alpha=1.0 for opaque, alpha<1 for transparent; do not use 0",
            ],
        )


def _check_asset_path(path, label: str, *,
                      error_cls=InvalidPrimitive, refs=()):
    """A staged asset path: run-relative, no escapes, under ``assets/``.

    The same rule `scene_context.referenced_object_assets` enforces when it
    decides what travels into a catalog, applied here so a bad path fails at
    author time — where the traceback names the call that made it — rather
    than at staging time, where it names a dict.
    """
    if not isinstance(path, str) or not path:
        raise error_cls(
            f"{label} must be a non-empty string, got {path!r}",
            refs=list(refs),
            evidence={"param": label, "value": repr(path),
                      "constraint": "run-relative path under assets/"},
        )
    parts = tuple(x for x in path.replace("\\", "/").split("/") if x)
    if path.startswith("/") or ".." in parts or not parts \
            or parts[0] != "assets":
        raise error_cls(
            f"{label} must be a run-relative path under assets/, got {path!r}",
            refs=list(refs),
            evidence={"param": label, "value": path,
                      "constraint": "starts with assets/, no .., not absolute"},
            suggested_repairs=[
                "Use importer-staged Texture metadata for image maps. Author "
                "new surfaces with Material(color=...) or Material(shader=...).",
            ],
        )


def _check_digest(digest, label: str, *, error_cls=InvalidPrimitive, refs=()):
    """A blake2b-128 hex digest — 32 hex chars, matching `_mesh_assets`."""
    ok = (isinstance(digest, str) and len(digest) == 32
          and all(c in "0123456789abcdef" for c in digest))
    if not ok:
        raise error_cls(
            f"{label} must be 32 lowercase hex chars, got {digest!r}",
            refs=list(refs),
            evidence={"param": label, "value": repr(digest),
                      "constraint": "blake2b-128 hex"},
        )


def _check_scale(x, label: str, *, error_cls=InvalidPrimitive, refs=()):
    """Tile size in meters: finite and strictly positive.

    No upper bound. A tile larger than the part is legal and simply means one
    region of the image covers the whole surface; that is a composition
    choice, not an error.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise error_cls(
            f"{label} must be a number, got {type(x).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": repr(x),
                      "constraint": "type:number"},
        )
    if not math.isfinite(float(x)) or float(x) <= 0.0:
        raise error_cls(
            f"{label} must be finite and > 0, got {x}",
            refs=list(refs),
            evidence={"param": label, "value": x,
                      "constraint": "float > 0"},
            suggested_repairs=[
                "scale_m is the real-world edge of one tile in meters "
                "(0.2-0.3 reads as planks on furniture)",
            ],
        )


def _check_emission_strength(x, label: str, *, error_cls=InvalidPrimitive, refs=()):
    """Radiant exitance in W/m². Finite and strictly positive.

    Zero is refused deliberately: `emission=None` already means "not a light
    source", and allowing a second spelling of it would put an `if strength >
    0` in every consumer — three of which are renderers that would then each
    decide independently whether a zero-strength emitter gets a shader node.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise error_cls(f"{label} must be a number, got {x!r}",
                        evidence={"param": label, "constraint": "float>0"},
                        refs=list(refs)) from None
    if not math.isfinite(v) or v <= 0.0:
        raise error_cls(
            f"{label} must be finite and > 0, got {v!r}",
            evidence={"param": label, "constraint": "float>0", "value": v},
            refs=list(refs),
            suggested_repairs=["drop `emission` entirely for a surface that "
                               "does not emit"],
        )
    return v


def _check_strength(x, label: str, *, error_cls=InvalidPrimitive, refs=()):
    """A map's own scalar weight: a finite number, with no range.

    No bound, because the bound belongs to the kind rather than to the field —
    glTF's `normalScale` is routinely > 1 while an occlusion strength is
    [0, 1]. Non-finite is rejected here rather than tolerated downstream:
    `float("nan")` survives a plain `float()` coercion, and
    `json.dumps` then writes the bare token `NaN`, which is not JSON. The IR
    would stop round-tripping through any strict reader, and the content hash
    would cover a value that cannot come back.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise error_cls(
            f"{label} must be a number, got {type(x).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": repr(x),
                      "constraint": "type:number"},
        )
    if not math.isfinite(float(x)):
        raise error_cls(
            f"{label} must be finite, got {x}",
            refs=list(refs),
            evidence={"param": label, "value": repr(x),
                      "constraint": "finite"},
        )


def check_texture(texture, label: str, *,
                  error_cls=InvalidPrimitive, refs=()) -> "Texture":
    """Validate that `texture` is a Texture instance."""
    if not isinstance(texture, Texture):
        raise error_cls(
            f"{label} must be a Texture instance, got "
            f"{type(texture).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": type(texture).__name__,
                      "constraint": "type:Texture"},
            suggested_repairs=[
                "Use importer-staged Texture metadata for image maps. Author "
                "new surfaces with Material(color=...) or Material(shader=...).",
            ],
        )
    return texture


#: Material fields a `shader` OWNS, and which an authored material therefore
#: may not also set. The rule is taught everywhere; the refusal that enforces
#: it at the authoring boundary is deferred -- see `check_material`. `alpha` and `emission` are deliberately not here: the
#: harness applies both outside the graph (`_apply_emission` in the render
#: child, and alpha through the transparency path), so they are the surface's
#: two properties a shader genuinely cannot express.
SHADER_OWNED_FIELDS = ("color", "metallic", "roughness")


def check_material(material, label: str, *,
                   error_cls=InvalidPrimitive, refs=()) -> "Material":
    """Validate a material at the **authoring** boundary.

    Two rules: it is a `Material`, and if it carries a `shader` it does not
    also carry the scalars that shader owns.

    **Why the exclusivity check lives here and not in `__post_init__`.** This
    is the authoring boundary and only the authoring boundary — its five
    callers are `Object.add_part`, `Object.make_box` and `Scene.set_room`'s
    three room-surface arguments, which is exactly the set of places an agent
    describes a *new* surface. `material_from_ir` does not come through here,
    so a stored IR that carries both representations still reads, still
    round-trips byte-identically, and still keeps its content hash — which
    `Material.shader`'s docstring promises and `test_shader_material.py`
    guards. Putting the rule in `__post_init__` would make reading such a
    stored IR an error.

    Builders that need their own error subclass (e.g. `InvalidRoom` from
    `Scene.set_room`) pass `error_cls=`; the message and evidence remain
    consistent with `Material.__post_init__`.
    """
    if not isinstance(material, Material):
        # **The likeliest slip is handing over the shader itself.** `@shader`
        # binds its name to a `Shader`, so `material=marble` reads perfectly
        # and is one wrapper short — and the generic advice below would send
        # the author to a flat scalar instead of telling them what to type.
        if isinstance(material, Shader):
            repairs = [
                f"wrap it: material=Material(shader={material.name})",
            ]
        else:
            repairs = [
                "pass a Material: Material(shader=<a @shader function>) for an "
                "authored surface, adding alpha= or emission= if the surface "
                "needs them",
            ]
        raise error_cls(
            f"{label} must be a Material instance, got "
            f"{type(material).__name__}",
            refs=list(refs),
            evidence={"param": label, "value": type(material).__name__,
                      "constraint": "type:Material"},
            suggested_repairs=repairs,
        )
    # **The exclusivity refusal is written and deliberately not armed yet.**
    #
    # `SHADER_OWNED_FIELDS` states the rule and every agent-facing surface
    # teaches the shader-only form. Refusing the pair is retroactive: every
    # `Material(shader=...)` in the recorded authored sources carries at least
    # one of the scalars, so arming this turns each into an `InvalidPrimitive`
    # at `add_part` / `set_room` and a replay of any of those sources into a
    # build failure.
    #
    # Arm it when either is true, and delete this block's comment with it:
    #   * the authored corpus has turned over under the new prompts, which the
    #     same measurement re-run will show; or
    #   * `Material` grows an explicit non-PBR `swatch=` field, so an author
    #     can still hand a flat color to the renderers that cannot run the
    #     graph without describing the same sockets twice.
    #
    # `tests/test_shader_material.py::test_authoring_a_shader_with_the_scalars_it_owns_is_refused`
    # is a strict xfail and will report the moment this is uncommented.
    return material


#: Keys of a texture's IR dict that are NOT maps. Everything else in that dict
#: is a map kind, which is what lets asset staging, catalog rebasing and the
#: URDF exporter carry a new kind without knowing its name.
def texture_maps_from_ir(texture: dict | None) -> dict:
    """``{kind: {"image", "digest", ...}}`` for every map in a texture's IR.

    THE definition of "what is a map", shared by every layer that moves one:
    `scene_context.referenced_object_assets`, `viewer.scene._rebase_material`,
    `render.urdf_export`, and the renderers. Adding a kind is a data change
    precisely because none of them enumerate kinds — they ask here.

    A texture IS its maps, so this filters for well-formedness rather than
    distinguishing maps from scalar siblings. It still filters, because IR
    reaching a renderer is not always IR this code wrote — an importer may
    merge report fields into the same block, and a value that is not a dict
    carrying an `image` is not a map whatever else it is.
    """
    if not isinstance(texture, dict):
        return {}
    return {k: v for k, v in texture.items()
            if isinstance(v, dict) and v.get("image")}


#: What a malformed texture / mapping block raises on its way through
#: `TextureMap` or `Mapping`: a missing key, a value of the wrong type, an
#: unparseable number, or this module's own validation. Caught so a material
#: nobody can parse degrades to its scalars — see `material_from_ir`.
#:
#: Deliberately NOT `BaseException`. These helpers run inside the renderers, so
#: a `KeyboardInterrupt` or `SystemExit` arriving mid-render has to reach the
#: caller rather than be reported as "this map did not parse"; neither is
#: something a texture block can cause, and swallowing them turns Ctrl-C during
#: a long render into a silently mis-drawn surface.
_MALFORMED_IR = (InvalidPrimitive, KeyError, TypeError, ValueError)


def _map_from_ir(d: dict) -> "TextureMap | None":
    """One map, or None if the block is not a well-formed one."""
    try:
        return TextureMap(image=d["image"], digest=d["digest"],
                          channel=d.get("channel"),
                          strength=d.get("strength"))
    except _MALFORMED_IR:
        return None


def _shader_from_ir(d) -> "Shader | None":
    """A `Shader` from its IR block, or None if it will not parse.

    Degrades rather than raising, for the same reason `_map_from_ir` does: this
    runs inside the renderers, and a material nobody can read must leave the
    part on its scalars rather than take the whole render down.
    """
    if not isinstance(d, dict):
        return None
    try:
        return Shader(name=d["name"], body=d["body"],
                      params=tuple(d.get("params") or ()))
    except _MALFORMED_IR:
        return None


def _emission_from_ir(d) -> "Emission | None":
    """One emission block, or None if it is absent or malformed.

    Degrades rather than raising, matching how a bad texture block is handled:
    this runs inside the renderers, where a material nobody can parse must not
    take the whole picture down. The surface then draws unlit, which is the
    same thing every consumer that cannot sample emission already does.
    """
    if not isinstance(d, dict):
        return None
    try:
        return Emission(color=tuple(d["color"]), strength=d["strength"])
    except _MALFORMED_IR:
        return None


def material_from_ir(d: dict | None,
                     *,
                     default_color: tuple[int, int, int]) -> Material:
    """Reconstitute a Material from its IR dict form.

    Missing PBR fields fall back to defaults — this keeps the renderer
    forward-compatible if older IR happens to omit them.

    **A shader material's absent scalars stay absent, so this is idempotent:**
    `material_from_ir(d).to_ir() == d`. A material with no shader still has its
    four scalars filled from the module defaults, which is what keeps older IR
    readable and every stored object's id stable. The scalar approximation a
    non-Cycles consumer needs comes from the packing boundary
    (`render/_world._material_to_render`), not from here.

    **A malformed texture block degrades to the scalars; it does not raise.**
    Two reasons, and both are about who calls this. It runs inside the
    renderers, where a material nobody can parse must not take the whole render
    down — a part drawn on its fallback color is a bad picture, an exception
    is no picture. And an importer may merge its own report fields into the
    same ``material`` dict (a note saying which atlas it declined to carry, and
    why), so a sibling key that is not a map is expected traffic rather than
    corruption. A map that will not parse is dropped; a texture left with no
    maps at all becomes no texture.
    """
    if d is None:
        return Material(color=default_color)
    # Used by the scalar branch below. A shader material reads `d` directly, so
    # that `default_color` never silently becomes an authored value.
    color = d.get("color", default_color)

    maps = tuple(sorted(
        (k, m) for k, m in
        ((k, _map_from_ir(v)) for k, v in texture_maps_from_ir(d.get("texture")).items())
        if m is not None))
    texture = Texture(map_items=maps) if maps else None

    # `mapping` only means something beside maps, and `Material` refuses the
    # pair the other way round — so drop it rather than raising when the
    # texture it sized did not survive.
    mapping = None
    raw_mapping = d.get("mapping")
    if texture is not None and isinstance(raw_mapping, dict):
        try:
            mapping = Mapping(scale_m=float(raw_mapping["scale_m"]))
        except _MALFORMED_IR:
            mapping = None

    # An emission map whose scalar did not survive would trip `Material`'s own
    # check, so the pair is dropped together — same posture as `mapping`.
    emission = _emission_from_ir(d.get("emission"))
    if emission is None and texture is not None and EMISSION in texture.maps:
        # Build the remainder BEFORE constructing, because `Texture` refuses an
        # empty map set — so dropping the only map has to become `None`, not a
        # Texture with nothing in it. (And `mapping` goes with it: it sizes maps
        # that no longer exist, which `Material` also refuses.)
        rest = tuple((k, m) for k, m in texture.map_items if k != EMISSION)
        texture = Texture(map_items=rest) if rest else None
        if texture is None:
            mapping = None

    shader = _shader_from_ir(d.get("shader"))
    if shader is not None and texture is not None:
        # `Material` refuses the pair, and this function must not raise. The
        # shader owns the sockets a texture would have driven, so the maps are
        # what goes; `mapping` sizes maps that no longer exist, so it follows.
        texture = None
        mapping = None

    if shader is not None:
        # **A shader material's absent scalars stay absent, so this function is
        # idempotent.** Filling them here would make `material_from_ir(d).to_ir()
        # != d`: the canonical bytes changed just by being parsed and written back,
        # which is a trap for anything that round-trips an IR — a rebase, a
        # migration, a regression oracle — and it contradicts the one thing
        # `shader` is supposed to mean, that the graph decides the surface.
        #
        # Nothing loses its fallback by this. The scalar approximation every
        # non-Cycles consumer draws is supplied at the PACKING boundary
        # (`render/_world._material_to_render`, which emits all four
        # unconditionally), which is where a renderer-specific default belongs —
        # not in the reader that defines what the IR says.
        return Material(
            color=tuple(color) if "color" in d else None,
            metallic=_opt_float(d, "metallic"),
            roughness=_opt_float(d, "roughness"),
            alpha=_opt_float(d, "alpha"),
            texture=texture,
            mapping=mapping,
            emission=emission,
            shader=shader,
        )

    # Legacy scalar material: filled from the module defaults exactly as it
    # always was, so older IR that omits a field still reads and every stored
    # object keeps its id.
    return Material(
        color=tuple(color),
        metallic=float(d.get("metallic", DEFAULT_METALLIC)),
        roughness=float(d.get("roughness", DEFAULT_ROUGHNESS)),
        alpha=float(d.get("alpha", DEFAULT_ALPHA)),
        texture=texture,
        mapping=mapping,
        emission=emission,
    )


def _opt_float(d: dict, key: str):
    """`float(d[key])` when the key is there, else None.

    Absence has to survive the read, so a missing key cannot become a default;
    a value that will not convert is dropped rather than raised on, matching
    this module's rule that malformed IR degrades inside a renderer.
    """
    if key not in d:
        return None
    try:
        return float(d[key])
    except (TypeError, ValueError):
        return None


__all__ = [
    "Material",
    "SHADER_UV_SCALE_M",
    "Emission",
    "Shader",
    "EMISSION",
    "Texture",
    "TextureMap",
    "Mapping",
    "texture_maps_from_ir",
    "check_material",
    "check_texture",
    "material_from_ir",
    "DEFAULT_PART_COLOR",
    "DEFAULT_FLOOR_COLOR",
    "DEFAULT_WALL_COLOR",
    "DEFAULT_METALLIC",
    "DEFAULT_ROUGHNESS",
    "DEFAULT_ALPHA",
]
