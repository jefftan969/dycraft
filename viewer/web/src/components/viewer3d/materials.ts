// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import * as THREE from 'three';
import type { UrdfVisual } from './urdf-parser';

export interface MaterialSpec {
  name?: string;
  color: THREE.Color;
  opacity: number;
  metalness: number;
  roughness: number;
  transmission: number;
  thickness: number;
  ior: number;
  clearcoat: number;
  clearcoatRoughness: number;
  envMapIntensity: number;
  /**
   * Run-relative path to a base-color texture, from the URDF's
   * `<texture filename>`. Resolved against the same base URL as
   * `<mesh filename>` — the run's `files/` route serves both.
   */
  texturePath?: string;
  /** Real-world edge length of one tile, in meters. */
  textureScaleM?: number;
  /**
   * Extra map paths by kind, sampled through the SAME UVs as the base color.
   * Carried opaquely from the URDF: a kind this file has no binding for is
   * simply not applied, and the material falls back to its scalars — which is
   * the right degradation, and why nothing between the library and here
   * enumerates kinds.
   */
  textureMaps?: Record<string, string>;
}

/**
 * Ask the canvas for a redraw, once a texture has finished decoding.
 *
 * **Without this the whole texture feature is invisible.** The viewer does not
 * run a continuous animation loop — it draws on demand, when something calls
 * `invalidate()`. Loading is asynchronous, so a texture that decodes after the
 * last draw is never composited, and the part keeps whatever it looked like
 * while the map was still empty. Because a textured material binds a WHITE
 * factor (so the image is not tinted by the authored color), an empty map does
 * not render as the fallback color — it renders **black**, until the viewer
 * happens to draw again because the user moved the camera.
 *
 * A module-level hook rather than a parameter because the loader is reached
 * through `createMaterial`, several layers below whoever owns the canvas, and
 * threading a callback through every one of them would put a rendering concern
 * into the signature of code that has no other reason to know about it.
 */
let requestRedraw: (() => void) | null = null;

export function setTextureRedrawHook(fn: (() => void) | null): void {
  requestRedraw = fn;
}

/**
 * Decoded textures, keyed by resolved URL.
 *
 * Mirrors `geometryTemplateCache` in `geometry-loader.ts`, and matters for the
 * same reason: one library material is bound to many parts, and each is a
 * separate `<visual>` in the URDF. The Python side caches by content digest for
 * the identical reason (`open3d_renderer.py::_TEXTURE_CACHE`).
 */
const textureCache = new Map<string, THREE.Texture>();

/**
 * Per-URL listeners told when a load fails, so every material that bound the
 * texture can fall back to its own color.
 *
 * A list rather than a single callback because the cache hands one Texture to
 * many materials — a library material bound to six parts is six `<visual>`
 * elements, and all six have to un-bind or five of them render a blank map.
 */
const textureErrorListeners = new Map<string, Set<() => void>>();

/** Join a base URL and a run-relative asset path, the way the loader does. */
export function resolveTextureUrl(baseUrl: string, path: string): string {
  return baseUrl.endsWith('/') ? `${baseUrl}${path}` : `${baseUrl}/${path}`;
}

/**
 * Load a base-color texture, calling `onError` if the fetch fails.
 *
 * Loading is asynchronous and three hands back a Texture immediately, so a
 * failure cannot be reported by returning null — an unfetched map renders as
 * blank white, which is *worse* than no texture at all because it discards the
 * color too. `onError` is how the caller un-binds and restores the fallback,
 * which is the whole reason an authored color still accompanies a texture, and
 * the same posture `open3d_renderer.py::_usable_texture` takes.
 *
 * The cache entry is dropped on failure, so a later mount retries rather than
 * inheriting a permanently blank texture from a transient 404.
 */
export function loadBaseColorTexture(url: string, onError?: () => void): THREE.Texture {
  return _loadTexture(url, THREE.SRGBColorSpace, onError);
}

/**
 * Load a DATA texture — roughness and anything else that is not a color.
 *
 * The color space is the whole reason this is a separate entry point.
 * Roughness is a measured scalar, not a color, so it must be sampled linearly;
 * decoding it as sRGB would apply a gamma curve to a number that never had one
 * and skew every value toward the glossy end.
 */
export function loadDataTexture(url: string, onError?: () => void): THREE.Texture {
  return _loadTexture(url, THREE.NoColorSpace, onError);
}

function _loadTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
  onError?: () => void,
): THREE.Texture {
  if (onError) {
    const listeners = textureErrorListeners.get(url) ?? new Set();
    listeners.add(onError);
    textureErrorListeners.set(url, listeners);
  }
  const cached = textureCache.get(url);
  if (cached) {
    return cached;
  }
  const texture = new THREE.TextureLoader().load(
    url,
    // Both paths request a redraw, and both are needed. On success the decoded
    // image has to reach a frame; on failure the listeners have just swapped
    // the material back to its flat color, and that has to reach one too — a
    // fallback nobody draws looks exactly like the failure it was correcting.
    () => requestRedraw?.(),
    undefined,
    () => {
      textureCache.delete(url);
      const listeners = textureErrorListeners.get(url);
      textureErrorListeners.delete(url);
      listeners?.forEach((listener) => listener());
      requestRedraw?.();
    },
  );
  // sRGB for base color, linear for data. A color PNG holds gamma-encoded
  // values three would otherwise sample as linear (washed out); a roughness PNG
  // holds measured scalars that never had a curve, and applying one skews the
  // whole surface glossy.
  texture.colorSpace = colorSpace;
  // Triplanar UVs are world coordinates over a tile size, so they run well
  // outside 0..1 — clamping would smear the edge pixel across a whole table.
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 8;
  textureCache.set(url, texture);
  return texture;
}

/**
 * The `three` namespace this module builds materials from.
 *
 * Exported only so `scripts/check-triplanar-uv.mjs` can stub `TextureLoader`
 * while asserting what `createMaterial` does with a Texture — node has no DOM,
 * so a real load cannot run there. Nothing in the app should import this.
 */
export const THREE_FOR_TESTS = THREE;

/**
 * What an absent `<pbr>` block stands for.
 *
 * `render/urdf_export.py` emits the block only when `metallic > 0` or
 * `roughness != 0.9`, so that an untextured part's URDF stays byte-identical
 * to what the pre-PBR exporter produced. Its absence therefore carries
 * information — these exact two numbers, the `sdk._material` defaults — rather
 * than meaning "unknown".
 *
 * Sound only because every URDF this viewer sees is generated: `viewer/app.py`
 * serves `model.urdf` out of `runs.run_urdf` -> `render.urdf_export`, and even
 * an imported foreign asset reaches the browser re-exported through it. If the
 * viewer ever loads a URDF it did not write, this assumption is the first
 * thing that breaks, and the symptom is every surface going matte.
 */
export const EXPORTED_DEFAULT_METALNESS = 0.0;
export const EXPORTED_DEFAULT_ROUGHNESS = 0.9;

/**
 * Distinct depth biases handed out before the sequence repeats, and the step
 * between them. Both are deliberately small — see `depthBiasForOrdinal`.
 */
const DEPTH_BIAS_CYCLE = 8;
const DEPTH_BIAS_STEP = 0.25;

/**
 * A deterministic tie-break for coplanar surfaces, in depth-buffer units.
 *
 * The bias goes to `polygonOffsetUnits` (never `polygonOffsetFactor` — see
 * `applyDepthBias`), so its value is a count of *resolvable depth steps*:
 * enough to give two surfaces at the same depth a stable order, no more.
 *
 * **It cycles, and that is the point.** A bias that grew with a running count
 * of every visual would depend on how big the model is, and a depth-buffer unit
 * is not a fixed distance: with the near plane a scene camera ends up with, a
 * few dozen of them are about ten centimeters of world depth, enough to
 * separate surfaces the scene IR asserts are coplanar and draw them in the
 * wrong order. Tie-breaking needs a handful of distinct values, not one per
 * part, so the sequence repeats every `DEPTH_BIAS_CYCLE` visuals and can never
 * grow with the model.
 *
 * Surfaces `DEPTH_BIAS_CYCLE` apart in the ordinal share a bias. They are in
 * different parts (the ordinal runs in link order, so one part's own surfaces
 * are adjacent and always differ), which is the case this was never solving.
 */
export function depthBiasForOrdinal(ordinal: number): number {
  return 1 + (ordinal % DEPTH_BIAS_CYCLE) * DEPTH_BIAS_STEP;
}

/**
 * Install a depth bias on a material — the one place that decides how a
 * `depthBiasForOrdinal` value becomes GL polygon-offset state.
 *
 * Two call sites need it: `createMaterial`, for a material built from the URDF
 * spec, and `applyLoadedMeshPresentation`, for a derived GLB that keeps the
 * material the bake wrote. One function, so a run with baked materials gets the
 * same bias from the same ordinal.
 *
 * `polygonOffsetFactor` stays 0 on purpose. GL's offset is
 * `factor * m + units * r`, where `m` is the polygon's maximum depth slope in
 * window space; `m` is a property of the *view*, so any non-zero factor makes
 * the bias slide while the camera moves, by an amount that grows without bound
 * as a surface turns edge-on. `units * r` is whole resolvable depth steps and
 * does not move.
 */
export function applyDepthBias(
  material: THREE.Material,
  depthBias: number | undefined,
): void {
  if (depthBias === undefined) {
    return;
  }
  material.polygonOffset = depthBias !== 0;
  material.polygonOffsetFactor = 0;
  material.polygonOffsetUnits = depthBias;
}

function srgbColor(r: number, g: number, b: number): THREE.Color {
  return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}

function softenBrightDiffuse(spec: MaterialSpec): void {
  const maxChannel = Math.max(spec.color.r, spec.color.g, spec.color.b);
  const minChannel = Math.min(spec.color.r, spec.color.g, spec.color.b);
  const isBrightNeutral = maxChannel > 0.78 && maxChannel - minChannel < 0.08;

  if (!isBrightNeutral || spec.metalness > 0.2 || spec.transmission > 0.01) {
    return;
  }

  spec.color.lerp(new THREE.Color(0.82, 0.82, 0.78), 0.38);
  spec.roughness = Math.max(spec.roughness, 0.62);
  spec.clearcoat = Math.min(spec.clearcoat, 0.08);
  spec.envMapIntensity = Math.min(spec.envMapIntensity, 0.55);
}

/**
 * Resolve visual material specification from URDF data
 * Defaults to a neutral gray if no material specified
 */
export function resolveVisualMaterialSpec(visual: UrdfVisual): MaterialSpec {
  const defaultSpec: MaterialSpec = {
    name: undefined,
    color: srgbColor(0.74, 0.74, 0.72),
    opacity: 1.0,
    metalness: 0.05,
    roughness: 0.48,
    transmission: 0,
    thickness: 0,
    ior: 1.45,
    clearcoat: 0,
    clearcoatRoughness: 0.2,
    envMapIntensity: 0.8,
  };

  if (!visual.material) {
    return defaultSpec;
  }

  const spec = { ...defaultSpec };
  spec.name = visual.material.name;

  if (visual.material.color) {
    const [r, g, b, a] = visual.material.color.rgba;
    spec.color = srgbColor(r, g, b);
    spec.opacity = a;
  }

  // **The authored scalars, and they outrank every guess below.** Most
  // untextured parts carry an authored roughness and many a metalness; with
  // `defaultSpec`'s values instead, a nylon strap at 0.92 and a crystal drop at
  // 0.05 would draw identically here while Cycles and Filament draw them at
  // opposite ends.
  //
  // Per field, not per block: under a `<shader>` marker the exporter emits
  // only the tags whose values the author actually wrote (see
  // `urdf-parser.ts`), so one of the two can be absent from a block that
  // exists. The fallback is the same suppressed-default pair either way.
  const pbr = visual.material.pbr;
  spec.metalness = pbr?.metalness ?? EXPORTED_DEFAULT_METALNESS;
  spec.roughness = pbr?.roughness ?? EXPORTED_DEFAULT_ROUGHNESS;

  if (visual.material.texture?.filename) {
    // A photographed surface, so the heuristics below are skipped entirely —
    // they exist to GUESS what the URDF cannot carry, and guessing over a
    // measured surface is strictly worse than not guessing.
    spec.texturePath = visual.material.texture.filename;
    spec.textureScaleM = visual.material.texture.scaleM;
    spec.textureMaps = visual.material.texture.maps;
    return spec;
  }

  return applyUncarriedPreset(spec);
}

/**
 * The name heuristics, reduced to the fields the URDF has no place for.
 *
 * `applyNamedMaterialPreset` and `softenBrightDiffuse` set `metalness` and
 * `roughness`, and `softenBrightDiffuse` also lerps `color`. All of these
 * arrive from the exporter (`<pbr>`, `<color rgba>`), so whatever the guessers did to them is discarded
 * and only `transmission` / `thickness` / `ior` / `clearcoat` /
 * `envMapIntensity` survive: refraction is genuinely not in the schema, and a
 * `<pbr>`-less consumer has no way to express it.
 *
 * Written as a restore-after rather than by pruning each branch so that a
 * preset added later cannot reintroduce the override by accident, and so the
 * set of carried fields is stated in exactly one place. The guessers do run;
 * their verdict on a carried field is simply not consulted.
 *
 * Both rarely match, because part names are `shell`, `visor`, `cradle`, not
 * material words. They are kept for
 * the glass case, where the guess is the only source of refraction.
 */
function applyUncarriedPreset(spec: MaterialSpec): MaterialSpec {
  const tuned = applyNamedMaterialPreset(spec);
  softenBrightDiffuse(tuned);
  tuned.color = spec.color;
  tuned.opacity = spec.opacity;
  tuned.metalness = spec.metalness;
  tuned.roughness = spec.roughness;
  return tuned;
}

function applyNamedMaterialPreset(spec: MaterialSpec): MaterialSpec {
  const name = spec.name?.toLowerCase() ?? '';
  const tuned = { ...spec };
  const tokens = name.split(/[^a-z0-9]+/).filter(Boolean);
  const hasToken = (token: string): boolean => tokens.includes(token);
  const hasAnyToken = (...candidates: string[]): boolean => candidates.some(hasToken);
  const mentionsTransparentDescriptor = hasAnyToken(
    'clear',
    'glass',
    'transparent',
    'translucent',
    'smoked',
    'frosted',
  );
  const mentionsTransparentPlasticFamily = hasAnyToken(
    'polycarbonate',
    'acrylic',
    'plexi',
    'plexiglass',
    'lexan',
  );
  const isClearRigidMaterial = (
    mentionsTransparentDescriptor
    || (mentionsTransparentPlasticFamily && tuned.opacity < 0.999)
  );

  if (isClearRigidMaterial) {
    tuned.metalness = 0.0;
    tuned.roughness = 0.08;
    tuned.transmission = Math.max(0.45, 1.0 - tuned.opacity * 0.55);
    tuned.thickness = 0.006;
    tuned.ior = 1.45;
    tuned.clearcoat = 0.9;
    tuned.clearcoatRoughness = 0.06;
    tuned.envMapIntensity = 1.25;
    return tuned;
  }

  if (name.includes('reflector')) {
    tuned.metalness = 0.9;
    tuned.roughness = 0.18;
    tuned.opacity = Math.max(tuned.opacity, 0.78);
    tuned.clearcoat = 0.18;
    tuned.clearcoatRoughness = 0.12;
    tuned.envMapIntensity = 1.4;
    return tuned;
  }

  if (name.includes('brass') || name.includes('metal')) {
    tuned.metalness = 0.88;
    tuned.roughness = 0.22;
    tuned.clearcoat = 0.16;
    tuned.clearcoatRoughness = 0.18;
    tuned.envMapIntensity = 1.3;
    return tuned;
  }

  if (name.includes('bakelite')) {
    tuned.metalness = 0.02;
    tuned.roughness = 0.38;
    tuned.clearcoat = 0.24;
    tuned.clearcoatRoughness = 0.2;
    tuned.envMapIntensity = 0.7;
    return tuned;
  }

  if (name.includes('felt')) {
    tuned.metalness = 0.0;
    tuned.roughness = 0.92;
    tuned.clearcoat = 0.0;
    tuned.envMapIntensity = 0.25;
  }

  return tuned;
}

/**
 * Create a THREE.js material from a material spec
 */
export function createMaterial(spec: MaterialSpec, options?: {
  wireframe?: boolean;
  transparent?: boolean;
  side?: THREE.Side;
  depthBias?: number;
  /**
   * Base URL a `spec.texturePath` is resolved against — the run's `files/`
   * route, the same base `<mesh filename>` uses. Resolution lives here rather
   * than on the spec because this is where the caller knows the base; a spec
   * stays a description of what the URDF said.
   */
  textureBaseUrl?: string;
}): THREE.MeshPhysicalMaterial {
  const transparent = options?.transparent ?? (spec.opacity < 0.999 || spec.transmission > 0.01);
  const side = options?.side ?? THREE.FrontSide;
  const depthBias = options?.depthBias ?? 0;

  // Built before the material exists so the error path can reach back into it;
  // `loadBaseColorTexture` cannot report failure by returning null, because
  // three resolves the fetch after handing the Texture back.
  let material: THREE.MeshPhysicalMaterial;
  const map = spec.texturePath && options?.textureBaseUrl
    ? loadBaseColorTexture(
        resolveTextureUrl(options.textureBaseUrl, spec.texturePath),
        () => {
          material.map = null;
          material.color.copy(spec.color);
          material.needsUpdate = true;
        },
      )
    : null;

  // Same story as the albedo: three's `roughnessFactor *= texelRoughness.g`
  // MULTIPLIES the map by the scalar, so binding the authored value alongside
  // it would apply the surface's own roughness twice. The scalar goes to 1.0
  // and stays the fallback for anything that cannot sample the map — which is
  // what a library entry's `base_roughness` exists to supply.
  //
  // `.g` is why the library ships a GRAYSCALE png: R == G == B, so one file
  // satisfies three's green channel and Open3D's single channel at once.
  const roughPath = spec.textureMaps?.roughness;
  const roughnessMap = roughPath && options?.textureBaseUrl
    ? loadDataTexture(
        resolveTextureUrl(options.textureBaseUrl, roughPath),
        () => {
          material.roughnessMap = null;
          material.roughness = spec.roughness;
          material.needsUpdate = true;
        },
      )
    : null;

  material = new THREE.MeshPhysicalMaterial({
    map,
    roughnessMap,
    // WHITE under a texture, and this is not cosmetic. three multiplies `map`
    // by `color` exactly as Filament multiplies `albedo_img` by `base_color`,
    // so leaving the authored brown here would darken the wood by its own
    // color, to about half of the authored image's brightness. `render/open3d_renderer.py::_pbr_material` binds white for the
    // same reason, and the agent prompt promises `color` is a fallback rather
    // than a tint. Both renderers have to keep that true or the promise is a
    // lie in whichever one drops it.
    color: map ? new THREE.Color(1, 1, 1) : spec.color,
    opacity: spec.opacity,
    metalness: spec.metalness,
    roughness: roughnessMap ? 1.0 : spec.roughness,
    transmission: spec.transmission,
    thickness: spec.thickness,
    ior: spec.ior,
    clearcoat: spec.clearcoat,
    clearcoatRoughness: spec.clearcoatRoughness,
    envMapIntensity: spec.envMapIntensity,
    transparent,
    depthWrite: !transparent,
    forceSinglePass: transparent,
    wireframe: options?.wireframe ?? false,
    side,
  });
  applyDepthBias(material, depthBias);
  return material;
}

/**
 * Create edge lines for a geometry
 */
export function createEdgeLines(geometry: THREE.BufferGeometry, color: number = 0x000000): THREE.LineSegments {
  const edges = new THREE.EdgesGeometry(geometry, 15); // 15 degree threshold
  const lineMaterial = new THREE.LineBasicMaterial({ color, opacity: 0.3, transparent: true });
  return new THREE.LineSegments(edges, lineMaterial);
}
