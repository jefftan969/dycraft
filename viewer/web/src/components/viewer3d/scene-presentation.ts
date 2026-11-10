import * as THREE from 'three';

const SEGMENTATION_PALETTE = [
  '#ff5a36',
  '#00b3ff',
  '#ffd400',
  '#16c47f',
  '#ff2f92',
  '#7c5cff',
  '#ff8a00',
  '#00c2a8',
  '#ff6b6b',
  '#145af2',
  '#c4ff0e',
  '#ff3d77',
] as const;
const SEGMENTATION_SNAPSHOT_KEY = '__articraftSegmentColorSnapshot__';
const STUDIO_SHADING_SNAPSHOT_KEY = '__articraftStudioShadingSnapshot__';

export type MaterialWithColor = THREE.Material & {
  color?: THREE.Color;
  emissive?: THREE.Color;
  map?: THREE.Texture | null;
  emissiveMap?: THREE.Texture | null;
  metalness?: number;
  roughness?: number;
  transmission?: number;
  clearcoat?: number;
  envMapIntensity?: number;
  depthWrite?: boolean;
  colorWrite?: boolean;
};

type SegmentMaterialSnapshot = {
  color?: THREE.Color;
  emissive?: THREE.Color;
  map?: THREE.Texture | null;
  emissiveMap?: THREE.Texture | null;
  metalness?: number;
  roughness?: number;
  transparent: boolean;
  opacity: number;
  depthWrite?: boolean;
  colorWrite?: boolean;
};

type StudioMaterialSnapshot = {
  roughness?: number;
  clearcoat?: number;
  envMapIntensity?: number;
  onBeforeCompile: THREE.Material['onBeforeCompile'];
  customProgramCacheKey: THREE.Material['customProgramCacheKey'];
};

export type SegmentEmphasis = 'default' | 'selected' | 'dimmed';

export function segmentColorForIndex(index: number): THREE.Color {
  const base = new THREE.Color(SEGMENTATION_PALETTE[index % SEGMENTATION_PALETTE.length]);
  const hueOffset = Math.floor(index / SEGMENTATION_PALETTE.length) * 0.07;
  return base.offsetHSL(hueOffset, 0.06, 0);
}

function storeSegmentMaterialSnapshot(material: MaterialWithColor): void {
  if (material.userData[SEGMENTATION_SNAPSHOT_KEY]) {
    return;
  }

  material.userData[SEGMENTATION_SNAPSHOT_KEY] = {
    color: material.color?.clone(),
    emissive: material.emissive?.clone(),
    map: material.map ?? null,
    emissiveMap: material.emissiveMap ?? null,
    metalness: material.metalness,
    roughness: material.roughness,
    transparent: material.transparent,
    opacity: material.opacity,
    depthWrite: material.depthWrite,
    colorWrite: material.colorWrite,
  } satisfies SegmentMaterialSnapshot;
}

function restoreSegmentSnapshot(
  material: MaterialWithColor,
  snapshot: SegmentMaterialSnapshot,
): void {
  if (material.color && snapshot.color) {
    material.color.copy(snapshot.color);
  }
  if (material.emissive) {
    if (snapshot.emissive) {
      material.emissive.copy(snapshot.emissive);
    } else {
      material.emissive.setRGB(0, 0, 0);
    }
  }
  if ('map' in material) {
    material.map = snapshot.map ?? null;
  }
  if ('emissiveMap' in material) {
    material.emissiveMap = snapshot.emissiveMap ?? null;
  }
  if ('metalness' in material && typeof snapshot.metalness === 'number') {
    material.metalness = snapshot.metalness;
  }
  if ('roughness' in material && typeof snapshot.roughness === 'number') {
    material.roughness = snapshot.roughness;
  }
  material.transparent = snapshot.transparent;
  material.opacity = snapshot.opacity;
  if ('depthWrite' in material && typeof snapshot.depthWrite === 'boolean') {
    material.depthWrite = snapshot.depthWrite;
  }
  if ('colorWrite' in material && typeof snapshot.colorWrite === 'boolean') {
    material.colorWrite = snapshot.colorWrite;
  }
}

export function applySegmentColor(
  material: MaterialWithColor,
  color: THREE.Color,
  emphasis: SegmentEmphasis,
): void {
  storeSegmentMaterialSnapshot(material);
  const snapshot = material.userData[SEGMENTATION_SNAPSHOT_KEY] as
    | SegmentMaterialSnapshot
    | undefined;
  if (!snapshot) {
    return;
  }
  restoreSegmentSnapshot(material, snapshot);

  const tint = color.clone();
  if (emphasis === 'dimmed') {
    tint.lerp(new THREE.Color('#d8d8d8'), 0.72);
  }

  if (material.color) {
    material.color.copy(tint);
  }
  if (material.emissive) {
    if (emphasis === 'selected') {
      material.emissive.setRGB(0.09, 0.09, 0.09);
    } else if (emphasis === 'dimmed') {
      material.emissive.setRGB(0.018, 0.018, 0.018);
    } else {
      material.emissive.setRGB(0.06, 0.06, 0.06);
    }
  }
  if ('map' in material) {
    material.map = null;
  }
  if ('emissiveMap' in material) {
    material.emissiveMap = null;
  }
  if ('metalness' in material && typeof material.metalness === 'number') {
    material.metalness = emphasis === 'dimmed' ? 0.02 : 0.14;
  }
  if ('roughness' in material && typeof material.roughness === 'number') {
    material.roughness = emphasis === 'dimmed' ? 0.9 : 0.72;
  }
  material.transparent = emphasis === 'dimmed';
  material.opacity = emphasis === 'dimmed' ? 0.1 : 1;
  if ('depthWrite' in material) {
    material.depthWrite = emphasis !== 'dimmed';
  }
  if ('colorWrite' in material) {
    material.colorWrite = true;
  }
  material.needsUpdate = true;
}

export function applySurfaceSampleMeshPresentation(material: MaterialWithColor): void {
  storeSegmentMaterialSnapshot(material);
  material.transparent = true;
  material.opacity = 0;
  if ('depthWrite' in material) {
    material.depthWrite = false;
  }
  if ('colorWrite' in material) {
    material.colorWrite = false;
  }
  material.needsUpdate = true;
}

function injectStudioShading(shader: THREE.WebGLProgramParametersWithUniforms): void {
  shader.vertexShader = shader.vertexShader
    .replace(
      '#include <common>',
      `
#include <common>
varying vec3 vArticraftWorldPosition;
varying vec3 vArticraftWorldNormal;
      `.trim(),
    )
    .replace(
      '#include <beginnormal_vertex>',
      `
#include <beginnormal_vertex>
vArticraftWorldNormal = normalize(mat3(modelMatrix) * objectNormal);
      `.trim(),
    )
    .replace(
      '#include <worldpos_vertex>',
      `
#include <worldpos_vertex>
vArticraftWorldPosition = worldPosition.xyz;
      `.trim(),
    );

  shader.fragmentShader = shader.fragmentShader
    .replace(
      '#include <common>',
      `
#include <common>
varying vec3 vArticraftWorldPosition;
varying vec3 vArticraftWorldNormal;
      `.trim(),
    )
    .replace(
      '#include <dithering_fragment>',
      `
vec3 articraftNormal = normalize(vArticraftWorldNormal);
vec3 articraftViewDir = normalize(cameraPosition - vArticraftWorldPosition);
float articraftKey = max(dot(articraftNormal, normalize(vec3(-0.38, 0.78, 0.50))), 0.0);
float articraftFill = max(dot(articraftNormal, normalize(vec3(0.68, 0.30, -0.44))), 0.0);
float articraftUnderside = 1.0 - smoothstep(-0.12, 0.42, articraftNormal.y);
float articraftRim = pow(1.0 - max(dot(articraftNormal, articraftViewDir), 0.0), 2.4);
float articraftForm = 0.82 + articraftKey * 0.16 + articraftFill * 0.035 + articraftRim * 0.075 - articraftUnderside * 0.075;
gl_FragColor.rgb *= clamp(articraftForm, 0.66, 1.08);
gl_FragColor.rgb += vec3(0.015) * articraftRim;
#include <dithering_fragment>
      `.trim(),
    );
}

/**
 * Whether the studio look is on, for the render options given.
 *
 * Exported because **two** subtrees need the same answer: the staged URDF
 * model, which `SceneCanvas` walks, and a native clip's evaluated meshes,
 * which are built outside that walk and were therefore drawn without it — env
 * lighting at `GLTFLoader`'s default 1.0 against the staged model's clamped
 * 0.62, and none of the form shading. That reads as a clip noticeably
 * brighter and flatter than the scene it animates. Diverging copies of this
 * condition is how that happened, so there is one.
 *
 * The three debug modes are excluded for the same reason as before: each
 * replaces a surface's appearance outright, and studio shading on top of a
 * segmentation color would just misreport it.
 */
export function shouldApplyStudioShading(options: {
  fancyGraphics?: boolean;
  showCollisions?: boolean;
  showSegmentColors?: boolean;
  showSurfaceSamples?: boolean;
}): boolean {
  return Boolean(options.fancyGraphics)
    && !options.showCollisions
    && !options.showSegmentColors
    && !options.showSurfaceSamples;
}


export function applyStudioShading(material: MaterialWithColor): void {
  if (material.userData[STUDIO_SHADING_SNAPSHOT_KEY]) {
    return;
  }

  if (material.transparent || material.opacity < 0.999 || (material.transmission ?? 0) > 0.01) {
    return;
  }

  material.userData[STUDIO_SHADING_SNAPSHOT_KEY] = {
    roughness: material.roughness,
    clearcoat: material.clearcoat,
    envMapIntensity: material.envMapIntensity,
    onBeforeCompile: material.onBeforeCompile,
    customProgramCacheKey: material.customProgramCacheKey,
  } satisfies StudioMaterialSnapshot;

  const snapshot = material.userData[STUDIO_SHADING_SNAPSHOT_KEY] as StudioMaterialSnapshot;
  if (typeof material.roughness === 'number') {
    material.roughness = Math.max(material.roughness, 0.56);
  }
  if (typeof material.clearcoat === 'number') {
    material.clearcoat = Math.min(material.clearcoat, 0.1);
  }
  if (typeof material.envMapIntensity === 'number') {
    material.envMapIntensity = Math.min(material.envMapIntensity, 0.62);
  }

  material.onBeforeCompile = (shader, renderer) => {
    snapshot.onBeforeCompile.call(material, shader, renderer);
    injectStudioShading(shader);
  };
  material.customProgramCacheKey = () => {
    const previousKey = snapshot.customProgramCacheKey.call(material);
    return `${previousKey}:articraft-studio-shading-v1`;
  };
  material.needsUpdate = true;
}

export function restoreStudioShading(material: MaterialWithColor): void {
  const snapshot = material.userData[STUDIO_SHADING_SNAPSHOT_KEY] as
    | StudioMaterialSnapshot
    | undefined;
  if (!snapshot) {
    return;
  }

  if (typeof snapshot.roughness === 'number') {
    material.roughness = snapshot.roughness;
  }
  if (typeof snapshot.clearcoat === 'number') {
    material.clearcoat = snapshot.clearcoat;
  }
  if (typeof snapshot.envMapIntensity === 'number') {
    material.envMapIntensity = snapshot.envMapIntensity;
  }
  material.onBeforeCompile = snapshot.onBeforeCompile;
  material.customProgramCacheKey = snapshot.customProgramCacheKey;
  material.needsUpdate = true;
  delete material.userData[STUDIO_SHADING_SNAPSHOT_KEY];
}

/**
 * Strip inherited studio state from a material, without restoring functions.
 *
 * **For a `clone()` of a material that was already studio-shaded**, which is
 * what a native clip makes when it borrows a staged part's baked material.
 * `Material.copy` carries the *modified* scalars, and it carries `userData`
 * through `JSON.parse(JSON.stringify(...))` — so the snapshot arrives with its
 * numbers intact and its `onBeforeCompile`/`customProgramCacheKey` silently
 * dropped, while the clone itself never inherited the injected shader at all.
 *
 * That half-state is worse than either end of it. `applyStudioShading` sees a
 * snapshot and early-returns, so the form shading never lands; and
 * `restoreStudioShading` would assign `onBeforeCompile = undefined` from the
 * hollow snapshot, which breaks the material's program on the next compile.
 *
 * So: put the numbers back to their pre-studio values, drop the snapshot, and
 * leave the function fields alone — the clone's are already the originals.
 * The material is then a clean baseline that `applyStudioShading` will treat
 * exactly as it treats the staged one.
 */
export function detachStudioShading(material: MaterialWithColor): void {
  const snapshot = material.userData[STUDIO_SHADING_SNAPSHOT_KEY] as
    | Partial<StudioMaterialSnapshot>
    | undefined;
  if (!snapshot) {
    return;
  }
  if (typeof snapshot.roughness === 'number') {
    material.roughness = snapshot.roughness;
  }
  if (typeof snapshot.clearcoat === 'number') {
    material.clearcoat = snapshot.clearcoat;
  }
  if (typeof snapshot.envMapIntensity === 'number') {
    material.envMapIntensity = snapshot.envMapIntensity;
  }
  delete material.userData[STUDIO_SHADING_SNAPSHOT_KEY];
  material.needsUpdate = true;
}


export function restoreSegmentMaterial(material: MaterialWithColor): void {
  const snapshot = material.userData[SEGMENTATION_SNAPSHOT_KEY] as
    | SegmentMaterialSnapshot
    | undefined;
  if (!snapshot) {
    return;
  }

  restoreSegmentSnapshot(material, snapshot);
  material.needsUpdate = true;
  delete material.userData[SEGMENTATION_SNAPSHOT_KEY];
}

export function withMeshMaterials(
  mesh: THREE.Mesh,
  visit: (material: MaterialWithColor) => void,
): void {
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (const material of materials) {
    if (material) {
      visit(material as MaterialWithColor);
    }
  }
}
