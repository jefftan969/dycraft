// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import type { UrdfVisualGeometry } from './urdf-parser';
import type { MaterialSpec } from './materials';
import { applyDepthBias, createMaterial, createEdgeLines } from './materials';
import { applyTriplanarUVs } from './triplanar-uv';
import { DERIVED_MATERIAL_USER_DATA_KEY } from './user-data-keys';
import type { SerializedObj } from '@/workers/loader-protocol';
import { available as workersAvailable, request as requestFromWorker } from '@/workers/pool';
import { objectFromSerialized } from './obj-transfer';

export interface LoadGeometryOptions {
  kind?: 'visual' | 'collision';
  materialSpec?: MaterialSpec;
  doubleSided?: boolean;
  depthBias?: number;
  assetRevisionKey?: string | null;
  /** Base URL for a material's base-color texture; see `createMaterial`. */
  textureBaseUrl?: string;
  /**
   * Keep the material the file arrived with instead of building one from the
   * URDF spec.
   *
   * For a **derived viewer asset**: a baked-atlas GLB carries the canonical
   * Cycles appearance as PBR maps on its own non-overlapping UVs, plus the
   * view-dependent lobes as `KHR_materials_*` scalars. Rebuilding the material
   * from the URDF's flat color would throw all of that away and show the very
   * approximation the bake exists to avoid.
   *
   * `side` and `depthBias` are still applied, so the double-sided toggle and
   * the coincident-surface bias keep working.
   */
  keepLoadedMaterial?: boolean;
}

const geometryTemplateCache = new Map<string, Promise<THREE.Object3D>>();

/**
 * How many mesh downloads may be in flight at once, across the whole app.
 *
 * A scene asks for one mesh per visual and does it all at once:
 * a large recorded scene is 5403 of them, another 868.
 * Handing that to the browser unbounded queues thousands of XHRs behind six
 * usable connections, and Chrome starts failing them outright
 * (`ERR_INSUFFICIENT_RESOURCES`, surfaced to `fetch`/XHR as the bare message
 * "Failed to fetch") — the whole scene then failed on the first casualty.
 *
 * Twelve is above the six-per-host limit of HTTP/1.1, so the connections stay
 * saturated, and low enough that the queue lives here — where it is ordered
 * and bounded — instead of in the network stack.
 */
const MAX_CONCURRENT_MESH_LOADS = 12;

let activeMeshLoads = 0;
const meshLoadWaiters: Array<() => void> = [];

async function withMeshLoadSlot<T>(load: () => Promise<T>): Promise<T> {
  if (activeMeshLoads >= MAX_CONCURRENT_MESH_LOADS) {
    await new Promise<void>((resolve) => meshLoadWaiters.push(resolve));
  }
  activeMeshLoads += 1;
  try {
    return await load();
  } finally {
    activeMeshLoads -= 1;
    meshLoadWaiters.shift()?.();
  }
}

function appendRevisionParam(url: string, assetRevisionKey: string | null | undefined): string {
  if (!assetRevisionKey) {
    return url;
  }

  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}rev=${encodeURIComponent(assetRevisionKey)}`;
}

/**
 * The cache key for one mesh: where it would be fetched from.
 *
 * Exported because `mesh-bundle` primes this cache with meshes that arrive in
 * an archive instead of from their own URL, and a key computed two ways is a
 * key that will eventually disagree — at which case the bundle would silently
 * do nothing and every mesh would be fetched again individually.
 */
export function meshTemplateUrl(
  baseUrl: string,
  filename: string,
  assetRevisionKey?: string | null,
): string {
  const url = baseUrl.endsWith('/')
    ? `${baseUrl}${filename}`
    : `${baseUrl}/${filename}`;
  return appendRevisionParam(url, assetRevisionKey);
}

/**
 * Put an already-downloaded mesh into the template cache.
 *
 * The bundle path: `mesh-bundle` fetches a few tar chunks instead of one
 * request per visual, and hands each entry here. Parsing is the same
 * `OBJLoader` the network path uses, so a primed template and a fetched one
 * are indistinguishable downstream — which is what lets any mesh the bundle
 * misses fall back to its own request with no special case.
 *
 * Returns false when the cache already has this URL (a template in flight
 * wins; it is the same bytes) or when the format cannot travel in an archive.
 */
export function primeGeometryTemplate(
  resolvedUrl: string,
  filename: string,
  data: Uint8Array,
): boolean {
  if (geometryTemplateCache.has(resolvedUrl)) {
    return false;
  }
  if (filename.split('.').pop()?.toLowerCase() !== 'obj') {
    return false;
  }
  const text = new TextDecoder().decode(data);
  const object = new OBJLoader().parse(text);
  geometryTemplateCache.set(resolvedUrl, Promise.resolve(rememberTemplate(object)));
  return true;
}

/**
 * Geometry held by the template cache, which every visual cloned from a template shares. A model reloaded in place
 * with the same meshes (its shader colors arrived) draws these again, so `useUrdfLoader` keeps them on the GPU then
 * rather than dispose them and upload them all again.
 */
const templateGeometries = new WeakSet<THREE.BufferGeometry>();

function rememberTemplate(object: THREE.Object3D): THREE.Object3D {
  object.traverse((child) => {
    const geometry = (child as { geometry?: THREE.BufferGeometry }).geometry;
    if (geometry) templateGeometries.add(geometry);
  });
  return object;
}

export function isTemplateGeometry(geometry: THREE.BufferGeometry): boolean {
  return templateGeometries.has(geometry);
}

/** The worker path's half of `primeGeometryTemplate`: an OBJ already parsed off the page. */
export function primeGeometryTemplateObject(resolvedUrl: string, object: THREE.Object3D): boolean {
  if (geometryTemplateCache.has(resolvedUrl)) {
    return false;
  }
  geometryTemplateCache.set(resolvedUrl, Promise.resolve(rememberTemplate(object)));
  return true;
}

function cloneCachedObject(root: THREE.Object3D): THREE.Object3D {
  const clone = root.clone(true);

  clone.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) {
      return;
    }

    const material = child.material;
    if (Array.isArray(material)) {
      child.material = material.map((entry) => entry.clone());
      return;
    }

    if (material) {
      child.material = material.clone();
    }
  });

  return clone;
}

function loadGeometryTemplate(
  resolvedUrl: string,
  extension: string | undefined,
): Promise<THREE.Object3D> {
  const cached = geometryTemplateCache.get(resolvedUrl);
  if (cached) {
    return cached;
  }

  // Inside the slot, not around the cache lookup: a second visual naming the
  // same digest joins the promise already in flight rather than taking a slot
  // of its own, which is what keeps a scene of eighty identical chairs to one
  // download.
  const pending = withMeshLoadSlot(async () => {
    if (extension === 'glb' || extension === 'gltf') {
      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(resolvedUrl);
      return gltf.scene;
    }

    if (extension === 'obj') {
      if (workersAvailable()) {
        // Parsed in a loader worker, so a scene of hundreds of meshes does not stall the page.
        let parsed: SerializedObj | null = null;
        await requestFromWorker({ kind: 'obj-file', url: resolvedUrl }, (reply) => {
          if (reply.type === 'objs') parsed = reply.entries[0]?.object ?? null;
        });
        if (!parsed) throw new Error(`no mesh parsed from ${resolvedUrl}`);
        return objectFromSerialized(parsed);
      }
      const loader = new OBJLoader();
      return loader.loadAsync(resolvedUrl);
    }

    throw new Error(`Unsupported mesh file format for ${resolvedUrl}`);
  });

  void pending.then(rememberTemplate, () => undefined);
  geometryTemplateCache.set(resolvedUrl, pending);
  void pending.catch(() => {
    if (geometryTemplateCache.get(resolvedUrl) === pending) {
      geometryTemplateCache.delete(resolvedUrl);
    }
  });

  return pending;
}

/**
 * Build a Three.js mesh from a URDF primitive geometry spec (box, cylinder, sphere).
 * Returns null for mesh-type geometries (use loadGeometryObject instead).
 */
export function buildPrimitiveMesh(
  geometry: UrdfVisualGeometry,
  materialSpec: MaterialSpec,
  options: Pick<LoadGeometryOptions, 'doubleSided' | 'depthBias' | 'textureBaseUrl'> = {},
): THREE.Mesh | null {
  let bufferGeometry: THREE.BufferGeometry;

  switch (geometry.type) {
    case 'box': {
      const [x, y, z] = geometry.size ?? [1, 1, 1];
      bufferGeometry = new THREE.BoxGeometry(x, y, z);
      break;
    }
    case 'cylinder': {
      const radius = geometry.radius ?? 1;
      const length = geometry.length ?? 1;
      bufferGeometry = new THREE.CylinderGeometry(radius, radius, length, 32);
      // URDF cylinders are aligned to the Z axis; Three.js uses Y.
      bufferGeometry.rotateX(Math.PI / 2);
      break;
    }
    case 'sphere': {
      const radius = geometry.radius ?? 1;
      bufferGeometry = new THREE.SphereGeometry(radius, 32, 16);
      break;
    }
    default:
      return null;
  }

  const material = createMaterial(materialSpec, {
    side: options.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
    depthBias: options.depthBias,
    textureBaseUrl: options.textureBaseUrl,
  });
  const mesh = new THREE.Mesh(bufferGeometry, material);
  // Before the mesh is placed by its link transform: triplanar UVs are derived
  // from the part's own frame, so the pattern belongs to the part rather than
  // to where the part stands. The Python side inserts at the same point, before
  // `mesh.transform()`.
  applyTextureUVs(mesh, materialSpec, { shared: false });
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return mesh;
}

/**
 * Give a mesh triplanar UVs when its material carries a texture.
 *
 * A spec with a `texturePath` but no `textureScaleM` is skipped deliberately —
 * `urdf-parser` leaves the scale undefined rather than guessing, and a guessed
 * tile size would put the viewport's grain at a different size from the
 * canonical render. Better to show the fallback color than a plausible
 * wrong picture.
 *
 * **`shared` must be true for anything out of `geometryTemplateCache`.**
 * `cloneCachedObject` clones materials by hand precisely because
 * `Object3D.clone()` shares them by reference — and it shares GEOMETRY by
 * reference too. Writing a `uv` attribute straight onto a loaded mesh would
 * reach back into the template and re-parameterize every other part built from
 * the same OBJ, at whatever tile size happened to be applied last.
 * Content-addressed meshes make that common rather than exotic: four identical
 * table legs are one cached template. A primitive's geometry is built fresh per
 * call and owns itself, so it passes false and skips the copy.
 *
 * The cost on the shared path is one geometry copy per textured part. It could
 * be avoided by keying the cache on `url|scaleM` and parameterizing the
 * template once, which is worth doing if a textured import ever gets large; at
 * CAD-solid sizes (hundreds of triangles) it is not worth the extra dimension.
 *
 * **A mesh that arrived with its own UVs keeps them.** A staged OBJ carries
 * per-corner coordinates — baked from the B-rep for a CAD solid, authored into
 * the asset for an imported one — and `OBJLoader` reads them into the `uv`
 * attribute. Those describe the surface in a way a projection cannot: they wrap
 * a barrel as a cylinder rather than cutting it into three planar patches, and
 * they put an atlas where its author meant it. Re-projecting over them is
 * silent, because the result still looks textured. This mirrors
 * `open3d_renderer.py::_apply_texture_uvs`, so the viewport and the canonical
 * render answer the same question the same way.
 */
export function applyTextureUVs(
  mesh: THREE.Mesh,
  spec: MaterialSpec | undefined,
  { shared }: { shared: boolean },
): void {
  if (!spec?.texturePath || !spec.textureScaleM) {
    return;
  }
  if (mesh.geometry.getAttribute('uv')) {
    return;
  }
  if (shared) {
    mesh.geometry = mesh.geometry.clone();
  }
  applyTriplanarUVs(mesh, spec.textureScaleM);
}

/**
 * Load an external mesh file (GLB/GLTF or OBJ) referenced by a URDF mesh geometry.
 * Returns a promise that resolves to a THREE.Group containing the loaded model.
 */
export async function loadGeometryObject(
  geometry: UrdfVisualGeometry,
  baseUrl: string,
  options: LoadGeometryOptions = {},
): Promise<THREE.Group> {
  const template = await loadGeometryTemplateFor(geometry, baseUrl, options.assetRevisionKey);
  return instantiateGeometryObject(template, geometry, options);
}

/** The shared template a mesh visual draws, loaded once per file. */
export function loadGeometryTemplateFor(
  geometry: UrdfVisualGeometry,
  baseUrl: string,
  assetRevisionKey?: string | null,
): Promise<THREE.Object3D> {
  if (geometry.type !== 'mesh' || !geometry.filename) {
    return Promise.reject(new Error(`loadGeometryObject requires a mesh geometry with a filename`));
  }
  const filename = geometry.filename;
  const extension = filename.split('.').pop()?.toLowerCase();
  if (extension !== 'glb' && extension !== 'gltf' && extension !== 'obj') {
    return Promise.reject(new Error(`Unsupported mesh file format: .${extension} (${filename})`));
  }
  return loadGeometryTemplate(meshTemplateUrl(baseUrl, filename, assetRevisionKey), extension);
}

/** One visual from its template: a clone with its own materials, scaled and presented as `options` say. */
export function instantiateGeometryObject(
  template: THREE.Object3D,
  geometry: UrdfVisualGeometry,
  options: LoadGeometryOptions = {},
): THREE.Group {
  const scale = geometry.scale ?? [1, 1, 1];
  const group = new THREE.Group();
  group.add(cloneCachedObject(template));
  group.scale.set(scale[0], scale[1], scale[2]);
  applyLoadedMeshPresentation(group, options);
  return group;
}

function applyLoadedMeshPresentation(root: THREE.Object3D, options: LoadGeometryOptions): void {
  const kind = options.kind ?? 'visual';
  const materialSpec = options.materialSpec;
  const side = options.doubleSided ? THREE.DoubleSide : THREE.FrontSide;
  const depthBias = options.depthBias;

  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) {
      return;
    }

    if (!child.geometry.attributes.normal) {
      child.geometry.computeVertexNormals();
    }

    if (options.keepLoadedMaterial && kind === 'visual') {
      // The file's own material wins, but the two options that are about
      // *drawing* rather than about appearance still apply.
      for (const material of Array.isArray(child.material)
        ? child.material
        : [child.material]) {
        if (!material) continue;
        material.side = side;
        // Through the shared writer, so this path cannot drift from
        // `createMaterial` (for example into `polygonOffsetFactor`, the
        // view-dependent term).
        applyDepthBias(material, depthBias);
        material.needsUpdate = true;
      }
      // Tagged so native playback can tell that this mesh's maps are
      // addressed on the baked atlas, and not lend its material to an
      // evaluated mesh without atlas UVs. See `DERIVED_MATERIAL_USER_DATA_KEY`.
      child.userData[DERIVED_MATERIAL_USER_DATA_KEY] = true;
      // No `applyTextureUVs`: the atlas UVs are in the file, and it would
      // return early on them anyway.
    } else if (materialSpec) {
      child.material = kind === 'collision'
        ? createMaterial(materialSpec, {
            side,
            transparent: true,
            depthBias,
          })
        : createMaterial(materialSpec, {
            side,
            depthBias,
            textureBaseUrl: options.textureBaseUrl,
          });
      // Visual only. A collision proxy is a translucent debug overlay and its
      // material deliberately ignores the texture, so parameterizing it would
      // pay `toNonIndexed` for geometry that will never sample an image.
      if (kind === 'visual') {
        applyTextureUVs(child, materialSpec, { shared: true });
      }
    }

    child.castShadow = kind === 'visual';
    child.receiveShadow = kind === 'visual';
    child.userData.articraftVisual = kind === 'visual';
    child.userData.articraftCollision = kind === 'collision';
    child.visible = kind === 'visual';

    if (kind === 'collision') {
      child.renderOrder = 10;
      child.add(createEdgeLines(child.geometry, materialSpec?.color.getHex() ?? 0x000000));
    }
  });
}

