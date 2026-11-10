// Decode a native clip's motion track, and drive a three.js group from it.
//
// A native animation run has no URDF joint channel to replay: `worker.py`
// walks Blender's depsgraph and writes what Blender *evaluated*, so the
// viewport's job is to draw those objects at those matrices rather than to
// re-derive them. `viewer/native_animation.py` ships that in two pieces, and
// the split is the whole point of this file:
//
//   * one **track** for the clip — objects, colors, and a `[frames][objects]`
//     block of 3x4 matrices;
//   * one **mesh** per distinct content digest, fetched once.
//
// `applyNativeFrame` is then a loop over objects writing `Matrix4`s, with no fetch, no decode, and no allocation per
// frame. The meshes are loaded around the playhead (`native-stream.ts`): a long film's deforming objects carry one
// per frame, more than a tab can hold.
//
// Two things here are load-bearing:
//
//   * **Matrices are rows 0..2, row-major.** Blender hands out `matrix_world`
//     row by row and the server drops the constant `[0,0,0,1]`; `Matrix4.set`
//     takes its arguments in that same row-major order, so the two line up with
//     no transpose. `Matrix4.fromArray` would NOT — it reads column-major, and
//     a matrix read in the wrong major is still a valid matrix, so the mistake
//     shows up as objects in plausibly wrong places rather than as an error.
//   * **Normals are creased, not smoothed.** An evaluated mesh arrives as
//     positions and triangles with no normals and no shading flags, so the
//     viewer has to pick. Smoothing everything (what `computeVertexNormals`
//     alone does) rounds off every hard edge, and a box comes out looking like
//     a pillow; creasing at 40 degrees keeps a box's edges hard and a cylinder's
//     barrel smooth, which is what Blender draws.

import * as THREE from "three";

import { creasedNormals } from "./creased-normals";

import { DERIVED_MATERIAL_USER_DATA_KEY } from "./user-data-keys";
import { detachStudioShading, type MaterialWithColor } from "./scene-presentation";

/** The layout `viewer/native_animation.py::TRACK_VERSION` emits. */
export const NATIVE_TRACK_VERSION = 1;

/**
 * What `scene-graph-builder` names a link group.
 *
 * Declared here, and imported back there, so there is one spelling of it — but
 * here rather than there because this module has to recognize the boundary and
 * is deliberately dependency-light: `check-native-track.mjs` bundles it under
 * node with only `three` external, so importing the graph builder (and its
 * loader chain) for one string would cost that check its footing.
 */
export const LINK_GROUP_PREFIX = "link:";

/** Where a hard edge stops being a crease and starts being a curved surface. */
export const NATIVE_CREASE_ANGLE = (40 * Math.PI) / 180;

/** Floats per object per frame: rows 0..2 of a 4x4, row-major. */
export const MATRIX_STRIDE = 12;

/**
 * What a node holds before its first frame, and on a frame where its object
 * does not exist. Shared and never disposed: `THREE.Mesh`'s constructor would
 * otherwise allocate one empty `BufferGeometry` per object that nothing ever
 * frees, since the frame loop replaces the reference rather than the contents.
 */
const EMPTY_GEOMETRY = new THREE.BufferGeometry();

export interface NativeTrackObject {
  name: string;
  /**
   * The staged part this object came from (`"hill/hill_body"`), or null when
   * the clip authored it in Blender and no staged part corresponds.
   *
   * This is what lets the viewport draw a staged run in its real colors
   * without re-evaluating it — see `nativeMaterial`.
   */
  identity: string | null;
  /** Blender's material color, LINEAR RGBA. */
  color: number[];
  /** Index into `NativeTrack.meshes`; one value when constant, -1 when absent. */
  mesh: number | number[];
  /** One value when constant over the clip. */
  hidden: boolean | boolean[];
}

export interface NativeTrack {
  version: number;
  /** The manifest id this track was built from; changes when the bundle does. */
  animation: string;
  n_frames: number;
  fps: number;
  /** Distinct mesh digests, indexed by `NativeTrackObject.mesh`. */
  meshes: string[];
  objects: NativeTrackObject[];
  /** `[frames][objects][12]`, row-major 3x4. */
  matrices: Float32Array;
}

export interface NativeMesh {
  digest: string;
  positions: Float32Array;
  indices: Uint32Array;
  /** A hash of `indices`: two meshes with the same one share their triangles (`viewer/native_animation.py`). */
  topology?: string;
}

/** `/native-positions`: one mesh's vertices only, for a deforming object whose triangles the viewport already has. */
export interface NativePositions {
  digest: string;
  positions: Float32Array;
  topology: string;
}

/** One baked part's atlas UVs, in evaluated-triangle corner order. */
export interface NativeAtlasEntry {
  /** `<ir_id>/<part>` — the granularity an atlas actually belongs to. */
  key: string;
  offset: number;
  n_corners: number;
  n_tris: number;
}

export interface NativeAtlas {
  /** `u16` normalized to [0,1], or `f32` when a bake escaped that range. */
  dtype: "u16" | "f32";
  atlases: NativeAtlasEntry[];
  objects: { name: string; uv_space: string; atlas: number }[];
  data: Uint16Array | Float32Array;
}

function envelope(buffer: ArrayBuffer): { header: Record<string, unknown>; offset: number } {
  if (buffer.byteLength < 4) throw new Error("native blob is empty");
  const length = new DataView(buffer).getUint32(0, true);
  if (4 + length > buffer.byteLength) {
    throw new Error("native blob header runs past the end of the response");
  }
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, length)),
  ) as Record<string, unknown>;
  if (header.version !== NATIVE_TRACK_VERSION) {
    throw new Error(
      `native blob version ${String(header.version)}, expected ${NATIVE_TRACK_VERSION}`,
    );
  }
  return { header, offset: 4 + length };
}

export function decodeNativeTrack(buffer: ArrayBuffer): NativeTrack {
  const { header, offset } = envelope(buffer);
  const track = header as unknown as NativeTrack;
  const count = track.n_frames * track.objects.length * MATRIX_STRIDE;
  if (offset + count * 4 !== buffer.byteLength) {
    throw new Error(
      `native track carries ${buffer.byteLength - offset} matrix bytes, ` +
      `expected ${count * 4} for ${track.n_frames} frames of ` +
      `${track.objects.length} objects`,
    );
  }
  track.matrices = new Float32Array(buffer, offset, count);
  return track;
}

export function decodeNativeMesh(buffer: ArrayBuffer): NativeMesh {
  const { header, offset } = envelope(buffer);
  const vertices = header.vertices as number;
  const triangles = header.triangles as number;
  const end = offset + vertices * 12 + triangles * 12;
  if (end !== buffer.byteLength) {
    throw new Error(
      `native mesh carries ${buffer.byteLength - offset} bytes, expected ` +
      `${end - offset} for ${vertices} vertices and ${triangles} triangles`,
    );
  }
  return {
    digest: header.digest as string,
    positions: new Float32Array(buffer, offset, vertices * 3),
    indices: new Uint32Array(buffer, offset + vertices * 12, triangles * 3),
    topology: typeof header.topology === "string" ? header.topology : undefined,
  };
}

/** A `/native-positions` reply: the mesh's vertices, exactly as its whole mesh carries them. */
export function decodeNativePositions(buffer: ArrayBuffer): NativePositions {
  const { header, offset } = envelope(buffer);
  const vertices = header.vertices as number;
  if (header.encoding !== "f32" || offset + vertices * 12 !== buffer.byteLength) {
    throw new Error(
      `native positions carry ${buffer.byteLength - offset} bytes, expected ${vertices * 12} for ${vertices} vertices`);
  }
  return {
    digest: header.digest as string,
    positions: new Float32Array(buffer, offset, vertices * 3),
    topology: header.topology as string,
  };
}

export function decodeNativeAtlas(buffer: ArrayBuffer): NativeAtlas {
  const { header, offset } = envelope(buffer);
  const atlas = header as unknown as NativeAtlas;
  const corners = (atlas.atlases ?? []).reduce((n, a) => n + a.n_corners, 0);
  const width = atlas.dtype === "f32" ? 4 : 2;
  if (offset + corners * 2 * width !== buffer.byteLength) {
    throw new Error(
      `native atlas carries ${buffer.byteLength - offset} uv bytes, expected ` +
      `${corners * 2 * width} for ${corners} corners`,
    );
  }
  atlas.data = atlas.dtype === "f32"
    ? new Float32Array(buffer, offset, corners * 2)
    : new Uint16Array(buffer, offset, corners * 2);
  return atlas;
}

/**
 * The uv attribute for one object, or null when it has no baked atlas.
 *
 * `normalized: true` on the `u16` path is what makes the fixed-point encoding
 * free: three divides by 65535 in the vertex shader, so nothing is converted
 * or copied on the way in.
 */
export function atlasAttribute(
  atlas: NativeAtlas | null,
  objectName: string,
): THREE.BufferAttribute | null {
  if (!atlas) return null;
  const row = atlas.objects.find((o) => o.name === objectName);
  if (!row || row.uv_space !== "atlas") return null;
  const entry = atlas.atlases[row.atlas];
  if (!entry) return null;
  const from = entry.offset * 2;
  const to = from + entry.n_corners * 2;
  return atlas.dtype === "f32"
    ? new THREE.BufferAttribute((atlas.data as Float32Array).subarray(from, to), 2)
    : new THREE.BufferAttribute((atlas.data as Uint16Array).subarray(from, to), 2, true);
}

/** A per-frame value that the server collapsed when it never changed. */
function at<T>(value: T | T[], frame: number): T {
  return Array.isArray(value) ? value[frame] : value;
}

export function meshIndexAt(object: NativeTrackObject, frame: number): number {
  return at(object.mesh, frame) ?? -1;
}

export function hiddenAt(object: NativeTrackObject, frame: number): boolean {
  return at(object.hidden, frame) ?? false;
}

export function nativeGeometry(mesh: NativeMesh): THREE.BufferGeometry {
  const { position, normal, index } = creasedNormals(mesh.positions, mesh.indices, NATIVE_CREASE_ANGLE);
  return nativeGeometryFromArrays(position, normal, undefined, undefined, index);
}

/**
 * A creased mesh the loader worker built (`creasedNormals`), as a geometry, indexed when `index` is given. `box`
 * (`[min, max]`) and `sphere` (`[center, radius]`) are its bounds, which three would otherwise compute on the first
 * frame that draws it.
 */
export function nativeGeometryFromArrays(
  position: Float32Array, normal: Float32Array, box?: number[], sphere?: number[], index?: Uint32Array,
): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
  if (index) geometry.setIndex(new THREE.BufferAttribute(index, 1));
  if (box) geometry.boundingBox = new THREE.Box3().set(new THREE.Vector3(box[0], box[1], box[2]),
                                                       new THREE.Vector3(box[3], box[4], box[5]));
  if (sphere) geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(sphere[0], sphere[1], sphere[2]), sphere[3]);
  return geometry;
}

/**
 * The URDF link a staged part is drawn under.
 *
 * `scene_ir_to_urdf` prefixes every link with its instance and a double
 * underscore, and `worker.py` splits the identity back on its FIRST slash —
 * so `wheel/rim` is link `wheel__rim` while a nested part keeps the rest of its
 * path intact. Replacing every slash would miss exactly those.
 */
export function linkNameForIdentity(identity: string): string {
  return identity.replace("/", "__");
}

/**
 * What to draw this object with.
 *
 * **The staged part's own material wins when there is one.** Blender's
 * `diffuse_color` is a viewport swatch, not the shader — a scene whose parts
 * are all authored through a Principled BSDF leaves it at Blender's default
 * 0.8 gray, which is how native playback came to draw every object the same
 * color. The staged scene beside it already knows what each part looks like,
 * and it is the same part: `identity` names it.
 *
 * **Maps are borrowed only when the object has atlas UVs to read them with.**
 * `worker.py` exports positions and triangles, so an evaluated mesh has no UVs
 * of its own and binding a textured material to it would sample one corner
 * texel over the whole surface — a confident flat color that is simply wrong.
 * When `/native-atlas` supplies the bake's own corner UVs, that objection goes
 * away and the baked maps are exactly the right thing to draw; without them a
 * baked material is still no use, and `stagedMaterial` declines to lend one.
 */
export function nativeMaterial(
  object: NativeTrackObject,
  links: Map<string, THREE.Group> | null,
  hasAtlasUVs = false,
): THREE.MeshStandardMaterial {
  const staged = object.identity
    ? stagedMaterial(links?.get(linkNameForIdentity(object.identity)), hasAtlasUVs)
    : null;
  if (staged) {
    // A baked material's appearance IS its maps, so the scalars alone would
    // draw it white — `createMaterial` whitens `color` under a map and pins
    // `roughness` to 1.0 under a roughness map for the same reason.
    if (hasAtlasUVs && staged.map) {
      const material = staged.clone();
      // The staged material may already be studio-shaded by the time the clip
      // finishes loading, and a clone of one is a half-state: modified
      // scalars, a hollow snapshot, no injected shader. Reset it to the
      // pre-studio baseline so the clip's own studio pass treats it exactly
      // as `SceneCanvas` treats the staged model.
      detachStudioShading(material as unknown as MaterialWithColor);
      material.side = THREE.DoubleSide;
      return material;
    }
    return new THREE.MeshStandardMaterial({
      color: staged.color.clone(),
      roughness: staged.roughness,
      metalness: staged.metalness,
      side: THREE.DoubleSide,
    });
  }
  const [r, g, b] = object.color;
  return new THREE.MeshStandardMaterial({
    // `setRGB` writes in three's working color space, which is linear — and
    // Blender's material colors are linear too. No conversion, deliberately.
    color: new THREE.Color().setRGB(r ?? 0.8, g ?? 0.8, b ?? 0.8),
    roughness: 0.65,
    side: THREE.DoubleSide,
  });
}

/**
 * The material of a staged link's OWN visual, or null when it has none to lend.
 *
 * Two ways of getting this wrong, both of which draw a whole scene in
 * plausible, consistent, wrong colors rather than failing.
 *
 * **A link group contains the links below it.** `scene-graph-builder` hangs a
 * child link off its parent through a `joint-frame:` group, so `traverse` walks
 * the entire kinematic subtree and the first visual it meets on any non-leaf
 * part belongs to some descendant. In a recorded wheel scene that is 11 visuals under
 * `link:rolling_wheel__earth` and 6 under `link:rolling_wheel__rim`, one each of
 * which is the part's own: the hill drew in turf green and the rim and hub in
 * tire black. So the walk stops at the next link rather than entering it.
 *
 * **A baked material has no color to lend.** A derived (baked-atlas) asset
 * keeps the material its GLB arrived with, where the appearance is the map and
 * `color` is the neutral multiplier in front of it. Copying the scalars and
 * dropping the map — all this path can do, since an evaluated mesh has no UVs —
 * turns every object white. `bakedMaterials` is on by default, so that was the
 * ordinary case: the clip played back as uniform gray while the staged scene
 * beside it was fully textured.
 */
function stagedMaterial(
  group: THREE.Group | undefined,
  allowDerived = false,
): THREE.MeshStandardMaterial | null {
  if (!group) return null;
  let found: THREE.MeshStandardMaterial | null = null;
  const visit = (object: THREE.Object3D): void => {
    if (found) return;
    // Another link begins here, and everything below it is a different part.
    if (object !== group && object.name.startsWith(LINK_GROUP_PREFIX)) return;
    // Visuals only: a link's other children are collision proxies, which are
    // drawn in a debug palette that has nothing to do with the part.
    if (object instanceof THREE.Mesh
        && object.userData.articraftVisual === true
        && (allowDerived
            || object.userData[DERIVED_MATERIAL_USER_DATA_KEY] !== true)) {
      const material = Array.isArray(object.material) ? object.material[0] : object.material;
      if (material instanceof THREE.MeshStandardMaterial) {
        found = material;
        return;
      }
    }
    for (const child of object.children) visit(child);
  };
  visit(group);
  return found;
}

/** The drawn form of a track: one Mesh per object, reused for every frame. */
export interface NativeScene {
  group: THREE.Group;
  track: NativeTrack;
  nodes: THREE.Mesh[];
  geometries: Map<string, THREE.BufferGeometry>;
  /**
   * Per object, the suffix its geometry is keyed under in `geometries`.
   *
   * **Load-bearing.** Evaluated geometry is deduplicated by content, so ten
   * robot parts can share one mesh digest while owning ten different atlas
   * islands — mesh #85 on this corpus. Keying geometry by digest alone would
   * put one part's UVs on all ten and draw nine of them with the wrong patch
   * of texture, which looks plausible and is wrong.
   */
  atlasKeys: (string | null)[];
  /** Per object, its atlas UVs, for the meshes that arrive after the scene is built; null when it has none. */
  atlasUVs: (THREE.BufferAttribute | null)[];
}

/**
 * `base` wearing `uv`: one geometry per object that has an atlas, built when the object first draws the mesh.
 *
 * The atlas has a UV per evaluated corner, so an indexed mesh is de-indexed for it, which copies its corners once;
 * a mesh that is not indexed hands its `position` and `normal` over by reference instead. Either way nothing here
 * mutates a base attribute.
 */
function withAtlasUVs(
  base: THREE.BufferGeometry,
  uv: THREE.BufferAttribute,
): THREE.BufferGeometry {
  // The atlas holds one UV per evaluated corner, in triangle order: an indexed mesh is de-indexed first, which
  // `toNonIndexed` does in exactly that order.
  const corners = base.index ? base.toNonIndexed() : base;
  const geometry = new THREE.BufferGeometry();
  for (const name of ["position", "normal"]) {
    const attribute = corners.getAttribute(name);
    if (attribute) geometry.setAttribute(name, attribute);
  }
  geometry.setAttribute("uv", uv);
  return geometry;
}

/** A mesh's evaluated corners: three per triangle, whether it is indexed or not. */
function cornerCount(geometry: THREE.BufferGeometry): number | undefined {
  return geometry.index ? geometry.index.count : geometry.getAttribute("position")?.count;
}

export function buildNativeScene(
  track: NativeTrack,
  geometries: Map<string, THREE.BufferGeometry>,
  links: Map<string, THREE.Group> | null,
  atlas: NativeAtlas | null = null,
): NativeScene {
  const group = new THREE.Group();
  group.name = "native-animation";
  const atlasKeys: (string | null)[] = [];
  const atlasUVs: (THREE.BufferAttribute | null)[] = [];
  const nodes = track.objects.map((object) => {
    // Resolved per object, then materialized for every digest that object can
    // wear — across this corpus exactly one, but the track permits more.
    const uv = atlasAttribute(atlas, object.name);
    let atlasKey: string | null = null;
    if (uv) {
      const row = atlas!.objects.find((o) => o.name === object.name)!;
      atlasKey = `|${atlas!.atlases[row.atlas].key}`;
      for (const digest of track.meshes) {
        const base = geometries.get(digest);
        const key = digest + atlasKey;
        if (!base || geometries.has(key)) continue;
        // Only where the corner counts agree. The server checks this against
        // frame 0; a mesh that disagrees here would be silently mis-mapped.
        if (cornerCount(base) !== uv.count) continue;
        geometries.set(key, withAtlasUVs(base, uv));
      }
    }
    atlasKeys.push(atlasKey);
    atlasUVs.push(uv);
    const mesh = new THREE.Mesh(
      EMPTY_GEOMETRY, nativeMaterial(object, links, Boolean(uv)));
    mesh.name = object.name;
    // The matrix is written per frame from the track, so three must not
    // recompute it from position/quaternion/scale, which stay at their
    // defaults and would silently reset it to identity on the first render.
    mesh.matrixAutoUpdate = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  });
  return { group, track, nodes, geometries, atlasKeys, atlasUVs };
}

/** How far a frame looks for an object's nearest built mesh when its own has not arrived and it shows nothing yet. */
const NEAREST_SEARCH_FRAMES = 240;

/** The geometry object `i` wears for `digest`: its atlas-UV variant when it has one, built on first use. */
function geometryFor(scene: NativeScene, i: number, digest: string): THREE.BufferGeometry | undefined {
  const base = scene.geometries.get(digest);
  const key = scene.atlasKeys[i];
  const uv = scene.atlasUVs[i];
  if (!key || !uv || !base) return base;
  const keyed = digest + key;
  let variant = scene.geometries.get(keyed);
  // Only where the corner counts agree; otherwise the object draws untextured rather than mis-mapped.
  if (!variant && cornerCount(base) === uv.count) {
    variant = withAtlasUVs(base, uv);
    scene.geometries.set(keyed, variant);
  }
  return variant ?? base;
}

/** The nearest frame's built mesh for an object that shows nothing yet and whose own mesh has not arrived. */
function nearestBuilt(scene: NativeScene, i: number, frame: number): THREE.BufferGeometry | undefined {
  const object = scene.track.objects[i];
  for (let d = 1; d <= NEAREST_SEARCH_FRAMES; d += 1) {
    for (const f of [frame - d, frame + d]) {
      if (f < 0 || f >= scene.track.n_frames) continue;
      const index = meshIndexAt(object, f);
      if (index < 0) continue;
      const geometry = geometryFor(scene, i, scene.track.meshes[index]);
      if (geometry) return geometry;
    }
  }
  return undefined;
}

/**
 * Point the scene at one frame.
 *
 * Allocation-free and idempotent: it writes into the Meshes built above and
 * swaps a geometry only on a frame where the object's digest changes, which is what makes scrubbing cheap.
 */
export function applyNativeFrame(scene: NativeScene, frame: number, stride = 1): void {
  const { track, nodes } = scene;
  const clamped = Math.max(0, Math.min(track.n_frames - 1, Math.trunc(frame)));
  const base = clamped * nodes.length * MATRIX_STRIDE;
  const m = track.matrices;
  // On a slow link the stream fetches deforming meshes only on every `stride`-th frame (`native-stream.ts`), and a
  // frame between counts as ready when the one on the stride before it is: that is the mesh it shows.
  const onStride = stride > 1 ? clamped - (clamped % stride) : clamped;
  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    const object = track.objects[i];
    const index = meshIndexAt(object, clamped);
    const digest = index < 0 ? null : track.meshes[index];
    // A mesh still loading (`native-stream.ts`): the object shows the stride's frame's mesh, or else keeps the mesh
    // it shows, or the nearest frame's, at this frame's placement, rather than vanishing until it arrives.
    let geometry = digest === null ? undefined : geometryFor(scene, i, digest);
    if (digest !== null && !geometry && onStride !== clamped) {
      const strideIndex = meshIndexAt(object, onStride);
      if (strideIndex >= 0) geometry = geometryFor(scene, i, track.meshes[strideIndex]);
    }
    if (digest !== null && !geometry) {
      geometry = node.geometry !== EMPTY_GEOMETRY ? node.geometry : nearestBuilt(scene, i, clamped);
    }
    node.visible = Boolean(geometry) && !hiddenAt(object, clamped);
    if (!node.visible) continue;
    if (node.geometry !== geometry) node.geometry = geometry!;
    const o = base + i * MATRIX_STRIDE;
    // Row-major, exactly as `Matrix4.set` reads its arguments — see the header.
    node.matrix.set(
      m[o], m[o + 1], m[o + 2], m[o + 3],
      m[o + 4], m[o + 5], m[o + 6], m[o + 7],
      m[o + 8], m[o + 9], m[o + 10], m[o + 11],
      0, 0, 0, 1,
    );
    // `matrixAutoUpdate` is off, so nothing else will notice the write.
    node.matrixWorldNeedsUpdate = true;
  }
}

export function disposeNativeScene(scene: NativeScene): void {
  for (const node of scene.nodes) {
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) material.dispose();
  }
  for (const geometry of scene.geometries.values()) geometry.dispose();
  scene.geometries.clear();
  scene.group.clear();
}
