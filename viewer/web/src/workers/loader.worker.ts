/// <reference lib="webworker" />
// The page's heavy parsing, off its main thread: JSON payloads, native meshes, and the OBJ mesh bundle.
//
// Every reply is sized so that receiving it is a short task on the page (`loader-protocol.ts`). Typed arrays are
// transferred, not copied.

import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

import { creasedNormals } from "../components/viewer3d/creased-normals";
import { decodeNativeMesh, decodeNativePositions, NATIVE_CREASE_ANGLE } from "../components/viewer3d/native-track";
import { readTarEntries } from "../components/viewer3d/tar";
import {
  type Transfer,
  JSON_SLICE_BYTES,
  JSON_SLICE_MIN_ITEMS,
  type LoaderReply,
  type LoaderRequest,
  type SerializedDrawable,
  type SerializedObj,
} from "./loader-protocol";

const scope = self as unknown as DedicatedWorkerGlobalScope;
const inFlight = new Map<number, AbortController>();

/**
 * Native mesh downloads since the last mesh reply, as the network stack timed them, handed to the page with the next
 * reply. Timed by the browser rather than here because this worker may be building another mesh while a body
 * streams in, which would make a fast link look slow. Cache hits (no bytes on the wire) say nothing about the link.
 */
const transfers: Transfer[] = [];
try {
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries() as PerformanceResourceTiming[]) {
      if (!/\/native-(mesh|positions)\//.test(entry.name) || entry.transferSize === 0 || !entry.responseStart) continue;
      transfers.push({
        start: performance.timeOrigin + entry.responseStart,
        end: performance.timeOrigin + entry.responseEnd,
        bytes: entry.encodedBodySize,
      });
      if (transfers.length > 64) transfers.shift();
    }
    // The observer has them; the buffer would otherwise fill at 250 and stop recording.
    performance.clearResourceTimings();
  }).observe({ type: "resource" });
} catch {
  // No Resource Timing in this worker: the page sees no transfers and never thins a clip's frames.
}

/** A deforming object's triangles, and the hash the server gives them, to check each frame against. */
interface Topology {
  indices: Uint32Array;
  topology?: string;
}

/**
 * Deforming objects' triangles, by the object's first mesh: fetched once per worker and shared by every frame of the
 * object this worker builds. Frames go to whichever worker is least busy, so each worker may fetch an object's
 * triangles once; the mesh is immutable, so after the first the browser's cache answers. Bounded by bytes, the least
 * recently used out first.
 */
const topologies = new Map<string, { promise: Promise<Topology>; bytes: number }>();
const TOPOLOGY_BUDGET_BYTES = 64 * 1024 * 1024;
let topologyBytes = 0;

function topologyOf(key: string, url: string): Promise<Topology> {
  const known = topologies.get(key);
  if (known) {
    // Most recently used goes last, so eviction takes the oldest first.
    topologies.delete(key);
    topologies.set(key, known);
    return known.promise;
  }
  // Under no request's id, so no request's progress counts it; and not tied to the asking request's signal, since
  // other frames of the object wait on the same fetch.
  const entry = {
    bytes: 0,
    promise: fetchBytes(null, url, new AbortController().signal).then((bytes): Topology => {
      const mesh = decodeNativeMesh(bytes.buffer as ArrayBuffer);
      return { indices: mesh.indices, topology: mesh.topology };
    }),
  };
  topologies.set(key, entry);
  entry.promise.then((topology) => {
    if (topologies.get(key) !== entry) return;
    entry.bytes = topology.indices.byteLength;
    topologyBytes += entry.bytes;
    for (const [other, held] of topologies) {
      if (topologyBytes <= TOPOLOGY_BUDGET_BYTES) break;
      if (other === key) continue;
      topologies.delete(other);
      topologyBytes -= held.bytes;
    }
  }, () => {
    if (topologies.get(key) === entry) topologies.delete(key);
  });
  return entry.promise;
}

/** One native mesh, built: creased normals, bounds, and the arrays handed to the page. */
function replyMesh(id: number, mesh: { digest: string; positions: Float32Array; indices: Uint32Array }): void {
  const { position, normal, index } = creasedNormals(mesh.positions, mesh.indices, NATIVE_CREASE_ANGLE);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const box = [...geometry.boundingBox!.min.toArray(), ...geometry.boundingBox!.max.toArray()];
  const sphere = [...geometry.boundingSphere!.center.toArray(), geometry.boundingSphere!.radius];
  reply({ id, type: "native-mesh", digest: mesh.digest, position, normal, index, box, sphere,
          transfers: transfers.splice(0) },
        [position.buffer, normal.buffer, index.buffer]);
}

function reply(message: LoaderReply, transfer: Transferable[] = []): void {
  // A buffer listed twice is refused, and two attributes may share one.
  scope.postMessage(message, [...new Set(transfer)]);
}

/** The body, with progress about every 100 ms. `total` is the declared length, null when the server gave none. */
/** A response's bytes; with `id`, its progress reported as that request's. */
async function fetchBytes(id: number | null, url: string, signal: AbortSignal): Promise<Uint8Array> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // Not JSON: the status line is all there is.
    }
    throw new Error(detail);
  }
  const declared = Number(response.headers.get("Content-Length"));
  const total = Number.isFinite(declared) && declared > 0 ? declared : null;
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const parts: Uint8Array[] = [];
  let received = 0;
  let reported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.byteLength;
    const now = performance.now();
    if (id !== null && now - reported > 100) {
      reported = now;
      reply({ id, type: "progress", received, total });
    }
  }
  if (id !== null) reply({ id, type: "progress", received, total });
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.byteLength;
  }
  return bytes;
}

/** Post a parsed payload as its small fields, then each large top-level array in slices. */
function replyJson(id: number, value: unknown, bytes: number): void {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    reply({ id, type: "json-head", value, arrays: {} });
    return;
  }
  const head: Record<string, unknown> = { ...(value as Record<string, unknown>) };
  const arrays: Record<string, number> = {};
  let sliced = 0;
  for (const [key, field] of Object.entries(head)) {
    if (Array.isArray(field) && field.length >= JSON_SLICE_MIN_ITEMS) {
      arrays[key] = field.length;
      sliced += field.length;
      head[key] = [];
    }
  }
  reply({ id, type: "json-head", value: head, arrays });
  if (sliced === 0) return;
  // The big arrays are nearly all of the payload, so its size over their items is each item's typical size.
  const perSlice = Math.max(16, Math.floor(JSON_SLICE_BYTES / Math.max(1, bytes / sliced)));
  for (const key of Object.keys(arrays)) {
    const items = (value as Record<string, unknown[]>)[key];
    for (let i = 0; i < items.length; i += perSlice) {
      reply({ id, type: "json-slice", key, items: items.slice(i, i + perSlice) });
    }
  }
}

function serializeObj(name: string, root: THREE.Object3D): { object: SerializedObj; transfer: Transferable[] } {
  const children: SerializedDrawable[] = [];
  const transfer: Transferable[] = [];
  root.traverse((child) => {
    const kind = child instanceof THREE.Mesh ? "Mesh"
      : child instanceof THREE.LineSegments ? "LineSegments"
        : child instanceof THREE.Points ? "Points" : null;
    if (!kind) return;
    const drawable = child as THREE.Mesh;
    const geometry = drawable.geometry as THREE.BufferGeometry;
    // What the page would otherwise do on first use (`applyLoadedMeshPresentation`), done here instead.
    if (kind === "Mesh" && !geometry.attributes.normal) geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const box = geometry.boundingBox!;
    const sphere = geometry.boundingSphere!;
    const attributes: SerializedDrawable["attributes"] = {};
    for (const [key, attribute] of Object.entries(geometry.attributes)) {
      const array = (attribute as THREE.BufferAttribute).array as Float32Array;
      attributes[key] = { array, itemSize: attribute.itemSize, normalized: attribute.normalized };
      transfer.push(array.buffer);
    }
    const materials = (Array.isArray(drawable.material) ? drawable.material : [drawable.material])
      .map((material) => ({
        name: material?.name ?? "",
        flatShading: Boolean((material as THREE.MeshPhongMaterial | undefined)?.flatShading),
        vertexColors: Boolean(material?.vertexColors),
      }));
    children.push({ kind, name: child.name, attributes, groups: geometry.groups.map((g) => ({
      start: g.start, count: g.count, materialIndex: g.materialIndex ?? 0 })), materials,
      box: [...box.min.toArray(), ...box.max.toArray()], sphere: [...sphere.center.toArray(), sphere.radius] });
  });
  return { object: { name, children }, transfer };
}

function parseObj(name: string, data: Uint8Array): { object: SerializedObj; transfer: Transferable[] } {
  return serializeObj(name, new OBJLoader().parse(new TextDecoder().decode(data)));
}

async function handle(request: Exclude<LoaderRequest, { kind: "abort" }>, signal: AbortSignal): Promise<void> {
  const { id } = request;
  if (request.kind === "json") {
    const bytes = await fetchBytes(id, request.url, signal);
    replyJson(id, JSON.parse(new TextDecoder().decode(bytes)), bytes.byteLength);
  } else if (request.kind === "native-mesh") {
    const bytes = await fetchBytes(id, request.url, signal);
    replyMesh(id, decodeNativeMesh(bytes.buffer as ArrayBuffer));
  } else if (request.kind === "native-frame") {
    // A deforming object's frame: its positions on the triangles every frame of the object shares.
    // The triangles failing to arrive is not this frame's failure: it is then fetched whole.
    const [bytes, topology] = await Promise.all([
      fetchBytes(id, request.url, signal), topologyOf(request.topologyKey, request.topologyUrl).catch(() => null)]);
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    const frame = decodeNativePositions(bytes.buffer as ArrayBuffer);
    if (topology?.topology !== undefined && frame.topology === topology.topology) {
      replyMesh(id, { digest: frame.digest, positions: frame.positions, indices: topology.indices });
    } else {
      // Not the same triangles after all, or none to hand: the whole mesh, as for any other.
      const whole = await fetchBytes(id, request.meshUrl, signal);
      replyMesh(id, decodeNativeMesh(whole.buffer as ArrayBuffer));
    }
  } else if (request.kind === "obj-chunk") {
    const bytes = await fetchBytes(id, request.url, signal);
    // One reply per entry: each is a few small arrays to receive.
    for (const entry of readTarEntries(bytes.buffer as ArrayBuffer)) {
      if (signal.aborted) return;
      if (entry.name.split(".").pop()?.toLowerCase() !== "obj") continue;
      try {
        const { object, transfer } = parseObj(entry.name, entry.data);
        reply({ id, type: "objs", entries: [{ name: entry.name, object }] }, transfer);
      } catch {
        // A mesh that will not parse is fetched on its own later, where the per-file path reports it.
      }
    }
  } else if (request.kind === "obj-file") {
    const bytes = await fetchBytes(id, request.url, signal);
    const { object, transfer } = parseObj(request.url, bytes);
    reply({ id, type: "objs", entries: [{ name: request.url, object }] }, transfer);
  }
}

scope.onmessage = (event: MessageEvent<LoaderRequest>) => {
  const request = event.data;
  if (request.kind === "abort") {
    inFlight.get(request.id)?.abort();
    inFlight.delete(request.id);
    return;
  }
  const controller = new AbortController();
  inFlight.set(request.id, controller);
  handle(request, controller.signal)
    .then(() => { if (!controller.signal.aborted) reply({ id: request.id, type: "done" }); })
    .catch((error: unknown) => {
      if (controller.signal.aborted) return;
      reply({ id: request.id, type: "error", message: error instanceof Error ? error.message : String(error) });
    })
    .finally(() => { inFlight.delete(request.id); });
};
