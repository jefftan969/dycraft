// Messages between the page and `loader.worker.ts`.
//
// The worker does a run's heavy parsing so the page stays responsive: it fetches, parses, and hands back the result in
// pieces small enough that receiving any one of them is a short task. A parsed object posted whole would cost the page
// a deserialization about as long as the parse, so a big JSON payload comes back as its small fields first, then each
// large top-level array in slices.

export type LoaderRequest =
  | { id: number; kind: "json"; url: string }
  | { id: number; kind: "native-mesh"; url: string }
  /**
   * One frame of a deforming object: its positions (`url`), on the triangles of the object's first mesh
   * (`topologyUrl`), fetched once per `topologyKey`; the whole mesh (`meshUrl`) when the triangles differ.
   */
  | { id: number; kind: "native-frame"; url: string; meshUrl: string; topologyUrl: string; topologyKey: string }
  | { id: number; kind: "obj-chunk"; url: string }
  | { id: number; kind: "obj-file"; url: string }
  | { id: number; kind: "abort" };

/** One OBJ object's drawable, as `OBJLoader` built it, reduced to what crosses `postMessage`. */
export interface SerializedDrawable {
  kind: "Mesh" | "LineSegments" | "Points";
  name: string;
  attributes: Record<string, { array: Float32Array; itemSize: number; normalized: boolean }>;
  groups: { start: number; count: number; materialIndex: number }[];
  materials: { name: string; flatShading: boolean; vertexColors: boolean }[];
  /** `[min xyz, max xyz]` and `[center xyz, radius]`, which three would otherwise compute on the first frame. */
  box: number[];
  sphere: number[];
}

export interface SerializedObj {
  name: string;
  children: SerializedDrawable[];
}

export type LoaderReply =
  | { id: number; type: "progress"; received: number; total: number | null }
  | { id: number; type: "json-head"; value: unknown; arrays: Record<string, number> }
  | { id: number; type: "json-slice"; key: string; items: unknown[] }
  | { id: number; type: "native-mesh"; digest: string; position: Float32Array; normal: Float32Array;
      index: Uint32Array; box: number[]; sphere: number[]; transfers?: Transfer[] }
  | { id: number; type: "objs"; entries: { name: string; object: SerializedObj }[] }
  | { id: number; type: "done" }
  | { id: number; type: "error"; message: string };

/**
 * One native mesh's download as the browser's network stack timed it (Resource Timing): first byte to last, in ms
 * since the epoch, and its bytes on the wire. What `native-stream.ts` measures the link by.
 */
export interface Transfer {
  start: number;
  end: number;
  bytes: number;
}

/** About this much JSON per slice: well under a millisecond to receive. */
export const JSON_SLICE_BYTES = 256 * 1024;

/** A top-level array this long is sent in slices rather than in the head. */
export const JSON_SLICE_MIN_ITEMS = 256;
