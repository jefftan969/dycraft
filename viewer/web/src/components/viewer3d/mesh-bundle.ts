// Fetch a scene's meshes as a few tar chunks instead of one request per visual.
//
// The server half is `viewer/mesh_bundle.py`, which explains the packing. This
// half reads the archives and primes the template cache the per-file loader
// reads, so bundling is a *prefetch*: anything it misses — an older server with
// no such route, a chunk that failed, a mesh whose path could not travel in a
// USTAR header — is fetched individually exactly as before. There is no mode to
// be in and nothing to keep consistent.
//
// Why it matters: a large scene names thousands of meshes, and thousands of
// queued XHRs is how Chrome ends up refusing requests outright.

import { available as workersAvailable, request as requestFromWorker } from '@/workers/pool';
import { meshTemplateUrl, primeGeometryTemplate, primeGeometryTemplateObject } from './geometry-loader';
import { objectFromSerialized } from './obj-transfer';

/** Chunks in flight. Low: each one is ~16 MB and parsing blocks the thread. */
const MAX_CONCURRENT_CHUNKS = 3;

/**
 * Entries parsed between yields to the event loop.
 *
 * `OBJLoader.parse` is synchronous, so a whole chunk back-to-back is a quarter
 * to half a second of frozen tab on the heavier scenes. Breaking it up keeps
 * the viewport interactive while the rest arrives.
 */
const PARSE_YIELD_EVERY = 32;

import { readTarEntries } from './tar';

export { readTarEntries, type TarEntry } from './tar';

export interface MeshBundleResult {
  /** Chunks the manifest offered. */
  chunks: number;
  /** Chunks that arrived and parsed. */
  chunksLoaded: number;
  /** Templates primed — meshes that will not be fetched individually. */
  primed: number;
  bytes: number;
}

/**
 * The bundle routes sit beside the file route.
 *
 * `runFileBase` builds `<api>/api/runs/<id>/files` and the loader resolves
 * every `<mesh filename>` against it, so the sibling is derivable — which is
 * worth one assumption here rather than threading a second base URL through
 * `App`, `ViewportPanel`, `SceneCanvas`, and `useUrdfLoader`. Anything that is
 * not that shape disables bundling instead of guessing.
 */
export function meshBundleBase(baseFileUrl: string): string | null {
  const suffix = '/files';
  if (!baseFileUrl.endsWith(suffix)) return null;
  return `${baseFileUrl.slice(0, -suffix.length)}/mesh-bundle`;
}

interface Manifest {
  files: number;
  bytes: number;
  chunks: { index: number; files: number; bytes: number }[];
}

interface PrefetchOptions {
  /** `runFileBase(runId)` — what the URDF's mesh paths resolve against. */
  baseUrl: string;
  /** Cache-buster and immutability key; the IR hash. */
  assetRevisionKey: string | null;
  /** Ask for the baked-asset variant's mesh set. */
  derived: boolean;
  /** Stop early — the selection changed while chunks were arriving. */
  isCancelled?: () => boolean;
  /** Chunks arrived and meshes primed so far, out of the manifest's. */
  onProgress?: (progress: MeshBundleProgress) => void;
}

export interface MeshBundleProgress {
  chunks: number;
  chunksLoaded: number;
  files: number;
  primed: number;
}

const yieldToEventLoop = (): Promise<void> =>
  new Promise((resolve) => { setTimeout(resolve, 0); });

/**
 * Download this run's mesh bundle and prime the template cache from it.
 *
 * Resolves to null when there is no bundle to speak of — no such route, an
 * unexpected base URL, nothing bundlable — in which case the caller simply
 * proceeds and every mesh is fetched on its own.
 *
 * Never throws: a failure here is a slower load, not a failed one.
 */
export async function prefetchMeshBundle({
  baseUrl,
  assetRevisionKey,
  derived,
  isCancelled = () => false,
  onProgress,
}: PrefetchOptions): Promise<MeshBundleResult | null> {
  const base = meshBundleBase(baseUrl);
  if (!base) return null;

  const query = new URLSearchParams();
  if (derived) query.set('derived', '1');
  const suffix = query.size > 0 ? `?${query}` : '';

  let manifest: Manifest;
  try {
    const response = await fetch(`${base}${suffix}`);
    if (!response.ok) return null;
    manifest = (await response.json()) as Manifest;
  } catch {
    return null;
  }
  if (isCancelled()) return null;
  if (!manifest?.chunks?.length) return null;

  const result: MeshBundleResult = {
    chunks: manifest.chunks.length, chunksLoaded: 0, primed: 0,
    bytes: manifest.bytes ?? 0,
  };

  // The revision rides on the chunk URL, not the manifest's: it is what makes
  // the response immutable, so a re-selected run is served from the browser
  // cache without touching the network.
  const chunkUrl = (index: number): string => {
    const params = new URLSearchParams(query);
    if (assetRevisionKey) params.set('rev', assetRevisionKey);
    const tail = params.size > 0 ? `?${params}` : '';
    return `${base}/${index}${tail}`;
  };

  const report = () => onProgress?.({ chunks: result.chunks, chunksLoaded: result.chunksLoaded,
    files: manifest.files, primed: result.primed });
  report();

  if (workersAvailable()) {
    // Off the page: each chunk goes to a loader worker, which fetches and parses it and sends back one mesh at a
    // time, so receiving any one is a short task. The pool parses chunks in parallel.
    const controller = new AbortController();
    const stop = window.setInterval(() => { if (isCancelled()) controller.abort(); }, 100);
    try {
      await Promise.all(manifest.chunks.map(async (chunk) => {
        try {
          await requestFromWorker({ kind: 'obj-chunk', url: chunkUrl(chunk.index) }, (reply) => {
            if (reply.type !== 'objs' || isCancelled()) return;
            for (const entry of reply.entries) {
              if (primeGeometryTemplateObject(meshTemplateUrl(baseUrl, entry.name, assetRevisionKey),
                                              objectFromSerialized(entry.object))) {
                result.primed += 1;
              }
            }
            report();
          }, controller.signal);
          result.chunksLoaded += 1;
          report();
        } catch {
          // This chunk's meshes go per-file.
        }
      }));
    } finally {
      window.clearInterval(stop);
    }
    return isCancelled() ? null : result;
  }

  const queue = manifest.chunks.map((chunk) => chunk.index);
  const worker = async (): Promise<void> => {
    for (;;) {
      const index = queue.shift();
      if (index === undefined || isCancelled()) return;
      let buffer: ArrayBuffer;
      try {
        const response = await fetch(chunkUrl(index));
        if (!response.ok) continue;
        buffer = await response.arrayBuffer();
      } catch {
        continue;                 // this chunk's meshes go per-file
      }
      if (isCancelled()) return;

      let since = 0;
      for (const entry of readTarEntries(buffer)) {
        try {
          if (primeGeometryTemplate(
                meshTemplateUrl(baseUrl, entry.name, assetRevisionKey),
                entry.name, entry.data)) {
            result.primed += 1;
          }
        } catch {
          // A mesh that will not parse is one the per-file path would also
          // fail on; leave it out and let that path report it.
        }
        since += 1;
        if (since >= PARSE_YIELD_EVERY) {
          since = 0;
          await yieldToEventLoop();
          if (isCancelled()) return;
        }
      }
      result.chunksLoaded += 1;
      report();
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(MAX_CONCURRENT_CHUNKS, queue.length) },
               () => worker()),
  );
  return result;
}
