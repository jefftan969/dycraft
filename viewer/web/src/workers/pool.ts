// The page's side of the loader workers (`loader.worker.ts`): a small pool, one request per call.
//
// Sized to the machine, so a scene's hundreds of meshes parse in parallel. Each request goes to the worker with the
// fewest in flight. Where workers cannot start (an old browser, or a script under node), `available()` is false and
// callers keep their in-page path.

import type { LoaderReply, LoaderRequest } from "./loader-protocol";

/** A request as the page writes it: any request but an abort, without the id the pool assigns. */
type RequestBody = LoaderRequest extends infer R ? R extends { kind: "abort" } ? never : Omit<R, "id"> : never;

interface Pending {
  worker: PooledWorker;
  onReply: (reply: LoaderReply) => void;
  resolve: () => void;
  reject: (error: Error) => void;
}

interface PooledWorker {
  worker: Worker;
  inFlight: number;
}

let pool: PooledWorker[] | null = null;
let broken = false;
let nextId = 1;
const pending = new Map<number, Pending>();

function size(): number {
  const cores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;
  return Math.max(2, Math.min(6, cores - 1));
}

function start(): PooledWorker[] | null {
  if (pool || broken) return pool;
  if (typeof Worker === "undefined") {
    broken = true;
    return null;
  }
  try {
    pool = Array.from({ length: size() }, () => {
      const pooled: PooledWorker = {
        worker: new Worker(new URL("./loader.worker.ts", import.meta.url), { type: "module" }),
        inFlight: 0,
      };
      pooled.worker.onmessage = (event: MessageEvent<LoaderReply>) => {
        const message = event.data;
        const entry = pending.get(message.id);
        if (!entry) return;
        if (message.type === "done" || message.type === "error") {
          pending.delete(message.id);
          entry.worker.inFlight -= 1;
          if (message.type === "done") entry.resolve();
          else entry.reject(new Error(message.message));
          return;
        }
        entry.onReply(message);
      };
      pooled.worker.onerror = (event) => {
        // A worker that fails to load: everything it holds fails, and callers fall back to the page.
        event.preventDefault();
        broken = true;
        for (const [id, entry] of pending) {
          if (entry.worker !== pooled) continue;
          pending.delete(id);
          entry.reject(new Error(`loader worker failed: ${event.message || "could not start"}`));
        }
      };
      return pooled;
    });
  } catch {
    broken = true;
    pool = null;
  }
  return pool;
}

/** Whether requests can go to workers here. */
export function available(): boolean {
  return start() !== null && !broken;
}

/**
 * Send one request, handing each intermediate reply to `onReply`. Resolves on the worker's `done`, rejects on its
 * error or on `signal` (which also stops the worker's fetch).
 */
export function request(
  body: RequestBody,
  onReply: (reply: LoaderReply) => void,
  signal?: AbortSignal,
): Promise<void> {
  const workers = start();
  if (!workers || broken) return Promise.reject(new Error("loader workers are unavailable"));
  if (signal?.aborted) return Promise.reject(new DOMException("aborted", "AbortError"));
  const worker = workers.reduce((a, b) => (b.inFlight < a.inFlight ? b : a));
  const id = nextId;
  nextId += 1;
  return new Promise<void>((resolve, reject) => {
    pending.set(id, { worker, onReply, resolve, reject });
    worker.inFlight += 1;
    signal?.addEventListener("abort", () => {
      if (!pending.delete(id)) return;
      worker.inFlight -= 1;
      worker.worker.postMessage({ id, kind: "abort" } satisfies LoaderRequest);
      reject(new DOMException("aborted", "AbortError"));
    }, { once: true });
    worker.worker.postMessage({ ...body, id } as LoaderRequest);
  });
}

/** Progress of one download: bytes so far, and the declared total when the server sent one. */
export type ByteProgress = (received: number, total: number | null) => void;

/** A JSON payload, parsed off the page and received in slices (`loader-protocol.ts`). */
export async function fetchJson<T>(url: string, onProgress?: ByteProgress, signal?: AbortSignal): Promise<T> {
  let value: unknown = null;
  await request({ kind: "json", url }, (message) => {
    if (message.type === "progress") onProgress?.(message.received, message.total);
    else if (message.type === "json-head") value = message.value;
    else if (message.type === "json-slice") {
      const target = (value as Record<string, unknown[]>)[message.key];
      for (const item of message.items) target.push(item);
    }
  }, signal);
  return value as T;
}
