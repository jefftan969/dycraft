import { useEffect, useRef, useState } from 'react';

import { encodeRunId } from '@/lib/url';

/**
 * Track the run's baked-atlas viewer assets, asking the server to build them.
 *
 * The browser cannot evaluate a `Material(shader=...)` node graph, so the
 * server samples it: a packed atlas is unwrapped on the real mesh, the original
 * shader is evaluated at every surface point, and the result is exported as a
 * per-part GLB. That asset is within a few tenths of a level (out of 255) of
 * `blender_renderer.render()` — closer to the canonical Cycles image than the
 * live Blender stream, which pays more than that to JPEG alone — and the
 * browser renders it locally, so orbiting, articulation, and Preview Motion are
 * as smooth as they are for any other mesh.
 *
 * The bake is a one-time cost per IR (15–95 s) and the result is cached by
 * content hash, so this is normally one round trip that answers `ready`. A
 * **scene** run needs one asset per catalog object it instances, and reports
 * `ready` only once every one of them has landed — `objects` / `objectsReady`
 * carry the progress meanwhile.
 * Until it does, the caller keeps the existing preview appearance — the
 * contract on failure is never to show an approximation of a shader without
 * saying so.
 */

export type DerivedStatus =
  | 'idle' | 'absent' | 'building' | 'ready' | 'partial' | 'unavailable' | 'error';

export interface DerivedAssetsState {
  status: DerivedStatus;
  /**
   * True once the URDF may safely be re-fetched with `?derived=1`.
   *
   * Also true for `partial`: a scene bakes one asset per catalog object, and
   * when some succeed and the rest permanently fail the derived URDF is a
   * *mixture* — real materials for what baked, staged meshes for what did not.
   * That is strictly better than drawing every object as a gray placeholder,
   * and `useUrdfLoader` decides per visual which material to keep, so both
   * halves draw correctly.
   */
  ready: boolean;
  /** Why there is no asset, or why one object of several has none. */
  reason: string | null;
  parts: number | null;
  bytes: number | null;
  warnings: string[];
  /** Objects this run needs baked, and how many are ready. Scene runs have
   *  several; an object run has one. Null before the first answer. */
  objects: number | null;
  objectsReady: number | null;
}

const IDLE: DerivedAssetsState = {
  status: 'idle', ready: false, reason: null, parts: null, bytes: null,
  warnings: [], objects: null, objectsReady: null,
};

/** While a bake is running. Long enough not to hammer a busy GPU box. */
const POLL_MS = 2000;

interface Payload {
  status?: string;
  reason?: string;
  parts?: number;
  bytes?: number;
  warnings?: string[];
  objects?: number;
  objects_ready?: number;
}

export function useDerivedAssets(
  runId: string | null,
  enabled: boolean,
): DerivedAssetsState {
  const [state, setState] = useState<DerivedAssetsState>(IDLE);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled || !runId) {
      setState(IDLE);
      return;
    }
    let cancelled = false;
    const base = import.meta.env.VITE_API_BASE ?? '';
    const url = `${base}/api/runs/${encodeRunId(runId)}/derived`;

    const poll = async (build: boolean) => {
      if (cancelled) return;
      try {
        const response = await fetch(build ? url : `${url}?build=0`, {
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const payload = (await response.json()) as Payload;
        if (cancelled) return;
        const status = (payload.status ?? 'error') as DerivedStatus;
        setState({
          status,
          ready: status === 'ready' || status === 'partial',
          reason: payload.reason ?? null,
          parts: payload.parts ?? null,
          bytes: payload.bytes ?? null,
          warnings: payload.warnings ?? [],
          objects: payload.objects ?? null,
          objectsReady: payload.objects_ready ?? null,
        });
        // Only `building` is worth waiting on. `unavailable`, `error`, and
        // `partial` are all settled — retrying any of them would be a loop
        // that reports the same thing forever. (`absent` is only reachable
        // from a pure query, which the first poll is not.)
        if (status === 'building' || status === 'absent') {
          timerRef.current = window.setTimeout(() => void poll(false), POLL_MS);
        }
      } catch (error) {
        if (cancelled) return;
        setState({
          ...IDLE, status: 'error',
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    };

    // The first call is the one allowed to start a bake; every poll after it
    // only observes, so a component that remounts cannot queue a second.
    void poll(true);

    return () => {
      cancelled = true;
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [enabled, runId]);

  return state;
}
