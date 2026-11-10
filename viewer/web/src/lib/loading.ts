// What is still loading for the selected run, one row per item, for the viewport's loading tray.
//
// Each loader owns its row: it sets it while it works and clears it when it finishes, fails, or is abandoned. The
// rows are shown in a fixed order, so the tray does not reshuffle as items come and go.

import { useSyncExternalStore } from "react";

export type LoadingKey = "preparing" | "geometry" | "colours" | "animation" | "trajectory" | "timeline";

export const LOADING_ORDER: readonly LoadingKey[] = [
  "preparing", "geometry", "colours", "animation", "trajectory", "timeline",
];

export interface LoadingRow {
  key: LoadingKey;
  label: string;
  /** What it is doing now: "chunks 3/8", "12 of 47 MB", "buffering frames 120-216". */
  detail: string;
  /** 0..1 when the amount of work is known. */
  fraction?: number;
}

let rows: LoadingRow[] = [];
const listeners = new Set<() => void>();

function publish(next: LoadingRow[]): void {
  rows = next;
  for (const listener of listeners) listener();
}

export function setLoading(key: LoadingKey, label: string, detail: string, fraction?: number): void {
  const row: LoadingRow = { key, label, detail, fraction };
  const existing = rows.find((r) => r.key === key);
  if (existing && existing.label === label && existing.detail === detail && existing.fraction === fraction) return;
  publish([...rows.filter((r) => r.key !== key), row]
    .sort((a, b) => LOADING_ORDER.indexOf(a.key) - LOADING_ORDER.indexOf(b.key)));
}

export function clearLoading(key: LoadingKey): void {
  if (rows.some((r) => r.key === key)) publish(rows.filter((r) => r.key !== key));
}

export function useLoadingRows(): LoadingRow[] {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => rows,
    () => rows,
  );
}

/** "12 of 47 MB", or "12 MB" when the total is unknown or the body arrives compressed. */
export function describeBytes(received: number, total: number | null): { detail: string; fraction?: number } {
  const mb = (n: number) => (n >= 10e6 ? Math.round(n / 1e6) : Math.round(n / 1e5) / 10);
  if (total && received <= total) return { detail: `${mb(received)} of ${mb(total)} MB`, fraction: received / total };
  return { detail: `${mb(received)} MB` };
}
