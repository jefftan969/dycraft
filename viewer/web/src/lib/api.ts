// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
// Thin fetch layer over dycraft's viewer API.
//
// In `--dev` the SPA is served by vite on :5173 while the API stays on :8770, so
// the base URL is configurable via VITE_API_BASE; in the built app both are on
// the same origin and the base is empty.

import type { FilmCameraTrack } from "@/components/viewer3d/film-camera";
import type {
  AgentTimeline,
  AnimationTimeline,
  PrepareStatus,
  RunDetail,
  RunSummary,
  TextFile,
  Trajectory,
} from "@/lib/types";

import type { Cutaway } from "@/components/viewer3d/cutaway";
import { encodeRunId } from "@/lib/url";
import { available as workersAvailable, fetchJson, type ByteProgress } from "@/workers/pool";

const API_BASE: string = import.meta.env.VITE_API_BASE ?? "";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // Non-JSON error body — the status line is all we have.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

/**
 * A large JSON payload, parsed in a loader worker and received in slices so the page never stalls on it
 * (`workers/loader-protocol.ts`); in the page when workers cannot start.
 */
async function getLargeJson<T>(path: string, onProgress?: ByteProgress, signal?: AbortSignal): Promise<T> {
  if (workersAvailable()) {
    try {
      return await fetchJson<T>(`${API_BASE}${path}`, onProgress, signal);
    } catch (error) {
      // A worker that failed to start falls back; a failed request or an abort is the answer.
      if (signal?.aborted || workersAvailable()) throw error;
    }
  }
  return getJson<T>(path);
}

/**
 * A run id is a path relative to `runs/`, so its separators must survive into
 * the URL for the server's `{run_id:path}` param to see them. Encoding the
 * whole id would turn `/` into `%2F` and the route would no longer match, so
 * each segment is encoded on its own.
 */
export async function fetchRuns(): Promise<RunSummary[]> {
  const data = await getJson<{ runs: RunSummary[] }>("/api/runs");
  return data.runs;
}

/** Whether a run is ready to open; asking starts preparing an installed film that is not. */
export function fetchPrepare(runId: string): Promise<PrepareStatus> {
  return getJson<PrepareStatus>(`/api/runs/${encodeRunId(runId)}/prepare`);
}

/** A native film's walls and ceilings, for the free camera's cutaway (`viewer/cutaway.py`). */
export function fetchCutaway(runId: string): Promise<Cutaway> {
  return getJson<Cutaway>(`/api/runs/${encodeRunId(runId)}/cutaway`);
}

export function fetchRun(runId: string): Promise<RunDetail> {
  return getJson<RunDetail>(`/api/runs/${encodeRunId(runId)}`);
}

export function fetchTrajectory(runId: string, onProgress?: ByteProgress, signal?: AbortSignal): Promise<Trajectory> {
  return getLargeJson<Trajectory>(`/api/runs/${encodeRunId(runId)}/trajectory`, onProgress, signal);
}

/**
 * Per-frame motion for an animation run. 404s on every other kind of run, so
 * only ask once `ir.kind === "animation"` — it is the one dense payload the
 * API serves and there is no reason for a scene run to pay for the round trip.
 */
export function fetchAnimation(runId: string): Promise<AnimationTimeline> {
  return getJson<AnimationTimeline>(
    `/api/runs/${encodeRunId(runId)}/animation`,
  );
}

/** A storyboard keyframe (`viewer/animation.py` `storyboard_keys`): `frame` is the film's own frame. */
export interface StoryboardKey {
  name: string;
  frame: number;
  camera: string | null;
  index: number;
}

/**
 * A film's storyboard keyframes, in frame order, with the film's first frame (its manifest's), which puts them on the
 * playhead's frames; no keyframes for a film without a storyboard.
 */
export function fetchStoryboard(runId: string): Promise<{ frame_start: number; keys: StoryboardKey[] }> {
  return getJson<{ frame_start: number; keys: StoryboardKey[] }>(`/api/runs/${encodeRunId(runId)}/storyboard`);
}

/** Keyframe `index`'s rendered view, small. */
export function storyboardThumbUrl(runId: string, index: number): string {
  return `${API_BASE}/api/runs/${encodeRunId(runId)}/storyboard/${index}.jpg`;
}

/** A film's camera at every frame, which the film view follows (`viewer/animation.py` `film_cameras`). */
export function fetchFilmCameras(runId: string): Promise<FilmCameraTrack> {
  return getJson<FilmCameraTrack>(`/api/runs/${encodeRunId(runId)}/film-cameras`);
}

/** One agent run's own actions, in lanes: the inside of a single run. */
export function fetchAgentTimeline(
  runId: string, onProgress?: ByteProgress, signal?: AbortSignal,
): Promise<AgentTimeline> {
  return getLargeJson<AgentTimeline>(`/api/runs/${encodeRunId(runId)}/agent_timeline`, onProgress, signal);
}

export function fetchTextFile(runId: string, path: string): Promise<TextFile> {
  return getJson<TextFile>(`/api/runs/${encodeRunId(runId)}/text/${path}`);
}

/**
 * Base URL the three.js loader resolves geometry against. It appends
 * `/model.urdf` for the robot and each URDF `<mesh filename>` verbatim, both of
 * which this route answers (see viewer/app.py).
 */
export function runFileBase(runId: string): string {
  return `${API_BASE}/api/runs/${encodeRunId(runId)}/files`;
}

export function runFileUrl(runId: string, path: string): string {
  return `${runFileBase(runId)}/${path}`;
}

async function fetchBinary(path: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) throw new Error(await response.text());
  return response.arrayBuffer();
}

/**
 * A native clip's whole motion track: every object, every frame, no geometry.
 *
 * One request for the clip rather than one per frame — see
 * `viewer/native_animation.py`. The geometry it refers to comes from
 * `fetchNativeMesh`, once per distinct mesh.
 */
export function fetchNativeTrack(runId: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  return fetchBinary(`/api/runs/${encodeRunId(runId)}/native-track`, signal);
}

/**
 * Baked atlas UVs, per object, for a native clip.
 *
 * Separate from the track because it depends on a bake that lands minutes
 * later and on its own schedule, and because a viewer with baked materials
 * switched off should not pay for it. An empty payload is a normal answer.
 */
export function fetchNativeAtlas(runId: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  return fetchBinary(`/api/runs/${encodeRunId(runId)}/native-atlas`, signal);
}

/**
 * One evaluated mesh by the digest of its own arrays.
 *
 * Served `immutable`, so the second visit to a run costs no bytes at all: the
 * URL cannot outlive the content it names.
 */
export function fetchNativeMesh(runId: string, digest: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  return fetchBinary(`/api/runs/${encodeRunId(runId)}/native-mesh/${digest}`, signal);
}

/** Where `fetchNativeMesh` reads from, for a loader worker to fetch itself. */
export function nativeMeshUrl(runId: string, digest: string): string {
  return `${API_BASE}/api/runs/${encodeRunId(runId)}/native-mesh/${digest}`;
}

/** One mesh's vertices only, for a deforming object's frames (`viewer/native_animation.py`). */
export function nativePositionsUrl(runId: string, digest: string): string {
  return `${API_BASE}/api/runs/${encodeRunId(runId)}/native-positions/${digest}`;
}

/** Settings the server's operator chose (`viewer/app.py`, `/api/config`). */
export interface ViewerConfig {
  /** Most built native-clip geometry to keep in the tab, in MB (`DYCRAFT_VIEWER_CLIP_BUDGET_MB`). */
  clip_budget_mb: number;
}

let config: Promise<ViewerConfig> | null = null;

/** Read once per page; an older server with no such route gets the defaults. */
export function fetchViewerConfig(): Promise<ViewerConfig> {
  config ??= getJson<ViewerConfig>("/api/config").catch(() => ({ clip_budget_mb: 1024 }));
  return config;
}
