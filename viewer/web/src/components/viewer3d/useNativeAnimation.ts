// Load a native clip's evaluated geometry, and pose it on every frame.
//
// Two jobs, kept apart:
//
//   * **Load** (keyed on the run) pulls the track, then the meshes the first picture needs: every mesh an object
//     keeps for the whole clip, and the current frame's deforming meshes. Then it builds one Mesh per object. The
//     rest of the deforming meshes follow the playhead (`native-stream.ts`).
//   * **Scrub** (keyed on the frame) writes matrices into those Meshes and moves the loading window.
//
// The staged model is hidden only once there is something to replace it with, and never between frames. `ready`
// tells the transport bar the first picture is up; `isFrameReady` lets playback wait for a frame's meshes, as a
// video player buffers.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as THREE from "three";

import {
  fetchNativeAtlas, fetchNativeMesh, fetchNativeTrack, fetchViewerConfig, nativeMeshUrl, nativePositionsUrl,
} from "@/lib/api";
import { clearLoading, setLoading } from "@/lib/loading";
import { available as workersAvailable, request as requestFromWorker } from "@/workers/pool";
import type { LoaderReply, Transfer } from "@/workers/loader-protocol";
import {
  applyStudioShading,
  restoreStudioShading,
  withMeshMaterials,
} from "./scene-presentation";
import {
  applyNativeFrame,
  buildNativeScene,
  decodeNativeAtlas,
  decodeNativeMesh,
  decodeNativeTrack,
  disposeNativeScene,
  nativeGeometry,
  nativeGeometryFromArrays,
  nativeMaterial,
  type NativeAtlas,
  type NativeScene,
  type NativeTrack,
} from "./native-track";
import { OWNED_ELSEWHERE_USER_DATA_KEY } from "./user-data-keys";
import { NativeStream } from "./native-stream";

/**
 * Meshes in flight at once: enough to keep every loader worker busy, since building one (its creased normals) costs
 * more than fetching it.
 */
const MESH_CONCURRENCY = 12;

export interface NativeAnimationState {
  /** Loading progress or the reason there is nothing to draw; empty when live. */
  notes: string[];
  /** True once the first picture is up and a frame can be shown. */
  ready: boolean;
  /** Whether every mesh a frame draws is built; playback waits on it. True when there is no clip. */
  isFrameReady: (frame: number) => boolean;
  /** How many frames from `frame` on are ready, counting at most `horizon`; all of them when there is no clip. */
  readyAhead: (frame: number, horizon: number) => number;
}

const ALWAYS_READY = () => true;
const ALL_AHEAD = (_frame: number, horizon: number) => horizon;
const IDLE: NativeAnimationState = { notes: [], ready: false, isFrameReady: ALWAYS_READY, readyAhead: ALL_AHEAD };

/** "2nd", "3rd", "4th" … for the tray. */
function ordinal(n: number): string {
  return n === 2 ? "2nd" : n === 3 ? "3rd" : `${n}th`;
}

/** How often the tray's animation row may update. */
const REPORT_MS = 100;

/**
 * One mesh, fetched and built in a loader worker; in the page when workers cannot start.
 *
 * A deforming object's mesh comes as its positions only, on the triangles of the object's first mesh, which the
 * worker fetches once and keeps (every frame of an object has the same triangles; the server says so, per mesh, and
 * the worker falls back to the whole mesh when one does not). That is about a third of the bytes, for a browser on
 * another machine.
 */
function meshLoader(runId: string, track: NativeTrack, onTransfers: (transfers: Transfer[], digest: string) => void) {
  // Each deforming object's meshes, by digest, to the object's first one: where its triangles come from.
  const firstOf = new Map<string, string>();
  for (const object of track.objects) {
    if (!Array.isArray(object.mesh)) continue;
    const first = object.mesh.find((index) => index >= 0);
    if (first === undefined) continue;
    for (const index of object.mesh) {
      if (index >= 0 && !firstOf.has(track.meshes[index])) firstOf.set(track.meshes[index], track.meshes[first]);
    }
  }
  return async (digest: string, signal: AbortSignal): Promise<THREE.BufferGeometry> => {
    if (workersAvailable()) {
      let geometry: THREE.BufferGeometry | null = null;
      const onReply = (reply: LoaderReply) => {
        if (reply.type === "native-mesh") {
          geometry = nativeGeometryFromArrays(reply.position, reply.normal, reply.box, reply.sphere, reply.index);
          if (reply.transfers?.length) onTransfers(reply.transfers, digest);
        }
      };
      const first = firstOf.get(digest);
      if (first !== undefined && first !== digest) {
        await requestFromWorker({
          kind: "native-frame", url: nativePositionsUrl(runId, digest), meshUrl: nativeMeshUrl(runId, digest),
          topologyUrl: nativeMeshUrl(runId, first), topologyKey: first,
        }, onReply, signal);
      } else {
        await requestFromWorker({ kind: "native-mesh", url: nativeMeshUrl(runId, digest) }, onReply, signal);
      }
      if (!geometry) throw new Error("no mesh came back");
      return geometry;
    }
    return nativeGeometry(decodeNativeMesh(await fetchNativeMesh(runId, digest, signal)));
  };
}

export function useNativeAnimation(
  runId: string | null,
  version: string | undefined,
  frame: number,
  links: Map<string, THREE.Group> | null,
  invalidate: () => void,
  /**
   * Whether the studio look is on — `shouldApplyStudioShading(renderOptions)`.
   *
   * The clip's meshes hang inside the robot group, so `SceneCanvas`'s studio
   * pass *traverses* them, but it filters on `articraftVisual`, which only
   * URDF-loaded visuals carry. The clip therefore drew with the env map at
   * `GLTFLoader`'s default intensity of 1.0 while the staged model around it
   * was clamped to 0.62, and without the form-shading term at all — brighter
   * and flatter than the scene it animates. Applied here rather than by
   * tagging the meshes, because `articraftVisual` also drives part picking,
   * the legend, and the segmentation modes, none of which have an entry for an
   * evaluated mesh.
   */
  studioShading = false,
  /** The playback rate: on a slow link it decides how many of a deforming object's frames are fetched. */
  rate = 1,
  /** Whether the clip is playing: only a stopped playhead's own frame is fetched exactly on a slow link. */
  playing = false,
): NativeAnimationState {
  const [state, setState] = useState<NativeAnimationState>(IDLE);
  // The built scene, read by the frame effect below. A ref rather than state
  // because scrubbing must not re-run the loader, and because the frame effect
  // has to see the scene the load effect built in the same commit.
  const scene = useRef<NativeScene | null>(null);
  const stream = useRef<NativeStream | null>(null);
  const reportRef = useRef<(() => void) | null>(null);
  const draw = useRef(invalidate);
  const rateRef = useRef(rate);
  rateRef.current = rate;
  const playingRef = useRef(playing);
  playingRef.current = playing;
  draw.current = invalidate;
  // Read once by the loader, so the first drawn frame is the playhead's rather
  // than frame 0; driven per frame by the effect at the bottom.
  const frameRef = useRef(frame);
  frameRef.current = frame;
  // The model's link groups. The model is reloaded in place when its shader colors arrive, which hands over new
  // link groups; the clip is moved to them (the effect below) rather than loaded again. So the load reads them
  // through this ref and is keyed only on whether there are any.
  const linksRef = useRef(links);
  linksRef.current = links;
  const hasLinks = Boolean(links);
  // The `world` link the clip is mounted beside and hides, with the visibility to give back, and the links its
  // materials were taken from.
  const mounted = useRef<{ world: THREE.Object3D; visible: boolean } | null>(null);
  const materialLinks = useRef<Map<string, THREE.Group> | null>(null);
  const studioRef = useRef(studioShading);
  studioRef.current = studioShading;

  useEffect(() => {
    const links = linksRef.current;
    if (!runId || !version || !links) {
      setState(IDLE);
      return;
    }
    const world = links.get("world");
    const root = world?.parent;
    if (!world || !root) {
      setState({ notes: ["Native animation: scene root is unavailable"], ready: false, isFrameReady: ALWAYS_READY,
                 readyAhead: ALL_AHEAD });
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    const isFrameReady = (f: number) => stream.current?.isReady(f) ?? false;
    const readyAhead = (f: number, horizon: number) => stream.current?.readyAhead(f, horizon) ?? 0;
    // Progress lives in the tray, so the notes carry only what is wrong.
    const notReady = (notes: string[]) => setState({ notes, ready: false, isFrameReady: ALWAYS_READY,
                                                     readyAhead: ALL_AHEAD });
    notReady([]);
    setLoading("animation", "Animation", "motion track");

    // What the tray says while the clip, or the window ahead of the playhead, fills; once it is full, nothing.
    // At most every `REPORT_MS`: meshes arrive far faster than anyone reads.
    let reportTimer: number | undefined;
    const reportNow = () => {
      reportTimer = undefined;
      const live = stream.current;
      if (!live || cancelled) return;
      const f = frameRef.current;
      const fps = live.fps;
      if (live.loadsWholeClip) {
        const { loaded, total } = live.progress();
        if (loaded >= total) clearLoading("animation");
        else setLoading("animation", "Animation", `clip meshes ${loaded}/${total}`, loaded / Math.max(1, total));
        return;
      }
      const ahead = Math.min(live.aheadFrames(), live.nFrames - f);
      const ready = live.readyAhead(f);
      // Said while it buffers, which is when someone wonders why: on a slow link the characters move in steps.
      const thinned = live.stride > 1 ? ` · every ${ordinal(live.stride)} frame on this connection` : "";
      if (ready >= ahead) clearLoading("animation");
      else setLoading("animation", "Animation",
                      `buffered ${(ready / fps).toFixed(1)} of ${(ahead / fps).toFixed(0)} s${thinned}`,
                      ready / Math.max(1, ahead));
    };
    const reportBuffer = () => {
      if (reportTimer === undefined) reportTimer = window.setTimeout(reportNow, REPORT_MS);
    };
    reportRef.current = reportBuffer;

    void (async () => {
      try {
        const track = decodeNativeTrack(
          await fetchNativeTrack(runId, controller.signal));
        if (cancelled) return;
        if (track.objects.length === 0) {
          // Nothing to put in the staged model's place, so leave it standing.
          // Hiding it here would trade a scene for an empty viewport and a
          // playhead that still moves, which is not an improvement on either.
          notReady(["Native animation: the clip evaluates no drawable geometry"]);
          clearLoading("animation");
          return;
        }
        const { clip_budget_mb: budgetMb } = await fetchViewerConfig();
        if (cancelled) return;
        const live = new NativeStream({
          track,
          // Each reply's downloads as the network timed them: the link `native-stream.ts` sizes its stride by.
          load: meshLoader(runId, track, (transfers, digest) => {
            stream.current?.noteTransfers(transfers, stream.current.isDeforming(digest));
          }),
          budgetBytes: budgetMb * 1e6,
          concurrency: MESH_CONCURRENCY,
          displayed: () => new Set((scene.current?.nodes ?? []).map((node) => node.geometry)),
          onChange: (digest) => {
            const built = scene.current;
            if (!built || cancelled) return;
            // Redrawn only for a mesh the current frame shows; the rest wait for their frame.
            if (digest !== null && stream.current?.digestsAt(frameRef.current).includes(digest)) {
              applyNativeFrame(built, frameRef.current, stream.current?.stride);
              draw.current();
            }
            reportBuffer();
          },
        });
        stream.current = live;
        live.setRate(rateRef.current);
        live.setPlaying(playingRef.current);
        await live.loadFirst(frameRef.current, (done, total) => {
          if (cancelled) return;
          setLoading("animation", "Animation", `first frame: meshes ${done}/${total}`, done / Math.max(1, total));
        });
        if (cancelled) return;
        if (live.geometries.size === 0) {
          notReady(live.noteFailures.map((f) => `Native animation: ${f}`));
          clearLoading("animation");
          return;
        }
        // Fetched after the meshes, and never allowed to fail the clip: the
        // atlas depends on a bake that lands minutes later on its own
        // schedule, so "no atlas yet" is the ordinary case and its answer is
        // the flat-color appearance this path draws without one.
        let atlas: NativeAtlas | null = null;
        try {
          const decoded = decodeNativeAtlas(
            await fetchNativeAtlas(runId, controller.signal));
          if (decoded.objects.length > 0) atlas = decoded;
        } catch {
          atlas = null;
        }
        if (cancelled) return;
        // The model as it is now: it may have been reloaded in place while the clip loaded.
        const current = linksRef.current ?? links;
        const currentWorld = current.get("world") ?? world;
        const built = buildNativeScene(track, live.geometries, current, atlas);
        built.group.userData[OWNED_ELSEWHERE_USER_DATA_KEY] = true;
        applyNativeFrame(built, frameRef.current, stream.current?.stride);
        (currentWorld.parent ?? root).add(built.group);
        // **Only now.** Hiding the staged model before its replacement exists
        // is what made the viewport go blank for the whole load.
        mounted.current = { world: currentWorld, visible: currentWorld.visible };
        currentWorld.visible = false;
        materialLinks.current = current;
        scene.current = built;
        setState({ notes: live.noteFailures.map((f) => `Native animation: ${f}`), ready: true, isFrameReady,
                   readyAhead });
        draw.current();
        live.setFrame(frameRef.current);
        reportBuffer();
      } catch (error) {
        if (cancelled || controller.signal.aborted) return;
        notReady([`Native animation: ${error instanceof Error ? error.message : String(error)}`]);
        clearLoading("animation");
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(reportTimer);
      clearLoading("animation");
      reportRef.current = null;
      if (mounted.current) mounted.current.world.visible = mounted.current.visible;
      mounted.current = null;
      materialLinks.current = null;
      const built = scene.current;
      scene.current = null;
      // The scene owns the stream's meshes once it exists; before that, the stream frees them itself.
      stream.current?.dispose(!built);
      stream.current = null;
      if (built) {
        built.group.removeFromParent();
        disposeNativeScene(built);
      }
      draw.current();
    };
    // `frame` is deliberately absent: it is read once through `frameRef` so the
    // first drawn frame is the playhead's, and driven from the effect below
    // afterwards. Listing it here is precisely the bug this replaced.
  }, [runId, version, hasLinks]);

  // The model reloaded in place (its shader colors arrived): move the clip to the new model, hide its `world`, and
  // take the objects' materials from its links again, so the clip wears the new colors. A layout effect, so the
  // move lands before the next frame is drawn.
  useLayoutEffect(() => {
    const built = scene.current;
    if (!built || !links || links === materialLinks.current) return;
    const world = links.get("world");
    const root = world?.parent;
    if (!world || !root) return;
    if (built.group.parent !== root) {
      built.group.removeFromParent();
      root.add(built.group);
    }
    if (mounted.current && mounted.current.world !== world) mounted.current.world.visible = mounted.current.visible;
    mounted.current = { world, visible: mounted.current?.visible ?? world.visible };
    world.visible = false;
    built.nodes.forEach((node, i) => {
      const previous = node.material;
      node.material = nativeMaterial(built.track.objects[i], links, Boolean(built.atlasUVs[i]));
      for (const material of Array.isArray(previous) ? previous : [previous]) material.dispose();
      if (studioRef.current) withMeshMaterials(node, applyStudioShading);
    });
    materialLinks.current = links;
    applyNativeFrame(built, frameRef.current, stream.current?.stride);
    draw.current();
  }, [links, state.ready]);

  // Keyed on the flag, and on `state.ready` so it also runs once the scene it
  // writes into exists: the load is async, so a pass that ran only on the
  // toggle would miss every clip that finished loading after it.
  useEffect(() => {
    const built = scene.current;
    if (!built) return;
    for (const node of built.nodes) {
      withMeshMaterials(node, (material) => {
        if (studioShading) applyStudioShading(material);
        else restoreStudioShading(material);
      });
    }
    draw.current();
  }, [studioShading, state.ready]);

  useEffect(() => {
    stream.current?.setRate(rate);
    stream.current?.setPlaying(playing);
  }, [rate, playing, state.ready]);

  // `state.ready` is a dependency so that the first frame after a load lands:
  // the effect is otherwise keyed only on `frame`, which does not change when
  // the scene it writes into comes into existence.
  useEffect(() => {
    if (!scene.current) return;
    stream.current?.setFrame(frame);
    applyNativeFrame(scene.current, frame, stream.current?.stride);
    draw.current();
    reportRef.current?.();
  }, [frame, state.ready]);

  return state;
}
