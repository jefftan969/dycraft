// Drive the viewport from a committed animation run.
//
// This is the animation-run counterpart to `preview-motion.ts`, and the two are
// mutually exclusive by construction: preview motion invents a joint sweep for
// an object that has no authored motion, while this replays motion that was
// authored, verified, and hashed. When a timeline is present the caller gates
// the preview sweep off — sweeping joints the animation also writes would just
// fight it.
//
// Playback is wall-clock, not frame-per-tick: at 24 fps on a 60 Hz display a
// tick-per-frame loop would run the clip 2.5x fast. Elapsed time maps to a
// frame index, so a dropped frame costs smoothness rather than duration — and
// a playback rate is then just a factor on that mapping.
//
// Slow motion holds each authored frame longer; it never interpolates between
// two of them. That is the honest reading of an evaluated clip: it is a dense
// sample grid, and what happens between frame 8 and frame 9 is not something
// the run committed to. At 0.25x on a 24 fps clip each frame is on screen for
// four display frames, which is exactly the resolution the verifiers checked.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type * as THREE from "three";

import { useNativeAnimation } from "./useNativeAnimation";

import type { AnimationTimeline } from "@/lib/types";
import type { UrdfSpec } from "./urdf-parser";
import {
  applyBodyPoses,
  clearBodyPoses,
  jointValuesAtFrame,
  resolvePlacementTargets,
} from "./animation-playback";

/** Whether the reader asked the system for less motion: then nothing starts playing by itself. */
function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
}

/** Wall-clock multipliers offered by the transport bar. */
export const PLAYBACK_RATES = [0.1, 0.25, 0.5, 1, 2, 3] as const;

/** The rate a session starts at: twice the film's speed, as the project page plays its films. */
export const DEFAULT_PLAYBACK_RATE = 2;

export interface AnimationPlaybackState {
  /** Current frame, always in `[0, nFrames)`; 0 when there is no timeline. */
  frame: number;
  playing: boolean;
  nFrames: number;
  fps: number;
  /** Wall-clock multiplier; 1 plays at the clip's authored fps. */
  rate: number;
  setRate: (rate: number) => void;
  setFrame: (frame: number) => void;
  step: (delta: number) => void;
  /** Move the playhead by `seconds` of clip, playing on if it was (the J and L keys). */
  seek: (seconds: number) => void;
  togglePlay: () => void;
  /** Whether playback starts again from the first frame at the last, rather than stopping there. */
  loop: boolean;
  setLoop: (loop: boolean) => void;
  /** Bodies the timeline moves that the loaded URDF has no placement joint for. */
  missing: string[];
  /** Native playback's loading progress, or why it has nothing to draw; empty when live. */
  notes: string[];
  /** The native clip's first picture is up (false when the run has no native clip). */
  nativeReady: boolean;
}

interface Options {
  /** The run whose native clip to load. */
  runId: string | null;
  timeline: AnimationTimeline | null;
  urdfSpec: UrdfSpec | null;
  jointNodes: Map<string, THREE.Object3D> | null;
  /** Link name -> group, for native playback. Null until meshes load. */
  linkNodes: Map<string, THREE.Group> | null;
  /**
   * The studio look is on — `shouldApplyStudioShading(renderOptions)`.
   *
   * A native clip builds its own meshes, so `SceneCanvas`'s studio pass never
   * reaches their materials; without this they render at the env map's full
   * intensity while the staged model around them is clamped, which is a clip
   * visibly brighter than its own scene.
   */
  studioShading: boolean;
  /** `useJointController`'s applier — commit so the sliders track playback. */
  applyJointValues: (
    values: Map<string, number>,
    options?: { commit?: boolean },
  ) => void;
  /** Ask the canvas to redraw; it renders on demand, not continuously. */
  invalidate: () => void;
}

/** Seconds of clip that must be ready before play starts, or resumes after a stall. */
const BUFFER_S = 2;

export function useAnimationPlayback({
  runId,
  timeline,
  urdfSpec,
  jointNodes,
  linkNodes,
  studioShading,
  applyJointValues,
  invalidate,
}: Options): AnimationPlaybackState {
  const [frame, setFrameState] = useState(0);
  const [playing, setPlaying] = useState(false);
  // Not reset when the timeline changes: a rate is how you are watching, not a
  // property of what you are watching. Pick 0.25x to study a contact, click
  // through three runs, and it should still be 0.25x.
  const [rate, setRate] = useState<number>(DEFAULT_PLAYBACK_RATE);
  const native = useNativeAnimation(
    runId, timeline?.native?.version, frame, linkNodes, invalidate,
    studioShading, rate, playing);
  // A native clip has nothing to pose until its evaluated geometry is resident.
  // Running the clock before then would advance the playhead over a viewport
  // that cannot follow it.
  const nativeLoading = Boolean(timeline?.native) && !native.ready;
  // Bumped by a seek, so a running loop restarts from the new playhead with a fresh clock.
  const [seekNonce, setSeekNonce] = useState(0);

  const nFrames = timeline?.n_frames ?? 0;
  const fps = timeline?.fps || 24;

  const { targets, missing } = useMemo(() => {
    if (!timeline || !urdfSpec || !jointNodes) {
      return { targets: [], missing: [] as string[] };
    }
    return resolvePlacementTargets(timeline, urdfSpec, jointNodes);
  }, [timeline, urdfSpec, jointNodes]);

  // A new run must not inherit the previous one's playhead: frame 40 of a
  // 61-frame clip is out of range on the next run's 31-frame one, and
  // `values[frame]` would silently be undefined — i.e. a frozen pose that
  // looks like a working scrub.
  //
  // A film opened plays by itself once it can draw, as the project page's films do (`autoplayRef`), unless the
  // reader asked for less motion; anything they do with the transport first wins.
  const autoplayRef = useRef(false);
  useEffect(() => {
    setFrameState(0);
    setPlaying(false);
    autoplayRef.current = Boolean(timeline) && !prefersReducedMotion();
  }, [timeline]);
  const drawable = timeline?.native ? native.ready : Boolean(jointNodes);
  useEffect(() => {
    if (!autoplayRef.current || !drawable || nFrames <= 1) return;
    autoplayRef.current = false;
    setPlaying(true);
  }, [drawable, nFrames]);

  // Not reset when the timeline changes, like the rate: whether to loop is how you are watching.
  const [loop, setLoop] = useState(true);
  const loopRef = useRef(loop);
  loopRef.current = loop;

  // Refs so the rAF loop below never has to be torn down and rebuilt on a
  // frame change — restarting it every frame would reset its own clock.
  const applyRef = useRef(applyJointValues);
  const invalidateRef = useRef(invalidate);
  useEffect(() => {
    applyRef.current = applyJointValues;
    invalidateRef.current = invalidate;
  }, [applyJointValues, invalidate]);

  // Pose on every frame change, and once on load so the viewport opens at
  // frame 0 of the animation rather than at the scene's rest pose.
  useEffect(() => {
    if (!timeline || !jointNodes) return;
    applyRef.current(jointValuesAtFrame(timeline, frame), { commit: true });
    applyBodyPoses(targets, frame);
    invalidateRef.current();
  }, [frame, jointNodes, targets, timeline]);

  // Selecting a different run unmounts nothing — the same graph may be reused
  // — so hand the bodies back at rest rather than leaving the last frame's
  // pose baked into a scene nobody is animating any more.
  useEffect(() => {
    return () => {
      clearBodyPoses(targets);
    };
  }, [targets]);

  const frameRef = useRef(frame);
  useEffect(() => {
    frameRef.current = frame;
  }, [frame]);

  // Read through refs so the loop below does not restart when the hook hands back new functions.
  const frameReadyRef = useRef(native.isFrameReady);
  frameReadyRef.current = native.isFrameReady;
  const readyAheadRef = useRef(native.readyAhead);
  readyAheadRef.current = native.readyAhead;

  useEffect(() => {
    if (!playing || nFrames <= 1 || nativeLoading) return;
    let raf = 0;
    let start = 0;
    let from = frameRef.current;
    // Buffered like a video: play starts, and resumes after a stall, only once `BUFFER_S` ahead (or the rest of the
    // clip) is ready, so a clip whose meshes load slower than it plays stops now and then rather than every frame.
    const want = (f: number) => Math.max(1, Math.min(Math.round(BUFFER_S * fps), nFrames - f));
    const buffered = (f: number) => readyAheadRef.current(f, want(f)) >= want(f);
    let buffering = !buffered(from);
    const tick = (now: number) => {
      if (buffering) {
        const f = frameRef.current;
        if (buffered(f)) {
          // Resume from the frame it held, with a fresh clock, rather than jumping ahead by the time spent waiting.
          buffering = false;
          from = f;
          start = now;
        }
        raf = requestAnimationFrame(tick);
        return;
      }
      if (!start) start = now;
      const advanced = Math.floor(((now - start) / 1000) * fps * rate);
      // Past the last frame: from the first again, or, with the loop off, stop on the last.
      if (!loopRef.current && from + advanced >= nFrames) {
        setFrameState(nFrames - 1);
        setPlaying(false);
        return;
      }
      const next = (from + advanced) % nFrames;
      if (next !== frameRef.current) {
        if (frameReadyRef.current(next)) setFrameState(next);
        else buffering = true;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // `frame` is deliberately absent from the deps: the loop reads the playhead
    // it started from once, through `frameRef`, and owns it until it stops.
    // Listing it would tear the loop down and reset its clock every frame.
    //
    // `rate` IS listed: changing it mid-play restarts the loop from the current
    // playhead with a fresh clock, which is what makes the new rate take effect
    // from now rather than retroactively rescaling the elapsed time.
    //
    // A native clip runs on this same loop, waiting on `isFrameReady` while
    // its meshes load around the playhead (`native-stream.ts`).
  }, [playing, nFrames, fps, rate, nativeLoading, seekNonce]);

  const setFrame = useCallback(
    (value: number) => {
      if (nFrames <= 0) return;
      autoplayRef.current = false;
      setPlaying(false);
      setFrameState(Math.max(0, Math.min(nFrames - 1, Math.round(value))));
    },
    [nFrames],
  );

  const step = useCallback(
    (delta: number) => {
      if (nFrames <= 0) return;
      autoplayRef.current = false;
      setPlaying(false);
      setFrameState((current) =>
        Math.max(0, Math.min(nFrames - 1, current + delta)),
      );
    },
    [nFrames],
  );

  const seek = useCallback(
    (seconds: number) => {
      if (nFrames <= 0) return;
      const next = Math.max(0, Math.min(nFrames - 1, frameRef.current + Math.round(seconds * fps)));
      frameRef.current = next;
      setFrameState(next);
      setSeekNonce((n) => n + 1);
    },
    [nFrames, fps],
  );

  const togglePlay = useCallback(() => {
    if (nFrames <= 1) return;
    autoplayRef.current = false;
    setPlaying((current) => {
      // Restarting from the last frame would play a zero-length clip.
      if (!current && frameRef.current >= nFrames - 1) setFrameState(0);
      return !current;
    });
  }, [nFrames]);

  return { frame, playing, nFrames, fps, rate, setRate, setFrame, step, seek,
           togglePlay, loop, setLoop, missing, notes: native.notes, nativeReady: Boolean(timeline?.native) && native.ready };
}
