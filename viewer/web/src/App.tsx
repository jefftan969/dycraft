// dycraft run viewer shell: run list | 3D viewport | inspector.
//
// Replaces Articraft's `ViewerShell`, which is built around their curated
// record store (staging, promotion, ratings, dataset collections). dycraft's
// `runs/` is an execution log, so the left pane is a plain reverse-chronological
// list and the inspector shows what a run actually produces: the trajectory,
// the verifier verdicts, and per-part geometry facts.

import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type JSX,
} from "react";
import type * as THREE from "three";

import { AppErrorBoundary } from "@/components/layout/AppErrorBoundary";
import { ViewportPanel } from "@/components/layout/ViewportPanel";
import { AnimationTimeline } from "@/components/inspector/AnimationTimeline";
import { PlaybackNotes } from "@/components/inspector/PlaybackNotes";
import { RenderOptionsPanel } from "@/components/inspector/RenderOptionsPanel";
import { useAnimationPlayback } from "@/components/viewer3d/useAnimationPlayback";
import { shouldApplyStudioShading } from "@/components/viewer3d/scene-presentation";
import { useDerivedAssets } from "@/components/viewer3d/useDerivedAssets";
import { buildFlippedJointValues } from "@/components/viewer3d/joint-flip";
import { useJointController } from "@/components/viewer3d/useJointController";
import { useRenderOptions } from "@/components/viewer3d/useRenderOptions";
import type { UrdfSpec } from "@/components/viewer3d/urdf-parser";
import {
  PREVIEW_UI_SYNC_MS,
  buildPreviewJointMotions,
  previewJointValue,
} from "@/components/viewer3d/preview-motion";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ClipPanel } from "@/panels/ClipPanel";
import { CodePanel } from "@/panels/CodePanel";
import { OverviewPanel } from "@/panels/OverviewPanel";
import { PartsPanel } from "@/panels/PartsPanel";
import { TrajectoryPanel } from "@/panels/TrajectoryPanel";
import { RunSidebar } from "@/panels/RunSidebar";
import { AgentTimeline } from "@/panels/AgentTimeline";
import { clearLoading, describeBytes, setLoading } from "@/lib/loading";
import { isTypingTarget } from "@/components/viewer3d/useCameraKeys";
import { SummaryPanel } from "@/panels/SummaryPanel";
import {
  fetchAgentTimeline,
  fetchAnimation,
  fetchFilmCameras,
  fetchStoryboard,
  storyboardThumbUrl,
  type StoryboardKey,
  runFileUrl,
  fetchPrepare,
  fetchRun,
  fetchRuns,
  fetchTrajectory,
  runFileBase,
} from "@/lib/api";
import { type FilmCameraTrack, shotsOf } from "@/components/viewer3d/film-camera";
import type {
  AgentTimeline as AgentTimelineData,
  AnimationTimeline as AnimationTimelineData,
  RunDetail,
  RunSummary,
  Trajectory,
} from "@/lib/types";

type SnapshotFn = (() => Promise<void>) | null;

/** Tabs a published film has nothing for: no context summary, no agent trajectory, no parts, no sources. */
const FILM_HIDDEN_TABS = new Set(["summary", "trajectory", "parts", "code"]);

/** Shown over the viewport while a film installed from the published set is prepared for its first open. */
const PREPARING_NOTE =
  "Preparing this film to play in 3D. The first open evaluates its Blender project, which can take a minute or " +
  "more for the larger films; later opens are immediate.";

function formatSeconds(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

const NO_STORYBOARD = { frame_start: 0, keys: [] as StoryboardKey[] };

export default function App(): JSX.Element {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  // Bumped by the sidebar's refresh button. Every per-run effect watches it, so
  // one click re-reads the selected run's detail, trajectory, and timeline as
  // well as the list — a run that was still going when it was opened is the
  // reason to refresh at all, and its panels are what went stale.
  const [reloadNonce, setReloadNonce] = useState(0);
  // The run the user chose, and the run whose data loads. They differ only while a film installed from the
  // published set is prepared for its first open: nothing of it loads until it is ready.
  const [requestedRunId, setRequestedRunId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [prepareMessage, setPrepareMessage] = useState<string | null>(null);

  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [trajectoryLoading, setTrajectoryLoading] = useState(false);
  const [trajectoryError, setTrajectoryError] = useState<string | null>(null);
  const [agentTimeline, setAgentTimeline] = useState<AgentTimelineData | null>(
    null,
  );
  // The one thing neither panel could do for the other: an action in the strip
  // and the turn that issued it are the same event seen twice, so selecting
  // either highlights both.
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);

  const [urdfSpec, setUrdfSpec] = useState<UrdfSpec | null>(null);
  const [jointNodes, setJointNodes] =
    useState<Map<string, THREE.Object3D> | null>(null);
  const [linkNodes, setLinkNodes] =
    useState<Map<string, THREE.Group> | null>(null);
  const [snapshot, setSnapshot] = useState<SnapshotFn>(null);
  const [codeFile, setCodeFile] = useState<string | null>(null);
  const [tab, setTab] = useState("overview");
  const [timeline, setTimeline] = useState<AnimationTimelineData | null>(null);
  // The film's camera at every frame, asked for once its clip is on screen (`SceneCanvas` follows it).
  const [filmTrack, setFilmTrack] = useState<FilmCameraTrack | null>(null);
  const [timelineError, setTimelineError] = useState<string | null>(null);

  const { options: renderOptions, setOption } = useRenderOptions();

  // The scene's committed rest configuration, keyed by URDF joint name. The
  // URDF carries dycraft's joint frame and cannot express a default pose, so
  // without this every joint with a non-zero authored default would draw at 0.
  // Memoized on `detail` so the controller's seed effect does not re-run — and
  // so a slider the user has dragged is not yanked back on every render.
  const restJointValues = useMemo(() => {
    const out = new Map<string, number>();
    for (const j of detail?.ir?.joints ?? []) {
      if (typeof j.default === "number" && Number.isFinite(j.default)
          && j.default !== 0) {
        out.set(j.name, j.default);
      }
    }
    return out;
  }, [detail]);

  const invalidateRef = useRef<(() => void) | null>(null);
  const invalidate = useCallback(() => {
    invalidateRef.current?.();
  }, []);

  const { jointValues, posedJointValues, setJointValue, applyJointValues, resetAll } =
    useJointController(jointNodes, urdfSpec, restJointValues, invalidate);

  // Preview motion runs off requestAnimationFrame and writes straight into the
  // three.js graph. Refs, not state: a per-frame setState would re-render the
  // whole shell 60x a second, and `onInvalidateReady` hands over a *function*,
  // which a useState setter would invoke rather than store.
  const [previewJointValues, setPreviewJointValues] = useState<
    Map<string, number>
  >(new Map());
  const jointValuesRef = useRef(jointValues);
  const previewUiLastSyncRef = useRef(0);

  useEffect(() => {
    jointValuesRef.current = jointValues;
  }, [jointValues]);

  const loadRuns = useCallback(() => {
    setRunsLoading(true);
    setRunsError(null);
    fetchRuns()
      .then((list) => setRuns(list))
      .catch((err: Error) => setRunsError(err.message))
      .finally(() => setRunsLoading(false));
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);
  // Nothing is selected on load, on purpose. Folders start collapsed, so
  // auto-selecting the first run would highlight a run inside a shut folder,
  // and the inspector would fill with a run the sidebar gives no sign of having
  // chosen. Almost every run lives in a folder, so that is the normal case.

  const handleRefresh = useCallback(() => {
    loadRuns();
    setReloadNonce((n) => n + 1);
  }, [loadRuns]);

  const requestedRun = useMemo(
    () => runs.find((r) => r.run_id === requestedRunId) ?? null,
    [runs, requestedRunId],
  );
  const needsPreparing = requestedRun?.prepare === "needed";
  // A published film has its film and its 3D playback, and none of a run's working records.
  const installedFilm = requestedRun?.installed === true;

  // A film installed from the published set is prepared the first time it is opened (`/prepare`), with its
  // progress in the loading tray; every other run loads at once.
  useEffect(() => {
    setPrepareMessage(null);
    if (!requestedRunId || !needsPreparing) {
      setSelectedRunId(requestedRunId);
      return;
    }
    setSelectedRunId(null);
    const runId = requestedRunId;
    const label = "Preparing film";
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const fail = (why: string) => {
      clearLoading("preparing");
      setPrepareMessage(`Preparing this film failed: ${why}`);
    };
    const poll = () => {
      fetchPrepare(runId)
        .then((status) => {
          if (cancelled) return;
          if (status.state === "ready") {
            clearLoading("preparing");
            setPrepareMessage(null);
            setSelectedRunId(runId);
            return;
          }
          if (status.state === "failed") {
            fail(status.error);
            return;
          }
          if (status.state === "preparing") {
            const elapsed = formatSeconds(status.seconds);
            if (status.frames && status.frames_done > 0) {
              const done = Math.min(status.frames_done, status.frames);
              setLoading("preparing", label, `frame ${done} of ${status.frames}, ${elapsed}`,
                         done / status.frames);
            } else {
              setLoading("preparing", label, `opening its Blender file, ${elapsed}`);
            }
          }
          timer = setTimeout(poll, 1000);
        })
        .catch((err: Error) => {
          if (!cancelled) fail(err.message);
        });
    };
    setPrepareMessage(PREPARING_NOTE);
    setLoading("preparing", label, "starting");
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      clearLoading("preparing");
    };
  }, [requestedRunId, needsPreparing]);

  // Selecting a run drops all per-run state: a stale URDF spec would leave the
  // joint sliders bound to the previous object's joints. So does a film being
  // prepared, which selects nothing until it is ready.
  useEffect(() => {
    setDetail(null);
    setUrdfSpec(null);
    setJointNodes(null);
    setLinkNodes(null);
    setCodeFile(null);
    setTrajectory(null);
    setTrajectoryError(null);
    setTrajectoryLoading(false);
    setTimeline(null);
    setTimelineError(null);
    if (!selectedRunId) return;
    let cancelled = false;

    // The model loads once the detail is in (`baseFileUrl`); until then the tray says what it is waiting for.
    setLoading("geometry", "Geometry", "reading the run");
    fetchRun(selectedRunId)
      .then((data) => {
        if (cancelled) return;
        setDetail(data);
        // Only an animation run has one, and it is the one dense payload the
        // API serves — chained off the detail rather than fired blind so a
        // scene run never pays for a 404 round trip.
        if (data.ir?.kind !== "animation") return;
        fetchAnimation(selectedRunId)
          .then((data2) => {
            if (!cancelled) setTimeline(data2);
          })
          .catch((err: Error) => {
            if (!cancelled) setTimelineError(err.message);
          });
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setRunsError(err.message);
        clearLoading("geometry");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId, reloadNonce]);

  // The trajectory is a run's largest payload (a long film's: 47 MB, 16 MB gzipped) and only its own tab shows it, so
  // it is fetched when that tab first opens (picking a turn on the timeline opens it), and kept until another run is
  // selected. Over a slow link that is seconds off every first open.
  const [trajectoryOpened, setTrajectoryOpened] = useState(false);
  useEffect(() => {
    setTrajectoryOpened(false);
  }, [selectedRunId, reloadNonce]);
  useEffect(() => {
    if (tab === "trajectory") setTrajectoryOpened(true);
  }, [tab, selectedRunId, reloadNonce]);
  useEffect(() => {
    if (!selectedRunId || !trajectoryOpened) return;
    let cancelled = false;
    const trajectoryAbort = new AbortController();
    setTrajectoryLoading(true);
    setLoading("trajectory", "Trajectory", "requested");
    fetchTrajectory(selectedRunId, (received, total) => {
      if (cancelled) return;
      const { detail, fraction } = describeBytes(received, total);
      setLoading("trajectory", "Trajectory", detail, fraction);
    }, trajectoryAbort.signal)
      .then((data) => {
        if (!cancelled) setTrajectory(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setTrajectoryError(err.message);
      })
      .finally(() => {
        if (!cancelled) {
          setTrajectoryLoading(false);
          clearLoading("trajectory");
        }
      });
    return () => {
      cancelled = true;
      trajectoryAbort.abort();
      clearLoading("trajectory");
    };
  }, [selectedRunId, reloadNonce, trajectoryOpened]);

  // Fetched separately from the trajectory even though it consumes it
  // server-side: the strip is cheap and useful immediately, and should not
  // wait on the turn cards.
  useEffect(() => {
    if (!selectedRunId) {
      setAgentTimeline(null);
      return;
    }
    let cancelled = false;
    const abort = new AbortController();
    setSelectedTurn(null);
    setLoading("timeline", "Timeline", "requested");
    fetchAgentTimeline(selectedRunId, (received, total) => {
      if (cancelled) return;
      const { detail, fraction } = describeBytes(received, total);
      setLoading("timeline", "Timeline", detail, fraction);
    }, abort.signal)
      .then((data) => {
        if (!cancelled) setAgentTimeline(data);
      })
      .catch(() => {
        // A run with no trace and no transcript has no strip, and nothing else
        // on the page depends on it, so failing quietly is right here.
        if (!cancelled) setAgentTimeline(null);
      })
      .finally(() => {
        if (!cancelled) clearLoading("timeline");
      });
    return () => {
      cancelled = true;
      abort.abort();
      clearLoading("timeline");
    };
  }, [selectedRunId, reloadNonce]);

  // The Clip trigger only exists for animation runs, so selecting any other
  // run while it is active would leave the inspector on a tab with no trigger
  // and no content.
  useEffect(() => {
    if (tab === "clip" && detail && detail.ir?.kind !== "animation") {
      setTab("overview");
    }
    if (installedFilm && FILM_HIDDEN_TABS.has(tab)) setTab("overview");
  }, [detail, installedFilm, tab]);

  // "Flip joints", from the Parts tab: one pose with every joint mirrored to
  // the opposite side of its own range. The pose goes through the same
  // controller the sliders use, so mimic joints follow their source and the
  // numbers in the panel move with the model. `commit` is what makes them
  // move — an uncommitted apply writes the graph only, which is right for a
  // 60 fps driver and wrong for a deliberate pose the user now wants to read
  // off and adjust.
  //
  // `restJointValues` is passed because a continuous joint has no limits to
  // mirror within and uses its rest value as the center instead.
  const flipJoints = useCallback(() => {
    if (!urdfSpec) return;
    const flipped = buildFlippedJointValues(urdfSpec.joints, jointValues,
                                            restJointValues);
    applyJointValues(flipped, { commit: true });
    invalidateRef.current?.();
  }, [applyJointValues, jointValues, restJointValues, urdfSpec]);

  const handleSelectTurn = useCallback((turn: number | null) => {
    setSelectedTurn(turn);
    // Pointing at a turn is only useful if the panel showing turns is open.
    if (turn !== null) setTab("trajectory");
  }, []);

  const handleSelectRun = useCallback((runId: string) => {
    setRequestedRunId(runId);
  }, []);

  const handleUrdfSpecChange = useCallback(
    (
      spec: UrdfSpec | null,
      nodes: Map<string, THREE.Object3D> | null,
      links: Map<string, THREE.Group> | null,
    ) => {
      setUrdfSpec(spec);
      setJointNodes(nodes);
      setLinkNodes(links);
    },
    [],
  );

  const showSource = useCallback((path: string) => {
    setCodeFile(path);
    setTab("code");
  }, []);

  // Store the exporter, don't invoke it. `useState`'s setter treats a bare
  // function argument as an updater and CALLS it, so `onSnapshotReady={
  // setSnapshot}` made every scene load run the PNG export — i.e. clicking a
  // run silently downloaded an image. The thunk makes React store it instead.
  const handleSnapshotReady = useCallback((fn: SnapshotFn) => {
    setSnapshot(() => fn);
  }, []);

  const handleInvalidateReady = useCallback((fn: (() => void) | null) => {
    invalidateRef.current = fn;
  }, []);

  // Baked-atlas viewer assets; the viewport picks a URDF variant from them.
  //
  // Suppressed under `showCollisions`, which replaces the visual geometry with
  // a translucent proxy — baking the real appearance for something about to be
  // drawn as a debug overlay is pure waste.
  const derived = useDerivedAssets(
    selectedRunId,
    Boolean(renderOptions.bakedMaterials) && !renderOptions.showCollisions,
  );

  // Replay the committed animation in the viewport. Owns the playhead; poses
  // the scene whenever the frame changes.
  const playback = useAnimationPlayback({
    runId: selectedRunId,
    timeline,
    urdfSpec,
    jointNodes,
    linkNodes,
    studioShading: shouldApplyStudioShading(renderOptions),
    applyJointValues,
    invalidate,
  });

  // The timeline's shots (a change of the film's camera is a cut) and its storyboard's keyframes, both in the
  // playhead's frames. The keyframes come in the film's own frames, with the film's first frame to subtract.
  const [storyboard, setStoryboard] = useState<{ frame_start: number; keys: StoryboardKey[] }>(NO_STORYBOARD);
  const isAnimation = detail?.ir?.kind === "animation";
  useEffect(() => {
    setStoryboard(NO_STORYBOARD);
    if (!selectedRunId || !isAnimation) return;
    let cancelled = false;
    fetchStoryboard(selectedRunId)
      .then((found) => {
        if (!cancelled) setStoryboard(found);
      })
      .catch(() => {
        // No storyboard: the timeline marks none.
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId, isAnimation]);
  const shots = useMemo(() => shotsOf(filmTrack), [filmTrack]);
  const keyframes = useMemo(
    () => storyboard.keys.map((key) => ({ ...key, frame: key.frame - storyboard.frame_start })),
    [storyboard],
  );

  // The film's camera at every frame: asked for once the clip is on screen, so the film's opening never waits on it,
  // and only for a film that opens on a camera of its own. A film without one keeps its opening shot.
  const hasFilmCamera = Boolean(timeline?.film_camera);
  useEffect(() => {
    setFilmTrack(null);
    if (!selectedRunId || !hasFilmCamera || !playback.nativeReady) return;
    let cancelled = false;
    fetchFilmCameras(selectedRunId)
      .then((track) => {
        if (!cancelled) setFilmTrack(track);
      })
      .catch(() => {
        // No camera to follow: the film view stays on its opening shot.
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId, hasFilmCamera, playback.nativeReady]);

  // Posing the graph once its nodes exist is `useJointController`'s job — it
  // is handed `invalidate` for exactly that. Posing it here, with the *rest*
  // pose, would make a baked-material swap (which rebuilds the graph) reset
  // every joint the user had moved.

  // "Preview motion": sweep every movable joint on its own phase-offset cycle.
  // `SceneCanvas` only switches to continuous rendering when the option is on —
  // it does not move anything itself, so without this driver the toggle just
  // burned frames on a static pose.
  //
  // Suppressed on an animation run: an invented sweep and an authored,
  // verified trajectory would be writing the same joints, and the sweep would
  // win whichever ran last.
  useEffect(() => {
    if (timeline) return;
    if (!urdfSpec || !renderOptions.autoAnimate) {
      previewUiLastSyncRef.current = 0;
      setPreviewJointValues(new Map());
      applyJointValues(jointValuesRef.current);
      return;
    }
    const motions = buildPreviewJointMotions(urdfSpec);
    if (motions.length === 0) {
      setPreviewJointValues(new Map());
      return;
    }

    let frameId = 0;
    const tick = (now: number) => {
      const elapsed = now / 1000;
      const next = new Map<string, number>();
      for (const motion of motions) {
        const phase =
          ((elapsed / motion.cycleSeconds + motion.phaseOffset) % 1 + 1) % 1;
        next.set(motion.joint.name, previewJointValue(motion.joint, phase));
      }
      // The graph updates every frame; the sliders only ~10x a second, since
      // re-rendering the inspector at 60 fps is pure waste.
      if (now - previewUiLastSyncRef.current >= PREVIEW_UI_SYNC_MS) {
        previewUiLastSyncRef.current = now;
        startTransition(() => setPreviewJointValues(new Map(next)));
      }
      applyJointValues(next);
      invalidateRef.current?.();
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frameId);
      previewUiLastSyncRef.current = 0;
      setPreviewJointValues(new Map());
      // Restore the pose the user had set before previewing.
      applyJointValues(jointValuesRef.current);
      invalidateRef.current?.();
    };
  }, [applyJointValues, renderOptions.autoAnimate, timeline, urdfSpec]);

  const displayedJointValues =
    renderOptions.autoAnimate && previewJointValues.size > 0
      ? previewJointValues
      : jointValues;

  // What went wrong in the 3D playback: meshes that did not load, bodies the
  // graph has no placement joint for, a timeline that would not load. Each can
  // make the picture silently wrong — a body that does not move looks like a body
  // that is not supposed to — so `PlaybackNotes` shows them in amber, ahead of
  // the clip's standing notes.
  const playbackProblems = useMemo(() => {
    const notes = [...playback.notes];
    for (const instance of playback.missing) {
      notes.push(
        `${instance} moves in the animation but has no placement joint in the ` +
          `loaded URDF — it is drawn at rest`,
      );
    }
    if (timelineError) notes.push(`timeline unavailable: ${timelineError}`);
    return notes;
  }, [playback.missing, playback.notes, timelineError]);
  // Only once the run's detail is in, which carries the IR id the mesh URLs are keyed by: loading before it would
  // fetch and parse the model under one set of URLs, then again under the other once the detail arrives.
  // Playback from the keyboard: K plays and pauses, J and L seek two seconds, as on video sites (Space flies the
  // camera, as in Minecraft). Window-wide, as a video site's are, while the flight keys and R count only over the
  // viewport (`useCameraKeys`). Never while a text field has the keyboard, nor with a modifier held.
  const playbackRef = useRef(playback);
  playbackRef.current = playback;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.repeat || isTypingTarget(event.target)) return;
      const p = playbackRef.current;
      if (p.nFrames <= 1) return;
      if (event.code === "KeyK") p.togglePlay();
      else if (event.code === "KeyJ") p.seek(-2);
      else if (event.code === "KeyL") p.seek(2);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const baseFileUrl = selectedRunId && detail ? runFileBase(selectedRunId) : null;
  const inspectorUrdfSpec = urdfSpec ? { joints: urdfSpec.joints } : null;
  const codeFiles = useMemo(() => {
    if (!detail) return [];
    const files = [...detail.source_files];
    if (detail.has_ir) files.push("ir.json");
    // An animation's input scene: the geometry on screen is entirely this
    // file's, so not being able to read it makes the run half-legible. A native
    // film without one draws from its own track, so there is no file to list.
    if (detail.ir?.kind === "animation" && (detail.ir.scene_id || !detail.ir.native)) {
      files.push("scene.json");
    }
    // A staged build123d source reached from the Parts panel is not in the
    // run's top-level listing, so make it selectable once requested.
    if (codeFile && !files.includes(codeFile)) files.unshift(codeFile);
    return files;
  }, [detail, codeFile]);

  // A committed scene whose catalog did not resolve has an `ir.json` and still
  // nothing to draw, so `has_ir` alone is the wrong gate for the overlay.
  const noGeometryNotice = useMemo(() => {
    if (!detail) return null;
    if (!detail.has_ir) {
      return "This run has no ir.json — geometry is written when a run commits.";
    }
    const ir = detail.ir;
    // An animation run draws its staged scene, so it fails the same way — and
    // additionally when that scene was never staged at all, unless it is a
    // native film, which draws itself from its own track.
    if (ir?.kind === "animation" && !ir.scene_id && !ir.native) {
      return (
        "This animation run has no scene.json — the frozen scene it animates " +
        "is staged into the run dir at launch, and everything drawn here " +
        "comes from that file rather than from ir.json."
      );
    }
    if (
      (ir?.kind === "scene" || ir?.kind === "animation") &&
      (ir.catalog_total ?? 0) > 0 &&
      (ir.catalog_resolved ?? 0) === 0
    ) {
      return (
        `This scene's ${ir.catalog_total} instance(s) do not resolve against ` +
        "catalog/ — the object catalog is snapshotted next to the scene IR " +
        "on a successful compile."
      );
    }
    return null;
  }, [detail]);

  return (
    <AppErrorBoundary>
      <div className="flex h-screen flex-col bg-[var(--bg-primary)] text-[var(--text-primary)]">
        <header className="flex items-center gap-2 border-b border-[var(--border-subtle)] px-3 py-2">
          <span className="text-[13px] font-semibold">dycraft</span>
          <span className="text-[13px] text-[var(--text-quaternary)]">
            / Run viewer
          </span>
          {requestedRunId ? (
            <span className="ml-auto font-mono text-[11px] text-[var(--text-tertiary)]">
              {requestedRunId}
            </span>
          ) : null}
        </header>

        <main className="flex min-h-0 flex-1 flex-col">
          <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
          <ResizablePanel defaultSize="22%" minSize="14%" className="min-w-0">
            <RunSidebar
              runs={runs}
              error={runsError}
              selectedRunId={requestedRunId}
              onSelect={handleSelectRun}
              onRefresh={handleRefresh}
              refreshing={runsLoading}
            />
          </ResizablePanel>
          <ResizableHandle />

          <ResizablePanel defaultSize="48%" minSize="25%" className="min-w-0">
            <div className="flex h-full min-h-0 flex-col">
              <div className="min-h-0 flex-1">
                <ViewportPanel
                  baseFileUrl={baseFileUrl}
                  assetRevisionKey={detail?.ir?.id ?? null}
                  selectionKey={selectedRunId}
                  jointPoseSignal={displayedJointValues}
                  derived={derived}
                  posedJointValues={posedJointValues}
                  renderOptions={renderOptions}
                  fitBounds={timeline?.bounds ?? null}
                  filmCamera={timeline?.film_camera ?? null}
                  filmTrack={filmTrack}
                  frame={playback.frame}
                  filmClipUrl={selectedRunId && detail?.ir?.kind === "animation" && detail.ir.clip
                    ? runFileUrl(selectedRunId, detail.ir.clip) : null}
                  playing={playback.playing}
                  rate={playback.rate}
                  fps={playback.fps}
                  nativeVersion={timeline?.native?.version ?? null}
                  nativeReady={playback.nativeReady}
                  onUrdfSpecChange={handleUrdfSpecChange}
                  onSnapshotReady={handleSnapshotReady}
                  onInvalidateReady={handleInvalidateReady}
                  idleMessage={prepareMessage}
                  disabledOverlay={
                    noGeometryNotice ? (
                      <p className="text-[11px] text-[var(--text-tertiary)]">
                        {noGeometryNotice}
                      </p>
                    ) : undefined
                  }
                />
              </div>
              {timeline ? (
                <AnimationTimeline
                  frame={playback.frame}
                  nFrames={playback.nFrames}
                  fps={playback.fps}
                  playing={playback.playing}
                  rate={playback.rate}
                  events={timeline.events}
                  onSetFrame={playback.setFrame}
                  onStep={playback.step}
                  onTogglePlay={playback.togglePlay}
                  onSetRate={playback.setRate}
                  loop={playback.loop}
                  onToggleLoop={() => playback.setLoop(!playback.loop)}
                  shots={shots}
                  keyframes={keyframes}
                  keyframeThumb={(index) => (selectedRunId ? storyboardThumbUrl(selectedRunId, index) : "")}
                />
              ) : null}
              <PlaybackNotes
                key={selectedRunId ?? ""}
                standing={timeline?.notes ?? []}
                problems={playbackProblems}
              />
            </div>
          </ResizablePanel>
          <ResizableHandle />

          <ResizablePanel defaultSize="30%" minSize="20%" className="min-w-0">
            <Tabs
              value={tab}
              onValueChange={setTab}
              className="flex h-full min-h-0 flex-col"
            >
              <TabsList className="custom-scrollbar min-w-0 shrink-0 overflow-x-auto">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                {installedFilm ? null : (
                  <>
                    <TabsTrigger value="summary">Summary</TabsTrigger>
                    <TabsTrigger value="trajectory">Trajectory</TabsTrigger>
                    <TabsTrigger value="parts">Parts</TabsTrigger>
                    <TabsTrigger value="code">Code</TabsTrigger>
                  </>
                )}
                <TabsTrigger value="render">Render</TabsTrigger>
                {/* Animation runs only — every other layer's `final_render`
                    writes the five-view snapshot the Overview already shows. */}
                {detail?.ir?.kind === "animation" ? (
                  <TabsTrigger value="clip">Clip</TabsTrigger>
                ) : null}
              </TabsList>

              <TabsContent value="overview" className="min-h-0 flex-1">
                {detail ? (
                  <OverviewPanel
                    detail={detail}
                    onSelectRun={handleSelectRun}
                    installed={installedFilm}
                  />
                ) : (
                  <p className="p-3 text-[11px] text-[var(--text-tertiary)]">
                    {prepareMessage
                      ? prepareMessage.startsWith("Preparing this film failed")
                        ? prepareMessage
                        : "Preparing this film. Its overview appears when it opens."
                      : "Select a run."}
                  </p>
                )}
              </TabsContent>

              <TabsContent value="summary" className="min-h-0 flex-1">
                {detail?.has_context_summary && selectedRunId ? (
                  <SummaryPanel runId={selectedRunId} />
                ) : detail ? (
                  <p className="p-3 text-[11px] text-[var(--text-tertiary)]">
                    No context summary recorded for this run.
                  </p>
                ) : (
                  <p className="p-3 text-[11px] text-[var(--text-tertiary)]">
                    Select a run.
                  </p>
                )}
              </TabsContent>

              <TabsContent value="trajectory" className="min-h-0 flex-1">
                <TrajectoryPanel
                  trajectory={trajectory}
                  loading={trajectoryLoading || (!trajectory && !trajectoryError)}
                  error={trajectoryError}
                  selectedTurn={selectedTurn}
                  onSelectTurn={setSelectedTurn}
                />
              </TabsContent>

              <TabsContent value="parts" className="min-h-0 flex-1">
                <PartsPanel
                  parts={detail?.ir?.parts ?? []}
                  instances={detail?.ir?.instances ?? []}
                  urdfSpec={inspectorUrdfSpec}
                  jointValues={displayedJointValues}
                  onJointChange={setJointValue}
                  onResetJoints={resetAll}
                  onFlipJoints={flipJoints}
                  // Preview motion and clip playback both write every joint
                  // per frame, so a flip would last exactly one frame.
                  motionDriven={Boolean(timeline) || renderOptions.autoAnimate}
                  onShowSource={showSource}
                />
              </TabsContent>

              <TabsContent value="code" className="min-h-0 flex-1">
                {selectedRunId && codeFiles.length > 0 ? (
                  <CodePanel
                    runId={selectedRunId}
                    files={codeFiles}
                    selected={codeFile}
                    onSelect={setCodeFile}
                  />
                ) : (
                  <p className="p-3 text-[11px] text-[var(--text-tertiary)]">
                    No source files.
                  </p>
                )}
              </TabsContent>

              <TabsContent value="render" className="min-h-0 flex-1">
                <RenderOptionsPanel
                  options={renderOptions}
                  onOptionChange={setOption}
                  onSnapshot={snapshot}
                />
              </TabsContent>

              <TabsContent value="clip" className="min-h-0 flex-1">
                {selectedRunId && detail?.ir?.kind === "animation" ? (
                  <ClipPanel
                    runId={selectedRunId}
                    clip={detail.ir.clip ?? null}
                    animation={detail.ir.animation}
                  />
                ) : null}
              </TabsContent>
            </Tabs>
          </ResizablePanel>
          </ResizablePanelGroup>
          {/* The run's own actions, bottom-docked so the viewport keeps the
              width. */}
          <div className="max-h-[46vh] shrink-0 overflow-y-auto custom-scrollbar">
            {/* A run with no recorded agent activity, such as a published film, has nothing to show here. */}
            {agentTimeline && agentTimeline.totals.n_events > 0 ? (
              <AgentTimeline
                timeline={agentTimeline}
                selectedTurn={selectedTurn}
                onSelectTurn={handleSelectTurn}
              />
            ) : null}
          </div>
        </main>
      </div>
    </AppErrorBoundary>
  );
}
