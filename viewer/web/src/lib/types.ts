// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
// Shapes returned by dycraft's viewer API (see viewer/runs.py, viewer/trajectory.py).
import type { FilmCamera } from '@/components/viewer3d/film-camera';

export type RunSummary = {
  /** Path relative to `runs/` — `achieve/kettle` for a run inside a folder. */
  run_id: string;
  /** Leaf label (meta.json's run_id, else the directory name). */
  name: string;
  /** What the run list calls it, ahead of its prompt: a published film's brief and title. */
  label?: string | null;
  /**
   * A film installed from the published set. It has its film and its 3D playback, and none of a run's working
   * records (agent trajectory, context summary, parts, sources), so the inspector leaves those tabs out.
   */
  installed?: boolean;
  /** Parent folders, `""` for a run sitting directly in `runs/`. */
  group: string;
  layer: string | null;
  model: string | null;
  reasoning_effort: string | null;
  runtime: string | null;
  outcome: string | null;
  prompt: string | null;
  started_at: string | null;
  finished_at: string | null;
  /**
   * Wall clock, in seconds — from the run's own session when it recorded one,
   * else `finished_at - started_at`. Null for a run that measured neither.
   */
  wall_seconds: number | null;
  /**
   * What the model read and wrote, when `meta.json` carries a `usage` block.
   * Null for every run that predates it, so treat its absence as "unknown"
   * rather than "zero". The per-turn breakdown is on `/trajectory`.
   */
  tokens: {
    gross_input: number;
    output: number;
    turns: number;
    /** A codex run's sub-agents, each its own session: their total tokens. */
    subagents?: { total: number; threads: number };
  } | null;
  has_ir: boolean;
  /**
   * What `ir.json` actually holds, read from the payload rather than from
   * `layer`. Counts below are per kind, so read the one that matches (`null`
   * when the run has no IR).
   */
  ir_kind: IrKind;
  n_parts: number;
  n_joints: number;
  /** Scene runs; on an animation run, the *staged* scene's instance count. */
  n_instances: number;
  /** Animation runs only. */
  n_frames: number;
  /**
   * A film installed from the published set that the viewer has not prepared yet: it is prepared the first time
   * it is opened (`/prepare`), and loads once that is done.
   */
  prepare?: "needed";
};

/**
 * An `animation` IR carries no geometry at all — what it animates is the
 * frozen SceneIR staged beside it as `scene.json`, which is where the viewer
 * gets the URDF, the parts, and the joints for such a run.
 */
export type IrKind = "object" | "scene" | "animation" | null;

export type VerifierReport = {
  name: string | null;
  verdict: string | null;
};

export type ExactAdjudication = {
  available?: boolean;
  considered?: number;
  sent?: number;
  cleared?: number;
  confirmed?: number;
  unknown?: number;
  not_adjudicable?: number;
  cache_hits?: number;
  reason?: string;
};

export type CompileReport = {
  step: number | null;
  ok: boolean | null;
  ir_hash: string | null;
  build_error: Record<string, unknown> | null;
  sandbox_violation: Record<string, unknown> | null;
  verifiers: VerifierReport[];
  exact_adjudication: ExactAdjudication | null;
};

export type PartSummary = {
  name: string | null;
  primitive: string | null;
  material: string | null;
  color: unknown;
  source: string | null;
  // `solid` parts only — derived from the B-rep at authoring time.
  volume?: number;
  fill_ratio?: number;
  face_census?: Record<string, number>;
  center?: number[];
  /** Scene IRs only: which instance this part came from. `name` is prefixed. */
  instance?: string | null;
};

/** One placed object in a scene IR, as the inspector shows it. */
export type SceneInstanceSummary = {
  name: string | null;
  object_name: string | null;
  object_ref: string | null;
  /** False when the catalog snapshot is missing this object — it won't draw. */
  in_catalog: boolean;
  placement_kind: string | null;
  /** Another instance's name, or "room". */
  parent: string | null;
  parent_anchor: string | null;
  child_anchor: string | null;
  n_parts: number;
  n_joints: number;
};

/** One run in a run's `sub_runs/`: a scene or motion rewriter. */
export type SubRunSummary = {
  /** Runs-dir-relative, so it selects like any other run. */
  run_id: string;
  name: string;
  layer: string | null;
  model: string | null;
  outcome: string | null;
  started_at: string | null;
  finished_at: string | null;
  has_ir: boolean;
};

export type JointSummary = {
  name: string;
  type: string;
  parent?: string;
  child?: string;
  limits?: number[] | null;
  default?: number;
};

export type RunDetail = {
  run_id: string;
  reasoning_effort: string | null;
  meta: Record<string, unknown>;
  compile_reports: CompileReport[];
  source_files: string[];
  renders: string[];
  has_ir: boolean;
  /** True when context_summary.md is present and safely servable. */
  has_context_summary: boolean;
  /** Runs this run started in its `sub_runs/` (its rewriters); `[]` if none. */
  sub_runs: SubRunSummary[];
  ir?: {
    kind: IrKind;
    id: string | null;
    name: string | null;
    root: string | null;
    /** Animation runs only — the facts, without the dense per-frame arrays. */
    animation?: AnimationSummary;
    /** Animation runs only; null until `final_render` has run. */
    clip?: string | null;
    /**
     * Animation runs only: a native film, drawn from its own evaluated track. It needs no staged scene, and a film
     * installed from the published set has none.
     */
    native?: boolean;
    /** Animation runs only: the staged scene the geometry below comes from. */
    scene_id?: string | null;
    scene_name?: string | null;
    /**
     * For a scene these are flattened across instances and prefixed
     * `<instance>__`, matching the URDF's link/joint names.
     */
    parts: PartSummary[];
    joints: JointSummary[];
    collision_allowances: { a: string; b: string; reason?: string }[];
    assertions: unknown[];
    // Scene IRs only.
    instances?: SceneInstanceSummary[];
    groups?: unknown[];
    room?: Record<string, unknown> | null;
    /** How many instances resolved against `catalog/`; 0 of N means no geometry. */
    catalog_resolved?: number;
    catalog_total?: number;
  };
};

// ---------------------------------------------------------------------------
// Animation (see viewer/animation.py)
// ---------------------------------------------------------------------------

export type AnimationEvent = {
  name: string | null;
  predicate: Record<string, unknown> | null;
  window: [number, number] | number[];
  /** First/last frame the window covers, `null` when it covers none. */
  window_frames?: [number, number] | null;
  /** Seconds, from the run's own EventsSatisfied report; null if unmet. */
  satisfied_at?: number | null;
  satisfied_frame?: number | null;
};

export type AnimationSummary = {
  fps: number | null;
  duration: number | null;
  n_frames: number;
  scene_ref: string | null;
  /** False means the staged scene is not the one this IR was built against. */
  scene_ref_ok: boolean | null;
  events: AnimationEvent[];
  provenance: { label?: string; frames?: [number, number] }[];
};

/**
 * One instance whose *placement* moves. `frames[i]` is an absolute world SE3,
 * `[tx,ty,tz,qw,qx,qy,qz]` — w-first, as everywhere in dycraft; three.js is
 * x-first, and `animation-playback.ts` is the one place that reorders.
 *
 * Absolute rather than a delta because the URDF nests a kinematic-follow child
 * under its parent's link: a delta applied at the child would compose with the
 * parent's and double it.
 */
export type AnimationBody = {
  instance: string;
  /** URDF link name — the child of the fixed joint that places the instance. */
  link: string;
  rest: number[];
  frames: number[][];
  /** Effective physical presence, after attachment inheritance. */
  exists: boolean[];
};

export type AnimationTimeline = {
  native?: { version: string };
  fps: number;
  duration: number | null;
  n_frames: number;
  /** URDF joint names and values, in dycraft's own joint frame — no offset. */
  joints: {
    name: string;
    instance: string | null;
    joint: string | null;
    type: string | null;
    values: number[];
  }[];
  bodies: AnimationBody[];
  /** Always empty: the native timeline still sends the key. Nothing reads it. */
  deform: unknown[];
  events: AnimationEvent[];
  provenance: { label?: string; frames?: [number, number] }[];
  clip: string | null;
  /**
   * World AABB over every frame, for the camera fit. Same union the offline
   * clip renderer frames on, so the viewport and the mp4 agree; null when the
   * kernel could not bound the scene.
   */
  bounds: { min: number[]; max: number[] } | null;
  /**
   * The camera the delivered film opens on, from its `camera.json`; null when
   * the run delivered no film. See `components/viewer3d/film-camera.ts`.
   */
  film_camera?: FilmCamera | null;
  /** Things this timeline cannot show, and why. Surfaced verbatim. */
  notes: string[];
};

// ---------------------------------------------------------------------------
// per-agent timeline  (viewer/agent_timeline.py)
// ---------------------------------------------------------------------------

/**
 * Nine categories on one shared band. The order is semantic, not
 * load-bearing: the palette in index.css clears the ALL-PAIRS colourblindness
 * floors, so no pair depends on never being adjacent.
 *
 * `model` and `judge` (the review sub-agent's grading) are LLM latency rather
 * than agent actions, and together they are most of a run, which is why
 * `totals` carries a separate `action_ms` denominator.
 */
export type AgentCategoryKey =
  | "model"
  | "judge"
  | "author"
  | "build"
  | "inspect"
  | "look"
  | "solve"
  | "deliver"
  | "aside";

export type AgentCategory = {
  key: AgentCategoryKey;
  label: string;
  blurb: string;
  n_events: number;
  ms: number;
};

export type AgentEventStatus = "ok" | "failed" | "running" | "unknown";

export type AgentTimelineEvent = {
  id: string;
  order: number;
  category: AgentCategoryKey;
  /** `op:compile`, `model.turn`, `op:review`, `aside.unknown`, … */
  action: string;
  /** Always non-empty: this is the segment's own visible label. */
  label: string;
  /** Harness step, when this came from a traced op. */
  step: number | null;
  /** Owning trajectory turn, for the cross-panel selection. */
  turn: number | null;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  status: AgentEventStatus;
  cache_hit: boolean | null;
  /** Bookkeeping with no duration — `record_deviation`, `.terminate`. */
  zero_duration: boolean;
  source: "trace" | "transcript";
  detail: {
    op?: string;
    tool?: string;
    dycraft_op?: string | null;
    model?: string | null;
    effort?: string | null;
    stop_reason?: string | null;
    thought?: boolean;
    text?: boolean;
    n_tool_calls?: number;
    /**
     * A model turn's internal stretches, measured from the per-block record
     * timestamps. `write` is the model composing a tool call's input — 31% of
     * an object run's model time — and is hatched inside the segment rather
     * than given a category of its own. Empty on the `api` path.
     */
    phases?: { kind: "think" | "text" | "write"; offset_ms: number; duration_ms: number }[];
    think_ms?: number;
    text_ms?: number;
    write_ms?: number;
    compile?: {
      attempt: number | null;
      is_finish: boolean | null;
      build_error: string | null;
      n_verifiers: number;
      n_failed: number;
      failed_verifiers: string[];
      verifiers: { name: string | null; verdict: string | null }[];
    } | null;
  };
};

export type AgentTimelineMark = {
  kind: string;
  at: string | null;
  label: string;
  failure: boolean;
  step: number | null;
  detail: Record<string, unknown> | null;
};

export type AgentTimeline = {
  run_id: string | null;
  layer: string;
  outcome: string | null;
  started_at: string | null;
  finished_at: string | null;
  run_finished: boolean;
  now: string | null;
  categories: AgentCategory[];
  events: AgentTimelineEvent[];
  marks: AgentTimelineMark[];
  totals: {
    categories: Record<AgentCategoryKey, { n: number; ms: number }>;
    /** Excludes `model` and `judge` — the only comparable denominator. */
    action_ms: number;
    model_ms: number;
    judge_ms: number;
    model_n: number;
    judge_n: number;
    all_ms: number;
    n_events: number;
  };
  /** What this timeline cannot show, and why. Surfaced verbatim. */
  notes: string[];
};

export type TrajectoryUsage = {
  /** Anthropic vocabulary: the UNCACHED remainder only, not the whole prompt. */
  input: number;
  cache_read: number;
  cache_write: number;
  /** Write tier matters: 1 h costs 1.6x the 5 m rate. */
  cache_write_5m: number;
  cache_write_1h: number;
  output: number;
  /** Articraft/OpenAI-style `prompt_tokens`: everything read, cache included. */
  gross_input: number;
  total: number;
};

export type CostBuckets = Partial<Record<
  "input" | "cache_read" | "cache_write" | "cache_write_5m" | "cache_write_1h" | "output",
  number
>>;

export type TurnBilling = {
  inference_geo: string | null;
  service_tier: string | null;
  speed: string | null;
  /** Combined rate multiplier: geo 1.1x, batch 0.5x, fast 2x. */
  multiplier: number | null;
};

export type TrajectoryToolCall = {
  name: string | null;
  input: unknown;
  result: string | null;
  is_error: boolean;
  dycraft_op: string | null;
  dycraft_step: number | null;
};

export type TrajectoryTurn = {
  index: number;
  ts: string | null;
  elapsed_ms: number | null;
  model: string | null;
  effort: string | null;
  thought_summary: string | null;
  text: string | null;
  tool_calls: TrajectoryToolCall[];
  usage: TrajectoryUsage;
  billing: TurnBilling;
  cost_usd: number | null;
  cost_buckets: CostBuckets | null;
  stop_reason: string | null;
};

export type Trajectory = {
  source: "claude_cli" | "codex_cli" | null;
  transcript_path: string | null;
  model: string | null;
  /** False when the producer did not persist reasoning text — see reasoning_note. */
  reasoning_available: boolean;
  reasoning_note: string | null;
  /** Per-model turn counts — a run can mix models via provider fallback. */
  models: { model: string; turns: number }[];
  turns: TrajectoryTurn[];
  totals: TrajectoryUsage & {
    turns: number;
    tool_calls: number;
    cost_usd: number | null;
    cost_buckets: CostBuckets;
    /** False when some turn's model has no rate card, so the total is partial. */
    cost_complete: boolean;
    unpriced_models: string[];
  };
};

export type TextFile = {
  path: string;
  size: number;
  truncated: boolean;
  text: string;
};

/** `/api/runs/<id>/prepare`: whether a run is ready to open (`viewer/films.py`). */
export type PrepareStatus =
  | { state: "ready" }
  | { state: "unprepared" }
  | { state: "failed"; error: string }
  | {
      state: "preparing";
      /** Frames evaluated so far; 0 while Blender opens the film, which for a heavy film is most of the wait. */
      frames_done: number;
      frames: number | null;
      seconds: number;
    };
