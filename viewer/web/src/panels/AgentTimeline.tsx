// One agent run's own actions, as a single labeled band with the legend below.
//
// One band can hold all nine categories because the palette in `index.css`
// clears the all-pairs color-blindness floors (normal-vision ΔE 17.4,
// min(protan,deutan) ΔE 10.4, against 15 / 8), so any two can sit side by
// side. The order here is semantic, not load-bearing, and every segment also
// carries a visible text label: identity is text, with color reinforcing it.
//
// Segments rarely overlap: model generation runs only up to its turn's first
// tool call. Parallel tool calls in one turn are rare, and are packed into an
// extra row of the same band rather than given a lane of their own.

import { useCallback, useEffect, useMemo, useRef, useState, type JSX } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type {
  AgentCategory,
  AgentCategoryKey,
  AgentTimeline as Timeline,
  AgentTimelineEvent,
  AgentTimelineMark,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { CompactNotes } from "@/components/inspector/CompactNotes";

type AgentTimelineProps = {
  timeline: Timeline;
  /** Selected trajectory turn, so the band can highlight its segments. */
  selectedTurn: number | null;
  onSelectTurn: (turn: number | null) => void;
};

const FILL: Record<AgentCategoryKey, string> = {
  model: "var(--cat-model)",
  judge: "var(--cat-judge)",
  author: "var(--cat-author)",
  build: "var(--cat-build)",
  inspect: "var(--cat-inspect)",
  look: "var(--cat-look)",
  solve: "var(--cat-solve)",
  deliver: "var(--cat-deliver)",
  aside: "var(--cat-aside)",
};

const INK: Record<AgentCategoryKey, string> = {
  model: "var(--cat-model-ink)",
  judge: "var(--cat-judge-ink)",
  author: "var(--cat-author-ink)",
  build: "var(--cat-build-ink)",
  inspect: "var(--cat-inspect-ink)",
  look: "var(--cat-look-ink)",
  solve: "var(--cat-solve-ink)",
  deliver: "var(--cat-deliver-ink)",
  aside: "var(--cat-aside-ink)",
};

type LayoutView = "time" | "order";
const VIEW_STORAGE_KEY = "dycraft-agent-timeline-view";
const COLLAPSE_STORAGE_KEY = "dycraft-agent-timeline-collapsed";

function readView(): LayoutView {
  try {
    return localStorage.getItem(VIEW_STORAGE_KEY) === "order" ? "order" : "time";
  } catch {
    return "time";
  }
}

/** Tall enough for an inline label, which is the point of the single band. */
const ROW_HEIGHT = 20;
const ROW_GAP = 3;
/** Below this many pixels a label cannot be read, so it is dropped. */
const MIN_LABEL_PX = 46;
/**
 * Column width in the order view. Fixed rather than `1/n` of the axis, and the
 * strip scrolls: at 82 segments an even split is 15px a column and not one
 * label fits, which defeats the view whose whole job is that every segment can
 * be read.
 */
const ORDER_COL_PX = 116;

function parseTime(value: string | null): number | null {
  if (!value) return null;
  const ms = new Date(value).getTime();
  return Number.isNaN(ms) ? null : ms;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  if (seconds < 90) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${minutes.toFixed(1)}m`;
  return `${(minutes / 60).toFixed(1)}h`;
}

function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Tick spacing that lands at most eight gridlines on the axis. */
export function tickStepMs(spanMs: number): number {
  const second = 1000;
  for (const step of [1, 5, 15, 30, 60, 300, 900, 1800, 3600, 7200, 14400]) {
    if (spanMs / (step * second) <= 8) return step * second;
  }
  return 28800 * second;
}

export type PlacedSegment = {
  event: AgentTimelineEvent;
  /** Left edge, as a fraction of the axis. */
  from: number;
  /** Width, as a fraction of the axis. 0 for a tick. */
  width: number;
  tick: boolean;
  /** 0 is the band's own row; >0 only where segments genuinely overlap. */
  row: number;
};

/**
 * Where each segment sits, and which row of the band it lands on.
 *
 * Two rules this exists to hold:
 *
 * 1. A segment with no end runs to `now` only while the run itself is going.
 *    Once the run has finished, nobody recorded a result and the duration is
 *    *unknown*; drawing it to `now` would grow the bar every time the page is
 *    opened.
 * 2. A `zero_duration` action (`record_deviation`, `.terminate`, `commit`) is
 *    a tick by definition, not by rounding, and must not widen the axis.
 *
 * Row packing is greedy over start-sorted segments — lowest row whose previous
 * segment has already ended. That is optimal in row count and keeps the run's
 * own sequence on row 0.
 *
 * Pure, and `now` is a parameter, so it is testable without a DOM — see
 * `scripts/check-agent-timeline.mjs`.
 */
export function layoutSegments(
  events: AgentTimelineEvent[],
  {
    startedAt,
    finishedAt,
    now,
    runFinished,
  }: {
    startedAt: string | null;
    finishedAt: string | null;
    now: number;
    runFinished: boolean;
  },
): { placed: PlacedSegment[]; span: number; origin: number; rows: number } {
  const stamps: number[] = [];
  for (const event of events) {
    const start = parseTime(event.started_at);
    if (start !== null) stamps.push(start);
    const end = parseTime(event.finished_at);
    if (end !== null) stamps.push(end);
    if (start !== null && end === null && !runFinished && !event.zero_duration) {
      stamps.push(now);
    }
  }
  const runStart = parseTime(startedAt);
  const runEnd = parseTime(finishedAt);
  if (runStart !== null) stamps.push(runStart);
  if (runEnd !== null) stamps.push(runEnd);

  const t0 = stamps.length ? Math.min(...stamps) : 0;
  const t1 = stamps.length ? Math.max(...stamps) : 1;
  // A run that began and ended in the same instant, or nothing datable: keep
  // the span nonzero so nothing collapses and the axis cannot divide by zero.
  const span = t1 > t0 ? t1 - t0 : 1;

  const prepared = events.map((event) => {
    const start = parseTime(event.started_at);
    const end = parseTime(event.finished_at);
    const running =
      start !== null && end === null && !runFinished && !event.zero_duration;
    const until =
      start === null
        ? null
        : event.zero_duration
          ? start
          : running
            ? now
            : (end ?? start);
    return {
      event,
      start,
      until,
      tick: event.zero_duration || end === null,
    };
  });
  prepared.sort((a, b) => (a.start ?? Infinity) - (b.start ?? Infinity));

  const rowEnds: number[] = [];
  const placed: PlacedSegment[] = prepared.map(({ event, start, until, tick }) => {
    if (start === null) {
      return { event, from: 0, width: 0, tick: true, row: 0 };
    }
    const end = until ?? start;
    // Boundary jitter is not concurrency. A turn's model window opens at the
    // record *before* the turn, which can precede the previous tool call's
    // recorded end by a few milliseconds, and treating that as overlap would
    // push a minutes-long segment onto a second row and make the band look
    // broken. Anything under a pixel of the current axis counts as abutting.
    const epsilon = Math.max(250, span * 0.002);
    let row = rowEnds.findIndex((last) => last - epsilon <= start);
    if (row === -1) {
      row = rowEnds.length;
      rowEnds.push(end);
    } else {
      rowEnds[row] = Math.max(rowEnds[row], end);
    }
    return {
      event,
      from: (start - t0) / span,
      width: Math.max(0, (end - start) / span),
      tick,
      row,
    };
  });
  return { placed, span, origin: t0, rows: Math.max(1, rowEnds.length) };
}

/** Narrow segments that start within this many pixels of each other on a row are drawn as one. */
const MERGE_PX = 3;
/** The width merging assumes before the plot has been measured. */
const UNMEASURED_WIDTH_PX = 1600;

interface DrawnSegment extends PlacedSegment {
  /** Every event this segment stands for, first one first. One for a segment wide enough to draw alone. */
  members: AgentTimelineEvent[];
}

/**
 * The segments to draw: each wide one alone, and the narrow ones that start in the same `MERGE_PX` of a row as one.
 *
 * A long run has thousands of events, and on a time axis most of them are slivers drawn at the 3 px minimum, on top
 * of one another. Drawing each as its own element would make the strip the page's heaviest render. Merged, the
 * count follows the strip's width instead. A merged segment opens its first event's turn and says how many it holds.
 */
function mergeNarrow(placed: PlacedSegment[], plotWidth: number): DrawnSegment[] {
  const width = plotWidth > 0 ? plotWidth : UNMEASURED_WIDTH_PX;
  const drawn: DrawnSegment[] = [];
  const open = new Map<number, { bin: number; segment: DrawnSegment }>();
  for (const segment of placed) {
    if (!segment.tick && segment.width * width >= MERGE_PX) {
      drawn.push({ ...segment, members: [segment.event] });
      continue;
    }
    const bin = Math.floor((segment.from * width) / MERGE_PX);
    const current = open.get(segment.row);
    if (current && current.bin === bin) {
      current.segment.members.push(segment.event);
      continue;
    }
    const merged: DrawnSegment = { ...segment, members: [segment.event] };
    open.set(segment.row, { bin, segment: merged });
    drawn.push(merged);
  }
  return drawn;
}

function TooltipRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div className="grid grid-cols-[74px_minmax(0,1fr)] gap-3 py-0.5">
      <dt className="text-[10px] text-[var(--text-tertiary)]">{label}</dt>
      <dd className="min-w-0 text-[10px] text-[var(--text-secondary)]">
        {children}
      </dd>
    </div>
  );
}

function SegmentTooltip({
  event,
  categoryLabel,
}: {
  event: AgentTimelineEvent;
  categoryLabel: string;
}): JSX.Element {
  const compile = event.detail.compile;
  return (
    <div className="w-[326px] p-2.5 font-normal">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 font-mono text-[8px] uppercase tracking-[0.09em] text-[var(--text-quaternary)]">
            <span
              aria-hidden="true"
              className="h-1.5 w-1.5 shrink-0 rounded-sm"
              style={{ background: FILL[event.category] }}
            />
            {categoryLabel} · {event.action}
          </p>
          <p className="mt-0.5 break-words text-[11px] font-semibold text-[var(--text-primary)]">
            {event.label}
          </p>
        </div>
        <span
          className={cn(
            "shrink-0 rounded-full border px-2 py-1 text-[9px]",
            event.status === "failed"
              ? "border-[var(--destructive)] text-[var(--destructive)]"
              : event.status === "ok"
                ? "border-[var(--border-subtle)] text-[var(--text-secondary)]"
                : "border-[var(--warning)] text-[var(--warning)]",
          )}
        >
          {event.status}
        </span>
      </div>

      <dl className="mt-2 border-t border-[var(--border-subtle)] pt-1.5">
        <TooltipRow label="Duration">
          <span className="font-mono">
            {event.zero_duration
              ? "instant (bookkeeping)"
              : event.duration_ms === null
                ? "unknown"
                : formatDuration(event.duration_ms)}
          </span>
        </TooltipRow>
        <TooltipRow label="Started">
          <time className="font-mono">{formatTimestamp(event.started_at)}</time>
        </TooltipRow>
        {event.step !== null ? (
          <TooltipRow label="Step">
            <span className="font-mono">{event.step}</span>
          </TooltipRow>
        ) : null}
        {event.turn !== null ? (
          <TooltipRow label="Turn">
            <span className="font-mono">
              {event.turn} · click to open it in Trajectory
            </span>
          </TooltipRow>
        ) : null}
        {event.detail.model ? (
          <TooltipRow label="Model">
            <span className="font-mono">{event.detail.model}</span>
          </TooltipRow>
        ) : null}
        {event.detail.write_ms ? (
          <TooltipRow label="Split">
            <span className="font-mono">
              {formatDuration(
                (event.detail.think_ms ?? 0) + (event.detail.text_ms ?? 0),
              )}{" "}
              thinking · {formatDuration(event.detail.write_ms)} composing the
              tool call (hatched)
            </span>
          </TooltipRow>
        ) : null}
        {event.detail.stop_reason ? (
          <TooltipRow label="Stop">
            <span className="font-mono">{event.detail.stop_reason}</span>
          </TooltipRow>
        ) : null}
        {event.cache_hit ? (
          <TooltipRow label="Cache">warm — nothing was recomputed</TooltipRow>
        ) : null}
      </dl>

      {compile ? (
        <div className="mt-2 border-t border-[var(--border-subtle)] pt-1.5">
          <p className="text-[10px] text-[var(--text-secondary)]">
            attempt {compile.attempt ?? "?"}
            {compile.is_finish ? " (finish)" : ""} ·{" "}
            {compile.n_verifiers - compile.n_failed}/{compile.n_verifiers}{" "}
            verifiers passed
          </p>
          {compile.build_error ? (
            <p className="mt-1 font-mono text-[10px] text-[var(--destructive)]">
              {compile.build_error}
            </p>
          ) : null}
          {compile.failed_verifiers.length ? (
            <p className="mt-1 font-mono text-[10px] text-[var(--destructive)]">
              {compile.failed_verifiers.join(", ")}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function MarkRow({
  marks,
  span,
  origin,
}: {
  marks: AgentTimelineMark[];
  span: number;
  origin: number;
}): JSX.Element {
  return (
    <div className="relative h-2.5">
      <TooltipProvider>
        {marks.map((mark, index) => {
          const at = parseTime(mark.at);
          if (at === null) return null;
          return (
            <Tooltip key={`${mark.kind}-${index}`}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  tabIndex={-1}
                  aria-label={mark.label}
                  className={cn(
                    "absolute top-0 h-full w-px -translate-x-1/2",
                    mark.failure
                      ? "bg-[var(--destructive)]"
                      : "bg-[var(--text-quaternary)]",
                  )}
                  style={{ left: `${((at - origin) / span) * 100}%` }}
                />
              </TooltipTrigger>
              <TooltipContent side="top" sideOffset={6} className="rounded-md">
                <span className="px-2 py-1 text-[10px]">
                  {mark.label} · {formatTimestamp(mark.at)}
                </span>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </TooltipProvider>
    </div>
  );
}

/**
 * One segment, in either view. Extracted so the two layouts differ only in the
 * geometry they hand it — the fill, ink, failure texture, label, and accessible
 * name are defined once and cannot drift between views.
 */
function Segment({
  event,
  count = 1,
  categoryLabel,
  selected,
  selectedTurn,
  onSelectTurn,
  showLabel,
  style,
}: {
  event: AgentTimelineEvent;
  /** Events merged into this segment (`mergeNarrow`); `event` is the first. */
  count?: number;
  categoryLabel: string;
  selected: boolean;
  selectedTurn: number | null;
  onSelectTurn: (turn: number | null) => void;
  showLabel: boolean;
  style: React.CSSProperties;
}): JSX.Element {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`${count > 1 ? `${count} events from ` : ""}${categoryLabel}: ${event.label}${
            event.duration_ms === null
              ? ""
              : `, ${formatDuration(event.duration_ms)}`
          }, ${event.status}`}
          onClick={() =>
            onSelectTurn(
              event.turn !== null && event.turn === selectedTurn
                ? null
                : event.turn,
            )
          }
          className={cn(
            "overflow-hidden rounded-[3px] text-left outline-none",
            "transition-[filter,box-shadow] hover:brightness-110",
            "focus-visible:ring-2 focus-visible:ring-[var(--text-primary)]",
            event.status === "failed" && "seg-failed",
            event.status === "unknown" && "seg-unknown",
            selected && "ring-1 ring-[var(--text-primary)] brightness-110",
            event.turn === null && "cursor-default",
          )}
          style={{
            ...style,
            backgroundColor: FILL[event.category],
            color: INK[event.category],
            opacity: event.status === "running" ? 0.45 : 1,
          }}
        >
          {/* The stretches where the model was composing a tool call rather
              than thinking, hatched in place within the turn's one segment.
              Behind the label, and inert to the pointer. */}
          {(event.detail.phases ?? [])
            .filter((phase) => phase.kind === "write")
            .map((phase, index) =>
              event.duration_ms && event.duration_ms > 0 ? (
                <span
                  key={index}
                  aria-hidden="true"
                  className="seg-writing pointer-events-none absolute inset-y-0"
                  style={{
                    left: `${(phase.offset_ms / event.duration_ms) * 100}%`,
                    width: `${(phase.duration_ms / event.duration_ms) * 100}%`,
                  }}
                />
              ) : null,
            )}
          {showLabel ? (
            <span className="pointer-events-none relative block truncate px-1 text-[9px] leading-[20px]">
              {/* Failure is spelled out in the label, not only drawn: the
                  hatch and the dashed outline are both easy to miss on a
                  narrow segment, and neither survives a screenshot at 1x. */}
              {event.status === "failed" ? "✕ " : ""}
              {event.status === "unknown" ? "? " : ""}
              {event.label}
            </span>
          ) : null}
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        sideOffset={9}
        className="rounded-lg p-0 shadow-[0_8px_24px_rgba(0,0,0,0.10)]"
      >
        {count > 1 ? (
          <p className="border-b border-[var(--border-subtle)] px-2.5 pt-2 pb-1.5 text-[10px] text-[var(--text-secondary)]">
            {count} events here, too close together to draw apart; the first, which a click opens:
          </p>
        ) : null}
        <SegmentTooltip event={event} categoryLabel={categoryLabel} />
      </TooltipContent>
    </Tooltip>
  );
}

/** The legend, below the band. Every category the layer can fill. */
function Legend({
  categories,
  view,
}: {
  categories: AgentCategory[];
  view: LayoutView;
}): JSX.Element {
  return (
    <ul className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 pb-1.5 pt-1">
      {categories.map((category) => {
        // Absolute time, not a share. With everything on one band the band
        // itself shows the proportion, and a percentage needs a denominator
        // the eye cannot see: against action-only time an object run's Build
        // reads "94%" beside a two-pixel sliver, which is true and useless.
        // `totals.action_ms` is still in the payload for anyone who wants it.
        return (
          <li
            key={category.key}
            className="flex shrink-0 items-center gap-1.5"
            title={category.blurb}
          >
            <span
              aria-hidden="true"
              className="h-2 w-2 shrink-0 rounded-[2px]"
              style={{ background: FILL[category.key] }}
            />
            <span className="text-[10px] text-[var(--text-secondary)]">
              {category.label}
            </span>
            <span className="font-mono text-[9px] text-[var(--text-quaternary)]">
              {view === "order"
                ? category.n_events
                : formatDuration(category.ms)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export function AgentTimeline({
  timeline,
  selectedTurn,
  onSelectTurn,
}: AgentTimelineProps): JSX.Element {
  const [view, setView] = useState<LayoutView>(readView);
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem(COLLAPSE_STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });
  // The plot's pixel width, so a label can be dropped when it cannot be read
  // rather than clipped mid-word.
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [plotWidth, setPlotWidth] = useState(0);
  useEffect(() => {
    const node = plotRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      setPlotWidth(entries[0]?.contentRect.width ?? 0);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [collapsed]);

  const now = Date.now();
  const { placed, span, origin, rows } = useMemo(
    () =>
      layoutSegments(timeline.events, {
        startedAt: timeline.started_at,
        finishedAt: timeline.finished_at,
        now,
        runFinished: timeline.run_finished,
      }),
    // `now` is re-read every render on purpose; listing it would defeat the memo.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [timeline],
  );
  const drawn = useMemo(() => mergeNarrow(placed, plotWidth), [placed, plotWidth]);

  /**
   * Run order, one fixed-width column per segment, horizontally scrollable.
   * This is the view that makes every label readable: in time, a 40 ms probe
   * on a six-hour axis is three pixels and the run's shape is all model and
   * judge. In order it is a filmstrip of what the agent actually did.
   */
  const ordered = useMemo(
    () => [...timeline.events].sort((a, b) => a.order - b.order),
    [timeline.events],
  );

  const chooseView = useCallback((next: LayoutView) => {
    setView(next);
    try {
      localStorage.setItem(VIEW_STORAGE_KEY, next);
    } catch {
      // Storage denied. The toggle still works, it just will not persist.
    }
  }, []);

  const toggleCollapsed = useCallback(() => {
    setCollapsed((previous) => {
      const next = !previous;
      try {
        localStorage.setItem(COLLAPSE_STORAGE_KEY, next ? "1" : "0");
      } catch {
        // As above.
      }
      return next;
    });
  }, []);

  const labelOf = useMemo(
    () => new Map(timeline.categories.map((c) => [c.key, c.label])),
    [timeline.categories],
  );

  const ticks: number[] = [];
  const step = tickStepMs(span);
  for (let t = 0; t <= span; t += step) ticks.push(t);

  const { totals } = timeline;
  const agentVar =
    timeline.layer === "object"
      ? "var(--agent-object)"
      : timeline.layer === "scene"
        ? "var(--agent-scene)"
        : timeline.layer === "animation"
          ? "var(--agent-animation)"
          : "var(--agent-plain)";
  const bandRows = view === "order" ? 1 : rows;
  const bandHeight = bandRows * ROW_HEIGHT + (bandRows - 1) * ROW_GAP;

  return (
    <section
      aria-label="Agent run timeline"
      className="shrink-0 border-t border-[var(--border-default)] bg-[var(--surface-0)]"
    >
      <div className="flex h-7 min-w-0 items-center gap-2.5 border-b border-[var(--border-subtle)] px-3">
        <button
          type="button"
          onClick={toggleCollapsed}
          aria-expanded={!collapsed}
          className="flex shrink-0 items-center gap-1 text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
        >
          {collapsed ? <ChevronRight size={11} /> : <ChevronDown size={11} />}
          <span className="font-mono text-[9px] font-medium uppercase tracking-[0.09em]">
            Run timeline
          </span>
        </button>
        {/* Agent identity is chrome only — a dot and a label, never a segment
            fill, so the identity hues and the category hues never have to be
            told apart from one another. */}
        <span className="flex shrink-0 items-center gap-1.5 rounded-full border border-[var(--border-subtle)] px-1.5 py-0.5 text-[9px] text-[var(--text-secondary)]">
          <span
            aria-hidden="true"
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: agentVar }}
          />
          {timeline.layer}
        </span>
        <span className="min-w-0 truncate font-mono text-[9px] text-[var(--text-quaternary)]">
          {timeline.run_id ?? ""}
        </span>
        <span className="ml-auto shrink-0 font-mono text-[9px] text-[var(--text-quaternary)]">
          {totals.n_events} segments · {formatDuration(totals.all_ms)}
          {view === "time" && rows > 1 ? ` · ${rows} rows` : ""}
        </span>
        <span
          role="group"
          aria-label="Timeline layout"
          className="flex shrink-0 overflow-hidden rounded-full border border-[var(--border-subtle)]"
        >
          {(["time", "order"] as const).map((option) => (
            <button
              key={option}
              type="button"
              aria-pressed={view === option}
              title={
                option === "time"
                  ? "Proportional in time — what the run actually spent itself on"
                  : "Equal columns in run order — every segment readable, which the time view cannot be"
              }
              onClick={() => chooseView(option)}
              className={cn(
                "px-2 py-0.5 text-[8px] uppercase tracking-[0.07em] transition-colors",
                view === option
                  ? "bg-[var(--surface-2)] text-[var(--text-secondary)]"
                  : "text-[var(--text-quaternary)] hover:text-[var(--text-tertiary)]",
              )}
            >
              {option}
            </button>
          ))}
        </span>
      </div>

      {collapsed ? null : (
        <>
          <div className="px-3 pt-1.5">
            <div className="relative h-3">
              {view === "time"
                ? ticks.map((t) => (
                    <span
                      key={t}
                      className="absolute top-0 -translate-x-1/2 font-mono text-[9px] text-[var(--text-quaternary)]"
                      style={{ left: `${(t / span) * 100}%` }}
                    >
                      {t === 0 ? "0" : formatDuration(t)}
                    </span>
                  ))
                : null}
            </div>

            {view === "time" ? (
              <MarkRow marks={timeline.marks} span={span} origin={origin} />
            ) : (
              <div className="h-2.5" />
            )}

            {view === "order" ? (
              <div
                className="custom-scrollbar overflow-x-auto overflow-y-hidden"
                style={{ height: `${ROW_HEIGHT + 4}px` }}
              >
                <div className="flex min-w-max items-start gap-0.5">
                  <TooltipProvider>
                    {ordered.map((event) => (
                      <Segment
                        key={event.id}
                        event={event}
                        categoryLabel={
                          labelOf.get(event.category) ?? event.category
                        }
                        selected={
                          selectedTurn !== null && event.turn === selectedTurn
                        }
                        onSelectTurn={onSelectTurn}
                        selectedTurn={selectedTurn}
                        showLabel
                        style={{
                          position: "relative",
                          width: `${ORDER_COL_PX}px`,
                          flex: "0 0 auto",
                          height: `${ROW_HEIGHT}px`,
                        }}
                      />
                    ))}
                  </TooltipProvider>
                </div>
              </div>
            ) : (
              <div
                ref={plotRef}
                className="relative"
                style={{ height: `${bandHeight}px` }}
              >
                <span
                  aria-hidden="true"
                  className="absolute inset-0 rounded-sm bg-[var(--surface-2)]"
                />
                {ticks.map((t) => (
                  <span
                    key={t}
                    aria-hidden="true"
                    className="absolute inset-y-0 w-px bg-[var(--border-subtle)]"
                    style={{ left: `${(t / span) * 100}%` }}
                  />
                ))}

                <TooltipProvider>
                  {drawn.map(({ event, members, from, width, tick, row }) => (
                    <Segment
                      key={event.id}
                      event={event}
                      count={members.length}
                      categoryLabel={labelOf.get(event.category) ?? event.category}
                      selected={
                        selectedTurn !== null && members.some((member) => member.turn === selectedTurn)
                      }
                      onSelectTurn={onSelectTurn}
                      selectedTurn={selectedTurn}
                      showLabel={width * plotWidth >= MIN_LABEL_PX}
                      style={{
                        position: "absolute",
                        left: `${from * 100}%`,
                        // Ticks and slivers stay visible and clickable.
                        width: tick ? "3px" : `max(3px, ${width * 100}%)`,
                        height: `${ROW_HEIGHT}px`,
                        top: `${row * (ROW_HEIGHT + ROW_GAP)}px`,
                      }}
                    />
                  ))}
                </TooltipProvider>
              </div>
            )}
          </div>

          <Legend categories={timeline.categories} view={view} />

          <CompactNotes
            label="Timeline notes"
            title="How this timeline's durations were derived from the run's records, and what those records leave unknown"
            standing={timeline.notes}
            className="text-[9px]"
          />
        </>
      )}
    </section>
  );
}
