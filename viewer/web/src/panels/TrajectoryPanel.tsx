// Full agent trajectory: per-turn reasoning, tool calls, results, and tokens.
//
// The turn renderer is Articraft's `TrajectoryTurnCard`, vendored unchanged.
// What differs is where the data comes from: dycraft normalizes on the server
// (viewer/trajectory.py) instead of shipping raw trace text to the client, so
// `lib/trajectory-adapter` reshapes the JSON rather than the card being reworked.

import { useEffect, useMemo, useRef, useState, type JSX } from "react";
import { AlertTriangle } from "lucide-react";

import { TrajectoryTurnCard } from "@/components/inspector/TrajectoryTurnCard";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { adaptTrajectory } from "@/lib/trajectory-adapter";
import { cn } from "@/lib/utils";
import type { Trajectory } from "@/lib/types";

/** Turn cards rendered at first, and added each time the list is scrolled to its end. */
const TURN_BATCH = 60;

function formatCount(value: number): string {
  return value.toLocaleString();
}

const SOURCE_LABEL: Record<string, string> = {
  claude_cli: "external Claude CLI transcript",
  codex_cli: "external Codex CLI rollout",
};

const BUCKET_LABEL: Record<string, string> = {
  input: "uncached input",
  cache_read: "cache read",
  cache_write: "cache write",
  cache_write_5m: "cache write (5 m)",
  cache_write_1h: "cache write (1 h)",
  output: "output",
};

function Notice({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div className="flex items-start gap-1.5 rounded-md bg-amber-500/[0.08] px-2 py-1.5 text-[10px] text-amber-600">
      <AlertTriangle className="mt-0.5 size-3 shrink-0 opacity-70" />
      <span>{children}</span>
    </div>
  );
}

/**
 * Where the money went, per bucket. The point of showing this is that token
 * share and cost share diverge sharply: cache reads are usually most of the
 * tokens and a minority of the cost, while output is a few percent of the
 * tokens and often half the bill.
 */
function CostTable({
  totals,
}: {
  totals: Trajectory["totals"];
}): JSX.Element | null {
  const entries = Object.entries(totals.cost_buckets ?? {}).filter(
    ([, usd]) => usd > 0,
  );
  if (entries.length === 0) return null;
  const sum = entries.reduce((acc, [, usd]) => acc + usd, 0);
  entries.sort((a, b) => b[1] - a[1]);
  return (
    <table className="w-full font-mono text-[10px]">
      <tbody>
        {entries.map(([bucket, usd]) => (
          <tr key={bucket}>
            <td className="text-[var(--text-tertiary)]">
              {BUCKET_LABEL[bucket] ?? bucket}
            </td>
            <td className="pl-2 text-right tabular-nums text-[var(--text-primary)]">
              ${usd.toFixed(4)}
            </td>
            <td className="pl-2 text-right tabular-nums text-[var(--text-quaternary)]">
              {sum > 0 ? `${((usd / sum) * 100).toFixed(0)}%` : ""}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type TrajectoryPanelProps = {
  trajectory: Trajectory | null;
  loading: boolean;
  error: string | null;
  /** Turn selected in the run timeline strip, or null. */
  selectedTurn?: number | null;
  onSelectTurn?: (turn: number | null) => void;
};

export function TrajectoryPanel({
  trajectory,
  loading,
  error,
  selectedTurn = null,
  onSelectTurn,
}: TrajectoryPanelProps): JSX.Element {
  const turnRefs = useRef(new Map<number, HTMLDivElement>());
  const turns = useMemo(
    () => (trajectory ? adaptTrajectory(trajectory) : []),
    [trajectory],
  );

  // A long run has thousands of turns, and rendering every card at once is one long task on the page. So the list
  // grows as it is scrolled: `TURN_BATCH` at a time, when the sentinel below the last card comes into view.
  const [shown, setShown] = useState(TURN_BATCH);
  useEffect(() => setShown(TURN_BATCH), [trajectory]);
  const sentinel = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const node = sentinel.current;
    if (!node || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setShown((n) => Math.min(turns.length, n + TURN_BATCH));
      }
    }, { rootMargin: "400px" });
    observer.observe(node);
    return () => observer.disconnect();
  }, [turns, shown]);

  // Clicking a bar in the strip picks a turn; bring it into view here, rendering the list as far as it first. A long
  // run is 200+ turns, so without this the selection is invisible.
  useEffect(() => {
    if (selectedTurn === null) return;
    const position = turns.findIndex((turn) => turn.index === selectedTurn);
    if (position >= shown) {
      setShown(position + TURN_BATCH);
      return;
    }
    turnRefs.current.get(selectedTurn)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [selectedTurn, turns, shown]);
  const baseTimestamp =
    turns.find((turn) => turn.timestamp != null)?.timestamp ?? null;

  if (loading) {
    return (
      <div className="space-y-3 p-3">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
    );
  }
  if (error) {
    return (
      <p className="p-3 text-[11px] text-red-400">
        Could not load trajectory: {error}
      </p>
    );
  }
  if (!trajectory || turns.length === 0) {
    return (
      <p className="p-3 text-[11px] text-[var(--text-tertiary)]">
        {trajectory?.reasoning_note ?? "No trajectory recorded for this run."}
      </p>
    );
  }

  const { totals } = trajectory;

  return (
    <ScrollArea className="h-full">
      <div className="space-y-3 p-3">
        <div className="space-y-2 border-b border-[var(--border-subtle)] pb-3">
          <div className="flex flex-wrap items-center gap-1.5">
            {/* One model chip per model actually used: a run can mix them when
                the provider falls back mid-run. */}
            {((trajectory.models ?? []).length
              ? trajectory.models
              : trajectory.model
                ? [{ model: trajectory.model, turns: totals.turns }]
                : // A totals-only envelope has no per-turn models to name.
                  // `meta.json`'s is on the Overview tab; a chip reading
                  // "unknown" next to real token counts only looks like a bug.
                  []
            ).map((m) => (
              <Badge key={m.model} variant="secondary">
                {m.model}
                {(trajectory.models ?? []).length > 1 ? ` ×${m.turns}` : ""}
              </Badge>
            ))}
            <Badge variant="secondary">{totals.turns} turns</Badge>
            <Badge variant="secondary">{totals.tool_calls} tool calls</Badge>
            {totals.cost_usd != null ? (
              <Badge variant={totals.cost_complete ? "success" : "secondary"}>
                {totals.cost_complete ? "" : "≥ "}${totals.cost_usd.toFixed(2)}
              </Badge>
            ) : null}
          </div>

          {/* Input and output are the two figures worth leading with. Summing
              them with cache reads — as this panel first did — inflates the
              apparent size of a run several-fold, because a cache read is a
              re-read of the same prompt and is priced at a tenth of an input
              token. */}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-0.5 font-mono text-[11px]">
            <div className="flex justify-between">
              <dt className="text-[var(--text-tertiary)]">input tokens</dt>
              <dd className="tabular-nums text-[var(--text-primary)]">
                {formatCount(totals.gross_input)}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-[var(--text-tertiary)]">output tokens</dt>
              <dd className="tabular-nums text-[var(--text-primary)]">
                {formatCount(totals.output)}
              </dd>
            </div>
          </dl>
          <p className="font-mono text-[10px] text-[var(--text-quaternary)]">
            input = uncached {formatCount(totals.input)} + cache read{" "}
            {formatCount(totals.cache_read)} + cache write{" "}
            {formatCount(totals.cache_write)}
            {totals.cache_write_1h ? " (1 h tier)" : ""}
          </p>

          <CostTable totals={totals} />

          {trajectory.source ? (
            <p className="font-mono text-[10px] text-[var(--text-quaternary)]">
              source: {SOURCE_LABEL[trajectory.source] ?? trajectory.source}
            </p>
          ) : null}
          {!trajectory.reasoning_available && trajectory.reasoning_note ? (
            <Notice>{trajectory.reasoning_note}</Notice>
          ) : null}
          {(totals.unpriced_models ?? []).length ? (
            <Notice>
              No price is known for {(totals.unpriced_models ?? []).join(", ")}, so this total covers only the
              turns that have one and is a lower bound.
            </Notice>
          ) : totals.cost_usd == null ? (
            <p className="text-[10px] text-[var(--text-quaternary)]">
              No prices are known for this run&apos;s models, so there is no cost to show.
            </p>
          ) : null}
        </div>

        {turns.slice(0, shown).map((turn) => (
          <div
            key={turn.index}
            ref={(element) => {
              if (element) turnRefs.current.set(turn.index, element);
              else turnRefs.current.delete(turn.index);
            }}
            onClick={() => onSelectTurn?.(turn.index)}
            aria-current={selectedTurn === turn.index ? "step" : undefined}
            className={cn(
              "border-b border-[var(--border-subtle)] pb-3 last:border-0",
              selectedTurn === turn.index &&
                "-mx-1 rounded-md bg-[var(--accent-soft)] px-1 ring-1 ring-[var(--accent)]",
            )}
          >
            <TrajectoryTurnCard turn={turn} baseTimestamp={baseTimestamp} />
          </div>
        ))}
        {shown < turns.length ? (
          <div ref={sentinel} className="py-2 text-center text-[10px] text-[var(--text-quaternary)]">
            {shown} of {turns.length} turns shown; scroll for more
          </div>
        ) : null}
      </div>
    </ScrollArea>
  );
}
