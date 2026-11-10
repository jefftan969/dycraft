// Map dycraft's normalized trajectory onto the shape the vendored
// TrajectoryTurnCard renders.
//
// The card was written against Articraft's own trace format, whose parser
// (`parseEnrichedTurns`) consumes raw JSONL text. dycraft normalizes on the
// server instead (viewer/trajectory.py), so this adapter takes that JSON and
// produces `EnrichedTraceTurn[]` directly, leaving the 945-line renderer
// untouched.
//
// One turn becomes an assistant message event followed by one `role: "tool"`
// event per tool call that returned a result — the same layout Articraft's
// parser produces, which is what the card's tool-call/result pairing expects.

import type {
  EnrichedTraceEvent,
  EnrichedTraceMessage,
  EnrichedTraceTurn,
} from "@/components/inspector/trajectory-types";
import type { Trajectory, TrajectoryTurn, TrajectoryUsage } from "@/lib/types";

/** `formatRelativeTime` renders `ts - base` as seconds, so emit seconds. */
function epochSeconds(iso: string | null): number | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : ms / 1000;
}

function toTraceUsage(usage: TrajectoryUsage) {
  return {
    // Articraft's vocabulary: `prompt_tokens` is GROSS input (cache included)
    // and `cached_tokens` is the subset served from cache. dycraft carries the
    // uncached remainder separately, so pass the gross figure through rather
    // than a partial sum.
    prompt_tokens: usage.gross_input,
    cached_tokens: usage.cache_read,
    candidates_tokens: usage.output,
    total_tokens: usage.total,
  };
}

function stringifyArgs(input: unknown): string {
  if (input == null) return "";
  if (typeof input === "string") return input;
  try {
    return JSON.stringify(input, null, 2);
  } catch {
    return String(input);
  }
}

function turnCost(turn: TrajectoryTurn): Record<string, unknown> {
  return {
    tokens: toTraceUsage(turn.usage),
    costs_usd: turn.cost_usd == null ? {} : { total: turn.cost_usd },
  };
}

function adaptTurn(turn: TrajectoryTurn): EnrichedTraceTurn {
  const timestamp = epochSeconds(turn.ts);
  const events: EnrichedTraceEvent[] = [];

  const assistant: EnrichedTraceMessage = {
    kind: "message",
    eventType: "message",
    role: "assistant",
    content: turn.text ?? "",
    thought_summary: turn.thought_summary ?? undefined,
    usage: toTraceUsage(turn.usage),
    timestamp: timestamp ?? undefined,
    tool_calls: turn.tool_calls.map((call, index) => ({
      id: `t${turn.index}-c${index}`,
      type: "function",
      function: {
        // Surface the dycraft op when we could identify one: `Bash` says
        // nothing, `Bash -> compile (step 4)` says everything.
        name: call.dycraft_op
          ? `${call.name ?? "tool"} → ${call.dycraft_op}` +
            (call.dycraft_step != null ? ` (step ${call.dycraft_step})` : "")
          : (call.name ?? "tool"),
        arguments: stringifyArgs(call.input),
      },
    })),
  };
  events.push(assistant);

  turn.tool_calls.forEach((call, index) => {
    if (call.result == null) return;
    events.push({
      kind: "message",
      eventType: "message",
      role: "tool",
      name: call.dycraft_op ?? call.name ?? "tool",
      tool_call_id: `t${turn.index}-c${index}`,
      content: call.result,
      timestamp: timestamp ?? undefined,
    });
  });

  return { index: turn.index, cost: turnCost(turn), events, timestamp };
}

export function adaptTrajectory(trajectory: Trajectory): EnrichedTraceTurn[] {
  return trajectory.turns.map(adaptTurn);
}
