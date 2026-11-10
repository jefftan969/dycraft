// Run browser: the `runs/` directory tree, ordered by name inside each folder.
//
// Deliberately not Articraft's `RecordBrowser`: its eight facets (rating, cost,
// author, category, harness, ...) describe a hand-curated dataset, and none of
// those fields exist in a dycraft run's meta.json. What is worth seeing here is
// what the run did — layer, model, outcome, part/joint counts.
//
// The shape mirrors the filesystem because that is the only grouping `runs/`
// actually carries: a run bundle records nothing about which experiment it
// belonged to, so a folder somebody made is the whole signal. Runs sitting
// directly in `runs/` stay a flat list at the top, exactly as before.

import { useMemo, useState, type JSX } from "react";
import { RefreshCw } from "lucide-react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { RunSummary } from "@/lib/types";

const OUTCOME_STYLE: Record<string, string> = {
  committed: "text-emerald-500",
  interrupted: "text-amber-500",
};

function shortTime(iso: string | null): string {
  if (!iso) return "";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return "";
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** `1.2M`, `26k`, `947` — a token count at a glance, not to the digit. */
function shortCount(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${Math.round(value / 1e3)}k`;
  return String(Math.round(value));
}

/** `18m`, `2h14m`, `47s`. */
function shortWall(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, "0")}m`;
}

type TreeNode = {
  name: string;
  /** Folder path relative to `runs/`; "" is the root. */
  path: string;
  folders: Map<string, TreeNode>;
  runs: RunSummary[];
  /**
   * The run that *is* this folder, when one exists — a run whose `sub_runs/`
   * children (its scene and motion rewriters) make a folder at its own path.
   * Rendered at the head of the folder so the parent sits directly above the
   * runs it started, instead of orphaned below every folder at the level
   * above.
   */
  selfRun: RunSummary | null;
  /** Run count in this subtree, for the folder badge. */
  count: number;
};

function emptyNode(name: string, path: string): TreeNode {
  return { name, path, folders: new Map(), runs: [], selfRun: null, count: 0 };
}

/**
 * Group a flat run list into the folder tree its `group` paths describe.
 *
 * The server already returns runs ordered by path, so pushing in order leaves
 * each folder's runs sorted by name — no re-sorting of runs here.
 *
 * Two passes, because a run can turn out to be a folder: `sub_runs/` children
 * are discovered as ordinary runs whose group starts with their parent's
 * run_id, and the parent must be placed inside the node those groups created.
 * Creating every folder first means the second pass can just look one up.
 */
function buildTree(runs: RunSummary[]): TreeNode {
  const root = emptyNode("", "");
  const byPath = new Map<string, TreeNode>([["", root]]);

  const ensureFolder = (group: string): TreeNode => {
    let node = root;
    if (!group) return node;
    for (const segment of group.split("/")) {
      const path = node.path ? `${node.path}/${segment}` : segment;
      let child = node.folders.get(segment);
      if (!child) {
        child = emptyNode(segment, path);
        node.folders.set(segment, child);
        byPath.set(path, child);
      }
      node = child;
    }
    return node;
  };

  for (const run of runs) ensureFolder(run.group);
  for (const run of runs) {
    const own = byPath.get(run.run_id);
    if (own) own.selfRun = run;
    else ensureFolder(run.group).runs.push(run);
  }

  const countUp = (node: TreeNode): number => {
    let total = node.runs.length + (node.selfRun ? 1 : 0);
    for (const child of node.folders.values()) total += countUp(child);
    node.count = total;
    return total;
  };
  countUp(root);
  return root;
}

/**
 * Folders by name, matching how the server orders the runs inside them.
 * `numeric` so `rev10` follows `rev2` rather than preceding it.
 */
function sortedFolders(node: TreeNode): TreeNode[] {
  return [...node.folders.values()].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true }),
  );
}

type RunSidebarProps = {
  runs: RunSummary[];
  error: string | null;
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  /** Re-reads `runs/` and reloads whatever is selected. */
  onRefresh: () => void;
  /** A refresh is in flight; the icon spins and the button is inert. */
  refreshing: boolean;
};

function RunRow({
  run,
  depth,
  selected,
  onSelect,
}: {
  run: RunSummary;
  depth: number;
  selected: boolean;
  onSelect: (runId: string) => void;
}): JSX.Element {
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(run.run_id)}
        style={{ paddingLeft: `${12 + depth * 12}px` }}
        className={cn(
          "w-full space-y-0.5 border-b border-[var(--border-subtle)] py-2 pr-3 text-left",
          selected
            ? "bg-[var(--bg-tertiary)]"
            : "hover:bg-[var(--bg-secondary)]",
        )}
      >
        <p className="line-clamp-2 text-[11px] leading-snug text-[var(--text-primary)]">
          {run.label ?? run.prompt ?? run.name}
        </p>
        <p className="flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-[var(--text-quaternary)]">
          <span className={OUTCOME_STYLE[run.outcome ?? ""] ?? undefined}>
            {run.outcome ?? "running"}
          </span>
          {run.layer ? <span>{run.layer}</span> : null}
          {run.model ? <span>{run.model}</span> : null}
          {run.reasoning_effort ? (
            <span>effort:{run.reasoning_effort}</span>
          ) : null}
          {run.has_ir ? (
            <span>
              {run.ir_kind === "scene"
                ? `${run.n_instances}i`
                : run.ir_kind === "animation"
                  ? // Frames first: it is what distinguishes two animations of
                    // the same scene, which the instance count cannot.
                    `${run.n_frames}f/${run.n_instances}i`
                  : `${run.n_parts}p/${run.n_joints}j`}
            </span>
          ) : null}
          {/* Cost of the run, in the two units that actually differ between
              arms of the same experiment: how long it took and how much the
              model read and wrote. Both are omitted rather than zeroed when
              the bundle never recorded them. */}
          {shortWall(run.wall_seconds) ? (
            <span>{shortWall(run.wall_seconds)}</span>
          ) : null}
          {run.tokens ? (
            <span
              title={
                `${run.tokens.turns} turns · ${run.tokens.gross_input.toLocaleString()} in · ${run.tokens.output.toLocaleString()} out` +
                (run.tokens.subagents
                  ? ` · sub-agents: ${run.tokens.subagents.threads} threads, ${run.tokens.subagents.total.toLocaleString()} tokens`
                  : "")
              }
            >
              {shortCount(run.tokens.gross_input)}&#8594;
              {shortCount(run.tokens.output)}
              {run.tokens.subagents ? ` +${shortCount(run.tokens.subagents.total)}` : null}
            </span>
          ) : null}
          <span>{shortTime(run.started_at)}</span>
        </p>
      </button>
    </li>
  );
}

function TreeLevel({
  node,
  depth,
  expanded,
  toggle,
  selectedRunId,
  onSelect,
}: {
  node: TreeNode;
  depth: number;
  expanded: Set<string>;
  toggle: (path: string) => void;
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}): JSX.Element {
  return (
    <ul>
      {sortedFolders(node).map((folder) => {
        const isOpen = expanded.has(folder.path);
        return (
          <li key={folder.path}>
            <button
              type="button"
              onClick={() => toggle(folder.path)}
              style={{ paddingLeft: `${12 + depth * 12}px` }}
              className="flex w-full items-center gap-1.5 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)] py-1.5 pr-3 text-left font-mono text-[10px] uppercase tracking-wide text-[var(--text-tertiary)] hover:text-[var(--text-primary)]"
            >
              <span className="inline-block w-2 text-[9px]">
                {isOpen ? "▾" : "▸"}
              </span>
              <span className="truncate normal-case tracking-normal">
                {folder.name}
              </span>
              <span className="ml-auto text-[var(--text-quaternary)]">
                {folder.count}
              </span>
            </button>
            {/*
              The run that *is* this folder stays visible while it is collapsed.
              Collapsing a parent run means "hide the runs it started", not
              "hide the run itself" — and since the folder is named after that
              run, hiding it would leave a header there is no way to select.
            */}
            {folder.selfRun ? (
              <ul>
                <RunRow
                  run={folder.selfRun}
                  depth={depth + 1}
                  selected={folder.selfRun.run_id === selectedRunId}
                  onSelect={onSelect}
                />
              </ul>
            ) : null}
            {isOpen ? (
              <TreeLevel
                node={folder}
                depth={depth + 1}
                expanded={expanded}
                toggle={toggle}
                selectedRunId={selectedRunId}
                onSelect={onSelect}
              />
            ) : null}
          </li>
        );
      })}
      {node.runs.map((run) => (
        <RunRow
          key={run.run_id}
          run={run}
          depth={depth}
          selected={run.run_id === selectedRunId}
          onSelect={onSelect}
        />
      ))}
    </ul>
  );
}

export function RunSidebar({
  runs,
  error,
  selectedRunId,
  onSelect,
  onRefresh,
  refreshing,
}: RunSidebarProps): JSX.Element {
  const tree = useMemo(() => buildTree(runs), [runs]);
  // Folders start collapsed, and the state tracks which are *open* rather than
  // which are shut. Tracking the shut ones would mean seeding the set with
  // every folder path — and `runs` arrives asynchronously, so there is nothing
  // to seed from on the first render. Empty-means-collapsed needs no seeding,
  // and a folder that appears on a later refetch is collapsed for free instead
  // of springing open.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (path: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (!next.delete(path)) next.add(path);
      return next;
    });

  return (
    <div className="flex h-full min-h-0 flex-col border-r border-[var(--border-subtle)]">
      {/*
        The list is fetched once on mount, so a run that finishes — or a whole
        corpus that is staged — while the viewer is open is invisible until it
        is re-read. Refresh re-reads it without a restart, which would lose the
        selection and every expanded folder.
      */}
      <div className="flex shrink-0 items-center gap-2 px-3 py-2 font-mono text-[10px] uppercase tracking-wide text-[var(--text-quaternary)]">
        <span>Runs ({runs.length})</span>
        <button
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
          title="Re-read runs/ from disk and reload the selected run"
          aria-label="Refresh run list"
          className="ml-auto rounded p-0.5 hover:bg-[var(--bg-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50"
        >
          <RefreshCw className={cn("size-3", refreshing && "animate-spin")} />
        </button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        {error ? (
          <p className="p-3 text-[11px] text-red-400">{error}</p>
        ) : runs.length === 0 ? (
          <p className="p-3 text-[11px] text-[var(--text-tertiary)]">
            No runs found.
          </p>
        ) : (
          <TreeLevel
            node={tree}
            depth={0}
            expanded={expanded}
            toggle={toggle}
            selectedRunId={selectedRunId}
            onSelect={onSelect}
          />
        )}
      </ScrollArea>
    </div>
  );
}
