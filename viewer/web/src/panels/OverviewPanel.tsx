// The run at a glance: the prompt that produced it, meta.json, and the
// per-compile verifier verdicts (including exact-collision adjudication, which
// is the number that says how many SAT reports were bounding-box artifacts).

import { ChevronRight } from "lucide-react";
import { useCallback, useState, type JSX } from "react";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type {
  AnimationSummary,
  CompileReport,
  RunDetail,
  SubRunSummary,
} from "@/lib/types";

const VERDICT_STYLE: Record<string, string> = {
  pass: "text-emerald-500",
  fail: "text-red-400",
  uncertain: "text-amber-500",
};

// Keys rendered as the header badges rather than in the raw meta table.
const PROMOTED_META = new Set([
  "run_id",
  "prompt",
  "layer",
  "model",
  "reasoning_effort",
  "runtime",
  "outcome",
]);

//: Which sections the reader has opened. Persisted because "collapsed by
//: default" is only tolerable if the one section you actually read stays open —
//: otherwise every run selection costs the same click again.
const SECTIONS_STORAGE_KEY = "dycraft-overview-sections";

/**
 * The persisted key for a section, with any live count stripped.
 *
 * Several titles carry one (`Slots (3)`, `Compile reports (5)`) and it changes
 * from run to run, so keying on the raw title would file the same section under
 * a new name every time the count moved — i.e. silently forget that it was open.
 */
function sectionKey(title: string): string {
  return title.replace(/\s*\(\d+\)$/, "");
}

function readOpenSections(): Record<string, boolean> {
  try {
    const parsed = JSON.parse(localStorage.getItem(SECTIONS_STORAGE_KEY) ?? "");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    // Absent, unparseable, or storage denied — everything starts collapsed.
    return {};
  }
}

/**
 * Collapsed by default: this panel stacks eleven sections, most of which are
 * long (a full prompt, every compile report, raw meta.json), and reading any
 * one of them meant scrolling past the others. Headers alone make the tab a
 * table of contents.
 *
 * Read from storage in the initializer rather than in an effect so the section
 * never renders open-then-closed on mount.
 */
function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}): JSX.Element {
  const key = sectionKey(title);
  const [open, setOpen] = useState(() => readOpenSections()[key] === true);

  const toggle = useCallback(() => {
    setOpen((current) => {
      const next = !current;
      try {
        localStorage.setItem(
          SECTIONS_STORAGE_KEY,
          // Re-read rather than merging into a captured copy: two panels can
          // mount over the life of a tab, and the stale one would drop the
          // other's writes.
          JSON.stringify({ ...readOpenSections(), [key]: next }),
        );
      } catch {
        // Storage denied. The toggle still works, it just will not persist.
      }
      return next;
    });
  }, [key]);

  return (
    <section className="space-y-1.5">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="group flex w-full items-center gap-1 text-left"
      >
        <ChevronRight
          aria-hidden="true"
          className={cn(
            "h-2.5 w-2.5 shrink-0 text-[var(--text-quaternary)] transition-transform duration-150",
            open ? "rotate-90" : "rotate-0",
          )}
        />
        <h3 className="font-mono text-[10px] uppercase tracking-wide text-[var(--text-quaternary)] transition-colors group-hover:text-[var(--text-tertiary)]">
          {title}
        </h3>
      </button>
      {open ? children : null}
    </section>
  );
}

function AdjudicationLine({ report }: { report: CompileReport }): JSX.Element | null {
  const adj = report.exact_adjudication;
  if (!adj || !adj.available) return null;
  return (
    <span className="font-mono text-[10px] text-[var(--text-tertiary)]">
      exact: {adj.considered ?? 0} considered · {adj.cleared ?? 0} cleared ·{" "}
      {adj.confirmed ?? 0} confirmed
      {adj.unknown ? ` · ${adj.unknown} unknown` : ""}
    </span>
  );
}

/**
 * What the animation declares, and whether it held.
 *
 * The scrubber marks satisfied events, but an *unmet* one has no frame to mark
 * — and that is the case worth seeing, since `EventsSatisfied` failing is how a
 * motion that does not match its description shows up. A committed run has none
 * by definition; a run that stopped short can.
 *
 * Renders nothing when the run declares no events and its scene matches: frames
 * and fps are already on the Run badge, so the heading would stand alone.
 */
function AnimationSection({
  animation,
}: {
  animation: AnimationSummary;
}): JSX.Element | null {
  const sceneMismatch = animation.scene_ref_ok === false;
  if (!sceneMismatch && animation.events.length === 0) return null;
  return (
    <Section title="Animation">
      <div className="space-y-1">
        {sceneMismatch ? (
          <p className="font-mono text-[10px] text-amber-500">
            scene.json is not the scene this IR was built against (
            {animation.scene_ref}) — every pose shown is misplaced
          </p>
        ) : null}
        <ul className="space-y-0.5">
          {animation.events.map((event) => (
            <li
              key={event.name ?? ""}
              className="flex items-center gap-2 font-mono text-[10px]"
            >
              <span
                className={
                  event.satisfied_at == null
                    ? "text-[var(--destructive)]"
                    : "text-[var(--success)]"
                }
              >
                {event.satisfied_at == null
                  ? "unmet"
                  : `t=${event.satisfied_at}`}
              </span>
              <span className="text-[var(--text-secondary)]">{event.name}</span>
              <span className="text-[var(--text-quaternary)]">
                window [{event.window.join(", ")}]
              </span>
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}

function SubRunsSection({
  subRuns,
  onSelectRun,
}: {
  subRuns: SubRunSummary[];
  onSelectRun?: (runId: string) => void;
}): JSX.Element {
  return (
    <Section title={`Sub-runs (${subRuns.length})`}>
      <ul className="space-y-0.5">
        {subRuns.map((sub) => (
          <li key={sub.run_id}>
            <button
              type="button"
              disabled={!onSelectRun}
              onClick={() => onSelectRun?.(sub.run_id)}
              className="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left font-mono text-[10px] hover:bg-[var(--bg-secondary)] disabled:hover:bg-transparent"
            >
              <span className="text-sky-500">{sub.name}</span>
              <span className="text-[var(--text-quaternary)]">{sub.layer}</span>
              <span
                className={
                  sub.outcome === "committed"
                    ? "ml-auto text-emerald-500"
                    : "ml-auto text-amber-500"
                }
              >
                {sub.outcome ?? "running"}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </Section>
  );
}

export function OverviewPanel({
  detail,
  onSelectRun,
  installed = false,
}: {
  detail: RunDetail;
  onSelectRun?: (runId: string) => void;
  /**
   * A film installed from the published set: it has no compile reports, and its `meta.json` is the record the
   * viewer made for it from `run.json`, so neither section is shown.
   */
  installed?: boolean;
}): JSX.Element {
  const meta = detail.meta ?? {};
  const rest = Object.entries(meta)
    .filter(([key]) => !PROMOTED_META.has(key))
    .sort(([a], [b]) => a.localeCompare(b));

  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 p-3">
        <Section title="Prompt">
          <p className="whitespace-pre-wrap text-[12px] leading-relaxed text-[var(--text-primary)]">
            {(meta.prompt as string) ?? "—"}
          </p>
        </Section>

        <Section title="Run">
          <div className="flex flex-wrap items-center gap-1.5">
            {meta.outcome ? (
              <Badge
                variant={meta.outcome === "committed" ? "success" : "secondary"}
              >
                {String(meta.outcome)}
              </Badge>
            ) : null}
            {meta.layer ? <Badge variant="secondary">{String(meta.layer)}</Badge> : null}
            {meta.model ? <Badge variant="secondary">{String(meta.model)}</Badge> : null}
            {detail.reasoning_effort ? (
              <Badge variant="secondary">
                effort {detail.reasoning_effort}
              </Badge>
            ) : null}
            {meta.runtime ? <Badge variant="secondary">{String(meta.runtime)}</Badge> : null}
            {detail.ir ? (
              <Badge variant="secondary">
                {detail.ir.kind === "scene"
                  ? `${detail.ir.instances?.length ?? 0} instances / ` +
                    `${detail.ir.parts.length}p`
                  : detail.ir.kind === "animation"
                    ? // Parts and joints here belong to the staged scene, so
                      // reporting them would read as this run's own output.
                      `${detail.ir.animation?.n_frames ?? 0} frames @ ` +
                      `${detail.ir.animation?.fps ?? "?"} fps`
                    : `${detail.ir.parts.length}p / ${detail.ir.joints.length}j`}
              </Badge>
            ) : null}
            {(detail.ir?.kind === "scene" || detail.ir?.kind === "animation") &&
            (detail.ir.catalog_total ?? 0) >
              (detail.ir.catalog_resolved ?? 0) ? (
              <Badge variant="secondary">
                <span className="text-amber-500">
                  {detail.ir.catalog_resolved ?? 0}/{detail.ir.catalog_total}{" "}
                  in catalog
                </span>
              </Badge>
            ) : null}
          </div>
        </Section>

        {detail.ir?.animation ? (
          <AnimationSection animation={detail.ir.animation} />
        ) : null}

        {detail.sub_runs.length > 0 ? (
          <SubRunsSection
            subRuns={detail.sub_runs}
            onSelectRun={onSelectRun}
          />
        ) : null}

        {installed ? null : (
        <Section title={`Compile reports (${detail.compile_reports.length})`}>
          {detail.compile_reports.length === 0 ? (
            <p className="text-[11px] text-[var(--text-tertiary)]">
              None.
            </p>
          ) : (
            <div className="space-y-2">
              {detail.compile_reports.map((report) => (
                <div
                  key={report.step ?? Math.random()}
                  className="rounded-md border border-[var(--border-subtle)] p-2"
                >
                  <div className="flex items-center gap-2 pb-1">
                    <span className="font-mono text-[11px] text-[var(--text-primary)]">
                      step {report.step ?? "?"}
                    </span>
                    <Badge variant={report.ok ? "success" : "secondary"}>
                      {report.ok ? "ok" : "failed"}
                    </Badge>
                    <AdjudicationLine report={report} />
                  </div>
                  <ul className="space-y-0.5">
                    {report.verifiers.map((verifier) => (
                      <li
                        key={verifier.name ?? Math.random()}
                        className="font-mono text-[10px]"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-[var(--text-tertiary)]">
                            {verifier.name}
                          </span>
                          <span
                            className={
                              VERDICT_STYLE[verifier.verdict ?? ""] ??
                              "text-[var(--text-quaternary)]"
                            }
                          >
                            {verifier.verdict}
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          )}
        </Section>
        )}

        {detail.ir && detail.ir.collision_allowances.length > 0 ? (
          <Section title="Collision allowances">
            <ul className="space-y-1">
              {detail.ir.collision_allowances.map((allowance) => (
                <li key={`${allowance.a}-${allowance.b}`} className="text-[10px]">
                  <span className="font-mono text-[var(--text-primary)]">
                    {allowance.a} × {allowance.b}
                  </span>
                  {allowance.reason ? (
                    <span className="text-[var(--text-tertiary)]">
                      {" "}
                      — {allowance.reason}
                    </span>
                  ) : null}
                </li>
              ))}
            </ul>
          </Section>
        ) : null}

        {installed ? null : (
        <Section title="meta.json">
          <dl className="space-y-0.5">
            {rest.map(([key, value]) => (
              <div
                key={key}
                className="flex items-start justify-between gap-3 font-mono text-[10px]"
              >
                <dt className="shrink-0 text-[var(--text-quaternary)]">{key}</dt>
                <dd className="min-w-0 break-words text-right text-[var(--text-tertiary)]">
                  {typeof value === "object" && value !== null
                    ? JSON.stringify(value)
                    : String(value)}
                </dd>
              </div>
            ))}
          </dl>
        </Section>
        )}
      </div>
    </ScrollArea>
  );
}
