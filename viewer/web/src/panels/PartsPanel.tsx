// Per-part geometry facts, then the movable-joint sliders.
//
// `fill_ratio` (volume / bounding-box volume) is the one number that says how
// badly an AABB misrepresents a part: a shelled carcass or a half-ring handle
// reports ~0.1, meaning ~90% of the box the collision broad-phase reasons about
// is air. It exists only for `solid` parts, derived from the B-rep at authoring
// time.

import { useMemo, type JSX } from "react";
import { FlipHorizontal2 } from "lucide-react";

import { JointPanel } from "@/components/inspector/JointPanel";
import type { UrdfJoint } from "@/components/inspector/JointSlider";
import { flippableJoints } from "@/components/viewer3d/joint-flip";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import type { PartSummary, SceneInstanceSummary } from "@/lib/types";

/** Below this, the bounding box is mostly air and SAT will over-report. */
const LOW_FILL = 0.35;

function censusLabel(census: Record<string, number> | undefined): string | null {
  if (!census) return null;
  return Object.entries(census)
    .sort(([, a], [, b]) => b - a)
    .map(([kind, count]) => `${kind}:${count}`)
    .join(" ");
}

function PartRow({
  part,
  onShowSource,
}: {
  part: PartSummary;
  onShowSource: (path: string) => void;
}): JSX.Element {
  const fill = part.fill_ratio;
  return (
    <div className="space-y-1 border-b border-[var(--border-subtle)] pb-2 last:border-0">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-[11px] text-[var(--text-primary)]">
          {part.name}
        </span>
        <Badge variant="secondary">{part.primitive}</Badge>
        {fill != null ? (
          <span
            className={`font-mono text-[10px] tabular-nums ${
              fill < LOW_FILL ? "text-amber-500" : "text-[var(--text-tertiary)]"
            }`}
            title="volume / bounding-box volume"
          >
            fill {fill.toFixed(3)}
          </span>
        ) : null}
        {part.volume != null ? (
          <span className="font-mono text-[10px] tabular-nums text-[var(--text-quaternary)]">
            {(part.volume * 1e6).toFixed(1)} cm³
          </span>
        ) : null}
      </div>
      {censusLabel(part.face_census) ? (
        <p className="font-mono text-[10px] text-[var(--text-quaternary)]">
          {censusLabel(part.face_census)}
        </p>
      ) : null}
      {part.source ? (
        <button
          type="button"
          onClick={() => onShowSource(part.source as string)}
          className="font-mono text-[10px] text-sky-500 underline-offset-2 hover:underline"
        >
          view build123d source
        </button>
      ) : null}
    </div>
  );
}

/**
 * A scene's placed objects, above its parts.
 *
 * `parent` is the point: a scene IR states placement relationally (mug
 * attached to table's `top` anchor), not as world coordinates, so the parent
 * chain is what explains where something ended up — and a placement bug shows
 * up here as the wrong parent or anchor long before it is visible in the
 * numbers. `in_catalog` is called out because an unresolved `object_ref` is
 * the one failure that makes an instance silently not draw at all.
 */
function InstanceRow({
  instance,
}: {
  instance: SceneInstanceSummary;
}): JSX.Element {
  return (
    <div className="space-y-0.5 border-b border-[var(--border-subtle)] pb-2 last:border-0">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-[11px] text-[var(--text-primary)]">
          {instance.name}
        </span>
        {instance.object_name && instance.object_name !== instance.name ? (
          <Badge variant="secondary">{instance.object_name}</Badge>
        ) : null}
        <span className="font-mono text-[10px] text-[var(--text-quaternary)]">
          {instance.n_parts}p/{instance.n_joints}j
        </span>
        {instance.in_catalog ? null : (
          <span
            className="font-mono text-[10px] text-amber-500"
            title="object_ref is not in this run's catalog/ — it will not draw"
          >
            not in catalog
          </span>
        )}
      </div>
      <p className="font-mono text-[10px] text-[var(--text-tertiary)]">
        {instance.placement_kind ?? "?"} → {instance.parent ?? "?"}
        {instance.parent_anchor ? `.${instance.parent_anchor}` : ""}
        {instance.child_anchor ? ` (self.${instance.child_anchor})` : ""}
      </p>
    </div>
  );
}

type PartsPanelProps = {
  parts: PartSummary[];
  /** Scene runs only; empty for an object run. */
  instances?: SceneInstanceSummary[];
  urdfSpec: { joints: UrdfJoint[] } | null;
  jointValues: Map<string, number>;
  onJointChange: (name: string, value: number) => void;
  onResetJoints: () => void;
  /** Mirrors every joint to the other side of its range; see `joint-flip`. */
  onFlipJoints: () => void;
  /**
   * True while Preview motion or a clip is writing the joints every frame.
   * A flip would be overwritten on the next frame, so the button says so
   * instead of appearing to do nothing.
   */
  motionDriven?: boolean;
  onShowSource: (path: string) => void;
};

export function PartsPanel({
  parts,
  instances,
  urdfSpec,
  jointValues,
  onJointChange,
  onResetJoints,
  onFlipJoints,
  motionDriven = false,
  onShowSource,
}: PartsPanelProps): JSX.Element {
  const flippable = useMemo(
    () => flippableJoints(urdfSpec?.joints ?? []),
    [urdfSpec],
  );
  const flipDisabled = flippable.length === 0 || motionDriven;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Above the scroll area, not inside it: on a scene with eighty
          instances the joints are hundreds of rows down, and this is the one
          control that acts on all of them at once. */}
      <div className="flex shrink-0 items-center gap-2 border-b border-[var(--border-subtle)] px-3 py-1.5">
        <Button
          variant="ghost"
          size="sm"
          onClick={onFlipJoints}
          disabled={flipDisabled}
          title={
            motionDriven
              ? "Preview motion or clip playback is driving the joints"
              : flippable.length === 0
                ? "No joint in this run has a range of motion to flip"
                : "Mirror every joint to the other side of its range at " +
                  "once; click again to put them back exactly"
          }
          className="h-6 gap-1 px-2 text-[11px] text-[var(--text-secondary)]"
        >
          <FlipHorizontal2 className="size-3" />
          Flip joints
        </Button>
        <span className="truncate font-mono text-[10px] text-[var(--text-quaternary)]">
          {flippable.length === 0
            ? "no joint has a range of motion"
            : `${flippable.length} joint${flippable.length === 1 ? "" : "s"} \u2192 mirrored in range`}
        </span>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-4 p-3">
          {instances && instances.length > 0 ? (
            <section className="space-y-2">
              <h3 className="font-mono text-[10px] uppercase tracking-wide text-[var(--text-quaternary)]">
                Instances ({instances.length})
              </h3>
              <div className="space-y-2">
                {instances.map((instance) => (
                  <InstanceRow
                    key={instance.name ?? Math.random()}
                    instance={instance}
                  />
                ))}
              </div>
            </section>
          ) : null}

          <section className="space-y-2">
            <h3 className="font-mono text-[10px] uppercase tracking-wide text-[var(--text-quaternary)]">
              Parts ({parts.length})
            </h3>
            {parts.length === 0 ? (
              <p className="text-[11px] text-[var(--text-tertiary)]">
                {instances && instances.length > 0
                  ? "None resolved — a scene's parts come from its catalog/, " +
                    "which is snapshotted next to the scene IR on a " +
                    "successful compile."
                  : "No IR — geometry is written when a run commits."}
              </p>
            ) : (
              <div className="space-y-2">
                {parts.map((part) => (
                  <PartRow
                    key={part.name ?? Math.random()}
                    part={part}
                    onShowSource={onShowSource}
                  />
                ))}
              </div>
            )}
          </section>

          <section className="space-y-2">
            <h3 className="font-mono text-[10px] uppercase tracking-wide text-[var(--text-quaternary)]">
              Joints
            </h3>
            <JointPanel
              urdfSpec={urdfSpec}
              jointValues={jointValues}
              onJointChange={onJointChange}
              onResetAll={onResetJoints}
            />
          </section>
        </div>
      </ScrollArea>
    </div>
  );
}
