// The rendered mp4 for an animation run.
//
// It is not redundant with scrubbing the IR in the 3D viewport, which is why
// both exist: the viewport draws the URDF's box/mesh proxies through the
// browser's material stack, while the clip is what the Filament renderer
// actually produced — the same pixels the pipeline would hand downstream. When
// the two disagree, that gap is the finding.

import { type JSX } from "react";

import { runFileUrl } from "@/lib/api";
import type { AnimationSummary } from "@/lib/types";

interface ClipPanelProps {
  runId: string;
  /** Run-root-relative path of the film (`bundle/film/clip.mp4`); null if unrendered. */
  clip: string | null;
  animation: AnimationSummary | undefined;
}

export function ClipPanel({
  runId,
  clip,
  animation,
}: ClipPanelProps): JSX.Element {
  if (!clip) {
    return (
      <p className="p-3 text-[11px] text-[var(--text-tertiary)]">
        No clip. `final_render` is a terminal op an animation agent may never
        reach — a run can commit its IR and still stop before rendering.
      </p>
    );
  }

  const fps = animation?.fps ?? null;
  const duration = animation?.duration ?? null;

  return (
    <div className="flex h-full min-h-0 flex-col gap-2 p-3">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[11px] text-[var(--text-secondary)]">
          {clip}
        </span>
        {fps ? (
          <span className="font-mono text-[10px] text-[var(--text-tertiary)]">
            {animation?.n_frames} frames · {fps} fps
            {duration ? ` · ${duration}s` : ""}
          </span>
        ) : null}
      </div>
      <video
        key={`${runId}/${clip}`}
        src={runFileUrl(runId, clip)}
        controls
        loop
        playsInline
        className="min-h-0 w-full rounded border border-[var(--border-default)] bg-black"
      />
    </div>
  );
}
