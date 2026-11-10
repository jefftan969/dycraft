// The viewport's loading tray: one row per item still loading for the selected run (`lib/loading.ts`).
//
// Always on top of the scene and never in its way: it takes no pointer events, and the canvas stays orbitable under
// it. It is a DOM overlay, so a snapshot (`canvas.toBlob`) never contains it.

import { type JSX } from "react";

import { useLoadingRows } from "@/lib/loading";
import { cn } from "@/lib/utils";

/** `inline`: placed by its parent (above the film in the viewport's corner) rather than in the corner itself. */
export function LoadingTray({ inline = false }: { inline?: boolean }): JSX.Element | null {
  const rows = useLoadingRows();
  if (rows.length === 0) return null;
  return (
    <div
      className={cn("pointer-events-none z-[7] w-[15rem] space-y-1 rounded bg-black/55 px-2 py-1.5",
                    !inline && "absolute bottom-2 right-2")}
      role="status"
      aria-live="polite"
    >
      {rows.map((row) => (
        <div key={row.key}>
          <div className="flex items-baseline justify-between gap-2 font-mono text-[9px] leading-tight text-white/90">
            <span>{row.label}</span>
            <span className="truncate text-white/70">{row.detail}</span>
          </div>
          <div className="mt-0.5 h-[2px] overflow-hidden rounded bg-white/15">
            {row.fraction === undefined ? (
              <div className="h-full w-1/3 animate-pulse bg-white/60" />
            ) : (
              <div className="h-full bg-white/80" style={{ width: `${Math.round(row.fraction * 100)}%` }} />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
