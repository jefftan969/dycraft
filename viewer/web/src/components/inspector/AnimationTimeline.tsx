// Transport bar under the 3D viewport for an animation run.
//
// The event ticks are the reason this is not just a slider. An animation's
// declared events are what make "does the motion match the description"
// checkable at all (design_dynamics.md), and `EventsSatisfied` already computed
// the frame each one fired at. Drawing those against the playhead turns the
// verifier's verdict into something you can scrub to: park on
// `lands_on_table`'s frame and see whether the mug is actually on the table.

import { useState, type JSX } from "react";

import { Button } from "@/components/ui/button";
import type { Shot } from "@/components/viewer3d/film-camera";
import { PLAYBACK_RATES } from "@/components/viewer3d/useAnimationPlayback";
import type { AnimationEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

interface AnimationTimelineProps {
  frame: number;
  nFrames: number;
  fps: number;
  playing: boolean;
  rate: number;
  events: AnimationEvent[];
  onSetFrame: (frame: number) => void;
  onStep: (delta: number) => void;
  onTogglePlay: () => void;
  onSetRate: (rate: number) => void;
  /** Whether playback starts again at the end (`useAnimationPlayback`); the toggle shows when given. */
  loop?: boolean;
  onToggleLoop?: () => void;
  /** The film's shots (`shotsOf`), drawn as a band under the scrubber; a click goes to a shot's first frame. */
  shots?: Shot[];
  /** The storyboard's keyframes, in the playhead's frames, marked under the shots; hovering shows its view. */
  keyframes?: { name: string; frame: number; camera: string | null; index: number }[];
  keyframeThumb?: (index: number) => string;
}

/** The keyframe nearest frame `f`, if one is within `reach` frames. */
function nearestKeyframe<K extends { frame: number }>(keys: K[], f: number, reach: number): K | null {
  let best: K | null = null;
  for (const k of keys) {
    const d = Math.abs(k.frame - f);
    if (d <= reach && (!best || d < Math.abs(best.frame - f))) best = k;
  }
  return best;
}

export function AnimationTimeline({
  frame,
  nFrames,
  fps,
  playing,
  rate,
  events,
  onSetFrame,
  onStep,
  onTogglePlay,
  onSetRate,
  loop,
  onToggleLoop,
  shots = [],
  keyframes = [],
  keyframeThumb,
}: AnimationTimelineProps): JSX.Element | null {
  const [hovered, setHovered] = useState<number | null>(null);
  if (nFrames <= 0) return null;

  const last = Math.max(1, nFrames - 1);
  const seconds = fps > 0 ? frame / fps : 0;
  const duration = fps > 0 ? last / fps : 0;

  // An unmet event has no frame — it is exactly the case where nothing on the
  // clip is worth pointing at, so it gets no tick and is left to the Overview
  // panel to explain.
  const ticks = events
    .filter((event) => typeof event.satisfied_frame === "number")
    .map((event) => ({
      name: event.name ?? "",
      frame: event.satisfied_frame as number,
    }));

  const nearest = ticks.find((tick) => tick.frame === frame);
  const shot = shots.find((s) => s.start <= frame && frame <= s.end);
  const keyframe = keyframes.find((k) => k.frame === frame);
  const hoveredKey = hovered === null ? undefined : keyframes.find((k) => k.index === hovered);
  const at = (f: number) => `${(Math.max(0, Math.min(last, f)) / last) * 100}%`;

  return (
    <div className="flex flex-col gap-1 border-t border-[var(--border-subtle)] bg-[var(--surface-1)] px-3 py-2">
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 font-mono text-[11px]"
          title="First frame"
          onClick={() => onSetFrame(0)}
        >
          ⏮
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 font-mono text-[11px]"
          title="Previous frame"
          onClick={() => onStep(-1)}
        >
          ◀
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 font-mono text-[11px]"
          title={playing ? "Pause" : "Play"}
          onClick={onTogglePlay}
        >
          {playing ? "⏸" : "▶"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 font-mono text-[11px]"
          title="Next frame"
          onClick={() => onStep(1)}
        >
          ▶|
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 font-mono text-[11px]"
          title="Last frame"
          onClick={() => onSetFrame(last)}
        >
          ⏭
        </Button>

        <div className="relative min-w-[4rem] flex-1">
          <input
            type="range"
            min={0}
            max={last}
            step={1}
            value={frame}
            onChange={(event) => onSetFrame(Number(event.target.value))}
            className="w-full accent-[var(--accent)]"
            aria-label="Animation frame"
          />
          {/* The film's shots, then its storyboard's keyframes, in two thin rows under the track. */}
          {shots.length > 0 ? (
            <div className="absolute inset-x-0 top-[calc(100%+3px)] h-[4px]" aria-label="Shots">
              {shots.map((s, k) => (
                <button
                  key={`${s.start}-${s.name}`}
                  type="button"
                  title={`${s.name} · f${s.start}–f${s.end}`}
                  onClick={() => onSetFrame(s.start)}
                  className={cn("absolute top-0 h-full border-l border-[var(--surface-1)]",
                                s === shot ? "bg-[var(--accent)]"
                                  : k % 2 ? "bg-[var(--text-quaternary)]/40" : "bg-[var(--text-quaternary)]/70")}
                  style={{ left: at(s.start), width: `calc(${at(s.end + 1)} - ${at(s.start)})` }}
                />
              ))}
            </div>
          ) : null}
          {keyframes.length > 0 ? (
            // The row, not each marker, takes the pointer: it picks the keyframe nearest the pointer, so keyframes a
            // few frames apart, which overlap at the scrubber's width, each have a slice of it.
            // From the keyboard: Left and Right step between keyframes, showing each, and Enter goes to one.
            <div
              className="absolute inset-x-0 top-[calc(100%+7px)] h-[10px] cursor-pointer rounded-sm outline-none
                         focus-visible:ring-1 focus-visible:ring-[var(--accent)]"
              role="listbox"
              aria-orientation="horizontal"
              tabIndex={0}
              aria-label="Storyboard keyframes"
              aria-activedescendant={hoveredKey ? `keyframe-${hoveredKey.index}` : undefined}
              onFocus={() => {
                // Functional: a click's own pointer move has just picked a keyframe, in this same event loop turn,
                // and that one stays; focus from the keyboard starts at the playhead.
                setHovered((h) => h ?? nearestKeyframe(keyframes, frame, Infinity)?.index ?? null);
              }}
              onBlur={() => setHovered(null)}
              onKeyDown={(event) => {
                const at = keyframes.findIndex((k) => k.index === hovered);
                if (!["ArrowRight", "ArrowLeft", "Enter", " "].includes(event.key)) return;
                // These are the camera's flight keys too (`useCameraKeys`, on the window): handled here, they go no
                // further, or Right would also fly the camera off the film's view.
                event.preventDefault();
                event.stopPropagation();
                if (event.key === "Enter" || event.key === " ") {
                  if (hoveredKey) onSetFrame(hoveredKey.frame);
                  return;
                }
                const right = event.key === "ArrowRight";
                // With none shown (after a click, the pointer gone), from where the film is: the next keyframe after
                // the playhead, or the last before it.
                const next = at >= 0
                  ? keyframes[Math.max(0, Math.min(keyframes.length - 1, at + (right ? 1 : -1)))]
                  : right ? keyframes.find((k) => k.frame > frame) ?? keyframes[keyframes.length - 1]
                    : [...keyframes].reverse().find((k) => k.frame < frame) ?? keyframes[0];
                setHovered(next.index);
              }}
              onPointerMove={(event) => {
                const box = event.currentTarget.getBoundingClientRect();
                const f = ((event.clientX - box.left) / Math.max(1, box.width)) * last;
                const near = nearestKeyframe(keyframes, f, (8 / Math.max(1, box.width)) * last);
                setHovered(near ? near.index : null);
              }}
              onPointerLeave={() => setHovered(null)}
              onClick={() => {
                if (hoveredKey) onSetFrame(hoveredKey.frame);
              }}
            >
              {keyframes.map((k) => (
                <span
                  key={k.index}
                  id={`keyframe-${k.index}`}
                  role="option"
                  aria-selected={k.index === hovered}
                  aria-label={`${k.name}, frame ${k.frame}`}
                  className={cn("pointer-events-none absolute top-[2px] h-[6px] w-[6px] -translate-x-1/2 rotate-45",
                                "rounded-[1px]",
                                k.frame === frame || k.index === hovered ? "bg-[var(--accent)]"
                                  : "bg-[var(--text-tertiary)]")}
                  style={{ left: at(k.frame) }}
                />
              ))}
            </div>
          ) : null}
          {hoveredKey && keyframeThumb ? (
            <div className="pointer-events-none absolute bottom-[calc(100%+6px)] z-20 w-[164px] -translate-x-1/2
                            rounded border border-[var(--border-default)] bg-[var(--surface-0)] p-0.5 shadow-md"
                 style={{ left: at(hoveredKey.frame) }}>
              <img src={keyframeThumb(hoveredKey.index)} alt="" className="block aspect-video w-full rounded-sm bg-black" />
              <p className="truncate px-0.5 pt-0.5 font-mono text-[9px] text-[var(--text-secondary)]">
                {hoveredKey.name} · f{hoveredKey.frame}{hoveredKey.camera ? ` · ${hoveredKey.camera}` : ""}
              </p>
            </div>
          ) : null}
          {/* Ticks sit under the track, positioned by frame fraction. */}
          <div className="pointer-events-none absolute inset-x-0 bottom-[-2px] h-[5px]">
            {ticks.map((tick) => (
              <span
                key={`${tick.name}-${tick.frame}`}
                title={`${tick.name} @ f${tick.frame}`}
                className={cn(
                  "absolute top-0 h-[5px] w-[2px] -translate-x-1/2 rounded-sm",
                  tick.frame === frame
                    ? "bg-[var(--accent)]"
                    : "bg-[var(--text-quaternary)]",
                )}
                style={{ left: `${(tick.frame / last) * 100}%` }}
              />
            ))}
          </div>
        </div>

        {/* Shrinks before the scrubber does: on a narrow viewport the playhead's slider matters more than its time. */}
        <span className="min-w-0 shrink truncate font-mono text-[11px] tabular-nums text-[var(--text-tertiary)]">
          f{String(frame).padStart(3, "0")}/{last} · {seconds.toFixed(3)}s /{" "}
          {duration.toFixed(2)}s
        </span>

        {/* Slow motion holds each authored frame longer — it never interpolates
            between two of them, because nothing between frame 8 and frame 9 is
            something the run committed to. */}
        <select
          value={rate}
          onChange={(event) => onSetRate(Number(event.target.value))}
          title="Playback rate — slow motion holds each authored frame longer"
          aria-label="Playback rate"
          className="h-6 shrink-0 cursor-pointer rounded border border-[var(--border-default)] bg-[var(--surface-0)] px-1 font-mono text-[10px] tabular-nums text-[var(--text-secondary)] outline-none focus-visible:border-[var(--accent)]"
        >
          {PLAYBACK_RATES.map((r) => (
            <option key={r} value={r}>
              {r}x
            </option>
          ))}
        </select>

        {onToggleLoop ? (
          <Button
            variant="ghost"
            size="sm"
            className={cn("h-6 shrink-0 px-1.5 font-mono text-[12px]",
                          loop ? "text-[var(--accent)]" : "text-[var(--text-quaternary)]")}
            title={loop ? "Loop on: plays again from the start at the end" : "Loop off: stops at the end"}
            aria-label="Loop"
            aria-pressed={Boolean(loop)}
            onClick={onToggleLoop}
          >
            ↻
          </Button>
        ) : null}
      </div>

      <p className={cn("h-[13px] truncate text-[10px] text-[var(--text-tertiary)]",
                       (shots.length > 0 || keyframes.length > 0) && "mt-[14px]")}>
        {[nearest ? `▲ ${nearest.name}` : "", shot ? `shot ${shot.name}` : "",
          keyframe ? `storyboard ${keyframe.name}` : ""].filter(Boolean).join(" · ")}
      </p>
    </div>
  );
}
