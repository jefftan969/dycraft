// The film beside its 3D playback: the rendered clip in a corner of the viewport, kept to the playhead, as the project
// page shows it. In the film's view the two show the same shot, the render and the evaluated geometry side by side;
// in a free view the clip is still there to compare against.
//
// The clip follows the playhead, never the other way round. Playing, it runs at the playback rate and is moved only
// when it has drifted (a seek costs a decode, so small drift is left alone). When the playhead stops advancing while
// playing (the 3D waiting for its meshes), the clip pauses too rather than running ahead; when the playhead goes
// back (the loop, a seek, a step), the clip goes with it. Stopped, it shows the playhead's frame. Muted, as the
// page's are.

import { useEffect, useRef, type JSX } from "react";

import { cn } from "@/lib/utils";

interface FilmPipProps {
  /** The rendered clip's URL. */
  src: string;
  /** The playhead's frame, and the clip's frames a second: the clip's time is `frame / fps`. */
  frame: number;
  fps: number;
  playing: boolean;
  rate: number;
  big: boolean;
  onToggleBig: () => void;
  onHide: () => void;
}

/** How far the clip may drift behind the playhead before it is moved, in seconds of film, at a rate. */
const behindTolerance = (rate: number) => 0.3 + 0.1 * rate;
/** How far ahead it may get before it waits for the playhead (the 3D buffering). */
const AHEAD_TOLERANCE_S = 0.2;

export function FilmPip({ src, frame, fps, playing, rate, big, onToggleBig, onHide }: FilmPipProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // The playhead's last frame and when it changed: a frame earlier than the last is a jump back, and no change for a
  // few frames' time while playing is the 3D waiting.
  const lastFrame = useRef(frame);
  const changedAt = useRef(performance.now());

  useEffect(() => {
    if (!playing) return;
    const stall = Math.max(300, (2.5 * 1000) / Math.max(1e-3, fps * rate));
    const timer = window.setInterval(() => {
      const video = videoRef.current;
      if (video && !video.paused && performance.now() - changedAt.current > stall) video.pause();
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing, fps, rate]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = rate;
  }, [rate]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !(fps > 0) || video.readyState < 1) return;
    const target = frame / fps;
    const back = frame < lastFrame.current;
    lastFrame.current = frame;
    changedAt.current = performance.now();
    if (!playing) {
      if (!video.paused) video.pause();
      if (Math.abs(video.currentTime - target) > 0.5 / fps) video.currentTime = target;
      return;
    }
    const drift = video.currentTime - target;
    // The playhead went back (the loop came round, a seek or a step back), or the clip is far ahead of it: the clip
    // goes there too.
    if ((back || drift > Math.max(1, rate)) && !video.seeking) {
      video.currentTime = target;
    } else if (drift > AHEAD_TOLERANCE_S) {
      // A little ahead, the playhead is waiting for meshes: the clip waits with it.
      if (!video.paused) video.pause();
      return;
    }
    if (drift < -behindTolerance(rate) && !video.seeking) video.currentTime = target;
    if (video.paused) void video.play().catch(() => {});
  }, [frame, fps, playing, rate]);

  return (
    <div className={cn("pointer-events-auto relative aspect-video w-full overflow-hidden rounded-md border-2",
                       "border-white/80 bg-black shadow-md", big ? "cursor-zoom-out" : "cursor-zoom-in")}
         title={big ? "Click to make the film smaller" : "Click to enlarge the film"}
         onClick={onToggleBig}>
      <video
        ref={videoRef}
        key={src}
        src={src}
        muted
        playsInline
        preload="auto"
        className="absolute inset-0 h-full w-full object-contain"
        onLoadedMetadata={(event) => {
          event.currentTarget.playbackRate = rate;
          if (fps > 0) event.currentTarget.currentTime = frame / fps;
        }}
      />
      <span className="pointer-events-none absolute left-1.5 top-1 text-[10px] font-medium text-white
                       [text-shadow:0_1px_3px_rgba(0,0,0,0.8)]">
        Rendered film
      </span>
      <button
        type="button"
        aria-label="Hide the film"
        title="Hide the film"
        onClick={(event) => {
          event.stopPropagation();
          onHide();
        }}
        className="absolute right-1 top-0.5 rounded px-1 text-[12px] leading-none text-white/90
                   [text-shadow:0_1px_3px_rgba(0,0,0,0.8)] hover:bg-black/40"
      >
        ×
      </button>
    </div>
  );
}
