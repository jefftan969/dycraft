// A native clip's meshes, loaded around the playhead rather than all at once.
//
// A long film's deforming objects carry one mesh per frame: thousands of meshes, gigabytes once built, more than a
// tab can hold and minutes to build. So the meshes an object keeps for the whole clip load first, with the current
// frame's deforming meshes. Then, when the whole clip's estimated size fits the byte budget, the rest of it loads
// from the playhead on; otherwise a window around the playhead does, `BEHIND_S` back and `AHEAD_S` ahead, nearest
// first, ahead before behind. Past the budget, the meshes least recently wanted are dropped, never one that is in
// the window or on screen. Playback waits for `readyAhead`, the way a video player buffers.
//
// **On a slow link, every k-th frame.** A long film's characters at twice its speed are hundreds of megabits a
// second, more than a tunnel to a remote browser carries. When the link measured from the downloads cannot carry the
// deforming meshes at the playback rate, the stream fetches them only on every `stride`-th frame and each one is
// held over the frames between, the way a video player drops quality rather than stopping: the playhead keeps time,
// the characters move in steps. The frame on screen when playback stops still loads exactly (while it plays, the
// playhead's own frame is never fetched: it would be passed before it arrived). A link at least
// `FAST_LINK_BYTES_PER_S` (a browser on the same machine) never thins anything.

import type * as THREE from "three";

import type { Transfer } from "@/workers/loader-protocol";

import type { NativeTrack } from "./native-track";

const AHEAD_S = 4;
const BEHIND_S = 1;

/** Failures kept for the notes; past this the rest are counted. */
const MAX_NOTED_FAILURES = 5;

/** The strides the stream may thin deforming objects' frames to. */
export const STRIDES = [1, 2, 3, 4, 6, 8];
/** A link this fast is never thinned: about 400 Mbit/s, well above any tunnel and far below a loopback. */
export const FAST_LINK_BYTES_PER_S = 50e6;
/** How far back the link is measured. */
const LINK_WINDOW_MS = 4000;
/** How often the stride may change, so it does not chase every burst. */
const STRIDE_HOLD_MS = 2000;
/**
 * With too few downloads to judge the link by, and the window full, the stride comes down a step this often: little to
 * download means the link is keeping up, and a stride must not stay high after the link recovers.
 */
const STRIDE_DECAY_MS = 8000;

/**
 * The link's speed from recent downloads: their bytes over the time any of them was arriving (the union of their
 * first-to-last-byte spans), so downloads in parallel are not counted twice and the server's own work before the
 * first byte is not counted at all. Null until there is enough to go on.
 */
export function linkBytesPerSecond(transfers: Transfer[], now: number, windowMs = LINK_WINDOW_MS): number | null {
  const recent = transfers.filter((t) => t.end >= now - windowMs && t.end >= t.start).sort((a, b) => a.start - b.start);
  const bytes = recent.reduce((sum, t) => sum + t.bytes, 0);
  if (recent.length < 6 || bytes < 1e6) return null;
  let busy = 0;
  let from = recent[0].start;
  let to = recent[0].end;
  for (const t of recent.slice(1)) {
    if (t.start > to) {
      busy += to - from;
      from = t.start;
      to = t.end;
    } else {
      to = Math.max(to, t.end);
    }
  }
  busy += to - from;
  return bytes / (Math.max(busy, 1) / 1000);
}

/**
 * The stride for a link: the smallest that carries the deforming meshes at the playback rate with a fifth to spare.
 * Coming back down needs two fifths to spare, so a link near the line does not flip it back and forth. Unchanged
 * while the link is unknown.
 */
export function chooseStride(link: number | null, bytesPerFilmSecond: number, rate: number, current: number): number {
  if (link === null || !(bytesPerFilmSecond > 0) || !(rate > 0)) return current;
  if (link >= FAST_LINK_BYTES_PER_S) return 1;
  const fits = (stride: number, share: number) => (bytesPerFilmSecond * rate) / stride <= share * link;
  const needed = STRIDES.find((stride) => fits(stride, 0.8)) ?? STRIDES[STRIDES.length - 1];
  if (needed >= current) return needed;
  return STRIDES.find((stride) => stride >= needed && stride < current && fits(stride, 0.6)) ?? current;
}

export interface NativeStreamOptions {
  track: NativeTrack;
  /** Fetch and build one mesh. */
  load: (digest: string, signal: AbortSignal) => Promise<THREE.BufferGeometry>;
  /** Most bytes of built geometry to keep. */
  budgetBytes: number;
  /** Meshes loading at once. */
  concurrency: number;
  /** A mesh arrived (its digest) or failed (null): redraw if it is on screen, and re-check buffering. */
  onChange: (digest: string | null) => void;
  /** The geometries on screen now, which must not be dropped. */
  displayed: () => Set<THREE.BufferGeometry>;
}

function geometryBytes(geometry: THREE.BufferGeometry): number {
  let bytes = geometry.index ? geometry.index.array.byteLength : 0;
  for (const attribute of Object.values(geometry.attributes)) {
    bytes += (attribute as THREE.BufferAttribute).array.byteLength;
  }
  return bytes;
}

export class NativeStream {
  /** Every built mesh by digest, plus the atlas-UV variants `applyNativeFrame` adds under `digest|key`. */
  readonly geometries = new Map<string, THREE.BufferGeometry>();
  readonly failures: string[] = [];
  private failedCount = 0;
  private readonly failed = new Set<string>();
  private readonly sizes = new Map<string, number>();
  private total = 0;
  private readonly lastWanted = new Map<string, number>();
  private readonly inFlight = new Map<string, AbortController>();
  private readonly persistent = new Set<string>();
  /** Per deforming object, its mesh index on each frame. */
  private readonly deforming: number[][] = [];
  private frame = 0;
  private generation = 0;
  private disposed = false;
  /** Whether the whole clip is loaded rather than a window, decided once the first picture's sizes are known. */
  private wholeClip = false;
  /** Deforming objects' meshes are fetched on every `stride`-th frame (see the header). */
  private strideNow = 1;
  private strideChangedAt = 0;
  private rate = 1;
  private playing = false;
  private readonly transfers: Transfer[] = [];
  /**
   * Wire bytes of the latest deforming meshes downloaded, to size a film second by. The latest only: the first ones
   * are whole meshes, three times a later frame's positions, and would make the link look slower than it is.
   */
  private readonly deformingWire: number[] = [];

  constructor(private readonly options: NativeStreamOptions) {
    const { track } = options;
    for (const object of track.objects) {
      if (Array.isArray(object.mesh)) this.deforming.push(object.mesh);
      else if (object.mesh >= 0) this.persistent.add(track.meshes[object.mesh]);
    }
  }

  get nFrames(): number {
    return this.options.track.n_frames;
  }

  get fps(): number {
    return this.options.track.fps || 24;
  }

  /** Meshes built so far, and how many the clip has (less those that failed). */
  progress(): { loaded: number; total: number } {
    return { loaded: this.sizes.size, total: this.options.track.meshes.length - this.failed.size };
  }

  get noteFailures(): string[] {
    const extra = this.failedCount - this.failures.length;
    return extra > 0 ? [...this.failures, `and ${extra} more meshes did not load`] : this.failures;
  }

  /** The deforming objects' meshes on one frame. */
  digestsAt(frame: number): string[] {
    const { track } = this.options;
    const out: string[] = [];
    for (const meshes of this.deforming) {
      const index = meshes[Math.max(0, Math.min(meshes.length - 1, frame))];
      if (index >= 0) out.push(track.meshes[index]);
    }
    return out;
  }

  /**
   * Whether every mesh this frame draws is built (or will never be, having failed). With a stride, a frame between two
   * fetched ones is ready when the one before it is: its meshes are held over it.
   */
  isReady(frame: number): boolean {
    return this.isBuilt(frame) || (this.strideNow > 1 && this.isBuilt(frame - (frame % this.strideNow)));
  }

  private isBuilt(frame: number): boolean {
    for (const digest of this.digestsAt(frame)) {
      if (!this.geometries.has(digest) && !this.failed.has(digest)) return false;
    }
    return true;
  }

  /** The stride deforming objects' frames are fetched at now: 1 unless the link is too slow for the playback rate. */
  get stride(): number {
    return this.strideNow;
  }

  /** Whether the clip is playing: a stopped playhead's own frame loads exactly, a moving one's is not fetched. */
  setPlaying(playing: boolean): void {
    if (playing === this.playing) return;
    this.playing = playing;
    this.pump();
  }

  /** The playback rate, which sets how many meshes a second the link has to carry. */
  setRate(rate: number): void {
    this.rate = rate > 0 ? rate : 1;
    this.restride(true);
  }

  /** Downloads as the network stack timed them (`loader.worker.ts`), and whether they were deforming meshes. */
  noteTransfers(transfers: Transfer[], deforming: boolean): void {
    for (const transfer of transfers) {
      this.transfers.push(transfer);
      if (deforming) {
        this.deformingWire.push(transfer.bytes);
        if (this.deformingWire.length > 32) this.deformingWire.shift();
      }
    }
    if (this.transfers.length > 256) this.transfers.splice(0, this.transfers.length - 256);
    this.restride(false);
  }

  /** Whether `digest` is one of a deforming object's meshes (as opposed to one an object keeps for the whole clip). */
  isDeforming(digest: string): boolean {
    return !this.persistent.has(digest);
  }

  /**
   * The deforming meshes' bytes on the wire per second of film around the playhead: how many distinct meshes the next
   * second of frames draws, at the average size downloaded so far.
   */
  private bytesPerFilmSecond(): number {
    if (this.deformingWire.length === 0) return 0;
    const meshBytes = this.deformingWire.reduce((sum, bytes) => sum + bytes, 0) / this.deformingWire.length;
    const fps = this.options.track.fps || 24;
    const last = Math.min(this.options.track.n_frames - 1, this.frame + Math.round(fps) - 1);
    const distinct = new Set<string>();
    for (let f = this.frame; f <= last; f += 1) for (const digest of this.digestsAt(f)) distinct.add(digest);
    const seconds = (last - this.frame + 1) / fps;
    return (distinct.size / Math.max(seconds, 1 / fps)) * meshBytes;
  }

  private restride(force: boolean): void {
    const now = performance.timeOrigin + performance.now();
    if (!force && now - this.strideChangedAt < STRIDE_HOLD_MS) return;
    const link = linkBytesPerSecond(this.transfers, now);
    let next = chooseStride(link, this.bytesPerFilmSecond(), this.rate, this.strideNow);
    if (link === null && next > 1 && now - this.strideChangedAt >= STRIDE_DECAY_MS
        && this.readyAhead(this.frame) >= Math.min(this.aheadFrames(), this.options.track.n_frames - this.frame)) {
      next = STRIDES[Math.max(0, STRIDES.indexOf(next) - 1)];
    }
    if (next === this.strideNow) return;
    this.strideNow = next;
    this.strideChangedAt = now;
    this.pump();
    this.options.onChange(null);
  }

  /** How many frames from `frame` on are ready, counting at most `horizon` (the window by default). */
  readyAhead(frame: number, horizon = this.aheadFrames()): number {
    const last = Math.min(this.options.track.n_frames - 1, frame + horizon - 1);
    let n = 0;
    for (let f = frame; f <= last && this.isReady(f); f += 1) n += 1;
    return n;
  }

  /** The window ahead of the playhead, in frames. */
  aheadFrames(): number {
    return Math.max(1, Math.round(AHEAD_S * (this.options.track.fps || 24)));
  }

  /**
   * The persistent meshes and this frame's deforming ones, which the first picture needs. Resolves once they are
   * all built or failed.
   */
  async loadFirst(frame: number, onProgress: (done: number, total: number) => void): Promise<void> {
    const wanted = [...new Set([...this.persistent, ...this.digestsAt(frame)])];
    let next = 0;
    let done = 0;
    onProgress(0, wanted.length);
    const worker = async () => {
      while (next < wanted.length && !this.disposed) {
        const digest = wanted[next];
        next += 1;
        await this.loadOne(digest);
        done += 1;
        onProgress(done, wanted.length);
      }
    };
    await Promise.all(Array.from({ length: Math.min(this.options.concurrency, wanted.length) }, worker));
    this.wholeClip = this.estimatedClipBytes() <= this.options.budgetBytes * 0.9;
  }

  /**
   * The whole clip's built size, estimated from the first picture: each deforming object keeps one topology from
   * frame to frame, so its frame-0 mesh gives the size of every one of its meshes.
   */
  estimatedClipBytes(): number {
    const { track } = this.options;
    let bytes = 0;
    for (const digest of this.persistent) bytes += this.sizes.get(digest) ?? 0;
    for (const meshes of this.deforming) {
      const first = meshes.find((index) => index >= 0);
      if (first === undefined) continue;
      const size = this.sizes.get(track.meshes[first]) ?? 0;
      bytes += size * new Set(meshes.filter((index) => index >= 0)).size;
    }
    return bytes;
  }

  /** Whether the stream loads the whole clip (it fits the budget) rather than a window. */
  get loadsWholeClip(): boolean {
    return this.wholeClip;
  }

  /** Move the playhead: the window follows it. */
  setFrame(frame: number): void {
    this.frame = Math.max(0, Math.min(this.options.track.n_frames - 1, Math.trunc(frame)));
    this.pump();
  }

  /** Stop loading. With `freeGeometry`, also free what was built: nothing else owns it yet. */
  dispose(freeGeometry = false): void {
    this.disposed = true;
    for (const controller of this.inFlight.values()) controller.abort();
    this.inFlight.clear();
    if (freeGeometry) {
      for (const geometry of this.geometries.values()) geometry.dispose();
      this.geometries.clear();
    }
  }

  /** The window's meshes, nearest first and ahead before behind, without repeats. */
  private windowOrder(): string[] {
    const { track } = this.options;
    const fps = track.fps || 24;
    // The whole clip: everything from the playhead to the end, then from the start up to it.
    const last = this.wholeClip ? track.n_frames - 1 : Math.min(track.n_frames - 1, this.frame + this.aheadFrames());
    const first = this.wholeClip ? 0 : Math.max(0, this.frame - Math.round(BEHIND_S * fps));
    const seen = new Set<string>();
    const order: string[] = [];
    const add = (f: number) => {
      for (const digest of this.digestsAt(f)) {
        if (!seen.has(digest)) {
          seen.add(digest);
          order.push(digest);
        }
      }
    };
    // With a stride, the frames on it; and a stopped playhead's own frame, so it shows exactly.
    const step = this.strideNow;
    const on = this.frame - (this.frame % step);
    if (!this.playing) add(this.frame);
    for (let f = on; f <= last; f += step) add(f);
    if (this.wholeClip) for (let f = 0; f < on; f += step) add(f);
    else for (let f = on - step; f >= first; f -= step) add(f);
    return order;
  }

  private pump(): void {
    if (this.disposed) return;
    this.generation += 1;
    const order = this.windowOrder();
    const now = this.generation;
    for (const digest of order) this.lastWanted.set(digest, now);
    // Requests for meshes the window has moved past are dropped, so a scrub does not wait behind them.
    const inWindow = new Set(order);
    for (const [digest, controller] of this.inFlight) {
      if (!inWindow.has(digest) && !this.persistent.has(digest)) {
        controller.abort();
        this.inFlight.delete(digest);
      }
    }
    for (const digest of order) {
      if (this.inFlight.size >= this.options.concurrency) break;
      if (this.geometries.has(digest) || this.failed.has(digest) || this.inFlight.has(digest)) continue;
      if (this.total >= this.options.budgetBytes && !this.evict(inWindow)) break;
      void this.loadOne(digest).then(() => this.pump());
    }
  }

  /** Drop the meshes least recently wanted, outside the window and off screen. False when none can go. */
  private evict(inWindow: Set<string>): boolean {
    const shown = this.options.displayed();
    const candidates = [...this.sizes.keys()]
      .filter((digest) => !inWindow.has(digest) && !this.persistent.has(digest))
      .sort((a, b) => (this.lastWanted.get(a) ?? 0) - (this.lastWanted.get(b) ?? 0));
    let freed = false;
    for (const digest of candidates) {
      if (this.total < this.options.budgetBytes * 0.9) break;
      const keys = [...this.geometries.keys()].filter((key) => key === digest || key.startsWith(`${digest}|`));
      if (keys.some((key) => shown.has(this.geometries.get(key)!))) continue;
      for (const key of keys) {
        this.geometries.get(key)!.dispose();
        this.geometries.delete(key);
      }
      this.total -= this.sizes.get(digest) ?? 0;
      this.sizes.delete(digest);
      freed = true;
    }
    return freed && this.total < this.options.budgetBytes;
  }

  private async loadOne(digest: string): Promise<void> {
    if (this.geometries.has(digest) || this.failed.has(digest) || this.inFlight.has(digest)) return;
    const controller = new AbortController();
    this.inFlight.set(digest, controller);
    try {
      const geometry = await this.options.load(digest, controller.signal);
      if (this.disposed || controller.signal.aborted) {
        geometry.dispose();
        return;
      }
      this.geometries.set(digest, geometry);
      const bytes = geometryBytes(geometry);
      this.sizes.set(digest, bytes);
      this.total += bytes;
      this.options.onChange(digest);
    } catch (error) {
      if (this.disposed || controller.signal.aborted) return;
      this.failed.add(digest);
      this.failedCount += 1;
      if (this.failures.length < MAX_NOTED_FAILURES) {
        this.failures.push(`${digest.slice(0, 12)}: ${error instanceof Error ? error.message : String(error)}`);
      }
      this.options.onChange(null);
    } finally {
      if (this.inFlight.get(digest) === controller) this.inFlight.delete(digest);
    }
  }
}
