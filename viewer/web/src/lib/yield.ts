// Hand the event loop a turn, so input, rendering, and other callbacks run between slices of a long job.
//
// A resolved promise is not enough: its continuation is a microtask and runs before anything else can. This queues a
// task: `scheduler.yield()` where the browser has it, which resumes ahead of other queued work, otherwise a
// MessageChannel message, which is not clamped the way a nested `setTimeout(0)` is.

/** How long a slice of main-thread work may run before it yields. */
export const SLICE_MS = 8;

export function yieldToMain(): Promise<void> {
  const scheduler = (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler;
  if (scheduler?.yield) return scheduler.yield();
  if (typeof MessageChannel === "undefined") return new Promise((resolve) => setTimeout(resolve, 0));
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      resolve();
    };
    channel.port2.postMessage(null);
  });
}

/** A clock for one long job: `due()` once a slice has run `SLICE_MS`, then `yield()` resets it. */
export function sliceClock(): { due: () => boolean; yield: () => Promise<void> } {
  let start = performance.now();
  return {
    due: () => performance.now() - start >= SLICE_MS,
    yield: async () => {
      await yieldToMain();
      start = performance.now();
    },
  };
}
