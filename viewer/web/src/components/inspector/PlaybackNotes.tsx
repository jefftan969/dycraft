// The notes about a clip's 3D playback, under the transport bar (`CompactNotes`): what the 3D view shows (gray), and
// meshes that did not load or bodies with nothing to place them by (amber). Loading progress is not here: the
// viewport's loading tray has it.

import { type JSX } from "react";

import { CompactNotes } from "./CompactNotes";

export function PlaybackNotes({ standing, problems }: { standing: string[]; problems: string[] }): JSX.Element | null {
  return (
    <CompactNotes
      label="3D playback"
      title="Notes about this clip's 3D playback: what it shows, and anything it could not load or place"
      standing={standing}
      problems={problems}
    />
  );
}
