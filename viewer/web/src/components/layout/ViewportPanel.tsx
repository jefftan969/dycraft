// Adapted from Articraft (github.com/mattzh72/articraft, Apache License 2.0; see LICENSE),
// modified for dycraft.
import { type JSX, type ReactNode, type RefObject } from "react";
import type * as THREE from "three";

import { SceneCanvas, type SnapshotExporter } from "@/components/viewer3d/SceneCanvas";
import type { RenderOptions } from "@/components/viewer3d/useRenderOptions";
import type { UrdfSpec } from "@/components/viewer3d/urdf-parser";
import type { DerivedAssetsState } from "@/components/viewer3d/useDerivedAssets";
import { cn } from "@/lib/utils";
import type { FilmCamera, FilmCameraTrack } from '@/components/viewer3d/film-camera';

interface ViewportPanelProps {
  baseFileUrl: string | null;
  assetRevisionKey: string | null;
  selectionKey: string | null;
  jointPoseSignal: Map<string, number>;
  /** Baked-atlas asset state, owned by `App`; see `SceneCanvas`. */
  derived: DerivedAssetsState;
  /** Live posed joints, for the Blender stream; see `SceneCanvas`. */
  posedJointValues?: RefObject<Map<string, number>>;
  renderOptions: RenderOptions;
  /** Dycraft-world AABB to frame on; animation runs pass the clip's union. */
  fitBounds?: { min: number[]; max: number[] } | null;
  filmCamera?: FilmCamera | null;
  /** The film's camera at every frame, and the playhead's frame; see `SceneCanvas`. */
  filmTrack?: FilmCameraTrack | null;
  frame?: number;
  /** The rendered film and the playback it keeps to; see `SceneCanvas`. */
  filmClipUrl?: string | null;
  playing?: boolean;
  rate?: number;
  fps?: number;
  /** The run's native clip version, for the walls cutaway; see `SceneCanvas`. */
  nativeVersion?: string | null;
  /** The native clip's first picture is up; see `SceneCanvas`. */
  nativeReady?: boolean;
  /** What the empty viewport says when nothing is selected; see `SceneCanvas`. */
  idleMessage?: string | null;
  onUrdfSpecChange: (
    spec: UrdfSpec | null,
    jointNodes: Map<string, THREE.Object3D> | null,
    linkNodes: Map<string, THREE.Group> | null,
  ) => void;
  onInvalidateReady?: (invalidate: (() => void) | null) => void;
  onSnapshotReady?: (snapshot: SnapshotExporter | null) => void;
  onLoadStateChange?: (state: {
    loading: boolean;
    error: string | null;
    missingArtifacts: boolean;
    collisionGeometryState: {
      declaredCount: number;
      primitiveCount: number;
      meshCount: number;
      loadedMeshCount: number;
      failedMeshCount: number;
      loading: boolean;
    };
  }) => void;
  overlayNotice?: ReactNode;
  disabledOverlay?: ReactNode;
}

export function ViewportPanel({
  baseFileUrl,
  assetRevisionKey,
  selectionKey,
  jointPoseSignal,
  derived,
  posedJointValues,
  renderOptions,
  fitBounds,
  filmCamera,
  filmTrack,
  frame,
  filmClipUrl,
  playing,
  rate,
  fps,
  nativeVersion,
  nativeReady,
  idleMessage,
  onUrdfSpecChange,
  onInvalidateReady,
  onSnapshotReady,
  onLoadStateChange,
  overlayNotice,
  disabledOverlay,
}: ViewportPanelProps): JSX.Element {
  // Map from useRenderOptions format to RenderOptionsPanel format for SceneCanvas
  const sceneRenderOptions = {
    fancyGraphics: renderOptions.fancyGraphics,
    showEdges: renderOptions.showEdges,
    showGrid: renderOptions.showGrid,
    showCollisions: renderOptions.showCollisions,
    showSegmentColors: renderOptions.showSegmentColors,
    showSurfaceSamples: renderOptions.showSurfaceSamples,
    doubleSided: renderOptions.doubleSided,
    autoAnimate: renderOptions.autoAnimate,
    showJointOverlay: renderOptions.showJointOverlay,
    showDeformation: renderOptions.showDeformation,
    blenderRender: renderOptions.blenderRender,
    bakedMaterials: renderOptions.bakedMaterials,
  };

  return (
    <section className="relative h-full w-full min-w-0 bg-[var(--surface-2)]">
      <div className="absolute inset-0 overflow-hidden">
        <div
          className={cn(
            "h-full w-full transition duration-200",
            disabledOverlay ? "pointer-events-none select-none blur-[8px] saturate-50 opacity-35" : "",
          )}
        >
          <SceneCanvas
            baseFileUrl={baseFileUrl}
            assetRevisionKey={assetRevisionKey}
            selectionKey={selectionKey}
            jointPoseSignal={jointPoseSignal}
            derived={derived}
            posedJointValues={posedJointValues}
            renderOptions={sceneRenderOptions}
            fitBounds={fitBounds}
            filmCamera={filmCamera}
            filmTrack={filmTrack}
            frame={frame}
            filmClipUrl={filmClipUrl}
            playing={playing}
            rate={rate}
            fps={fps}
            nativeVersion={nativeVersion}
            nativeReady={nativeReady}
            idleMessage={idleMessage}
            onUrdfSpecChange={onUrdfSpecChange}
            onInvalidateReady={onInvalidateReady}
            onSnapshotReady={onSnapshotReady}
            onLoadStateChange={onLoadStateChange}
          />
        </div>
      </div>
      {overlayNotice ? <div className="absolute inset-x-0 bottom-0 z-[5] p-3">{overlayNotice}</div> : null}
      {disabledOverlay ? <div className="absolute inset-0 z-10">{disabledOverlay}</div> : null}
    </section>
  );
}
