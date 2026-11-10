# Vendored Frontend Components

`viewer3d/`, `ui/`, `layout/`, `inspector/`, and `../lib/` hold code from
[Articraft](https://github.com/mattzh72/articraft), distributed under the Apache
License 2.0 (`LICENSE` at the repository root; see also `NOTICE`),
alongside files written for DyCraft.

- **Articraft's files, unmodified:** every file in these directories that is not
  listed below and carries no header.
- **Articraft's files, modified for DyCraft:** each starts with a header saying
  "Adapted from Articraft …, modified".
- **Written for DyCraft:**
  `inspector/AnimationTimeline.tsx`,
  `inspector/MarkdownArticle.tsx`,
  `inspector/trajectory-parsing.ts`,
  `layout/AppErrorBoundary.tsx`,
  `viewer3d/animation-playback.ts`,
  `viewer3d/camera-keys.ts`,
  `viewer3d/joint-flip.ts`,
  `viewer3d/mesh-bundle.ts`,
  `viewer3d/native-track.ts`,
  `viewer3d/preview-motion.ts`,
  `viewer3d/scene-presentation.ts`,
  `viewer3d/snapshot-export.ts`,
  `viewer3d/triplanar-uv.ts`,
  `viewer3d/useAnimationPlayback.ts`,
  `viewer3d/useBlenderStream.ts`,
  `viewer3d/useCameraKeys.ts`,
  `viewer3d/useDerivedAssets.ts`,
  `viewer3d/useNativeAnimation.ts`,
  `viewer3d/user-data-keys.ts`,
  `../lib/clipboard.ts`,
  `../lib/trajectory-adapter.ts`.

DyCraft-specific UI otherwise lives in `../panels/` and `../App.tsx`; the seam
between the two is `../lib/trajectory-adapter.ts`, which reshapes DyCraft's API
responses into the types `inspector/trajectory-types.ts` declares. Dropped from
the vendored set: `browser/`, `dashboard/`, `InspectPanel`, `MetadataPanel`,
`TracePanel`, `CodePanel`, `InspectorTabs`, `TrajectoryInspectorModal`,
`AppHeader`, `Inspector`, `ui/chart`, all bound to Articraft's curated record
store, which has no DyCraft counterpart.
