# Project history

## August 2026: reusable highlight library and cross-source compilations (unreleased after 1.3.0)

Current local `main` separates source analysis from final Reel assembly. This work has not yet been published as a new immutable Docker tag; public image `1.3.0` predates it:

- source videos now produce reusable scored highlight clips instead of one mandatory Reel;
- a 70% source-relative library floor retains a wider candidate pool, while 87% is only a recommendation marker;
- neither the old six-point quota nor the 55-second budget applies during extraction or compilation;
- SQLite indexes clip metadata and file paths across jobs, preserves inactive legacy results, and stores ordered many-to-many compilation items; MP4 bytes remain regular files under `data/`;
- the desktop library filters by source, recording date, score, duration, and timeline, with top-six-per-filter and top-six-per-source batch actions;
- compilation building runs in the background, uses NVENC when available, and normalizes missing audio with silence.

Legacy jobs can expose only files that older versions actually exported. `rebuild-library` creates a separate clip-set directory and atomically activates it after a successful run; NVDEC/NVENC accelerate media I/O while the current heuristic scoring remains CPU-side. It never deletes the prior clips or Reel.

The 2026-08-23 persistence audit documented the hybrid SQLite/filesystem boundary and found 5 source videos (15.20 GiB), 56 human annotations, and 135 clip rows: 102 active v2, 30 inactive legacy, and 3 inactive v1. All 56 annotations are positive `highlight` labels, so this is useful error-analysis data but not yet a complete supervised training set. No custom compilation had been submitted at that snapshot. The target cloud lifecycle is Pixel → Google Drive handoff → desktop processing/cache → pCloud video archive, while live SQLite remains local with small snapshots. No pCloud credentials, upload worker, remote catalog, hydration, or automatic local eviction exist in this version. Package metadata still reports 1.3.0, so the next release must bump every version surface and refuse to overwrite the existing public tag before publishing.

## August 2026: threshold-selected point reels (version 1.3, superseded extraction policy)

Version 1.3 removes the per-video six-point quota from the default selection policy:

- candidates must reach 87% of the strongest score from the same source;
- the 55-second Reel budget remains, while the point-count cap is disabled by default;
- `analysis.json` persists every candidate, its score threshold, and the selection decision;
- padding is assigned only among selected points, so rejected neighbours no longer trim context;
- hard-cut H.264/AAC exports normalize timestamps and use signed CTS offsets so NVENC B-frames remain decodable in MP4;
- five real GPU reruns selected 22 points instead of the previous fixed 30, with every Reel fully decoding and no pipeline warnings.

The relative score rule is an interim heuristic rather than a calibrated probability. It adapts to source-level score shifts but still retains the strongest candidate whenever a source has any candidate at all.

The later reusable-library design above supersedes the 87%-plus-55-second extraction rule: 70% now controls material preservation, 87% is only a recommendation marker, and final duration is chosen at compilation time.

## February 2026: first prototype

Commits `67aee40` and `1c53645` established the first runnable pipeline:

- YOLO-World searched the opening frames for a table.
- YOLO-Pose tracked people around an expanded table region.
- A VIP score treated persistent player presence as rally activity.
- OpenCV read frames and FFmpeg stream-copy exported proposed intervals.

This proved that local automatic clipping was feasible, but it did not distinguish a rally from waiting, picking up balls, or conversation.

## June 2026: experiments on the original architecture

Commits `34a2f46` through `cdad2b7` explored several useful ideas:

- workspace-relative storage and a parameter-tuning loop;
- generic COCO sports-ball detections and trajectory direction changes;
- optional Gemini clip verification and intensity ranking;
- scene-change detection, repeated table detection, and aspect-ratio-aware play zones;
- a watch-folder/import utility, spatial player roles, stitched reels, and an offline report.

These commits remain in Git history for reference. They were not carried into the active runtime because the underlying rally state still depended on player presence, generic sports-ball detections were only diagnostic, and the tuner optimized clip count rather than labeled precision or recall. OpenCV frame indexing and FFmpeg stream-copy also left mobile VFR, rotation, and exact-cut problems unresolved. The optional Gemini path sent clips to a third party, which does not fit the new local-first default.

## August 2026: local-first pipeline restart

Version 0.2 replaced the experiment scripts with a package-oriented system:

- a phone-facing LAN page with resumable chunk upload and persistent offsets;
- SQLite-backed upload and processing jobs that recover after restart;
- timestamp-based FFmpeg audio/video decoding for mobile formats;
- adaptive audio-transient and localized-motion signal fusion;
- ranked rally candidates, accurate H.264/AAC cuts, and a combined reel;
- synthetic media, API, recovery, export, and end-to-end tests;
- an explicit labeled-data evaluation plan before adding trained models.

The earlier work informed failure analysis, while the production path was intentionally rebuilt around measurable rally evidence and reliable mobile ingestion.

## August 2026: point-based social reel editing

Version 0.3 changed the product unit from a long activity interval to one scored point:

- impact groups are split at point-sized gaps and are never merged into long rally blocks;
- nearby points divide their quiet gap so exported clips do not duplicate serves or reactions;
- ranking uses a maximum-point count and a target Reel duration;
- selected points are rendered on a 9:16 blurred-background canvas;
- video and audio cross-dissolve only between points, with no fade-out after the final point;
- the real 10-minute Pixel recording produced a six-point, 44.2-second vertical Reel during validation.

## August 2026: source-format output and transfer UX

Version 0.4 separated analysis output from publishing format:

- the default Reel now preserves the source resolution and aspect ratio;
- the vertical blurred-background renderer remains optional for a later publish step;
- the LAN page explains the phone-to-PC-to-phone workflow in three steps;
- completed jobs provide inline video preview, a primary MP4 download, Web Share fallback, and secondary per-point downloads;
- preview responses are inline while explicit downloads use attachment headers;
- stable job polling no longer rebuilds the video player and interrupts playback.
