# Project history

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
