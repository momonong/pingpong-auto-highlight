# Project history

## August 2026: formal candidate gate and detector v4 (unreleased after 1.3.0)

The candidate-generation and runtime-library identities are now deliberately separate. `candidate-generation-v4` creates complete JSON/NPZ candidate evidence without exporting MP4 or mutating the runtime database; `highlight-library-v3` is the corresponding runtime processor version used only when a new job or an explicit library rebuild exports clips.

The receipt-valid development run at commit `7e0881d` used strict NVIDIA NVDEC on an NVIDIA GeForce RTX 5090 Laptop GPU. Across five development sources and 56 positive `highlight` annotations, v4 reached 51/56 strict candidate recall (91.07%). It produced 481 candidates over 108.658377 source minutes (4.426718/min), covered 46.3734% of the source timelines by candidate-core union, had a maximum core of 18.957 seconds, and left zero overlapping candidate pairs. Recall and every candidate-burden guardrail passed, so the decision is `GO_RANKING`; the previous formal v3 baseline had reached only 4/56 (7.14%) and stopped at the detector.

This is a `valid-development-regression`, not held-out accuracy. All 56 annotations are positive and there are no explicit `exclude` labels, so precision and other false-positive-dependent metrics remain explicitly abstained. The run did not activate new clips: the existing five sources still have 102 active `highlight-library-v2` clips. Moving them to v3 requires an explicit, successful per-job rebuild; no evaluation command performs that switch.

## August 2026: pCloud archive foundation (unreleased after 1.3.0)

The first additive pCloud phase is implemented without changing local playback or deletion behavior:

- a pinned rclone 1.75.0 binary is included in the source-built Docker image, while portable setup scripts create an OAuth config outside Git;
- `pcloud doctor/bootstrap/plan/archive/status/verify` provides an explicit operator workflow, with no startup-triggered bulk upload;
- canonical `HighlightCraft/archive-v1` paths rename originals, active clips, and final compilations by date/type/stable ID while retaining original names in deterministic manifests;
- every transfer uses a unique staging path, local SHA-1/SHA-256, pCloud `copyfile noover=1`, remote size/SHA-1 verification, and crash reconciliation before SQLite marks the object verified;
- the first transfer attempt freezes content/manifest checksums, while catalog preflight rejects account, backend-root, or archive-root drift before another object is registered;
- `storage_objects` records independent local/archive state, remote alias/region/account/root/file ID/path, checksums, attempts, and errors;
- the desktop library loads active and inactive indexed clips, then filters lifecycle, local availability, and archive state without exposing remote paths or credentials;
- remote-only clips remain discoverable but cannot be previewed or compiled until hydration exists, and staging cleanup is idempotent when pCloud has already removed an object;
- the long-lived OAuth config is absent from the web container and mounted only into the operator-run `pcloud-admin` container;
- this phase does not delete Drive or local files and does not yet provide background upload, hydration, eviction, or database snapshots.

The real-data non-transfer plan on 2026-08-23 resolved all 5 originals and 102 active v2 clips with no missing local file, totaling about 16.1 GiB. It initialized the additive `storage_objects` schema but registered no object; no pCloud OAuth was performed and no remote folder or file was created.

## August 2026: reusable highlight library and cross-source compilations (unreleased after 1.3.0)

Current local `main` separates source analysis from final Reel assembly. This work has not yet been published as a new immutable Docker tag; public image `1.3.0` predates it:

- source videos now produce reusable scored highlight clips instead of one mandatory Reel;
- a 70% source-relative library floor retains a wider candidate pool, while 87% is only a recommendation marker;
- neither the old six-point quota nor the 55-second budget applies during extraction or compilation;
- SQLite indexes clip metadata and file paths across jobs, preserves inactive legacy results, and stores ordered many-to-many compilation items; MP4 bytes remain regular files under `data/`;
- the desktop library filters by source, recording date, score, duration, and timeline, with top-six-per-filter and top-six-per-source batch actions;
- compilation building runs in the background, uses NVENC when available, and normalizes missing audio with silence.

Legacy jobs can expose only files that older versions actually exported. `rebuild-library` creates a separate clip-set directory and atomically activates it after a successful run; NVDEC/NVENC accelerate media I/O while the current heuristic scoring remains CPU-side. It never deletes the prior clips or Reel.

The 2026-08-23 persistence audit documented the hybrid SQLite/filesystem boundary and found 5 source videos (15.20 GiB), 56 human annotations, and 135 clip rows: 102 active v2, 30 inactive legacy, and 3 inactive v1. All 56 annotations are positive `highlight` labels, so this is useful error-analysis data but not yet a complete supervised training set. No custom compilation had been submitted at that snapshot. The target cloud lifecycle is Pixel → Google Drive handoff → desktop processing/cache → pCloud video archive, while live SQLite remains local with small snapshots. The later pCloud foundation above adds explicit archive transfer/catalog support, but credentials, background upload, hydration, automatic snapshots, and local eviction still do not exist. Package metadata still reports 1.3.0, so the next release must bump every version surface and refuse to overwrite the existing public tag before publishing.

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
