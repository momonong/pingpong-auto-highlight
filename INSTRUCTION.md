# 📖 TTHAC User Guide & Algorithmic Instructions

This document provides detailed instructions on how to use TTHAC, how the underlying tracking algorithms operate, and how to configure and tune parameters to get the best highlights.

---

## 🛠️ CLI Command Guide

TTHAC operates through two CLI scripts.

### 1. Main Pipeline ([main.py](file:///home/ubuntu/projects/pingpong-auto-highlight/main.py))
Analyze a video and extract highlights:
```bash
python main.py /path/to/video.mp4
```

#### Optional Overrides:
* `--min_rally_duration 2.5` - Enforce longer rallies (seconds).
* `--vip_warmup_score 15` - Lower the score threshold needed to promote a player to a "VIP".
* `--core_zone_expansion 1.6` - Expand play detection area surrounding the table.

### 2. Video Import & Preprocessing ([import_tool.py](file:///home/ubuntu/projects/pingpong-auto-highlight/import_tool.py))
Optimize large files, download links, or listen for uploads:

* **Compress large local files**:
  ```bash
  python import_tool.py compress /path/to/large_video.mp4 -o ./storage/game_optimized.mp4
  ```
* **Fetch from YouTube / URL and highlight**:
  ```bash
  python import_tool.py url-import "https://www.youtube.com/watch?v=VIDEO_ID" --compress
  ```
* **Run automated upload folder daemon**:
  ```bash
  python import_tool.py watch --dir ./storage/uploads --archive ./storage/processed
  ```

---

## ⚙️ Tuning Parameters Reference

Algorithm variables are located in [config/settings.py](file:///home/ubuntu/projects/pingpong-auto-highlight/config/settings.py) under the `ALGO_PARAMS` dictionary.

| Configuration Param | Default | Purpose | Adjusting Impact |
| :--- | :--- | :--- | :--- |
| `table_search_frames` | `90` | Frames to search at start for table. | Increase (e.g. 200) if the table is obstructed or starts late. |
| `min_rally_duration` | `1.5` | Min clip duration (seconds). | Lower to capture quick points; raise to keep only long rallies. |
| `max_dropout_duration` | `3.0` | Tolerated gap (seconds) where players are hidden. | Raise to prevent highlights from cutting off during camera switches. |
| `vip_warmup_score` | `20` | Frames score required to begin recording a player. | Lower to record immediately; raise to filter out warmups. |
| `score_in_core` | `5` | Score bonus when pose is in core zone. | Raise to prioritize action strictly centered around the table. |
| `core_zone_expansion` | `1.4` | Table bounding box expansion scale. | Raise (e.g. 1.6) if players stand far behind the table. |

### Tuning Scenarios Heuristics
* **Casual Warmup / Multi-ball Practice**:
  Set `min_rally_duration` high (e.g., 3.0) and lower `score_in_core` to ignore pick-ups. Set a long `max_dropout_duration` (e.g., 5.0) to capture the entire training block as a single clip.
* **Intense Match Play**:
  Keep `min_rally_duration` around 2.0. Increase `score_in_core` to 8 to prioritize powerful shots, and decrease `max_dropout_duration` to 2.0 to clip immediately when the point finishes.

---

## 🧠 Algorithmic Deep Dive

### 1. Spatial Role Tracking
Standard object tracking assigns volatile IDs (e.g. ID `1` switches to ID `5` when a player bends over or gets occluded). To fix this, TTHAC tracks fixed **spatial slots**:
1. It calculates the centroid of player keypoints.
2. Checks camera perspective: if table width > table height, it assumes a **Side View**; otherwise a **Vertical View**.
3. In a Vertical View, players are sorted by their Y coordinates into `Player_Far` (Top) and `Player_Near` (Bottom). In a Side View, they are sorted by X into `Player_Left` and `Player_Right`.
4. The tracking database registers points to these slots rather than YOLO's ID, stabilizing player histories.

### 📊 2. Local Hit Count Estimation
The horizontal (or vertical) coordinate of the ball is monitored frame-by-frame during a rally:
1. It computes horizontal speed `vx` (or vertical speed `vy` depending on camera orientation).
2. It detects a sign change (positive velocity to negative velocity or vice versa) that crosses a noise threshold.
3. It enforces a **0.3s cooldown** (maximum of ~3 hits per second) to prevent tracking noise from registering as false racket hits.
4. The estimated hits are stored in the output data and plotted in the offline report.
