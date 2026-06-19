# 🏓 Table Tennis Highlight Clipper (TTHAC)

TTHAC is an automated, AI-powered video highlight generation system designed specifically for ping pong matches. It utilizes advanced computer vision models to detect the table, track players via pose estimation, identify active rallies, and perform lightning-fast, lossless video clipping using FFmpeg.

---

## 🚀 Key Features

* **Auto Table Detection**: Uses `yolov8l-worldv2.pt` (YOLO-World) to automatically detect the table boundaries and compute an optimized core activity zone.
* **Player Pose & ID Tracking**: Leverages `yolo11l-pose.pt` to detect human keypoints (hips, knees, ankles) and track player IDs across video frames.
* **Intelligent Rally Extraction**: Employs a state machine that tracks player "VIP score" based on proximity and duration within the table zone. It automatically defines rallies, tolerating temporary occlusions.
* **Lossless Fast Cutting**: Uses FFmpeg stream copying (`-c copy`) to segment highlight clips in fractions of a second without re-encoding, preserving original video quality.
* **Configurable Algorithm Params**: Tuning parameters for sensitivity, core zone expansion, minimum durations, and dropout padding.

---

## 📂 Project Structure

```text
pingpong-auto-highlight/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized settings and algorithm parameters
├── core/
│   ├── __init__.py
│   ├── detectors.py         # TableDetector (YOLO-World) & PoseEngine (YOLO-Pose)
│   └── tracker.py           # VIPGameTracker & PlayerStats state machine
├── README.md                # System overview and quick start guide
├── agent.md                 # Specification for the agentic tuning system
├── main.py                  # CLI entry point to process video
└── requirements.txt         # Python dependencies
```

---

## 🛠️ Getting Started

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg**: Ensure `ffmpeg` is installed and available in your system's `PATH`.
   * **Linux**: `sudo apt install ffmpeg`
   * **macOS**: `brew install ffmpeg`
   * **Windows**: Download from the official website and add to environment variables.

### Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment (optional but recommended).
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running TTHAC

To process a ping pong video and extract highlights, run the entry point script with the path to your video file:

```bash
python main.py /path/to/your/video.mp4
```

Highlights will be exported to the output directory defined in your configurations (organized under a subdirectory matching the input video name).

---

## ⚙️ Configuration & Algorithm Parameters

All configurations are defined in [config/settings.py](file:///home/ubuntu/projects/pingpong-auto-highlight/config/settings.py). Key parameters include:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `table_search_frames` | `int` | `90` | Number of frames to scan at the start to find the ping pong table. |
| `min_rally_duration` | `float` | `1.5` | Minimum duration (seconds) required for a sequence to qualify as a highlight. |
| `max_dropout_duration` | `float` | `3.0` | Allowed time (seconds) for VIP players to be missing or inactive before ending a rally. |
| `vip_warmup_score` | `int` | `20` | Score threshold required for a tracked person to be promoted to a "VIP player". |
| `score_in_frame` | `int` | `1` | Base score added to a player's tracker per frame they are visible. |
| `score_in_core` | `int` | `5` | Additional score added per frame when keypoints are detected inside the table's core zone. |
| `core_zone_expansion` | `float` | `1.4` | Scale factor to expand the bounding box of the table to define the core play area. |

---

## 🤖 Agentic Tuning System (Roadmap)

To elevate TTHAC into an autonomous agentic system, we are developing an intelligent feedback loop (`agent.md`) that will:
1. **Self-Correct Table Detection**: Adjust parameters (e.g., scan length, prompts) or fallback to center zones when table detection fails.
2. **Tune Algorithm Parameters**: Dynamically adjust parameters like `score_in_core` and `vip_warmup_score` by evaluating the distribution and length of generated clips.
3. **Validate Outputs**: Automatically verify clip quality (e.g., checking if players are active, validating files) to ensure only relevant highlights are saved.