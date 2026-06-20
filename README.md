# 🏓 Table Tennis Highlight Clipper (TTHAC)

TTHAC is an automated, AI-powered video highlight generation system designed specifically for ping pong matches. It utilizes computer vision models to detect the table, track players via pose estimation, track the ball trajectory, identify active rallies, and perform fast, lossless video clipping using FFmpeg.

---

## 🚀 Key Features

* **Dynamic Table & Camera Switch Detection**: Automatically tracks the table and updates play boundaries, even during camera cuts or slow pans.
* **Aspect-Ratio-Aware Play Zones**: Adapts play boundaries for standard baseline views, side profile views, and corner diagonal angles.
* **Pose & Spatial ID Tracking**: Leverages human keypoints to lock onto active players, stabilizing IDs based on spatial orientation to prevent tracker ID flickering.
* **Ball Hit Trajectory Analysis**: Tracks horizontal and vertical ball velocity changes to count racket hits per rally.
* **Lossless Fast Clipping**: Uses FFmpeg stream copying (`-c copy`) to segment highlight clips in fractions of a second without quality degradation.
* **Multimodal Verification & Reports**: Stitches highlights into a unified reel, generating interactive HTML dashboards locally or fetching AI grading via Gemini VLM.
* **Enterprise Ingestion Tool**: Pre-compresses large video files, downloads streams directly from YouTube, and monitors watch folders automatically.

---

## 🛠️ Getting Started

### Prerequisites
* **Python 3.8+**
* **FFmpeg**: Ensure `ffmpeg` is installed and available in your environment.
  * **Linux**: `sudo apt install ffmpeg`
  * **macOS**: `brew install ffmpeg`

### Installation
1. Clone this repository and navigate to the root directory.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Run
Process a table tennis video and generate highlights:
```bash
python main.py /path/to/your/video.mp4
```

---

## 📚 Documentation Registry

To keep the repository clean and modular, documentation is separated by target audiences:

1. **[INSTRUCTION.md](file:///home/ubuntu/projects/pingpong-auto-highlight/INSTRUCTION.md)**: The end-user guide. Contains detailed parameter descriptions, tuning recommendations for different play styles, and CLI commands for the import tool.
2. **[ARCHITECTURE.md](file:///home/ubuntu/projects/pingpong-auto-highlight/ARCHITECTURE.md)**: The system design specification. Contains data flow charts, file structures, component responsibilities, and model registries.
3. **[AGENT.md](file:///home/ubuntu/projects/pingpong-auto-highlight/AGENT.md)**: Rules, constraints, and optimization loops specifically written for AI coding assistants and automatic tuning pipelines.
4. **[docs/history.md](file:///home/ubuntu/projects/pingpong-auto-highlight/docs/history.md)**: Historical commit timeline, author contributions, and milestone breakdowns.
5. **[docs/feature_roadmap.md](file:///home/ubuntu/projects/pingpong-auto-highlight/docs/feature_roadmap.md)**: Features progression stages and future evolutionary direction.