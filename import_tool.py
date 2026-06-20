#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess
import time
import shutil
from pathlib import Path

# Try to import tqdm for progress visualization if available
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)

def compress_video(input_path: Path, output_path: Path, target_resolution: str = "1280x720", fps: int = 30) -> bool:
    """
    Compresses a large video using FFmpeg to make it highly optimized for AI analysis.
    Downscales resolution, sets frame rate, and applies H.264 compression (CRF 28).
    This keeps table/pose detection accurate but slashes file size by up to 90%.
    """
    print(f"\n[Compress] Compressing video for AI processing:")
    print(f"  Input:  {input_path} ({get_file_size_mb(input_path):.2f} MB)")
    
    # Check if FFmpeg is installed
    if not shutil.which("ffmpeg"):
        print("❌ Error: FFmpeg is not installed or not in PATH. Compression failed.")
        return False
        
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"scale={target_resolution}",
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", "28",           # Slightly higher CRF (lower quality, much smaller size) - perfect for CV tracking
        "-preset", "faster",     # Fast encoding speed
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]
    
    start_time = time.time()
    try:
        # Run command and capture stderr for error logging if needed
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("❌ FFmpeg execution failed:")
            print(res.stderr.decode('utf-8'))
            return False
            
        elapsed = time.time() - start_time
        print(f"✅ Compression completed in {elapsed:.1f} seconds.")
        print(f"  Output: {output_path} ({get_file_size_mb(output_path):.2f} MB)")
        reduction = (1 - (output_path.stat().st_size / input_path.stat().st_size)) * 100
        print(f"  Storage Reduction: {reduction:.1f}% smaller!")
        return True
    except Exception as e:
        print(f"❌ Error compressing video: {e}")
        return False

def download_video_url(url: str, output_dir: Path) -> Path:
    """
    Downloads a video from a URL.
    Prefers using `yt-dlp` for platforms like YouTube/Twitch,
    and falls back to standard file download if it's a direct URL.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try yt-dlp first
    if shutil.which("yt-dlp"):
        print(f"\n[Download] URL detected. Downloading with yt-dlp: {url}")
        # Standard filename structure
        out_template = str(output_dir / "%(title)s_%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", "mp4/best",
            "-o", out_template,
            url
        ]
        
        try:
            subprocess.run(cmd, check=True)
            # Find the most recently created file in output_dir
            files = sorted(output_dir.glob("*.mp4"), key=os.path.getmtime, reverse=True)
            if files:
                print(f"✅ Successfully downloaded via yt-dlp: {files[0]}")
                return files[0]
        except subprocess.CalledProcessError as e:
            print(f"⚠️ yt-dlp failed or returned an error: {e}")
            
    # Direct url / fallback
    print(f"\n[Download] Attempting direct download: {url}")
    filename = url.split("/")[-1].split("?")[0]
    if not filename.endswith((".mp4", ".mkv", ".avi", ".mov")):
        filename = f"downloaded_video_{int(time.time())}.mp4"
        
    target_path = output_dir / filename
    
    # Determine downloading tool
    if shutil.which("wget"):
        cmd = ["wget", "-O", str(target_path), url]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "-o", str(target_path), url]
    else:
        print("❌ Error: No download tools (yt-dlp, wget, curl) found in path.")
        return None
        
    try:
        subprocess.run(cmd, check=True)
        if target_path.exists() and target_path.stat().st_size > 0:
            print(f"✅ Successfully downloaded direct file: {target_path} ({get_file_size_mb(target_path):.2f} MB)")
            return target_path
    except Exception as e:
        print(f"❌ Direct download failed: {e}")
        
    return None

def start_watch_folder(watch_dir: Path, processed_dir: Path, pipeline_params: list):
    """
    Monitors a directory for incoming files.
    When a file finishes uploading (size is stable), it triggers the main pipeline.
    """
    watch_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Daemon] Starting Watch Folder Daemon on: {watch_dir.resolve()}")
    print(f"  Processed videos will be archived in: {processed_dir.resolve()}")
    print("  Waiting for videos to process... Press Ctrl+C to stop.")
    
    # Store processed file hashes or paths to avoid reprocessing
    history_file = watch_dir / ".processed_history.txt"
    processed_history = set()
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as fh:
            processed_history = set(line.strip() for line in fh if line.strip())
            
    try:
        while True:
            # Look for video files
            for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov"):
                for video_file in watch_dir.glob(ext):
                    # Skip files currently being modified/written
                    if str(video_file) in processed_history:
                        continue
                        
                    print(f"\n[Daemon] Found new video candidate: {video_file.name}")
                    
                    # Verify file is not still uploading (size stability test)
                    last_size = video_file.stat().st_size
                    time.sleep(3)
                    curr_size = video_file.stat().st_size
                    
                    if last_size != curr_size:
                        print(f"  [Daemon] File is still being uploaded/written (Size: {get_file_size_mb(video_file):.2f} MB). Waiting...")
                        continue
                    
                    if curr_size == 0:
                        continue
                        
                    print(f"  [Daemon] File stabilized ({get_file_size_mb(video_file):.2f} MB). Starting highlight clipper...")
                    
                    # Run main pipeline
                    cmd = [sys.executable, "main.py", str(video_file)] + pipeline_params
                    print(f"  [Daemon] Running: {' '.join(cmd)}")
                    
                    proc_res = subprocess.run(cmd)
                    
                    # Archive processed file to prevent infinite loops
                    archive_path = processed_dir / video_file.name
                    if archive_path.exists():
                        # Append timestamp to filename if duplicate
                        archive_path = processed_dir / f"{video_file.stem}_{int(time.time())}{video_file.suffix}"
                        
                    shutil.move(str(video_file), str(archive_path))
                    print(f"  [Daemon] Moved video to archive: {archive_path.name}")
                    
                    # Log in history
                    processed_history.add(str(video_file))
                    with open(history_file, 'a', encoding='utf-8') as fh:
                        fh.write(f"{video_file}\n")
                        
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[Daemon] Watch folder daemon stopped.")

def main():
    parser = argparse.ArgumentParser(description="TTHAC Long Video Importer & Optimization Utility")
    
    # Action groups
    subparsers = parser.add_subparsers(dest="command", help="Importer Action Modes")
    
    # 1. Compress Command
    compress_parser = subparsers.add_parser("compress", help="Compress large local videos for fast AI processing")
    compress_parser.add_argument("input_video", type=str, help="Path to large input video file")
    compress_parser.add_argument("--output", "-o", type=str, default=None, help="Output compressed path")
    compress_parser.add_argument("--resolution", type=str, default="1280x720", help="Output resolution (default: 1280x720)")
    compress_parser.add_argument("--fps", type=int, default=30, help="Output framerate (default: 30)")
    
    # 2. URL Import Command
    url_parser = subparsers.add_parser("url-import", help="Import & highlight video directly from a web URL/YouTube/Drive")
    url_parser.add_argument("url", type=str, help="Remote URL of the video file or streaming site")
    url_parser.add_argument("--compress", action="store_true", help="Compress the video after download before processing")
    
    # 3. Watch Folder Daemon Command
    watch_parser = subparsers.add_parser("watch", help="Monitor directory for new videos and automatically run clipper")
    watch_parser.add_argument("--dir", type=str, default="./storage/uploads", help="Directory to monitor")
    watch_parser.add_argument("--archive", type=str, default="./storage/processed", help="Directory to move processed videos to")
    
    # Universal main processing args that should be forwarded to main.py
    for p in [compress_parser, url_parser, watch_parser]:
        p.add_argument("--min_rally_duration", type=float, default=None)
        p.add_argument("--max_dropout_duration", type=float, default=None)
        p.add_argument("--vip_warmup_score", type=int, default=None)
        p.add_argument("--score_in_core", type=int, default=None)
        p.add_argument("--core_zone_expansion", type=float, default=None)

    args, unknown = parser.parse_known_args()
    
    # Build forwardable params to main.py
    forward_params = []
    for param in [
        "min_rally_duration", "max_dropout_duration",
        "vip_warmup_score", "score_in_core", "core_zone_expansion"
    ]:
        val = getattr(args, param, None)
        if val is not None:
            forward_params.append(f"--{param}")
            forward_params.append(str(val))
            
    if not args.command:
        parser.print_help()
        return

    # Execute Commands
    if args.command == "compress":
        in_path = Path(args.input_video)
        if not in_path.exists():
            print(f"❌ Error: Input video '{in_path}' does not exist.")
            sys.exit(1)
            
        out_path = Path(args.output) if args.output else in_path.parent / f"{in_path.stem}_optimized{in_path.suffix}"
        success = compress_video(in_path, out_path, target_resolution=args.resolution, fps=args.fps)
        if not success:
            sys.exit(1)
            
    elif args.command == "url-import":
        storage_dir = Path("./storage/downloads")
        downloaded_file = download_video_url(args.url, storage_dir)
        if not downloaded_file:
            print("❌ Download failed.")
            sys.exit(1)
            
        target_file = downloaded_file
        if args.compress:
            compressed_file = downloaded_file.parent / f"{downloaded_file.stem}_optimized{downloaded_file.suffix}"
            if compress_video(downloaded_file, compressed_file):
                # Clean up raw download to save space
                downloaded_file.unlink()
                target_file = compressed_file
                
        # Run pipeline
        cmd = [sys.executable, "main.py", str(target_file)] + forward_params
        print(f"\n[Pipeline] Running Highlight Clipper on imported video:")
        print(f"  Executing: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.command == "watch":
        watch_dir = Path(args.dir)
        archive_dir = Path(args.archive)
        start_watch_folder(watch_dir, archive_dir, forward_params)

if __name__ == "__main__":
    main()
