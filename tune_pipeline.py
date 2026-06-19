#!/usr/bin/env python3
import sys
import os
import subprocess
import re
from pathlib import Path
import cv2

def get_video_duration(video_path: Path) -> float:
    """Uses OpenCV to calculate the duration of a video file in seconds."""
    if not video_path.exists():
        return 0.0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0.0

def run_highlight_pipeline(video_path: Path, params: dict) -> dict:
    """Runs main.py with the specified parameters and captures output metrics."""
    cmd = [sys.executable, "main.py", str(video_path)]
    for key, val in params.items():
        cmd.append(f"--{key}")
        cmd.append(str(val))
    
    print(f"\n[Tuner] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    stdout = result.stdout
    stderr = result.stderr
    
    # Parse metrics from logs
    table_found = "Table Found:" in stdout
    fallback_used = "Warning: No table detected. Using Center 50%" in stdout
    
    # Parse highlight count from stdout
    highlight_count = 0
    match = re.search(r"Found (\d+) highlights", stdout)
    if match:
        highlight_count = int(match.group(1))
        
    return {
        "success": result.returncode == 0,
        "table_found": table_found and not fallback_used,
        "fallback_used": fallback_used,
        "highlight_count": highlight_count,
        "stdout": stdout,
        "stderr": stderr
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python tune_pipeline.py <video_file_path> [min_clips] [max_clips]")
        sys.exit(1)
        
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
        
    # Configure clip targets (Defaults: target between 3 and 10 highlights)
    min_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_clips = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    print(f"=== Starting TTHAC Automated Parameter Tuner ===")
    print(f"Input Video: {video_path.name}")
    print(f"Target Highlights Count: {min_clips} to {max_clips}")
    
    # Base configuration starting points
    params = {
        "table_search_frames": 90,
        "min_rally_duration": 1.5,
        "max_dropout_duration": 3.0,
        "vip_warmup_score": 20,
        "score_in_frame": 1,
        "score_in_core": 5,
        "core_zone_expansion": 1.4
    }
    
    max_iterations = 5
    best_run = None
    best_score_diff = float('inf')
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        print(f"Current Tuning Params: {params}")
        
        run_res = run_highlight_pipeline(video_path, params)
        
        if not run_res["success"]:
            print(f"[Tuner] Execution failed! Error:")
            print(run_res["stderr"])
            break
            
        highlight_count = run_res["highlight_count"]
        table_found = run_res["table_found"]
        fallback_used = run_res["fallback_used"]
        
        # Verify generated clips and check their actual duration
        video_stem = video_path.stem
        # Base storage path uses local project root / storage (configured in settings)
        clips_dir = Path(__file__).parent / "storage" / "clips" / video_stem
        
        actual_clips = []
        if clips_dir.exists():
            actual_clips = sorted(list(clips_dir.glob("highlight_*.mp4")))
            
        clip_durations = [get_video_duration(clip) for clip in actual_clips]
        
        print(f"[Tuner] Results - Highlight Count (from Log): {highlight_count}, Verified Files: {len(actual_clips)}")
        if clip_durations:
            print(f"[Tuner] Verified Clips Durations: {', '.join([f'{d:.1f}s' for d in clip_durations])}")
            print(f"[Tuner] Average Clip Duration: {sum(clip_durations)/len(clip_durations):.1f}s")
            
        # Determine deviation from targets
        if highlight_count < min_clips:
            score_diff = min_clips - highlight_count
        elif highlight_count > max_clips:
            score_diff = highlight_count - max_clips
        else:
            score_diff = 0
            
        # Record best run
        if score_diff < best_score_diff:
            best_score_diff = score_diff
            best_run = {
                "iteration": iteration,
                "params": params.copy(),
                "highlight_count": highlight_count,
                "table_found": table_found,
                "clip_durations": clip_durations
            }
            
        # Check success condition
        if score_diff == 0:
            print(f"\n🎉 Success! Target highlights reached on iteration {iteration}.")
            break
            
        # Apply tuning heuristics
        if highlight_count == 0:
            print("[Tuner] Symptom: 0 highlights detected.")
            if fallback_used:
                print("[Tuner] Tuning Action: Table detection failed. Searching more frames.")
                params["table_search_frames"] = min(300, params["table_search_frames"] + 60)
            else:
                print("[Tuner] Tuning Action: Table found, but no rally captured. Lowering VIP warmup criteria.")
                params["vip_warmup_score"] = max(3, int(params["vip_warmup_score"] * 0.4))
                params["core_zone_expansion"] = round(min(2.0, params["core_zone_expansion"] + 0.2), 2)
        elif highlight_count < min_clips:
            print(f"[Tuner] Symptom: Too few highlights ({highlight_count} < {min_clips}).")
            print("[Tuner] Tuning Action: Making detection more sensitive (Lowering warmup, extending core zone, reducing min duration).")
            params["vip_warmup_score"] = max(3, int(params["vip_warmup_score"] * 0.6))
            params["min_rally_duration"] = round(max(1.0, params["min_rally_duration"] - 0.3), 2)
            params["max_dropout_duration"] = round(min(6.0, params["max_dropout_duration"] + 0.5), 2)
        elif highlight_count > max_clips:
            print(f"[Tuner] Symptom: Too many highlights ({highlight_count} > {max_clips}).")
            print("[Tuner] Tuning Action: Making detection stricter (Raising warmup, increasing min duration, reducing dropout tolerance).")
            params["vip_warmup_score"] = min(100, int(params["vip_warmup_score"] * 1.5))
            params["min_rally_duration"] = round(min(5.0, params["min_rally_duration"] + 0.5), 2)
            params["max_dropout_duration"] = round(max(1.5, params["max_dropout_duration"] - 0.5), 2)
            params["core_zone_expansion"] = round(max(1.1, params["core_zone_expansion"] - 0.1), 2)

    # Output final summary
    print("\n================ Tuning Summary ================")
    if best_run:
        print(f"Best Configuration Found (Iteration {best_run['iteration']}):")
        print(f"  Params: {best_run['params']}")
        print(f"  Highlight Clips Generated: {best_run['highlight_count']}")
        if best_run['clip_durations']:
            print(f"  Clip Durations: {', '.join([f'{d:.1f}s' for d in best_run['clip_durations']])}")
        if best_score_diff == 0:
            print("Status: Target constraints successfully satisfied.")
        else:
            print(f"Status: Underperformed target by {best_score_diff} highlights, but best configuration saved.")
    else:
        print("Status: No successful run executed.")
    print("================================================")

if __name__ == "__main__":
    main()
