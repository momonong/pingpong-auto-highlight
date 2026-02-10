import sys
import cv2
import subprocess  # <--- 新增這個
from pathlib import Path
from tqdm import tqdm

# 導入模組
from config import settings
from core.detectors import TableDetector, PoseEngine
from core.tracker import VIPGameTracker

# --- 新增：更穩定的剪輯函式 (不依賴 moviepy 版本) ---
def fast_cut_video(input_path: str, output_path: str, start_time: float, end_time: float):
    """
    使用 FFmpeg 直接剪輯 (Stream Copy)，不重新編碼，速度最快且畫質無損。
    """
    cmd = [
        "ffmpeg", "-y",             # -y: 自動覆蓋檔案
        "-ss", str(start_time),     # 開始時間 (必須放在 -i 之前以加速搜尋)
        "-i", input_path,           # 輸入檔案
        "-t", str(end_time - start_time), # 持續時間
        "-c", "copy",               # 影像與聲音直接複製 (不編碼)
        "-avoid_negative_ts", "1",  # 修正時間戳記
        str(output_path)            # 輸出檔案
    ]
    # 執行指令，並隱藏冗長的輸出
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main(video_path_str: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"=== Table Tennis Highlight Clipper (TTHAC) v1.2 ===")
    print(f"Processing: {video_path.name}")
    print(f"Target Model Storage: {settings.MODEL_DIR}")
    
    # 1. 初始化模型 (傳入 Name 和 Path)
    # 這樣 detectors 裡面就會執行「下載 -> 移動」的動作
    world_detector = TableDetector(
        settings.WORLD_MODEL_NAME, 
        settings.WORLD_MODEL_PATH
    )
    pose_engine = PoseEngine(
        settings.POSE_MODEL_NAME, 
        settings.POSE_MODEL_PATH
    )
    # 2. 尋找球桌 (增加搜尋範圍到 90 幀 = 3秒)
    # 有時候影片剛開始會有人擋住鏡頭，多看幾秒比較準
    table_box = world_detector.find_table_roi(str(video_path), search_frames=90)
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if table_box:
        print(f"✅ Table Found: {table_box}")
        core_zone = world_detector.calculate_core_zone(table_box, (width, height), 
                                                     settings.ALGO_PARAMS['core_zone_expansion'])
    else:
        # Fallback: 如果真的找不到桌子，我們假設桌子在畫面正中央 50% 的區域
        # 這樣比「全螢幕」好，至少能過濾掉邊緣的路人
        print("⚠️ Warning: No table detected. Using Center 50% as core zone.")
        cw, ch = width * 0.5, height * 0.5
        cx, cy = width / 2, height / 2
        core_zone = (int(cx - cw/2), int(cy - ch/2), int(cx + cw/2), int(cy + ch/2))
    
    print(f"Core Zone: {core_zone}")
    
    # 3. 初始化追蹤器
    tracker = VIPGameTracker(settings.ALGO_PARAMS, core_zone)
    
    # 4. 主迴圈
    print("Starting Analysis Loop...")
    pbar = tqdm(total=total_frames, unit="frame")
    
    # 記錄每 100 幀印出一次 Debug 資訊
    debug_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        results = pose_engine.track(frame)
        tracker.update(current_time, results)
        
        # --- [新增] Debug 區塊 ---
        debug_counter += 1
        if debug_counter % 100 == 0: # 每 100 幀 (約 3 秒) 印一次
            # 找出目前分數最高的幾個 ID
            top_players = sorted(tracker.players.values(), key=lambda p: p.score, reverse=True)[:3]
            stats = [f"ID:{p.id}(Score:{p.score})" for p in top_players]
            # 使用 tqdm.write 才不會打亂進度條
            # 顯示目前狀態：Is_Rallying? 以及前三名分數
            # tqdm.write(f"Time:{current_time:.1f}s | Rally:{tracker.is_rallying} | Stats: {stats}")
            pass 
        # -----------------------

        pbar.update(1)
        
    cap.release()
    pbar.close()
    
    # 5. 輸出剪輯
    print(f"\nAnalysis Complete. Found {len(tracker.captured_rallies)} highlights.")
    
    if tracker.captured_rallies:
        video_output_dir = settings.OUTPUT_DIR / video_path.stem
        video_output_dir.mkdir(exist_ok=True)
        
        print(f"Exporting clips to {video_output_dir}...")
        for i, (start, end) in enumerate(tracker.captured_rallies):
            out_name = video_output_dir / f"highlight_{i+1:03d}.mp4"
            end = min(end, total_frames/fps)
            
            # 使用新的剪輯函式
            fast_cut_video(str(video_path), str(out_name), start, end)
            
        print(f"✅ All Done! Saved to {video_output_dir}")
    else:
        print("No highlights found. Try adjusting 'score_in_core' or 'min_rally_duration' in settings.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_file_path>")
    else:
        main(sys.argv[1])