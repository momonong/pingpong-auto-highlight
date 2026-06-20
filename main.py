import sys
import cv2
import subprocess  # <--- 新增這個
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

# 導入模組
from config import settings
from core.detectors import TableDetector, PoseEngine, BallDetector
from core.tracker import VIPGameTracker

# --- 新增：更穩定的剪輯合併與 AI 導演函式 ---
def concatenate_videos(clips, output_path):
    """
    使用 FFmpeg 快速無損合併多個影片剪輯。
    """
    if not clips:
        return
    file_list_path = output_path.parent / "file_list.txt"
    # 寫入暫存的合併清單
    with open(file_list_path, "w", encoding="utf-8") as f:
        for clip_path, _, _, _ in clips:
            f.write(f"file '{Path(clip_path).name}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(file_list_path),
        "-c", "copy",
        str(output_path)
    ]
    # 在輸出目錄下執行，避免路徑轉義問題
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(output_path.parent))
    if file_list_path.exists():
        file_list_path.unlink()

def run_agentic_director(clips, output_dir, api_key):
    """
    呼叫 Gemini VLM (多模態 API) 自動驗證、評分並描述每一段 Highlight，最後過濾並重新排序合併。
    """
    try:
        from google import genai
    except ImportError:
        print("[Agentic Director] 正在安裝 google-genai 函式庫...")
        subprocess.run([sys.executable, "-m", "pip", "install", "google-genai"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            from google import genai
        except ImportError:
            print("[Agentic Director] ❌ 無法安裝或載入 google-genai 函式庫。跳過 VLM 驗證。")
            return
            
    client = genai.Client(api_key=api_key)
    print("\n[Agentic Director] 🎬 開始進行 AI Highlight 影片導演分析與過濾...")
    
    report_lines = [
        "# 🎬 Table Tennis Highlight Analysis Report",
        f"\n**Source Video:** {output_dir.name}",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "\n| 剪輯檔名 | 時間區間 | 球體活動率 | AI 精彩度評分 | 該球勝方 | 精彩對決內容描述 | 狀態 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    verified_clips = []
    
    for i, (clip_path, start, end, ball_ratio) in enumerate(clips):
        print(f"  [{i+1}/{len(clips)}] 正在上傳與分析: {clip_path.name}...")
        try:
            # 上傳影片到 Gemini 暫存空間
            video_file = client.files.upload(file=str(clip_path))
            
            # 等待影片處理完成
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)
                
            if video_file.state.name == "FAILED":
                print(f"    ⚠️ 影片上傳處理失敗: {clip_path.name}")
                continue
                
            prompt = (
                "You are an expert table tennis referee and sports video director. Analyze this video clip of a table tennis rally. "
                "Provide a JSON response with the following keys:\n"
                "1. 'is_valid_rally': boolean (true if it represents actual table tennis play/rally, false if it's just players picking up the ball, walking around, or adjusting equipment)\n"
                "2. 'intensity_score': integer between 1 and 10 (rating how exciting the rally is based on exchange length, smashes, player movements)\n"
                "3. 'winner': string ('Player Left', 'Player Right', or 'Unknown')\n"
                "4. 'description': string (a short, exciting 1-sentence description of the rally highlights in Traditional Chinese)\n"
                "Respond ONLY with a valid JSON block."
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[video_file, prompt]
            )
            
            text = response.text
            
            # 擷取 JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                
            import json
            res_data = json.loads(text.strip())
            
            is_valid = res_data.get("is_valid_rally", True)
            intensity = res_data.get("intensity_score", 5)
            winner = res_data.get("winner", "Unknown")
            desc = res_data.get("description", "精彩乒乓球來回對決。")
            
            status = "✅ 經 AI 驗證" if is_valid else "❌ 雜訊/誤判 (過濾)"
            report_lines.append(
                f"| {clip_path.name} | {start:.1f}s - {end:.1f}s | {ball_ratio:.1%} | {intensity}/10 | {winner} | {desc} | {status} |"
            )
            
            if is_valid:
                verified_clips.append((clip_path, start, end, ball_ratio, intensity))
                
            # 刪除檔案，避免佔用雲端空間
            client.files.delete(name=video_file.name)
            
        except Exception as e:
            print(f"    ⚠️ VLM 分析出錯 ({clip_path.name}): {e}")
            report_lines.append(
                f"| {clip_path.name} | {start:.1f}s - {end:.1f}s | {ball_ratio:.1%} | N/A | Unknown | VLM 處理錯誤或超時 | ⚠️ 未能驗證 |"
            )
            verified_clips.append((clip_path, start, end, ball_ratio, 5))
            
    # 寫入 Markdown 報告
    report_path = output_dir / "highlight_report.md"
    report_path.write_text("\n".join(report_lines), encoding='utf-8')
    print(f"\n[Agentic Director] AI 分析報告已儲存至: {report_path}")
    
    # 根據 AI 精彩度評分，將 Verified Clips 重新排序（高分到低分）並重新合併
    if verified_clips:
        verified_clips.sort(key=lambda x: x[4], reverse=True)
        print(f"[Agentic Director] 正在將 {len(verified_clips)} 段經驗證的 Highlight (已依精彩度評分排序) 合併為單一精華影片...")
        concatenate_videos([(vc[0], vc[1], vc[2], vc[3]) for vc in verified_clips], output_dir / "final_highlight_reel.mp4")
        print(f"🎉 [Agentic Director] 完成！已合併精華影片儲存至: {output_dir / 'final_highlight_reel.mp4'}")
    else:
        print("[Agentic Director] ⚠️ 沒有任何片段通過 AI 乒乓球來回對決驗證。")

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

def is_scene_change(prev_frame, curr_frame, threshold=0.55) -> bool:
    """
    Detects if there is a scene cut/change between two frames using downsampled HSV histogram correlation.
    """
    if prev_frame is None or curr_frame is None:
        return False
    try:
        prev_small = cv2.resize(prev_frame, (128, 128))
        curr_small = cv2.resize(curr_frame, (128, 128))
        
        hsv_prev = cv2.cvtColor(prev_small, cv2.COLOR_BGR2HSV)
        hsv_curr = cv2.cvtColor(curr_small, cv2.COLOR_BGR2HSV)
        
        hist_prev = cv2.calcHist([hsv_prev], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [16, 16], [0, 180, 0, 256])
        
        cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)
        
        metric = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        return metric < threshold
    except Exception:
        return False

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
    ball_detector = BallDetector(
        settings.BALL_MODEL_NAME,
        settings.BALL_MODEL_PATH
    )
    # 2. 尋找球桌 (使用 settings 中的搜尋範圍)
    # 有時候影片剛開始會有人擋住鏡頭，多看幾秒比較準
    table_box = world_detector.find_table_roi(str(video_path), search_frames=settings.ALGO_PARAMS['table_search_frames'])
    
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
    current_table_box = table_box
    current_core_zone = core_zone
    tracker = VIPGameTracker(settings.ALGO_PARAMS, current_core_zone)
    
    # 4. 主迴圈
    print("Starting Analysis Loop...")
    pbar = tqdm(total=total_frames, unit="frame")
    
    debug_counter = 0
    frame_idx = 0
    prev_frame_small = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # 偵測鏡頭切換 (Scene Cut)
        scene_cut = False
        if frame_idx % 5 == 0:  # 每 5 幀檢查一次以節省計算資源
            curr_small = cv2.resize(frame, (128, 128))
            if prev_frame_small is not None:
                if is_scene_change(prev_frame_small, curr_small, threshold=0.55):
                    scene_cut = True
                    tqdm.write(f"  [Camera Shift] Frame {frame_idx}: Detected camera angle/scene switch at {current_time:.1f}s.")
            prev_frame_small = curr_small
        
        # 每 90 幀 (約 3 秒) 或者在鏡頭切換時，動態重新偵測球桌以適應不同視角或鏡頭移動
        if scene_cut or (frame_idx % 90 == 0):
            new_box = world_detector.detect_table_in_frame(frame, conf_threshold=0.15)
            if new_box is not None:
                current_table_box = new_box
                current_core_zone = world_detector.calculate_core_zone(
                    new_box, (width, height), settings.ALGO_PARAMS['core_zone_expansion']
                )
                # tqdm.write(f"  [Dynamic Table] Frame {frame_idx}: Updated core zone to {current_core_zone}")
            else:
                if scene_cut:
                    # 如果切換了鏡頭而且完全找不到球桌（例如特寫畫面），暫時關閉核心判定區以避免誤判
                    current_table_box = None
                    current_core_zone = None
                    # tqdm.write(f"  [Dynamic Table] Frame {frame_idx}: Table lost on scene cut. Suspending core zone.")

        results = pose_engine.track(frame)
        ball_pos = ball_detector.detect(frame)
        
        # 傳入動態的 core_zone 給追蹤器
        tracker.update(current_time, results, ball_pos, core_zone=current_core_zone)
        
        # --- Debug 區塊 ---
        debug_counter += 1
        if debug_counter % 100 == 0: # 每 100 幀印一次
            top_players = sorted(tracker.players.values(), key=lambda p: p.score, reverse=True)[:3]
            stats = [f"ID:{p.id}(Score:{p.score})" for p in top_players]
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
        exported_clips = []
        for i, (start, end, ball_ratio) in enumerate(tracker.captured_rallies):
            out_name = video_output_dir / f"highlight_{i+1:03d}.mp4"
            end = min(end, total_frames/fps)
            
            # 使用新的剪輯函式
            fast_cut_video(str(video_path), str(out_name), start, end)
            exported_clips.append((out_name, start, end, ball_ratio))
            
        print(f"✅ All individual clips exported. Saved to {video_output_dir}")
        
        # 預設無損合併所有剪輯
        default_reel_path = video_output_dir / "final_highlight_reel.mp4"
        concatenate_videos(exported_clips, default_reel_path)
        print(f"✅ Lossless compilation video created at: {default_reel_path}")

        # AI 導演過濾與評分 (如果有 API Key)
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            run_agentic_director(exported_clips, video_output_dir, api_key)
        else:
            print("\n💡 Tip: To enable automatic AI verification, intensity scoring, and video sorting, set your Gemini API key:")
            print("   export GEMINI_API_KEY='your-key-here'")
    else:
        print("No highlights found. Try adjusting 'score_in_core' or 'min_rally_duration' in settings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Tennis Highlight Clipper (TTHAC)")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--table_search_frames", type=int, default=None, help="Frames to search for the table")
    parser.add_argument("--min_rally_duration", type=float, default=None, help="Min rally duration in seconds")
    parser.add_argument("--max_dropout_duration", type=float, default=None, help="Max dropout duration in seconds")
    parser.add_argument("--vip_warmup_score", type=int, default=None, help="VIP warmup score threshold")
    parser.add_argument("--score_in_frame", type=int, default=None, help="Score per visible frame")
    parser.add_argument("--score_in_core", type=int, default=None, help="Score per frame in core zone")
    parser.add_argument("--core_zone_expansion", type=float, default=None, help="Core zone expansion ratio")
    
    args = parser.parse_args()
    
    # 套用外部參數覆蓋設定
    for param in [
        "table_search_frames", "min_rally_duration", "max_dropout_duration",
        "vip_warmup_score", "score_in_frame", "score_in_core", "core_zone_expansion"
    ]:
        val = getattr(args, param)
        if val is not None:
            settings.ALGO_PARAMS[param] = val
            
    main(args.video_path)