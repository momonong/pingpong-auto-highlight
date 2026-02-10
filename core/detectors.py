import cv2
import shutil
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Tuple

def load_model_safely(model_name: str, target_path: Path, device: str) -> YOLO:
    """
    強制將模型檔案管理在指定路徑 (D槽)。
    如果 D 槽沒有，就下載並移動過去。
    """
    # 1. 如果 D 槽已經有檔案，直接讀取絕對路徑
    if target_path.exists():
        print(f"[Loader] Found model at {target_path}, loading...")
        return YOLO(str(target_path), task='detect') # task='detect' is safer to infer

    # 2. 如果 D 槽沒有，先用檔名初始化 (這會觸發下載到目前目錄)
    print(f"[Loader] Model not found at {target_path}. Downloading...")
    temp_model = YOLO(model_name) 
    
    # 3. 下載完後，檢查是否出現在根目錄，並移動到 D 槽
    local_file = Path(model_name)
    if local_file.exists():
        print(f"[Loader] Moving {local_file} to {target_path}...")
        shutil.move(str(local_file), str(target_path))
        # 4. 移動完後，重新從 D 槽讀取
        return YOLO(str(target_path))
    else:
        # 萬一 Ultralytics 真的聽話下載到設定的目錄了，就直接回傳 temp_model
        return temp_model

class TableDetector:
    def __init__(self, model_name: str, model_path: Path, device: str = '0'):
        # 使用新的安全載入邏輯
        self.model = load_model_safely(model_name, model_path, device)
        self.device = device

    def find_table_roi(self, video_path: str, search_frames: int = 90) -> Optional[Tuple[int, int, int, int]]:
        cap = cv2.VideoCapture(video_path)
        
        # 多種 Prompts 增加成功率
        prompts = ["ping pong table", "table", "tennis table"]
        self.model.set_classes(prompts)
        
        max_area = 0
        best_box = None
        
        print(f"Scanning for table (First {search_frames} frames)...")
        for i in range(search_frames):
            ret, frame = cap.read()
            if not ret: break
            
            if i % 5 != 0: continue 

            results = self.model.predict(frame, verbose=False, device=self.device, conf=0.1)
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    
                    frame_area = frame.shape[0] * frame.shape[1]
                    if area < frame_area * 0.05: continue

                    if area > max_area:
                        max_area = area
                        best_box = (int(x1), int(y1), int(x2), int(y2))
        
        cap.release()
        return best_box

    @staticmethod
    def calculate_core_zone(table_box, frame_wh, expansion=1.2):
        tx1, ty1, tx2, ty2 = table_box
        w_img, h_img = frame_wh
        w_table, h_table = tx2 - tx1, ty2 - ty1
        cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        
        zone_w = w_table * expansion
        zone_h = h_table * expansion * 1.5 
        
        zx1 = max(0, cx - zone_w / 2)
        zy1 = max(0, cy - zone_h / 2)
        zx2 = min(w_img, cx + zone_w / 2)
        zy2 = min(h_img, cy + zone_h / 2)
        
        return (int(zx1), int(zy1), int(zx2), int(zy2))

class PoseEngine:
    def __init__(self, model_name: str, model_path: Path, device: str = '0'):
        # 使用新的安全載入邏輯
        self.model = load_model_safely(model_name, model_path, device)
        self.device = device
        
    def track(self, frame, persist=True):
        return self.model.track(frame, persist=persist, verbose=False, device=self.device)