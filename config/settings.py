import os
from pathlib import Path
from ultralytics import settings as ultralytics_settings

# --- 1. 基礎路徑設定 (D 槽) ---
# 確保這個路徑是你想要存放所有資料的地方
BASE_STORAGE_DIR = Path("D:/AI_Project_Data/TableTennis_Highlight")

# 自動建立子目錄
MODEL_DIR = BASE_STORAGE_DIR / "weights"
OUTPUT_DIR = BASE_STORAGE_DIR / "clips"
LOG_DIR = BASE_STORAGE_DIR / "logs"

# 確保目錄存在
for p in [MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# 嘗試告訴 Ultralytics 改設定 (作為備案)
ultralytics_settings.update({'weights_dir': str(MODEL_DIR)})

# --- 2. 模型設定 (關鍵修改：分離檔名與路徑) ---
# 檔名 (用來告訴 YOLO 要下載哪一個)
WORLD_MODEL_NAME = "yolov8l-worldv2.pt"
POSE_MODEL_NAME = "yolo11l-pose.pt"

# 絕對路徑 (用來告訴程式檔案應該在哪裡)
WORLD_MODEL_PATH = MODEL_DIR / WORLD_MODEL_NAME
POSE_MODEL_PATH = MODEL_DIR / POSE_MODEL_NAME

# --- 3. 演算法參數 ---
ALGO_PARAMS = {
    "table_search_frames": 90,
    "min_rally_duration": 1.5,      # 超短回合也要
    "max_dropout_duration": 3.0,    # 允許 ID 消失久一點 (容忍遮擋)
    "vip_warmup_score": 20,         # 幾乎是「一站上球桌就開始錄」
    "score_in_frame": 1,
    "score_in_core": 5,             # 核心區加權
    "core_zone_expansion": 1.4      # 擴大判定範圍
}