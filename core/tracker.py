from collections import defaultdict
from typing import Dict, List, Set, Tuple

class PlayerStats:
    """單一玩家的狀態資料結構"""
    def __init__(self, player_id: int):
        self.id = player_id
        self.score = 0
        self.frames_in_core = 0  # 新增：記錄在核心區待了幾幀
        self.last_seen_time = 0.0
        self.is_vip = False

class VIPGameTracker:
    """管理所有玩家積分與 Rally 狀態判斷"""
    def __init__(self, config: dict, core_zone: Tuple[int, int, int, int]):
        self.cfg = config
        self.core_zone = core_zone  # (x1, y1, x2, y2)
        self.players: Dict[int, PlayerStats] = {}
        
        # Rally 狀態
        self.is_rallying = False
        self.rally_start_time = 0.0
        self.last_active_time = 0.0
        self.captured_rallies = []

    def _is_in_zone(self, point: Tuple[float, float]) -> bool:
        px, py = point
        x1, y1, x2, y2 = self.core_zone
        return x1 <= px <= x2 and y1 <= py <= y2

    def update(self, current_time: float, track_results) -> None:
        """每一幀呼叫此函式更新狀態"""
        current_frame_ids = []
        
        if track_results[0].boxes.id is not None:
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.data.cpu().numpy()
            
            for tid, kp in zip(track_ids, keypoints):
                if tid not in self.players:
                    self.players[tid] = PlayerStats(tid)
                
                player = self.players[tid]
                current_frame_ids.append(tid)
                player.last_seen_time = current_time
                
                # --- [修正 1] 身體特徵點檢查 ---
                # 原本只看腳踝 (15, 16)，現在加入 膝蓋(13, 14) 和 臀部(11, 12)
                # 只要任何一點在核心區，就算得分
                # Keypoint indices: 11-12 (Hips), 13-14 (Knees), 15-16 (Ankles)
                check_points = [kp[11], kp[12], kp[13], kp[14], kp[15], kp[16]]
                
                in_core = False
                for cx, cy, conf in check_points:
                    if conf > 0.3: # 稍微降低信心門檻
                        if self._is_in_zone((cx, cy)):
                            in_core = True
                            break # 只要有一點在裡面就算數
                
                # --- 計分邏輯 ---
                score_gain = self.cfg['score_in_frame']
                if in_core:
                    score_gain += self.cfg['score_in_core']
                    player.frames_in_core += 1
                
                player.score += score_gain
                
                # VIP 晉升檢查
                if player.score > self.cfg['vip_warmup_score']:
                    player.is_vip = True

        # --- [修正 2] 寬鬆版狀態判定 ---
        # 找出當前在畫面中的 VIP
        active_vips_in_frame = []
        for pid in current_frame_ids:
            if self.players[pid].is_vip:
                active_vips_in_frame.append(pid)
        
        is_active_moment = False
        
        # 條件放寬：
        # 只要有「至少 1 位」VIP 在畫面中，並且該 VIP 最近有在核心區活動，我們就視為 Rally 進行中。
        # 這樣即使其中一人被擋住，或者只是練球，也能錄下來。
        if len(active_vips_in_frame) >= 1:
            # 進一步檢查：這些在場的 VIP，真的有在核心區打球嗎？
            # 我們檢查他們的 score 是否足夠高 (代表長期在核心區)
            strong_vip_present = False
            for vid in active_vips_in_frame:
                if self.players[vid].frames_in_core > 30: # 至少在核心區待過 1 秒
                    strong_vip_present = True
                    break
            
            if strong_vip_present:
                is_active_moment = True
        
        self._manage_state(is_active_moment, current_time, active_vips_in_frame)

    def _manage_state(self, is_active: bool, now: float, current_vips: List[int]):
        """狀態機管理"""
        if is_active:
            self.last_active_time = now
            if not self.is_rallying:
                self.is_rallying = True
                self.rally_start_time = now
        else:
            # 檢查 Dropout
            if self.is_rallying and (now - self.last_active_time > self.cfg['max_dropout_duration']):
                self.is_rallying = False
                rally_end_time = self.last_active_time
                
                duration = rally_end_time - self.rally_start_time
                if duration >= self.cfg['min_rally_duration']:
                    # 加入 Padding (往前多抓一點，確保發球有被錄到)
                    final_start = max(0, self.rally_start_time - 3.0) 
                    final_end = rally_end_time + 2.0
                    
                    self.captured_rallies.append((final_start, final_end))
                    print(f"✅ Highlight: {final_start:.1f}s - {final_end:.1f}s (Dur: {duration:.1f}s) | Active VIPs: {current_vips}")