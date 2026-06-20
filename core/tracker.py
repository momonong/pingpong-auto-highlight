from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

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

        # 桌球軌跡相關
        self.ball_history: List[Tuple[float, float, float]] = []  # 儲存 (time, x, y)
        self.rally_ball_positions: List[Tuple[float, float, float]] = [] # 新增: 追蹤當前 rally 中的所有球座標
        self.ball_detections_in_current_rally = 0
        self.frames_in_current_rally = 0

    def _is_in_zone(self, point: Tuple[float, float]) -> bool:
        if self.core_zone is None:
            return False
        px, py = point
        x1, y1, x2, y2 = self.core_zone
        return x1 <= px <= x2 and y1 <= py <= y2

    def update(self, current_time: float, track_results, ball_pos: Tuple[int, int] = None, core_zone: Tuple[int, int, int, int] = None) -> None:
        """每一幀呼叫此函式更新狀態"""
        if core_zone is not None:
            self.core_zone = core_zone
        current_frame_ids = []
        
        # 更新球軌跡
        if ball_pos is not None:
            bx, by = ball_pos
            self.ball_history.append((current_time, bx, by))
            if self.is_rallying:
                self.ball_detections_in_current_rally += 1
                self.rally_ball_positions.append((current_time, bx, by))
                
        # 移除超過 2.0 秒的軌跡
        self.ball_history = [h for h in self.ball_history if current_time - h[0] <= 2.0]
        
        if self.is_rallying:
            self.frames_in_current_rally += 1
        
        # Map raw track IDs to stabilized spatial IDs to handle ID switching
        stabilized_detections = []
        if track_results[0].boxes.id is not None:
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.data.cpu().numpy()
            
            candidates = []
            for tid, kp in zip(track_ids, keypoints):
                valid_kps = kp[kp[:, 2] > 0.3]
                if len(valid_kps) > 0:
                    center = np.mean(valid_kps[:, :2], axis=0)
                    candidates.append((tid, kp, center))
            
            if candidates:
                if self.core_zone is not None:
                    zx1, zy1, zx2, zy2 = self.core_zone
                    tc_x = (zx1 + zx2) / 2.0
                    tc_y = (zy1 + zy2) / 2.0
                    is_side_view = (zx2 - zx1) / max(1.0, zy2 - zy1) > 1.2
                else:
                    tc_x, tc_y = 640.0, 360.0
                    is_side_view = False
                
                # Filter by distance to table center to exclude background people (keep at most 2 players)
                candidates.sort(key=lambda c: (c[2][0] - tc_x)**2 + (c[2][1] - tc_y)**2)
                active_candidates = candidates[:2]
                
                if len(active_candidates) == 2:
                    c1, c2 = active_candidates
                    if is_side_view:
                        if c1[2][0] < c2[2][0]:
                            stabilized_detections.append((-100, c1[1]))
                            stabilized_detections.append((-101, c2[1]))
                        else:
                            stabilized_detections.append((-101, c1[1]))
                            stabilized_detections.append((-100, c2[1]))
                    else:
                        if c1[2][1] < c2[2][1]:
                            stabilized_detections.append((-100, c1[1]))
                            stabilized_detections.append((-101, c2[1]))
                        else:
                            stabilized_detections.append((-101, c1[1]))
                            stabilized_detections.append((-100, c2[1]))
                elif len(active_candidates) == 1:
                    c = active_candidates[0]
                    if is_side_view:
                        spatial_id = -100 if c[2][0] < tc_x else -101
                    else:
                        spatial_id = -100 if c[2][1] < tc_y else -101
                    stabilized_detections.append((spatial_id, c[1]))

        for tid, kp in stabilized_detections:
            if tid not in self.players:
                self.players[tid] = PlayerStats(tid)
            
            player = self.players[tid]
            current_frame_ids.append(tid)
            player.last_seen_time = current_time
            
            # --- [修正 1] 身體特徵點檢查 ---
            check_points = [kp[11], kp[12], kp[13], kp[14], kp[15], kp[16]]
            
            in_core = False
            for cx, cy, conf in check_points:
                if conf > 0.3:
                    if self._is_in_zone((cx, cy)):
                        in_core = True
                        break
            
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

    def _estimate_hits(self) -> int:
        """
        根據球的運動軌跡估算擊球次數。
        橫向鏡頭看 X 軸速度變化，縱向鏡頭看 Y 軸速度變化。
        """
        if len(self.rally_ball_positions) < 3:
            return 0
        
        # 決定相機方向
        if self.core_zone is not None:
            zx1, zy1, zx2, zy2 = self.core_zone
            is_side_view = (zx2 - zx1) / max(1.0, zy2 - zy1) > 1.2
        else:
            is_side_view = False
            
        vel_list = []
        for i in range(1, len(self.rally_ball_positions)):
            dt = self.rally_ball_positions[i][0] - self.rally_ball_positions[i-1][0]
            if dt > 0:
                coord_idx = 1 if is_side_view else 2 # 1=x, 2=y
                dv = (self.rally_ball_positions[i][coord_idx] - self.rally_ball_positions[i-1][coord_idx]) / dt
                vel_list.append((self.rally_ball_positions[i][0], dv))
        
        hits = 0
        current_dir = 0
        last_change_time = 0.0
        
        for t, dv in vel_list:
            if abs(dv) > 80.0:  # 速度門檻 (像素/秒)
                new_dir = 1 if dv > 0 else -1
                if current_dir == 0:
                    current_dir = new_dir
                    last_change_time = t
                elif new_dir != current_dir and (t - last_change_time > 0.3):
                    hits += 1
                    current_dir = new_dir
                    last_change_time = t
        return hits

    def _manage_state(self, is_active: bool, now: float, current_vips: List[int]):
        """狀態機管理"""
        if is_active:
            self.last_active_time = now
            if not self.is_rallying:
                self.is_rallying = True
                self.rally_start_time = now
                self.rally_ball_positions = [] # 開始時清空球跡
                self.ball_detections_in_current_rally = 0
                self.frames_in_current_rally = 0
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
                    
                    # 計算球活動比例
                    ball_ratio = 0.0
                    if self.frames_in_current_rally > 0:
                        ball_ratio = self.ball_detections_in_current_rally / self.frames_in_current_rally
                    
                    estimated_hits = self._estimate_hits()
                    
                    self.captured_rallies.append((final_start, final_end, ball_ratio, estimated_hits))
                    print(f"✅ Highlight Proposal: {final_start:.1f}s - {final_end:.1f}s (Dur: {duration:.1f}s, Ball Activity: {ball_ratio:.1%}, Hits: {estimated_hits}) | Active VIPs: {current_vips}")