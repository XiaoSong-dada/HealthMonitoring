import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
from tracking_utils import KalmanFilter, TrackState, joint_stracks, sub_stracks, remove_duplicate_stracks, linear_assignment, iou_distance


@dataclass
class Object:
    """检测目标类"""
    rect: Tuple[int, int, int, int]  # (x, y, width, height)
    prob: float  # 检测置信度


class STrack:
    """单个目标跟踪器"""
    
    def __init__(self, tlwh: np.ndarray, score: float):
        """
        初始化目标跟踪器
        
        参数:
            tlwh: 目标框坐标及尺寸 [top_left_x, top_left_y, width, height]
            score: 检测得分
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.is_activated = False
        self.track_id = 0
        self.state = TrackState.New
        self.score = score
        self.max_frame_id = 0
        self.start_frame = 0
        self.frame_count = 0
        self.end_frame = 0
        self.time_since_update = 0
        self.history = []
        self.hits = []
        self.hit_streak = 0

    @property
    def tlwh(self):
        """获取当前边界框"""
        return self._tlwh
    
    @property
    def xyah(self):
        """获取卡尔曼滤波器状态"""
        return self.mean[:4]
    
    @property
    def tlbr(self):
        """转换为 [top left, bottom right] 坐标格式"""
        ret = self.xyah_to_tlbr(self.xyah)
        return ret
    
    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """将 [top left, bottom right] 转换为 [top left, width height] 格式"""
        ret = np.copy(tlbr)
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """将 [top left, width height] 转换为卡尔曼滤波器使用的 [center x, center y, aspect ratio, height] 格式"""
        ret = np.copy(tlwh)
        ret[:2] += tlwh[2:] / 2
        ret[2] /= tlwh[3]
        return ret
    
    @staticmethod
    def xyah_to_tlbr(xyah: np.ndarray) -> np.ndarray:
        """将 [center x, center y, aspect ratio, height] 转换为 [top left, bottom right] 格式"""
        x, y, a, h = xyah
        w = a * h
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2], dtype=np.float32)
    
    def predict(self, kalman_filter: KalmanFilter):
        """预测目标位置"""
        mean, covariance = kalman_filter.predict(self.mean, self.covariance)
        self.mean = mean
        self.covariance = covariance
        if self.state != TrackState.Tracked:
            self.hit_streak = 0
        self.time_since_update += 1
    
    def update(self, detection: "STrack"):
        """更新目标状态"""
        self.time_since_update = 0
        self.history = []
        self.hits.append(detection.score)
        self.hit_streak += 1
        self.score = detection.score
        
        xyah = detection.xyah
        mean, covariance = self.kalman_filter.update(self.mean, self.covariance, xyah)
        self.mean = mean
        self.covariance = covariance
        xyah = self.mean[:4]
        tlbr = STrack.xyah_to_tlbr(xyah)
        self._tlwh = STrack.tlbr_to_tlwh(tlbr)
    
    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """激活跟踪器"""
        self.track_id = self._get_next_id()
        mean, covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.mean = mean
        self.covariance = covariance
        self.is_activated = True
        self.frame_count = frame_id
        self.start_frame = frame_id
        self.hit_streak = 1
        self.hits = [self.score]
    
    def re_activate(self, detection: "STrack", frame_id: int, new_id: bool = False):
        """重新激活跟踪器"""
        self.time_since_update = 0
        self.hit_streak += 1
        self.hits.append(detection.score)
        self.score = detection.score
        
        xyah = detection.xyah
        mean, covariance = self.kalman_filter.update(self.mean, self.covariance, xyah)
        self.mean = mean
        self.covariance = covariance
        xyah = self.mean[:4]
        tlbr = STrack.xyah_to_tlbr(xyah)
        self._tlwh = STrack.tlbr_to_tlwh(tlbr)
        
        self.frame_count = frame_id
        if new_id:
            self.track_id = self._get_next_id()
    
    def mark_lost(self):
        """标记为目标丢失"""
        self.state = TrackState.Lost
    
    def mark_removed(self):
        """标记为目标移除"""
        self.state = TrackState.Removed
    
    def _get_next_id(self):
        """获取下一个track_id"""
        if not hasattr(STrack, 'next_id'):
            STrack.next_id = 0
        ret = STrack.next_id
        STrack.next_id = (STrack.next_id + 1) % math.pow(2, 32)
        return ret


class BYTETracker:
    """多目标跟踪器"""
    
    def __init__(self, frame_rate: int = 30, track_buffer: int = 30):
        """
        初始化BYTETracker
        
        参数:
            frame_rate: 视频帧率
            track_buffer: 跟踪缓冲区大小
        """
        self.track_thresh = 0.5     # 跟踪阈值
        self.high_thresh = 0.6      # 高阈值
        self.match_thresh = 0.8     # 匹配阈值
        
        self.frame_id = 0           # 当前帧ID
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)  # 最大丢失时间
        
        self.tracked_stracks = []   # 跟踪中的轨迹
        self.lost_stracks = []     # 丢失的轨迹
        self.removed_stracks = []  # 移除的轨迹

        # 添加 KalmanFilter 实例
        self.kalman_filter = KalmanFilter()  # 注意：和 STrack 内部使用的保持一致

        print("Init ByteTrack!")
    
    def update(self, objects: List[Object]) -> List[STrack]:
        """
        更新跟踪器
        
        参数:
            objects: 当前帧的检测对象列表
            
        返回:
            List[STrack]: 跟踪结果列表
        """
        self.frame_id += 1
        
        # Step 1: 获取检测结果
        detections = []
        detections_low = []
        
        if objects:
            for obj in objects:
                # 将矩形框转换为[top left, bottom right]格式
                x, y, width, height = obj.rect
                tlbr = [x, y, x + width, y + height]
                
                # 创建检测跟踪器
                score = obj.prob
                strack = STrack(np.array(tlbr, dtype=np.float32), score)
                
                # 根据阈值分组
                if score >= self.track_thresh:
                    detections.append(strack)
                else:
                    detections_low.append(strack)
        
        # 分离未确认和已确认的跟踪
        unconfirmed = []
        tracked_stracks = []
        
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Step 2: 第一次关联，使用IOU
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # 对所有轨迹进行预测
        for track in strack_pool:
            track.predict(self.kalman_filter)
        
        # 计算IOU距离
        dists = iou_distance(strack_pool, detections)
        
        # 进行线性分配
        matches, u_track, u_detection = [], [], []
        if dists:  # 检查 dists 是否为空
            matches, u_track, u_detection = linear_assignment(dists, self.match_thresh)
        else:
            u_track = list(range(len(strack_pool)))
            u_detection = list(range(len(detections)))
        
        # 处理匹配结果
        activated_stracks = []
        refind_stracks = []
        
        for track_idx, detection_idx in matches:
            track = strack_pool[track_idx]
            detection = detections[detection_idx]
            
            if track.state == TrackState.Tracked:
                track.update(detection)
                activated_stracks.append(track)
            else:
                track.re_activate(detection, self.frame_id, False)
                refind_stracks.append(track)
        
        # Step 3: 第二次关联，使用低分检测
        detections_cp = [detections[i] for i in u_detection]
        detections = detections_low
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        # 计算IOU距离
        dists = iou_distance(r_tracked_stracks, detections)
        
        # 进行线性分配
        matches, u_track, u_detection = [], [], []
        if dists:  # 检查 dists 是否为空
            matches, u_track, u_detection = linear_assignment(dists, 0.5)
        else:
            u_track = list(range(len(r_tracked_stracks)))
            u_detection = list(range(len(detections)))
        
        # 处理匹配结果
        for track_idx, detection_idx in matches:
            track = r_tracked_stracks[track_idx]
            detection = detections[detection_idx]
            
            if track.state == TrackState.Tracked:
                track.update(detection)
                activated_stracks.append(track)
            else:
                track.re_activate(detection, self.frame_id, False)
                refind_stracks.append(track)
        
        # 处理未匹配的跟踪
        for idx in u_track:
            track = r_tracked_stracks[idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                self.lost_stracks.append(track)
        
        # 处理未确认的跟踪
        detections = detections_cp
        dists = iou_distance(unconfirmed, detections)
        
        # 进行线性分配
        matches, u_unconfirmed, u_detection = [], [], []
        if dists:  # 检查 dists 是否为空
            matches, u_unconfirmed, u_detection = linear_assignment(dists, 0.7)
        else:
            u_unconfirmed = list(range(len(unconfirmed)))
            u_detection = list(range(len(detections)))
        
        # 处理匹配结果
        for track_idx, detection_idx in matches:
            unconfirmed[track_idx].update(detections[detection_idx])
            activated_stracks.append(unconfirmed[track_idx])
        
        # 处理未匹配的未确认跟踪
        for idx in u_unconfirmed:
            unconfirmed[idx].mark_removed()
            self.removed_stracks.append(unconfirmed[idx])
        
        # Step 4: 初始化新跟踪
        for idx in u_detection:
            detection = detections[idx]
            if detection.score < self.high_thresh:
                continue
            detection.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(detection)
        
        # Step 5: 更新状态
        # 更新丢失的跟踪
        removed_stracks = []
        lost_stracks = []  # 添加这一行以定义 lost_stracks
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame() > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
            else:
                lost_stracks.append(track)  # 保留未超时的丢失轨迹
        
        self.removed_stracks.extend(removed_stracks)
        
        # 清理已移除的跟踪
        self.tracked_stracks = [t for t in tracked_stracks 
                            if t.state == TrackState.Tracked]
        
        # 合并跟踪
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        
        # 更新丢失的跟踪
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        
        # 移除重复跟踪
        resa, resb = [], []
        # 先复制
        resa = self.tracked_stracks.copy()
        resb = self.lost_stracks.copy()

        # 调用函数，只传入两个跟踪列表
        resa, resb = remove_duplicate_stracks(resa, resb)
        
        self.tracked_stracks = resa
        self.lost_stracks = resb
        
        # 返回当前活跃的跟踪
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks