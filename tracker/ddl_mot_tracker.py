#--*-- coding:utf-8 --*--
import numpy as np
from numba import jit
from collections import OrderedDict, deque
import itertools

from utils.nms_wrapper import nms_detections
from utils.log import logger 
from utils.kalman_filter import KalmanFilter 
from tracker import matching 
from tracker.DDTracker import DDTracker 
from tracker.basetrack import BaseTrack, TrackState 
from models.classification.classifier import PatchClassifier


class STrack(BaseTrack):
    def __init__(self, tlwh, score, from_det=True):
        # 新创建的轨迹需要连续两帧出现才能激活
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None 
        self.mean, self.covariance = None, None 
        self.is_activated = False  # 未激活状态

        self.score = score 
        self.from_det = from_det
        self.tracklet_len = 0  # 当前轨迹长度
        self.time_by_tracking = 0  # 完全依靠跟踪器连续跟踪的长度
        
        self.tracker = DDTracker()  # 每条轨迹有一个单独的跟踪器

    def activate(self, kalman_filter, frame_id, image):
        # 新轨迹的起始
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        del self._tlwh 

        self.time_since_update = 0
        self.time_by_tracking = 0  # 依据检测来确定轨迹的起点,因此为0
        self.tracklet_len = 0
        self.state = TrackState.Tracked  # 跟踪态
        self.frame_id = frame_id 
        self.start_frame = frame_id 
        # 初始化跟踪器
        self.tracker.init(image, self.tlwh)

    def predict(self):
        # 每条轨迹进行一定的预测
        if self.time_since_update > 0: # 表示lost状态
            self.tracklet_len = 0
        
        self.time_since_update += 1
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.tracker.update(self.tlwh)

    def self_tracking(self, image):
        # 该轨迹自带检测器在新图像上的预测
        tlwh = self.tracker.predict(image) if self.tracker else self.tlwh 
        return tlwh 

    def tracking_distance(self, detections, image):
        # 由跟踪器计算匹配程度
        if len(detections) == 0:
            return np.empty((1,0), dtype=np.float)
        cost_vector = np.zeros(len(detections))
        cost_vector = matching.dd_gate_cost_matrix(self.kalman_filter, cost_vector, self, detections)

        possible_idx = cost_vector < np.inf 
        tlwhs = np.asarray([d.tlwh for d in detections])
        cost_vector[possible_idx] = 1 - self.tracker.matching(image, tlwhs)
        return cost_vector 

    def update(self, new_track, frame_id, image):
        # 由匹配跟踪到的detection 更新当前轨迹和跟踪器
        self.frame_id = frame_id 
        self.time_since_update = 0
        if new_track.from_det:
            self.time_by_tracking = 0
        else:
            self.time_by_tracking += 1
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh 
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.tracker.update(new_track.tlwh, image)  # 更新下一帧匹配的位置,以及kernel

    def re_activate(self, new_track, frame_id, image):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracker.update(new_track.tlwh, image)  
    
    @property
    @jit
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    @jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)
    
    def tracklet_score(self):
        # score = (1 - np.exp(-0.6 * self.hit_streak)) * np.exp(-0.03 * self.time_by_tracking)

        score = max(0, 1 - np.log(1 + 0.05 * self.time_by_tracking)) * (self.tracklet_len - self.time_by_tracking > 2)
        # score = max(0, 1 - np.log(1 + 0.05 * self.n_tracking)) * (1 - np.exp(-0.6 * self.hit_streak))
        return score
    
    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class OnlineTracker(object):
    def __init__(self, min_det_score=0.4, min_cls_dist=0.5, max_time_lost=30,
        use_tracking=True):
        self.min_det_score = min_det_score  # 检测置信度阈值
        self.min_cls_dist = min_cls_dist    # 模板匹配的阈值
        self.max_time_lost = max_time_lost
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.use_tracking = use_tracking
        self.classifier = PatchClassifier()

        self.frame_id = 0

    def update(self, image, tlwhs, det_scores=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        """ step 1. prediction, 卡尔曼滤波粗略估计目标新的位置 """
        for strack in itertools.chain(self.tracked_stracks, self.lost_stracks):
            strack.predict()
        
        """ step 2. scoring and selection 对包含预测的候选框nms处理 """
        if det_scores is None:
            det_scores = np.ones(len(tlwhs), dtype=float)
        detections = [STrack(tlwh, score, from_det=True) for tlwh, score in zip(tlwhs, det_scores)]

        if self.classifier is None:
            pred_dets = []
        else:
            self.classifier.update(image)

            n_dets = len(tlwhs)
            if self.use_tracking:
                tracks = [STrack(t.self_tracking(image), t.tracklet_score(), from_det=False)
                          for t in itertools.chain(self.tracked_stracks, self.lost_stracks) if t.is_activated]
                detections.extend(tracks)
            rois = np.asarray([d.tlbr for d in detections], dtype=np.float32)

            cls_scores = self.classifier.predict(rois)
            scores = np.asarray([d.score for d in detections], dtype=np.float)
            scores[0:n_dets] = 1.
            scores = scores * cls_scores
            # nms
            if len(detections) > 0:
                keep = nms_detections(rois, scores.reshape(-1), nms_thresh=0.3)
                mask = np.zeros(len(rois), dtype=np.bool)
                mask[keep] = True
                keep = np.where(mask & (scores >= self.min_det_score))[0]
                detections = [detections[i] for i in keep]
                scores = scores[keep]
                for d, score in zip(detections, scores):
                    d.score = score
            pred_dets = [d for d in detections if not d.from_det]
            detections = [d for d in detections if d.from_det]
        
        """step 3. association for tracked 对tracked轨迹进行关联跟踪"""
        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        dists = []
        for track in self.tracked_stracks:
            dists.append(track.tracking_distance(detections))
        dists = np.stack(dists, axis=0)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.min_cls_dist)
        for itracked, idet in matches:
            tracked_stracks[itracked].update(detections[idet], self.frame_id, image)
        
        # matching for missing targets
        detections = [detections[i] for i in u_detection]
        dists = []
        for track in self.lost_stracks:
            dists.append(track.tracking_distance(detections))
        dists = np.stack(dists, axis=0)
        matches, _, u_detection = matching.linear_assignment(dists, thresh=self.min_cls_dist)
        for ilost, idet in matches:
            self.lost_stracks[ilost].re_activate(detections[idet], self.frame_id, image)
            refind_stracks.append(self.lost_stracks[ilost])
        
        # remaining tracked
        # tracked
        len_det = len(u_detection)
        detections = [detections[i] for i in u_detection] + pred_dets
        r_tracked_stracks = [tracked_stracks[i] for i in u_track]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            r_tracked_stracks[itracked].update(detections[idet], self.frame_id, image)
        for it in u_track:
            track = r_tracked_stracks[it]
            track.mark_lost()
            lost_stracks.append(track)

        # unconfirmed
        detections = [detections[i] for i in u_detection if i < len_det]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, image)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ step 4. init new stracks 初始化新的轨迹 """
        for inew in u_detection:
            track = detections[inew]
            if not track.from_det or track.score < 0.6:
                continue 
            track.activate(self.kalman_filter, self.frame_id, image)
            activated_stracks.append(track)
        
        """ step 5. update states 更新跟踪器状态 """
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.tracked_stracks.extend(activated_stracks)
        self.tracked_stracks.extend(refind_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # output_stracks = self.tracked_stracks + self.lost_stracks

        # get scores of lost tracks
        rois = np.asarray([t.tlbr for t in self.lost_stracks], dtype=np.float32)
        lost_cls_scores = self.classifier.predict(rois)
        out_lost_stracks = [t for i, t in enumerate(self.lost_stracks)
                            if lost_cls_scores[i] > 0.3 and self.frame_id - t.end_frame <= 4]
        output_tracked_stracks = [track for track in self.tracked_stracks if track.is_activated]

        output_stracks = output_tracked_stracks + out_lost_stracks

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_stracks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks

        