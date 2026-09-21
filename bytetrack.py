# -*- coding: utf-8 -*-
"""ByteTrack 다중 객체 추적 — numpy/scipy만 쓰는 독립 구현.

torch 없는 백엔드(trt_backend.py)에서 ultralytics 트래커 대신 쓴다. ultralytics 코드를
복사하지 않고 논문(ByteTrack, Zhang et al. 2022) 알고리즘을 그대로 구현했다:

  1. 칼만 필터로 기존 트랙 위치 예측
  2. 1차 매칭: 높은 점수 탐지 <-> 추적중+잃어버린 트랙 (IoU 거리, 점수 융합)
  3. 2차 매칭: 낮은 점수 탐지 <-> 1차에서 못 맞춘 '추적중' 트랙 (가림 중 저점수 탐지 회수)
  4. 남은 고점수 탐지 <-> 미확정 트랙, 그래도 남으면 새 트랙
  5. 못 맞춘 트랙은 lost, track_buffer 프레임 지나면 제거

기본값은 ultralytics bytetrack.yaml과 같다 (high 0.25 / low 0.1 / new 0.25 / buffer 30 / match 0.8).
호출자(dump_monitor_jetson.py)는 PERSON_CONF 이상만 넘기므로 실제로는 1차 매칭이 대부분이다.

사용:
    tr = ByteTrack(frame_rate=15)
    tracks = tr.update(dets)        # dets: (N, 5) [x1,y1,x2,y2,score] -> [(track_id, [x1,y1,x2,y2], score)]
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- Kalman (x, y, a, h)
class KalmanFilter:
    """상태 8차원: 중심 x, y, 종횡비 a(w/h), 높이 h + 각 속도. 등속 모델."""

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)            # x += vx
        self.H = np.eye(4, 8)
        self.w_pos, self.w_vel = 1 / 20, 1 / 160

    def initiate(self, z):
        mean = np.zeros(8)
        mean[:4] = z
        h = z[3]
        std = [2 * self.w_pos * h, 2 * self.w_pos * h, 1e-2, 2 * self.w_pos * h,
               10 * self.w_vel * h, 10 * self.w_vel * h, 1e-5, 10 * self.w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.w_pos * h, self.w_pos * h, 1e-2, self.w_pos * h,
               self.w_vel * h, self.w_vel * h, 1e-5, self.w_vel * h]
        Q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.w_pos * h, self.w_pos * h, 1e-1, self.w_pos * h]))
        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ mean
        mean = mean + K @ y
        cov = (np.eye(8) - K @ self.H) @ cov
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=float)


def xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost, thresh):
    """헝가리안 매칭. cost > thresh 인 쌍은 매칭에서 제외. (matches, unmatched_rows, unmatched_cols)"""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = cost.copy()
    c[c > thresh] = thresh + 1e-3
    rows, cols = linear_sum_assignment(c)
    matches = [(r, c_) for r, c_ in zip(rows, cols) if cost[r, c_] <= thresh]
    mr = set(r for r, _ in matches); mc = set(c_ for _, c_ in matches)
    return matches, [r for r in range(cost.shape[0]) if r not in mr], \
        [c_ for c_ in range(cost.shape[1]) if c_ not in mc]


# ---------------------------------------------------------------- Track
NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class Track:
    _next_id = 0
    kf = KalmanFilter()

    def __init__(self, box, score):
        self.box_det = list(box)
        self.score = score
        self.mean, self.cov = None, None
        self.state = NEW
        self.is_activated = False
        self.id = 0
        self.frame_id = 0
        self.start_frame = 0

    @classmethod
    def next_id(cls):
        cls._next_id += 1
        return cls._next_id

    @classmethod
    def reset_ids(cls):
        cls._next_id = 0

    @property
    def box(self):
        return xyah_to_xyxy(self.mean) if self.mean is not None else self.box_det

    def predict(self):
        if self.state != TRACKED:
            self.mean[7] = 0                    # 잃어버린 트랙은 높이 속도 고정
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

    def activate(self, frame_id):
        self.id = self.next_id()
        self.mean, self.cov = self.kf.initiate(xyxy_to_xyah(self.box_det))
        self.state = TRACKED
        self.is_activated = frame_id == 1        # 첫 프레임 외엔 다음 매칭에서 확정
        self.frame_id = self.start_frame = frame_id

    def update(self, det, frame_id, reactivate=False):
        self.box_det, self.score = list(det.box_det), det.score
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(self.box_det))
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id


class ByteTrack:
    def __init__(self, track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                 track_buffer=30, match_thresh=0.8, fuse_score=True, frame_rate=30):
        self.high, self.low, self.new_thresh = track_high_thresh, track_low_thresh, new_track_thresh
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.max_lost = int(frame_rate / 30.0 * track_buffer)
        self.tracked, self.lost, self.removed = [], [], []
        self.frame_id = 0
        Track.reset_ids()

    def reset(self):
        self.__init__(self.high, self.low, self.new_thresh, self.max_lost, self.match_thresh,
                      self.fuse_score, 30)

    # ------------------------------------------------------------
    def _dist(self, tracks, dets):
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        d = 1 - iou_matrix([t.box for t in tracks], [x.box_det for x in dets])
        if self.fuse_score:                      # 점수 높은 탐지를 우대
            d = 1 - (1 - d) * np.array([x.score for x in dets])[None, :]
        return d

    def update(self, dets):
        """dets: (N,5) [x1,y1,x2,y2,score] -> [(id, [x1,y1,x2,y2], score)] (확정 트랙만)"""
        self.frame_id += 1
        dets = np.asarray(dets, float).reshape(-1, 5)
        high = [Track(b[:4], b[4]) for b in dets if b[4] >= self.high]
        low = [Track(b[:4], b[4]) for b in dets if self.low <= b[4] < self.high]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + self.lost
        for t in pool:
            t.predict()

        # 1차: 고점수 탐지 <-> 추적중 + lost
        matches, u_track, u_det = assign(self._dist(pool, high), self.match_thresh)
        activated, refound, lost_now = [], [], []
        for r, c in matches:
            t = pool[r]
            was_lost = t.state == LOST
            t.update(high[c], self.frame_id)
            (refound if was_lost else activated).append(t)

        # 2차: 저점수 탐지 <-> 1차에서 못 맞춘 '추적중' 트랙
        r_tracks = [pool[i] for i in u_track if pool[i].state == TRACKED]
        d2 = 1 - iou_matrix([t.box for t in r_tracks], [x.box_det for x in low]) if r_tracks and low \
            else np.zeros((len(r_tracks), len(low)))
        m2, u_track2, _ = assign(d2, 0.5)
        for r, c in m2:
            r_tracks[r].update(low[c], self.frame_id)
            activated.append(r_tracks[r])
        for i in u_track2:
            t = r_tracks[i]
            t.state = LOST
            lost_now.append(t)

        # 미확정 트랙 <-> 남은 고점수 탐지
        rest = [high[i] for i in u_det]
        m3, u_unconf, u_det3 = assign(self._dist(unconfirmed, rest), 0.7)
        for r, c in m3:
            unconfirmed[r].update(rest[c], self.frame_id)
            activated.append(unconfirmed[r])
        for i in u_unconf:
            unconfirmed[i].state = REMOVED
            self.removed.append(unconfirmed[i])

        # 새 트랙
        for i in u_det3:
            t = rest[i]
            if t.score >= self.new_thresh:
                t.activate(self.frame_id)
                activated.append(t)

        # lost 만료
        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_lost:
                t.state = REMOVED
                self.removed.append(t)

        # 목록 재구성
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _merge(self.tracked, activated)
        self.tracked = _merge(self.tracked, refound)
        self.lost = [t for t in self.lost if t.state == LOST and t not in self.tracked]
        self.lost = _merge(self.lost, lost_now)
        self.lost = [t for t in self.lost if t.state == LOST]
        self.tracked, self.lost = _drop_dups(self.tracked, self.lost)
        return [(t.id, [float(v) for v in t.box], float(t.score))
                for t in self.tracked if t.is_activated]


def _merge(a, b):
    seen = {id(t) for t in a}
    return a + [t for t in b if id(t) not in seen]


def _drop_dups(tracked, lost):
    """추적중/lost 트랙이 같은 자리를 가리키면(IoU>0.85) 더 오래된 쪽을 남긴다."""
    if not tracked or not lost:
        return tracked, lost
    iou = iou_matrix([t.box for t in tracked], [t.box for t in lost])
    dup_t, dup_l = set(), set()
    for i, j in zip(*np.where(iou > 0.85)):
        age_t = tracked[i].frame_id - tracked[i].start_frame
        age_l = lost[j].frame_id - lost[j].start_frame
        (dup_l if age_t > age_l else dup_t).add(j if age_t > age_l else i)
    return [t for i, t in enumerate(tracked) if i not in dup_t], \
        [t for j, t in enumerate(lost) if j not in dup_l]


__all__ = ["ByteTrack", "KalmanFilter", "iou_matrix"]
