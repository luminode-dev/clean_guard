# -*- coding: utf-8 -*-
"""주석 영상 기록 — 상시 운영에서 디스크를 채우지 않는 회전 기록기.

단일 annotated.mp4로 계속 쓰면 24/7 운영에서 디스크가 반드시 찬다
(1080p mp4v 기준 시간당 수 GB). 이 모듈은 세 가지로 그것을 막는다.

- 분할 : segment_min 분마다 새 파일(annotated_YYYYmmdd_HHMMSS.mp4)로 넘어간다
- 보존 : 최신 keep개만 남기고 오래된 세그먼트를 지운다
- 가드 : 남은 디스크가 min_free_mb 아래로 내려가면 녹화를 스스로 멈춘다.
         이벤트 로그(events.jsonl)와 증거 스냅샷이 영상보다 중요하므로,
         디스크를 비워야 할 때 먼저 포기하는 쪽은 영상이다.

가드가 걸린 뒤 보존 정리로 공간이 회복되면 다음 세그먼트 경계에서 녹화를 재개한다.
"""
import shutil
import time
from pathlib import Path

import cv2

PREFIX = "annotated_"
SUFFIX = ".mp4"
FOURCC = "mp4v"
FREE_CHECK_FRAMES = 300      # 세그먼트 중간에도 이 주기로 여유 공간을 다시 본다


def free_mb(path):
    """path가 속한 파일시스템의 여유 공간(MB). 실패하면 무한대로 봐서 가드를 비활성."""
    try:
        return shutil.disk_usage(str(path)).free / (1024 * 1024)
    except OSError:
        return float("inf")


class RotatingRecorder:
    """시간 단위로 분할하고 오래된 것을 지우는 VideoWriter 래퍼.

    enabled=False면 모든 호출이 no-op이라 호출부에 분기를 두지 않아도 된다.
    """

    def __init__(self, out_dir, fps, size, segment_min=60.0, keep=24,
                 min_free_mb=1000.0, enabled=True):
        self.dir = Path(out_dir)
        self.fps = float(fps) if fps and fps > 0 else 10.0
        self.size = (int(size[0]), int(size[1]))
        self.segment_sec = max(1.0, float(segment_min) * 60.0)
        self.keep = max(1, int(keep))
        self.min_free_mb = float(min_free_mb)
        self.enabled = bool(enabled) and self.size[0] > 0 and self.size[1] > 0
        self._w = None
        self._started = 0.0
        self._frames_in_seg = 0
        self.n_segments = 0
        self.n_pruned = 0
        self.n_frames = 0
        self.paused = False          # 디스크 가드로 멈춘 상태
        self._warned_pause = False

    # ------------------------------------------------------------ 내부
    def _path_for_now(self):
        return self.dir / f"{PREFIX}{time.strftime('%Y%m%d_%H%M%S')}{SUFFIX}"

    def _segments(self):
        """오래된 것부터 정렬된 세그먼트 목록 (파일명이 시각순이라 이름 정렬로 충분)."""
        return sorted(self.dir.glob(f"{PREFIX}*{SUFFIX}"))

    def _prune(self):
        segs = self._segments()
        for p in segs[:max(0, len(segs) - self.keep)]:
            try:
                p.unlink()
                self.n_pruned += 1
            except OSError:
                pass

    def current_path(self):
        """지금 쓰고 있는 세그먼트 경로 (없으면 None). 회수 대상에서 제외하기 위함."""
        if self._w is None:
            return None
        segs = self._segments()
        return segs[-1] if segs else None

    def _close_writer(self):
        if self._w is not None:
            try:
                self._w.release()
            except Exception:                                    # noqa: BLE001
                pass
            self._w = None

    def _open_writer(self):
        """새 세그먼트 시작. 공간이 모자라면 열지 않고 paused로 둔다."""
        self._close_writer()
        self._prune()                      # 열기 전에 먼저 비워 공간을 확보한다
        if free_mb(self.dir) < self.min_free_mb:
            self.paused = True
            if not self._warned_pause:
                print(f"[녹화 경고] 여유 공간 {free_mb(self.dir):.0f}MB < "
                      f"{self.min_free_mb:.0f}MB -> 영상 녹화 중단 "
                      f"(이벤트 로그·스냅샷은 계속 기록)")
                self._warned_pause = True
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path_for_now()
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*FOURCC),
                            self.fps, self.size)
        if not w.isOpened():
            print(f"[녹화 경고] 세그먼트를 열 수 없음: {p.name} -> 녹화 비활성")
            self.enabled = False
            return False
        self._w = w
        self._started = time.monotonic()
        self._frames_in_seg = 0
        self.n_segments += 1
        if self.paused:
            print(f"[녹화] 공간 회복 -> 녹화 재개 ({p.name})")
        self.paused = False
        self._warned_pause = False
        return True

    # ------------------------------------------------------------ 공개
    def write(self, frame):
        """프레임 1장 기록. 세그먼트 경계면 회전하고, 공간이 없으면 조용히 건너뛴다."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if self._w is None or now - self._started >= self.segment_sec:
            # paused 상태에서도 세그먼트 주기마다 재개를 시도한다
            if self._w is None and self.paused and now - self._started < self.segment_sec:
                return False
            if not self._open_writer():
                self._started = now        # 다음 주기까지 재시도 보류
                return False
        self._frames_in_seg += 1
        if self._frames_in_seg % FREE_CHECK_FRAMES == 0 and \
                free_mb(self.dir) < self.min_free_mb:
            # 세그먼트 도중에 공간이 말랐다 -> 지금 닫고 멈춘다
            self._close_writer()
            self._prune()
            self.paused = True
            self._started = now
            if not self._warned_pause:
                print(f"[녹화 경고] 기록 중 여유 공간 부족 -> 영상 녹화 중단")
                self._warned_pause = True
            return False
        self._w.write(frame)
        self.n_frames += 1
        return True

    def close(self):
        self._close_writer()
        self._prune()

    def summary(self):
        segs = self._segments()
        mb = sum(p.stat().st_size for p in segs if p.exists()) / (1024 * 1024)
        return (f"녹화: 세그먼트 {self.n_segments}개 생성 · {self.n_frames}프레임 · "
                f"보관 {len(segs)}개({mb:.0f}MB) · 삭제 {self.n_pruned}개"
                + (" · 공간부족으로 중단됨" if self.paused else ""))


class OutputSpace:
    """출력 폴더의 디스크를 감시하고, 모자라면 오래된 것부터 지운다 (FIFO).

    보존 개수(--video-keep)만으로는 부족하다. 해상도·비트레이트에 따라 세그먼트
    크기가 크게 달라지므로, 개수가 아니라 **남은 공간**을 기준으로도 회수해야 한다.

    purge_order
      "video-first" (기본) : 주석 영상을 오래된 것부터 전부 지우고, 그래도 모자라면
                             이벤트 스냅샷을 오래된 것부터. 영상보다 증거가 중요하므로
                             먼저 버리는 쪽은 영상이다.
      "oldest-first"       : 종류를 가리지 않고 순수하게 오래된 파일부터 (엄격한 FIFO).

    events.jsonl은 어떤 경우에도 지우지 않는다 — 용량이 작고, 그것이 곧 이벤트 기록이다.

    공간과 별개로 expire()는 보존 기간이 지난 증거를 지운다 (handoff §3.3 기준안:
    모자이크본 90일, 원본(*_raw.*) 30일). 0이면 기간 삭제 안 함.
    """

    VIDEO_GLOB = "annotated_*.mp4"
    EVENT_GLOB = "event_*.*"            # 스냅샷 jpg + 클립 mp4 + 방송 음성 wav (events.jsonl은 안 걸림)

    def __init__(self, out_dir, min_free_mb=1000.0, headroom_mb=200.0,
                 purge_order="video-first", keep_days=90.0, raw_keep_days=30.0):
        self.dir = Path(out_dir)
        self.min_free_mb = float(min_free_mb)
        # 임계선에 딱 맞춰 멈추면 곧바로 다시 걸리므로 조금 더 확보하고 끝낸다
        self.target_mb = self.min_free_mb + max(0.0, float(headroom_mb))
        self.purge_order = purge_order
        self.n_deleted = 0
        self.mb_freed = 0.0
        self.n_starved = 0          # 지울 게 없어 회수에 실패한 횟수
        self.keep_days, self.raw_keep_days = float(keep_days), float(raw_keep_days)
        self.n_expired = 0

    def _by_age(self, glob, exclude):
        """glob에 걸리는 파일을 오래된 순으로. (mtime, path) 목록."""
        out = []
        for p in self.dir.glob(glob):
            if exclude is not None and p == exclude:
                continue
            try:
                out.append((p.stat().st_mtime, p))
            except OSError:
                pass
        return sorted(out)

    def _candidates(self, exclude=None):
        """삭제 후보를 지울 순서대로. exclude는 현재 기록 중인 파일."""
        videos = self._by_age(self.VIDEO_GLOB, exclude)
        events = self._by_age(self.EVENT_GLOB, exclude)
        if self.purge_order == "oldest-first":
            return [p for _, p in sorted(videos + events)]
        return [p for _, p in videos] + [p for _, p in events]

    def free_mb(self):
        return free_mb(self.dir)

    def reclaim(self, exclude=None):
        """여유가 임계 미만이면 목표치까지 오래된 것부터 삭제. (삭제수, 회수MB) 반환."""
        if self.free_mb() >= self.min_free_mb:
            return 0, 0.0
        deleted, freed = 0, 0.0
        for p in self._candidates(exclude=exclude):
            if self.free_mb() >= self.target_mb:
                break
            try:
                mb = p.stat().st_size / (1024 * 1024)
                p.unlink()
            except OSError:
                continue
            deleted += 1
            freed += mb
        if deleted:
            self.n_deleted += deleted
            self.mb_freed += freed
            print(f"[디스크] 여유 부족 -> 오래된 파일 {deleted}개 삭제 "
                  f"({freed:.0f}MB 회수, 현재 {self.free_mb():.0f}MB)")
        elif self.free_mb() < self.min_free_mb:
            self.n_starved += 1
            if self.n_starved == 1:
                print(f"[디스크 경고] 여유 {self.free_mb():.0f}MB인데 지울 출력물이 없다. "
                      f"다른 곳을 비우거나 --min-free-mb를 낮출 것")
        return deleted, freed

    def expire(self, now=None, exclude=None):
        """보존 기간이 지난 증거 파일(스냅샷·클립·방송 음성)을 지운다. 삭제 수 반환."""
        now = time.time() if now is None else now
        n = 0
        for mtime, p in self._by_age(self.EVENT_GLOB, exclude):
            days = self.raw_keep_days if "_raw." in p.name else self.keep_days
            if days <= 0 or now - mtime < days * 86400:
                continue
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        if n:
            self.n_expired += n
            print(f"[디스크] 보존 기간 경과 증거 {n}개 삭제 "
                  f"(모자이크 {self.keep_days:g}일 · 원본 {self.raw_keep_days:g}일)", flush=True)
        return n

    def summary(self):
        s = (f"디스크: 오래된 출력물 {self.n_deleted}개 삭제({self.mb_freed:.0f}MB 회수) "
             f"· 현재 여유 {self.free_mb():.0f}MB")
        if self.n_expired:
            s += f" · 보존 기간 경과 {self.n_expired}개 삭제"
        if self.n_starved:
            s += f" · 회수 실패 {self.n_starved}회"
        return s


__all__ = ["RotatingRecorder", "OutputSpace", "free_mb"]
