#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""사건 전후 클립 (media.clip) — jetson_handoff.md §3.3 / jetson_data.md §4.1.

상시 녹화 없이(--no-save-video) 사건 앞뒤 N초만 mp4로 만든다.

- push(frame): 주석 프레임(모자이크 적용본)을 720p JPEG로 링버퍼에 넣는다.
  원본 프레임을 그대로 쥐면 10초 x 10fps x 2.7MB = 270MB라 JPEG(~100KB)로 압축해 둔다.
- start(event_id, path): 링버퍼의 직전 pre_s초 + 이후 post_s초 프레임을 모아
  백그라운드 스레드에서 H.264 mp4로 인코딩하고, 끝나면 on_done(event_id, path).
- 인코딩은 srt_sender와 같은 gst-launch x264(4:2:0 constrained-baseline) — 대시보드
  <video>가 PC·모바일 브라우저 모두에서 재생된다. cv2 mp4v는 브라우저가 못 푼다.
  Orin Nano에는 하드웨어 인코더가 없으므로 nice 10으로 돌려 감시 루프를 방해하지 않는다.
- 사건 직후 바로 다음 사건이 나도 프레임(JPEG bytes)은 공유하므로 메모리가 겹쳐 늘지 않는다.
"""
import os
import queue
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

ENC_PIPELINE = (
    "fdsrc fd=0 ! "
    "rawvideoparse format=bgr width={w} height={h} framerate={fps}/1 ! "
    "videoconvert ! video/x-raw,format=I420 ! "
    "x264enc bitrate={kbps} speed-preset=veryfast key-int-max={fps} bframes=0 ! "
    "video/x-h264,profile=constrained-baseline ! "
    "h264parse ! mp4mux faststart=true ! filesink location={path}"
)


def encode_mp4(frames, path, fps, size, kbps=1500, timeout=120.0):
    """JPEG bytes 목록 -> H.264 mp4. 성공하면 True."""
    w, h = size
    tmp = f"{path}.part"
    cmd = ["gst-launch-1.0", "-q", "-e"] + ENC_PIPELINE.format(
        w=w, h=h, fps=max(1, int(fps)), kbps=int(kbps), path=tmp).split()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, preexec_fn=lambda: os.nice(10))
    try:
        for jpg in frames:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (w, h):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        rc = proc.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
        proc.wait()
        rc = -1
    if rc == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)
        return True
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return False


class _Job:
    __slots__ = ("event_id", "path", "t_end", "frames")

    def __init__(self, event_id, path, t_end, frames):
        self.event_id, self.path, self.t_end, self.frames = event_id, path, t_end, frames


class EventClipper:
    def __init__(self, on_done, pre_s=5.0, post_s=5.0, size=(1280, 720), max_fps=10.0,
                 quality=80, kbps=1500, verbose=True):
        self.on_done = on_done
        self.pre_s, self.post_s = float(pre_s), float(post_s)
        self.size, self.quality, self.kbps = tuple(size), int(quality), int(kbps)
        # 간격 하한을 1/fps 그대로 두면 프레임 지터(99ms 도착)마다 버려 실제로 절반만 남는다 -> 여유 20%
        self.min_dt = 0.8 / max_fps if max_fps > 0 else 0.0
        self.verbose = verbose
        self.ring = deque()                       # (t, jpeg bytes)
        self.jobs = []
        self.made = self.failed = 0
        self._t_last = None
        self._q = queue.Queue()
        self._th = threading.Thread(target=self._loop, name="clip-encoder", daemon=True)
        self._th.start()

    def push(self, frame, t=None):
        t = time.monotonic() if t is None else t
        if self._t_last is not None and t - self._t_last < self.min_dt:
            return
        self._t_last = t
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        item = (t, buf.tobytes())
        self.ring.append(item)
        while self.ring and t - self.ring[0][0] > self.pre_s:
            self.ring.popleft()
        for job in list(self.jobs):
            job.frames.append(item)
            if t >= job.t_end:
                self.jobs.remove(job)
                self._q.put(job)

    def start(self, event_id, path, t=None):
        t = time.monotonic() if t is None else t
        self.jobs.append(_Job(event_id, str(path), t + self.post_s, list(self.ring)))

    def pending(self):
        return len(self.jobs) + self._q.qsize()

    def _loop(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            fr = job.frames
            if len(fr) < 2:
                self.failed += 1
                continue
            span = fr[-1][0] - fr[0][0]
            fps = round((len(fr) - 1) / span) if span > 0 else 10
            t0 = time.monotonic()
            ok = encode_mp4([j for _, j in fr], job.path, fps, self.size, self.kbps)
            if not ok:
                self.failed += 1
                if self.verbose:
                    print(f"[클립] 인코딩 실패 — {os.path.basename(job.path)}", flush=True)
                continue
            self.made += 1
            if self.verbose:
                print(f"[클립] {os.path.basename(job.path)} {span:.0f}초 {len(fr)}프레임 "
                      f"{os.path.getsize(job.path) // 1024}KiB (인코딩 "
                      f"{time.monotonic() - t0:.1f}s)", flush=True)
            try:
                self.on_done(job.event_id, job.path)
            except Exception as e:                   # noqa: BLE001
                print(f"[클립] 전달 실패({type(e).__name__}: {e})", flush=True)

    def close(self, timeout=30.0):
        """진행 중인 클립은 지금까지 모은 프레임으로 마무리하고 인코딩이 끝날 때까지 기다린다."""
        for job in self.jobs:
            self._q.put(job)
        self.jobs = []
        self._q.put(None)
        self._th.join(timeout)

    def summary(self):
        return f"클립: {self.made}개 생성" + (f" · 실패 {self.failed}개" if self.failed else "")
