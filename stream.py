# -*- coding: utf-8 -*-
"""입력 스트림 — RTSP 상시 운영을 위한 자동 재연결 + 최신 프레임 유지.

파일 입력과 라이브 입력은 요구가 정반대다.

- 파일  : 모든 프레임을 순서대로 빠짐없이 봐야 한다. 끝(EOF)나면 정상 종료.
- 라이브: 끊기면 다시 붙어야 하고, 처리가 스트림보다 느리면 **밀린 프레임을 버려야 한다**.
          버리지 않으면 OpenCV 내부 버퍼에 프레임이 쌓여 처리 시점이 실시간에서 계속
          뒤처진다. 몇십 분 뒤에는 "방금 버린 사람"에게 몇 분 전 사건을 방송하게 되어
          경고 방송이라는 기능 자체가 무의미해진다.

FrameSource가 둘을 같은 인터페이스로 감싼다.

    src = FrameSource(args.source).open()
    while True:
        ok, frame = src.read()
        if not ok:
            break            # 파일 EOF 또는 재연결 포기
    src.close()

라이브 소스는 별도 읽기 스레드가 계속 최신 프레임만 들고 있고, read()는 아직 안 준
가장 최근 프레임 하나를 돌려준다. 그 사이 도착한 프레임은 버려지며 개수만 센다
(cap.read()는 GIL을 놓으므로 추론 스레드를 막지 않는다).
"""
import os
import threading
import time

import cv2

LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "rtmps://",
                "http://", "https://", "udp://", "tcp://")

# RTSP가 끊겼을 때 cap.read()가 무한정 매달리지 않도록 FFMPEG에 타임아웃을 건다.
# OpenCV는 VideoCapture 생성 시점에 이 환경변수를 읽으므로 import 시 미리 넣어 둔다.
# (사용자가 이미 지정했으면 존중한다)
DEFAULT_FFMPEG_OPTS = "rtsp_transport;tcp|stimeout;5000000"   # TCP, 5초
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", DEFAULT_FFMPEG_OPTS)


def is_live(source):
    """웹캠 번호 또는 네트워크 스트림이면 True (= 재연결·프레임 드롭 대상)."""
    s = str(source)
    return s.isdigit() or s.lower().startswith(LIVE_SCHEMES)


def _norm(source):
    """웹캠 번호는 int로, 나머지는 문자열 그대로."""
    s = str(source)
    return int(s) if s.isdigit() else s


class FrameSource:
    """재연결·프레임드롭을 처리하는 VideoCapture 래퍼.

    reconnect_max_retries=0 이면 무한 재시도 (상시 감시의 기본값).
    startup_timeout 안에 첫 연결이 안 되면 open()이 RuntimeError를 낸다
    — URL 오타 같은 설정 실수를 몇 초 안에 알려주기 위해서다.
    """

    def __init__(self, source, reconnect_delay=2.0, reconnect_max_delay=30.0,
                 reconnect_max_retries=0, startup_timeout=30.0, drop_stale=True):
        self.source = _norm(source)
        self.live = is_live(source)
        self.delay0 = max(0.1, float(reconnect_delay))
        self.delay_max = max(self.delay0, float(reconnect_max_delay))
        self.max_retries = int(reconnect_max_retries)
        self.startup_timeout = float(startup_timeout)
        self.drop_stale = bool(drop_stale) and self.live
        self.cap = None
        self.fps = 10.0
        self.width = 0
        self.height = 0
        self.n_reconnect = 0
        self.n_dropped = 0
        self.n_read = 0
        self._thread = None
        self._cv = threading.Condition()
        self._latest = None
        self._seq = 0
        self._last_seq = 0
        self._done = False
        self._stop = threading.Event()

    # ------------------------------------------------------------ 연결
    def _open_cap(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return None
        if self.live:
            # 백엔드가 지원하면 내부 버퍼를 1로 줄인다 (지연 누적 완화).
            # FFMPEG 백엔드는 무시하는 경우가 많아 읽기 스레드 드롭이 본 방어선이다.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:                                    # noqa: BLE001
                pass
        return cap

    def _read_meta(self):
        f = self.cap.get(cv2.CAP_PROP_FPS)
        # RTSP는 0이나 비정상값(90000 등)을 주는 경우가 있다
        self.fps = f if f and 0 < f <= 120 else 10.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def open(self):
        deadline = time.monotonic() + self.startup_timeout
        delay = self.delay0
        while True:
            self.cap = self._open_cap()
            if self.cap is not None:
                break
            if not self.live or time.monotonic() >= deadline:
                raise RuntimeError(f"입력을 열 수 없음: {self.source}")
            print(f"[스트림] 첫 연결 실패 -> {delay:.0f}s 후 재시도")
            time.sleep(delay)
            delay = min(delay * 2, self.delay_max)
        self._read_meta()
        if self.drop_stale:
            self._thread = threading.Thread(target=self._reader, name="stream-reader",
                                            daemon=True)
            self._thread.start()
        return self

    def _reconnect(self):
        """끊긴 연결을 다시 세운다. 성공하면 True, 포기하면 False."""
        tries, delay = 0, self.delay0
        while not self._stop.is_set():
            tries += 1
            if self.max_retries and tries > self.max_retries:
                print(f"[스트림] 재연결 {self.max_retries}회 실패 -> 포기")
                return False
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:                                    # noqa: BLE001
                pass
            self.cap = None
            if self._stop.wait(delay):
                return False
            cap = self._open_cap()
            if cap is not None:
                self.cap = cap
                self.n_reconnect += 1
                self._read_meta()
                print(f"[스트림] 재연결 성공 (누적 {self.n_reconnect}회, {tries}번째 시도)")
                return True
            print(f"[스트림] 재연결 실패 {tries}회 -> {delay:.0f}s 후 재시도")
            delay = min(delay * 2, self.delay_max)
        return False

    # ------------------------------------------------------------ 읽기
    def _reader(self):
        """라이브 전용 읽기 스레드. 항상 최신 프레임 한 장만 들고 있는다."""
        while not self._stop.is_set():
            cap = self.cap
            if cap is None:
                if not self._reconnect():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                print("[스트림] 프레임 수신 끊김 -> 재연결 시도")
                if not self._reconnect():
                    break
                continue
            with self._cv:
                if self._latest is not None and self._seq != self._last_seq:
                    self.n_dropped += 1      # 소비되기 전에 덮어쓴다 = 버린 프레임
                self._latest = frame
                self._seq += 1
                self._cv.notify()
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def read(self, timeout=1.0):
        """(ok, frame). ok=False는 '더 볼 것이 없음'(파일 EOF / 재연결 포기 / close)."""
        if not self.drop_stale:
            # 파일: 모든 프레임을 순서대로. EOF면 종료.
            if self.cap is None:
                return False, None
            ok, frame = self.cap.read()
            if ok:
                self.n_read += 1
            return ok, frame
        deadline = time.monotonic() + max(0.05, timeout)
        with self._cv:
            while True:
                if self._seq != self._last_seq:
                    self._last_seq = self._seq
                    self.n_read += 1
                    return True, self._latest
                if self._done or self._stop.is_set():
                    return False, None
                if not self._cv.wait(timeout=max(0.01, deadline - time.monotonic())):
                    if time.monotonic() >= deadline:
                        # 아직 살아 있지만 이번 주기에 새 프레임이 없었다.
                        # 호출부가 --show의 q 입력 등을 처리할 수 있게 일단 돌려준다.
                        return True, None

    # ------------------------------------------------------------ 종료
    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:                                        # noqa: BLE001
            pass
        self.cap = None

    def summary(self):
        s = f"스트림: {self.n_read}프레임 수신"
        if self.live:
            s += f" · 재연결 {self.n_reconnect}회"
            if self.drop_stale:
                s += f" · 지연분 {self.n_dropped}프레임 드롭"
        return s


__all__ = ["FrameSource", "is_live"]
