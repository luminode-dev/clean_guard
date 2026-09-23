#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""libsrt(ctypes) 기반 SRT 송신기 — MPEG-TS 바이트를 서버 MediaMTX로 push한다.

JetPack 6 GStreamer(1.20.3)의 srtsink은 실서버 핸드셰이크 직후 SIGSEGV가 나서
(보드 실측 2026-09-23) libsrt-gnutls를 직접 호출한다. ffmpeg 불필요.

단독 실행(테스트): stdin으로 TS를 받아 송신
  gst-launch-1.0 videotestsrc ... ! mpegtsmux ! fdsink | \
      python3 srt_sender.py --host cleanguard.duckdns.org --port 8890 \
                            --streamid 'publish:site01_cam1:jetson01:비번'
"""
import ctypes
import ctypes.util
import os
import time

SRT_CHUNK = 1316            # TS 188바이트 x 7 — SRT 권장 페이로드

# ---- libsrt 상수 (srt.h 1.4.x) ----
SRTO_STREAMID = 46
SRTO_LATENCY = 23           # us가 아니라 ms
SRTO_SNDSYN = 1             # 블로킹 송신
SRT_ERROR = -1


class SrtSender:
    """caller 모드 SRT 연결 1개. send()가 실패하면 끊긴 것 — 호출측이 재연결한다."""

    def __init__(self, host, port, streamid, latency_ms=200):
        self.host, self.port, self.streamid = host, int(port), streamid
        self.latency_ms = latency_ms
        self._lib = self._load()
        self._sock = None

    @staticmethod
    def _load():
        for name in ("srt-gnutls", "srt"):
            path = ctypes.util.find_library(name)
            if path:
                lib = ctypes.CDLL(path)
                # 64비트에서 포인터 반환형을 지정하지 않으면 int로 잘려 segfault가 난다
                lib.srt_getlasterror_str.restype = ctypes.c_char_p
                lib.srt_getlasterror_str.argtypes = []
                lib.srt_create_socket.restype = ctypes.c_int
                lib.srt_connect.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
                lib.srt_setsockflag.argtypes = [ctypes.c_int, ctypes.c_int,
                                                ctypes.c_void_p, ctypes.c_int]
                lib.srt_sendmsg2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                             ctypes.c_int, ctypes.c_void_p]
                lib.srt_startup()
                return lib
        raise RuntimeError("libsrt를 찾을 수 없음 (libsrt-gnutls)")

    def connect(self):
        lib = self._lib
        s = lib.srt_create_socket()
        if s == SRT_ERROR:
            raise RuntimeError("srt_create_socket 실패")
        sid = self.streamid.encode()
        if lib.srt_setsockflag(s, SRTO_STREAMID, sid, len(sid)) == SRT_ERROR:
            lib.srt_close(s)
            raise RuntimeError("streamid 설정 실패")
        lat = ctypes.c_int(self.latency_ms)
        lib.srt_setsockflag(s, SRTO_LATENCY, ctypes.byref(lat), ctypes.sizeof(lat))
        syn = ctypes.c_int(1)
        lib.srt_setsockflag(s, SRTO_SNDSYN, ctypes.byref(syn), ctypes.sizeof(syn))

        # sockaddr_in 구성 (IPv4)
        import socket as pysock
        ip = pysock.gethostbyname(self.host)
        addr = ctypes.create_string_buffer(16)
        ctypes.memmove(addr, bytes([2, 0]) + self.port.to_bytes(2, "big")
                       + pysock.inet_aton(ip), 8)
        if lib.srt_connect(s, addr, 16) == SRT_ERROR:
            err = ctypes.string_at(lib.srt_getlasterror_str()).decode(errors="replace")
            lib.srt_close(s)
            raise ConnectionError(f"SRT 연결 실패 {self.host}:{self.port} — {err}")
        self._sock = s

    def send(self, data: bytes):
        """TS 바이트를 1316B 단위로 송신. 실패 시 ConnectionError."""
        lib = self._lib
        for i in range(0, len(data), SRT_CHUNK):
            chunk = data[i:i + SRT_CHUNK]
            if lib.srt_sendmsg2(self._sock, chunk, len(chunk), None) == SRT_ERROR:
                err = ctypes.string_at(lib.srt_getlasterror_str()).decode(errors="replace")
                raise ConnectionError(f"SRT 송신 실패 — {err}")

    def close(self):
        if self._sock is not None:
            self._lib.srt_close(self._sock)
            self._sock = None


class SrtListener:
    """테스트·진단용 SRT 수신기 (listener 모드). accept() 후 recv()로 TS 바이트를 받는다."""

    def __init__(self, port, host="127.0.0.1"):
        self._lib = SrtSender._load()
        lib = self._lib
        lib.srt_bind.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.srt_accept.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.srt_recvmsg2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                                     ctypes.c_int, ctypes.c_void_p]
        lib.srt_getsockflag.argtypes = [ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        import socket as pysock
        self._srv = lib.srt_create_socket()
        addr = ctypes.create_string_buffer(16)
        ctypes.memmove(addr, bytes([2, 0]) + int(port).to_bytes(2, "big")
                       + pysock.inet_aton(host), 8)
        if lib.srt_bind(self._srv, addr, 16) == SRT_ERROR:
            raise RuntimeError("SRT bind 실패: "
                               + ctypes.string_at(lib.srt_getlasterror_str()).decode())
        if lib.srt_listen(self._srv, 1) == SRT_ERROR:
            raise RuntimeError("SRT listen 실패")
        self._peer = None

    def accept(self):
        self._peer = self._lib.srt_accept(self._srv, None, None)
        return self._peer != SRT_ERROR

    def peer_streamid(self):
        buf = ctypes.create_string_buffer(512)
        n = ctypes.c_int(512)
        self._lib.srt_getsockflag(self._peer, SRTO_STREAMID, buf, ctypes.byref(n))
        return buf.raw[:n.value].decode(errors="replace")

    def recv(self, maxlen=SRT_CHUNK):
        buf = ctypes.create_string_buffer(maxlen)
        n = self._lib.srt_recvmsg2(self._peer, buf, maxlen, None)
        return buf.raw[:n] if n > 0 else b""

    def close(self):
        for s in (self._peer, self._srv):
            if s not in (None, SRT_ERROR):
                self._lib.srt_close(s)


# x264 소프트웨어 인코딩 — Orin Nano에는 영상 '인코더' 하드웨어(NVENC)가 없다
# (디코더만 있음, 보드 실측 2026-09-23. 인수인계 문서의 'NVENC 여유'는 오기).
ENC_PIPELINE = (
    "fdsrc fd=0 ! "
    "rawvideoparse format=bgr width={w} height={h} framerate={fps}/1 ! "
    "videoconvert ! "
    "x264enc bitrate={kbps} speed-preset=veryfast tune=zerolatency "
    "key-int-max={fps} bframes=0 ! "
    "h264parse ! mpegtsmux ! fdsink fd=1"
)


class SrtStreamer:
    """주석 프레임을 720p H.264 MPEG-TS로 인코딩해 SRT push하는 상시 송출기.

    - push(frame): 논블로킹에 가깝다. fps 초과분은 버리고, 크기가 다르면 리사이즈
    - 서버가 죽거나 끊기면 지수 백오프(5초 -> 최대 30초)로 무한 재연결, 그동안 프레임은 버린다
    - 인코더(gst-launch 서브프로세스)가 죽으면 다음 push에서 재기동
    """

    def __init__(self, host, port, streamid, size=(1280, 720), fps=10,
                 bitrate_kbps=1500, latency_ms=200):
        import subprocess
        import threading
        self._sub, self._th = subprocess, threading
        self.host, self.port, self.streamid = host, int(port), streamid
        self.size, self.fps = tuple(size), int(fps)
        self.kbps, self.latency_ms = int(bitrate_kbps), int(latency_ms)
        self.frames_in = self.sent_bytes = self.reconnects = 0
        self.connected = False
        self._next_t = 0.0
        self._enc = None
        self._stop = threading.Event()
        self._start_encoder()
        self._pump = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump.start()

    def _start_encoder(self):
        w, h = self.size
        cmd = ["gst-launch-1.0", "-q"] + ENC_PIPELINE.format(
            w=w, h=h, fps=self.fps, kbps=self.kbps).split()
        self._enc = self._sub.Popen(cmd, stdin=self._sub.PIPE, stdout=self._sub.PIPE,
                                    stderr=self._sub.DEVNULL, bufsize=0)

    def push(self, frame):
        now = time.monotonic()
        if now < self._next_t:
            return
        self._next_t = now + 1.0 / self.fps
        if self._enc.poll() is not None:            # 인코더 사망 -> 재기동
            self._start_encoder()
        import cv2
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        try:
            self._enc.stdin.write(frame.tobytes())
            self.frames_in += 1
        except (BrokenPipeError, OSError):
            pass                                    # 다음 push에서 재기동

    def _pump_loop(self):
        """인코더 stdout의 TS를 읽어 SRT로 보낸다. 연결 관리도 여기서."""
        tx = None
        retry_at, backoff = 0.0, 5.0
        while not self._stop.is_set():
            buf = self._enc.stdout.read(SRT_CHUNK * 8)
            if not buf:                             # 인코더 재기동 대기
                time.sleep(0.2)
                continue
            if tx is None:
                if time.monotonic() < retry_at:
                    continue                        # 끊긴 동안 TS는 버린다
                try:
                    tx = SrtSender(self.host, self.port, self.streamid, self.latency_ms)
                    tx.connect()
                    self.connected = True
                    backoff = 5.0
                    print(f"[송출] SRT 연결 {self.host}:{self.port}", flush=True)
                except Exception as e:              # noqa: BLE001
                    tx = None
                    retry_at = time.monotonic() + backoff
                    print(f"[송출] 연결 실패({type(e).__name__}: {str(e)[:80]}) "
                          f"-> {backoff:.0f}초 후 재시도", flush=True)
                    backoff = min(backoff * 2, 30.0)
                    continue
            try:
                tx.send(buf)
                self.sent_bytes += len(buf)
            except ConnectionError as e:
                print(f"[송출] 끊김({str(e)[:80]}) -> 재연결", flush=True)
                tx.close()
                tx = None
                self.connected = False
                self.reconnects += 1
                retry_at = time.monotonic() + backoff
        if tx:
            tx.close()

    def close(self):
        self._stop.set()
        try:
            self._enc.stdin.close()
            self._enc.terminate()
        except OSError:
            pass
        self._pump.join(timeout=3)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="stdin의 MPEG-TS를 SRT로 송신")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--streamid", required=True)
    ap.add_argument("--latency-ms", type=int, default=200)
    a = ap.parse_args()

    tx = SrtSender(a.host, a.port, a.streamid, a.latency_ms)
    tx.connect()
    print(f"[SRT] 연결됨 {a.host}:{a.port}", flush=True)
    fd = os.open("/dev/stdin", os.O_RDONLY) if False else 0
    sent = 0
    t0 = time.time()
    try:
        while True:
            buf = os.read(fd, SRT_CHUNK * 32)
            if not buf:
                break
            tx.send(buf)
            sent += len(buf)
            if time.time() - t0 >= 5:
                print(f"[SRT] 송신 {sent/1024:.0f} KiB", flush=True)
                t0 = time.time()
    finally:
        tx.close()
        print(f"[SRT] 종료 · 총 {sent/1024:.0f} KiB", flush=True)


if __name__ == "__main__":
    main()
