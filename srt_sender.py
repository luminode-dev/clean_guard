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
#
# 타임스탬프는 실제 시각 기준이다 (2026-10-05). 예전 파이프라인(fdsrc ! rawvideoparse framerate=10/1)은
# 들어온 프레임 수 x 0.1초로 PTS를 매겼는데, 감시 루프의 송출 간격 제한이 지터 때문에 프레임을 절반쯤
# 버려 실제로는 4~5fps가 들어갔다 -> 벽시계 12초에 영상 6초, 서버 녹화기가 10~13초마다
# "detected drift between recording duration and absolute time"으로 리셋했다.
# 지금은 appsrc(is-live, do-timestamp)가 넣는 순간의 파이프라인 시계로 PTS를 찍고,
# videorate가 그 PTS를 보고 정확히 {fps}fps CFR로 복제·삭제한다.
ENC_PIPELINE = (
    "appsrc name=src is-live=true do-timestamp=true format=time block=false "
    "max-bytes={qbytes} leaky-type=downstream "
    "caps=\"video/x-raw,format=BGR,width={w},height={h},framerate=0/1\" ! "
    # BGR을 그냥 넘기면 x264가 Y444로 받아 High 4:4:4 프로파일이 된다 -> PC 브라우저(소프트웨어
    # 디코딩)는 재생되지만 모바일 하드웨어 디코더는 못 풀어 WebRTC가 무한 로딩 (2026-09-26 실측).
    # 4:2:0 + constrained-baseline으로 고정해 모든 모바일 브라우저가 재생하게 한다.
    "videoconvert ! videorate ! video/x-raw,format=I420,framerate={fps}/1 ! "
    "x264enc bitrate={kbps} speed-preset=veryfast tune=zerolatency "
    "key-int-max={fps} bframes=0 ! "                 # GOP 1초, B-frame 없음
    "video/x-h264,profile=constrained-baseline ! "
    "h264parse ! mpegtsmux alignment=7 ! "           # 버퍼 1개 = TS 7패킷 = SRT 메시지 1개(1316B)
    "appsink name=sink sync=false max-buffers=2000 drop=true"
)
STALE_S = 3.0               # 감시 루프가 이만큼 프레임을 안 주면 마지막 프레임에 정지 표시를 얹어 계속 보낸다


class TsEncoder:
    """BGR 프레임 -> 실시간 타임스탬프 H.264 MPEG-TS (GStreamer in-process).

    - offer(frame): 감시 루프는 최신 프레임 참조만 넘긴다 (논블로킹, 복사 없음).
    - 페이싱 스레드가 1/fps마다 최신 프레임을 appsrc에 넣는다. 감시 루프가 느리거나(4~5fps)
      지터가 있어도 PTS는 넣은 시각이고, videorate가 빈 칸을 복제해 CFR을 맞춘다.
    - 감시 루프가 멈추면(카메라 재연결 등) 마지막 프레임을 'NO SIGNAL' 표시와 함께 계속 보낸다
      -> 영상 시간이 끊기지 않아 서버 녹화가 실제 시각과 어긋나지 않는다.
    - read(timeout): 인코딩된 TS 바이트 (1316B 배수). 파이프라인 오류 시 자동 재기동.
    """

    def __init__(self, size=(1280, 720), fps=10, bitrate_kbps=1500):
        import threading
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        self._Gst = Gst
        self.size, self.fps, self.kbps = tuple(size), int(fps), int(bitrate_kbps)
        self.frames_in = self.restarts = 0
        self._latest, self._latest_t = None, 0.0
        self._stale_src = self._stale_img = None
        self._lock = threading.Lock()                   # 파이프라인 교체 vs read()
        self._pipe = self._src = self._sink = None
        self._stop = threading.Event()
        self._start()
        self._pace = threading.Thread(target=self._pace_loop, name="srt-pace", daemon=True)
        self._pace.start()

    def _start(self):
        Gst = self._Gst
        w, h = self.size
        desc = ENC_PIPELINE.format(w=w, h=h, fps=self.fps, kbps=self.kbps,
                                   qbytes=w * h * 3 * 3)        # appsrc 대기열 최대 3프레임
        pipe = Gst.parse_launch(desc)
        src, sink = pipe.get_by_name("src"), pipe.get_by_name("sink")
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("인코더 파이프라인 시작 실패")
        with self._lock:
            self._pipe, self._src, self._sink = pipe, src, sink

    def _restart(self, why):
        print(f"[송출] 인코더 재기동 ({why})", flush=True)
        with self._lock:
            old, self._pipe = self._pipe, None
        if old is not None:
            old.set_state(self._Gst.State.NULL)
        self.restarts += 1
        self._start()

    def offer(self, frame):
        self._latest, self._latest_t = frame, time.monotonic()

    def _frame_to_send(self):
        frame = self._latest
        if frame is None:
            return None
        import cv2
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if time.monotonic() - self._latest_t < STALE_S:
            return frame
        if self._stale_src is not self._latest:         # 정지 화면은 한 번만 만든다
            img = frame.copy()
            img[:] = (img * 0.5).astype(img.dtype)
            cv2.putText(img, "NO SIGNAL", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 0, 255), 4, cv2.LINE_AA)
            self._stale_src, self._stale_img = self._latest, img
        return self._stale_img

    def _pace_loop(self):
        Gst = self._Gst
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            next_t += period
            now = time.monotonic()
            if next_t < now - 1.0:                      # 크게 밀렸으면 따라잡지 않는다 (videorate가 메움)
                next_t = now
            if next_t > now:
                self._stop.wait(next_t - now)
            pipe = self._pipe
            if pipe is None:
                continue
            msg = pipe.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is not None:
                why = (msg.parse_error()[0].message if msg.type == Gst.MessageType.ERROR
                       else "EOS")
                try:
                    self._restart(why[:80])
                except Exception as e:                  # noqa: BLE001 — 다음 주기에 다시
                    print(f"[송출] 인코더 재기동 실패({e})", flush=True)
                continue
            frame = self._frame_to_send()
            if frame is None:
                continue
            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            if self._src.emit("push-buffer", buf) == Gst.FlowReturn.OK:
                self.frames_in += 1

    def read(self, timeout=0.5):
        """인코딩된 TS 바이트. 없으면 b""."""
        with self._lock:
            sink = self._sink
        if sink is None:
            time.sleep(timeout)
            return b""
        sample = sink.emit("try-pull-sample", int(timeout * 1e9))
        if sample is None:
            return b""
        buf = sample.get_buffer()
        ok, info = buf.map(self._Gst.MapFlags.READ)
        if not ok:
            return b""
        try:
            return bytes(info.data)
        finally:
            buf.unmap(info)

    def close(self):
        self._stop.set()
        self._pace.join(timeout=2)
        with self._lock:
            pipe, self._pipe, self._sink = self._pipe, None, None
        if pipe is not None:
            pipe.set_state(self._Gst.State.NULL)


class SrtStreamer:
    """주석 프레임을 720p H.264 MPEG-TS로 인코딩해 SRT push하는 상시 송출기.

    - push(frame): 논블로킹. 최신 프레임만 넘기고 인코딩·송출 속도(fps)는 TsEncoder가 맞춘다
    - 서버가 죽거나 끊기면 지수 백오프(5초 -> 최대 30초)로 무한 재연결, 그동안 TS는 버린다
    - 인코더 파이프라인이 오류를 내면 TsEncoder가 재기동
    """

    def __init__(self, host, port, streamid, size=(1280, 720), fps=10,
                 bitrate_kbps=1500, latency_ms=200):
        import threading
        self.host, self.port, self.streamid = host, int(port), streamid
        self.size, self.fps = tuple(size), int(fps)
        self.kbps, self.latency_ms = int(bitrate_kbps), int(latency_ms)
        self.sent_bytes = self.reconnects = 0
        self.connected = False
        self._enc = TsEncoder(self.size, self.fps, self.kbps)
        self._stop = threading.Event()
        self._pump = threading.Thread(target=self._pump_loop, name="srt-pump", daemon=True)
        self._pump.start()

    @property
    def frames_in(self):
        return self._enc.frames_in

    def push(self, frame):
        self._enc.offer(frame)

    def _pump_loop(self):
        """인코더의 TS를 읽어 SRT로 보낸다. 연결 관리도 여기서."""
        tx = None
        retry_at, backoff = 0.0, 5.0
        while not self._stop.is_set():
            buf = self._enc.read(0.5)
            if not buf:
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
        self._pump.join(timeout=3)
        self._enc.close()


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
