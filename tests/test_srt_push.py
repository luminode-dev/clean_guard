#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SRT 송출(srt_sender.py) 검증 — 서버 없이 로컬 루프백으로.

1) SrtStreamer -> 로컬 SrtListener: 연결, streamid 전달, TS(0x47) 수신
2) 수신기 강제 종료 -> 자동 재연결(백오프) -> 재수신
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from srt_sender import SrtListener, SrtStreamer  # noqa: E402

PORT = 18890
SID = "publish:site01_cam1:jetson01:testpw"
fails = []


def check(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


class Receiver:
    """스레드에서 accept + recv 누적."""

    def __init__(self, port):
        self.listener = SrtListener(port)
        self.data = b""
        self.streamid = None
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        if not self.listener.accept():
            return
        self.streamid = self.listener.peer_streamid()
        while True:
            buf = self.listener.recv()
            if not buf:
                break
            self.data += buf

    def close(self):
        self.listener.close()


print("1) 로컬 수신기로 송출")
rx = Receiver(PORT)
st = SrtStreamer("127.0.0.1", PORT, SID, size=(320, 240), fps=5, bitrate_kbps=300)
frame = np.zeros((240, 320, 3), np.uint8)
frame[60:180, 100:220] = (0, 255, 0)

t_end = time.time() + 6
while time.time() < t_end:
    frame[:, :, 2] = int(time.time() * 50) % 255      # 화면 변화
    st.push(frame)
    time.sleep(0.05)

check("SRT 연결됨", st.connected)
check("streamid 전달", rx.streamid == SID, f"{rx.streamid!r}")
check("TS 수신 (0x47 sync)", len(rx.data) > 5000 and rx.data[0] == 0x47,
      f"{len(rx.data)} bytes")
check("인코더 투입 프레임 > 20", st.frames_in > 20, f"{st.frames_in}")

print("2) 수신기 종료 -> 자동 재연결")
rx.close()
t_end = time.time() + 3                               # 끊김 감지 유도
while time.time() < t_end:
    st.push(frame)
    time.sleep(0.05)
rx2 = Receiver(PORT)                                  # 같은 포트로 재기동
t_end = time.time() + 12                              # 첫 백오프 5초 + 여유
while time.time() < t_end and not (st.connected and len(rx2.data) > 5000):
    st.push(frame)
    time.sleep(0.05)

check("재연결됨", st.connected and st.reconnects >= 1, f"reconnects={st.reconnects}")
check("재연결 후 TS 수신", len(rx2.data) > 5000 and rx2.data[0] == 0x47,
      f"{len(rx2.data)} bytes")

st.close()
rx2.close()

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 SRT 송출 테스트 통과")
