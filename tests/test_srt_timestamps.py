#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SRT 송출 타임스탬프 검증 (2026-10-05 drift 수정) — 서버 없이 인코더 출력 TS를 직접 해석.

감시 루프가 실제로 4~5fps(지터 포함)로 프레임을 줄 때:
1) 영상 PTS 경과 시간 == 벽시계 경과 시간 (예전: 12초 동안 6초만 흐름)
2) 출력은 10fps CFR (PES 간격 100ms)
3) 키프레임 간격 최대 1초 (GOP 10), B-frame 없음 (PTS 단조 증가 = DTS 순서와 같음)
4) 감시 루프가 3초 넘게 멈춰도 영상 시간은 계속 흐른다

usage: python3 tests/test_srt_timestamps.py [초, 기본 30]
"""
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from srt_sender import TsEncoder  # noqa: E402

fails = []


def check(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


def video_pes(ts):
    """TS 바이트 -> [(pts초, IDR 여부)] — 비디오 PES마다. IDR은 H.264 NAL type 5 존재로 판정."""
    out, cur, vpid = [], None, None
    for i in range(0, len(ts) - 187, 188):
        p = ts[i:i + 188]
        if p[0] != 0x47:
            continue
        pid = ((p[1] & 0x1F) << 8) | p[2]
        afc = (p[3] >> 4) & 3
        if afc == 2:
            continue
        off = 5 + p[4] if afc == 3 else 4
        payload = p[off:]
        if p[1] & 0x40 and payload[:3] == b"\x00\x00\x01" and payload[3] == 0xE0:
            if cur is not None:
                out.append(cur)
            vpid = pid
            b = payload[9:14]
            pts = (((b[0] >> 1) & 7) << 30) | (b[1] << 22) | ((b[2] >> 1) << 15) \
                | (b[3] << 7) | (b[4] >> 1)
            cur = [pts / 90000.0, bytearray(payload[9 + payload[8]:])]
        elif pid == vpid and cur is not None:
            cur[1].extend(payload)
    if cur is not None:
        out.append(cur)
    res = []
    for pts, es in out:
        idr, k = False, es.find(b"\x00\x00\x01")
        while k >= 0 and k + 3 < len(es):
            if es[k + 3] & 0x1F == 5:
                idr = True
                break
            k = es.find(b"\x00\x00\x01", k + 3)
        res.append((pts, idr))
    return res


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    enc = TsEncoder(size=(1280, 720), fps=10, bitrate_kbps=1500)
    data = bytearray()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            data.extend(enc.read(0.2))
    th = threading.Thread(target=reader, daemon=True)
    th.start()

    frame = np.random.default_rng(0).integers(40, 200, (720, 1280, 3), dtype=np.uint8)
    t0 = time.monotonic()
    pause = (dur * 0.5, dur * 0.5 + 5.0)               # 중간에 5초 멈춤 (카메라 재연결 흉내)
    n = 0
    while time.monotonic() - t0 < dur:
        el = time.monotonic() - t0
        if not pause[0] <= el < pause[1]:
            f = frame.copy()                        # 화면 일부만 바뀐다 (전체가 바뀌면 x264 장면 전환 IDR)
            f[200:400, (n * 10) % 1100:(n * 10) % 1100 + 150] = 255
            enc.offer(f)
            n += 1
        time.sleep(1 / 4.5 * random.uniform(0.6, 1.4))  # 4~5fps + 지터
    wall = time.monotonic() - t0
    time.sleep(1.0)
    stop.set()
    th.join()
    enc.close()

    pes = video_pes(bytes(data))
    print(f"감시 루프 투입 {n}프레임 ({n / wall:.1f}fps) · 벽시계 {wall:.1f}s · TS {len(data) // 1024}KiB "
          f"· 영상 PES {len(pes)}개 · 재기동 {enc.restarts}회")
    pts = [p for p, _ in pes]
    span = pts[-1] - pts[0] if len(pts) > 1 else 0
    check("영상 시간 == 벽시계 (±3%)", abs(span - wall) / wall < 0.03 + 1.5 / wall,
          f"PTS {span:.2f}s vs 벽시계 {wall:.2f}s")
    gaps = np.diff(pts) if len(pts) > 1 else np.array([0])
    check("출력 10fps CFR (간격 100ms±1ms)", np.all(np.abs(gaps - 0.1) < 0.0015),
          f"간격 min {gaps.min()*1000:.1f} / max {gaps.max()*1000:.1f} ms, "
          f"{(len(pts) - 1) / max(span, 1e-6):.2f}fps")
    check("PTS 단조 증가 (B-frame 없음)", np.all(gaps > 0))
    keys = [p for p, k in pes if k]
    kg = np.diff(keys) if len(keys) > 1 else np.array([0])
    # 상한 1초 (key-int-max). 정지 표시처럼 화면 전체가 바뀌면 x264가 장면 전환 IDR을 더 넣는다
    check("키프레임 간격 최대 1초", len(keys) >= span - 1 and kg.max() < 1.02,
          f"{len(keys)}개 / {span:.0f}초, 간격 {kg.min():.2f}~{kg.max():.2f}s")
    br = len(data) * 8 / wall / 1000
    check("비트레이트 ~1.5Mbps", 1000 < br < 2000, f"{br:.0f} kbps")
    print(f"\n{'모두 통과' if not fails else f'실패 {len(fails)}건: {fails}'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
