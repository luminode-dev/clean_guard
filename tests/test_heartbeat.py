#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""하트비트(heartbeat.py) 검증 — 로컬 가짜 API로.

1) 시작 즉시 1회 + 주기 전송, Bearer 토큰·device_id·ts 오프셋
2) 본문 구조가 jetson_data.md §3과 맞는지 (pipeline/tts/mosaic/system/issues)
3) fps = 두 하트비트 사이 frames_total 증가량 / 경과 시간
4) issues 자동 판정: stream_lost, low_fps, tts_worker_down + 감시 루프가 넘긴 것 유지
5) 서버 다운 -> 예외 없이 실패 집계, 복구되면 다시 전송
"""
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heartbeat import Heartbeat                              # noqa: E402
from test_event_sender import TOKEN, FakeAPI, wait_until     # noqa: E402

PORT = 18012
fails = []


def check(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeAPI.heartbeats.clear()
    FakeAPI.mode = "ok"

    state = {"frames": 0, "stream": "ok", "worker": "ready", "issues": []}

    def collect():
        return {"pipeline": {"frames_total": state["frames"], "stream": state["stream"],
                             "infer_ms": {"person": 21.0, "waste": 19.0},
                             "tracked_persons": 2, "tracked_objects": 1},
                "tts": {"worker": state["worker"], "played_total": 3, "suppressed_total": 1,
                        "synth_fail_total": 0, "last_synth_ms": 4400},
                "mosaic": {"mode": "face", "faces_total": 5, "head_fallback_total": 1},
                "issues": list(state["issues"])}

    print("[1] 시작 즉시 전송 + 본문 구조")
    hb = Heartbeat(f"http://127.0.0.1:{PORT}", TOKEN, "JT-GN-0001", collect, interval=5)
    check("첫 하트비트 즉시 도착", wait_until(lambda: len(FakeAPI.heartbeats) >= 1, 3))
    b = FakeAPI.heartbeats[0]
    check("device_id", b.get("device_id") == "JT-GN-0001")
    check("ts 오프셋", b["ts"][-6] in "+-", b["ts"])
    for k in ("uptime_s", "pipeline", "tts", "mosaic", "system", "issues"):
        check(f"필드 {k}", k in b)
    sysb = b["system"]
    for k in ("cpu_pct", "gpu_pct", "mem_pct", "temp_c", "disk_free_mb", "power_mode", "net"):
        check(f"system.{k}", k in sysb, str(sysb.get(k)))
    check("온도 cpu/gpu 실측", {"cpu", "gpu"} <= set(sysb["temp_c"]), str(sysb["temp_c"]))
    check("첫 회 fps는 None (기준 없음)", b["pipeline"]["fps"] is None)
    check("정상이면 issues 비어 있음", b["issues"] == [] or b["issues"] == ["clock_unsynced"],
          str(b["issues"]))

    print("[2] fps 계산 (다음 주기)")
    t = time.monotonic()
    while time.monotonic() - t < 4.5:          # 약 15fps로 프레임 증가
        state["frames"] += 1
        time.sleep(1 / 15)
    check("두 번째 하트비트", wait_until(lambda: len(FakeAPI.heartbeats) >= 2, 4))
    fps = FakeAPI.heartbeats[1]["pipeline"]["fps"]
    check("fps ≈ 프레임 증가율", fps is not None and 8 <= fps <= 16, str(fps))
    check("CPU%는 두 번째부터 계산", isinstance(FakeAPI.heartbeats[1]["system"]["cpu_pct"], int))
    hb.close()

    print("[3] issues 자동 판정")
    state.update(stream="reconnecting", worker="down", issues=["queue_full"])
    hb = Heartbeat(f"http://127.0.0.1:{PORT}", TOKEN, "JT-GN-0001", collect, interval=5)
    n0 = len(FakeAPI.heartbeats)
    check("도착", wait_until(lambda: len(FakeAPI.heartbeats) > n0, 3))
    iss = FakeAPI.heartbeats[-1]["issues"]
    check("stream_lost", "stream_lost" in iss, str(iss))
    check("tts_worker_down", "tts_worker_down" in iss)
    check("감시 루프가 넘긴 queue_full 유지", "queue_full" in iss)
    hb.close()
    state.update(stream="ok", worker="ready", issues=[])
    hb = Heartbeat(f"http://127.0.0.1:{PORT}", TOKEN, "JT-GN-0001", collect, interval=5)
    time.sleep(0.3)
    hb._prev = (time.monotonic() - 10, state["frames"] - 20)     # 10초에 20프레임 = 2fps
    body = hb.build()
    check("low_fps", "low_fps" in body["issues"], str(body["pipeline"]["fps"]))
    hb.close()

    print("[4] 서버 다운 -> 실패 집계, 예외 없음 -> 복구")
    FakeAPI.mode = "down"
    hb = Heartbeat(f"http://127.0.0.1:{PORT}", TOKEN, "JT-GN-0001", collect, interval=5)
    check("실패 집계", wait_until(lambda: hb.failed >= 1, 3))
    FakeAPI.mode = "ok"
    n0 = len(FakeAPI.heartbeats)
    check("복구 후 send_once 성공", hb.send_once() and len(FakeAPI.heartbeats) > n0)
    hb.close()
    check("summary", "하트비트" in hb.summary(), hb.summary())

    srv.shutdown()
    print(f"\n{'실패: ' + ', '.join(fails) if fails else '전체 통과'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
