#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026-10-04 작업 요청 검증 — 로컬 가짜 API로.

1) ROI 판정: 하단 중앙 기준, exclude 우선, include 없으면 전체 (roi-editor.js isMonitored와 동일)
2) ROI 문서 검증: 구역 수·꼭짓점 수·좌표 범위·type
3) ROI 동기화: 하트비트 config.roi_version -> 응답 버전이 크면 GET /api/device/config
   -> config/roi.json 원자적 저장 -> 적용 -> ack ok / 잘못된 문서는 ack error + 같은 버전 재시도 안 함
4) 회수 판정: 사라지고 confirm_s 지나면 retrieved(after_s), 사람에 가려진 동안·스트림 공백은 안 셈,
   10분 넘게 그대로면 종료
5) event_update: EventSender.update() -> POST /api/events/{id}/update (이벤트 JSON 뒤에)
6) 썸네일: PUT /api/devices/{id}/thumbnail, image/jpeg, 640x360 이하
7) 보존 기간: 모자이크 90일 · 원본 30일 지난 증거 삭제, events.jsonl은 유지
8) GPU 부하: 주기 중 표본 평균 (순간값 0 문제)
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heartbeat                                              # noqa: E402
from heartbeat import Heartbeat, SystemProbe, ThumbnailUploader   # noqa: E402
from recorder import OutputSpace                              # noqa: E402
from retrieval import RetrievalWatch                          # noqa: E402
from roi import RoiConfig, crop_rect, is_monitored, validate             # noqa: E402
from test_event_sender import TOKEN, FakeAPI, make_sender, sample_event, wait_until  # noqa: E402

PORT = 18010            # make_sender()가 이 포트를 쓴다
fails = []

ROI_DOC = {
    "version": 4, "updated_at": "2026-10-04T05:29:10Z", "updated_by": "admin",
    "frame_ref": [1280, 720],
    "zones": [
        {"id": "z1", "name": "수거함 앞", "type": "include",
         "points": [[0.148, 0.346], [0.698, 0.296], [0.65, 0.9], [0.12, 0.92]]},
        {"id": "z2", "name": "도로", "type": "exclude",
         "points": [[0.005, 0.005], [0.995, 0.005], [0.995, 0.2], [0.005, 0.2]]},
    ]}


def check(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


class FakeAPI2(FakeAPI):
    """FakeAPI + ROI 설정·ack·후속 보고·썸네일."""
    roi_doc, roi_version = None, 0
    acks, updates, thumbs, config_gets = [], [], [], 0

    def do_GET(self):                                          # noqa: N802
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._reply(401, {})
        if self.path == "/api/device/config":
            FakeAPI2.config_gets += 1
            return self._reply(200, {"device_id": "JT-GN-0001", "roi": FakeAPI2.roi_doc})
        self._reply(404, {})

    def do_PUT(self):                                          # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/api/devices/JT-GN-0001/thumbnail":
            FakeAPI2.thumbs.append((self.headers.get("Content-Type"), body))
            return self._reply(200, {"ok": True})
        self._reply(404, {})

    def do_POST(self):                                         # noqa: N802
        if self.path == "/api/heartbeat":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            FakeAPI.heartbeats.append(json.loads(body))
            return self._reply(200, {"device_id": "JT-GN-0001", "status": "online",
                                     "config": {"roi_version": FakeAPI2.roi_version}})
        if self.path == "/api/device/config/ack":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            FakeAPI2.acks.append(json.loads(body))
            return self._reply(200, {"ok": True})
        if self.path.endswith("/update"):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            FakeAPI2.updates.append((self.path.split("/")[3], json.loads(body),
                                     len(FakeAPI.events)))
            return self._reply(200, {"ok": True})
        return super().do_POST()


def test_rules():
    print("[1] ROI 판정 규칙")
    zones = validate(ROI_DOC)
    W, H = 1280, 720
    check("구역 안 (하단 중앙)", is_monitored([500, 300, 560, 400], W, H, zones))
    # 박스 윗부분은 구역 밖(위)이지만 하단 중앙이 안 -> 감시
    check("하단 중앙 기준", is_monitored([500, 50, 560, 400], W, H, zones))
    check("구역 밖", not is_monitored([1100, 600, 1200, 700], W, H, zones))
    check("제외 구역이 우선", not is_monitored([500, 20, 560, 100], W, H, zones))
    only_exc = [z for z in zones if z["type"] == "exclude"]
    check("include 없으면 전체 감시", is_monitored([1100, 600, 1200, 700], W, H, only_exc))
    check("구역 없음 = 전체", is_monitored([0, 0, 10, 10], W, H, []))

    print("[1-1] 쓰레기 탐지 크롭 영역")
    sq = [{"type": "include", "points": [(0.5, 0.5), (0.7, 0.5), (0.7, 0.8), (0.5, 0.8)]}]
    x1, y1, x2, y2 = crop_rect(sq, W, H)
    check("include 외접 사각형 + 여유 (위쪽 더 넉넉히)",
          x1 < 640 and x2 > 896 and y1 <= 360 - 0.15 * H + 1 and y2 >= 576, str((x1, y1, x2, y2)))
    check("프레임 안으로 자름", crop_rect([{"type": "include", "points": [(0, 0), (1, 0), (1, 1)]}],
                                    W, H) == (0, 0, W, H))
    check("include 없으면 None (전체 프레임)", crop_rect(only_exc, W, H) is None
          and crop_rect([], W, H) is None)

    print("[2] ROI 문서 검증")
    def bad(doc):
        try:
            validate(doc)
            return None
        except ValueError as e:
            return str(e)
    check("정상 문서", bad(ROI_DOC) is None)
    check("꼭짓점 2개 거부", bad({"zones": [{"type": "include", "points": [[0, 0], [1, 1]]}]}))
    check("17개 구역 거부", bad({"zones": [ROI_DOC["zones"][0]] * 17}))
    check("범위 밖 좌표 거부", bad({"zones": [{"type": "include",
                                         "points": [[0, 0], [1.5, 0], [1, 1]]}]}))
    check("type 오류 거부", bad({"zones": [{"type": "maybe", "points": [[0, 0], [1, 0], [1, 1]]}]}))
    check("좌표 파싱 오류 거부", bad({"zones": [{"type": "include",
                                          "points": [["a", 0], [1, 0], [1, 1]]}]}))
    check("zones 비어 있음 허용 (= 전체)", bad({"version": 1, "zones": []}) is None)


def test_sync(tmp):
    print("[3] ROI 원격 동기화 (하트비트 -> GET -> 저장 -> ack)")
    FakeAPI.heartbeats.clear()
    FakeAPI2.acks.clear()
    FakeAPI2.roi_doc, FakeAPI2.roi_version, FakeAPI2.config_gets = ROI_DOC, 4, 0
    path = tmp / "config" / "roi.json"
    roi = RoiConfig(path, base_url=f"http://127.0.0.1:{PORT}", token=TOKEN, verbose=False)
    check("파일 없으면 v0 · 전체 감시", roi.version == 0 and roi.zones == [])

    hb = Heartbeat(f"http://127.0.0.1:{PORT}", TOKEN, "JT-GN-0001",
                   lambda: {"pipeline": {"frames_total": 0, "stream": "ok"}},
                   interval=5, verbose=False, roi=roi)
    check("첫 하트비트 config.roi_version=0",
          wait_until(lambda: FakeAPI.heartbeats, 3)
          and FakeAPI.heartbeats[0].get("config") == {"roi_version": 0})
    check("v4 적용 + ack ok", wait_until(lambda: FakeAPI2.acks, 3)
          and FakeAPI2.acks[0] == {"roi_version": 4, "ok": True}, str(FakeAPI2.acks))
    check("RoiConfig v4 · 구역 2개", roi.version == 4 and len(roi.zones) == 2)
    check("config/roi.json 저장", path.is_file()
          and json.loads(path.read_text(encoding="utf-8"))["version"] == 4)
    check("임시 파일 안 남음", not list(path.parent.glob(".*.tmp")))
    hb.send_once()
    check("다음 하트비트는 roi_version=4", FakeAPI.heartbeats[-1]["config"] == {"roi_version": 4})
    check("같은 버전 재요청 안 함", FakeAPI2.config_gets == 1)

    print("    잘못된 v5 -> ack error, 기존 v4 유지, 같은 버전 재시도 안 함")
    FakeAPI2.roi_doc = {"version": 5, "zones": [{"id": "z9", "type": "include",
                                                 "points": [[0, 0], [1, 1]]}]}
    FakeAPI2.roi_version = 5
    hb.send_once()
    check("ack ok=false + error", FakeAPI2.acks[-1].get("ok") is False
          and FakeAPI2.acks[-1].get("roi_version") == 5
          and "points" in FakeAPI2.acks[-1].get("error", ""), str(FakeAPI2.acks[-1]))
    check("기존 v4 유지 (메모리·파일)", roi.version == 4
          and json.loads(path.read_text(encoding="utf-8"))["version"] == 4)
    n_gets = FakeAPI2.config_gets
    hb.send_once()
    check("실패 버전 재요청 안 함", FakeAPI2.config_gets == n_gets)
    check("하트비트는 적용 버전(4) 보고", FakeAPI.heartbeats[-1]["config"] == {"roi_version": 4})

    print("    재시작 -> config/roi.json으로 복원 (서버 없이)")
    hb.close()
    roi2 = RoiConfig(path, verbose=False)
    check("파일에서 v4 복원", roi2.version == 4 and len(roi2.zones) == 2)
    check("서버 미연결이면 sync 안 함", roi2.sync(9) is False)

    print("    송출 프레임에 구역 그리기")
    img = np.zeros((720, 1280, 3), np.uint8)
    roi2.draw(img)
    check("외곽선이 그려짐", int(img[:, :, 1].max()) > 0 and int(img[:, :, 2].max()) > 0)


def test_retrieval():
    print("[4] 회수 판정")
    got = []
    w = RetrievalWatch(lambda eid, s: got.append((eid, s)), window_s=600, confirm_s=60)
    box = [100, 100, 200, 200]
    t = 0.0
    w.add("E1", box, t)
    for _ in range(300):                       # 30초 동안 그대로 있음
        t += 0.1
        w.update(t, [box], [])
    for _ in range(590):                       # 59초 안 보임 -> 아직 확정 아님
        t += 0.1
        w.update(t, [], [])
    check("confirm 전엔 보고 안 함", got == [])
    for _ in range(20):
        t += 0.1
        w.update(t, [], [])
    check("사라진 뒤 60초 -> retrieved", got == [("E1", 30)], str(got))

    got.clear()
    w = RetrievalWatch(lambda eid, s: got.append((eid, s)), window_s=600, confirm_s=60)
    t = 0.0
    w.add("E2", box, t)
    for _ in range(1200):                      # 120초 동안 사람이 가리고 서 있음 (탐지 안 됨)
        t += 0.1
        w.update(t, [], [[90, 50, 210, 260]])
    check("사람에 가려진 동안은 안 셈", got == [])
    for _ in range(50):                        # 다시 보임 -> 부재 초기화
        t += 0.1
        w.update(t, [[105, 102, 198, 205]], [])
    check("살짝 움직인 같은 자리 = 있음", not got and w.items[0].absent_s == 0)

    w.update(t + 500, [], [])                  # 스트림 500초 공백 뒤 첫 프레임
    check("스트림 공백은 상한(2초)만 셈", not got and w.items[0].absent_s <= 2.0)

    got.clear()
    w = RetrievalWatch(lambda eid, s: got.append((eid, s)), window_s=600, confirm_s=60)
    w.add("E3", box, 0.0)
    t = 0.0
    while t < 700:
        t += 1.0
        w.update(t, [box], [])
    check("10분 넘게 그대로면 판정 종료", not w.items and w.n_expired == 1 and not got)


def test_update_and_thumb(tmp):
    print("[5] event_update 전송 (이벤트 JSON 뒤)")
    FakeAPI.mode = "ok"
    FakeAPI.events.clear()
    FakeAPI2.updates.clear()
    s = make_sender(tmp / "q", min_free_mb=0)   # 보드 루트 여유가 2GB 미만이면 큐 정리가 끼어든다
    eid = s.submit(sample_event(1))
    s.update(eid, "retrieved", after_s=312)
    check("update 도착", wait_until(lambda: FakeAPI2.updates, 5))
    path_id, body, n_events_before = FakeAPI2.updates[0]
    check("경로 event_id", path_id == eid)
    check("본문", body.get("update") == "retrieved" and body.get("after_s") == 312
          and body.get("device_id") == "JT-GN-0001" and body.get("event_id") == eid
          and body.get("ts", "")[-6] in "+-", str(body))
    check("내부 필드 안 샘", not any(k.startswith("_") for k in body))
    check("이벤트 JSON이 먼저", n_events_before >= 1)
    check("큐 비움", wait_until(lambda: s.pending() == 0, 3))
    s.close()

    print("[6] 썸네일 PUT")
    FakeAPI2.thumbs.clear()
    th = ThumbnailUploader(f"http://127.0.0.1:{PORT}", TOKEN, "JT-GN-0001",
                           interval=1, verbose=False)
    th.offer(np.full((1440, 2560, 3), 90, np.uint8))
    th.offer(np.zeros((1440, 2560, 3), np.uint8))             # 주기 안 -> 무시
    check("1장 도착", wait_until(lambda: FakeAPI2.thumbs, 3) and len(FakeAPI2.thumbs) == 1)
    ctype, data = FakeAPI2.thumbs[0]
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    check("image/jpeg", ctype == "image/jpeg")
    check("640x360", img is not None and img.shape[:2] == (360, 640),
          str(None if img is None else img.shape))
    check("5MB 이하", len(data) < 5 * 1024 * 1024)
    th.close()


def test_expire(tmp):
    print("[7] 보존 기간 삭제")
    d = tmp / "out"
    d.mkdir()
    now = time.time()
    files = {"event_0001.jpg": 100, "event_0001_raw.jpg": 100, "event_0002.jpg": 40,
             "event_0002_raw.jpg": 40, "event_0003.mp4": 10, "events.jsonl": 400}
    for name, age_days in files.items():
        p = d / name
        p.write_bytes(b"x")
        os.utime(p, (now - age_days * 86400, now - age_days * 86400))
    sp = OutputSpace(d, keep_days=90, raw_keep_days=30)
    n = sp.expire(now=now)
    left = sorted(p.name for p in d.iterdir())
    check("90일 넘은 모자이크·30일 넘은 원본 삭제", n == 3, str(left))
    check("남은 것", left == ["event_0002.jpg", "event_0003.mp4", "events.jsonl"], str(left))


def test_gpu():
    print("[8] GPU 부하 평균")
    seq = iter(["0", "500", "0", "1000"])
    orig = heartbeat._read
    heartbeat._read = lambda path, default=None: (next(seq) if path == heartbeat.GPU_LOAD
                                                  else orig(path, default))
    try:
        pr = SystemProbe(tegrastats=False)
        for _ in range(3):
            pr.tick()
        s = pr.sample()                        # sample()이 마지막 표본(1000)을 더함
        check("표본 평균 (0,500,0,1000) -> 38%", s["gpu_pct"] == 38, str(s["gpu_pct"]))
    finally:
        heartbeat._read = orig


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), FakeAPI2)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="cg_roi_"))
    test_rules()
    test_sync(tmp)
    test_retrieval()
    test_update_and_thumb(tmp)
    test_expire(tmp)
    test_gpu()
    srv.shutdown()
    print(f"\n{'모두 통과' if not fails else f'실패 {len(fails)}건: {fails}'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
