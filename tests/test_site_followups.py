#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026-10-05 작업 요청 검증 — 로컬 가짜 API로.

1) 설치 위치 동기화: 하트비트 config.site_version -> 응답 버전이 크면 GET /api/device/config
   -> config/site.json 원자적 저장 -> ack {site_version, ok} -> 방송 장소명 교체
2) location null이면 이전 좌표를 남기지 않음 / 잘못된 문서는 ack error + 기존 유지
3) 서버 없이 재시작해도 site.json으로 동작, ROI와 함께 하트비트에 두 버전 보고
4) GPU: tegrastats GR3D_FREQ 평균, tegrastats 없으면 sysfs 폴백
5) 회수 판정 상태 파일: 재시작 후 이어서 판정, 시간 지난 건 버림
6) 하트비트 disk_low 기준 2000MB, output_free_mb 보고
"""
import json
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import heartbeat                                              # noqa: E402
from heartbeat import Heartbeat, SystemProbe, TegraStats      # noqa: E402
from retrieval import RetrievalWatch                          # noqa: E402
from roi import RoiConfig                                     # noqa: E402
from site_config import SiteConfig                            # noqa: E402
from test_event_sender import FakeAPI, wait_until, TOKEN      # noqa: E402
from test_roi_followups import FakeAPI2, ROI_DOC, PORT        # noqa: E402

fails = []

SITE_DOC = {"site_id": "SITE-GN-0001", "name": "역삼동 수거함 앞",
            "address": "서울 강남구 역삼동 1-1",
            "location": {"lat": 37.5001, "lng": 127.0362},
            "region": {"sido": "서울특별시", "sigungu": "강남구"},
            "version": 3, "updated_at": "2026-10-05T04:30:00Z", "updated_by": "admin",
            "location_source": "map"}


def check(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


class FakeAPI3(FakeAPI2):
    """FakeAPI2 + 설치 위치(site)."""
    site_doc, site_version = None, 0

    def do_GET(self):                                          # noqa: N802
        if self.path == "/api/device/config" and \
                self.headers.get("Authorization") == f"Bearer {TOKEN}":
            FakeAPI2.config_gets += 1
            return self._reply(200, {"device_id": "JT-GN-0001", "roi": FakeAPI2.roi_doc,
                                     "site": FakeAPI3.site_doc})
        return super().do_GET()

    def do_POST(self):                                         # noqa: N802
        if self.path == "/api/heartbeat":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            FakeAPI.heartbeats.append(json.loads(body))
            return self._reply(200, {"device_id": "JT-GN-0001", "status": "online",
                                     "config": {"roi_version": FakeAPI2.roi_version,
                                                "site_version": FakeAPI3.site_version}})
        return super().do_POST()


def test_site(tmp):
    print("[1] 설치 위치 동기화 (하트비트 -> GET -> site.json -> ack -> 방송 장소명)")
    FakeAPI.heartbeats.clear()
    FakeAPI2.acks.clear()
    FakeAPI2.roi_doc, FakeAPI2.roi_version = ROI_DOC, 4
    FakeAPI3.site_doc, FakeAPI3.site_version = SITE_DOC, 3
    url = f"http://127.0.0.1:{PORT}"
    roi = RoiConfig(tmp / "config" / "roi.json", base_url=url, token=TOKEN, verbose=False)
    names = []
    path = tmp / "config" / "site.json"
    site = SiteConfig(path, base_url=url, token=TOKEN, verbose=False,
                      on_apply=lambda c: names.append(c.name))
    check("파일 없으면 v0 · 미설정", site.version == 0 and not site.loaded)
    hb = Heartbeat(url, TOKEN, "JT-GN-0001",
                   lambda: {"pipeline": {"frames_total": 0, "stream": "ok"}},
                   interval=5, verbose=False, roi=roi, configs=[site], output_path=str(tmp),
                   tegrastats=False)
    check("첫 하트비트 config에 두 버전", wait_until(lambda: FakeAPI.heartbeats, 3)
          and FakeAPI.heartbeats[0].get("config") == {"roi_version": 0, "site_version": 0},
          str(FakeAPI.heartbeats[0].get("config") if FakeAPI.heartbeats else None))
    check("site ack ok", wait_until(lambda: {"site_version": 3, "ok": True} in FakeAPI2.acks, 3),
          str(FakeAPI2.acks))
    check("roi ack도 따로", {"roi_version": 4, "ok": True} in FakeAPI2.acks)
    check("SiteConfig v3 · 이름·좌표", site.version == 3 and site.name == "역삼동 수거함 앞"
          and site.location == (37.5001, 127.0362))
    saved = json.loads(path.read_text(encoding="utf-8"))
    check("site.json에 서버 문서 통째 저장", saved == SITE_DOC)
    check("임시 파일 안 남음", not list(path.parent.glob(".*.tmp")))
    check("방송 장소명 교체 콜백", names == ["역삼동 수거함 앞"], str(names))
    hb.send_once()
    check("다음 하트비트는 site_version=3", FakeAPI.heartbeats[-1]["config"]
          == {"roi_version": 4, "site_version": 3})
    check("output_free_mb 보고", isinstance(FakeAPI.heartbeats[-1]["system"].get("output_free_mb"),
                                          int))

    print("[2] location null -> 이전 좌표 비움 / 잘못된 문서 -> ack error")
    FakeAPI3.site_doc = dict(SITE_DOC, version=4, name="역삼동 2번 수거함", location=None)
    FakeAPI3.site_version = 4
    hb.send_once()
    check("v4 적용 · 좌표 없음", site.version == 4 and site.location is None
          and site.name == "역삼동 2번 수거함")
    check("site.json location null",
          json.loads(path.read_text(encoding="utf-8"))["location"] is None)
    FakeAPI3.site_doc = dict(SITE_DOC, version=5, location={"lat": 200, "lng": 127})
    FakeAPI3.site_version = 5
    hb.send_once()
    ack = FakeAPI2.acks[-1]
    check("ack ok=false + 사유", ack.get("site_version") == 5 and ack.get("ok") is False
          and "lat" in ack.get("error", ""), str(ack))
    check("기존 v4 유지", site.version == 4
          and json.loads(path.read_text(encoding="utf-8"))["version"] == 4)
    hb.close()

    print("[3] 재시작 -> site.json으로 복원 (서버 없이)")
    site2 = SiteConfig(path, verbose=False)
    check("파일에서 v4 복원", site2.loaded and site2.version == 4
          and site2.name == "역삼동 2번 수거함" and site2.location is None)
    check("서버 미연결이면 sync 안 함", site2.sync(9) is False)
    bad = tmp / "config" / "bad_site.json"
    bad.write_text("{broken", encoding="utf-8")
    check("깨진 파일 -> 미설정", not SiteConfig(bad, verbose=False).loaded)


def test_announcer_location():
    print("[4] 방송 장소명 교체")
    try:
        from tts import Announcer, build_phrase
    except Exception as e:                                     # noqa: BLE001
        print(f"  (건너뜀: tts 모듈 없음 {e})")
        return
    ann = Announcer(enabled=False, location="정문", verbose=False)
    ann.set_location("역삼동 수거함 앞")
    check("location 교체", ann.location == "역삼동 수거함 앞")
    check("문구에 새 지점명", build_phrase("검은색", "쓰레기봉투", ann.location)
          .startswith("역삼동 수거함 앞에"))
    ann.set_location("")
    check("빈 이름 -> 장소 없음", ann.location is None)


def test_gpu():
    print("[5] GPU 부하: tegrastats GR3D_FREQ")
    line = ("10-05-2026 13:44:54 RAM 5268/7620MB (lfb 3x4MB) SWAP 167/3810MB CPU [32%@729] "
            "GR3D_FREQ 42% cpu@47.5C")
    check("GR3D_FREQ 파싱", heartbeat._GR3D.search(line).group(1) == "42")
    pr = SystemProbe(tegrastats=True)
    if pr._tegra is not None and pr._tegra.alive:
        time.sleep(2.2)
        n = len(pr._tegra.samples)
        g = pr.sample()["gpu_pct"]
        check("보드 tegrastats 표본 수집", n >= 3 and isinstance(g, int) and 0 <= g <= 100,
              f"표본 {n}개 · gpu_pct={g}")
    else:
        print("  (보드 tegrastats 없음 — 실측 건너뜀)")
    pr.close()
    orig = heartbeat.subprocess.Popen

    def no_tegrastats(*a, **k):
        raise FileNotFoundError("tegrastats")
    heartbeat.subprocess.Popen = no_tegrastats
    try:
        check("tegrastats 없으면 alive=False", not TegraStats().alive)
    finally:
        heartbeat.subprocess.Popen = orig
    seq = iter(["0", "500", "0", "1000"])
    orig_read = heartbeat._read
    heartbeat._read = lambda path, default=None: (next(seq) if path == heartbeat.GPU_LOAD
                                                  else orig_read(path, default))
    try:
        pr = SystemProbe(tegrastats=False)
        for _ in range(3):
            pr.tick()
        check("sysfs 폴백 평균 -> 38%", pr.sample()["gpu_pct"] == 38)
    finally:
        heartbeat._read = orig_read


def test_retrieval_restart(tmp):
    print("[6] 회수 판정 상태 파일 — 재시작 후 이어서")
    st = tmp / "queue" / "retrieval.json"
    box = [100, 100, 200, 200]
    got = []
    w = RetrievalWatch(lambda e, s: got.append((e, s)), window_s=600, confirm_s=60, state_path=st)
    t0 = time.monotonic()
    w.add("E1", box, t0 - 40)                  # 40초 전 사건
    w.update(t0 - 10, [box], [])               # 10초 전까지 보였음
    w.add("E_OLD", box, t0 - 900)              # 판정 시간(600+60초) 지난 사건
    check("상태 파일 기록", st.is_file() and len(json.loads(st.read_text())) == 2)
    w2 = RetrievalWatch(lambda e, s: got.append((e, s)), window_s=600, confirm_s=60,
                        state_path=st)
    check("재시작 후 1건 복원 (지난 건 버림)", w2.n_restored == 1
          and w2.items[0].event_id == "E1", f"{w2.n_restored}")
    t = time.monotonic()
    for _ in range(61):
        t += 1.0
        w2.update(t, [], [])
    check("재시작 후 사라짐 확정 -> after_s는 재시작 전 기준", got and got[0][0] == "E1"
          and 29 <= got[0][1] <= 31, str(got))
    check("확정 후 상태 파일 비움", json.loads(st.read_text()) == [])


def test_disk_low(tmp):
    print("[7] disk_low 기준 2000MB")
    hb = Heartbeat("http://127.0.0.1:1", TOKEN, "JT-GN-0001", lambda: {}, interval=3600,
                   verbose=False, tegrastats=False)
    hb.close()
    orig = hb.probe.sample
    for free, expect in ((1999, True), (2000, False)):
        hb.probe.sample = lambda free=free: dict(orig(), disk_free_mb=free)
        check(f"disk_free_mb={free} -> disk_low {'있음' if expect else '없음'}",
              ("disk_low" in hb.build()["issues"]) == expect)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), FakeAPI3)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="cg_site_"))
    test_site(tmp)
    test_announcer_location()
    test_gpu()
    test_retrieval_restart(tmp)
    test_disk_low(tmp)
    srv.shutdown()
    print(f"\n{'모두 통과' if not fails else f'실패 {len(fails)}건: {fails}'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
