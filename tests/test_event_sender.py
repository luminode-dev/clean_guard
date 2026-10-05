#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""이벤트 전송(event_sender.py) 검증 — 서버 없이 로컬 가짜 API로.

1) submit -> POST /api/events(JSON) -> /media(multipart) -> 큐 파일 삭제
2) 서버 다운 상태에서 쌓인 큐 -> 서버 복구 후 오래된 순 재전송 (유실 없음)
3) 200 duplicate / 409 -> 성공 처리
4) 422(계약 위반) -> bad/ 격리, 큐를 막지 않음
5) 큐 상한 -> 가장 오래된 것부터 폐기 + queue_full 보고
6) ULID: 26자·시간순 정렬
7) attach(): 늦게 생긴 방송 mp3를 같은 event_id로 media만 추가 업로드 (JSON 재전송 없음)
9) media 응답 files.<필드>.size·sha256이 로컬과 같을 때만 로컬 파일 삭제 (2026-10-05)
10) 다르거나 확인이 없으면 로컬 보관 + 그 파일만 백오프 재전송, 뒤의 사건은 막지 않음
11) 방송 음성 wav도 확인 대상: 확인 없으면 로컬 보관·재전송, 확인되면 삭제 / 415 -> bad/ 격리
12) 401 -> 큐 유지
"""
import email.parser
import email.policy
import hashlib
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import event_sender                                    # noqa: E402
from event_sender import EventSender, ulid             # noqa: E402

event_sender.BACKOFF_START = 0.2                       # 테스트는 기다릴 수 없다
event_sender.BACKOFF_MAX = 0.5

PORT = 18010
TOKEN = "cg_testtoken"
fails = []


def check(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


class FakeAPI(BaseHTTPRequestHandler):
    """수신 내용을 클래스 변수에 쌓는 가짜 관제 서버."""
    events, media, mode = [], [], "ok"    # mode: ok | down | duplicate | reject
    media_mode = "verify"                 # verify(size·sha256 회신) | nofiles | mismatch | wav415
    heartbeats = []

    def do_POST(self):                                     # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._reply(401, {"error": "bad token"})
        if FakeAPI.mode == "down":
            return self._reply(500, {"error": "down"})
        if self.path == "/api/heartbeat":
            FakeAPI.heartbeats.append(json.loads(body))
            return self._reply(200, {"ok": True})
        if self.path == "/api/events":
            if FakeAPI.mode == "reject":
                return self._reply(422, {"error": "detection.class 누락"})
            ev = json.loads(body)
            if FakeAPI.mode == "duplicate" or any(
                    e["event_id"] == ev["event_id"] for e in FakeAPI.events):
                return self._reply(200, {"duplicate": True})
            FakeAPI.events.append(ev)
            return self._reply(201, {"ok": True})
        if self.path.endswith("/media"):
            FakeAPI.media.append((self.path.split("/")[3], body))
            if FakeAPI.media_mode == "wav415":
                return self._reply(415, {"error": "announce_audio must be WAV"})
            if FakeAPI.media_mode == "nofiles":
                return self._reply(200, {"snapshot": "/snapshots/x.jpg"})
            files = {}
            for field, data in parse_multipart(body, self.headers["Content-Type"]).items():
                sha = hashlib.sha256(data).hexdigest()
                if FakeAPI.media_mode == "mismatch":
                    sha = "0" * 64
                files[field] = {"size": len(data), "sha256": sha, "path": f"/x/{field}"}
            return self._reply(200, {"ok": True, "files": files})
        self._reply(404, {})

    def _reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                             # 테스트 출력 오염 방지
        pass


def parse_multipart(body, ctype):
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {ctype}\r\n\r\n".encode() + body)
    return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in msg.iter_parts()}


def fake_file(path, data=b"\xff\xd8fakejpeg\xff\xd9"):
    Path(path).write_bytes(data)
    return Path(path)


def wait_until(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def make_sender(qdir, **kw):
    return EventSender(f"http://127.0.0.1:{PORT}", TOKEN, "JT-GN-0001",
                       "SITE-GN-0001", qdir, **kw)


def sample_event(seq=1):
    return {"detection": {"class": "쓰레기봉투", "class_id": 0, "color": "검은색",
                          "night": None, "conf": 0.75, "bbox": [10, 20, 30, 40],
                          "frame_size": [2560, 1440]},
            "suspect": {"obj_id": seq, "owner_pid": 4, "owner_matched": False,
                        "person_bbox_at_drop": [1, 2, 3, 4]},
            "announce": {"played": True, "phrase": "…", "suppressed_reason": None},
            "debug": {"frame": 100 + seq, "local_seq": seq, "model_format": "engine"}}


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="cg_evq_"))

    print("[1] ULID")
    ids = [ulid(ts_ms=1000 + i) for i in range(3)]
    check("26자", all(len(u) == 26 for u in ids))
    check("시간순 정렬", ids == sorted(ids))

    print("[2] 정상 전송 (JSON -> multipart -> 큐 삭제)")
    snap = tmp / "event_0001.jpg"
    snap.write_bytes(b"\xff\xd8fakejpeg\xff\xd9")
    s = make_sender(tmp / "q1")
    eid = s.submit(sample_event(1), media={"snapshot": snap, "snapshot_raw": None})
    check("event_id는 ULID", len(eid) == 26)
    check("JSON 도착", wait_until(lambda: len(FakeAPI.events) == 1))
    ev = FakeAPI.events[0]
    check("장치 식별자 부여", ev["device_id"] == "JT-GN-0001" and ev["site_id"] == "SITE-GN-0001")
    check("ts 오프셋 포함", ev["ts"][-6] in "+-" or ev["ts"].endswith("Z"), ev["ts"])
    check("media는 파일명만", ev["media"] == {"snapshot": "event_0001.jpg"})
    check("내부 상태 필드 미전송", "_upload" not in ev and "_json_sent" not in ev)
    check("multipart 도착", wait_until(lambda: len(FakeAPI.media) == 1))
    mid, mbody = FakeAPI.media[0]
    check("media가 event_id로 감", mid == eid)
    check("파일 내용 포함", b"fakejpeg" in mbody and b'name="snapshot"' in mbody)
    check("큐 비움", wait_until(lambda: s.pending() == 0))
    s.close()

    print("[3] 서버 다운 -> 큐 보관 -> 복구 후 오래된 순 재전송")
    FakeAPI.mode = "down"
    s = make_sender(tmp / "q2")
    for i in range(3):
        s.submit(sample_event(i + 1))
        time.sleep(0.01)                       # ULID 순서 보장
    time.sleep(0.5)
    check("전송 안 됨·큐 보관", s.pending() == 3 and len(FakeAPI.events) == 1)
    FakeAPI.mode = "ok"
    check("복구 후 전부 전송", wait_until(lambda: len(FakeAPI.events) == 4))
    seqs = [e["debug"]["local_seq"] for e in FakeAPI.events[1:]]
    check("오래된 순", seqs == [1, 2, 3], str(seqs))
    check("큐 비움", wait_until(lambda: s.pending() == 0))
    s.close()

    print("[4] 중복(200 duplicate) = 성공 처리")
    FakeAPI.mode = "duplicate"
    s = make_sender(tmp / "q3")
    s.submit(sample_event(9))
    check("큐에서 제거", wait_until(lambda: s.pending() == 0))
    check("sent 집계", s.sent == 1)
    s.close()
    FakeAPI.mode = "ok"

    print("[5] 422 -> bad/ 격리 (큐 안 막힘)")
    FakeAPI.mode = "reject"
    s = make_sender(tmp / "q4")
    s.submit(sample_event(1))
    check("bad/로 이동", wait_until(lambda: len(list((tmp / "q4" / "bad").glob("*.json"))) == 1))
    check("큐 비움·rejected 집계", s.pending() == 0 and s.rejected == 1)
    s.close()
    FakeAPI.mode = "ok"

    print("[6] 큐 상한 -> 오래된 것 폐기 + queue_full")
    FakeAPI.mode = "down"
    s = make_sender(tmp / "q5", max_queue=3)
    for i in range(5):
        s.submit(sample_event(i + 1))
        time.sleep(0.01)
    check("상한 유지", s.pending() <= 3, f"pending={s.pending()}")
    check("폐기 집계·issues", s.dropped >= 2 and s.issues() == ["queue_full"])
    s.close()
    FakeAPI.mode = "ok"

    print("[7] attach — 방송 mp3 추가 업로드")
    FakeAPI.events.clear(); FakeAPI.media.clear()
    fake_file(snap)
    mp3 = tmp / "event_0001.mp3"
    mp3.write_bytes(b"ID3fakemp3")
    s = make_sender(tmp / "q6")
    eid = ulid()
    s.submit(dict(sample_event(1), event_id=eid), media={"snapshot": snap})
    check("없는 파일은 attach 거부", s.attach(eid, {"announce_audio": tmp / "none.mp3"}) is False)
    check("attach 큐 등록", s.attach(eid, {"announce_audio": mp3}))
    check("media 2회 도착", wait_until(lambda: len(FakeAPI.media) == 2))
    check("JSON은 1회만", len(FakeAPI.events) == 1)
    amid, abody = FakeAPI.media[-1]
    check("같은 event_id", amid == eid)
    check("필드명·MIME", b'name="announce_audio"' in abody and b"audio/mpeg" in abody
          and b"ID3fakemp3" in abody)
    check("큐 비움", wait_until(lambda: s.pending() == 0))
    s.close()

    print("[8] attach — 서버 다운 중이면 원 이벤트 뒤에서 대기, 복구 후 순서대로")
    FakeAPI.events.clear(); FakeAPI.media.clear()
    FakeAPI.mode = "down"
    fake_file(snap)
    fake_file(mp3, b"ID3fakemp3")
    s = make_sender(tmp / "q7")
    eid = ulid()
    s.submit(dict(sample_event(2), event_id=eid), media={"snapshot": snap})
    s.attach(eid, {"announce_audio": mp3})
    names = sorted(p.name for p in (tmp / "q7").glob("*.json"))
    check("큐 정렬: 이벤트 -> 음성", names == [f"{eid}.json", f"{eid}~announce_audio.json"],
          str(names))
    FakeAPI.mode = "ok"
    check("복구 후 JSON 1 + media 2", wait_until(
        lambda: len(FakeAPI.events) == 1 and len(FakeAPI.media) == 2))
    check("스냅샷 먼저, 음성 나중", b'name="snapshot"' in FakeAPI.media[0][1]
          and b'name="announce_audio"' in FakeAPI.media[1][1])
    s.close()

    print("[9] 서버 size·sha256 확인 -> 로컬 삭제")
    FakeAPI.events.clear(); FakeAPI.media.clear()
    snap, clip = fake_file(tmp / "e9.jpg"), fake_file(tmp / "e9.mp4", b"fakemp4" * 1000)
    s = make_sender(tmp / "q9")
    eid = s.submit(sample_event(9), media={"snapshot": snap})
    s.attach(eid, {"clip": clip})
    check("큐 비움", wait_until(lambda: s.pending() == 0))
    check("확인된 스냅샷·클립 로컬 삭제", not snap.exists() and not clip.exists()
          and s.media_deleted == 2, f"deleted={s.media_deleted}")
    s.close()

    print("[10] 확인 불일치 -> 로컬 보관·그 파일만 백오프 재전송, 뒤 사건은 진행")
    FakeAPI.events.clear(); FakeAPI.media.clear()
    FakeAPI.media_mode = "mismatch"
    snap = fake_file(tmp / "e10.jpg")
    s = make_sender(tmp / "q10")
    eid1 = s.submit(sample_event(10), media={"snapshot": snap})
    time.sleep(0.01)
    s.submit(sample_event(11))
    check("뒤 사건 JSON은 전송됨", wait_until(lambda: len(FakeAPI.events) == 2))
    check("불일치 스냅샷은 로컬 보관", snap.exists())
    check("큐에 미디어 1건 남음", wait_until(lambda: s.pending() == 1)
          and (tmp / "q10" / f"{eid1}.json").exists())
    check("재전송 반복 (백오프)", wait_until(lambda: len(FakeAPI.media) >= 3, 4.0),
          f"media {len(FakeAPI.media)}회")
    rec = json.loads((tmp / "q10" / f"{eid1}.json").read_text(encoding="utf-8"))
    check("JSON 재전송 안 함", rec.get("_json_sent") is True and len(FakeAPI.events) == 2)
    FakeAPI.media_mode = "nofiles"
    n = len(FakeAPI.media)
    check("확인 없는 응답도 보관 유지", wait_until(lambda: len(FakeAPI.media) > n, 4.0)
          and snap.exists() and s.pending() == 1)
    FakeAPI.media_mode = "verify"
    check("확인되면 삭제·큐 비움", wait_until(lambda: s.pending() == 0, 6.0)
          and not snap.exists())
    s.close()

    print("[11] 방송 음성 wav — 확인 없으면 보관·재전송, 확인되면 삭제, 415면 격리")
    FakeAPI.events.clear(); FakeAPI.media.clear()
    FakeAPI.media_mode = "nofiles"
    wav = fake_file(tmp / "e11.wav", b"RIFF\x24\x00\x00\x00WAVEfmt fakewav")
    s = make_sender(tmp / "q11")
    eid = s.submit(sample_event(12))
    s.attach(eid, {"announce_audio": wav})
    check("확인 없으면 음성 로컬 보관·큐 유지", wait_until(
        lambda: len(FakeAPI.media) >= 2, 3) and wav.exists() and s.pending() == 1)
    check("MIME audio/wav", any(b'name="announce_audio"' in m[1] and b"audio/wav" in m[1]
                                 for m in FakeAPI.media))
    FakeAPI.media_mode = "verify"
    check("확인되면 음성 삭제·큐 비움", wait_until(lambda: s.pending() == 0, 6.0)
          and not wav.exists())
    s.close()
    FakeAPI.media_mode = "wav415"
    wav = fake_file(tmp / "e11b.wav", b"not really wav")
    s = make_sender(tmp / "q11b")
    s.attach(ulid(), {"announce_audio": wav})
    check("415 -> bad/ 격리 (큐 안 막힘)", wait_until(
        lambda: len(list((tmp / "q11b" / "bad").glob("*.json"))) == 1) and s.pending() == 0)
    s.close()
    FakeAPI.media_mode = "verify"

    print("[12] 401 -> 큐 유지")
    s = EventSender(f"http://127.0.0.1:{PORT}", "wrong", "JT-GN-0001", "SITE-GN-0001",
                    tmp / "q12")
    s.submit(sample_event(13))
    time.sleep(0.6)
    check("큐 유지", s.pending() == 1)
    s._stop.set(); s._wake.set(); s._th.join(2)

    srv.shutdown()
    print(f"\n{'실패: ' + ', '.join(fails) if fails else '전체 통과'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
