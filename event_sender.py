#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""이벤트 서버 전송 (--report) — jetson_handoff.md §3 계약.

감시 루프는 submit()으로 이벤트를 디스크 큐(queue/events/*.json)에 넣고 즉시 돌아온다.
전송은 백그라운드 스레드가 맡는다:

  1) POST {API_URL}/api/events            (JSON, Bearer 장비토큰)
  2) POST {API_URL}/api/events/{id}/media (multipart: 스냅샷/원본/클립)
  3) 둘 다 성공하면 큐 파일 삭제
  4) media 응답의 files.<필드>.size·sha256이 로컬 파일과 같으면 그 로컬 파일을 지운다 (2026-10-05:
     젯슨 저장 공간 부족). 응답이 없거나 다르면 지우지 않고 그 파일만 다시 올린다 — 항목별
     지수 백오프(5초 -> 5분). 확인 안 된 미디어가 뒤의 사건·후속 보고를 막지는 않는다.
     확인 대상은 VERIFY_FIELDS(스냅샷·원본·클립·방송 음성 wav). 그 밖의 필드는 서버가 확인해
     주면 지우고, 언급이 없으면 업로드 성공으로 치고 로컬 파일은 보존 기간 삭제에 맡긴다.

방송 음성(announce_audio, wav)·전후 클립(clip, mp4)처럼 이벤트보다 늦게 만들어지는 파일은
attach()로 같은 event_id에 media 업로드만 따로 큐에 넣는다 (JSON은 다시 보내지 않는다).
회수 후속 보고는 update()로 넣는다 -> POST /api/events/{id}/update (§3.2).

- 실패·5xx·회선 끊김: 지수 백오프 5초 → 최대 5분, 오래된 것부터 무한 재시도.
  재전송 시 event_id가 같아 서버가 중복 제거한다 (200 {"duplicate":true} / 409도 성공).
- 401(토큰 오류): 보관한 채 로그만 남기고 계속 재시도 (§3.5).
- 400/403/413/415/422(계약 위반·용량 초과·형식 거절 — 재시도해도 똑같음): queue/events/bad/로
  치워 큐를 막지 않는다.
- 큐 상한(기본 5,000건) 또는 디스크가 거의 찼을 때(기본 200MB 미만) 가장 오래된 것부터 삭제하고
  issues()에 "queue_full"을 올린다 (하트비트 연계용). 이 하한은 출력물 회수 기준(--min-free-mb,
  2000MB)과 다르다 — 같게 두면 여유 2000MB 미만 보드에서 서버가 끊길 때마다 사건이 버려진다
  (2026-10-24, 2번 보드 여유 293MB). 디스크 확보는 OutputSpace가 출력물을 지워서 한다.
- pending_media(): 아직 서버 확인 전인 미디어 경로 — OutputSpace가 이것은 지우지 않는다.

외부 의존성 없음 (urllib). HTTPS 인증서는 시스템 CA를 쓴다.
"""
import hashlib
import json
import os
import random
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

BACKOFF_START = 5.0
BACKOFF_MAX = 300.0
HTTP_TIMEOUT = 15.0
MEDIA_TYPES = {".mp4": "video/mp4", ".wav": "audio/wav", ".mp3": "audio/mpeg",
               ".jpg": "image/jpeg"}
# 서버 확인 전엔 절대 안 지우는 파일 (announce_audio: 2026-10-05 작업 7 — WAV, 확인 후 삭제)
VERIFY_FIELDS = ("snapshot", "snapshot_raw", "clip", "announce_audio")
QUEUE_MIN_FREE_MB = 200.0   # 이 아래면 큐도 오래된 것부터 버린다 (디스크가 꽉 차 아무것도 못 쓰기 직전)
DEFERRED = "deferred"       # _send_one 결과: 서버는 받았지만 미디어 확인 실패 -> 이 항목만 나중에 재시도

# ---------------------------------------------------------------- ULID
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"          # Crockford


def ulid(ts_ms=None) -> str:
    """26자 ULID — 시간순 정렬 가능, 오프라인 중 생성해도 충돌 없음."""
    ts = int(time.time() * 1000) if ts_ms is None else int(ts_ms)
    rnd = int.from_bytes(os.urandom(10), "big")
    n = (ts << 80) | rnd
    return "".join(_B32[(n >> (5 * i)) & 31] for i in range(25, -1, -1))


def now_iso() -> str:
    """ISO 8601 + 로컬 오프셋 (예: 2026-09-24T14:02:41+09:00). 오프셋 필수 (§4)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- multipart
def _multipart(files):
    """{필드: 파일경로} -> (body bytes, content-type). 스냅샷 수 KB라 메모리 조립로 충분."""
    boundary = "----cleanguard-" + uuid.uuid4().hex
    parts = []
    for field, path in files.items():
        p = Path(path)
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{field}"; filename="{p.name}"\r\n'
            f"Content-Type: {MEDIA_TYPES.get(p.suffix.lower(), 'image/jpeg')}"
            "\r\n\r\n".encode()
            + p.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def file_digest(path):
    """(크기, sha256 hex)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def server_matches(info, size, sha):
    """media 응답의 files.<필드> 정보가 로컬 파일(size, sha256)과 같은가."""
    if not isinstance(info, dict):
        return False
    try:
        return (int(info.get("size")) == size
                and str(info.get("sha256", "")).strip().lower() == sha)
    except (TypeError, ValueError):
        return False


class EventSender:
    """디스크 큐 + 백그라운드 재전송. 감시 루프를 절대 막지 않는다."""

    def __init__(self, base_url, token, device_id, site_id, queue_dir,
                 max_queue=5000, min_free_mb=QUEUE_MIN_FREE_MB):
        self.base_url = base_url.rstrip("/")
        self.token, self.device_id, self.site_id = token, device_id, site_id
        self.queue_dir = Path(queue_dir)
        self.bad_dir = self.queue_dir / "bad"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.max_queue, self.min_free_mb = int(max_queue), float(min_free_mb)
        self.sent = self.dropped = self.rejected = 0
        self.media_deleted = self.media_unverified = 0
        self.queue_full = False
        self._retry = {}            # 큐 파일명 -> (시도 횟수, 다음 시도 monotonic) — 미디어 확인 실패분
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    # ------------------------------------------------------------ 큐 넣기
    def submit(self, event: dict, media: dict | None = None) -> str:
        """이벤트를 큐에 쓰고 event_id를 돌려준다. media={필드: 로컬 경로}."""
        event_id = event.get("event_id") or ulid()
        record = dict(event, event_id=event_id,
                      device_id=self.device_id, site_id=self.site_id)
        record.setdefault("ts", now_iso())
        # media 필드는 파일명만 (§3.3 — 파일은 JSON 성공 후 별도 업로드)
        media = {k: str(v) for k, v in (media or {}).items() if v and Path(v).is_file()}
        record.setdefault("media", {})
        for k, p in media.items():
            record["media"].setdefault(k, Path(p).name)
        record["_upload"] = media          # 로컬 상태 (서버 전송 전에 뗀다)
        record["_json_sent"] = False
        self._prune()
        tmp = self.queue_dir / f".{event_id}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.rename(self.queue_dir / f"{event_id}.json")     # 원자적 쓰기 (§3.5)
        self._wake.set()
        return event_id

    def attach(self, event_id: str, media: dict) -> bool:
        """이미 submit()한 이벤트에 파일을 추가 업로드한다 (예: 합성이 끝난 방송 음성).

        큐 파일명 '{event_id}~{필드}.json'은 ULID 정렬상 원 이벤트 바로 뒤에 온다
        ('~'가 '.'보다 크다). 원 이벤트 JSON이 막혀 있으면 그 뒤에서 함께 기다린다.
        """
        media = {k: str(v) for k, v in (media or {}).items() if v and Path(v).is_file()}
        if not media:
            return False
        record = {"event_id": event_id, "device_id": self.device_id,
                  "site_id": self.site_id, "_upload": media, "_json_sent": True}
        self._prune()
        name = f"{event_id}~{'+'.join(sorted(media))}"
        tmp = self.queue_dir / f".{name}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.rename(self.queue_dir / f"{name}.json")
        self._wake.set()
        return True

    def update(self, event_id: str, update: str, **fields) -> bool:
        """이미 submit()한 이벤트의 후속 보고 (§3.2 event_update). 예: update("retrieved", after_s=312).

        큐 파일명 '{event_id}~update.json' — 원 이벤트 뒤에 정렬되어 JSON이 먼저 나간다.
        """
        record = dict(fields, event_id=event_id, device_id=self.device_id,
                      ts=now_iso(), update=update, _kind="update")
        self._prune()
        name = f"{event_id}~update-{update}"
        tmp = self.queue_dir / f".{name}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.rename(self.queue_dir / f"{name}.json")
        self._wake.set()
        return True

    def pending(self) -> int:
        return len(list(self.queue_dir.glob("*.json")))

    def pending_media(self):
        """큐에 남은(서버 확인 전) 미디어 파일 경로 집합 (resolve된 Path)."""
        out = set()
        for qfile in self.queue_dir.glob("*.json"):
            try:
                upload = json.loads(qfile.read_text(encoding="utf-8")).get("_upload") or {}
            except (OSError, ValueError, AttributeError):
                continue
            for path in upload.values():
                out.add(Path(path).resolve())
        return out

    def issues(self):
        return ["queue_full"] if self.queue_full else []

    def close(self, timeout=5.0):
        self._stop.set()
        self._wake.set()
        self._th.join(timeout)
        # 종료 직전 최종 드레인 1회 — 짧게 돌다 꺼져도 서버가 살아 있으면 다 나간다.
        # 실패하면 큐에 남아 다음 실행 때 재전송된다.
        for qfile in sorted(self.queue_dir.glob("*.json")):
            if not self._send_one(qfile):
                break

    # ------------------------------------------------------------ 내부
    def _prune(self):
        """큐 상한·디스크 여유 확인. 넘치면 가장 오래된 것부터 버린다 (§3.5)."""
        files = sorted(self.queue_dir.glob("*.json"))
        free_mb = shutil.disk_usage(self.queue_dir).free / 1e6
        over = (len(files) - self.max_queue + 1) if len(files) >= self.max_queue else 0
        if free_mb < self.min_free_mb:
            over = max(over, 1)
        for p in files[:over]:
            p.unlink(missing_ok=True)
            self.dropped += 1
        self.queue_full = self.queue_full or over > 0

    def _post(self, path, data, content_type):
        req = urllib.request.Request(
            self.base_url + path, data=data, method="POST",
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status, r.read()

    def _verify_media(self, files, raw):
        """media 응답 확인. 확인된 로컬 파일은 지우고, 다시 올려야 할 {필드: 경로}를 돌려준다."""
        try:
            resp = json.loads(raw) if raw else {}
        except ValueError:
            resp = {}
        got = resp.get("files") if isinstance(resp, dict) else None
        got = got if isinstance(got, dict) else {}
        retry = {}
        for field, path in files.items():
            try:
                size, sha = file_digest(path)
            except OSError:
                continue                                     # 그새 지워짐 — 다시 올릴 수 없다
            if server_matches(got.get(field), size, sha):
                try:
                    os.unlink(path)
                    self.media_deleted += 1
                except OSError:
                    pass
            elif field in VERIFY_FIELDS:
                retry[field] = path
        return retry

    def _send_one(self, qfile: Path):
        """큐 파일 하나 처리. 완료(삭제/격리)하면 True, 재시도 필요하면 False(큐 전체 대기),
        서버가 미디어를 받았다는 확인이 없으면 DEFERRED(이 항목만 백오프 후 재시도)."""
        try:
            rec = json.loads(qfile.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            qfile.unlink(missing_ok=True)                    # 깨진 파일
            return True
        upload = rec.pop("_upload", {})
        json_sent = rec.pop("_json_sent", False)
        kind = rec.pop("_kind", "event")
        try:
            if kind == "update":
                self._post(f"/api/events/{rec['event_id']}/update",
                           json.dumps(rec, ensure_ascii=False).encode(), "application/json")
            elif not json_sent:
                body = json.dumps(rec, ensure_ascii=False).encode()
                self._post("/api/events", body, "application/json")
                # 성공 표시를 남겨 재시도 시 JSON을 또 보내지 않는다
                #(서버는 중복 제거하지만 회선을 아낀다)
                self._rewrite(qfile, dict(rec, _upload=upload, _json_sent=True))
            files = {k: p for k, p in upload.items() if Path(p).is_file()}
            if files:
                body, ctype = _multipart(files)
                _, raw = self._post(f"/api/events/{rec['event_id']}/media", body, ctype)
                retry = self._verify_media(files, raw)
                if retry:
                    return self._defer(qfile, rec, retry)
            qfile.unlink(missing_ok=True)
            self._retry.pop(qfile.name, None)
            self.sent += 1
            return True
        except urllib.error.HTTPError as e:
            if e.code in (200, 409):                         # 중복 — 성공 취급 (§3.5)
                qfile.unlink(missing_ok=True)
                self.sent += 1
                return True
            if e.code == 401:                                # 토큰 문제 — 보관·재시도
                print(f"[전송] 401 토큰 오류 — {qfile.name} 보관 후 재시도", flush=True)
                return False
            if e.code in (400, 403, 413, 415, 422) or (e.code == 404 and kind == "update"):
                # 계약 위반(또는 서버에 없는 사건의 후속 보고) — 재시도해도 같다 -> 격리
                self.bad_dir.mkdir(exist_ok=True)
                qfile.rename(self.bad_dir / qfile.name)
                self.rejected += 1
                print(f"[전송] {e.code} 거부 — {qfile.name} -> bad/ 격리", flush=True)
                return True
            print(f"[전송] HTTP {e.code} — 재시도 예정", flush=True)
            return False
        except Exception as e:                               # noqa: BLE001 — DNS/타임아웃 등
            print(f"[전송] 실패({type(e).__name__}: {str(e)[:60]}) — 재시도 예정", flush=True)
            return False

    @staticmethod
    def _rewrite(qfile, record):
        """큐 파일 원자적 갱신 (임시 파일 -> rename)."""
        tmp = qfile.with_name(f".{qfile.stem}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(qfile)

    def _defer(self, qfile, rec, retry):
        """확인 안 된 미디어만 남겨 큐 파일을 고쳐 쓰고 이 항목의 다음 시도 시각을 정한다."""
        n = self._retry.get(qfile.name, (0, 0.0))[0] + 1
        wait = min(BACKOFF_START * 2 ** (n - 1), BACKOFF_MAX)
        self._retry[qfile.name] = (n, time.monotonic() + wait)
        self.media_unverified += 1
        self._rewrite(qfile, dict(rec, _upload=retry, _json_sent=True))
        print(f"[전송] 미디어 수신 확인 안 됨({'+'.join(sorted(retry))}) — 로컬 보관, "
              f"{wait:.0f}초 뒤 재전송 ({n}회째)", flush=True)
        return DEFERRED

    def _loop(self):
        backoff = BACKOFF_START
        while not self._stop.is_set():
            files = sorted(self.queue_dir.glob("*.json"))    # ULID 이름 = 오래된 순
            if not files:                                    # 빈 큐 — submit까지 잠든다
                self.queue_full = False
                backoff = BACKOFF_START
                self._retry.clear()
                self._wake.clear()
                if self.pending():                           # glob과 clear 사이에 들어온 것 — 놓치지 않게
                    continue
                self._wake.wait()
                continue
            progressed, blocked, next_due = False, False, None
            now = time.monotonic()
            for qfile in files:
                if self._stop.is_set():
                    return
                due = self._retry.get(qfile.name, (0, 0.0))[1]
                if due > now:                                # 미디어 재전송 대기 중 — 건너뛴다
                    next_due = due if next_due is None else min(next_due, due)
                    continue
                r = self._send_one(qfile)
                if r is DEFERRED:
                    due = self._retry[qfile.name][1]
                    next_due = due if next_due is None else min(next_due, due)
                elif r:
                    progressed = True
                else:
                    blocked = True
                    break                                    # 순서 보장 — 막힌 자리에서 대기
            if progressed and self.pending() == 0:
                self.queue_full = False
            self._wake.clear()
            if blocked:                                      # 회선·서버 문제 — 큐 전체 백오프
                self._wake.wait(backoff * (1 + 0.1 * random.random()))
                backoff = min(backoff * 2, BACKOFF_MAX)
                continue
            backoff = BACKOFF_START
            if next_due is not None:                         # 남은 건 미디어 재전송 대기분뿐
                self._wake.wait(max(0.05, next_due - time.monotonic()))
            elif not self.pending():
                self._wake.wait()
