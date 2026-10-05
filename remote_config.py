#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""서버가 버전으로 관리하는 원격 설정의 공통 흐름 — ROI(roi.py)·설치 위치(site.py).

  1) 하트비트 본문 config.<KEY>_version = 지금 적용 중인 버전 (처음 0)
  2) 하트비트 응답 config.<KEY>_version 이 더 크면 GET /api/device/config 로 받아 온다
  3) 응답의 <KEY> 객체 검증 -> config/<KEY>.json 에 원자적 저장(임시 파일 -> rename) -> 즉시 적용
  4) POST /api/device/config/ack {<KEY>_version, ok[, error]} — 실패 사유는 대시보드에 그대로 표시

- 서버가 안 돼도 마지막 config/<KEY>.json 으로 계속 동작한다.
- 같은 버전이 검증에 실패하면 다시 받지 않는다 (30초마다 같은 오류 ack 도배 방지).
- 하위 클래스는 KEY·LABEL과 _parse(doc)(검증 실패 시 ValueError)·_set(parsed)만 정의한다.
"""
import json
import threading
import urllib.request
from pathlib import Path

HTTP_TIMEOUT = 10.0


class RemoteConfig:
    KEY = ""                    # /api/device/config 응답의 객체 이름 (roi / site)
    LABEL = ""                  # 로그 머리말
    MISSING = "기본값"           # 파일이 없거나 깨졌을 때의 동작 (로그용)

    def __init__(self, path, base_url=None, token=None, verbose=True, on_apply=None):
        self.path = Path(path)
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token
        self.verbose = verbose
        self.on_apply = on_apply            # 서버 설정을 새로 적용할 때마다 on_apply(self)
        self.version = 0
        self.generation = 0                 # 적용할 때마다 +1 — 감시 루프가 바뀐 것을 알아채는 용도
        self.last_error = None
        self._failed_version = None
        self._lock = threading.Lock()       # sync() 중복 실행 방지
        self.load()

    @property
    def version_key(self):
        return f"{self.KEY}_version"

    # ------------------------------------------------------------ 로컬
    def load(self):
        """config/<KEY>.json을 읽어 적용. 없거나 깨졌으면 기본값(version 0)."""
        if not self.path.is_file():
            return False
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            self._apply(self._parse(doc), int(doc.get("version") or 0))
            return True
        except (OSError, ValueError, TypeError, AttributeError) as e:
            self._log(f"{self.path.name} 읽기 실패({e}) -> {self.MISSING}")
            return False

    def _parse(self, doc):
        """서버/파일 문서 -> 적용할 값. 규칙 위반이면 ValueError(대시보드에 표시될 사유)."""
        raise NotImplementedError

    def _set(self, parsed):
        raise NotImplementedError

    def _apply(self, parsed, version):
        self._set(parsed)
        self.version = int(version)
        self.generation += 1

    def _save(self, doc):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def describe(self):
        return f"v{self.version}"

    # ------------------------------------------------------------ 서버
    def _request(self, method, path, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            raw = r.read()
        return json.loads(raw) if raw else {}

    def _ack(self, version, error=None):
        body = {self.version_key: int(version), "ok": error is None}
        if error is not None:
            body["error"] = str(error)[:200]
        try:
            self._request("POST", "/api/device/config/ack", body)
        except Exception as e:                               # noqa: BLE001 — 다음 하트비트가 대신 알린다
            self._log(f"ack 전송 실패({type(e).__name__}) — 하트비트로 대체")

    def sync(self, server_version):
        """하트비트 응답의 config.<KEY>_version을 받아 필요하면 내려받아 적용한다.

        하트비트 스레드에서 부른다. 예외를 밖으로 내지 않는다.
        """
        try:
            server_version = int(server_version)
        except (TypeError, ValueError):
            return False
        if (self.base_url is None or server_version <= self.version
                or server_version == self._failed_version):
            return False
        if not self._lock.acquire(blocking=False):
            return False
        try:
            try:
                resp = self._request("GET", "/api/device/config")
            except Exception as e:                           # noqa: BLE001 — 회선 문제: 다음 주기에 재시도
                self._log(f"v{server_version} 받기 실패({type(e).__name__}: {str(e)[:60]}) "
                          "— 다음 하트비트에 재시도")
                return False
            doc = (resp or {}).get(self.KEY)
            version = server_version
            try:
                if not isinstance(doc, dict):
                    raise ValueError(f"{self.KEY} missing in /api/device/config")
                version = int(doc.get("version") or server_version)
                parsed = self._parse(doc)                    # 검증 -> 저장 -> 적용 순서
                self._save(doc)
                self._apply(parsed, version)
            except (ValueError, TypeError, OSError, AttributeError) as e:
                self._failed_version = server_version
                self.last_error = str(e)
                self._log(f"v{version} 적용 실패: {e} -> 기존 v{self.version} 유지")
                self._ack(version, error=e)
                return False
            self.last_error = None
            self._failed_version = None
            self._log(f"서버 설정 적용: {self.describe()}")
            self._ack(version)
            if self.on_apply is not None:
                try:
                    self.on_apply(self)
                except Exception as e:                       # noqa: BLE001
                    self._log(f"적용 후 처리 실패({type(e).__name__}: {e})")
            return True
        finally:
            self._lock.release()

    def _log(self, msg):
        if self.verbose:
            print(f"[{self.LABEL}] {msg}", flush=True)


__all__ = ["RemoteConfig"]
