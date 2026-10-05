#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""설치 위치 원격 설정 -> config/site.json (2026-10-05 작업 요청 §2, ROI와 같은 흐름: remote_config.py).

대시보드에서 운영자가 지점명·주소·좌표를 저장하면 서버가 site_version을 올린다.
하트비트 응답으로 알아채 GET /api/device/config 의 site 객체를 받아 저장하고 ack한다.

  site = {"site_id", "name", "address", "location": {"lat", "lng"} | null, "region": {...},
          "version", "updated_at", "updated_by", "location_source"}

- 방송 문구의 장소명은 name을 쓴다 (예전 .env CAM_LOCATION / --tts-location 대체).
  site.json이 아직 없을 때(서버에서 한 번도 못 받음)만 그 값으로 대신한다.
- 서버 문서를 통째로 저장한다 — location이 null이면 이전 좌표를 남기지 않고 비운다.
- 서버에 연결이 안 돼도 마지막 site.json으로 동작한다.
(파일명이 site.py면 파이썬 표준 모듈 site를 가려 인터프리터 시작이 깨지므로 site_config.py)
"""
from remote_config import RemoteConfig


def _coord(v, lo, hi, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"location.{what} is not a number")
    if not lo <= float(v) <= hi:
        raise ValueError(f"location.{what} out of range [{lo},{hi}]")
    return float(v)


def validate_site(doc):
    """site 문서 -> (site_id, name, location(lat, lng) | None). 위반이면 ValueError."""
    if not isinstance(doc, dict):
        raise ValueError("site is not an object")
    site_id = doc.get("site_id")
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id missing")
    name = doc.get("name")
    if name is None:
        name = ""
    if not isinstance(name, str):
        raise ValueError("name is not a string")
    if len(name) > 100:
        raise ValueError("name too long (> 100)")
    loc = doc.get("location")
    if loc is not None:
        if not isinstance(loc, dict):
            raise ValueError("location is not an object or null")
        loc = (_coord(loc.get("lat"), -90, 90, "lat"), _coord(loc.get("lng"), -180, 180, "lng"))
    return site_id.strip(), name.strip(), loc


class SiteConfig(RemoteConfig):
    """현재 설치 위치. 감시 루프·방송은 name만 읽는다 (문자열 교체라 락 불필요)."""
    KEY, LABEL, MISSING = "site", "설치 위치", "기본 장소명 사용"

    def __init__(self, path, base_url=None, token=None, verbose=True, on_apply=None):
        self.site_id = None
        self.name = None                 # site.json이 없으면 None (기본 장소명 사용)
        self.location = None
        super().__init__(path, base_url, token, verbose, on_apply)

    @property
    def loaded(self):
        return self.name is not None

    def _parse(self, doc):
        return validate_site(doc)

    def _set(self, parsed):
        self.site_id, self.name, self.location = parsed

    def describe(self):
        if not self.loaded:
            return f"v{self.version} · 미설정"
        loc = (f"{self.location[0]:.6f},{self.location[1]:.6f}" if self.location
               else "좌표 없음")
        return f"v{self.version} · {self.site_id} '{self.name}' · {loc}"


__all__ = ["SiteConfig", "validate_site"]
