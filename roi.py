#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""감시 구역(ROI) 원격 설정 — jetson_handoff.md §3.6 / 2026-10-04 작업 요청 §1.

운영자가 대시보드에서 그린 다각형을 서버가 버전으로 관리한다. 젯슨 쪽 흐름:

  1) 하트비트 본문 config.roi_version = 지금 적용 중인 버전 (처음 0)
  2) 하트비트 응답 config.roi_version 이 더 크면 GET /api/device/config 로 받아 온다
  3) 검증 -> config/roi.json 에 원자적 저장(임시 파일 -> rename) -> 즉시 적용
  4) POST /api/device/config/ack {roi_version, ok[, error]} — 실패 사유는 대시보드에 그대로 표시
  (이 흐름은 설치 위치(site.py)와 같아 remote_config.RemoteConfig로 공통화했다)

- 서버가 안 돼도 마지막 config/roi.json 으로 계속 동작한다. 파일이 없으면 화면 전체 감시.
- 좌표는 0~1 정규화 [x, y]. 판정 기준점은 bbox 하단 중앙 (대시보드 roi-editor.js isMonitored와 동일).
- 쓰레기 탐지는 include 구역의 외접 사각형(+여유)만 잘라 돌리고, 구역 밖 탐지는 버린다
  (crop_rect). 사람 탐지는 화면 전체 — 구역 밖 행인도 얼굴 모자이크해야 하므로.
- exclude 가 include 보다 우선. include 가 하나도 없으면 화면 전체가 감시 구역.
- 같은 버전이 검증에 실패하면 다시 받지 않는다 (30초마다 같은 오류 ack 도배 방지).
"""
from remote_config import RemoteConfig

MAX_ZONES = 16
MIN_POINTS, MAX_POINTS = 3, 32
ZONE_COLORS = {"include": (0, 200, 0), "exclude": (0, 0, 255)}     # BGR


def point_in_polygon(x, y, poly):
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def is_monitored(bbox, frame_w, frame_h, zones):
    """bbox(x1,y1,x2,y2 픽셀)의 하단 중앙이 감시 대상인가."""
    x, y = (bbox[0] + bbox[2]) / 2 / frame_w, bbox[3] / frame_h
    if any(point_in_polygon(x, y, z["points"]) for z in zones if z["type"] == "exclude"):
        return False
    inc = [z for z in zones if z["type"] == "include"]
    return not inc or any(point_in_polygon(x, y, z["points"]) for z in inc)


# 탐지 크롭 여유: 판정은 bbox 하단 중앙이라 물체 몸통은 구역 위로 삐져나온다 -> 위쪽을 더 넉넉히
CROP_PAD = {"x": 0.05, "top": 0.15, "bottom": 0.03}       # 프레임 폭/높이 비율


def crop_rect(zones, frame_w, frame_h, pad=CROP_PAD):
    """쓰레기 탐지를 돌릴 영역 (x1, y1, x2, y2 픽셀). include 구역들의 외접 사각형 + 여유.
    include가 없으면(화면 전체 감시) None -> 프레임 전체를 쓴다."""
    pts = [p for z in zones if z["type"] == "include" for p in z["points"]]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1 = max(0, int((min(xs) - pad["x"]) * frame_w))
    x2 = min(frame_w, int(round((max(xs) + pad["x"]) * frame_w)))
    y1 = max(0, int((min(ys) - pad["top"]) * frame_h))
    y2 = min(frame_h, int(round((max(ys) + pad["bottom"]) * frame_h)))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def validate(doc):
    """ROI 문서 -> 정리된 zones 목록. 규칙 위반이면 ValueError(대시보드에 표시될 사유)."""
    if not isinstance(doc, dict):
        raise ValueError("roi is not an object")
    zones = doc.get("zones")
    if zones is None:
        zones = []
    if not isinstance(zones, list):
        raise ValueError("zones is not a list")
    if len(zones) > MAX_ZONES:
        raise ValueError(f"too many zones ({len(zones)} > {MAX_ZONES})")
    out = []
    for k, z in enumerate(zones):
        zid = z.get("id", k) if isinstance(z, dict) else k
        if not isinstance(z, dict):
            raise ValueError(f"zone {zid}: not an object")
        if z.get("type") not in ("include", "exclude"):
            raise ValueError(f"zone {zid}: type must be include|exclude")
        pts = z.get("points")
        if not isinstance(pts, list) or not MIN_POINTS <= len(pts) <= MAX_POINTS:
            raise ValueError(f"zone {zid}: needs {MIN_POINTS}-{MAX_POINTS} points")
        clean = []
        for p in pts:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError, KeyError):
                raise ValueError(f"zone {zid}: polygon parse error") from None
            if not (-0.01 <= x <= 1.01 and -0.01 <= y <= 1.01):     # 반올림 오차만 허용
                raise ValueError(f"zone {zid}: point out of range [0,1]")
            clean.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
        out.append({"id": str(z.get("id", k)), "name": str(z.get("name") or ""),
                    "type": z["type"], "points": clean})
    return out


class RoiConfig(RemoteConfig):
    """현재 ROI + 서버 동기화(remote_config.py). 감시 루프는 monitored()/draw()/generation만 쓴다.

    zones는 통째로 새 리스트로 바꿔 끼우므로 (감시 루프는 읽기만) 락이 필요 없다.
    """
    KEY, LABEL, MISSING = "roi", "ROI", "화면 전체 감시"

    def __init__(self, path, base_url=None, token=None, verbose=True):
        self.zones = []
        super().__init__(path, base_url, token, verbose)

    def _parse(self, doc):
        return validate(doc)

    def _set(self, zones):
        self.zones = zones

    def monitored(self, bbox, frame_w, frame_h):
        zones = self.zones
        return True if not zones else is_monitored(bbox, frame_w, frame_h, zones)

    def describe(self):
        n_inc = sum(1 for z in self.zones if z["type"] == "include")
        n_exc = len(self.zones) - n_inc
        return (f"v{self.version} · 감시 {n_inc}개 · 제외 {n_exc}개" if self.zones
                else f"v{self.version} · 화면 전체")

    def draw(self, frame):
        """송출·녹화 프레임에 구역 외곽선과 이름을 그린다.

        반투명 채우기는 2560x1440에서 구역당 ~7ms(보드 실측)라 외곽선만 그린다 (~0.1ms).
        """
        zones = self.zones
        if not zones:
            return
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        th = max(2, round(w / 640))
        for z in zones:
            pts = np.array([(round(x * w), round(y * h)) for x, y in z["points"]], np.int32)
            color = ZONE_COLORS[z["type"]]
            cv2.polylines(frame, [pts], True, color, th, cv2.LINE_AA)
            # 이름은 한글이 깨지므로(Hershey 폰트) id와 종류만
            tag = f"{z['id']} {'IN' if z['type'] == 'include' else 'EX'}"
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            cv2.putText(frame, tag, (int(x0) + th * 2, int(y0) + 12 * th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th, color, max(1, th // 2))


__all__ = ["RoiConfig", "is_monitored", "point_in_polygon", "validate"]
