#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""회수(되가져감) 판정 — jetson_handoff.md §3.2 / jetson_data.md §4.2 outcome.retrieved.

사건 후 window_s(기본 10분) 안에 그 물체가 사라지고, 그 뒤로 confirm_s 동안 다시 보이지 않으면
"retrieved"로 보고한다. after_s = 마지막으로 보인 시각 - 사건 시각.

- '보임' 판정은 물체 추적 ID가 아니라 사건 bbox 자리다. 같은 자리(IoU)에 감시 루프가 잡고 있는
  물체가 있으면 아직 있는 것 — 탐지 깜빡임으로 객체가 새 ID로 다시 생겨도 회수로 오판하지 않는다.
- 사람이 그 자리를 가리고 있는 동안은 '안 보임'을 세지 않는다 (서 있는 사람에 가려진 것과
  들고 간 것을 구분할 수 없다). 들고 가면 사람이 떠난 뒤부터 센다.
- 부재 시간은 실제로 프레임을 본 시간만 센다 (프레임 간격 상한 max_gap_s). 스트림이 끊겼다
  돌아온 직후에 '그동안 안 보였다'고 회수 처리하지 않기 위함.
- window_s가 지나도록 계속 보이면 판정 종료 (회수 아님 — 보고 없음).
- state_path를 주면 판정 중인 사건을 그 파일에 실제 시각으로 적어 둔다 (원자적 쓰기). 서비스가
  재시작돼도(코드 배포·오류 재기동) 이어서 판정한다 — 재시작이 잦아 회수 보고가 한 번도 안 나가던 문제
  (2026-10-05). 부재 시간은 재시작 후부터 다시 센다 (중단 동안 안 본 것을 '없었다'로 치지 않는다).
  라이브 입력 전용: t는 time.monotonic()이어야 한다.
"""
import json
import time
from pathlib import Path

IOU_PRESENT = 0.3           # 이 이상 겹치는 물체가 있으면 '아직 있음'
OCCLUDE_FRAC = 0.3          # 사람 박스가 물체 박스의 이 비율 이상을 덮으면 '가려짐'


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _covered(box, by):
    """box 면적 중 by가 덮은 비율."""
    ix = max(0.0, min(box[2], by[2]) - max(box[0], by[0]))
    iy = max(0.0, min(box[3], by[3]) - max(box[1], by[1]))
    return ix * iy / max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)


class _Watch:
    __slots__ = ("event_id", "box", "t_event", "last_present", "absent_s")

    def __init__(self, event_id, box, t_event):
        self.event_id, self.box = event_id, list(box)
        self.t_event = self.last_present = t_event
        self.absent_s = 0.0


class RetrievalWatch:
    """사건별 회수 감시. update()를 매 프레임 부르고, 확정되면 on_retrieved(event_id, after_s)."""

    def __init__(self, on_retrieved, window_s=600.0, confirm_s=120.0, max_gap_s=2.0,
                 state_path=None):
        self.on_retrieved = on_retrieved
        self.window_s, self.confirm_s = float(window_s), float(confirm_s)
        self.max_gap_s = float(max_gap_s)
        self.items = []
        self._t_prev = None
        self.n_retrieved = self.n_expired = 0
        self.state_path = Path(state_path) if state_path else None
        self._t_saved = 0.0
        self.n_restored = self._load()

    # ------------------------------------------------------------ 상태 파일
    def _load(self):
        if self.state_path is None or not self.state_path.is_file():
            return 0
        try:
            rows = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        off = time.monotonic() - time.time()          # 실제 시각 -> 이 프로세스의 monotonic
        for r in rows if isinstance(rows, list) else []:
            try:
                t_ev, t_last = float(r["t_event"]), float(r["last_present"])
                if time.time() - t_ev > self.window_s + self.confirm_s:
                    continue                          # 판정 시간이 이미 지났다
                w = _Watch(str(r["event_id"]), [float(v) for v in r["box"]], t_ev + off)
                w.last_present = t_last + off
                self.items.append(w)
            except (KeyError, TypeError, ValueError):
                continue
        self._save()
        return len(self.items)

    def _save(self):
        self._t_saved = time.monotonic()
        if self.state_path is None:
            return
        off = time.time() - time.monotonic()
        rows = [{"event_id": w.event_id, "box": w.box, "t_event": round(w.t_event + off, 3),
                 "last_present": round(w.last_present + off, 3)} for w in self.items]
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(f".{self.state_path.name}.tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            pass

    def add(self, event_id, box, t_event):
        self.items.append(_Watch(event_id, box, t_event))
        self._save()

    def update(self, t, object_boxes, person_boxes):
        dt = 0.0 if self._t_prev is None else min(max(0.0, t - self._t_prev), self.max_gap_s)
        self._t_prev = t
        n_before = len(self.items)
        for w in list(self.items):
            if any(_iou(w.box, b) >= IOU_PRESENT for b in object_boxes):
                w.last_present, w.absent_s = t, 0.0
                if t - w.t_event > self.window_s:        # 10분이 지나도 그대로 있다 -> 종료
                    self.items.remove(w)
                    self.n_expired += 1
                continue
            if any(_covered(w.box, p) >= OCCLUDE_FRAC for p in person_boxes):
                continue                                  # 가려져서 판단 불가 — 세지 않는다
            w.absent_s += dt
            if w.absent_s >= self.confirm_s:
                self.items.remove(w)
                if w.last_present - w.t_event <= self.window_s:
                    self.n_retrieved += 1
                    self.on_retrieved(w.event_id, max(0, round(w.last_present - w.t_event)))
                else:
                    self.n_expired += 1
        if self.state_path is not None and (
                len(self.items) != n_before or (self.items and t - self._t_saved >= 30)):
            self._save()                              # 목록 변경 또는 30초마다 last_present 갱신분

    def summary(self):
        return (f"회수 판정: 회수 {self.n_retrieved}건 · 미회수 종료 {self.n_expired}건"
                + (f" · 판정 중 {len(self.items)}건" if self.items else ""))
