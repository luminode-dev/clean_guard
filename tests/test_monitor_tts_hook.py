#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_monitor_jetson.py의 방송 훅 통합 테스트 (ultralytics/모델/스피커 불필요).

ultralytics.YOLO를 스텁으로 대체하고 '사람이 파란 물체를 들고 와서 내려놓고 떠나는'
합성 영상(PNG 시퀀스)을 만들어 main()을 실행한다. 검증 대상:
  - 이벤트가 실제로 발화하는지
  - events.jsonl에 color 필드가 들어가는지 ('파란색')
  - Announcer.announce가 (색상, 클래스)로 호출되는지

  python3 tests/test_monitor_tts_hook.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

W, H = 320, 240
N_FRAMES = 60
DROP_BOX = [50, 100, 90, 140]            # 바닥에 놓인 폐기물
BLUE = (220, 60, 30)                     # BGR
BG = 128
DROP_FRAME = 21                          # 이 프레임부터 바닥에 존재

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (기대 {want!r})"))
    if not ok:
        fails.append(name)


def check_true(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


# --- 시나리오: 프레임별 (사람 박스, 폐기물 박스, 소지중 여부) ---
def person_x(f):
    if f <= 20:
        return 200 - (f - 1) * 4          # 200 -> 124 로 접근
    if f < 30:
        return 120                        # 내려놓고 잠시 머무름
    return min(400, 120 + (f - 29) * 15)  # 이탈


def scene(f):
    px = person_x(f)
    pbox = [px - 25, 70, px + 25, 190]
    if f <= 20:                           # 소지중: 사람 박스 안쪽
        return pbox, [px - 15, 105, px + 15, 135], True
    if f >= DROP_FRAME:                   # 바닥
        return pbox, list(DROP_BOX), False
    return pbox, None, False


def render_frames(dirpath):
    for f in range(1, N_FRAMES + 1):
        img = np.full((H, W, 3), BG, np.uint8)
        pbox, wbox, _ = scene(f)
        x1, y1, x2, y2 = (int(v) for v in pbox)
        cv2.rectangle(img, (max(0, x1), y1), (min(W, x2), y2), (200, 200, 200), -1)
        if wbox is not None:
            a, b, c, d = (int(v) for v in wbox)
            cv2.rectangle(img, (max(0, a), b), (min(W, c), d), BLUE, -1)
        cv2.imwrite(str(Path(dirpath) / f"frame_{f:04d}.png"), img)


# --- ultralytics 스텁 ---
class _Box:
    def __init__(self, xyxy, cls=0, conf=0.9):
        self.xyxy = [np.asarray(xyxy, dtype=np.float32)]
        self.cls = cls
        self.conf = conf


class _Boxes:
    def __init__(self, boxes, ids=None):
        self._boxes = boxes
        self.id = ids

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Counter:
    f = 0


class FakeYOLO:
    """path에 'person'이 들어있으면 사람 모델, 아니면 폐기물 모델처럼 동작."""

    def __init__(self, path):
        self.is_person = "person" in str(path)

    def track(self, frame, **kw):
        _Counter.f += 1
        pbox, _w, _c = scene(_Counter.f)
        return [_Result(_Boxes([_Box(pbox)], ids=[1]))]

    def predict(self, frame, **kw):
        f = _Counter.f
        if self.is_person or f == 0:                  # 워밍업 호출은 빈 결과
            return [_Result(_Boxes([]))]
        _p, wbox, _c = scene(f)
        if wbox is None:
            return [_Result(_Boxes([]))]
        return [_Result(_Boxes([_Box(wbox, cls=3, conf=0.85)]))]   # cls 3 = '페트'


fake = types.ModuleType("ultralytics")
fake.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake

import dump_monitor_jetson as dm                                   # noqa: E402
from tts import Announcer                                          # noqa: E402

print("1) 합성 영상 + 스텁 모델로 main() 실행")
calls = []
_orig_announce = Announcer.announce


def spy(self, color, waste_name):
    calls.append((color, waste_name))
    return _orig_announce(self, color, waste_name)


Announcer.announce = spy

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    frames_dir = td / "frames"
    frames_dir.mkdir()
    render_frames(frames_dir)
    check("생성된 프레임 수", len(list(frames_dir.glob("*.png"))), N_FRAMES)

    # 캐시를 NullBackend로 미리 채워 실시간 합성 없이 캐시 히트 경로를 타게 한다
    from tts import PhraseCache, make_backend
    cache = PhraseCache(td / "cache", make_backend("null"))
    cache.prerender(["파란색"], ["페트"], verbose=False)

    sys.argv = ["dump_monitor_jetson.py",
                "--source", str(frames_dir / "frame_%04d.png"),
                "--name", "ttshook", "--out", str(td / "out"),
                "--no-save-video", "--tts", "--tts-backend", "null",
                "--tts-cache", str(td / "cache"),
                "--tts-cooldown", "0", "--tts-repeat-window", "0"]
    dm.main()

    print("\n2) 이벤트 기록")
    ev_path = td / "out" / "ttshook" / "events.jsonl"
    check_true("events.jsonl 생성", ev_path.exists())
    events = [json.loads(l) for l in ev_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    check_true("이벤트 1건 이상", len(events) >= 1, f"{len(events)}건")
    if events:
        ev = events[0]
        check("클래스", ev["class"], "페트")
        check("color 필드", ev["color"], "파란색")
        check("소지품 매칭으로 투기자 특정", ev["owner_matched"], True)
        check("투기자 pid", ev["owner_pid"], 1)
        check_true("스냅샷 저장", (td / "out" / "ttshook" / "event_0001.jpg").exists())

    print("\n3) 방송 호출")
    check_true("announce 호출됨", len(calls) >= 1, f"{calls}")
    if calls:
        check("첫 방송 인자", calls[0], ("파란색", "페트"))

Announcer.announce = _orig_announce

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 방송 훅 테스트 통과")
