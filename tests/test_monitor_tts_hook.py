#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_monitor_jetson.py의 방송 훅 통합 테스트 (ultralytics/모델/스피커 불필요).

ultralytics.YOLO를 스텁으로 대체하고 '사람이 파란 물체를 들고 와서 내려놓고 떠나는'
합성 영상(PNG 시퀀스)을 만들어 main()을 실행한다. 검증 대상:
  - 이벤트가 실제로 발화하는지
  - events.jsonl에 color 필드가 들어가는지 ('파란색')
  - Announcer.announce가 (색상, 클래스)로 호출되고 실시간 합성(NullBackend) -> 재생까지 가는지
  - 야간 적외선/저조도 장면에서는 색상 없이(None) 방송하는지
  - --mosaic head 로 스냅샷의 머리 영역이 가려지고, --raw-snapshot 원본은 그대로인지

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
DROP_FRAME = 21                          # 이 프레임부터 바닥에 존재

# 장면 스타일: (배경, 사람, 폐기물, 상단 띠) BGR
STYLES = {
    # 낮: 회색 바닥 + 파란 페트 + 위쪽에 색이 있는 벽/하늘 띠
    "day":  dict(bg=(128, 128, 128), person=(200, 200, 200), waste=(220, 60, 30),
                 band=(200, 170, 120)),
    # 야간 적외선: 전 픽셀 B=G=R (흑백)
    "ir":   dict(bg=(128, 128, 128), person=(200, 200, 200), waste=(60, 60, 60),
                 band=(150, 150, 150)),
    # 저조도: 전체가 어두움. 물체 자체는 파란색으로 판정될 만큼 밝아
    # (색상 생략이 '야간' 판정 때문임을 확인) 배경 차분(MAD>0.05)도 통과한다
    "dark": dict(bg=(20, 22, 20), person=(60, 60, 60), waste=(120, 40, 20),
                 band=(30, 25, 20)),
}

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


def render_frames(dirpath, style, textured_person=False):
    st = STYLES[style]
    rng = np.random.default_rng(0)
    for f in range(1, N_FRAMES + 1):
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = st["bg"]
        img[:24, :] = st["band"]
        pbox, wbox, _ = scene(f)
        x1, y1, x2, y2 = (int(v) for v in pbox)
        cv2.rectangle(img, (max(0, x1), y1), (min(W, x2), y2), st["person"], -1)
        if textured_person:                   # 모자이크 효과를 측정하려면 사람에 질감이 있어야 함
            xa, xb = max(0, x1), min(W, x2)
            if xb > xa:
                noise = rng.integers(-40, 41, (y2 - y1, xb - xa, 3))
                img[y1:y2, xa:xb] = np.clip(img[y1:y2, xa:xb].astype(np.int16) + noise,
                                            0, 255).astype(np.uint8)
        if wbox is not None:
            a, b, c, d = (int(v) for v in wbox)
            cv2.rectangle(img, (max(0, a), b), (min(W, c), d), st["waste"], -1)
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

calls = []
announcers = []
_orig_announce = Announcer.announce


def spy(self, color, waste_name):
    calls.append((color, waste_name))
    announcers.append(self)
    return _orig_announce(self, color, waste_name)


Announcer.announce = spy


def run_scenario(td, style, name=None, extra_args=(), textured_person=False):
    """합성 영상을 만들고 main()을 실행. (events, calls, announcer) 반환."""
    name = name or style
    calls.clear()
    announcers.clear()
    _Counter.f = 0
    frames_dir = td / f"frames_{name}"
    frames_dir.mkdir()
    render_frames(frames_dir, style, textured_person=textured_person)
    check("생성된 프레임 수", len(list(frames_dir.glob("*.png"))), N_FRAMES)
    # 사전 렌더링 캐시 없이, 기본(live) 모드로 NullBackend 즉석 합성 경로를 탄다
    sys.argv = ["dump_monitor_jetson.py",
                "--source", str(frames_dir / "frame_%04d.png"),
                "--name", name, "--out", str(td / "out"),
                "--no-save-video", "--tts", "--tts-backend", "null",
                "--tts-cache", str(td / "no_cache"),
                "--tts-cooldown", "0", "--tts-repeat-window", "0", *extra_args]
    dm.main()
    ev_path = td / "out" / name / "events.jsonl"
    check_true("events.jsonl 생성", ev_path.exists())
    events = [json.loads(l) for l in ev_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    check_true("이벤트 1건 이상", len(events) >= 1, f"{len(events)}건")
    return events, list(calls), (announcers[0] if announcers else None)


with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    print("1) 낮 장면: 파란 페트 투기 -> 색상 포함 방송 (실시간 합성)")
    events, got_calls, ann = run_scenario(td, "day")
    if events:
        ev = events[0]
        check("클래스", ev["class"], "페트")
        check("color 필드", ev["color"], "파란색")
        check("night 필드", ev["night"], None)
        check("소지품 매칭으로 투기자 특정", ev["owner_matched"], True)
        check("투기자 pid", ev["owner_pid"], 1)
        check_true("스냅샷 저장", (td / "out" / "day" / "event_0001.jpg").exists())
    check_true("announce 호출됨", len(got_calls) >= 1, f"{got_calls}")
    if got_calls:
        check("첫 방송 인자", got_calls[0], ("파란색", "페트"))
    check_true("Announcer가 live 모드", ann is not None and ann.mode == "live")
    if ann is not None:
        check_true("즉석 합성 -> 재생까지 완료", ann.player.played >= 1,
                   f"played={ann.player.played} synth_fail={ann.n_synth_fail}")
        check("합성 실패 0건", ann.n_synth_fail, 0)

    print("\n2) 야간 적외선(흑백) 장면: 종류만 방송, 색상 None")
    events, got_calls, ann = run_scenario(td, "ir")
    if events:
        check("color 필드", events[0]["color"], None)
        check("night 필드", events[0]["night"], "적외선")
    if got_calls:
        check("방송 인자(색상 없음)", got_calls[0], (None, "페트"))

    print("\n3) 저조도 장면: 종류만 방송, 색상 None")
    events, got_calls, ann = run_scenario(td, "dark")
    if events:
        check("color 필드", events[0]["color"], None)
        check("night 필드", events[0]["night"], "저조도")
    if got_calls:
        check("방송 인자(색상 없음)", got_calls[0], (None, "페트"))

    print("\n4) --mosaic head --raw-snapshot: 스냅샷 머리 영역 가림, 원본은 그대로")
    from mosaic import FaceMosaic, head_box
    apply_calls = []
    _orig_apply = FaceMosaic.apply

    def spy_apply(self, frame, person_boxes):
        apply_calls.append(len(person_boxes))
        return _orig_apply(self, frame, person_boxes)

    FaceMosaic.apply = spy_apply
    try:
        events, got_calls, ann = run_scenario(
            td, "day", name="mosaic", extra_args=["--mosaic", "head", "--raw-snapshot"],
            textured_person=True)
    finally:
        FaceMosaic.apply = _orig_apply
    check("프레임마다 apply 호출", len(apply_calls), N_FRAMES)
    check_true("매 프레임 사람 박스 전달", all(n == 1 for n in apply_calls))
    if events:
        ev = events[0]
        check("이벤트 자체는 동일하게 발화", (ev["class"], ev["color"]), ("페트", "파란색"))
        mos = cv2.imread(str(td / "out" / "mosaic" / "event_0001.jpg"))
        raw = cv2.imread(str(td / "out" / "mosaic" / "event_0001_raw.jpg"))
        check_true("모자이크 스냅샷 저장", mos is not None)
        check_true("원본 스냅샷 저장(_raw)", raw is not None)
        if mos is not None and raw is not None:
            pbox, _w, _c = scene(ev["frame"])
            _, hy1, _, hy2 = (int(v) for v in head_box(pbox))
            hx1, hx2 = max(0, int(pbox[0])), min(W, int(pbox[2]))   # 사람 박스 안쪽만

            def flat_ratio(img):
                """이웃 픽셀과 거의 같은(|차|<=3) 픽셀 비율. 노이즈 질감은 낮고, 픽셀화된
                블록 안은 균일하므로 높다 (JPEG 오차 허용)."""
                roi = img[hy1:hy2, hx1:hx2].astype(np.int16)
                d = np.abs(roi[:, 1:] - roi[:, :-1]).max(axis=2)
                return float((d <= 3).mean())

            f_r, f_m = flat_ratio(raw), flat_ratio(mos)
            check_true("머리 영역: 원본은 질감, 모자이크본은 블록 균일",
                       f_r < 0.3 and f_m > 0.6, f"균일 비율 {f_r:.2f} -> {f_m:.2f}")
            by1 = hy2 + 10                          # 머리 아래 몸통 영역은 두 파일이 같아야 함
            body_m = mos[by1:int(pbox[3]), hx1:hx2]
            body_r = raw[by1:int(pbox[3]), hx1:hx2]
            diff = np.abs(body_m.astype(np.int16) - body_r.astype(np.int16)).mean()
            check_true("몸통 영역은 두 스냅샷이 동일(JPEG 오차 이내)", diff < 3.0, f"평균차 {diff:.2f}")

Announcer.announce = _orig_announce

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 방송 훅 테스트 통과")
