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
  - --reid: 투기자가 나갔다 새 트래커 id로 돌아오면 이전 pid로 재식별되는지 (가짜 임베더)

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
DROP_BOX = [60, 140, 100, 180]           # 바닥에 놓인 폐기물 — 내려놓는 순간 사람 박스(x 95~145)와 겹친다
                                         # (귀속은 '처음 탐지된 프레임에 겹친 사람'만, 2026-10-04)
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


PERSON_TRACK = None     # 시나리오별 트래커 출력 오버라이드: f -> [(pid, box)] (None이면 pid 1 고정)
SCENE = None            # 시나리오 오버라이드: f -> (사람 박스|None, 폐기물 박스|None, 소지 여부)


def cur_scene(f):
    return (SCENE or scene)(f)


def scene_slow_leave(f):
    """기본 장면과 같되 천천히 떠난다 — 발화 시점(정지 조건 STATIC_FRAMES 이후)에도
    사람이 화면 안에 있어야 스냅샷의 머리 모자이크를 검사할 수 있다."""
    pbox, wbox, c = scene(f)
    if f >= 30:
        px = 120 + (f - 29) * 5
        pbox = [px - 25, 70, px + 25, 190]
    return pbox, wbox, c


def scene_flicker(f):
    """내려놓은 직후 투기자가 떠나고, 쓰레기 탐지가 2초(20프레임) 끊겼다가 같은 자리에서 다시 잡힌다
    (뿌연 유리창 너머 원거리 봉투, 2026-10-04 현장). 같은 물체로 이어 붙여 투기자 연결을 유지해야 한다."""
    pbox, wbox, c = scene(f)
    if 31 <= f <= 50:
        wbox = None
    return pbox, wbox, c


def scene_walk_into(f):
    """이미 놓여 있던 쓰레기(사람 없을 때 등장)에 나중에 사람이 걸어 들어와 겹쳤다가 떠난다.
    처음 탐지 때 겹친 사람이 없으므로 귀속되지 않아야 하고 이벤트도 없어야 한다.
    (이전 로직은 등장 전후 40프레임 안에 근처 사람이 있으면 귀속해 이 경우 발화했다)"""
    wbox = list(DROP_BOX) if f >= 8 else None
    if f < 20:
        return None, wbox, False
    px = max(110, 250 - (f - 20) * 8) if f <= 50 else 110 + (f - 50) * 15
    return [px - 25, 70, px + 25, 190], wbox, False


# --- 오탐 시나리오 (2026-09-26 실측: 흰 우산 쓴 보행자를 '흰색 쓰레기봉투 투기'로 판정) ---
# 재현 조건: 우산은 걸어 들어오는 도중(15프레임~)부터 검출되고(처음부터 보이면 '기존 적치물'로
# 빠진다), 사람은 곁에 있어도 트래커 id가 바뀌거나 검출이 끊기면 '주인 부재'로 계산된다.
UMB_FROM = 15


def scene_umbrella(f):
    """우산이 상체를 가려 사람 박스는 다리만, 우산은 그 위에 붙어 함께 이동."""
    px = 20 + f * 4
    umb = [px - 30, 95, px + 30, 132] if f >= UMB_FROM else None
    return [px - 15, 130, px + 15, 200], umb, False


def track_umbrella(f):
    """걷는 도중 트래커 id 전환 (1 -> 2) — 우산에 가려 흔한 일."""
    return [(1 if f < 30 else 2, scene_umbrella(f)[0])]


def scene_occluded_walk(f):
    """사람이 25프레임까지만 잡히고 이후엔 우산에 완전히 가려 검출이 끊긴 채
    '물체'만 계속 걸어간다 — 정지 조건만으로 걸러져야 한다."""
    px = 20 + f * 4
    pbox = [px - 15, 130, px + 15, 200] if f <= 25 else None
    umb = [px - 30, 40, px + 30, 80] if f >= UMB_FROM else None
    return pbox, umb, False

def render_frames(dirpath, style, textured_person=False, n_frames=N_FRAMES):
    st = STYLES[style]
    rng = np.random.default_rng(0)
    for f in range(1, n_frames + 1):
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = st["bg"]
        img[:24, :] = st["band"]
        pbox, wbox, _ = cur_scene(f)
        x1, y1, x2, y2 = (int(v) for v in (pbox or (0, 0, 0, 0)))
        if pbox is not None:
            cv2.rectangle(img, (max(0, x1), y1), (min(W, x2), y2), st["person"], -1)
        if textured_person and pbox is not None:                   # 모자이크 효과를 측정하려면 사람에 질감이 있어야 함
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
    waste_inputs = []       # 폐기물 모델이 받은 프레임 크기 (ROI 크롭 확인용)


def _view_offset(img):
    """img가 전체 프레임을 잘라 낸 뷰(frame[y0:y1, x0:x1])면 (x0, y0). 아니면 (0, 0).
    ROI 크롭 시 스텁이 실제 모델처럼 '크롭 기준 좌표'를 돌려주기 위함."""
    base = img.base
    if base is None or getattr(base, "ndim", 0) != 3:
        return 0, 0
    d = img.__array_interface__["data"][0] - base.__array_interface__["data"][0]
    return (d % base.strides[0]) // base.strides[1], d // base.strides[0]


class FakeYOLO:
    """path에 'person'이 들어있으면 사람 모델, 아니면 폐기물 모델처럼 동작."""

    def __init__(self, path):
        self.is_person = "person" in str(path)

    def track(self, frame, **kw):
        _Counter.f += 1
        if PERSON_TRACK is not None:
            plist = PERSON_TRACK(_Counter.f)
            return [_Result(_Boxes([_Box(b) for _, b in plist], ids=[pid for pid, _ in plist]))]
        pbox, _w, _c = cur_scene(_Counter.f)
        if pbox is None:
            return [_Result(_Boxes([], ids=[]))]
        return [_Result(_Boxes([_Box(pbox)], ids=[1]))]

    def predict(self, frame, **kw):
        f = _Counter.f
        if self.is_person or f == 0:                  # 워밍업 호출은 빈 결과
            return [_Result(_Boxes([]))]
        _Counter.waste_inputs.append(frame.shape[:2])
        _p, wbox, _c = cur_scene(f)
        if wbox is None:
            return [_Result(_Boxes([]))]
        x0, y0 = _view_offset(frame)
        h, w = frame.shape[:2]
        b = [min(max(wbox[0] - x0, 0), w), min(max(wbox[1] - y0, 0), h),
             min(max(wbox[2] - x0, 0), w), min(max(wbox[3] - y0, 0), h)]
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:                       # 크롭 밖이면 안 보인다
            return [_Result(_Boxes([]))]
        return [_Result(_Boxes([_Box(b, cls=3, conf=0.85)]))]   # cls 3 = '페트'


fake = types.ModuleType("ultralytics")
fake.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake

import dump_monitor_jetson as dm                                   # noqa: E402
from tts import Announcer                                          # noqa: E402

calls = []
announcers = []
_orig_announce = Announcer.announce


call_kw = []


def spy(self, color, waste_name, **kw):
    calls.append((color, waste_name))
    call_kw.append(kw)
    announcers.append(self)
    return _orig_announce(self, color, waste_name, **kw)


Announcer.announce = spy


def run_scenario(td, style, name=None, extra_args=(), textured_person=False, n_frames=N_FRAMES,
                 expect_event=True):
    """합성 영상을 만들고 main()을 실행. (events, calls, announcer) 반환."""
    name = name or style
    calls.clear()
    announcers.clear()
    _Counter.f = 0
    frames_dir = td / f"frames_{name}"
    frames_dir.mkdir()
    render_frames(frames_dir, style, textured_person=textured_person, n_frames=n_frames)
    check("생성된 프레임 수", len(list(frames_dir.glob("*.png"))), n_frames)
    # 사전 렌더링 캐시 없이, 기본(live) 모드로 NullBackend 즉석 합성 경로를 탄다
    sys.argv = ["dump_monitor_jetson.py",
                "--source", str(frames_dir / "frame_%04d.png"),
                "--name", name, "--out", str(td / "out"),
                "--no-save-video", "--tts", "--tts-backend", "null",
                "--tts-cache", str(td / "no_cache"),
                "--tts-cooldown", "0", "--tts-repeat-window", "0",
                # 현장 config/roi.json·site.json을 끌어오지 않게 (뒤에 오는 extra_args가 우선)
                "--roi-file", str(td / "no_roi.json"), "--site-file", str(td / "no_site.json"),
                *extra_args]
    dm.main()
    ev_path = td / "out" / name / "events.jsonl"
    check_true("events.jsonl 생성", ev_path.exists())
    events = [json.loads(l) for l in ev_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    if expect_event:
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

    print("\n2-1) 오탐 방지: 우산 쓴 보행자 (우산이 사람 박스 위에 붙어 이동)")
    SCENE, PERSON_TRACK = scene_umbrella, track_umbrella
    events, got_calls, ann = run_scenario(td, "day", name="umbrella", expect_event=False)
    PERSON_TRACK = None
    check("이벤트 0건", len(events), 0)
    check("방송 0건", len(got_calls), 0)

    print("\n2-2) 오탐 방지: 사람이 가려져 안 잡힌 채 물체만 이동 (정지 조건)")
    SCENE = scene_occluded_walk
    events, got_calls, ann = run_scenario(td, "day", name="occluded_walk", expect_event=False)
    check("이벤트 0건", len(events), 0)
    SCENE = None

    print("\n2-3) 귀속 규칙: 이미 있던 쓰레기에 나중에 사람이 다가가 겹침 -> 연결·이벤트 없음")
    SCENE = scene_walk_into
    events, got_calls, ann = run_scenario(td, "day", name="walk_into", expect_event=False,
                                          n_frames=80)
    check("이벤트 0건", len(events), 0)
    check("방송 0건", len(got_calls), 0)
    SCENE = None

    print("\n2-5-1) 탐지가 2초 끊겼다 같은 자리에서 다시 잡혀도 투기자 연결 유지 -> 이벤트")
    SCENE = scene_flicker
    events, got_calls, ann = run_scenario(td, "day", name="flicker", n_frames=90)
    if events:
        check("투기자 pid 유지", events[0]["owner_pid"], 1)
    SCENE = None

    print("\n2-4) ROI: 감시 구역 밖에 투기 -> 쓰레기 모델이 구역만 보므로 이벤트 없음")
    roi_right = td / "roi_right.json"            # 화면 오른쪽 절반만 감시 (투기 지점 x 60~100은 밖)
    roi_right.write_text(json.dumps({"version": 1, "zones": [
        {"id": "z1", "type": "include", "points": [[0.6, 0.3], [1, 0.3], [1, 1], [0.6, 1]]}]}))
    _Counter.waste_inputs = []
    events, got_calls, ann = run_scenario(td, "day", name="roi_out", expect_event=False,
                                          extra_args=["--roi-file", str(roi_right)])
    check("이벤트 0건", len(events), 0)
    sizes = set(_Counter.waste_inputs[1:])        # [0]은 워밍업
    check_true("쓰레기 모델은 구역 영역만 받음", sizes and all(w < W for _h, w in sizes),
               f"{sizes}")

    print("\n2-5) ROI: 감시 구역 안에 투기 -> 기존처럼 이벤트, 좌표는 프레임 기준으로 복원")
    roi_left = td / "roi_left.json"              # 왼쪽 절반 (투기 지점 포함)
    roi_left.write_text(json.dumps({"version": 1, "zones": [
        {"id": "z1", "type": "include", "points": [[0, 0.4], [0.5, 0.4], [0.5, 1], [0, 1]]}]}))
    _Counter.waste_inputs = []
    events, got_calls, ann = run_scenario(td, "day", name="roi_in",
                                          extra_args=["--roi-file", str(roi_left)])
    if events:
        check("이벤트 박스 = 실제 투기 위치", events[0]["box"], DROP_BOX)
    sizes = set(_Counter.waste_inputs[1:])
    check_true("쓰레기 모델 입력이 크롭됨", sizes and all(w < W for _h, w in sizes), f"{sizes}")

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
    SCENE = scene_slow_leave
    try:
        events, got_calls, ann = run_scenario(
            td, "day", name="mosaic", extra_args=["--mosaic", "head", "--raw-snapshot"],
            textured_person=True)
    finally:
        FaceMosaic.apply = _orig_apply
        SCENE_USED = SCENE
        SCENE = None
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
            pbox, _w, _c = SCENE_USED(ev["frame"])
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

    print("\n5) --reid: 투기 후 나갔다가 새 트래커 id로 돌아온 사람 -> 이전 pid로 재식별")
    from person_reid import PersonReID

    class FakeEncoder:
        """박스 폭을 정체성으로 쓰는 가짜 임베더 (같은 폭 = 같은 사람)."""
        def __init__(self):
            self.rng = np.random.default_rng(0)
            self.base = {}

        def __call__(self, frame, dets):
            out = []
            for cx, cy, w, h in dets:
                k = int(round(w))
                if k not in self.base:
                    v = self.rng.normal(size=512)
                    self.base[k] = v / np.linalg.norm(v)
                v = self.base[k] + self.rng.normal(size=512) * 0.05
                out.append(v / np.linalg.norm(v))
            return out

    RETURN_F = 70

    def track_leave_and_return(f):
        if f < 30:                                   # 접근·내려놓기 (pid 1)
            return [(1, scene(f)[0])]
        if f < RETURN_F:                             # 화면 밖 (트래커 출력 없음, 40프레임)
            return []
        return [(9, [95, 70, 145, 190])]             # 새 트래커 id 9로 물건 옆에 복귀 (폭 50 = 같은 사람)

    merges_seen = []
    _orig_update = PersonReID.update

    def spy_update(self, frame, persons, frame_idx, now=None, embs=None):
        out, merges = _orig_update(self, frame, persons, frame_idx, now, embs)
        merges_seen.extend(merges)
        return out, merges

    PersonReID.update = spy_update
    _orig_loader = dm.load_reid_encoder
    dm.load_reid_encoder = lambda p=None, device=None: FakeEncoder()
    PERSON_TRACK = track_leave_and_return
    try:
        events, got_calls, ann = run_scenario(td, "day", name="reid", extra_args=["--reid"],
                                              n_frames=90)
    finally:
        PERSON_TRACK = None
        PersonReID.update = _orig_update
        dm.load_reid_encoder = _orig_loader
    check_true("이벤트는 사람이 떠난 뒤 발화", bool(events) and 30 <= events[0]["frame"] < RETURN_F,
               f"frame={events[0]['frame'] if events else None}")
    if events:
        check("투기자 pid", events[0]["owner_pid"], 1)
    check("재식별 병합", merges_seen, [(9, 1)])

    print("\n6) --report: 이벤트 JSON + 스냅샷이 관제 서버 API로 전송")
    import os
    import threading
    import time as _time
    from http.server import ThreadingHTTPServer

    from test_event_sender import TOKEN, FakeAPI, wait_until

    RPT_PORT = 18011
    srv = ThreadingHTTPServer(("127.0.0.1", RPT_PORT), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeAPI.events, FakeAPI.media, FakeAPI.mode = [], [], "ok"
    # load_env_file은 setdefault라 미리 넣은 환경변수가 .env보다 우선한다
    os.environ.update(API_URL=f"http://127.0.0.1:{RPT_PORT}", DEVICE_TOKEN=TOKEN,
                      DEVICE_ID="JT-GN-0001", SITE_ID="SITE-GN-0001")
    # 썸네일(PUT)은 test_roi_followups.py에서 따로 본다 — 여기 가짜 서버는 PUT이 없다
    events, got_calls, ann = run_scenario(td, "day", name="report",
                                          extra_args=["--report", "--thumbnail-sec", "0"])
    check_true("서버에 이벤트 도착", wait_until(lambda: len(FakeAPI.events) >= 1))
    if FakeAPI.events:
        srv_ev = FakeAPI.events[0]
        check("event_id ULID", len(srv_ev["event_id"]), 26)
        check("device_id", srv_ev["device_id"], "JT-GN-0001")
        check("detection.class", srv_ev["detection"]["class"], "페트")
        check("detection.color", srv_ev["detection"]["color"], "파란색")
        check("frame_size", srv_ev["detection"]["frame_size"], [W, H])
        check("suspect.owner_pid", srv_ev["suspect"]["owner_pid"], 1)
        check_true("person_bbox_at_drop 기록",
                   isinstance(srv_ev["suspect"]["person_bbox_at_drop"], list)
                   and len(srv_ev["suspect"]["person_bbox_at_drop"]) == 4)
        check("announce.played", srv_ev["announce"]["played"], True)
        check_true("announce.phrase에 클래스 포함",
                   "페트" in (srv_ev["announce"]["phrase"] or ""))
        check("media 파일명", srv_ev["media"],
              {"snapshot": "event_0001.jpg", "clip": "event_0001.mp4", "clip_range_s": [-5, 5]})
    check_true("스냅샷 multipart 도착", wait_until(lambda: len(FakeAPI.media) >= 1))
    if FakeAPI.media:
        check_true("snapshot 파트 포함", b'name="snapshot"' in FakeAPI.media[0][1])
    check_true("announce에 audio_out(wav) 전달", bool(call_kw and str(call_kw[-1].get("audio_out", "")).endswith(".wav")))
    # 방송 음성은 합성 뒤 attach로 따로 올라간다 (같은 event_id, JSON 재전송 없음)
    check_true("방송 wav multipart 도착", wait_until(
        lambda: any(b'name="announce_audio"' in m[1] for m in FakeAPI.media)))
    audio = [m for m in FakeAPI.media if b'name="announce_audio"' in m[1]]
    if audio and FakeAPI.events:
        check("wav가 같은 event_id로", audio[0][0], FakeAPI.events[0]["event_id"])
        check_true("audio/wav", b"audio/wav" in audio[0][1])
        check("이벤트 JSON은 1회만", len(FakeAPI.events), 1)
    # 전후 클립: 종료 시 모은 만큼 mp4로 마무리해 같은 event_id로 따로 올라간다
    check_true("클립 multipart 도착", wait_until(
        lambda: any(b'name="clip"' in m[1] for m in FakeAPI.media)))
    clip = [m for m in FakeAPI.media if b'name="clip"' in m[1]]
    if clip and FakeAPI.events:
        check("클립이 같은 event_id로", clip[0][0], FakeAPI.events[0]["event_id"])
        check_true("video/mp4 + ftyp", b"video/mp4" in clip[0][1] and b"ftyp" in clip[0][1])
    qdir = td / "out" / "queue" / "events"
    check_true("전송 후 큐 비움", not list(qdir.glob("*.json")))
    # --report면 하트비트도 함께 나간다 (시작 즉시 1회)
    check_true("하트비트 도착", len(FakeAPI.heartbeats) >= 1)
    if FakeAPI.heartbeats:
        hb = FakeAPI.heartbeats[-1]
        check("하트비트 device_id", hb.get("device_id"), "JT-GN-0001")
        check_true("pipeline.stream 보고", hb["pipeline"].get("stream") in ("ok", "lost"))
        check_true("pipeline.infer_ms 키", set(hb["pipeline"].get("infer_ms", {})) == {"person", "waste"})
        check_true("tts.worker 보고", hb.get("tts", {}).get("worker") in ("ready", "starting", "cache"))

    print("\n7) --report: 서버 다운이어도 감시는 돌고, 이벤트는 큐에 보관")
    FakeAPI.mode = "down"
    events, got_calls, ann = run_scenario(td, "day", name="report_down",
                                          extra_args=["--report", "--thumbnail-sec", "0"])
    check_true("이벤트는 정상 발화", len(events) >= 1)
    check_true("큐에 보관됨", len(list(qdir.glob("*.json"))) >= 1)
    FakeAPI.mode = "ok"
    srv.shutdown()

Announcer.announce = _orig_announce

print("\n8) 귀속: 쓰레기 첫 탐지 프레임에 사람 탐지가 빠져도 방금 놓친 사람(예측 위치)이면 연결")
bag = [100, 150, 140, 190]
o = dm.WasteObject(bag, 0, 0.5, 50, None)
how = dm.link_owner(o, bag, 0, None, [], {}, 50, recent=[(7, [120, 60, 170, 200])])
check("깜빡인 사람에게 연결", (o.owner_pid, o.person_at_birth), (7, True))
check_true("로그에 보정 표시", "깜빡임 보정" in how, how)
o = dm.WasteObject(bag, 0, 0.5, 50, None)
dm.link_owner(o, bag, 0, None, [(3, [90, 60, 130, 200])], {}, 50, recent=[(7, [120, 60, 170, 200])])
check("이번 프레임 탐지가 있으면 그 사람 우선", o.owner_pid, 3)
o = dm.WasteObject(bag, 0, 0.5, 50, None)
how = dm.link_owner(o, bag, 0, None, [], {}, 50, recent=[(7, [300, 60, 350, 200])])
check("멀리 있던 사람은 연결 안 함", (o.owner_pid, o.person_at_birth), (None, False))

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 방송 훅 테스트 통과")
