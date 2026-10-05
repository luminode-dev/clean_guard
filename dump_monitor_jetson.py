# -*- coding: utf-8 -*-
"""무단투기 의심 이벤트 감지 — Jetson 배포판.

- 사람: COCO yolo26s (class 0, 없으면 yolo26n) + BoT-SORT(--tracker, 기본) / 폐기물: waste10 10클래스
- 모델은 models/ 폴더에서 .engine(TensorRT) > .onnx > .pt 순으로 자동 선택
- --backend ultralytics(기본) | trt: trt는 torch/ultralytics 없이 TensorRT(.engine)·onnxruntime(.onnx)을
  직접 호출 (trt_backend.py + bytetrack.py). 메모리 ~1GB 절약, 시작 빠름. .pt 폴백은 없음
- 로직: 귀속은 '쓰레기가 처음 탐지된 프레임에 사람 박스와 겹친 경우'만 (이미 있던 쓰레기에
  나중에 다가간 사람은 연결하지 않음), LOST 객체 재식별, owner 이탈 시 발화
  (개발 PC의 work/scripts/dump_monitor.py v2와 동일 로직)
- --tts: 발화 시 '색상 + 쓰레기 종류'를 특정한 한국어 경고를 그 자리에서 합성해 방송 (tts/ 참고)
  야간 적외선/저조도 프레임에서는 색상 판정이 무의미하므로 종류만 방송한다
- --mosaic head|face: 출력물(스냅샷/영상/화면)의 사람 얼굴을 픽셀화 (mosaic.py 참고).
  추론은 원본 프레임으로 하므로 탐지 정확도에는 영향 없음
- --reid: 사람 재식별. 두 층으로 쓴다
  (1) BoT-SORT 외형 매칭: 겹쳐 지나가는 사람끼리 id가 바뀌지 않게 (매 프레임, 탐지마다 임베딩)
  (2) 갤러리(person_reid.py): 화면 밖으로 나갔다 돌아온 사람에게 이전 pid를 돌려줌.
      trt 백엔드에선 (1)이 구한 임베딩을 그대로 재사용해 추가 추론이 없다
- 감시 구역(ROI): config/roi.json의 include/exclude 다각형 (roi.py). 쓰레기 모델은 구역을 감싼
  사각형만 잘라 돌리고 구역 밖 탐지는 버린다 (사람 모델은 모자이크 때문에 화면 전체).
  --report면 대시보드에서 그린 구역을 하트비트로 받아 와 즉시 적용한다
- 설치 위치: config/site.json (site_config.py). --report면 대시보드에서 저장한 지점명·좌표를 같은
  방식으로 받아 온다. 방송 장소명은 site.json의 name (없을 때만 --tts-location / .env CAM_LOCATION)
- --report 부가 기능: 회수 판정(retrieval.py) -> event_update, 전후 5초 클립(clip.py) -> media.clip,
  5초 썸네일(heartbeat.ThumbnailUploader)

usage:
  python3 dump_monitor_jetson.py --source rtsp://... --name cam01
  python3 dump_monitor_jetson.py --source 0 --name webcam --show
  python3 dump_monitor_jetson.py --source clip.mp4 --name test --no-save-video
  python3 dump_monitor_jetson.py --source 0 --name cam01 --tts --tts-location "정문 수거함 앞"
  python3 dump_monitor_jetson.py --source 0 --name cam01 --mosaic face --reid
"""
import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from mosaic import FaceMosaic
from person_reid import PersonReID, load_encoder as load_reid_encoder
from event_sender import ulid
from recorder import OutputSpace, RotatingRecorder
from roi import RoiConfig, crop_rect, is_monitored
from site_config import SiteConfig
from stream import FrameSource

try:                                          # TTS는 선택 기능 — 없어도 감시는 동작
    from tts import Announcer, build_phrase, dominant_color_name
    from tts.color_naming import crop_box, night_mode
except Exception as _tts_err:                 # noqa: BLE001
    Announcer = build_phrase = dominant_color_name = crop_box = night_mode = None
    _TTS_IMPORT_ERROR = _tts_err
else:
    _TTS_IMPORT_ERROR = None

HERE = Path(__file__).parent

WASTE_CONF = 0.15        # 유리창 너머 원거리 봉투가 0.07~0.28로 흔들려 0.20 -> 0.15 (2026-10-04 현장 테스트).
                         # 오탐은 '첫 탐지 때 사람과 겹침' 규칙과 배경 비교가 거른다
PERSON_CONF = 0.25       # 유리창 너머 저대비 보행자가 0.25~0.35에서 깜빡여 0.35 -> 0.25 (2026-10-04 현장 테스트)
# (ultralytics 백엔드) 트래커 설정 명시 고정: 버전별 기본 트래커가 달라 pid 특성이 흔들리는 것 방지
TRACKER_YAML = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}
STABLE_AGE = 12          # 폐기물이 '잔류'로 간주되는 누적 탐지 프레임
MISS_TOLERANCE = 15      # 이 프레임까지는 ACTIVE 유지 (탐지 깜빡임 허용). 10fps 기준 1.5초
                         # (15fps 때 8프레임이었는데 10fps 현장에서 0.8초 넘는 끊김이 잦았다)
LOST_KEEP = 300          # LOST 상태 보관 프레임 (재식별 대상)
REID_DIST_FACTOR = 1.2   # 재식별 허용 거리: 객체 대각선의 N배
REID_HIST_MIN = 0.5      # 재식별 색상 유사도 하한
SAME_SPOT_IOU = 0.5      # 사라졌던 물체와 이만큼 겹치는 자리에서 다시 잡히면 색·종류와 상관없이 같은 물체.
                         # 뿌연 화면에선 색 비교가 실패해 같은 봉투가 새 물체로 다시 태어나며
                         # 투기자 연결과 누적 탐지 수를 잃었다 (2026-10-04 현장 테스트)
CARRY_IOU = 0.30         # 사람 박스와 이 이상 겹치면 '소지 쓰레기'
CARRY_WINDOW = 60        # 소지품 목격 후 귀속 후보로 보는 프레임 수
NEAR_FACTOR = 2.0        # 사람-객체 근접 판정: 객체 대각선의 N배
BIRTH_WINDOW = 40        # 사람 이력·배경 비교용 과거 프레임 보관 길이 (귀속 판정에는 안 씀)
DEDUP_IOU = 0.6          # 클래스 무관 탐지 중복 제거 임계
CLEAR_FRAMES = 5         # 사람 부재 연속 프레임 후 발화
STATIC_FRAMES = 20       # 발화 전 제자리에 멈춰 있어야 하는 최소 프레임 (~1.3s @15fps)
STATIC_TOL = 0.3         # 기준 위치에서 중심이 객체 대각선의 이 비율 넘게 벗어나면 '이동 중'
                         # -> 사람과 함께 움직이는 물체(우산·가방)는 버려진 것이 아니다 (2026-09-26 오탐)
LOST_LINK_FRAMES = 10    # 탐지가 빠진 지 이 프레임 이내인 사람은 칼만 예측 위치에 '있는' 것으로 본다
                         # (귀속·투기자 근처 판정. 사람 탐지 깜빡임 한 번에 투기자를 놓치지 않게)
ROI_GRACE = 5            # 구역 변경 후 이 프레임 안에 처음 잡힌 물체는 원래 있던 것으로 본다
                         # (구역을 넓히면 그 자리에 놓여 있던 쓰레기가 '새로' 탐지되므로)
BG_MIN_GAP = 5           # 배경 비교: 등장 최소 N프레임 전 프레임과 대조
BG_MAD_THRESH = 0.05     # 영역 평균절대차(0~1)가 이보다 작으면 '원래 있던 물체'로 간주
                         # (야간 저대비에서 실제 투기까지 억제하지 않도록 보수적으로 설정)

WASTE_NAMES = ["쓰레기봉투", "대형가구", "가전제품", "페트", "캔", "병",
               "스티로폼", "종이박스", "의류", "플라스틱"]


ENV_FILE = Path(os.environ.get("CLEAN_GUARD_ENV") or (HERE / ".env"))


def load_env_file(path=None):
    """KEY=VALUE 설정 파일을 환경변수로 읽는다 (외부 의존성 없음).

    RTSP 비밀번호를 명령줄에도 셸 이력에도 남기지 않으려면 여기에 적는다.
    이미 설정된 환경변수는 덮어쓰지 않는다 — 명시적으로 넘긴 쪽이 항상 우선.
    비밀번호가 들어가는 파일이므로 권한이 느슨하면 경고한다.
    """
    p = Path(path) if path else ENV_FILE
    if not p.is_file():
        return None
    try:
        if p.stat().st_mode & 0o077:
            print(f"[설정 경고] {p.name}을 다른 사용자가 읽을 수 있습니다 -> chmod 600 {p}")
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            if k:
                os.environ.setdefault(k, v)
        return p
    except OSError as e:                                         # noqa: BLE001
        print(f"[설정 경고] {p}를 읽을 수 없음 ({e}) -> 무시하고 계속")
        return None


# 사람 모델 기본값: 앞에서부터 models/에 파일이 있는 것. s는 n보다 GPU 시간 +1.4ms(384x640, 보드 실측)
# 대신 유리창 너머 저대비 보행자를 더 잘 잡았다 (2026-10-04, 사용자 결정으로 s 채택)
PERSON_MODELS = ("person_yolo26s", "person_yolo26n")


def default_person_model():
    for stem in PERSON_MODELS:
        if any((HERE / "models" / (stem + e)).exists() for e in (".engine", ".onnx", ".pt")):
            return stem
    return PERSON_MODELS[-1]


def load_model(stem_or_path: str):
    """(ultralytics 백엔드) 모델 로드. 경로가 주어지면 그대로, 아니면 models/에서
    .engine > .onnx > .pt 순으로 시도한다. 로드/워밉업이 실패하면 다음 포맷으로 폴백
    (예: JetPack 업그레이드로 .engine이 무효화된 경우 .onnx/.pt로 계속 동작)."""
    from ultralytics import YOLO          # torch를 끌어오므로 이 백엔드를 쓸 때만 import
    if stem_or_path and Path(stem_or_path).suffix:
        return YOLO(stem_or_path), stem_or_path
    cands = [HERE / "models" / (stem_or_path + ext) for ext in (".engine", ".onnx", ".pt")]
    cands = [p for p in cands if p.exists()]
    if not cands:
        raise FileNotFoundError(f"models/{stem_or_path}.(engine|onnx|pt) 없음")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for p in cands:
        try:
            m = YOLO(str(p))
            m.predict(dummy, verbose=False)          # 실제 추론까지 확인
            return m, str(p)
        except Exception as e:
            print(f"[경고] {p.name} 사용 불가 ({type(e).__name__}: {str(e)[:120]}) -> 다음 포맷 시도")
    raise RuntimeError(f"{stem_or_path}: 사용 가능한 모델 포맷이 없음")


class UltralyticsBackend:
    """ultralytics YOLO + 내장 트래커(BoT-SORT/ByteTrack). .engine/.onnx/.pt 모두 지원 (torch 상주)."""

    name = "ultralytics"

    def __init__(self, waste, person, tracker="botsort"):
        self.waste, self.waste_path = load_model(waste or "waste10_yolo26n")
        self.person, self.person_path = load_model(person or default_person_model())
        self.tracker = tracker
        self.tracker_cfg = TRACKER_YAML[tracker]
        self.last_embs = None           # 트래커 임베딩은 꺼낼 수 없어 갤러리가 따로 구한다

    def recent_lost(self, max_age):
        """ultralytics 트래커의 최근 놓친 트랙 [(id, 예측 xyxy)]. 내부 구조가 다르면 빈 목록."""
        try:
            tr = self.person.predictor.trackers[0]
            return [(int(t.track_id), [float(v) for v in t.xyxy]) for t in tr.lost_stracks
                    if t.is_activated and tr.frame_id - t.frame_id <= max_age]
        except Exception:                                         # noqa: BLE001
            return []

    def attach_reid(self, encoder, model_path=None):
        """BoT-SORT에 외형 매칭을 켠다. ultralytics는 설정 파일로만 받으므로 yaml을 만들어 쓴다
        (gmc none: 고정 카메라. 나머지는 ultralytics botsort.yaml 기본값)."""
        if self.tracker != "botsort" or encoder is None:
            return
        from person_reid import find_models
        cands = find_models(model_path)
        if not cands:
            return
        cfg = HERE / "cache" / "botsort_reid.yaml"
        cfg.parent.mkdir(exist_ok=True)
        cfg.write_text(
            "tracker_type: botsort\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\n"
            "new_track_thresh: 0.25\ntrack_buffer: 30\nmatch_thresh: 0.8\nfuse_score: True\n"
            "gmc_method: none\nproximity_thresh: 0.5\nappearance_thresh: 0.8\n"
            f"with_reid: True\nmodel: {cands[0]}\n", encoding="utf-8")
        self.tracker_cfg = str(cfg)

    def track_persons(self, frame):
        pr = self.person.track(frame, classes=[0], conf=PERSON_CONF, persist=True,
                               tracker=self.tracker_cfg, verbose=False)[0]
        persons = []
        if pr.boxes is not None:
            ids = pr.boxes.id
            for k, b in enumerate(pr.boxes):
                pid = int(ids[k]) if ids is not None else -(k + 1)
                persons.append((pid, b.xyxy[0].tolist()))
        return persons

    def detect_waste(self, frame):
        wr = self.waste.predict(frame, conf=WASTE_CONF, verbose=False)[0]
        if wr.boxes is None:
            return []
        return [(b.xyxy[0].tolist(), int(b.cls), float(b.conf)) for b in wr.boxes]

    def reid_encoder(self, model_path):
        return load_reid_encoder(model_path)


class TrtBackend:
    """torch 없는 백엔드: TensorRT(.engine) / onnxruntime(.onnx) 직접 호출 + 독립 BoT-SORT/ByteTrack."""

    name = "trt"

    def __init__(self, waste, person, tracker="botsort"):
        from bytetrack import BoTSORT, ByteTrack
        from trt_backend import load_detector
        self.waste, self.waste_path = load_detector(waste or "waste10_yolo26n")
        self.person, self.person_path = load_detector(person or default_person_model())
        # ultralytics도 frame_rate=30 고정 (buffer 30프레임)
        self.tracker = (BoTSORT if tracker == "botsort" else ByteTrack)(frame_rate=30)
        self.encoder = None
        self.last_embs = None           # {pid: 이번 프레임 임베딩} — 갤러리(PersonReID)가 재사용

    def recent_lost(self, max_age):
        return self.tracker.recent_lost(max_age)

    def attach_reid(self, encoder, model_path=None):
        """BoT-SORT 외형 매칭용 인코더. ByteTrack이면 쓰지 않는다."""
        if hasattr(self.tracker, "frame_feats"):
            self.encoder = encoder

    def track_persons(self, frame):
        boxes, conf, _ = self.person.detect(frame, conf=PERSON_CONF, classes=[0])
        feats = None
        if self.encoder is not None and len(boxes):
            xywh = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
            try:
                feats = self.encoder(frame, xywh)
            except Exception as e:                                # noqa: BLE001
                print(f"[ReID 경고] 트래커 임베딩 실패 ({type(e).__name__}: {str(e)[:80]})")
        tracks = self.tracker.update(np.c_[boxes, conf], feats)
        self.last_embs = dict(self.tracker.frame_feats) if feats is not None else None
        return [(tid, box) for tid, box, _s in tracks]

    def detect_waste(self, frame):
        boxes, conf, cls = self.waste.detect(frame, conf=WASTE_CONF)
        return [(b.tolist(), int(c), float(s)) for b, c, s in zip(boxes, cls, conf)]

    def reid_encoder(self, model_path):
        from trt_backend import load_reid
        try:
            enc, path = load_reid(model_path or "yolo26n-reid")
            print(f"[ReID] 인코더 로드: {Path(path).name} ({self.name})")
            return enc
        except Exception as e:                                    # noqa: BLE001
            print(f"[ReID 경고] {type(e).__name__}: {str(e)[:120]} -> 재식별 비활성")
            return None


def make_backend(name, waste, person, tracker="botsort"):
    return (TrtBackend if name == "trt" else UltralyticsBackend)(waste, person, tracker)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def diag(b):
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def color_hist(frame, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:max(y1 + 2, y2), x1:max(x1 + 2, x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def hist_sim(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def region_unchanged(frame_gray, obj, frame_buffer):
    """객체 등장 이전의 과거 프레임과 현재 프레임에서 객체 영역을 비교.
    거의 같으면(True) '원래 있던 물체'가 뒤늦게 탐지된 것 -> 발화 금지."""
    x1, y1, x2, y2 = map(int, obj.box)
    x1, y1 = max(0, x1), max(0, y1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    cur = frame_gray[y1:y2, x1:x2]
    pre = [e for e in frame_buffer if e[0] <= obj.first_seen - BG_MIN_GAP]
    if not pre:
        # 영상/스트림 시작 직후 등장 -> 새 물체라는 증거가 없으니 기존 적치물 취급
        return obj.first_seen <= BG_MIN_GAP + 2
    # 사람이 그 영역에 없던 가장 오래된 프레임과 비교
    for fi, gray, plist in pre:
        if any(iou(obj.box, pb) > 0.05 for _, pb in plist):
            continue
        old = gray[y1:y2, x1:x2]
        if old.shape != cur.shape or old.size == 0:
            continue
        mad = float(np.mean(np.abs(cur.astype(np.int16) - old.astype(np.int16)))) / 255.0
        return mad < BG_MAD_THRESH
    return False           # 과거 프레임마다 사람이 있었으면 판단 보류(발화 허용)


def held_overhead(obj_box, pbox):
    """사람 박스 바로 위에 얹힌 물체 (우산·머리에 인 짐).
    우산이 상체를 가리면 사람 박스가 다리만 남아 물체가 박스 '위 바깥'에 놓이므로
    inside/IoU로는 소지품으로 안 잡힌다. 가로로 절반 이상 겹치고 물체 아랫변이
    사람 윗변 근처(위로 물체 높이 절반 ~ 아래로 사람 높이 절반)에 있으면 소지로 본다."""
    ow, oh = obj_box[2] - obj_box[0], obj_box[3] - obj_box[1]
    ph = pbox[3] - pbox[1]
    overlap = min(obj_box[2], pbox[2]) - max(obj_box[0], pbox[0])
    return (ow > 0 and overlap > 0.5 * ow
            and pbox[1] - 0.5 * oh <= obj_box[3] <= pbox[1] + 0.5 * ph)


def person_near(obj_box, person_boxes, factor=NEAR_FACTOR):
    cx, cy = center(obj_box)
    d0 = diag(obj_box)
    for p in person_boxes:
        px, py = center(p)
        fx, fy = (p[0] + p[2]) / 2, p[3]
        d = min(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5,
                ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5)
        if d < factor * max(d0, 40):
            return True
    return False


def save_snapshot(path, image, box, label, width=0):
    """증거 스냅샷 저장. width>0이고 원본이 더 크면 그 폭으로 줄여서 쓴다.

    축소를 먼저 하고 박스·라벨을 그린다. 원본에 그린 뒤 줄이면 선과 글자가 같이
    작아져 640에서 잘 안 보이기 때문이다. 박스 좌표는 축소 비율로 환산한다.
    분석·색상 판정은 원본 해상도로 이미 끝난 뒤이므로 정확도에는 영향이 없다.
    """
    s = 1.0
    if width and image.shape[1] > width:
        s = width / float(image.shape[1])
        img = cv2.resize(image, (int(width), max(1, int(round(image.shape[0] * s)))),
                         interpolation=cv2.INTER_AREA)
    else:
        img = image.copy()
    w = img.shape[1]
    th = max(1, int(round(w / 320)))                 # 폭에 비례한 선 두께
    fs = max(0.4, w / 1100.0)                        # 폭에 비례한 글자 크기
    x1, y1, x2, y2 = [int(round(v * s)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), th)
    cv2.putText(img, label, (int(w * 0.02), int(round(28 * fs / 0.58))),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th)
    return cv2.imwrite(str(path), img)


def best_color_crop(frame, box):
    """색상 판정용 64x64 크롭 사본. TTS 모듈이 없으면 None."""
    if crop_box is None:
        return None
    c = crop_box(frame, box)
    if c is None or c.size == 0:
        return None
    return cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA)


def overlap_area(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def link_owner(o, box, cls, hist, persons, carried, frame_idx, recent=()):
    """새 쓰레기 객체 o의 투기자 귀속 — '처음 탐지된 프레임'에 박스가 겹친 사람만 후보다.

    이미 놓여 있던 쓰레기에 나중에 사람이 다가간 경우(산책·수거·구경)는 연결하지 않는다.
    등장 이후 프레임에서 사람을 다시 찾지 않으므로, 연결되지 않은 물체는 끝까지 발화하지 않는다.
    겹친 사람이 여럿이면 직전까지 들고 있던 소지품(carried)과 색·종류가 맞는 사람, 그다음 겹친 면적 순.
    이번 프레임에 탐지된 사람이 없으면 recent(방금 탐지가 빠진 사람의 예측 위치)로 본다.
    반환: 로그용 설명 문자열.
    """
    touching, via = [], ""
    for cands, tag in ((persons, ""), (recent, " · 탐지 깜빡임 보정")):
        touching = [(pid, overlap_area(box, pb)) for pid, pb in cands]
        touching = [(pid, a) for pid, a in touching if a > 0]
        if touching:
            via = tag
            break
    if not touching:
        return "겹친 사람 없음 -> 연결 안 함 (발화 대상 아님)"

    def carry_score(pid):
        info = carried.get(pid)
        if info is None or frame_idx - info["last_frame"] > CARRY_WINDOW:
            return 0.0
        return hist_sim(hist, info["hist"]) + (0.5 if info["cls"] == cls else 0.0)

    scored = [(carry_score(pid), a, pid) for pid, a in touching]
    best_s, _, pid = max(scored)
    o.owner_pid, o.person_at_birth = pid, True
    o.owner_strong = best_s >= 0.6          # 소지품 매칭까지 됐는가 (이벤트의 owner_matched)
    return f"투기자 P{pid} 연결{' (소지품 일치)' if o.owner_strong else ''}{via}"


class WasteObject:
    _next_id = 1

    def __init__(self, box, cls, conf, frame_idx, hist, frame=None):
        self.id = WasteObject._next_id
        WasteObject._next_id += 1
        self.box = box
        self.cls = cls
        self.conf = conf
        self.hist = hist
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.age = 1
        self.anchor = center(box)       # 정지 판정 기준점 — 움직이면 갱신
        self.still_since = frame_idx    # 기준점에 머문 시작 프레임
        self.baseline = frame_idx <= 2
        self.person_at_birth = False
        self.owner_pid = None
        self.owner_strong = False
        self.fired = False
        self.clear_count = 0
        self.last_reason = None     # 판정 로그(--log-objects) 중복 방지
        # 방송 색상은 '가장 확신했던 프레임'의 크롭에서 뽑는다 (발화 프레임 1장보다 안정적)
        self.best_conf = -1.0
        self.best_crop = None
        self.best_night = None      # 그 크롭을 딴 프레임의 조명 상태 ("적외선"/"저조도"/None)
        self._update_crop(frame, box, conf)

    def _update_crop(self, frame, box, conf):
        if frame is None or conf <= self.best_conf:
            return
        c = best_color_crop(frame, box)
        if c is not None:
            self.best_conf, self.best_crop = conf, c
            self.best_night = night_mode(frame) if night_mode is not None else None

    def seen(self, box, conf, frame_idx, hist, frame=None):
        self._update_crop(frame, box, conf)
        cx, cy = center(box)
        if ((cx - self.anchor[0]) ** 2 + (cy - self.anchor[1]) ** 2) ** 0.5 \
                > STATIC_TOL * max(diag(box), 20):
            self.anchor, self.still_since = (cx, cy), frame_idx
        self.box, self.conf, self.last_seen = box, conf, frame_idx
        self.age += 1
        if hist is not None:
            self.hist = hist if self.hist is None else 0.7 * self.hist + 0.3 * hist


def main():
    env_path = load_env_file()          # .env를 먼저 읽어 아래 기본값으로 쓴다
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="영상 파일 / RTSP URL / 웹캠 번호. 생략하면 .env 또는 환경변수의 "
                         "CAM_URL을 쓴다 (비밀번호가 명령줄·ps·셸 이력에 남지 않는다)")
    ap.add_argument("--name", default=os.environ.get("CAM_NAME") or "cam",
                    help="출력 폴더명 (.env의 CAM_NAME)")
    ap.add_argument("--out", default=str(HERE / "output"), help="출력 루트")
    ap.add_argument("--backend", default="ultralytics", choices=["ultralytics", "trt"],
                    help="ultralytics=YOLO+내장 트래커(.engine/.onnx/.pt, torch 상주) / "
                         "trt=torch 없이 TensorRT·onnxruntime 직접 호출 (.engine/.onnx)")
    ap.add_argument("--waste-model", default=None)
    ap.add_argument("--person-model", default=None,
                    help="사람 모델 (기본: models/의 person_yolo26s, 없으면 person_yolo26n)")
    ap.add_argument("--show", action="store_true", help="화면 표시 (모니터 연결 시)")
    ap.add_argument("--no-save-video", action="store_true", help="주석 영상 저장 생략")
    ap.add_argument("--push", action="store_true",
                    help="주석 프레임을 관제 서버로 SRT 송출 (720p/10fps H.264). "
                         ".env의 SRT_HOST/SRT_STREAMID 필요. 끊기면 자동 재연결")
    ap.add_argument("--report", action="store_true",
                    help="이벤트 JSON+스냅샷을 관제 서버 API로 전송 (event_sender.py). "
                         ".env의 API_URL/DEVICE_TOKEN/DEVICE_ID/SITE_ID 필요. "
                         "실패 시 디스크 큐에 보관 후 재전송 — 회선이 끊겨도 유실 없음")
    ap.add_argument("--heartbeat-sec", type=float, default=30.0,
                    help="--report 사용 시 장치 상태 하트비트 주기(초). 서버는 90초 미수신이면 "
                         "offline 판정. 0=끔 (기본 30)")
    ap.add_argument("--roi-file", default=str(HERE / "config" / "roi.json"),
                    help="감시 구역 파일. 없으면 화면 전체 감시. --report면 서버 설정으로 자동 갱신")
    ap.add_argument("--site-file", default=str(HERE / "config" / "site.json"),
                    help="설치 위치 파일 (지점명·좌표). --report면 서버 설정으로 자동 갱신. "
                         "방송 장소명은 여기 name을 쓰고, 파일이 없을 때만 --tts-location")
    ap.add_argument("--clip-sec", type=float, default=5.0,
                    help="--report 사용 시 사건 전후 이 초만큼 mp4 클립을 만들어 업로드. 0=끔 "
                         "(기본 5 — 서버도 전후 5초로 잘라 저장한다)")
    ap.add_argument("--thumbnail-sec", type=float, default=5.0,
                    help="--report 사용 시 대시보드 썸네일 업로드 주기(초). 0=끔 (기본 5)")
    ap.add_argument("--retrieve-window-min", type=float, default=10.0,
                    help="사건 후 이 시간 안에 물체가 사라지면 '회수'로 보고 (기본 10분). 0=회수 판정 끔")
    ap.add_argument("--retrieve-confirm-sec", type=float, default=120.0,
                    help="사라진 뒤 이 시간 동안 다시 안 보여야 회수로 확정 (사람에 가려진 시간 제외, "
                         "기본 120초). 짧으면 탐지 깜빡임을 회수로 오판한다")
    ap.add_argument("--tts", action="store_true", help="이벤트 발생 시 음성 경고 방송")
    ap.add_argument("--tts-mode", default="live", choices=["live", "cache"],
                    help="live=이벤트마다 문구를 즉석 합성해 방송(기본) / "
                         "cache=prerender_tts.py로 만든 사전 렌더링 wav만 재생")
    ap.add_argument("--tts-backend", default="supertonic", choices=["supertonic", "null"],
                    help="합성 백엔드 (null=무음, 배선 점검용)")
    ap.add_argument("--tts-cache", default=None,
                    help="사전 렌더링 wav 폴더 (기본 cache/tts). live 모드에선 합성 실패 시 폴백용")
    ap.add_argument("--tts-location", default=os.environ.get("CAM_LOCATION") or None,
                    help="방송에 넣을 장소명 (.env의 CAM_LOCATION). config/site.json이 있으면 "
                         "그쪽 name이 우선. cache 모드면 그 장소 전용 캐시가 있어야 함")
    ap.add_argument("--tts-cooldown", type=float, default=12.0,
                    help="방송 간 최소 간격(초). 문장 길이(약 9초)보다 크게 두어 대기열 적체를 막는다")
    ap.add_argument("--tts-repeat-window", type=float, default=30.0,
                    help="같은 (색상,종류) 조합 재방송 억제 시간(초)")
    ap.add_argument("--tts-no-runtime-synth", action="store_true",
                    help="(cache 모드) 캐시 미스 시 즉석 합성하지 않고 방송을 건너뜀")
    ap.add_argument("--tts-model", default="supertonic-2", help="합성 모델")
    ap.add_argument("--tts-voice", default=os.environ.get("TTS_VOICE") or "M4",
                    help="목소리 (M1~M5/F1~F5). 기본은 .env의 TTS_VOICE, 없으면 M4")
    ap.add_argument("--tts-steps", type=int, default=int(os.environ.get("TTS_STEPS") or 8),
                    help="합성 품질(디노이징 스텝, 1~100). 클수록 고품질·느림. "
                         "기본은 .env의 TTS_STEPS, 없으면 8")
    ap.add_argument("--tts-say-location", action="store_true",
                    default=os.environ.get("TTS_SAY_LOCATION", "").strip() in ("1", "true", "yes"),
                    help="방송 문구 앞에 장소명(site.json name)을 붙인다. 기본은 끔 "
                         "(2026-10-05 사용자 결정). .env TTS_SAY_LOCATION=1")
    ap.add_argument("--mosaic", default="off", choices=["off", "head", "face"],
                    help="출력물의 사람 얼굴 픽셀화. head=사람 박스 상단(모델 불필요) / "
                         "face=YuNet 얼굴 탐지(못 찾으면 head 폴백)")
    ap.add_argument("--face-model", default=None,
                    help="YuNet onnx 경로 (기본 models/face_detection_yunet_2023mar.onnx)")
    ap.add_argument("--raw-snapshot", action="store_true",
                    help="--mosaic 사용 시 모자이크 없는 원본 스냅샷도 event_NNNN_raw.jpg로 저장 "
                         "(증거 보관용. 접근 통제 필요)")
    ap.add_argument("--reid", action="store_true",
                    help="사람 재식별 활성화 (models/yolo26n-reid.engine|onnx 필요)")
    ap.add_argument("--reid-model", default=None, help="Re-ID 모델 경로 (기본 models/yolo26n-reid.*)")
    ap.add_argument("--log-objects", action="store_true",
                    default=os.environ.get("LOG_OBJECTS", "").strip() in ("1", "true", "yes"),
                    help="쓰레기 물체마다 등장·투기자 연결·판정 단계를 로그로 남김 (.env LOG_OBJECTS=1)")
    ap.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack"],
                    help="사람 트래커. botsort=IoU+외형(--reid일 때) 매칭 / bytetrack=IoU만 (이전 동작)")
    ap.add_argument("--reid-ttl", type=float, default=600.0,
                    help="사라진 사람을 재식별 대상으로 기억하는 시간(초). 회수 판정 시간과 맞출 것")
    # --- 상시 운영(RTSP) 관련 ---
    ap.add_argument("--video-segment-min", type=float, default=60.0,
                    help="주석 영상을 이 분 단위로 분할 (기본 60). 상시 운영에서 디스크 고갈 방지")
    ap.add_argument("--video-keep", type=int, default=24,
                    help="보관할 영상 세그먼트 개수. 초과분은 오래된 것부터 삭제 (기본 24)")
    ap.add_argument("--min-free-mb", type=float, default=2000.0,
                    help="출력 폴더 여유가 이 값 아래면 오래된 출력물부터 삭제하고, "
                         "그래도 모자라면 녹화를 중단한다 (기본 2000MB). "
                         "출력을 RAM(/dev/shm)에 두면 메모리 여유에 맞춰 더 크게 잡을 것")
    ap.add_argument("--purge-order", default="video-first",
                    choices=["video-first", "oldest-first"],
                    help="공간 회수 순서. video-first=주석 영상을 먼저 다 지우고 "
                         "그래도 모자라면 스냅샷(각각 오래된 것부터) / "
                         "oldest-first=종류 무관 순수 FIFO")
    ap.add_argument("--keep-days", type=float, default=90.0,
                    help="증거(스냅샷·클립·방송 음성) 보존 일수. 지나면 삭제. 0=기간 삭제 안 함 (기본 90)")
    ap.add_argument("--raw-keep-days", type=float, default=30.0,
                    help="모자이크 없는 원본 스냅샷(*_raw.jpg) 보존 일수 (기본 30)")
    ap.add_argument("--snapshot-width", type=int, default=640,
                    help="이벤트 스냅샷 저장 폭(px). 0=원본 그대로. 기본 640 "
                         "(분석은 원본 해상도로 하고 저장만 줄인다)")
    ap.add_argument("--reconnect-delay", type=float, default=2.0,
                    help="스트림 끊김 후 첫 재연결 대기(초). 실패할수록 2배씩 늘어난다")
    ap.add_argument("--reconnect-max-delay", type=float, default=30.0,
                    help="재연결 대기 상한(초)")
    ap.add_argument("--reconnect-max-retries", type=int, default=0,
                    help="재연결 최대 시도 횟수. 0=무한 (상시 감시 기본값)")
    ap.add_argument("--no-drop-frames", action="store_true",
                    help="라이브 입력에서 밀린 프레임을 버리지 않는다. "
                         "전 프레임 분석이 필요한 검증용이며, 상시 운영에서는 지연이 누적된다")
    args = ap.parse_args()
    log_objects = args.log_objects
    # 탐지 임계는 현장마다 다르다 (유리창 반사·거리) -> .env로 코드 수정 없이 조정
    global WASTE_CONF, PERSON_CONF
    try:
        WASTE_CONF = float(os.environ.get("WASTE_CONF") or WASTE_CONF)
        PERSON_CONF = float(os.environ.get("PERSON_CONF") or PERSON_CONF)
    except ValueError as e:
        ap.error(f".env의 WASTE_CONF/PERSON_CONF는 숫자여야 합니다 ({e})")

    if env_path is not None:
        print(f"[설정] {env_path} 읽음")
    if log_objects:
        print("[물체] 판정 과정 로그 활성 (등장·투기자 연결·판정 단계)")
    print(f"[탐지 임계] 쓰레기 {WASTE_CONF:.2f} · 사람 {PERSON_CONF:.2f}")
    # --source가 없으면 CAM_URL에서 읽는다. 명령줄에 두면 ps aux로 비밀번호가 보이고
    # 셸 이력에도 남으므로, 자격증명이 들어간 RTSP URL은 이쪽을 쓰는 편이 안전하다.
    if not args.source:
        args.source = os.environ.get("CAM_URL", "").strip()
        if not args.source:
            ap.error(f"카메라 주소가 없습니다. {ENV_FILE}에 CAM_URL을 적거나 "
                     f"--source로 넘기세요 (.env.example 참고)")
        print("[입력] CAM_URL 사용")

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    announcer = None
    if args.tts:
        if Announcer is None:
            print(f"[TTS 경고] tts 모듈 로드 실패 ({_TTS_IMPORT_ERROR}) -> 방송 없이 계속")
        else:
            tts_kw = ({} if args.tts_backend == "null"
                      else {"model": args.tts_model, "voice": args.tts_voice,
                            "steps": args.tts_steps})
            announcer = Announcer(
                mode=args.tts_mode,
                cache_dir=args.tts_cache, backend=args.tts_backend,
                location=args.tts_location if args.tts_say_location else None,
                cooldown=args.tts_cooldown,
                repeat_window=args.tts_repeat_window,
                allow_runtime_synth=not args.tts_no_runtime_synth, **tts_kw)
    backend = make_backend(args.backend, args.waste_model, args.person_model, args.tracker)
    reid = PersonReID(backend.reid_encoder(args.reid_model) if args.reid else None, ttl_s=args.reid_ttl)
    if args.reid and reid.enabled:
        backend.attach_reid(reid.enc, args.reid_model)
        print(f"[ReID] 활성 · 갤러리 보관 {args.reid_ttl:.0f}s")
    print(f"[트래커] {args.tracker}"
          + (" + 외형 매칭" if args.tracker == "botsort" and args.reid and reid.enabled else " (IoU)"))
    mosaic = FaceMosaic(args.mosaic, model_path=args.face_model)
    if mosaic.enabled:
        print(f"[모자이크] {mosaic.mode} 모드"
              + (" · 원본 스냅샷 별도 저장" if args.raw_snapshot else ""))
    print(f"backend={backend.name} waste={Path(backend.waste_path).name} "
          f"person={Path(backend.person_path).name} -> {out_dir}")

    try:
        src = FrameSource(args.source,
                          reconnect_delay=args.reconnect_delay,
                          reconnect_max_delay=args.reconnect_max_delay,
                          reconnect_max_retries=args.reconnect_max_retries,
                          drop_stale=not args.no_drop_frames).open()
    except RuntimeError as e:
        print(e)
        return
    fps, W, H = src.fps, src.width, src.height
    if src.live:
        print(f"[스트림] 라이브 입력 {W}x{H} @{fps:.0f}fps · "
              + ("밀린 프레임 드롭" if src.drop_stale else "전 프레임 처리")
              + (" · 재연결 무한" if not args.reconnect_max_retries
                 else f" · 재연결 최대 {args.reconnect_max_retries}회"))
    recorder = RotatingRecorder(out_dir, fps, (W, H),
                                segment_min=args.video_segment_min,
                                keep=args.video_keep,
                                min_free_mb=args.min_free_mb,
                                enabled=not args.no_save_video)
    if recorder.enabled:
        seg = (f"{recorder.segment_sec / 60:.0f}분" if recorder.segment_sec >= 60
               else f"{recorder.segment_sec:.0f}초")
        print(f"[녹화] {seg} 단위 분할 · 최신 {args.video_keep}개 보관 "
              f"· 여유 {args.min_free_mb:.0f}MB 미만이면 중단")
    space = OutputSpace(out_dir, min_free_mb=args.min_free_mb,
                        purge_order=args.purge_order,
                        keep_days=args.keep_days, raw_keep_days=args.raw_keep_days)
    print(f"[디스크] 여유 {space.free_mb():.0f}MB · 임계 {args.min_free_mb:.0f}MB "
          f"· 회수 순서 {args.purge_order} · 스냅샷 "
          + (f"{args.snapshot_width}px 폭" if args.snapshot_width else "원본 해상도"))
    pusher = None
    if args.push:
        # 주소·자격은 .env에서 읽는다 (명령줄에 두면 ps·셸 이력에 남는다)
        srt_host = os.environ.get("SRT_HOST", "").strip()
        srt_sid = os.environ.get("SRT_STREAMID", "").strip()
        if not srt_host or not srt_sid:
            ap.error(f"--push에는 {ENV_FILE}의 SRT_HOST와 SRT_STREAMID가 필요합니다"
                     " (.env.example 참고)")
        from srt_sender import SrtStreamer
        pusher = SrtStreamer(srt_host, int(os.environ.get("SRT_PORT", "8890")), srt_sid,
                             latency_ms=int(os.environ.get("SRT_LATENCY_MS", "200")))
        print(f"[송출] SRT {srt_host}:{pusher.port} · {pusher.size[0]}x{pusher.size[1]}"
              f"@{pusher.fps}fps x264 {pusher.kbps}kbps · 끊기면 자동 재연결")
    reporter = None
    if args.report:
        # 토큰·식별자는 .env에서 읽는다 (jetson_handoff.md §1 — 코드·git에 넣지 말 것)
        api_url = os.environ.get("API_URL", "").strip()
        api_token = os.environ.get("DEVICE_TOKEN", "").strip()
        device_id = os.environ.get("DEVICE_ID", "").strip()
        site_id = os.environ.get("SITE_ID", "").strip()
        if not (api_url and api_token and device_id and site_id):
            ap.error(f"--report에는 {ENV_FILE}의 API_URL, DEVICE_TOKEN, "
                     "DEVICE_ID, SITE_ID가 필요합니다 (.env.example 참고)")
        from event_sender import EventSender
        reporter = EventSender(api_url, api_token, device_id, site_id,
                               queue_dir=Path(args.out) / "queue" / "events",
                               min_free_mb=args.min_free_mb)
        n_queued = reporter.pending()
        print(f"[전송] {api_url} · 장치 {device_id}"
              + (f" · 이전 미전송 {n_queued}건 재시도" if n_queued else "")
              + " · 실패 시 큐 보관 후 재전송")
    roi = RoiConfig(args.roi_file,
                    base_url=os.environ.get("API_URL", "").strip() if reporter else None,
                    token=os.environ.get("DEVICE_TOKEN", "").strip() if reporter else None)
    print(f"[ROI] {roi.describe()}" + (" · 서버 설정 동기화" if reporter else "")
          + ("" if Path(args.roi_file).is_file() else f" ({Path(args.roi_file).name} 없음)"))
    site = SiteConfig(args.site_file,
                      base_url=os.environ.get("API_URL", "").strip() if reporter else None,
                      token=os.environ.get("DEVICE_TOKEN", "").strip() if reporter else None,
                      on_apply=lambda c: (announcer.set_location(place_name())
                                          if announcer else None))

    def place_name():
        """방송 장소명: 끔(기본)이면 None. 켜면 site.json의 name, 아직 못 받았으면
        --tts-location(.env CAM_LOCATION)."""
        if not args.tts_say_location:
            return None
        return (site.name or None) if site.loaded else args.tts_location
    print(f"[설치 위치] {site.describe()}" + (" · 서버 설정 동기화" if reporter else "")
          + (f" · 방송 장소명 '{place_name() or '(없음)'}'" if args.tts_say_location
             else " · 방송에 장소명 안 넣음"))
    if announcer is not None and site.loaded and args.tts_say_location:
        announcer.set_location(place_name())
    roi_seen = (roi.generation, roi.zones)     # 구역이 바뀌면 기존 물체를 재판정하기 위함
    roi_changed_at = -ROI_GRACE - 1            # 마지막 구역 변경 프레임
    clipper = thumb = retrieval = None
    if reporter is not None:
        api_url, api_token = os.environ["API_URL"].strip(), os.environ["DEVICE_TOKEN"].strip()
        if args.clip_sec > 0:
            from clip import EventClipper
            clipper = EventClipper(lambda eid, path: reporter.attach(eid, {"clip": path}),
                                   pre_s=args.clip_sec, post_s=args.clip_sec)
            print(f"[클립] 사건 전후 {args.clip_sec:g}초 720p H.264 -> 사건에 첨부")
        if args.thumbnail_sec > 0:
            from heartbeat import ThumbnailUploader
            thumb = ThumbnailUploader(api_url, api_token, os.environ["DEVICE_ID"].strip(),
                                      interval=args.thumbnail_sec)
        if args.retrieve_window_min > 0:
            from retrieval import RetrievalWatch

            def on_retrieved(eid, after_s):
                print(f"[회수] 사건 {eid}: {after_s}초 뒤 물체 사라짐 -> retrieved 보고", flush=True)
                reporter.update(eid, "retrieved", after_s=after_s)
            # 판정 중인 사건은 큐 옆에 적어 둔다 — 재시작돼도 이어서 판정 (라이브 입력만)
            retrieval = RetrievalWatch(on_retrieved, window_s=args.retrieve_window_min * 60,
                                       confirm_s=args.retrieve_confirm_sec,
                                       state_path=(Path(args.out) / "queue" / "retrieval.json"
                                                   if src.live else None))
            print(f"[회수] 사건 후 {args.retrieve_window_min:g}분 안에 사라지고 "
                  f"{args.retrieve_confirm_sec:g}초간 재등장 없으면 보고"
                  + (f" · 재시작 전 판정 중이던 {retrieval.n_restored}건 이어서"
                     if retrieval.n_restored else ""))
    next_expire = 0.0                            # 보존 기간 삭제는 1시간에 한 번
    events_f = open(out_dir / "events.jsonl", "a", encoding="utf-8")

    objects, carried = [], {}
    person_hist = deque(maxlen=BIRTH_WINDOW * 2)
    frame_buffer = deque(maxlen=BIRTH_WINDOW + 10)   # (frame_idx, gray, persons)
    n_events, frame_idx = 0, 0
    show_ready = False              # --show 창은 첫 프레임에서 한 번만 만든다
    # 하트비트가 읽는 감시 루프 값 (추론 시간은 지수이동평균 ms)
    live = {"persons": 0, "infer_ms": {"person": None, "waste": None}}

    def note_infer(kind, t_start):
        ms = (time.perf_counter() - t_start) * 1000
        prev = live["infer_ms"][kind]
        live["infer_ms"][kind] = round(ms if prev is None else prev * 0.9 + ms * 0.1, 1)

    heartbeat = None
    if reporter is not None and args.heartbeat_sec > 0:
        from heartbeat import Heartbeat

        def hb_collect():
            return {
                "pipeline": {"frames_total": frame_idx, "stream": src.state,
                             "infer_ms": dict(live["infer_ms"]),
                             "tracked_persons": live["persons"],
                             "tracked_objects": sum(1 for o in objects
                                                    if frame_idx - o.last_seen <= MISS_TOLERANCE)},
                "tts": announcer.status() if announcer is not None else {"worker": "off"},
                "mosaic": {"mode": args.mosaic, "faces_total": mosaic.n_faces,
                           "head_fallback_total": mosaic.n_heads},
                "issues": reporter.issues()
                          + ([] if str(backend.waste_path).endswith(".engine") else ["model_fallback"]),
            }

        heartbeat = Heartbeat(os.environ["API_URL"].strip(), os.environ["DEVICE_TOKEN"].strip(),
                              os.environ["DEVICE_ID"].strip(), hb_collect,
                              interval=args.heartbeat_sec, roi=roi, configs=[site],
                              output_path=args.out)
    t0 = time.time()

    while True:
        ok, frame = src.read()
        if not ok:
            break                       # 파일 EOF 또는 재연결 포기
        if frame is None:
            # 연결은 살아 있는데 이번 주기에 새 프레임이 없었다 (일시적 정체).
            # --show의 종료 입력만 처리하고 넘어간다.
            if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        frame_idx += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t_inf = time.perf_counter()
        persons = backend.track_persons(frame)
        note_infer("person", t_inf)
        live["persons"] = len(persons)
        # 재식별: 돌아온 사람에게 이전 pid를 돌려주고, 그동안 새 pid로 쌓인 상태를 옛 pid로 이관
        persons, merges = reid.update(frame, persons, frame_idx, embs=backend.last_embs)
        for old_pid, new_pid in merges:
            if old_pid in carried:
                carried[new_pid] = carried.pop(old_pid)
            for o in objects:
                if o.owner_pid == old_pid:
                    o.owner_pid = new_pid
            for i, (fi, plist) in enumerate(person_hist):
                person_hist[i] = (fi, [(new_pid if pid == old_pid else pid, pb) for pid, pb in plist])
        person_boxes = [p[1] for p in persons]
        seen_pids = {pid for pid, _ in persons}
        recent = [(pid, b) for pid, b in ((reid.canonical(t), b)
                                          for t, b in backend.recent_lost(LOST_LINK_FRAMES))
                  if pid not in seen_pids]
        person_hist.append((frame_idx, list(persons)))
        frame_buffer.append((frame_idx, frame_gray, list(persons)))

        ground_dets = []
        fh, fw = frame.shape[:2]
        # 이번 프레임에 쓸 구역 (하트비트 스레드가 중간에 바꿔도 한 프레임 안에선 같은 구역)
        roi_gen, zones = roi.generation, roi.zones
        if roi_gen != roi_seen[0]:
            roi_changed_at = frame_idx
        rect = crop_rect(zones, fw, fh)
        t_inf = time.perf_counter()
        if rect is None:
            waste_dets = backend.detect_waste(frame)
        else:                       # 구역을 감싼 영역만 모델에 넣고 좌표를 프레임 기준으로 되돌린다
            x0, y0, x1, y1 = rect
            waste_dets = [([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], c, sc)
                          for b, c, sc in backend.detect_waste(frame[y0:y1, x0:x1])]
        note_infer("waste", t_inf)
        if zones:                   # 구역 밖(하단 중앙 기준) 탐지는 소지품·물체 추적에도 쓰지 않는다
            waste_dets = [d for d in waste_dets if is_monitored(d[0], fw, fh, zones)]
        for box, cls, conf in waste_dets:
            holder = None
            for pid, pbox in persons:
                cx, cy = center(box)
                inside = pbox[0] <= cx <= pbox[2] and pbox[1] <= cy <= pbox[3]
                if inside or iou(box, pbox) > CARRY_IOU or held_overhead(box, pbox):
                    holder = pid
                    break
            if holder is not None:
                carried[holder] = {"cls": cls, "hist": color_hist(frame, box),
                                   "last_frame": frame_idx, "box": box}
            else:
                ground_dets.append((box, cls, conf))
        ground_dets.sort(key=lambda d: -d[2])
        deduped = []
        for d in ground_dets:
            if all(iou(d[0], k[0]) < DEDUP_IOU for k in deduped):
                deduped.append(d)
        ground_dets = deduped

        active = [o for o in objects if frame_idx - o.last_seen <= MISS_TOLERANCE]
        lost = [o for o in objects if o not in active]
        matched = set()
        for box, cls, conf in ground_dets:
            h = color_hist(frame, box)
            best, best_iou = None, 0.3
            for o in active:
                if o.id in matched:
                    continue
                v = iou(box, o.box)
                if v > best_iou:
                    best, best_iou = o, v
            if best is not None:
                best.seen(box, conf, frame_idx, h, frame=frame)
                matched.add(best.id)
                continue
            best, best_score = None, 0.0
            for o in lost:
                if o.id in matched:
                    continue
                v = iou(box, o.box)
                if v >= SAME_SPOT_IOU:          # 같은 자리 -> 색 비교 없이 같은 물체 (겹칠수록 우선)
                    if 1.0 + v > best_score:
                        best, best_score = o, 1.0 + v
                    continue
                if o.cls != cls:
                    continue
                cx, cy = center(box); ox, oy = center(o.box)
                dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if dist > REID_DIST_FACTOR * max(diag(o.box), 60):
                    continue
                s = hist_sim(h, o.hist)
                if s >= REID_HIST_MIN and s > best_score:
                    best, best_score = o, s
            if best is not None:
                best.seen(box, conf, frame_idx, h, frame=frame)
                matched.add(best.id)
                continue
            o = WasteObject(box, cls, conf, frame_idx, h, frame=frame)
            if frame_idx - roi_changed_at <= ROI_GRACE:
                o.baseline = True   # 구역이 막 바뀌어 보이기 시작한 물체 = 원래 있던 것
            how = link_owner(o, box, cls, h, persons, carried, frame_idx, recent)
            if log_objects:
                print(f"[물체] #{o.id} {WASTE_NAMES[cls]} 등장({conf:.2f}) · "
                      + ("적치물 (시작 직후/구역 변경 직후)" if o.baseline else how), flush=True)
            objects.append(o)
        objects = [o for o in objects if frame_idx - o.last_seen <= LOST_KEEP]
        # 소지품 기록도 같은 기준으로 정리한다. 위 귀속 탐색이 carried 전체를 훑으므로
        # 안 지우면 상시 운영에서 누적된 pid만큼 매 탐지마다 느려진다.
        # (CARRY_WINDOW 초과분은 어차피 귀속 후보에서 제외되므로 판정 결과는 그대로)
        if frame_idx % 300 == 0:
            carried = {pid: info for pid, info in carried.items()
                       if frame_idx - info["last_frame"] <= CARRY_WINDOW}
            # 녹화가 꺼져 있어도 스냅샷은 쌓이므로 주기적으로 공간을 본다
            space.reclaim(exclude=recorder.current_path())
            if time.monotonic() >= next_expire:
                space.expire(exclude=recorder.current_path())
                next_expire = time.monotonic() + 3600

        # 클립·회수 판정의 시계: 라이브는 실제 시각, 파일은 영상 시각 (파일은 실시간보다 빨리 돈다)
        now_t = time.monotonic() if src.live else frame_idx / fps
        if roi_gen != roi_seen[0]:
            # 구역이 바뀌었다: 그동안 감시 밖이라 발화 판정을 안 받던 물체가 새 구역 안에 들어오면
            # 이미 오래 놓여 있던 것이다 -> 기존 적치물로 돌린다 (구역 변경만으로 사건이 나지 않게)
            old = roi_seen[1]
            n_rebased = 0
            for o in objects:
                if o.fired or o.baseline:
                    continue
                was_monitored = not old or is_monitored(o.box, fw, fh, old)
                if not was_monitored and (not zones or is_monitored(o.box, fw, fh, zones)):
                    o.baseline = True
                    n_rebased += 1
            roi_seen = (roi_gen, zones)
            print(f"[ROI] 구역 변경 적용 ({roi.describe()})"
                  + (f" · 새로 감시 구역에 든 기존 물체 {n_rebased}개는 적치물로 처리" if n_rebased else ""),
                  flush=True)
        if retrieval is not None:
            retrieval.update(now_t,
                             [o.box for o in objects if frame_idx - o.last_seen <= MISS_TOLERANCE],
                             person_boxes)

        # 추론·색상 크롭·재식별이 모두 끝난 뒤, 출력(스냅샷/영상/화면)에 쓰일 프레임의 얼굴을 가린다.
        # raw는 원본 스냅샷 저장과 색상 판정 폴백(best_crop 없을 때)에만 쓴다.
        raw = frame
        if mosaic.enabled:
            raw = frame.copy()
            mosaic.apply(frame, person_boxes)

        def note(o, reason):
            """연결된 물체의 판정 단계가 바뀔 때만 한 줄 (--log-objects / LOG_OBJECTS=1)."""
            if log_objects and o.person_at_birth and reason != o.last_reason:
                o.last_reason = reason
                print(f"[물체] #{o.id} {WASTE_NAMES[o.cls]} (P{o.owner_pid}) {reason}", flush=True)

        # 투기자 근처 판정용: 이번 프레임 탐지 + 방금 탐지가 빠진 사람(예측 위치)
        near_people = list(persons) + list(recent)
        for o in objects:
            if o.baseline or o.fired:
                continue
            if frame_idx - o.last_seen > MISS_TOLERANCE:
                note(o, "탐지 끊김 (다시 잡히면 이어서 판정)")
                continue
            if not roi.monitored(o.box, fw, fh):
                continue            # 감시 구역 밖이거나 제외 구역 안
            if not o.person_at_birth:
                continue
            if o.age < STABLE_AGE:
                note(o, f"안정화 대기 (탐지 {STABLE_AGE}프레임 필요)")
                continue
            if frame_idx - o.still_since < STATIC_FRAMES:
                note(o, "움직이는 중 (정지 대기)")
                continue            # 아직 움직이는 중이거나 막 멈췄다 — 들고 가는 물체 오탐 방지
            if o.owner_pid is not None:
                owner_here = any(pid == o.owner_pid and person_near(o.box, [pb])
                                 for pid, pb in near_people)
            else:
                owner_here = person_near(o.box, [pb for _, pb in near_people])
            if owner_here:
                note(o, "투기자 아직 근처")
                o.clear_count = 0
                continue
            o.clear_count += 1
            if o.clear_count < CLEAR_FRAMES:
                note(o, "투기자 이탈 확인 중")
                continue
            if any(x.fired and iou(o.box, x.box) > 0.5 for x in objects if x is not o):
                note(o, "같은 자리에서 이미 발화 -> 생략")
                o.fired = True
                continue
            # 배경 차분: 등장 이전에도 그 자리에 같은 것이 있었다면 기존 적치물 -> 강등
            if region_unchanged(frame_gray, o, frame_buffer):
                note(o, "등장 전에도 같은 자리에 있었음 (배경 동일) -> 적치물")
                o.baseline = True
                continue
            note(o, "발화")
            o.fired = True
            n_events += 1
            color, night = None, None
            if dominant_color_name is not None:
                # best_crop(최고 확신 프레임)이 있으면 그것으로, 없으면 현재 프레임에서.
                # 야간 적외선/저조도 프레임이면 색상은 신뢰할 수 없으므로 판정하지 않는다.
                if o.best_crop is not None:
                    night = o.best_night
                    if night is None:
                        color = dominant_color_name(o.best_crop)
                else:
                    night = night_mode(raw)
                    if night is None:
                        color = dominant_color_name(raw, o.box)
            ev = {"event": n_events, "frame": frame_idx, "sec": round(frame_idx / fps, 1),
                  "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "class": WASTE_NAMES[o.cls], "color": color, "night": night,
                  "conf": round(o.conf, 2),
                  "obj_id": o.id, "owner_pid": o.owner_pid,
                  "owner_matched": o.owner_strong, "box": [round(v) for v in o.box]}
            events_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            events_f.flush()
            print(f"[이벤트 {n_events}] {ev['ts']} 무단투기 의심: "
                  f"{color + ' ' if color else ''}{ev['class']}"
                  f"{f' ({night}: 색상 생략)' if night else ''} "
                  f"obj#{o.id} 투기자 pid={o.owner_pid}"
                  f"{'(소지품 매칭)' if o.owner_strong else ''}", flush=True)
            announced = False
            # event_id를 먼저 정해 둔다 — 방송 음성은 몇 초 뒤 합성이 끝나야 생기므로
            # 재생이 끝나면 같은 event_id로 그 wav만 따로 올린다 (reporter.attach)
            event_id = ulid() if reporter is not None else None
            if announcer is not None:
                audio_kw = {}
                if reporter is not None:
                    audio_kw = dict(
                        audio_out=out_dir / f"event_{n_events:04d}.wav",
                        on_audio=lambda wav, eid=event_id:
                            reporter.attach(eid, {"announce_audio": wav}))
                announced = announcer.announce(color, WASTE_NAMES[o.cls], **audio_kw)
            label = f"DUMPING SUSPECT #{n_events} (person {o.owner_pid})"
            space.reclaim(exclude=recorder.current_path())   # 쓰기 전에 자리 확보
            snap_path = out_dir / f"event_{n_events:04d}.jpg"
            save_snapshot(snap_path, frame, o.box, label, args.snapshot_width)
            raw_path = None
            if mosaic.enabled and args.raw_snapshot:
                raw_path = out_dir / f"event_{n_events:04d}_raw.jpg"
                save_snapshot(raw_path, raw, o.box, label, args.snapshot_width)
            media_meta = {}
            if clipper is not None:
                clip_path = out_dir / f"event_{n_events:04d}.mp4"
                clipper.start(event_id, clip_path, now_t)
                media_meta = {"clip": clip_path.name,
                              "clip_range_s": [-round(args.clip_sec), round(args.clip_sec)]}
            if retrieval is not None:
                retrieval.add(event_id, o.box, now_t)
            if reporter is not None:
                # 투기자의 마지막 목격 위치 — 스냅샷에 투기자를 함께 표시하기 위함
                owner_box = None
                if o.owner_pid is not None:
                    for _fi, plist in reversed(person_hist):
                        hit = next((pb for pid, pb in plist if pid == o.owner_pid), None)
                        if hit is not None:
                            owner_box = [round(v) for v in hit]
                            break
                reporter.submit({           # ts는 submit()이 ISO 8601+오프셋으로 채운다
                    "event_id": event_id,
                    "detection": {"class": ev["class"], "class_id": int(o.cls),
                                  "color": color, "night": night,
                                  "conf": ev["conf"], "bbox": ev["box"],
                                  "frame_size": [W, H]},
                    "suspect": {"obj_id": o.id, "owner_pid": o.owner_pid,
                                "owner_matched": o.owner_strong,
                                "person_bbox_at_drop": owner_box},
                    "announce": {
                        "played": announced,
                        "phrase": (build_phrase(color, WASTE_NAMES[o.cls], place_name())
                                   if build_phrase else None),
                        "suppressed_reason": (None if announced or announcer is None
                                              else announcer.last_suppressed_reason)},
                    "debug": {"frame": frame_idx, "local_seq": n_events,
                              "model_format": Path(backend.waste_path).suffix[1:]},
                    "media": media_meta,
                }, media={"snapshot": snap_path, "snapshot_raw": raw_path})

        if (recorder.enabled or args.show or pusher is not None
                or clipper is not None or thumb is not None):
            roi.draw(frame)
            for pid, p in persons:
                x1, y1, x2, y2 = map(int, p)
                has_carry = pid in carried and frame_idx - carried[pid]["last_frame"] <= 5
                color = (0, 120, 255) if has_carry else (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"P{pid}{'*' if has_carry else ''}",
                            (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            for o in objects:
                if frame_idx - o.last_seen > MISS_TOLERANCE:
                    continue
                x1, y1, x2, y2 = map(int, o.box)
                color = (0, 0, 255) if o.fired else ((180, 180, 180) if o.baseline
                                                     else (0, 165, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = f"#{o.id}" + (f"<P{o.owner_pid}" if o.owner_pid is not None else "")
                cv2.putText(frame, tag, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if n_events:
                cv2.putText(frame, f"EVENTS: {n_events}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            recorder.write(frame)
            if pusher is not None or clipper is not None or thumb is not None:
                # 송출·클립·썸네일은 모두 720p — 한 번만 줄여서 같이 쓴다
                small = (frame if (fw, fh) == (1280, 720)
                         else cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA))
                if pusher is not None:
                    pusher.push(small)
                if clipper is not None:
                    clipper.push(small, now_t)
                if thumb is not None:
                    thumb.offer(small)
            if args.show:
                # 2560x1440 원본을 그대로 그리면 화면을 넘치고 표시만으로 ~6fps로 떨어진다 (보드 실측)
                # -> 표시용으로만 720p로 줄이고 창 크기는 사용자가 조절할 수 있게 둔다
                if not show_ready:
                    cv2.namedWindow("dump_monitor", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("dump_monitor", 1280, 720)
                    show_ready = True
                sh = 720 / frame.shape[0]
                cv2.imshow("dump_monitor", frame if sh >= 1 else
                           cv2.resize(frame, (round(frame.shape[1] * sh), 720),
                                      interpolation=cv2.INTER_AREA))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    dt = time.time() - t0          # 방송 잔여 재생 대기 전에 측정 (FPS 왜곡 방지)
    src.close()
    recorder.close()
    events_f.close()
    print(f"완료: {frame_idx}프레임, {dt:.1f}s ({frame_idx / max(dt, 0.1):.1f} FPS), "
          f"이벤트 {n_events}건 -> {out_dir}")
    print(f"[{src.summary()}]")
    if recorder.enabled:
        print(f"[{recorder.summary()}]")
    print(f"[{space.summary()}]")
    if mosaic.enabled:
        print(f"[{mosaic.summary()}]")
    if reid.enabled:
        print(f"[{reid.summary()}]")
    if pusher is not None:
        print(f"[송출] {pusher.frames_in}프레임 인코딩 · {pusher.sent_bytes / 1024:.0f}KiB 송신 "
              f"· 재연결 {pusher.reconnects}회")
        pusher.close()
    if heartbeat is not None:
        heartbeat.close()
        print(f"[{heartbeat.summary()}]")
    if thumb is not None:
        thumb.close()
        print(f"[{thumb.summary()}]")
    if retrieval is not None:
        print(f"[{retrieval.summary()}]")
    if clipper is not None:
        clipper.close()            # 진행 중 클립을 마무리해 reporter.attach까지 넘긴다
        print(f"[{clipper.summary()}]")
    if announcer is not None:
        # 전송기보다 먼저 닫는다 — 남은 합성·재생이 끝나며 나오는 방송 wav를 reporter.attach로
        # 넘겨야 아래 최종 드레인 때 함께 나간다
        announcer.close()          # 큐에 남은 방송을 끝까지 재생한 뒤 종료
    if reporter is not None:
        left = reporter.pending()
        print(f"[전송] {reporter.sent}건 전송 완료"
              + (f" · 미전송 {left}건 큐 보관 (다음 실행 때 재시도)" if left else "")
              + (f" · 큐 초과 폐기 {reporter.dropped}건" if reporter.dropped else "")
              + (f" · 서버 거부 {reporter.rejected}건 (queue/events/bad/)"
                 if reporter.rejected else ""))
        reporter.close()


if __name__ == "__main__":
    main()
