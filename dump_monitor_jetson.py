# -*- coding: utf-8 -*-
"""무단투기 의심 이벤트 감지 — Jetson 배포판.

- 사람: COCO yolo26n (class 0) + ByteTrack / 폐기물: waste10 10클래스
- 모델은 models/ 폴더에서 .engine(TensorRT) > .onnx > .pt 순으로 자동 선택
- 로직: 소지품-사람 귀속, LOST 객체 재식별(Re-ID), owner 이탈 시 발화
  (개발 PC의 work/scripts/dump_monitor.py v2와 동일 로직)
- --tts: 발화 시 '색상 + 쓰레기 종류'를 특정한 한국어 경고를 그 자리에서 합성해 방송 (tts/ 참고)
  야간 적외선/저조도 프레임에서는 색상 판정이 무의미하므로 종류만 방송한다
- --mosaic head|face: 출력물(스냅샷/영상/화면)의 사람 얼굴을 픽셀화 (mosaic.py 참고).
  추론은 원본 프레임으로 하므로 탐지 정확도에는 영향 없음
- --reid: 사람 재식별 (person_reid.py). 화면 밖으로 나갔다 돌아온 사람에게 이전 pid를 돌려줘
  owner 부재 판정·회수 귀속이 끊기지 않게 한다. yolo26n-reid 임베딩, 사람당 3프레임마다 1회

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
from ultralytics import YOLO

from mosaic import FaceMosaic
from person_reid import PersonReID, load_encoder as load_reid_encoder
from recorder import OutputSpace, RotatingRecorder
from stream import FrameSource

try:                                          # TTS는 선택 기능 — 없어도 감시는 동작
    from tts import Announcer, dominant_color_name
    from tts.color_naming import crop_box, night_mode
except Exception as _tts_err:                 # noqa: BLE001
    Announcer = dominant_color_name = crop_box = night_mode = None
    _TTS_IMPORT_ERROR = _tts_err
else:
    _TTS_IMPORT_ERROR = None

HERE = Path(__file__).parent

WASTE_CONF = 0.20
PERSON_CONF = 0.35
TRACKER = "bytetrack.yaml"   # 명시 고정: ultralytics 버전별 기본 트래커가 달라 pid 특성이 흔들리는 것 방지
STABLE_AGE = 12          # 폐기물이 '잔류'로 간주되는 누적 탐지 프레임
MISS_TOLERANCE = 8       # 이 프레임까지는 ACTIVE 유지 (탐지 깜빡임 허용)
LOST_KEEP = 300          # LOST 상태 보관 프레임 (재식별 대상)
REID_DIST_FACTOR = 1.2   # 재식별 허용 거리: 객체 대각선의 N배
REID_HIST_MIN = 0.5      # 재식별 색상 유사도 하한
CARRY_IOU = 0.30         # 사람 박스와 이 이상 겹치면 '소지 쓰레기'
CARRY_WINDOW = 60        # 소지품 목격 후 귀속 후보로 보는 프레임 수
NEAR_FACTOR = 2.0        # 사람-객체 근접 판정: 객체 대각선의 N배
BIRTH_WINDOW = 40        # 객체 등장 전후 사람 존재 탐색 윈도우
DEDUP_IOU = 0.6          # 클래스 무관 탐지 중복 제거 임계
CLEAR_FRAMES = 5         # 사람 부재 연속 프레임 후 발화
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


def load_model(stem_or_path: str):
    """모델 로드. 경로가 주어지면 그대로, 아니면 models/에서
    .engine > .onnx > .pt 순으로 시도한다. 로드/워밉업이 실패하면 다음 포맷으로 폴백
    (예: JetPack 업그레이드로 .engine이 무효화된 경우 .onnx/.pt로 계속 동작)."""
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
        self.baseline = frame_idx <= 2
        self.person_at_birth = False
        self.owner_pid = None
        self.owner_strong = False
        self.fired = False
        self.clear_count = 0
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
    ap.add_argument("--waste-model", default=None)
    ap.add_argument("--person-model", default=None)
    ap.add_argument("--show", action="store_true", help="화면 표시 (모니터 연결 시)")
    ap.add_argument("--no-save-video", action="store_true", help="주석 영상 저장 생략")
    ap.add_argument("--tts", action="store_true", help="이벤트 발생 시 음성 경고 방송")
    ap.add_argument("--tts-mode", default="live", choices=["live", "cache"],
                    help="live=이벤트마다 문구를 즉석 합성해 방송(기본) / "
                         "cache=prerender_tts.py로 만든 사전 렌더링 wav만 재생")
    ap.add_argument("--tts-backend", default="supertonic", choices=["supertonic", "null"],
                    help="합성 백엔드 (null=무음, 배선 점검용)")
    ap.add_argument("--tts-cache", default=None,
                    help="사전 렌더링 wav 폴더 (기본 cache/tts). live 모드에선 합성 실패 시 폴백용")
    ap.add_argument("--tts-location", default=os.environ.get("CAM_LOCATION") or None,
                    help="방송에 넣을 장소명 (.env의 CAM_LOCATION). "
                         "cache 모드면 그 장소 전용 캐시가 있어야 함")
    ap.add_argument("--tts-cooldown", type=float, default=12.0,
                    help="방송 간 최소 간격(초). 문장 길이(약 9초)보다 크게 두어 대기열 적체를 막는다")
    ap.add_argument("--tts-repeat-window", type=float, default=30.0,
                    help="같은 (색상,종류) 조합 재방송 억제 시간(초)")
    ap.add_argument("--tts-no-runtime-synth", action="store_true",
                    help="(cache 모드) 캐시 미스 시 즉석 합성하지 않고 방송을 건너뜀")
    ap.add_argument("--tts-model", default="supertonic-2", help="합성 모델")
    ap.add_argument("--tts-voice", default="M4", help="목소리 (M1~M5/F1~F5)")
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
    ap.add_argument("--reid-ttl", type=float, default=600.0,
                    help="사라진 사람을 재식별 대상으로 기억하는 시간(초). 회수 판정 시간과 맞출 것")
    # --- 상시 운영(RTSP) 관련 ---
    ap.add_argument("--video-segment-min", type=float, default=60.0,
                    help="주석 영상을 이 분 단위로 분할 (기본 60). 상시 운영에서 디스크 고갈 방지")
    ap.add_argument("--video-keep", type=int, default=24,
                    help="보관할 영상 세그먼트 개수. 초과분은 오래된 것부터 삭제 (기본 24)")
    ap.add_argument("--min-free-mb", type=float, default=1000.0,
                    help="여유 디스크가 이 값 아래면 오래된 출력물부터 삭제하고, "
                         "그래도 모자라면 녹화를 중단한다 (기본 1000MB)")
    ap.add_argument("--purge-order", default="video-first",
                    choices=["video-first", "oldest-first"],
                    help="공간 회수 순서. video-first=주석 영상을 먼저 다 지우고 "
                         "그래도 모자라면 스냅샷(각각 오래된 것부터) / "
                         "oldest-first=종류 무관 순수 FIFO")
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

    if env_path is not None:
        print(f"[설정] {env_path} 읽음")
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
                      else {"model": args.tts_model, "voice": args.tts_voice})
            announcer = Announcer(
                mode=args.tts_mode,
                cache_dir=args.tts_cache, backend=args.tts_backend,
                location=args.tts_location, cooldown=args.tts_cooldown,
                repeat_window=args.tts_repeat_window,
                allow_runtime_synth=not args.tts_no_runtime_synth, **tts_kw)
    reid = PersonReID(load_reid_encoder(args.reid_model) if args.reid else None, ttl_s=args.reid_ttl)
    if args.reid and reid.enabled:
        print(f"[ReID] 활성 · 갤러리 보관 {args.reid_ttl:.0f}s")
    mosaic = FaceMosaic(args.mosaic, model_path=args.face_model)
    if mosaic.enabled:
        print(f"[모자이크] {mosaic.mode} 모드"
              + (" · 원본 스냅샷 별도 저장" if args.raw_snapshot else ""))
    waste_model, waste_path = load_model(args.waste_model or "waste10_yolo26n")
    person_model, person_path = load_model(args.person_model or "person_yolo26n")
    print(f"waste={Path(waste_path).name} person={Path(person_path).name} -> {out_dir}")

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
                        purge_order=args.purge_order)
    print(f"[디스크] 여유 {space.free_mb():.0f}MB · 임계 {args.min_free_mb:.0f}MB "
          f"· 회수 순서 {args.purge_order} · 스냅샷 "
          + (f"{args.snapshot_width}px 폭" if args.snapshot_width else "원본 해상도"))
    events_f = open(out_dir / "events.jsonl", "a", encoding="utf-8")

    objects, carried = [], {}
    person_hist = deque(maxlen=BIRTH_WINDOW * 2)
    frame_buffer = deque(maxlen=BIRTH_WINDOW + 10)   # (frame_idx, gray, persons)
    n_events, frame_idx = 0, 0
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

        pr = person_model.track(frame, classes=[0], conf=PERSON_CONF, persist=True,
                                tracker=TRACKER, verbose=False)[0]
        persons = []
        if pr.boxes is not None:
            ids = pr.boxes.id
            for k, b in enumerate(pr.boxes):
                pid = int(ids[k]) if ids is not None else -(k + 1)
                persons.append((pid, b.xyxy[0].tolist()))
        # 재식별: 돌아온 사람에게 이전 pid를 돌려주고, 그동안 새 pid로 쌓인 상태를 옛 pid로 이관
        persons, merges = reid.update(frame, persons, frame_idx)
        for old_pid, new_pid in merges:
            if old_pid in carried:
                carried[new_pid] = carried.pop(old_pid)
            for o in objects:
                if o.owner_pid == old_pid:
                    o.owner_pid = new_pid
            for i, (fi, plist) in enumerate(person_hist):
                person_hist[i] = (fi, [(new_pid if pid == old_pid else pid, pb) for pid, pb in plist])
        person_boxes = [p[1] for p in persons]
        person_hist.append((frame_idx, list(persons)))
        frame_buffer.append((frame_idx, frame_gray, list(persons)))

        wr = waste_model.predict(frame, conf=WASTE_CONF, verbose=False)[0]
        ground_dets = []
        if wr.boxes is not None:
            for b in wr.boxes:
                box = b.xyxy[0].tolist()
                cls, conf = int(b.cls), float(b.conf)
                holder = None
                for pid, pbox in persons:
                    cx, cy = center(box)
                    inside = pbox[0] <= cx <= pbox[2] and pbox[1] <= cy <= pbox[3]
                    if inside or iou(box, pbox) > CARRY_IOU:
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
                if o.id in matched or o.cls != cls:
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
            best_pid, best_s = None, 0.0
            for pid, info in carried.items():
                if frame_idx - info["last_frame"] > CARRY_WINDOW:
                    continue
                if not any(p[0] == pid and person_near(box, [p[1]]) for p in persons):
                    continue
                s = hist_sim(h, info["hist"]) + (0.5 if info["cls"] == cls else 0.0)
                if s > best_s:
                    best_pid, best_s = pid, s
            if best_pid is not None and best_s >= 0.6:
                o.owner_pid, o.owner_strong, o.person_at_birth = best_pid, True, True
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

        # 추론·색상 크롭·재식별이 모두 끝난 뒤, 출력(스냅샷/영상/화면)에 쓰일 프레임의 얼굴을 가린다.
        # raw는 원본 스냅샷 저장과 색상 판정 폴백(best_crop 없을 때)에만 쓴다.
        raw = frame
        if mosaic.enabled:
            raw = frame.copy()
            mosaic.apply(frame, person_boxes)

        for o in objects:
            if o.baseline or o.fired or frame_idx - o.last_seen > MISS_TOLERANCE:
                continue
            if not o.person_at_birth:
                for fi, plist in person_hist:
                    if abs(fi - o.first_seen) > BIRTH_WINDOW:
                        continue
                    near = [(pid, pb) for pid, pb in plist if person_near(o.box, [pb])]
                    if near:
                        o.person_at_birth = True
                        if o.owner_pid is None:
                            o.owner_pid = near[0][0]
                        break
            if not (o.person_at_birth and o.age >= STABLE_AGE):
                continue
            if o.owner_pid is not None:
                owner_here = any(pid == o.owner_pid and person_near(o.box, [pb])
                                 for pid, pb in persons)
            else:
                owner_here = person_near(o.box, person_boxes)
            if owner_here:
                o.clear_count = 0
                continue
            o.clear_count += 1
            if o.clear_count < CLEAR_FRAMES:
                continue
            if any(x.fired and iou(o.box, x.box) > 0.5 for x in objects if x is not o):
                o.fired = True
                continue
            # 배경 차분: 등장 이전에도 그 자리에 같은 것이 있었다면 기존 적치물 -> 강등
            if region_unchanged(frame_gray, o, frame_buffer):
                o.baseline = True
                continue
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
            if announcer is not None:
                announcer.announce(color, WASTE_NAMES[o.cls])
            label = f"DUMPING SUSPECT #{n_events} (person {o.owner_pid})"
            space.reclaim(exclude=recorder.current_path())   # 쓰기 전에 자리 확보
            save_snapshot(out_dir / f"event_{n_events:04d}.jpg", frame, o.box,
                          label, args.snapshot_width)
            if mosaic.enabled and args.raw_snapshot:
                save_snapshot(out_dir / f"event_{n_events:04d}_raw.jpg", raw, o.box,
                              label, args.snapshot_width)

        if recorder.enabled or args.show:
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
            if args.show:
                cv2.imshow("dump_monitor", frame)
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
    if announcer is not None:
        announcer.close()          # 큐에 남은 방송을 끝까지 재생한 뒤 종료


if __name__ == "__main__":
    main()
