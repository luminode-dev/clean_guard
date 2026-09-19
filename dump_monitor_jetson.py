# -*- coding: utf-8 -*-
"""무단투기 의심 이벤트 감지 — Jetson 배포판.

- 사람: COCO yolo26n (class 0) + ByteTrack / 폐기물: waste10 10클래스
- 모델은 models/ 폴더에서 .engine(TensorRT) > .onnx > .pt 순으로 자동 선택
- 로직: 소지품-사람 귀속, LOST 객체 재식별(Re-ID), owner 이탈 시 발화
  (개발 PC의 work/scripts/dump_monitor.py v2와 동일 로직)
- --tts: 발화 시 '색상 + 쓰레기 종류'를 특정한 한국어 경고를 그 자리에서 합성해 방송 (tts/ 참고)
  야간 적외선/저조도 프레임에서는 색상 판정이 무의미하므로 종류만 방송한다

usage:
  python3 dump_monitor_jetson.py --source rtsp://... --name cam01
  python3 dump_monitor_jetson.py --source 0 --name webcam --show
  python3 dump_monitor_jetson.py --source clip.mp4 --name test --no-save-video
  python3 dump_monitor_jetson.py --source 0 --name cam01 --tts --tts-location "정문 수거함 앞"
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="영상 파일 / RTSP URL / 웹캠 번호")
    ap.add_argument("--name", default="cam", help="출력 폴더명")
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
    ap.add_argument("--tts-location", default=None,
                    help="방송에 넣을 장소명 (cache 모드면 그 장소 전용 캐시가 있어야 함)")
    ap.add_argument("--tts-cooldown", type=float, default=12.0,
                    help="방송 간 최소 간격(초). 문장 길이(약 9초)보다 크게 두어 대기열 적체를 막는다")
    ap.add_argument("--tts-repeat-window", type=float, default=30.0,
                    help="같은 (색상,종류) 조합 재방송 억제 시간(초)")
    ap.add_argument("--tts-no-runtime-synth", action="store_true",
                    help="(cache 모드) 캐시 미스 시 즉석 합성하지 않고 방송을 건너뜀")
    ap.add_argument("--tts-model", default="supertonic-2", help="합성 모델")
    ap.add_argument("--tts-voice", default="M4", help="목소리 (M1~M5/F1~F5)")
    args = ap.parse_args()

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
    waste_model, waste_path = load_model(args.waste_model or "waste10_yolo26n")
    person_model, person_path = load_model(args.person_model or "person_yolo26n")
    print(f"waste={Path(waste_path).name} person={Path(person_path).name} -> {out_dir}")

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print(f"입력을 열 수 없음: {args.source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if not args.no_save_video:
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    events_f = open(out_dir / "events.jsonl", "a", encoding="utf-8")

    objects, carried = [], {}
    person_hist = deque(maxlen=BIRTH_WINDOW * 2)
    frame_buffer = deque(maxlen=BIRTH_WINDOW + 10)   # (frame_idx, gray, persons)
    n_events, frame_idx = 0, 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pr = person_model.track(frame, classes=[0], conf=PERSON_CONF, persist=True,
                                verbose=False)[0]
        persons = []
        if pr.boxes is not None:
            ids = pr.boxes.id
            for k, b in enumerate(pr.boxes):
                pid = int(ids[k]) if ids is not None else -(k + 1)
                persons.append((pid, b.xyxy[0].tolist()))
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
                    night = night_mode(frame)
                    if night is None:
                        color = dominant_color_name(frame, o.box)
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
            snap = frame.copy()
            x1, y1, x2, y2 = map(int, o.box)
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(snap, f"DUMPING SUSPECT #{n_events} (person {o.owner_pid})",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.imwrite(str(out_dir / f"event_{n_events:04d}.jpg"), snap)

        if writer is not None or args.show:
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
            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("dump_monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    dt = time.time() - t0          # 방송 잔여 재생 대기 전에 측정 (FPS 왜곡 방지)
    cap.release()
    if writer is not None:
        writer.release()
    events_f.close()
    print(f"완료: {frame_idx}프레임, {dt:.1f}s ({frame_idx / max(dt, 0.1):.1f} FPS), "
          f"이벤트 {n_events}건 -> {out_dir}")
    if announcer is not None:
        announcer.close()          # 큐에 남은 방송을 끝까지 재생한 뒤 종료


if __name__ == "__main__":
    main()
