#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""torch 없는 백엔드(trt_backend.py + bytetrack.py) 검증.

1) 추적기 단위 테스트 — 합성 박스 (의존성: numpy/scipy)
2) torch/ultralytics를 import하지 않는 서브프로세스에서 탐지·Re-ID가 도는지
3) ultralytics 결과와의 일치 (같은 ONNX, 같은 프레임) — ultralytics가 설치된 PC에서만

  python3 tests/test_trt_backend.py
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ASSET = ROOT / "tests" / "assets" / "street_bag.jpg"

from bytetrack import ByteTrack, iou_matrix  # noqa: E402

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


# ------------------------------------------------------------------ 1) ByteTrack
print("1) ByteTrack — 합성 궤적")
tr = ByteTrack(frame_rate=30)


def walk(f, x0, dx, y=100, w=40, h=100, s=0.9):
    x = x0 + dx * f
    return [x, y, x + w, y + h, s]


ids_a, ids_b = set(), set()
for f in range(1, 41):
    out = tr.update([walk(f, 50, 3), walk(f, 500, -3, y=300)])
    for tid, box, s in out:
        (ids_a if box[1] < 200 else ids_b).add(tid)
check("두 사람이 각각 하나의 id 유지 (A)", len(ids_a), 1)
check("두 사람이 각각 하나의 id 유지 (B)", len(ids_b), 1)
check_true("id가 서로 다름", ids_a != ids_b)
id_a = next(iter(ids_a))

# 10프레임 가림(탐지 없음) 후 같은 자리 근처 재등장 -> 같은 id
for f in range(41, 51):
    tr.update([walk(f, 500, -3, y=300)])
out = tr.update([walk(51, 50, 3), walk(51, 500, -3, y=300)])
got = [tid for tid, box, _ in out if box[1] < 200]
check("10프레임 가림 후 복귀 -> 같은 id (칼만 예측 매칭)", got, [id_a])

# track_buffer(30) 넘게 사라지면 새 id
for f in range(52, 100):
    tr.update([walk(f, 500, -3, y=300)])
out = tr.update([walk(100, 50, 3), walk(100, 500, -3, y=300)])
out = tr.update([walk(101, 50, 3), walk(101, 500, -3, y=300)])     # 새 트랙은 두 번째 프레임에 확정
got = [tid for tid, box, _ in out if box[1] < 200]
check_true("30프레임 넘게 사라지면 새 id", bool(got) and got[0] != id_a, f"{got} vs 이전 {id_a}")

# 저점수 탐지: 새 트랙은 만들지 않지만(new_thresh 0.25) 기존 트랙은 2차 매칭으로 유지
tr2 = ByteTrack()
for f in range(1, 6):
    tr2.update([walk(f, 50, 3, s=0.9)])
kept = [tr2.update([walk(f, 50, 3, s=0.15)]) for f in range(6, 12)]
check_true("저점수(0.15) 탐지로도 기존 트랙 유지", all(len(o) == 1 for o in kept))
tr3 = ByteTrack()
out = [tr3.update([walk(f, 50, 3, s=0.15)]) for f in range(1, 6)]
check_true("저점수만으론 새 트랙 생성 안 함", all(len(o) == 0 for o in out))
check("iou_matrix 정합", round(float(iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0]), 3), 0.333)

# ------------------------------------------------------------------ 2) torch 없는 프로세스
print("\n2) torch/ultralytics 없이 탐지 + Re-ID 실행 (서브프로세스)")
child = r'''
import sys, json
sys.path.insert(0, %r)
import numpy as np, cv2
from trt_backend import load_detector, load_reid, ByteTrack
img = cv2.imread(%r)
det_p, pp = load_detector("person_yolo26n"); det_w, wp = load_detector("waste10_yolo26n")
pb, pc, _ = det_p.detect(img, conf=0.35, classes=[0])
wb, wc, wcls = det_w.detect(img, conf=0.2)
enc, rp = load_reid("yolo26n-reid")
dets = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]] for b in pb], np.float32)
embs = enc(img, dets)
tr = ByteTrack(); tracks = tr.update(np.c_[pb, pc]); tracks = tr.update(np.c_[pb, pc])
print(json.dumps({"torch": "torch" in sys.modules, "ultralytics": "ultralytics" in sys.modules,
                  "persons": pb.round(2).tolist(), "pconf": pc.round(4).tolist(),
                  "waste": wb.round(2).tolist(), "wconf": wc.round(4).tolist(), "wcls": wcls.tolist(),
                  "emb_dim": [None if e is None else len(e) for e in embs],
                  "emb": [None if e is None else (e / np.linalg.norm(e)).round(5).tolist() for e in embs],
                  "tracks": len(tracks), "paths": [pp, wp, rp]}))
'''
r = subprocess.run([sys.executable, "-c", child % (str(ROOT), str(ASSET))], capture_output=True,
                   text=True, encoding="utf-8", cwd=str(ROOT))
line = [l for l in r.stdout.splitlines() if l.startswith("{")]
check_true("서브프로세스 정상 종료", r.returncode == 0 and line, r.stderr[-600:])
res = json.loads(line[-1]) if line else {}
if res:
    check("torch 미적재", res["torch"], False)
    check("ultralytics 미적재", res["ultralytics"], False)
    check_true("사람 탐지 ≥ 3", len(res["persons"]) >= 3, f"{len(res['persons'])}명")
    check("폐기물 탐지: 쓰레기봉투 1건", (len(res["waste"]), res["wcls"]), (1, [0]))
    check_true("Re-ID 임베딩 512차원", all(d == 512 for d in res["emb_dim"]))
    check("ByteTrack 확정 트랙 수 = 사람 수", res["tracks"], len(res["persons"]))
    print(f"       모델: {[Path(p).name for p in res['paths']]}")

# ------------------------------------------------------------------ 3) ultralytics와 일치
print("\n3) ultralytics 결과와 일치 (같은 ONNX·프레임)")
try:
    from ultralytics import YOLO
    from ultralytics.trackers.utils.reid import ReID
except Exception as e:                                             # noqa: BLE001
    print(f"  (건너뜀) ultralytics 없음: {e}")
    YOLO = None
if YOLO is not None and res:
    img = cv2.imread(str(ASSET))
    pm, wm = YOLO(str(ROOT / "models/person_yolo26n.onnx")), YOLO(str(ROOT / "models/waste10_yolo26n.onnx"))
    up = pm.predict(img, conf=0.35, classes=[0], verbose=False)[0]
    uw = wm.predict(img, conf=0.2, verbose=False)[0]

    def match(ub, uc, tb, tc):
        if len(ub) != len(tb):
            return 0.0, 1.0
        tb = np.array(tb).reshape(-1, 4); tc = np.array(tc)
        worst_iou, worst_c = 1.0, 0.0
        for b, c in zip(ub, uc):
            ious = iou_matrix([b], tb)[0]
            j = int(ious.argmax())
            worst_iou = min(worst_iou, float(ious[j])); worst_c = max(worst_c, abs(float(c) - tc[j]))
        return worst_iou, worst_c

    wi, wc_ = match(up.boxes.xyxy.numpy(), up.boxes.conf.numpy(), res["persons"], res["pconf"])
    check_true("사람: 개수 같고 박스 IoU ≥ 0.99, conf 차 ≤ 1e-3", wi >= 0.99 and wc_ <= 1e-3,
               f"min IoU {wi:.4f}, max Δconf {wc_:.5f}, n={len(up.boxes)}")
    wi, wc_ = match(uw.boxes.xyxy.numpy(), uw.boxes.conf.numpy(), res["waste"], res["wconf"])
    check_true("폐기물: 개수 같고 박스 IoU ≥ 0.99, conf 차 ≤ 1e-3", wi >= 0.99 and wc_ <= 1e-3,
               f"min IoU {wi:.4f}, max Δconf {wc_:.5f}, n={len(uw.boxes)}")

    enc = ReID(str(ROOT / "models/yolo26n-reid.onnx"), imgsz=224, device="cpu")
    pb = np.array(res["persons"]).reshape(-1, 4)
    dets = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]] for b in pb], np.float32)
    ue = enc(img, dets)
    cos = [float(np.dot(np.array(a) / np.linalg.norm(a), np.array(b) / np.linalg.norm(b)))
           for a, b in zip(ue, res["emb"]) if a is not None and b is not None]
    check_true("Re-ID 임베딩 코사인 ≥ 0.999", bool(cos) and min(cos) >= 0.999,
               f"min cos {min(cos):.5f}" if cos else "비교 불가")

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 torch-free 백엔드 테스트 통과")
