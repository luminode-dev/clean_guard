#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""안면 모자이크 단위 테스트 (YOLO/스피커 불필요. YuNet 모델은 있으면 함께 검증).

  python3 tests/test_mosaic.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mosaic import (DEFAULT_FACE_MODEL, HEAD_RATIO, FaceMosaic, head_box,  # noqa: E402
                    load_face_detector, pixelate)

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


def noisy(h=240, w=320, seed=0):
    """픽셀화 효과를 측정할 수 있도록 고주파 노이즈 프레임을 만든다."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def local_var(img, box):
    x1, y1, x2, y2 = map(int, box)
    return float(img[y1:y2, x1:x2].astype(np.float32).var())


print("1) pixelate: 영역 안은 뭉개지고 밖은 그대로")
img = noisy()
orig = img.copy()
box = [100, 50, 180, 130]
check_true("적용됨", pixelate(img, box))
check_true("영역 안 분산 크게 감소", local_var(img, box) < local_var(orig, box) * 0.3,
           f"{local_var(orig, box):.0f} -> {local_var(img, box):.0f}")
mask = np.ones(img.shape[:2], bool)
mask[50:130, 100:180] = False
check_true("영역 밖 변화 없음", np.array_equal(img[mask], orig[mask]))

print("\n2) pixelate: 경계 클램프 / 너무 작은 영역")
img = noisy()
check_true("프레임 밖으로 넘치는 박스", pixelate(img, [-50, -50, 40, 40]))
check_true("완전히 밖인 박스는 무시", not pixelate(img, [500, 500, 600, 600]))
check_true("1px 박스는 무시", not pixelate(img, [10, 10, 11, 11]))
check_true("float 좌표 허용", pixelate(img, [10.4, 10.6, 60.2, 70.9]))

print("\n3) head_box: 사람 박스 상단 + 좌우 여유")
hb = head_box([100, 0, 140, 200])
check("상단 y", hb[1], 0)
check("머리 높이 = 키 * HEAD_RATIO", round(hb[3] - hb[1]), round(200 * HEAD_RATIO))
check_true("좌우로 넓어짐", hb[0] < 100 and hb[2] > 140, f"{hb}")

print("\n4) FaceMosaic head 모드")
fm = FaceMosaic("head")
check_true("enabled", fm.enabled)
img = noisy()
orig = img.copy()
persons = [[40, 60, 90, 200], [200, 40, 260, 220]]
n = fm.apply(img, persons)
check("가린 영역 수", n, 2)
check("head 카운트", fm.n_heads, 2)
for pb in persons:
    hb = head_box(pb)
    check_true(f"머리 영역 뭉개짐 {pb}", local_var(img, hb) < local_var(orig, hb) * 0.3)
    body = [pb[0], pb[1] + (pb[3] - pb[1]) * 0.5, pb[2], pb[3]]
    check_true(f"몸통은 그대로 {pb}", np.array_equal(
        img[int(body[1]):int(body[3]), int(body[0]):int(body[2])],
        orig[int(body[1]):int(body[3]), int(body[0]):int(body[2])]))
check("사람 없으면 0", fm.apply(img, []), 0)

print("\n5) off 모드는 아무것도 안 함")
fm_off = FaceMosaic("off")
img = noisy()
orig = img.copy()
check("반환 0", fm_off.apply(img, persons), 0)
check_true("프레임 불변", np.array_equal(img, orig))
check_true("enabled=False", not fm_off.enabled)

print("\n6) face 모드 — 모델 없으면 head로 폴백")
fm_nomodel = FaceMosaic("face", model_path="models/__없는파일__.onnx")
check("모드 강등", fm_nomodel.mode, "head")

if DEFAULT_FACE_MODEL.exists() and hasattr(cv2, "FaceDetectorYN"):
    print("\n7) face 모드 — YuNet 로드 + 얼굴 없는 사람은 head 폴백")
    det = load_face_detector()
    check_true("YuNet 로드", det is not None)
    fm_face = FaceMosaic("face")
    check("모드 유지", fm_face.mode, "face")
    img = noisy()
    orig = img.copy()
    n = fm_face.apply(img, persons)                # 노이즈엔 얼굴이 없다 -> head 폴백
    check("가린 영역 수", n, 2)
    check("head 폴백 카운트", fm_face.n_heads, 2)
    check("얼굴 카운트", fm_face.n_faces, 0)
    check_true("작은 사람 박스(<32px)도 예외 없이 head 처리",
               fm_face.apply(img, [[5, 5, 20, 30]]) == 1)
    check_true("summary 문자열", "face" in fm_face.summary())
else:
    print(f"\n7) (건너뜀) YuNet 모델 없음 또는 cv2.FaceDetectorYN 미지원: {DEFAULT_FACE_MODEL}")

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 모자이크 테스트 통과")
