#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""색상 판정 단위 테스트 (모델/네트워크 불필요).

  python3 tests/test_color_naming.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts.color_naming import COLOR_NAMES, crop_box, dominant_color_name  # noqa: E402

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {got!r} (기대 {want!r})")
    if not ok:
        fails.append(name)


def patch(bgr, size=120):
    """단색 패치 이미지."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = bgr
    return img


def framed(bgr, bg=(128, 128, 128), size=120, inner=0.5):
    """중앙에만 색이 있고 바깥은 배경색인 이미지 -> 마진 크롭 검증용."""
    img = patch(bg, size)
    m = int(size * (1 - inner) / 2)
    img[m:size - m, m:size - m] = bgr
    return img


FULL_BOX = [0, 0, 120, 120]

print("1) 순색 패치 (박스 전체)")
CASES = [
    ("빨강", (0, 0, 220), "빨간색"),
    ("주황", (0, 140, 255), "주황색"),
    ("노랑", (0, 230, 240), "노란색"),
    ("초록", (60, 180, 60), "초록색"),
    ("파랑", (220, 60, 30), "파란색"),
    ("보라", (200, 40, 140), "보라색"),
    ("분홍", (190, 120, 250), "분홍색"),
    ("갈색", (30, 55, 95), "갈색"),
    ("흰색", (245, 245, 245), "흰색"),
    ("회색", (128, 128, 128), "회색"),
    ("검정", (20, 20, 20), "검은색"),
]
for label, bgr, want in CASES:
    check(label, dominant_color_name(patch(bgr), FULL_BOX), want)

print("\n2) 반환값은 항상 COLOR_NAMES 안에 있어야 함")
for label, bgr, _ in CASES:
    got = dominant_color_name(patch(bgr), FULL_BOX)
    if got not in COLOR_NAMES:
        fails.append(f"{label} 범위밖")
        print(f"  FAIL {label}: {got!r}가 COLOR_NAMES에 없음")
print(f"  OK   11색 모두 COLOR_NAMES 소속 ({len(COLOR_NAMES)}색 정의)")

print("\n3) 마진 크롭이 배경(회색)을 걷어내는지")
for label, bgr, want in [("빨강/회색배경", (0, 0, 220), "빨간색"),
                         ("파랑/회색배경", (220, 60, 30), "파란색")]:
    check(label, dominant_color_name(framed(bgr), FULL_BOX), want)

print("\n4) 유채색 비율이 낮으면 무채색으로 (검은 봉투에 낀 컬러 노이즈)")
img = patch((25, 25, 25))
img[:12, :12] = (0, 0, 230)          # 전체의 1% 정도만 빨강
check("검은봉투+빨강노이즈", dominant_color_name(img, FULL_BOX), "검은색")

print("\n5) 방어 동작")
check("None 프레임", dominant_color_name(None, FULL_BOX), None)
check("0크기 박스", dominant_color_name(patch((0, 0, 220)), [10, 10, 10, 10]), None)
check("박스 밖 좌표", dominant_color_name(patch((0, 0, 220)), [500, 500, 600, 600]), None)
check("crop_box 정상", crop_box(patch((0, 0, 220)), FULL_BOX) is not None, True)
# box=None이면 이미 잘린 크롭으로 취급
check("크롭 직접 입력", dominant_color_name(patch((220, 60, 30))), "파란색")

print("\n6) 큰 프레임 안의 작은 박스")
big = patch((128, 128, 128), 480)
big[100:200, 300:380] = (30, 200, 30)
check("초록 객체", dominant_color_name(big, [300, 100, 380, 200]), "초록색")

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 색상 테스트 통과")
