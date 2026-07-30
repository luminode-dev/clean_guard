# -*- coding: utf-8 -*-
"""폐기물 크롭 -> 한국어 색상 이름 (HSV 규칙 기반, 신규 의존성 없음).

dump_monitor_jetson.py의 color_hist()와 같은 크롭 규약(경계 클램프)을 따르되,
Re-ID용 히스토그램이 아니라 방송 문구에 쓸 '대표 색 이름' 하나를 뽑는다.

cv2는 판정 시점에 import 한다. TTS 전용 venv(onnxruntime + numpy2, cv2 없음)에서
prerender_tts.py가 이 모듈을 import 해도 깨지지 않게 하기 위함이다.
"""
import numpy as np

# 방송 문구에 쓰이는 11색. prerender_tts.py가 이 순서로 캐시를 만든다.
COLOR_NAMES = ["빨간색", "주황색", "노란색", "초록색", "파란색", "보라색",
               "분홍색", "갈색", "흰색", "회색", "검은색"]

_ACHROMATIC = {"흰색", "회색", "검은색"}

# --- 판정 임계 (OpenCV HSV: H 0~179, S/V 0~255) ---
V_BLACK = 55          # 이보다 어두우면 무조건 검은색
S_ACHROMA = 45        # 이보다 채도가 낮으면 무채색(흰/회)
V_WHITE = 190         # 무채색 중 이보다 밝으면 흰색
V_BROWN = 120         # 적/주황 계열이 이보다 어두우면 갈색
CHROMA_PREFER = 0.30  # 유채색 픽셀 비율이 이 이상이면 유채색으로 명명
MARGIN = 0.15         # 박스 상하좌우에서 잘라낼 비율 (배경 오염 제거)

# 유채색 H 구간 -> 이름. (하한, 상한, 이름), 상한 포함.
_HUE_BINS = [
    (0, 9, "빨간색"),
    (10, 20, "주황색"),
    (21, 33, "노란색"),
    (34, 77, "초록색"),
    (78, 130, "파란색"),
    (131, 155, "보라색"),
    (156, 169, "분홍색"),
    (170, 179, "빨간색"),
]


def crop_box(frame_bgr, box, margin=MARGIN):
    """박스 안쪽 (1-2*margin) 영역을 잘라 반환. 실패 시 None."""
    if frame_bgr is None or box is None:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw > 8 and bh > 8:
        x1 += bw * margin
        x2 -= bw * margin
        y1 += bh * margin
        y2 -= bh * margin
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def dominant_color_name(frame_bgr, box=None):
    """대표 색 이름을 반환. 판정 불가면 None.

    box가 None이면 frame_bgr 자체를 이미 잘린 크롭으로 취급한다
    (WasteObject.best_crop 경로).
    """
    crop = crop_box(frame_bgr, box) if box is not None else frame_bgr
    if crop is None or crop.size == 0:
        return None
    return _classify(crop)


def _classify(crop_bgr):
    import cv2

    small = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]

    black = V < V_BLACK
    low_s = (S < S_ACHROMA) & ~black
    white = low_s & (V >= V_WHITE)
    gray = low_s & ~white
    chroma = ~(black | white | gray)

    counts = {}
    n = H.size
    counts["검은색"] = int(black.sum())
    counts["흰색"] = int(white.sum())
    counts["회색"] = int(gray.sum())

    chroma_n = int(chroma.sum())
    if chroma_n:
        # 어두운 적/주황은 갈색으로 (H 0~25 & V < V_BROWN)
        brownish = chroma & (H <= 25) & (V < V_BROWN)
        counts["갈색"] = int(brownish.sum())
        rest = chroma & ~brownish
        for lo, hi, name in _HUE_BINS:
            m = rest & (H >= lo) & (H <= hi)
            c = int(m.sum())
            if c:
                counts[name] = counts.get(name, 0) + c

    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        return None

    if chroma_n / n >= CHROMA_PREFER:
        chromatic = {k: v for k, v in counts.items() if k not in _ACHROMATIC}
        if chromatic:
            return max(chromatic.items(), key=lambda kv: kv[1])[0]
    return max(counts.items(), key=lambda kv: kv[1])[0]
