# -*- coding: utf-8 -*-
"""사람 안면 모자이크 (OpenCV만 사용, 추가 pip 의존성 없음).

출력물(이벤트 스냅샷 / annotated.mp4 / --show 화면)에 찍히는 사람 얼굴을 픽셀화한다.
추론에 쓰는 프레임은 저장되지 않으므로 출력 직전에만 적용하면 탐지·재식별에는 영향이 없다.

두 모드:
- head : 사람 박스 상단 HEAD_RATIO 영역을 픽셀화. 모델 불필요, 비용 0.
         사람 탐지가 된 한 항상 가려지지만 얼굴보다 넓게 가린다
- face : YuNet(cv2.FaceDetectorYN, OpenCV 4.5.4+ 내장)으로 사람 박스 크롭 안에서 얼굴을 찾아
         얼굴만 정확히 가리고, 못 찾으면(뒷모습·모자·저조도) 그 사람은 head로 폴백.
         전체 프레임이 아니라 사람 크롭에만 돌리므로 Orin Nano CPU에서도 프레임당 수 ms 수준

모델: models/face_detection_yunet_2023mar.onnx (약 230KB, opencv_zoo)
"""
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
DEFAULT_FACE_MODEL = HERE / "models" / "face_detection_yunet_2023mar.onnx"

MODES = ("off", "head", "face")
HEAD_RATIO = 0.18        # 사람 박스 높이 중 머리로 볼 상단 비율
HEAD_PAD = 0.10          # 머리 영역 좌우로 더 넓힐 비율 (박스 폭 기준)
FACE_PAD = 0.25          # 얼굴 박스 사방으로 더 넓힐 비율 (턱·이마·귀 포함)
FACE_SCORE = 0.5         # YuNet 신뢰도 임계
FACE_NMS = 0.3
PERSON_CROP_PAD = 0.15   # 얼굴 탐지용 사람 크롭 여유 (박스가 머리를 살짝 자르는 경우 대비)
MIN_CROP = 32            # 이보다 작은 크롭은 YuNet에 넣지 않고 head로 처리
CELLS = 8                # 픽셀화 강도: 영역의 짧은 변을 이 칸 수로 뭉갠다 (얼굴 크기에 비례)
MIN_BLOCK = 6            # 작은 얼굴도 최소 이 픽셀 크기의 블록으로


def _clamp(box, w, h):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def pixelate(frame, box, cells=CELLS, min_block=MIN_BLOCK):
    """box 영역을 제자리에서 픽셀화. 영역이 2px 미만이면 무시.

    블록 크기를 영역 크기에 비례시켜(짧은 변 / cells) 큰 얼굴도 작은 얼굴도
    같은 정도로 식별 불가하게 만든다. 고정 픽셀 블록은 큰 얼굴에서 윤곽이 남는다."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp(box, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return False
    block = max(min_block, min(bw, bh) // cells)
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, bw // block), max(1, bh // block)),
                       interpolation=cv2.INTER_AREA)      # 블록 전체 평균 (LINEAR는 2x2만 샘플)
    frame[y1:y2, x1:x2] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    return True


def head_box(pbox, ratio=HEAD_RATIO, pad=HEAD_PAD):
    """사람 박스에서 머리 영역 추정 (상단 ratio, 좌우 pad만큼 확장)."""
    x1, y1, x2, y2 = pbox
    bw = x2 - x1
    return [x1 - bw * pad, y1, x2 + bw * pad, y1 + (y2 - y1) * ratio]


def load_face_detector(model_path=None):
    """YuNet 로더. cv2에 FaceDetectorYN이 없거나 모델이 없으면 None (호출자가 head로 폴백)."""
    path = Path(model_path) if model_path else DEFAULT_FACE_MODEL
    if not hasattr(cv2, "FaceDetectorYN"):
        print(f"[모자이크 경고] 이 OpenCV({cv2.__version__})에는 FaceDetectorYN이 없음 "
              f"(4.5.4+ 필요) -> head 모드로 동작")
        return None
    if not path.exists():
        print(f"[모자이크 경고] 얼굴 모델 없음: {path} -> head 모드로 동작. "
              f"README의 'YuNet 모델' 항목 참고")
        return None
    try:
        return cv2.FaceDetectorYN.create(str(path), "", (320, 320), FACE_SCORE, FACE_NMS, 5000)
    except Exception as e:                                        # noqa: BLE001
        print(f"[모자이크 경고] 얼굴 모델 로드 실패 ({type(e).__name__}: {e}) -> head 모드로 동작")
        return None


def detect_faces_in(frame, pbox, detector, pad=PERSON_CROP_PAD):
    """사람 박스(여유 pad 포함) 크롭에서 얼굴을 찾아 프레임 좌표 박스 목록을 반환."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = pbox
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1, cx2, cy2 = _clamp([x1 - bw * pad, y1 - bh * pad, x2 + bw * pad, y2 + bh * pad], w, h)
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw < MIN_CROP or ch < MIN_CROP:
        return []
    crop = frame[cy1:cy2, cx1:cx2]
    detector.setInputSize((cw, ch))
    _, faces = detector.detect(crop)
    if faces is None:
        return []
    out = []
    for fx, fy, fw, fh in faces[:, :4]:
        px, py = fw * FACE_PAD, fh * FACE_PAD
        out.append([cx1 + fx - px, cy1 + fy - py, cx1 + fx + fw + px, cy1 + fy + fh + py])
    return out


class FaceMosaic:
    """감시 루프가 프레임마다 apply(frame, person_boxes)를 호출한다."""

    def __init__(self, mode="head", model_path=None, cells=CELLS):
        if mode not in MODES:
            raise ValueError(f"mode는 {MODES} 중 하나여야 함: {mode!r}")
        self.mode = mode
        self.cells = cells
        self.detector = load_face_detector(model_path) if mode == "face" else None
        if mode == "face" and self.detector is None:
            self.mode = "head"
        self.n_faces = 0          # YuNet으로 가린 얼굴 수
        self.n_heads = 0          # head 추정으로 가린 수

    @property
    def enabled(self):
        return self.mode != "off"

    def apply(self, frame, person_boxes):
        """person_boxes의 얼굴/머리를 제자리에서 픽셀화. 가린 영역 수 반환."""
        if self.mode == "off" or not person_boxes:
            return 0
        n = 0
        for pb in person_boxes:
            faces = []
            if self.detector is not None:
                try:
                    faces = detect_faces_in(frame, pb, self.detector)
                except Exception as e:                            # noqa: BLE001
                    print(f"[모자이크 경고] 얼굴 탐지 실패 ({type(e).__name__}: {e}) -> head")
                    faces = []
            if faces:
                for fb in faces:
                    n += pixelate(frame, fb, self.cells)
                self.n_faces += len(faces)
            else:
                n += pixelate(frame, head_box(pb), self.cells)
                self.n_heads += 1
        return n

    def summary(self):
        if self.mode == "face":
            return f"모자이크(face): 얼굴 {self.n_faces}건 · head 폴백 {self.n_heads}건"
        return f"모자이크(head): {self.n_heads}건"


__all__ = ["FaceMosaic", "MODES", "DEFAULT_FACE_MODEL", "pixelate", "head_box",
           "load_face_detector", "detect_faces_in"]
