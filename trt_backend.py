# -*- coding: utf-8 -*-
"""torch 없는 추론 백엔드 — TensorRT(.engine) 또는 onnxruntime(.onnx) 직접 호출.

ultralytics는 학습·내보내기·모든 백엔드를 하나의 API로 다루기 위해 torch 위에 만들어져 있어,
만들어진 엔진을 돌리기만 하는 배포 장비에서는 torch(1GB+ 상주)가 잉여다. 이 모듈은 그 경로에서
ultralytics가 실제로 하는 일만 numpy/OpenCV로 다시 구현한다:

  전처리  : 레터박스(114 패딩, 중앙 정렬) -> BGR->RGB -> /255 -> NCHW
  추론    : TensorRT Python API + cuda-python 로 GPU 버퍼 관리  (PC 개발용: onnxruntime)
  후처리  : YOLO26 e2e 출력 (1, 300, 6)=[x1,y1,x2,y2,conf,cls] -> conf 필터 -> 원좌표 복원
  Re-ID   : ultralytics ReID와 같은 크롭 규약(gain 1.02, pad 10) -> 224x224 -> /255 -> 임베딩

검증: PC에서 onnxruntime 경로가 ultralytics 결과와 일치함 (박스 IoU >= 0.994, conf 차 <= 5e-4).
TensorRT 경로는 같은 그래프를 다른 런타임이 실행하는 것이라 수학은 같지만 **보드에서 확인 필요**.

런타임 선택: 경로 확장자로 결정. .engine -> TensorRT, .onnx -> onnxruntime.
Jetson에서는 .engine만 쓰면 onnxruntime(numpy 2 충돌)이 필요 없다.
"""
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PAD_VALUE = 114


# =============================================================== 런타임
class _Runtime:
    """공통 인터페이스: run(x: np.ndarray NCHW float32) -> np.ndarray (배치 첫 출력)."""

    name = "?"
    input_dtype = np.float32

    def run(self, x):
        raise NotImplementedError

    def close(self):
        pass


class OrtRuntime(_Runtime):
    """onnxruntime CPU/CUDA. PC 개발·테스트용. (Jetson 기본 cv2와 numpy 충돌 -> 보드에선 engine 권장)"""

    name = "onnxruntime"

    def __init__(self, path, providers=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def run(self, x):
        return self.sess.run(None, {self.inp: np.ascontiguousarray(x, dtype=np.float32)})[0]


class TrtRuntime(_Runtime):
    """TensorRT 10 (JetPack 6) + cuda-python. 동적 배치/크기 엔진 지원.

    보드 검증 필요 항목: (1) cuda-python import 경로 (12.x: cuda.cudart / 12.6+: cuda.bindings.runtime)
    (2) FP16 입력 엔진일 때 dtype 변환 (3) 동적 shape set_input_shape 후 출력 크기.
    """

    name = "tensorrt"

    def __init__(self, path):
        import tensorrt as trt
        try:
            from cuda.bindings import runtime as cudart      # cuda-python >= 12.6
        except ImportError:                                  # noqa: BLE001
            from cuda import cudart                          # cuda-python 12.x
        self.trt, self.cudart = trt, cudart
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"엔진 역직렬화 실패: {path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.out = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.inp))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out))
        self.dynamic = -1 in tuple(self.engine.get_tensor_shape(self.inp))
        self._check(cudart.cudaStreamCreate())
        self.stream = self._last
        self.d_in = self.d_out = None
        self.in_bytes = self.out_bytes = 0
        if not self.dynamic:
            self._alloc(tuple(self.engine.get_tensor_shape(self.inp)))

    def _check(self, ret):
        err, *vals = ret
        if err != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA 오류: {err}")
        self._last = vals[0] if len(vals) == 1 else vals
        return self._last

    def _alloc(self, in_shape):
        self.ctx.set_input_shape(self.inp, in_shape)
        out_shape = tuple(self.ctx.get_tensor_shape(self.out))
        in_bytes = int(np.prod(in_shape)) * np.dtype(self.input_dtype).itemsize
        out_bytes = int(np.prod(out_shape)) * np.dtype(self.output_dtype).itemsize
        if in_bytes > self.in_bytes:
            if self.d_in is not None:
                self._check(self.cudart.cudaFree(self.d_in))
            self.d_in = self._check(self.cudart.cudaMalloc(in_bytes)); self.in_bytes = in_bytes
        if out_bytes > self.out_bytes:
            if self.d_out is not None:
                self._check(self.cudart.cudaFree(self.d_out))
            self.d_out = self._check(self.cudart.cudaMalloc(out_bytes)); self.out_bytes = out_bytes
        self.ctx.set_tensor_address(self.inp, int(self.d_in))
        self.ctx.set_tensor_address(self.out, int(self.d_out))
        self.out_shape = out_shape
        self.in_shape = in_shape

    def run(self, x):
        x = np.ascontiguousarray(x, dtype=self.input_dtype)
        if self.dynamic or tuple(x.shape) != tuple(self.in_shape):
            self._alloc(tuple(x.shape))
        cudart, kind = self.cudart, self.cudart.cudaMemcpyKind
        self._check(cudart.cudaMemcpyAsync(self.d_in, x.ctypes.data, x.nbytes,
                                           kind.cudaMemcpyHostToDevice, self.stream))
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT 실행 실패")
        out = np.empty(self.out_shape, dtype=self.output_dtype)
        self._check(cudart.cudaMemcpyAsync(out.ctypes.data, self.d_out, out.nbytes,
                                           kind.cudaMemcpyDeviceToHost, self.stream))
        self._check(cudart.cudaStreamSynchronize(self.stream))
        return out

    def close(self):
        for d in (self.d_in, self.d_out):
            if d is not None:
                self.cudart.cudaFree(d)
        self.d_in = self.d_out = None


def open_runtime(path):
    path = Path(path)
    if path.suffix == ".engine":
        return TrtRuntime(path)
    if path.suffix == ".onnx":
        return OrtRuntime(path)
    raise ValueError(f"지원하지 않는 모델 포맷: {path.name} (.engine 또는 .onnx)")


def find_model(stem_or_path, exts=(".engine", ".onnx")):
    """models/에서 .engine > .onnx 순으로 존재하는 후보 목록 (경로가 주어지면 그것만)."""
    if stem_or_path and Path(stem_or_path).suffix:
        return [Path(stem_or_path)]
    return [p for p in (HERE / "models" / (stem_or_path + e) for e in exts) if p.exists()]


def load_with_fallback(stem_or_path, factory, label):
    """후보를 순서대로 열어 보고(워밍업 포함) 처음 성공한 것을 쓴다. ultralytics 경로와 같은 정책."""
    cands = find_model(stem_or_path)
    if not cands:
        raise FileNotFoundError(f"models/{stem_or_path}.(engine|onnx) 없음")
    for p in cands:
        try:
            obj = factory(p)
            obj.warmup()
            return obj, str(p)
        except Exception as e:                                    # noqa: BLE001
            print(f"[경고] {label} {p.name} 사용 불가 ({type(e).__name__}: {str(e)[:120]}) -> 다음 포맷 시도")
    raise RuntimeError(f"{label}: 사용 가능한 모델 포맷이 없음")


# =============================================================== 전처리
def letterbox(img, size=640, pad=PAD_VALUE):
    """ultralytics LetterBox(auto=False, scaleup=True)와 동일: 비율 유지 리사이즈 + 중앙 패딩."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad,) * 3)
    return out, r, (left, top)


def to_input(bgr):
    """HWC BGR uint8 -> NCHW RGB float32 [0,1]"""
    x = bgr[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def resize_bilinear(img, size):
    """torch F.interpolate(mode="bilinear", align_corners=False)와 같은 float 이중선형 보간.

    cv2.resize(INTER_LINEAR)는 고정소수점이라 픽셀당 최대 ~0.8 차이가 나서 Re-ID 임베딩
    코사인이 0.996까지 떨어진다. 이 구현은 최대차 0.005 (224x224 크롭 1장 ~2ms)."""
    H, W = img.shape[:2]

    def coords(n_in, n_out):
        x = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
        x0 = np.floor(x).astype(int)
        return x0, np.minimum(x0 + 1, n_in - 1), (x - x0).astype(np.float32)

    y0, y1, wy = coords(H, size)
    x0, x1, wx = coords(W, size)
    f = img.astype(np.float32)
    wx = wx[None, :, None]
    wy = wy[:, None, None]
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


# =============================================================== 탐지기
class Detector:
    """YOLO26 e2e 엔진. detect(frame, conf, classes) -> (boxes xyxy (N,4), conf (N,), cls (N,))"""

    def __init__(self, path, imgsz=640):
        self.rt = open_runtime(path)
        self.imgsz = imgsz
        self.path = str(path)

    def warmup(self):
        self.detect(np.zeros((self.imgsz, self.imgsz, 3), np.uint8))

    def detect(self, frame, conf=0.25, classes=None):
        lb, r, (px, py) = letterbox(frame, self.imgsz)
        out = self.rt.run(to_input(lb))
        out = np.asarray(out, dtype=np.float32).reshape(-1, 6)       # (300, 6)
        keep = out[:, 4] >= conf
        if classes is not None:
            keep &= np.isin(out[:, 5].astype(int), list(classes))
        out = out[keep]
        boxes = out[:, :4].copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, out[:, 4], out[:, 5].astype(int)

    def close(self):
        self.rt.close()


# =============================================================== Re-ID 인코더
class ReIDEncoder:
    """ultralytics ReID(비-.pt 경로)와 같은 규약. __call__(img, dets_xywh) -> [emb (512,) | None]

    크롭: xywh -> wh*1.02+10 -> xyxy(int 절삭) -> 클립.  입력: BGR /255, 224x224 bilinear(torch 동일).
    동적 배치 엔진이면 한 번에, 고정 배치 엔진이면 잘라서 넣는다.
    """

    def __init__(self, path, imgsz=224, gain=1.02, pad=10, max_batch=16):
        self.rt = open_runtime(path)
        self.imgsz, self.gain, self.pad, self.max_batch = imgsz, gain, pad, max_batch
        self.path = str(path)

    def warmup(self):
        self(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), np.array([[112, 112, 100, 200]], np.float32))

    def _crop(self, img, det):
        cx, cy, w, h = det[:4]
        w, h = w * self.gain + self.pad, h * self.gain + self.pad
        x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
        H, W = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    def __call__(self, img, dets):
        crops = [self._crop(img, d) for d in np.asarray(dets, dtype=np.float32).reshape(-1, 4)]
        valid = [i for i, c in enumerate(crops) if c is not None and c.size]
        embs = [None] * len(crops)
        if not valid:
            return embs
        batch = np.stack([resize_bilinear(crops[i], self.imgsz) for i in valid]
                         ).transpose(0, 3, 1, 2) / 255.0
        outs = []
        for s in range(0, len(batch), self.max_batch):
            outs.append(np.asarray(self.rt.run(np.ascontiguousarray(batch[s:s + self.max_batch])),
                                   dtype=np.float32).reshape(-1, 512))
        feats = np.concatenate(outs, 0)
        for k, i in enumerate(valid):
            embs[i] = feats[k]
        return embs

    def close(self):
        self.rt.close()


def load_detector(stem_or_path):
    return load_with_fallback(stem_or_path, Detector, "탐지")


def load_reid(stem_or_path="yolo26n-reid"):
    return load_with_fallback(stem_or_path, ReIDEncoder, "Re-ID")


__all__ = ["Detector", "ReIDEncoder", "ByteTrack", "load_detector", "load_reid",
           "letterbox", "open_runtime", "TrtRuntime", "OrtRuntime"]

from bytetrack import ByteTrack  # noqa: E402  (재수출: 백엔드 사용자가 한 곳에서 가져가도록)
