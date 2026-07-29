#!/usr/bin/env bash
# TensorRT 엔진 변환 — 반드시 Jetson 보드 위에서 실행할 것.
# (.engine은 빌드한 GPU 아키텍처 전용이라 PC에서 만든 것은 Jetson에서 동작하지 않음)
set -e
cd "$(dirname "$0")"

echo "== TensorRT 변환 시작 (수 분 소요, FP16) =="

# 방법 1) ultralytics 내장 export (권장) — .pt에서 직접 변환
python3 - <<'EOF'
from ultralytics import YOLO
for stem in ["waste10_yolo26n", "person_yolo26n"]:
    m = YOLO(f"models/{stem}.pt")
    path = m.export(format="engine", imgsz=640, half=True, device=0)
    import shutil
    shutil.move(path, f"models/{stem}.engine")
    print(f"OK: models/{stem}.engine")
EOF

# 방법 2) ultralytics 실패 시 trtexec로 ONNX에서 직접 빌드 (주석 해제)
# /usr/src/tensorrt/bin/trtexec --onnx=models/waste10_yolo26n.onnx \
#     --saveEngine=models/waste10_yolo26n.engine --fp16 --memPoolSize=workspace:2048
# /usr/src/tensorrt/bin/trtexec --onnx=models/person_yolo26n.onnx \
#     --saveEngine=models/person_yolo26n.engine --fp16 --memPoolSize=workspace:2048

echo "== 완료. dump_monitor_jetson.py 가 .engine 을 자동으로 사용합니다 =="
