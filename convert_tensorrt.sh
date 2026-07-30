#!/usr/bin/env bash
# Build Jetson Orin Nano-specific TensorRT FP16 engines.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

readonly IMG_SIZE="${IMG_SIZE:-640}"
readonly TRT_WORKSPACE_MB="${TRT_WORKSPACE_MB:-2048}"
readonly TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
readonly MODELS=(waste10_yolo26n person_yolo26n)

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "${IMG_SIZE}" =~ ^[0-9]+$ ]] || die "IMG_SIZE must be an integer"
[[ "${TRT_WORKSPACE_MB}" =~ ^[0-9]+$ ]] || die "TRT_WORKSPACE_MB must be an integer"
command -v python3 >/dev/null || die "python3 is required"
[[ -d models ]] || die "models directory not found"

echo "== TensorRT FP16 conversion (Jetson Orin Nano) =="
echo "image=${IMG_SIZE} batch=1 workspace=${TRT_WORKSPACE_MB}MiB"

build_with_ultralytics() {
  python3 - <<'PY'
from pathlib import Path
import shutil
from ultralytics import YOLO

img_size = int(__import__("os").environ["IMG_SIZE"])
for stem in ("waste10_yolo26n", "person_yolo26n"):
    target = Path("models") / f"{stem}.engine"
    exported = YOLO(f"models/{stem}.pt").export(
        format="engine", imgsz=img_size, batch=1, half=True, device=0,
        verbose=False,
    )
    exported = Path(str(exported))
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"empty engine: {target}")
    print(f"OK: {target}")
PY
}

build_with_trtexec() {
  [[ -x "${TRTEXEC}" ]] || die "trtexec not found at ${TRTEXEC}; set TRTEXEC to its path"
  for stem in "${MODELS[@]}"; do
    local onnx="models/${stem}.onnx"
    local engine="models/${stem}.engine"
    [[ -s "${onnx}" ]] || die "missing ONNX model: ${onnx}"
    "${TRTEXEC}" --onnx="${onnx}" --saveEngine="${engine}" \
      --fp16 --memPoolSize="workspace:${TRT_WORKSPACE_MB}" \
      --verbose
    [[ -s "${engine}" ]] || die "trtexec produced no engine: ${engine}"
    echo "OK: ${engine}"
  done
}

export IMG_SIZE TRT_WORKSPACE_MB
if build_with_ultralytics; then
  :
else
  echo "Ultralytics export failed; falling back to trtexec ONNX build." >&2
  build_with_trtexec
fi

for stem in "${MODELS[@]}"; do
  test -s "models/${stem}.engine" || die "engine verification failed: models/${stem}.engine"
done
echo "== Complete: both FP16 engines are ready =="
