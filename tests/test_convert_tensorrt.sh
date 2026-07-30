#!/usr/bin/env bash
set -euo pipefail

script="$(cd "$(dirname "$0")/.." && pwd)/convert_tensorrt.sh"

grep -q 'TRT_WORKSPACE_MB' "$script"
grep -Fq 'imgsz=img_size' "$script"
grep -q 'IMG_SIZE:-640' "$script"
grep -q 'batch=1' "$script"
grep -q -- '--memPoolSize="workspace:' "$script"
grep -q 'waste10_yolo26n' "$script"
grep -q 'person_yolo26n' "$script"
grep -q 'trtexec' "$script"
grep -q 'test -s' "$script"
bash -n "$script"

echo "conversion script contract: PASS"
