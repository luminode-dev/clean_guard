#!/usr/bin/env bash
# TTS 방송 환경 셋업 (Jetson Orin Nano / JetPack 6 / Python 3.10)
#
# 감시 프로세스(dump_monitor_jetson.py)에 onnxruntime을 들이지 않기 위해
# TTS 의존성은 전용 venv(.venv-tts)에 격리한다.
#
#   bash setup_tts.sh                    # supertonic-2 (약 270MB, 기본)
#   ST_MODEL=supertonic-3 bash setup_tts.sh
#   ST_VOICE=F2 bash setup_tts.sh
#
# 멱등: 이미 된 단계는 건너뛴다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ST_VENV:-$HERE/.venv-tts}"
MODEL="${ST_MODEL:-supertonic-2}"
VOICE="${ST_VOICE:-M4}"
MODEL_DIR="${ST_MODEL_DIR:-$HERE/models/tts/$MODEL}"
MIN_FREE_MB="${ST_MIN_FREE_MB:-600}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[경고] %s\033[0m\n' "$*"; }
die() { printf '\033[31m[실패] %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. 디스크 여유 --------------------------------------------------------
say "1/6 디스크 여유 확인"
FREE_MB=$(df -Pm "$HERE" | awk 'NR==2 {print $4}')
echo "  여유: ${FREE_MB}MB (필요: 최소 ${MIN_FREE_MB}MB)"
if [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
  warn "디스크 여유가 부족합니다. 아래를 정리한 뒤 다시 실행하세요:"
  du -sh "$HOME"/Downloads/* 2>/dev/null | sort -rh | head -5 || true
  echo "  예) rm -f ~/Downloads/*.deb"
  die "여유 공간 부족 (${FREE_MB}MB < ${MIN_FREE_MB}MB)"
fi

# --- 2. venv --------------------------------------------------------------
say "2/6 TTS 전용 venv 준비: $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  # 시스템 site-packages와 완전히 분리한다: onnxruntime이 numpy 2.x를 끌어오는데
  # 시스템 cv2 4.8은 numpy 1.x로 빌드되어 있어 섞이면 감시 파이프라인이 깨진다.
  # ensurepip이 없으므로 --without-pip으로 만들고 get-pip.py로 부트스트랩.
  python3 -m venv --without-pip "$VENV" \
    || die "python3 -m venv 실패 (sudo apt install python3-venv 필요할 수 있음)"
  echo "  venv 생성 완료 (시스템 site-packages와 격리)"
else
  if grep -q "include-system-site-packages *= *true" "$VENV/pyvenv.cfg" 2>/dev/null; then
    warn "기존 venv가 시스템 site-packages를 공유합니다. 격리 모드로 전환합니다."
    sed -i 's/include-system-site-packages *= *true/include-system-site-packages = false/' \
      "$VENV/pyvenv.cfg"
  fi
  echo "  이미 존재 — 건너뜀"
fi

if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "  pip 부트스트랩 중..."
  TMP_GETPIP=$(mktemp /tmp/get-pip-XXXX.py)
  curl -fsSL -o "$TMP_GETPIP" https://bootstrap.pypa.io/pip/get-pip.py \
    || die "get-pip.py 다운로드 실패 (네트워크 확인)"
  "$VENV/bin/python" "$TMP_GETPIP" --no-warn-script-location || die "pip 부트스트랩 실패"
  rm -f "$TMP_GETPIP"
fi
echo "  pip: $("$VENV/bin/python" -m pip --version)"

# --- 3. 패키지 ------------------------------------------------------------
say "3/6 supertonic + onnxruntime 설치"
if "$VENV/bin/python" -c "import supertonic, onnxruntime" >/dev/null 2>&1; then
  echo "  이미 설치됨 — 건너뜀"
else
  "$VENV/bin/python" -m pip install --no-cache-dir --no-warn-script-location \
    "supertonic>=1.3" "onnxruntime>=1.19" "soundfile>=0.12" "huggingface-hub>=0.20" \
    || die "패키지 설치 실패"
fi
"$VENV/bin/python" - <<'PY'
import onnxruntime, supertonic, numpy
print(f"  supertonic={supertonic.__version__} onnxruntime={onnxruntime.__version__} "
      f"numpy={numpy.__version__}")
PY

# --- 4. 모델 --------------------------------------------------------------
say "4/6 모델 다운로드: $MODEL -> $MODEL_DIR"
if [ -f "$MODEL_DIR/onnx/vocoder.onnx" ]; then
  echo "  이미 존재 — 건너뜀 ($(du -sh "$MODEL_DIR" | cut -f1))"
else
  mkdir -p "$(dirname "$MODEL_DIR")"
  "$VENV/bin/python" - "$MODEL" "$MODEL_DIR" <<'PY'
import sys
from supertonic.loader import download_model
model, out = sys.argv[1], sys.argv[2]
print(f"  HuggingFace에서 {model} 내려받는 중 (수백 MB, 시간이 걸립니다)...")
download_model(out, model)
print("  완료")
PY
  # 추론에 쓰이지 않는 샘플/이미지 제거 (디스크 절약)
  rm -rf "$MODEL_DIR/audio_samples" "$MODEL_DIR/img" "$MODEL_DIR/.cache" 2>/dev/null || true
  echo "  크기: $(du -sh "$MODEL_DIR" | cut -f1)"
fi

# --- 5. 오디오 출력 -------------------------------------------------------
say "5/6 오디오 출력 확인"
if command -v pactl >/dev/null 2>&1; then
  pactl list short sinks | sed 's/^/  sink: /' || warn "PulseAudio sink 조회 실패"
fi
PLAYER=""
for p in paplay aplay play; do command -v "$p" >/dev/null 2>&1 && { PLAYER=$p; break; }; done
[ -n "$PLAYER" ] && echo "  재생 명령: $PLAYER" || warn "재생 명령(paplay/aplay/play) 없음 — 방송 불가"

# --- 6. 스모크 + 사전 렌더링 ---------------------------------------------
say "6/6 합성 스모크 테스트"
"$VENV/bin/python" "$HERE/prerender_tts.py" --check --model "$MODEL" \
  --model-dir "$MODEL_DIR" --voice "$VOICE" || die "스모크 테스트 실패"

cat <<EOF

셋업 완료.

다음 단계:
  # 감시 + 방송 실행 (이벤트마다 문구를 즉석 합성 — 사전 작업 불필요)
  python3 $HERE/dump_monitor_jetson.py --source 0 --name cam01 --tts
  python3 $HERE/dump_monitor_jetson.py --source 0 --name cam01 --tts --tts-location "정문 수거함 앞"

  # (선택) 합성 실패 시 폴백용 / --tts-mode cache 용 사전 렌더링 120개
  $VENV/bin/python $HERE/prerender_tts.py --all --model $MODEL --model-dir $MODEL_DIR --voice $VOICE
EOF
