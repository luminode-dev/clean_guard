# -*- coding: utf-8 -*-
"""TTS 백엔드 — 텍스트 목록을 wav 파일로 렌더링한다.

- SupertonicBackend : TTS 전용 venv의 python으로 tts/synth_worker.py를 실행.
                      감시 프로세스는 onnxruntime을 import 하지 않는다.
- NullBackend       : 무음 wav. 모델 없이 전 경로(문구/캐시/재생) 검증용.

두 백엔드 모두 render(jobs) 시그니처를 공유한다.
  jobs: [(text, out_path), ...] -> 성공한 out_path 개수
"""
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent                       # clean_guard/
WORKER = HERE / "synth_worker.py"
DEFAULT_VENV = PKG_ROOT / ".venv-tts"
MODEL_ROOT = PKG_ROOT / "models" / "tts"     # setup_tts.sh가 여기에 내려받는다


def resolve_model_dir(model, model_dir=None):
    """모델 폴더 결정: 명시값 > models/tts/<model> > None(supertonic 기본 캐시).

    setup_tts.sh는 models/tts/ 아래에 두므로, 사전 렌더링과 런타임 폴백이
    같은 가중치를 쓰도록 여기서 한 곳에 모아 해석한다.
    """
    if model_dir:
        return str(model_dir)
    local = MODEL_ROOT / model
    if (local / "onnx" / "vocoder.onnx").exists():
        return str(local)
    return None


def find_venv_python(venv_dir=None):
    """TTS venv의 python 경로. 없으면 supertonic이 현재 인터프리터에 있는지 보고 폴백."""
    cand = Path(venv_dir) if venv_dir else DEFAULT_VENV
    py = cand / "bin" / "python"
    if py.exists():
        return str(py)
    try:
        import supertonic  # noqa: F401
        return sys.executable
    except Exception:
        return None


class NullBackend:
    """무음 wav 생성기. 텍스트 길이에 비례한 길이의 16kHz mono 무음."""

    name = "null"
    sample_rate = 16000

    def __init__(self, chars_per_sec=7.0, **_kw):
        self.chars_per_sec = chars_per_sec

    def available(self):
        return True

    def render(self, jobs, verbose=False):
        ok = 0
        for text, out in jobs:
            sec = max(0.4, len(text) / self.chars_per_sec)
            n = int(self.sample_rate * sec)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.sample_rate)
                f.writeframes(b"\x00\x00" * n)
            ok += 1
            if verbose:
                print(f"  [null] {sec:4.1f}s  {text[:40]}")
        return ok


class SupertonicBackend:
    """Supertonic ONNX 합성. 항상 별도 프로세스에서 실행된다."""

    name = "supertonic"

    def __init__(self, model="supertonic-2", model_dir=None, voice="M4", lang="ko",
                 steps=8, speed=1.0, threads=2, venv_dir=None, timeout=600):
        self.model = model
        self.model_dir = resolve_model_dir(model, model_dir)
        self.voice = voice
        self.lang = lang
        self.steps = steps
        self.speed = speed
        self.threads = threads
        self.timeout = timeout
        self.python = find_venv_python(venv_dir)

    def available(self):
        return self.python is not None and WORKER.exists()

    def render(self, jobs, verbose=False):
        if not jobs:
            return 0
        if not self.available():
            raise RuntimeError(
                "Supertonic 실행 환경 없음. setup_tts.sh를 먼저 실행하세요 "
                f"(찾은 경로: {DEFAULT_VENV})")
        for _t, out in jobs:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = [{"text": t, "out": str(o)} for t, o in jobs]
        fd, jobs_path = tempfile.mkstemp(suffix=".json", prefix="tts_jobs_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            cmd = [self.python, str(WORKER), "--jobs", jobs_path,
                   "--model", self.model, "--voice", self.voice,
                   "--lang", self.lang, "--steps", str(self.steps),
                   "--speed", str(self.speed), "--threads", str(self.threads)]
            if self.model_dir:
                cmd += ["--model-dir", self.model_dir]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout * max(1, len(jobs)))
            ok = 0
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("ok"):
                    ok += 1
                    if verbose:
                        print(f"  [supertonic] {r['sec']:4.1f}s  {Path(r['out']).name}")
                else:
                    print(f"[TTS 경고] 합성 실패 {Path(r.get('out', '?')).name}: "
                          f"{r.get('err')}")
            if ok == 0:
                tail = (proc.stderr or "").strip().splitlines()[-6:]
                raise RuntimeError("합성 결과 없음 (rc=%d)\n%s"
                                   % (proc.returncode, "\n".join(tail)))
            return ok
        finally:
            try:
                os.unlink(jobs_path)
            except OSError:
                pass


def make_backend(name="supertonic", **kw):
    if name == "null":
        return NullBackend(**{k: v for k, v in kw.items() if k == "chars_per_sec"})
    if name == "supertonic":
        return SupertonicBackend(**kw)
    raise ValueError(f"알 수 없는 TTS 백엔드: {name}")


def wav_duration(path):
    """wav 길이(초). 실패 시 0.0"""
    try:
        with wave.open(str(path), "rb") as f:
            return f.getnframes() / float(f.getframerate() or 1)
    except Exception:
        return 0.0


__all__ = ["NullBackend", "SupertonicBackend", "make_backend", "resolve_model_dir",
           "find_venv_python", "wav_duration", "DEFAULT_VENV", "MODEL_ROOT", "WORKER"]
