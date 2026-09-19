# -*- coding: utf-8 -*-
"""TTS 백엔드 — 텍스트를 wav 파일로 렌더링한다.

- SupertonicBackend : TTS 전용 venv의 python으로 tts/synth_worker.py를 실행.
                      감시 프로세스는 onnxruntime을 import 하지 않는다.
- NullBackend       : 무음 wav. 모델 없이 전 경로(문구/재생) 검증용.

두 백엔드 모두 같은 인터페이스를 갖는다.
  render(jobs)          일괄: [(text, out_path), ...] -> 성공한 개수 (사전 렌더링용)
  render_one(text, out) 단건: 실시간 방송용. Supertonic은 모델을 올려 둔 상주 워커에
                        보내므로 두 번째 호출부터는 합성 시간만 든다
  warmup()              상주 워커를 미리 띄운다 (첫 방송 지연 제거)
  close()               상주 워커 종료
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
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

    def render_one(self, text, out):
        return self.render([(text, out)]) == 1

    def warmup(self):
        return True

    def close(self):
        pass


class SupertonicBackend:
    """Supertonic ONNX 합성. 항상 별도 프로세스에서 실행된다.

    - render(): 일괄 모드. 워커를 한 번 띄워 작업 목록을 처리하고 종료 (사전 렌더링)
    - render_one(): 상주 모드. `synth_worker.py --serve` 프로세스를 유지하며
      stdin/stdout JSON 라인으로 작업을 주고받는다. 모델 로드(수 초~수십 초)는
      처음 한 번만 든다. 워커가 죽으면 다음 호출에서 자동으로 다시 띄운다.
    """

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
        self._proc = None
        self._lock = threading.Lock()
        self.load_sec = None

    def available(self):
        return self.python is not None and WORKER.exists()

    def _base_cmd(self):
        cmd = [self.python, str(WORKER),
               "--model", self.model, "--voice", self.voice,
               "--lang", self.lang, "--steps", str(self.steps),
               "--speed", str(self.speed), "--threads", str(self.threads)]
        if self.model_dir:
            cmd += ["--model-dir", self.model_dir]
        return cmd

    # --- 상주 워커 (실시간 방송) ---
    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _start_server(self):
        """워커를 띄우고 ready 신호까지 기다린다. 호출자가 _lock을 잡고 있어야 한다."""
        if not self.available():
            raise RuntimeError(
                "Supertonic 실행 환경 없음. setup_tts.sh를 먼저 실행하세요 "
                f"(찾은 경로: {DEFAULT_VENV})")
        self._kill()
        # stderr는 버린다: onnxruntime 로그가 감시 출력에 섞이는 것을 막고,
        # 실패 원인은 워커가 stdout JSON(err 필드)으로 돌려준다
        self._proc = subprocess.Popen(
            self._base_cmd() + ["--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        line = self._proc.stdout.readline()
        try:
            r = json.loads(line) if line.strip() else {}
        except ValueError:
            r = {}
        if not r.get("ready"):
            self._kill()
            raise RuntimeError(f"합성 워커 시작 실패: {r.get('err') or '응답 없음'}")
        self.load_sec = r.get("load_sec")

    def _kill(self):
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            if p.poll() is None:
                if p.stdin:
                    p.stdin.close()
                p.wait(timeout=3)
        except Exception:                                        # noqa: BLE001
            pass
        if p.poll() is None:
            p.kill()

    def warmup(self):
        """상주 워커를 미리 띄운다. 성공하면 True."""
        with self._lock:
            if not self._alive():
                self._start_server()
            return True

    def render_one(self, text, out):
        """문장 하나를 상주 워커로 합성. 성공하면 True, 실패하면 RuntimeError."""
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self._alive():
                self._start_server()
            p = self._proc
            try:
                p.stdin.write(json.dumps({"text": text, "out": str(out)},
                                         ensure_ascii=False) + "\n")
                p.stdin.flush()
                line = p.stdout.readline()
            except (OSError, ValueError) as e:
                self._kill()
                raise RuntimeError(f"합성 워커 통신 실패: {e}")
            if not line:
                self._kill()
                raise RuntimeError("합성 워커가 응답 없이 종료됨")
            try:
                r = json.loads(line)
            except ValueError:
                raise RuntimeError(f"합성 워커 응답 해석 실패: {line[:80]!r}")
            if not r.get("ok"):
                raise RuntimeError(f"합성 실패: {r.get('err')}")
            return True

    def close(self):
        with self._lock:
            self._kill()

    # --- 일괄 모드 (사전 렌더링) ---
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
            cmd = self._base_cmd() + ["--jobs", jobs_path]
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
