# -*- coding: utf-8 -*-
"""Supertonic 합성 워커 — TTS 전용 venv의 python으로 실행되는 독립 스크립트.

감시 프로세스(dump_monitor_jetson.py)에 onnxruntime/supertonic 의존성을
들이지 않기 위해, 합성은 항상 이 스크립트를 서브프로세스로 띄워 수행한다.
표준 라이브러리 + supertonic 패키지만 import 한다.

usage:
  # 일괄 모드 (사전 렌더링): 작업 목록을 처리하고 종료
  python3 synth_worker.py --jobs jobs.json [--model supertonic-2] [--voice M4]
                          [--lang ko] [--steps 8] [--speed 1.0] [--model-dir DIR]
  # 상주 모드 (실시간 방송): 모델을 한 번 올리고 stdin으로 작업을 계속 받는다
  python3 synth_worker.py --serve [같은 옵션]

jobs.json: [{"text": "...", "out": "/abs/path.wav"}, ...]
--serve   : 준비되면 {"ready": true, "load_sec": float} 한 줄을 먼저 출력하고,
            이후 stdin 한 줄({"text":..., "out":...})마다 결과 한 줄을 돌려준다.
            빈 줄 또는 EOF를 받으면 종료.
stdout: 작업당 JSON 한 줄 {"out":..., "ok":bool, "sec":float, "err":str|null}
"""
import argparse
import json
import sys
import time
import wave


def write_wav_int16(path, wav, sample_rate, peak=0.89):
    """float32 (-1..1) 또는 int16 배열을 16-bit mono PCM wav로 저장.

    peak > 0이면 피크 정규화한다. Supertonic 원본은 -11dB 정도로 야외 스피커에는
    작으므로, 방송 음량을 확보하기 위해 헤드룸만 남기고 끌어올린다.
    """
    import numpy as np

    a = np.asarray(wav).squeeze()
    if a.ndim > 1:                      # (1, T) 또는 다채널 -> mono
        a = a.reshape(a.shape[-1]) if a.shape[0] == 1 else a.mean(axis=0)
    a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32)
    if peak > 0:
        mx = float(np.abs(a).max())
        if mx > 1e-4:
            a *= peak / mx
    a = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(a.tobytes())


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def synth_one(tts, style, args, job):
    """작업 하나를 합성해 결과 dict를 반환한다 (예외는 err 필드로)."""
    out = job["out"]
    try:
        wav, dur = tts.synthesize(job["text"], voice_style=style,
                                  total_steps=args.steps, speed=args.speed,
                                  lang=args.lang)
        write_wav_int16(out, wav, tts.sample_rate)
        sec = float(dur[0]) if hasattr(dur, "__len__") else float(dur)
        return {"out": out, "ok": True, "sec": round(sec, 2), "err": None}
    except Exception as e:                                       # noqa: BLE001
        return {"out": out, "ok": False, "sec": 0.0, "err": f"{type(e).__name__}: {e}"}


def serve(tts, style, args):
    """stdin 한 줄당 작업 하나. 모델은 이미 올라와 있으므로 합성 시간만 든다."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        try:
            job = json.loads(line)
        except ValueError as e:
            emit({"out": None, "ok": False, "sec": 0.0, "err": f"잘못된 요청: {e}"})
            continue
        emit(synth_one(tts, style, args, job))
    return 0


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--jobs", help="작업 목록 JSON 경로 (일괄 모드)")
    mode.add_argument("--serve", action="store_true", help="상주 모드 (stdin으로 작업 수신)")
    ap.add_argument("--model", default="supertonic-2")
    ap.add_argument("--model-dir", default=None, help="모델 디렉터리 (미지정 시 기본 캐시)")
    ap.add_argument("--voice", default="M4", help="M1~M5 / F1~F5")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--steps", type=int, default=8, help="디노이징 스텝(클수록 고품질/느림)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op 스레드")
    ap.add_argument("--no-download", action="store_true", help="모델 자동 다운로드 금지")
    args = ap.parse_args()

    t0 = time.monotonic()
    try:
        from supertonic import TTS

        tts = TTS(model=args.model, model_dir=args.model_dir,
                  auto_download=not args.no_download,
                  intra_op_num_threads=args.threads)
        style = tts.get_voice_style(args.voice)
    except Exception as e:                                       # noqa: BLE001
        if args.serve:
            emit({"ready": False, "err": f"{type(e).__name__}: {e}"})
            return 1
        raise

    if args.serve:
        emit({"ready": True, "load_sec": round(time.monotonic() - t0, 1)})
        return serve(tts, style, args)

    with open(args.jobs, encoding="utf-8") as f:
        jobs = json.load(f)
    failed = 0
    for job in jobs:
        r = synth_one(tts, style, args, job)
        failed += 0 if r["ok"] else 1
        emit(r)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
