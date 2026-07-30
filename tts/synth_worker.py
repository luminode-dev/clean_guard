# -*- coding: utf-8 -*-
"""Supertonic 합성 워커 — TTS 전용 venv의 python으로 실행되는 독립 스크립트.

감시 프로세스(dump_monitor_jetson.py)에 onnxruntime/supertonic 의존성을
들이지 않기 위해, 합성은 항상 이 스크립트를 서브프로세스로 띄워 수행한다.
표준 라이브러리 + supertonic 패키지만 import 한다.

usage:
  python3 synth_worker.py --jobs jobs.json [--model supertonic-2] [--voice M4]
                          [--lang ko] [--steps 8] [--speed 1.0] [--model-dir DIR]

jobs.json: [{"text": "...", "out": "/abs/path.wav"}, ...]
stdout: 작업당 JSON 한 줄 {"out":..., "ok":bool, "sec":float, "err":str|null}
"""
import argparse
import json
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="작업 목록 JSON 경로")
    ap.add_argument("--model", default="supertonic-2")
    ap.add_argument("--model-dir", default=None, help="모델 디렉터리 (미지정 시 기본 캐시)")
    ap.add_argument("--voice", default="M4", help="M1~M5 / F1~F5")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--steps", type=int, default=8, help="디노이징 스텝(클수록 고품질/느림)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op 스레드")
    ap.add_argument("--no-download", action="store_true", help="모델 자동 다운로드 금지")
    args = ap.parse_args()

    with open(args.jobs, encoding="utf-8") as f:
        jobs = json.load(f)

    from supertonic import TTS

    tts = TTS(model=args.model, model_dir=args.model_dir,
              auto_download=not args.no_download,
              intra_op_num_threads=args.threads)
    style = tts.get_voice_style(args.voice)

    failed = 0
    for job in jobs:
        out = job["out"]
        try:
            wav, dur = tts.synthesize(job["text"], voice_style=style,
                                      total_steps=args.steps, speed=args.speed,
                                      lang=args.lang)
            write_wav_int16(out, wav, tts.sample_rate)
            sec = float(dur[0]) if hasattr(dur, "__len__") else float(dur)
            print(json.dumps({"out": out, "ok": True, "sec": round(sec, 2),
                              "err": None}, ensure_ascii=False), flush=True)
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(json.dumps({"out": out, "ok": False, "sec": 0.0,
                              "err": f"{type(e).__name__}: {e}"},
                             ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
