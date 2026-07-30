#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""방송 문구 사전 렌더링 CLI.

이벤트 발생 시 지연을 0으로 만들기 위해 (색상 11 + 색상없음) × 쓰레기 10종 = 120개
문장을 미리 합성해 cache/tts/ 에 wav로 저장한다.

TTS 전용 venv의 python으로 실행하는 것을 권장한다:
  .venv-tts/bin/python prerender_tts.py --all

usage:
  prerender_tts.py --check                 # 1문장 스모크 (기본 재생까지)
  prerender_tts.py --all                   # 120문장 전체
  prerender_tts.py --all --force           # 캐시 무시하고 재생성
  prerender_tts.py --all --location "정문 수거함 앞"
  prerender_tts.py --list                  # 만들 문구 목록만 출력
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tts import COLOR_NAMES, PhraseCache, build_phrase, make_backend   # noqa: E402
from tts.announcer import DEFAULT_CACHE_DIR                            # noqa: E402
from tts.backends import wav_duration                                  # noqa: E402
from tts.player import detect_player_cmd                               # noqa: E402

# dump_monitor_jetson.py:43 WASTE_NAMES와 동일해야 한다 (모델 클래스 순서)
WASTE_NAMES = ["쓰레기봉투", "대형가구", "가전제품", "페트", "캔", "병",
               "스티로폼", "종이박스", "의류", "플라스틱"]


def play(path):
    cmd = detect_player_cmd()
    if cmd is None:
        print("[경고] 재생 명령(paplay/aplay/play)이 없어 재생을 건너뜁니다")
        return
    subprocess.run(cmd + [str(path)], check=False)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="1문장 스모크 테스트")
    mode.add_argument("--all", action="store_true", help="전체 조합 렌더링")
    mode.add_argument("--list", action="store_true", help="문구 목록만 출력")
    ap.add_argument("--cache", default=None, help=f"캐시 폴더 (기본 {DEFAULT_CACHE_DIR})")
    ap.add_argument("--backend", default="supertonic", choices=["supertonic", "null"])
    ap.add_argument("--model", default="supertonic-2")
    ap.add_argument("--model-dir", default=None, help="모델 폴더 (기본 ~/.cache/supertonic*)")
    ap.add_argument("--voice", default="M4", help="M1~M5 / F1~F5")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--steps", type=int, default=8, help="디노이징 스텝")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--location", default=None, help="방송에 넣을 장소명")
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    ap.add_argument("--play", action="store_true", help="생성 후 샘플 1개 재생")
    ap.add_argument("--no-play", action="store_true", help="--check에서 재생 생략")
    args = ap.parse_args()

    cache_dir = Path(args.cache) if args.cache else DEFAULT_CACHE_DIR

    if args.list:
        n = 0
        for waste in WASTE_NAMES:
            for color in list(COLOR_NAMES) + [None]:
                n += 1
                print(f"{n:3d}. {build_phrase(color, waste, args.location)}")
        print(f"\n총 {n}개 -> {cache_dir}")
        return 0

    backend_kw = {}
    if args.backend == "supertonic":
        backend_kw = dict(model=args.model, model_dir=args.model_dir, voice=args.voice,
                          lang=args.lang, steps=args.steps, speed=args.speed,
                          threads=args.threads)
    backend = make_backend(args.backend, **backend_kw)
    cache = PhraseCache(cache_dir, backend, args.location)

    if args.check:
        color, waste = "파란색", "페트"
        text = build_phrase(color, waste, args.location)
        print(f"스모크: {text}")
        out = cache.path_for(color, waste)
        if args.force and out.exists():
            out.unlink()
        got = cache.ensure(color, waste, verbose=True)
        if got is None:
            print("[실패] 합성 결과 없음")
            return 1
        dur = wav_duration(got)
        print(f"[성공] {got} ({got.stat().st_size/1000:.0f}KB, {dur:.1f}s)")
        if dur < 0.3:
            print("[경고] 길이가 지나치게 짧습니다 — 합성이 정상인지 확인하세요")
        if not args.no_play:
            play(got)
        return 0

    made, skipped = cache.prerender(COLOR_NAMES, WASTE_NAMES, force=args.force)
    n, size = cache.stats()
    print(f"\n생성 {made}건 · 건너뜀 {skipped}건 · 캐시 총 {n}개 ({size/1e6:.1f}MB) -> {cache_dir}")
    expected = (len(COLOR_NAMES) + 1) * len(WASTE_NAMES)
    if made + skipped < expected:
        print(f"[경고] 기대 {expected}개 중 {made + skipped}개만 처리됨")
    if args.play:
        sample = cache.lookup("파란색", "페트") or next(iter(cache_dir.glob("*.wav")), None)
        if sample:
            play(sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
