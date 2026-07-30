# -*- coding: utf-8 -*-
"""방송 파사드 — 감시 루프가 쓰는 유일한 진입점.

원칙: TTS 쪽에서 무슨 일이 생기든 감시 파이프라인은 절대 멈추지 않는다.
모든 공개 메서드는 예외를 삼키고 경고만 출력한다.
"""
import time
from pathlib import Path

from .backends import make_backend
from .cache import PhraseCache
from .phrase import build_phrase
from .player import WavPlayer

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = HERE.parent / "cache" / "tts"


class Announcer:
    """캐시된 방송 wav를 쿨다운/중복억제 규칙에 따라 재생한다."""

    def __init__(self, cache_dir=None, backend="supertonic", location=None,
                 cooldown=12.0, repeat_window=30.0, allow_runtime_synth=True,
                 enabled=True, verbose=True, **backend_kw):
        self.enabled = enabled
        self.location = location
        self.cooldown = float(cooldown)
        self.repeat_window = float(repeat_window)
        self.allow_runtime_synth = allow_runtime_synth
        self.verbose = verbose
        self.last_play = 0.0
        self.last_by_combo = {}
        self.n_played = 0
        self.n_suppressed = 0
        self.n_missing = 0
        self.player = None
        self.cache = None
        if not enabled:
            return
        try:
            be = make_backend(backend, **backend_kw) if allow_runtime_synth else None
            self.cache = PhraseCache(cache_dir or DEFAULT_CACHE_DIR, be, location)
            self.player = WavPlayer().start()
            if not self.player.available():
                print("[TTS 경고] 재생 명령(paplay/aplay/play)을 찾지 못했습니다. 방송 비활성화")
                self.enabled = False
            else:
                n, size = self.cache.stats()
                print(f"[TTS] 방송 활성 · 캐시 {n}개({size/1e6:.1f}MB) "
                      f"· 재생={' '.join(self.player.cmd)}"
                      + (f" · 위치='{location}'" if location else ""))
        except Exception as e:                                   # noqa: BLE001
            print(f"[TTS 경고] 초기화 실패 ({type(e).__name__}: {e}) -> 방송 비활성화")
            self.enabled = False

    def _suppressed(self, combo, now):
        if now - self.last_play < self.cooldown:
            return "쿨다운"
        prev = self.last_by_combo.get(combo)
        if prev is not None and now - prev < self.repeat_window:
            return "중복"
        return None

    def announce(self, color, waste_name):
        """방송 요청. 실제로 큐에 넣었으면 True."""
        if not self.enabled or self.cache is None:
            return False
        try:
            now = time.monotonic()
            combo = (color, waste_name)
            why = self._suppressed(combo, now)
            if why:
                self.n_suppressed += 1
                if self.verbose:
                    print(f"[TTS] 방송 생략({why}): {build_phrase(color, waste_name)[:30]}…")
                return False

            wav = self.cache.lookup(color, waste_name)
            if wav is None:
                if not self.allow_runtime_synth:
                    self.n_missing += 1
                    print(f"[TTS 경고] 캐시 미스 & 실시간 합성 비활성: "
                          f"{color or '무색상'}/{waste_name}")
                    return False
                # 캐시 미스는 드물어야 정상. 합성이 수 초 걸릴 수 있음을 알린다.
                print(f"[TTS] 캐시 미스 -> 실시간 합성: {color or '무색상'}/{waste_name}")
                wav = self.cache.ensure(color, waste_name)
                if wav is None:
                    self.n_missing += 1
                    return False

            if not self.player.submit(wav):
                return False
            self.last_play = now
            self.last_by_combo[combo] = now
            self.n_played += 1
            if self.verbose:
                print(f"[TTS] 방송: {build_phrase(color, waste_name, self.location)}")
            return True
        except Exception as e:                                   # noqa: BLE001
            print(f"[TTS 경고] 방송 실패 ({type(e).__name__}: {e})")
            return False

    def close(self, drain=True):
        if self.player is not None:
            try:
                self.player.stop(drain=drain)
            except Exception:                                    # noqa: BLE001
                pass
        if self.enabled:
            print(f"[TTS] 방송 {self.n_played}건 · 억제 {self.n_suppressed}건"
                  + (f" · 실패 {self.n_missing}건" if self.n_missing else ""))
