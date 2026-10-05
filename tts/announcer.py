# -*- coding: utf-8 -*-
"""방송 파사드 — 감시 루프가 쓰는 유일한 진입점.

원칙: TTS 쪽에서 무슨 일이 생기든 감시 파이프라인은 절대 멈추지 않는다.
모든 공개 메서드는 예외를 삼키고 경고만 출력한다.

두 가지 동작 모드:
- live  (기본): 이벤트가 나면 문구를 그 자리에서 합성해 방송한다. 합성은 별도
                스레드 + 상주 워커 프로세스에서 하므로 감시 루프는 막히지 않는다.
                합성이 실패하면 cache/tts/에 같은 문구의 사전 렌더링본이 있을 때만 그것을 튼다.
- cache        : 사전 렌더링된 wav만 재생한다 (prerender_tts.py로 만든 것).
                캐시 미스는 allow_runtime_synth에 따라 즉석 합성(블로킹) 또는 생략.
"""
import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

from .backends import make_backend
from .cache import PhraseCache
from .phrase import build_phrase
from .player import WavPlayer

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = HERE.parent / "cache" / "tts"
MODES = ("live", "cache")


class Announcer:
    """쿨다운/중복억제 규칙에 따라 방송을 합성(live) 또는 조회(cache)해 재생한다."""

    def __init__(self, cache_dir=None, backend="supertonic", location=None,
                 cooldown=12.0, repeat_window=30.0, allow_runtime_synth=True,
                 enabled=True, verbose=True, mode="live", **backend_kw):
        if mode not in MODES:
            raise ValueError(f"mode는 {MODES} 중 하나여야 함: {mode!r}")
        self.mode = mode
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
        self.last_suppressed_reason = None   # 직전 announce()의 생략 사유 (이벤트 보고용)
        self.n_missing = 0
        self.n_synth_fail = 0
        self.last_synth_ms = None             # 하트비트 tts.last_synth_ms
        self._worker_ok = None                # 직전 예열/합성 성공 여부 (None=아직 모름)
        self.player = None
        self.cache = None
        self.backend = None
        self._synth_q = None
        self._synth_thread = None
        self._stop = threading.Event()
        if not enabled:
            return
        try:
            self.player = WavPlayer().start()
            if not self.player.available():
                print("[TTS 경고] 재생 명령(paplay/aplay/play)을 찾지 못했습니다. 방송 비활성화")
                self.enabled = False
                return
            if mode == "live":
                self.backend = make_backend(backend, **backend_kw)
                # 캐시는 합성 실패 시 폴백 조회 전용 (백엔드 없음 -> ensure 불가)
                self.cache = PhraseCache(cache_dir or DEFAULT_CACHE_DIR, None, location)
                self._synth_q = queue.Queue(maxsize=2)
                self._synth_thread = threading.Thread(
                    target=self._synth_loop, name="tts-synth", daemon=True)
                self._synth_thread.start()
                self._synth_q.put_nowait(None)          # 첫 작업 = 워커 예열
                print(f"[TTS] 방송 활성 (실시간 합성, 백엔드={self.backend.name}) "
                      f"· 재생={' '.join(self.player.cmd)}"
                      + (f" · 위치='{location}'" if location else ""))
            else:
                be = make_backend(backend, **backend_kw) if allow_runtime_synth else None
                self.cache = PhraseCache(cache_dir or DEFAULT_CACHE_DIR, be, location)
                n, size = self.cache.stats()
                print(f"[TTS] 방송 활성 (캐시 재생) · 캐시 {n}개({size/1e6:.1f}MB) "
                      f"· 재생={' '.join(self.player.cmd)}"
                      + (f" · 위치='{location}'" if location else ""))
        except Exception as e:                                   # noqa: BLE001
            print(f"[TTS 경고] 초기화 실패 ({type(e).__name__}: {e}) -> 방송 비활성화")
            self.enabled = False

    # ------------------------------------------------------------------ 공통
    def set_location(self, location):
        """방송 장소명 교체 (대시보드 설치 위치 변경 -> site_config.py). 다음 방송부터 적용.
        cache 모드는 장소별 캐시라 새 장소명은 첫 방송 때 즉석 합성된다."""
        location = location or None
        if location == self.location:
            return
        self.location = location
        if self.cache is not None:
            self.cache.location = location
        if self.verbose:
            print(f"[TTS] 장소명 변경 -> '{location or '(없음)'}'", flush=True)

    def _suppressed(self, combo, now):
        if now - self.last_play < self.cooldown:
            return "쿨다운"
        prev = self.last_by_combo.get(combo)
        if prev is not None and now - prev < self.repeat_window:
            return "중복"
        return None

    def announce(self, color, waste_name, audio_out=None, on_audio=None):
        """방송 요청. 실제로 큐에 넣었으면 True. 감시 루프를 막지 않는다.

        audio_out(wav 경로)을 주면 방송 음성을 그 경로에 남기고(서버 전송용 — 재생한 파일
        그대로, 변환 없음), 재생이 끝나면 on_audio(audio_out)를 재생 스레드에서 호출한다.
        생략·합성 실패로 방송하지 않으면 호출되지 않는다.
        """
        if not self.enabled:
            return False
        try:
            now = time.monotonic()
            combo = (color, waste_name)
            why = self._suppressed(combo, now)
            self.last_suppressed_reason = why
            if why:
                self.n_suppressed += 1
                if self.verbose:
                    print(f"[TTS] 방송 생략({why}): {build_phrase(color, waste_name)[:30]}…")
                return False
            audio = None
            if audio_out:
                # 이벤트 번호는 실행마다 1부터라 지난 실행의 같은 이름 파일이 남아 있을 수 있다.
                # 합성이 실패하면 그 옛 파일이 전송되므로 미리 지운다.
                try:
                    os.unlink(audio_out)
                except OSError:
                    pass
                audio = (str(audio_out), on_audio)
            ok = (self._announce_live(color, waste_name, audio) if self.mode == "live"
                  else self._announce_cached(color, waste_name, audio))
            if not ok:
                self.last_suppressed_reason = "합성실패"
            if ok:
                self.last_play = now
                self.last_by_combo[combo] = now
                self.n_played += 1
            return ok
        except Exception as e:                                   # noqa: BLE001
            self.last_suppressed_reason = "합성실패"
            print(f"[TTS 경고] 방송 실패 ({type(e).__name__}: {e})")
            return False

    # ------------------------------------------------------------------ live
    def _announce_live(self, color, waste_name, audio=None):
        """합성 작업을 큐에 넣고 즉시 반환. 큐가 차 있으면 오래된 요청을 버린다."""
        item = (color, waste_name, audio)
        try:
            self._synth_q.put_nowait(item)
        except queue.Full:
            try:
                self._synth_q.get_nowait()
            except queue.Empty:
                pass
            self._synth_q.put_nowait(item)
        if self.verbose:
            print(f"[TTS] 합성 요청: {build_phrase(color, waste_name, self.location)}")
        return True

    def _synth_loop(self):
        """합성 스레드. None = 예열, (color, waste, audio) = 합성 후 재생."""
        while not self._stop.is_set():
            try:
                item = self._synth_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                self._warmup()
                continue
            color, waste_name, audio = item
            text = build_phrase(color, waste_name, self.location)
            if audio:                               # 서버 전송용 경로에 바로 합성 -> 그 파일을 재생
                out = audio[0]
            else:
                fd, out = tempfile.mkstemp(suffix=".wav", prefix="tts_live_")
                os.close(fd)
            t0 = time.monotonic()
            try:
                self.backend.render_one(text, out)
                dt = time.monotonic() - t0
                self.last_synth_ms = round(dt * 1000)
                self._worker_ok = True
                if self._play(out, audio, cleanup=not audio) and self.verbose:
                    print(f"[TTS] 방송 (합성 {dt:.1f}s): {text}")
            except Exception as e:                               # noqa: BLE001
                self.n_synth_fail += 1
                self._worker_ok = False
                try:
                    os.unlink(out)
                except OSError:
                    pass
                print(f"[TTS 경고] 실시간 합성 실패 ({type(e).__name__}: {e})")
                self._fallback_cached(color, waste_name, audio)

    def _play(self, wav, audio=None, cleanup=False):
        """wav 재생 요청. audio=(서버용 경로, on_audio)면 재생 뒤 on_audio(경로).
        사전 렌더링 wav는 캐시 원본을 지키기 위해 서버용 경로로 복사해서 튼다
        (업로드 확인 뒤 로컬 파일이 지워지므로)."""
        if not audio:
            return self.player.submit(wav, cleanup=cleanup)
        path, on_audio = audio
        if os.path.abspath(str(wav)) != os.path.abspath(path):
            try:
                shutil.copyfile(wav, path)
            except OSError as e:
                print(f"[TTS 경고] 방송 음성 복사 실패 ({e}) -> 서버 첨부 없이 방송")
                return self.player.submit(wav, cleanup=cleanup)
        return self.player.submit(path, on_done=on_audio)

    def _warmup(self):
        t0 = time.monotonic()
        try:
            self.backend.warmup()
            self._worker_ok = True
            if self.verbose:
                print(f"[TTS] 합성 워커 준비 완료 ({time.monotonic() - t0:.1f}s)")
        except Exception as e:                                   # noqa: BLE001
            self._worker_ok = False
            print(f"[TTS 경고] 합성 워커 예열 실패 ({type(e).__name__}: {e}) "
                  f"-> 첫 방송 때 다시 시도")

    def _fallback_cached(self, color, waste_name, audio=None):
        wav = self.cache.lookup(color, waste_name) if self.cache else None
        if wav is None:
            self.n_missing += 1
            return
        if self._play(wav, audio):
            print(f"[TTS] 사전 렌더링본으로 대체 방송: {wav.name}")

    # ----------------------------------------------------------------- cache
    def _announce_cached(self, color, waste_name, audio=None):
        wav = self.cache.lookup(color, waste_name)
        if wav is None:
            if not self.allow_runtime_synth:
                self.n_missing += 1
                print(f"[TTS 경고] 캐시 미스 & 실시간 합성 비활성: "
                      f"{color or '무색상'}/{waste_name}")
                return False
            # 캐시 미스는 드물어야 정상. 합성이 수 초 걸릴 수 있음을 알린다.
            print(f"[TTS] 캐시 미스 -> 즉석 합성: {color or '무색상'}/{waste_name}")
            wav = self.cache.ensure(color, waste_name)
            if wav is None:
                self.n_missing += 1
                return False
        if not self._play(wav, audio):
            return False
        if self.verbose:
            print(f"[TTS] 방송: {build_phrase(color, waste_name, self.location)}")
        return True

    def status(self):
        """하트비트용 tts 블록 (jetson_data.md §3)."""
        if not self.enabled:
            worker = "off"
        elif self.mode != "live":
            worker = "cache"
        else:
            worker = {True: "ready", False: "down", None: "starting"}[self._worker_ok]
        return {"worker": worker, "played_total": self.n_played,
                "suppressed_total": self.n_suppressed,
                "synth_fail_total": self.n_synth_fail, "last_synth_ms": self.last_synth_ms}

    # ----------------------------------------------------------------- 종료
    def close(self, drain=True, timeout=30):
        """drain=True면 대기 중인 합성·재생을 최대 timeout초까지 끝낸 뒤 종료."""
        if self._synth_thread is not None:
            try:
                if drain:
                    deadline = time.monotonic() + timeout
                    while not self._synth_q.empty() and time.monotonic() < deadline:
                        time.sleep(0.2)
                self._stop.set()
                self._synth_thread.join(timeout=timeout)
            except Exception:                                    # noqa: BLE001
                pass
            self._synth_thread = None
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:                                    # noqa: BLE001
                pass
        if self.player is not None:
            try:
                self.player.stop(drain=drain)
            except Exception:                                    # noqa: BLE001
                pass
        if self.enabled:
            print(f"[TTS] 방송 {self.n_played}건 · 억제 {self.n_suppressed}건"
                  + (f" · 합성실패 {self.n_synth_fail}건" if self.n_synth_fail else "")
                  + (f" · 미방송 {self.n_missing}건" if self.n_missing else ""))
