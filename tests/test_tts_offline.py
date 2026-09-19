#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTS 방송 모듈 오프라인 통합 테스트 (모델/네트워크/스피커 불필요).

NullBackend로 문구 -> 실시간 합성 / 캐시 -> 재생 큐 전 경로를 검증한다.

  python3 tests/test_tts_offline.py
"""
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts import (Announcer, COLOR_NAMES, PhraseCache, build_phrase,  # noqa: E402
                 make_backend, phrase_key)
from tts.phrase import SPEECH_NAMES, eul_reul, has_batchim           # noqa: E402
from tts.player import WavPlayer                                     # noqa: E402

WASTE_NAMES = ["쓰레기봉투", "대형가구", "가전제품", "페트", "캔", "병",
               "스티로폼", "종이박스", "의류", "플라스틱"]

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (기대 {want!r})"))
    if not ok:
        fails.append(name)


def check_true(name, cond, note=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{(' — ' + note) if note else ''}")
    if not cond:
        fails.append(name)


print("1) 조사 판정")
check("받침있음(캔)", has_batchim("캔"), True)
check("받침있음(페트병)", has_batchim("페트병"), True)      # 병 = 받침 ㅇ
check("받침없음(쓰레기봉투)", has_batchim("쓰레기봉투"), False)
check("캔 조사", eul_reul("캔"), "을")
check("페트병 조사", eul_reul("페트병"), "을")
check("쓰레기봉투 조사", eul_reul("쓰레기봉투"), "를")
check("대형 가구 조사(공백 무시)", eul_reul("대형 가구"), "를")
check("플라스틱 조사", eul_reul("플라스틱"), "을")
check("빈 문자열", has_batchim(""), False)

print("\n2) 문구 생성")
check("색상+종류", build_phrase("파란색", "페트"),
      "파란색 페트병을 무단으로 버리셨습니다. 되가져가 주시기 바랍니다. 이곳은 CCTV 녹화 중입니다.")
check("색상 없음", build_phrase(None, "캔"),
      "캔을 무단으로 버리셨습니다. 되가져가 주시기 바랍니다. 이곳은 CCTV 녹화 중입니다.")
check("장소 포함", build_phrase("검은색", "쓰레기봉투", "정문 앞").split(" 무단")[0],
      "정문 앞에 검은색 쓰레기봉투를")
check_true("모든 클래스 매핑 존재", all(w in SPEECH_NAMES for w in WASTE_NAMES),
           f"{len(SPEECH_NAMES)}개")

print("\n3) 캐시 키")
check_true("동일 입력 -> 동일 키",
           phrase_key("파란색", "페트") == phrase_key("파란색", "페트"))
check_true("다른 색상 -> 다른 키",
           phrase_key("파란색", "페트") != phrase_key("빨간색", "페트"))
check_true("장소 다르면 다른 키",
           phrase_key("파란색", "페트") != phrase_key("파란색", "페트", "정문 앞"))
keys = {phrase_key(c, w) for w in WASTE_NAMES for c in list(COLOR_NAMES) + [None]}
check("전체 조합 키 충돌 없음", len(keys), (len(COLOR_NAMES) + 1) * len(WASTE_NAMES))

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    print("\n4) NullBackend 렌더링 + 캐시")
    cache = PhraseCache(td / "cache", make_backend("null"))
    check_true("최초 조회는 미스", cache.lookup("파란색", "페트") is None)
    p = cache.ensure("파란색", "페트")
    check_true("ensure 후 wav 생성", p is not None and p.exists(), str(p))
    with wave.open(str(p), "rb") as f:
        check("mono", f.getnchannels(), 1)
        check("16-bit", f.getsampwidth(), 2)
        check_true("길이 > 0.4s", f.getnframes() / f.getframerate() > 0.4,
                   f"{f.getnframes() / f.getframerate():.1f}s")
    check_true("두 번째 조회는 히트", cache.lookup("파란색", "페트") is not None)
    check_true("index.json 기록", (td / "cache" / "index.json").exists())

    print("\n5) 전체 사전 렌더링 (120조합)")
    made, skipped = cache.prerender(COLOR_NAMES, WASTE_NAMES, verbose=False)
    n, size = cache.stats()
    check("총 wav 개수", n, (len(COLOR_NAMES) + 1) * len(WASTE_NAMES))
    check("이미 있던 1건 건너뜀", skipped, 1)
    print(f"       생성 {made}건 · 총 {size/1e6:.1f}MB")
    misses = [(c, w) for w in WASTE_NAMES for c in list(COLOR_NAMES) + [None]
              if cache.lookup(c, w) is None]
    check("모든 조합 캐시 히트", misses, [])
    _, skipped2 = cache.prerender(COLOR_NAMES, WASTE_NAMES, verbose=False)
    check("재실행 시 전부 건너뜀", skipped2, n)

    print("\n6) 재생기 (재생 명령 없이 큐 동작만)")
    pl = WavPlayer(maxsize=2, cmd=["true"]).start()   # 'true'는 즉시 성공하는 no-op
    check_true("available", pl.available())
    for _ in range(5):
        pl.submit(p)
    time.sleep(0.8)
    pl.stop()
    check_true("재생 시도됨", pl.played > 0, f"played={pl.played} dropped={pl.dropped}")

    print("\n7) Announcer(cache 모드) 쿨다운 / 중복억제")
    ann = Announcer(mode="cache", cache_dir=td / "cache", backend="null", cooldown=0.0,
                    repeat_window=60.0, verbose=False)
    ann.player.cmd = ["true"]                          # 스피커 없이 검증
    check_true("활성화", ann.enabled)
    check_true("1차 방송", ann.announce("파란색", "페트"))
    check_true("같은 조합 즉시 재방송 억제", not ann.announce("파란색", "페트"))
    check_true("다른 조합은 통과", ann.announce("빨간색", "캔"))
    check_true("색상 None도 통과", ann.announce(None, "의류"))
    check("억제 카운트", ann.n_suppressed, 1)

    ann2 = Announcer(mode="cache", cache_dir=td / "cache", backend="null", cooldown=99.0,
                     repeat_window=0.0, verbose=False)
    ann2.player.cmd = ["true"]
    check_true("쿨다운 전 1차 통과", ann2.announce("흰색", "스티로폼"))
    check_true("쿨다운 중 다른 조합도 억제", not ann2.announce("검은색", "병"))
    ann.close(drain=False)
    ann2.close(drain=False)

    print("\n8) 장애 내성 — cache 모드 캐시 미스 + 합성 금지")
    ann3 = Announcer(mode="cache", cache_dir=td / "empty", backend="null", cooldown=0.0,
                     repeat_window=0.0, allow_runtime_synth=False, verbose=False)
    ann3.player.cmd = ["true"]
    check_true("미스 시 False 반환(예외 없음)", not ann3.announce("파란색", "페트"))
    check("미스 카운트", ann3.n_missing, 1)
    ann3.close(drain=False)

    print("\n9) 장애 내성 — 비활성 Announcer")
    ann4 = Announcer(enabled=False)
    check_true("enabled=False", not ann4.enabled)
    check_true("announce가 조용히 False", not ann4.announce("파란색", "페트"))
    ann4.close()

    print("\n10) live 모드 — 이벤트 시 즉석 합성 -> 재생 -> 임시파일 정리")
    played_paths = []
    live = Announcer(mode="live", cache_dir=td / "empty", backend="null",
                     cooldown=0.0, repeat_window=0.0, verbose=False)
    live.player.cmd = ["true"]
    _orig_submit = live.player.submit

    def spy_submit(path, cleanup=False):
        played_paths.append((str(path), cleanup))
        return _orig_submit(path, cleanup=cleanup)

    live.player.submit = spy_submit
    check_true("활성화", live.enabled)
    t0 = time.monotonic()
    check_true("캐시 없이도 방송 요청 수락", live.announce("파란색", "페트"))
    check_true("announce는 즉시 반환(합성을 기다리지 않음)", time.monotonic() - t0 < 0.1)
    live.announce(None, "캔")                       # 색상 없는 문구도 합성
    t_end = time.monotonic() + 5
    while len(played_paths) < 2 and time.monotonic() < t_end:
        time.sleep(0.05)
    check("합성된 wav 2건 재생 큐 투입", len(played_paths), 2)
    check_true("임시 wav는 cleanup 플래그로 투입", all(c for _, c in played_paths))
    live.close()
    check_true("재생 후 임시 wav 삭제됨",
               all(not os.path.exists(p_) for p_, _ in played_paths))
    check("합성 실패 0건", live.n_synth_fail, 0)
    check("방송 카운트", live.n_played, 2)

    print("\n11) live 모드 — 합성 실패 시 사전 렌더링본 폴백")
    live2 = Announcer(mode="live", cache_dir=td / "cache", backend="null",
                      cooldown=0.0, repeat_window=0.0, verbose=False)
    live2.player.cmd = ["true"]
    fb_paths = []
    _orig_submit2 = live2.player.submit

    def spy_submit2(path, cleanup=False):
        fb_paths.append(str(path))
        return _orig_submit2(path, cleanup=cleanup)

    live2.player.submit = spy_submit2

    def boom(text, out):
        raise RuntimeError("워커 죽음")

    live2.backend.render_one = boom
    live2.announce("파란색", "페트")              # 5)에서 캐시된 조합
    t_end = time.monotonic() + 5
    while not fb_paths and time.monotonic() < t_end:
        time.sleep(0.05)
    check("폴백 재생 1건", len(fb_paths), 1)
    check_true("캐시 wav로 재생", bool(fb_paths) and fb_paths[0].startswith(str(td / "cache")))
    check("합성 실패 카운트", live2.n_synth_fail, 1)
    live2.close(drain=False)

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 TTS 오프라인 테스트 통과")
