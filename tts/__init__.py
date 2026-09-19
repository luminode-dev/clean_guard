# -*- coding: utf-8 -*-
"""무단투기 경고 방송 모듈.

감시 루프(dump_monitor_jetson.py)에서 쓰는 것은 세 가지뿐:

    from tts import Announcer, dominant_color_name, night_mode

    ann = Announcer(location="○○동 수거함 앞")     # 기본 mode="live": 즉석 합성
    if night_mode(frame) is None:                 # 야간 적외선/저조도면 색상 생략
        color = dominant_color_name(frame, box)   # '파란색' 등, 실패 시 None
    ann.announce(color, "페트")                    # 합성 스레드에 넘기고 즉시 반환
    ...
    ann.close()

음성 합성(supertonic/onnxruntime)은 전용 venv의 상주 서브프로세스에서만 일어나므로
감시 프로세스에는 새 런타임 의존성이 없다. 사전 렌더링(cache 모드)은 prerender_tts.py 참고.
"""
from .announcer import DEFAULT_CACHE_DIR, Announcer
from .backends import NullBackend, SupertonicBackend, make_backend
from .cache import PhraseCache
from .color_naming import COLOR_NAMES, dominant_color_name, night_mode
from .phrase import SPEECH_NAMES, build_phrase, phrase_key
from .player import WavPlayer

__all__ = ["Announcer", "DEFAULT_CACHE_DIR", "PhraseCache", "WavPlayer",
           "COLOR_NAMES", "dominant_color_name", "night_mode",
           "SPEECH_NAMES", "build_phrase", "phrase_key",
           "make_backend", "NullBackend", "SupertonicBackend"]
