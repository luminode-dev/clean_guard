# -*- coding: utf-8 -*-
"""무단투기 경고 방송 모듈.

감시 루프(dump_monitor_jetson.py)에서 쓰는 것은 두 가지뿐:

    from tts import Announcer, dominant_color_name

    ann = Announcer(location="○○동 수거함 앞")
    color = dominant_color_name(frame, box)     # '파란색' 등, 실패 시 None
    ann.announce(color, "페트")                 # 캐시된 wav를 논블로킹 재생
    ...
    ann.close()

음성 합성(supertonic/onnxruntime)은 전용 venv의 서브프로세스에서만 일어나므로
감시 프로세스에는 새 런타임 의존성이 없다. 캐시 생성은 prerender_tts.py 참고.
"""
from .announcer import DEFAULT_CACHE_DIR, Announcer
from .backends import NullBackend, SupertonicBackend, make_backend
from .cache import PhraseCache
from .color_naming import COLOR_NAMES, dominant_color_name
from .phrase import SPEECH_NAMES, build_phrase, phrase_key
from .player import WavPlayer

__all__ = ["Announcer", "DEFAULT_CACHE_DIR", "PhraseCache", "WavPlayer",
           "COLOR_NAMES", "dominant_color_name",
           "SPEECH_NAMES", "build_phrase", "phrase_key",
           "make_backend", "NullBackend", "SupertonicBackend"]
