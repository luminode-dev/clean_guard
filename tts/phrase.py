# -*- coding: utf-8 -*-
"""방송 문구 생성 + 캐시 키.

dump_monitor_jetson.py의 WASTE_NAMES(모델 클래스명)를 방송에 자연스러운
표현으로 바꾸고, 색상/조사를 붙여 한 문장을 만든다.
"""
import hashlib

# 모델 클래스명 -> 방송용 표현 (dump_monitor_jetson.py:43 WASTE_NAMES 기준)
SPEECH_NAMES = {
    "쓰레기봉투": "쓰레기봉투",
    "대형가구": "대형 가구",
    "가전제품": "가전제품",
    "페트": "페트병",
    "캔": "캔",
    "병": "병",
    "스티로폼": "스티로폼",
    "종이박스": "종이 상자",
    "의류": "의류",
    "플라스틱": "플라스틱",
}

WARNING = "되가져가 주시기 바랍니다. 이곳은 CCTV 녹화 중입니다."


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지. 한글이 아니면 False."""
    for ch in reversed(word.strip()):
        if ch.isspace():
            continue
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28 != 0
        return False
    return False


def eul_reul(word: str) -> str:
    return "을" if has_batchim(word) else "를"


def speech_name(waste_name: str) -> str:
    return SPEECH_NAMES.get(waste_name, waste_name)


def build_phrase(color, waste_name, location=None) -> str:
    """예: '○○동 3번 수거함 앞에 파란색 페트병을 무단으로 버리셨습니다. ...'"""
    item = speech_name(waste_name)
    subject = f"{color} {item}" if color else item
    prefix = f"{location}에 " if location else ""
    return (f"{prefix}{subject}{eul_reul(item)} 무단으로 버리셨습니다. {WARNING}")


def phrase_key(color, waste_name, location=None) -> str:
    """문구 내용에 대한 안정적인 파일명(확장자 제외)."""
    text = build_phrase(color, waste_name, location)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
