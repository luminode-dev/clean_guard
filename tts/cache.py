# -*- coding: utf-8 -*-
"""렌더링된 방송 wav 캐시.

이벤트 시점의 지연을 0으로 만들기 위해 (색상 × 쓰레기종류) 조합을 미리 렌더링해 둔다.
런타임에는 조회만 하고, 미스일 때만(드묾) 백엔드를 호출한다.
"""
import json
from pathlib import Path

from .phrase import build_phrase, phrase_key

INDEX_NAME = "index.json"


class PhraseCache:
    def __init__(self, cache_dir, backend=None, location=None):
        self.dir = Path(cache_dir)
        self.backend = backend
        self.location = location
        self._index = {}
        self._load_index()

    # --- index (디버깅/점검용 key -> text 매핑) ---
    def _index_path(self):
        return self.dir / INDEX_NAME

    def _load_index(self):
        p = self._index_path()
        if p.exists():
            try:
                self._index = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                self._index = {}

    def _save_index(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path().write_text(
            json.dumps(self._index, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- 조회 / 생성 ---
    def path_for(self, color, waste_name):
        key = phrase_key(color, waste_name, self.location)
        return self.dir / f"{key}.wav"

    def lookup(self, color, waste_name):
        """캐시 히트면 Path, 아니면 None."""
        p = self.path_for(color, waste_name)
        return p if p.exists() and p.stat().st_size > 44 else None

    def ensure(self, color, waste_name, verbose=False):
        """캐시에 없으면 합성해서 채우고 Path 반환. 실패 시 None."""
        hit = self.lookup(color, waste_name)
        if hit is not None:
            return hit
        if self.backend is None:
            return None
        text = build_phrase(color, waste_name, self.location)
        out = self.path_for(color, waste_name)
        if self.backend.render([(text, out)], verbose=verbose) < 1:
            return None
        self._index[out.stem] = text
        self._save_index()
        return out if out.exists() else None

    def prerender(self, colors, waste_names, force=False, verbose=True):
        """(색상 + 색상없음) × 쓰레기종류 전체를 렌더링. (생성수, 건너뜀수) 반환."""
        combos = [(c, w) for w in waste_names for c in list(colors) + [None]]
        jobs, made = [], {}
        skipped = 0
        for color, waste in combos:
            out = self.path_for(color, waste)
            if not force and self.lookup(color, waste) is not None:
                skipped += 1
                made[out.stem] = build_phrase(color, waste, self.location)
                continue
            text = build_phrase(color, waste, self.location)
            jobs.append((text, out))
            made[out.stem] = text
        n = 0
        if jobs:
            if self.backend is None:
                raise RuntimeError("백엔드가 없어 사전 렌더링을 할 수 없습니다")
            if verbose:
                print(f"사전 렌더링 {len(jobs)}건 (건너뜀 {skipped}건) -> {self.dir}")
            n = self.backend.render(jobs, verbose=verbose)
        self._index.update(made)
        self._save_index()
        return n, skipped

    def stats(self):
        wavs = list(self.dir.glob("*.wav")) if self.dir.exists() else []
        return len(wavs), sum(p.stat().st_size for p in wavs)
