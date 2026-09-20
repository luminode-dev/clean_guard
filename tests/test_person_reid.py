#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""사람 재식별 갤러리 로직 단위 테스트 (모델 불필요 — 가짜 인코더 사용).

  python3 tests/test_person_reid.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from person_reid import (MAX_PENDING_FRAMES, MIN_BOX_H, PROBATION,  # noqa: E402
                         PersonReID)

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


class FakeEncoder:
    """박스 폭(w)을 '사람 정체성'으로 쓴다: 같은 폭 = 같은 사람. 기준 벡터 + 약간의 노이즈."""

    def __init__(self, noise=0.05, seed=0):
        self.rng = np.random.default_rng(seed)
        self.base = {}
        self.noise = noise
        self.calls = 0

    def vec(self, ident):
        if ident not in self.base:
            v = self.rng.normal(size=512)
            self.base[ident] = v / np.linalg.norm(v)
        return self.base[ident]

    def __call__(self, frame, dets):
        self.calls += len(dets)
        out = []
        for cx, cy, w, h in dets:
            v = self.vec(int(round(w))) + self.rng.normal(size=512) * self.noise
            out.append(v / np.linalg.norm(v))
        return out


def box(ident_w, x=100, y=100, h=120):
    return [x, y, x + ident_w, y + h]


FRAME = np.zeros((480, 640, 3), np.uint8)

print("1) 같은 사람이 나갔다 돌아오면 이전 pid로 병합")
enc = FakeEncoder()
r = PersonReID(enc, verbose=False)
merges_all = []
for f in range(1, 41):                                   # 사람 A(폭 50) raw 1 로 40프레임
    out, m = r.update(FRAME, [(1, box(50, x=100 + f))], f, now=f * 0.1)
    merges_all += m
check("초기 canonical pid", out[0][0], 1)
check("병합 없음", merges_all, [])
for f in range(41, 100):                                 # 사라짐 (59프레임)
    r.update(FRAME, [], f, now=f * 0.1)
seen = []
for f in range(100, 100 + PROBATION + 2):                # raw 7 로 재등장
    out, m = r.update(FRAME, [(7, box(50, x=300))], f, now=f * 0.1)
    seen.append(out[0][0]); merges_all += m
check("판정 전(probation)엔 새 pid 그대로", seen[0], 7)
check("probation 후 이전 pid 반환", seen[-1], 1)
check("merges 목록", merges_all, [(7, 1)])
check("재식별 카운트", r.n_merged, 1)

print("\n2) 다른 사람은 병합하지 않음")
enc = FakeEncoder()
r = PersonReID(enc, verbose=False)
for f in range(1, 41):
    r.update(FRAME, [(1, box(50))], f, now=f * 0.1)
for f in range(41, 100):
    r.update(FRAME, [], f, now=f * 0.1)
merges = []
for f in range(100, 100 + PROBATION + 2):
    out, m = r.update(FRAME, [(2, box(60))], f, now=f * 0.1)   # 폭 60 = 다른 사람
    merges += m
check("병합 없음", merges, [])
check("새 pid 유지", out[0][0], 2)

print("\n3) 갤러리에 비슷한 후보가 둘이면(margin 미달) 병합 보류")
enc = FakeEncoder(noise=0.0)
r = PersonReID(enc, verbose=False)
enc.base[50] = enc.vec(50)
twin = enc.base[50] + np.random.default_rng(1).normal(size=512) * 0.004  # 거의 같은 외형(cos≈0.99)의 다른 사람
enc.base[51] = twin / np.linalg.norm(twin)
for f in range(1, 41):
    r.update(FRAME, [(1, box(50, x=50)), (2, box(51, x=400))], f, now=f * 0.1)
for f in range(41, 100):
    r.update(FRAME, [], f, now=f * 0.1)
merges = []
for f in range(100, 100 + PROBATION + 2):
    _, m = r.update(FRAME, [(9, box(50, x=200))], f, now=f * 0.1)
    merges += m
check("1위-2위 차이 부족 -> 병합 안 함", merges, [])

print("\n4) TTL이 지난 갤러리는 버림")
enc = FakeEncoder()
r = PersonReID(enc, ttl_s=5.0, verbose=False)
for f in range(1, 41):
    r.update(FRAME, [(1, box(50))], f, now=f * 0.1)
for f in range(41, 200):
    r.update(FRAME, [], f, now=f * 0.1)                     # 마지막 관측 4.0s -> 19.9s: 15.9s 경과
check_true("갤러리에서 제거됨", 1 not in r.tracks)
merges = []
for f in range(200, 200 + PROBATION + 2):
    _, m = r.update(FRAME, [(3, box(50))], f, now=f * 0.1)
    merges += m
check("TTL 경과 후 재등장은 새 사람", merges, [])

print("\n5) 아직 보이는 사람과는 병합하지 않음 (동시 존재 = 다른 사람)")
enc = FakeEncoder()
r = PersonReID(enc, verbose=False)
for f in range(1, 41):
    r.update(FRAME, [(1, box(50, x=50))], f, now=f * 0.1)
merges = []
for f in range(41, 41 + PROBATION + 2):                    # A가 계속 보이는데 같은 외형 raw 2 등장
    _, m = r.update(FRAME, [(1, box(50, x=50)), (2, box(50, x=400))], f, now=f * 0.1)
    merges += m
check("병합 없음", merges, [])

print("\n6) 작은 박스는 임베딩하지 않고, 기한이 지나면 그냥 새 사람으로 확정")
enc = FakeEncoder()
r = PersonReID(enc, verbose=False)
for f in range(1, MAX_PENDING_FRAMES + 5):
    r.update(FRAME, [(1, box(50, h=MIN_BOX_H - 5))], f, now=f * 0.1)
check("임베딩 호출 0", enc.calls, 0)
check_true("pending 해소", 1 not in r.pending)
check_true("트랙은 유지(임베딩 없음)", 1 in r.tracks and r.tracks[1].emb is None)

print("\n7) 임베딩 주기 — 기존 트랙은 EMBED_EVERY 프레임마다만 호출")
enc = FakeEncoder()
r = PersonReID(enc, embed_every=3, verbose=False)
for f in range(1, 31):
    r.update(FRAME, [(1, box(50))], f, now=f * 0.1)
check_true("호출 수가 프레임 수보다 훨씬 적음", enc.calls < 20, f"calls={enc.calls} (30프레임)")

print("\n8) 인코더 없음 -> 통과(pass-through)")
r = PersonReID(None)
out, m = r.update(FRAME, [(5, box(50))], 1)
check("그대로 반환", (out[0][0], m), (5, []))
check_true("enabled=False", not r.enabled)
check_true("summary", "비활성" in r.summary())

print()
if fails:
    print(f"실패 {len(fails)}건: {fails}")
    sys.exit(1)
print("모든 Re-ID 테스트 통과")
