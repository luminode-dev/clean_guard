#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""입력 스트림 / 회전 녹화 테스트 (모델·카메라·네트워크 불필요).

상시 운영(RTSP)에서 문제가 됐던 세 가지를 검증한다.
- 영상이 무한히 커지는 것    -> RotatingRecorder 분할·보존·디스크 가드
- 스트림 끊김 시 프로세스 종료 -> FrameSource 재연결 (여기서는 판정 로직과 파일 경로만)
- 지연 누적                  -> drop_stale 설정이 입력 종류에 맞게 켜지는지

  python3 tests/test_stream_recorder.py
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recorder import RotatingRecorder, free_mb          # noqa: E402
from stream import FrameSource, is_live                 # noqa: E402

W, H, NFRAMES = 64, 48, 30
fails = []


def ok(cond, msg):
    print(f"  {'OK  ' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(msg)


def make_clip(path, n=NFRAMES):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    for i in range(n):
        f = np.full((H, W, 3), i * 7 % 255, np.uint8)
        w.write(f)
    w.release()
    return path


def frame():
    return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)


print("\n1) is_live — 입력 종류 판정")
for s, want in [("rtsp://cam/1", True), ("RTSP://cam/1", True), ("0", True),
                ("http://x/s.m3u8", True), ("clip.mp4", False), ("/a/b.avi", False)]:
    ok(is_live(s) is want, f"{s} -> live={want}")

print("\n2) 파일 입력 — 전 프레임 순차 처리 후 EOF")
tmp = Path(tempfile.mkdtemp(prefix="cg_stream_"))
try:
    clip = make_clip(tmp / "clip.mp4")
    src = FrameSource(str(clip)).open()
    ok(not src.live, "파일은 live=False")
    ok(not src.drop_stale, "파일은 프레임 드롭 안 함 (전 프레임 필요)")
    ok(src.width == W and src.height == H, f"해상도 {src.width}x{src.height}")
    n, none_frames = 0, 0
    while True:
        good, f = src.read()
        if not good:
            break
        if f is None:
            none_frames += 1
            continue
        n += 1
    ok(n >= NFRAMES - 2, f"프레임 {n}장 수신 (기대 ~{NFRAMES})")
    ok(none_frames == 0, "파일 경로에서는 빈 프레임을 돌려주지 않음")
    ok(src.n_dropped == 0, "파일은 드롭 0")
    src.close()

    print("\n3) 없는 입력 — 즉시 오류 (URL 오타를 몇 초 안에 알림)")
    t0 = time.monotonic()
    try:
        FrameSource(str(tmp / "없는파일.mp4")).open()
        ok(False, "예외가 나야 함")
    except RuntimeError:
        ok(time.monotonic() - t0 < 5, "파일 입력은 재시도 없이 즉시 실패")

    print("\n4) RotatingRecorder — 분할 · 보존 · 요약")
    out = tmp / "rec"
    out.mkdir()
    r = RotatingRecorder(out, 10.0, (W, H), segment_min=1 / 60, keep=2, min_free_mb=0)
    deadline = time.monotonic() + 2.6          # 세그먼트 1초 -> 3개 정도
    while time.monotonic() < deadline:
        r.write(frame())
        time.sleep(0.02)
    r.close()
    segs = sorted(out.glob("annotated_*.mp4"))
    ok(r.n_segments >= 2, f"세그먼트 {r.n_segments}개 생성 (분할 동작)")
    ok(len(segs) <= 2, f"보존 정책 적용: 남은 파일 {len(segs)}개 (keep=2)")
    ok(r.n_pruned >= 1, f"오래된 세그먼트 {r.n_pruned}개 삭제")
    ok(r.n_frames > 0 and "녹화:" in r.summary(), "요약 출력 정상")

    print("\n5) 디스크 가드 — 여유 부족이면 녹화만 포기")
    out2 = tmp / "rec2"
    out2.mkdir()
    huge = free_mb(out2) + 10_000_000          # 절대 만족 못 할 임계
    r2 = RotatingRecorder(out2, 10.0, (W, H), segment_min=60, keep=5, min_free_mb=huge)
    wrote = any(r2.write(frame()) for _ in range(5))
    r2.close()
    ok(not wrote, "공간 부족 시 프레임을 쓰지 않음")
    ok(r2.paused, "paused 상태로 전환")
    ok(not list(out2.glob("annotated_*.mp4")), "세그먼트 파일이 생기지 않음")
    ok("중단" in r2.summary(), "요약에 중단 표시")

    print("\n6) OutputSpace — 공간 부족 시 오래된 것부터 삭제")
    import recorder as rec_mod

    def seed(d):
        """영상 3개 + 스냅샷 3개를 시각 순서대로 만든다 (video1이 가장 오래됨)."""
        d.mkdir(parents=True, exist_ok=True)
        (d / "events.jsonl").write_text("{}\n", encoding="utf-8")
        made = []
        for i, name in enumerate(["annotated_20260101_000001.mp4", "event_0001.jpg",
                                  "annotated_20260101_000002.mp4", "event_0002.jpg",
                                  "annotated_20260101_000003.mp4", "event_0003.jpg"]):
            p = d / name
            p.write_bytes(b"x" * 1024)
            import os
            os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))   # 생성 순서 = mtime 순서
            made.append(p)
        return made

    def with_fake_free(d, min_free, order, per_delete=50.0, base=100.0, exclude=None):
        """삭제 1개당 per_delete MB가 회복되는 가짜 디스크로 reclaim을 돌린다."""
        total = len(list(d.glob("annotated_*.mp4"))) + len(list(d.glob("event_*.jpg")))
        orig = rec_mod.free_mb

        def fake(_p):
            left = len(list(d.glob("annotated_*.mp4"))) + len(list(d.glob("event_*.jpg")))
            return base + (total - left) * per_delete

        rec_mod.free_mb = fake
        try:
            sp = rec_mod.OutputSpace(d, min_free_mb=min_free, headroom_mb=0,
                                     purge_order=order)
            return sp, sp.reclaim(exclude=exclude)
        finally:
            rec_mod.free_mb = orig

    d1 = tmp / "sp1"
    seed(d1)
    sp1, (n1, _) = with_fake_free(d1, min_free=200.0, order="video-first")
    left_v = sorted(p.name for p in d1.glob("annotated_*.mp4"))
    left_e = sorted(p.name for p in d1.glob("event_*.jpg"))
    ok(n1 == 2, f"필요한 만큼만 삭제: {n1}개 (100MB -> 200MB)")
    ok(left_e == ["event_0001.jpg", "event_0002.jpg", "event_0003.jpg"],
       "video-first: 스냅샷은 하나도 안 지움")
    ok(left_v == ["annotated_20260101_000003.mp4"],
       f"video-first: 오래된 영상부터 삭제 (남은 것 {left_v})")
    ok((d1 / "events.jsonl").exists(), "events.jsonl은 절대 삭제하지 않음")

    d2 = tmp / "sp2"
    seed(d2)
    sp2, (n2, _) = with_fake_free(d2, min_free=200.0, order="oldest-first")
    gone = not (d2 / "annotated_20260101_000001.mp4").exists() and \
        not (d2 / "event_0001.jpg").exists()
    ok(n2 == 2 and gone, "oldest-first: 종류 무관하게 가장 오래된 2개 삭제")
    ok((d2 / "annotated_20260101_000002.mp4").exists(), "더 새 영상은 보존")

    d3 = tmp / "sp3"
    seed(d3)
    keep = d3 / "annotated_20260101_000001.mp4"
    sp3, _ = with_fake_free(d3, min_free=200.0, order="video-first", exclude=keep)
    ok(keep.exists(), "기록 중인 세그먼트는 exclude로 보호됨")

    d4 = tmp / "sp4"
    d4.mkdir()
    (d4 / "events.jsonl").write_text("{}\n", encoding="utf-8")
    sp4, (n4, _) = with_fake_free(d4, min_free=200.0, order="video-first")
    ok(n4 == 0 and sp4.n_starved == 1, "지울 게 없으면 경고만 남기고 계속")

    print("\n7) enabled=False — 호출부 분기 없이 no-op")
    out3 = tmp / "rec3"
    out3.mkdir()
    r3 = RotatingRecorder(out3, 10.0, (W, H), enabled=False)
    ok(r3.write(frame()) is False, "write()가 조용히 False")
    r3.close()
    ok(not list(out3.glob("*.mp4")), "파일 생성 없음")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if fails:
    print(f"실패 {len(fails)}건:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("모든 스트림/녹화 테스트 통과")
