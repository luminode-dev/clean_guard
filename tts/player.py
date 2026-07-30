# -*- coding: utf-8 -*-
"""논블로킹 wav 재생기.

감시 루프는 큐에 경로만 넣고 즉시 돌아간다. 실제 재생은 데몬 스레드가 담당하며,
방송이 겹치지 않도록 한 번에 하나씩 순차 재생한다.
"""
import queue
import shutil
import subprocess
import threading
import time


def detect_player_cmd():
    """사용 가능한 재생 명령을 [prog, *args] 형태로 반환. 없으면 None."""
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    if shutil.which("play"):                 # sox
        return ["play", "-q"]
    return None


class WavPlayer:
    """maxsize를 넘으면 가장 오래된 항목을 버린다 (최신 이벤트 우선)."""

    def __init__(self, maxsize=3, cmd=None, timeout=30):
        self.q = queue.Queue(maxsize=maxsize)
        self.cmd = cmd or detect_player_cmd()
        self.timeout = timeout
        self._stop = threading.Event()
        self._thread = None
        self.played = 0
        self.dropped = 0

    def available(self):
        return self.cmd is not None

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="tts-player", daemon=True)
        self._thread.start()
        return self

    def submit(self, wav_path):
        """재생 요청. 큐가 가득 차면 오래된 것을 버리고 넣는다."""
        if self.cmd is None:
            return False
        try:
            self.q.put_nowait(str(wav_path))
            return True
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(str(wav_path))
                return True
            except queue.Full:
                self.dropped += 1
                return False

    def _run(self):
        while not self._stop.is_set():
            try:
                path = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            if path is None:
                break
            try:
                subprocess.run(self.cmd + [path], timeout=self.timeout,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.played += 1
            except Exception as e:                               # noqa: BLE001
                print(f"[TTS 경고] 재생 실패 ({type(e).__name__}): {path}")

    def stop(self, drain=True, timeout=15):
        """drain=True면 큐에 남은 방송을 최대 timeout초까지 재생하고 종료."""
        thread = self._thread
        if thread is None:
            return
        if drain:
            deadline = time.monotonic() + timeout
            while not self.q.empty() and time.monotonic() < deadline:
                time.sleep(0.2)
        self._stop.set()
        thread.join(timeout=self.timeout + 1)
        self._thread = None
