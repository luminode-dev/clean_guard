#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""하트비트 (--report와 함께 동작) — jetson_handoff.md §3.4 / jetson_data.md §3 계약.

  POST {API_URL}/api/heartbeat   (30초 주기, Bearer 장비토큰)

서버는 90초 동안 못 받으면 장치를 offline으로 판정하고, 대시보드 장치 카드의
fps·스트림·CPU/GPU·온도·디스크·TTS 값을 이 본문으로 채운다.

- 이벤트와 달리 큐에 쌓지 않는다: 지난 상태를 나중에 보내 봐야 의미가 없다. 실패하면 다음 주기에 새로.
- 감시 루프 값은 collect() 콜백으로 읽는다 (정수·문자열 읽기뿐이라 락 불필요).
- 시스템 값은 /proc·/sys에서 직접 읽는다 (psutil 불필요).
- GPU 부하는 tegrastats의 GR3D_FREQ(0.5초 간격)를 주기 동안 모아 평균을 보낸다 (2026-10-05 요청).
  순간값이라 추론 사이 유휴 순간엔 0이 나오므로 (10fps 추론에서 표본의 절반가량) 평균이 필요하다.
  tegrastats가 없으면 /sys .../load를 1초마다 읽어 같은 방식으로 평균.
- disk_free_mb는 루트(eMMC) 여유. 증거·큐를 RAM(/dev/shm)에 두는 운용에선 output_free_mb가 따로 간다.
  disk_low 기준은 서버 경보와 같은 2,000MB (jetson_data.md §6).
- configs=[RoiConfig, SiteConfig]를 넘기면 본문 config.<키>_version으로 적용 버전을 알리고,
  응답의 버전이 더 크면 그 자리에서 새 설정을 받아 적용한다 (remote_config.py).

ThumbnailUploader: 5초마다 최신 프레임을 640x360 JPEG로 PUT (handoff §3.4, 대시보드 카드·지도).
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

HTTP_TIMEOUT = 10.0
GPU_LOAD = "/sys/devices/platform/bus@0/17000000.gpu/load"      # Orin: 0~1000
LOW_FPS = 5.0
HIGH_TEMP_C = 85.0
DISK_LOW_MB = 2000.0        # 서버 disk_low 경보 기준과 같게 (jetson_data.md §6)
TEGRASTATS_MS = 500
_GR3D = re.compile(r"GR3D_FREQ (\d+)%")


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


class TegraStats:
    """tegrastats를 띄워 두고 GR3D_FREQ(GPU 부하 %) 표본을 모은다. 없거나 죽으면 alive=False."""

    def __init__(self, interval_ms=TEGRASTATS_MS):
        self.samples = []
        self._proc = None
        try:
            self._proc = subprocess.Popen(["tegrastats", "--interval", str(int(interval_ms))],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          text=True, bufsize=1)
        except OSError:
            return
        threading.Thread(target=self._read, name="tegrastats", daemon=True).start()

    @property
    def alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _read(self):
        for line in self._proc.stdout:
            m = _GR3D.search(line)
            if m:
                self.samples.append(int(m.group(1)))

    def take(self):
        out, self.samples = self.samples, []
        return out

    def close(self):
        if self.alive:
            self._proc.terminate()


class SystemProbe:
    """CPU·GPU·메모리·온도·디스크·네트워크. CPU%와 tx_kbps는 직전 호출 대비 변화량."""

    def __init__(self, disk_path="/", output_path=None, tegrastats=True):
        self.disk_path = disk_path
        self.output_path = output_path
        self._cpu_prev = self._cpu_times()
        self._net_prev = None
        self.power_mode = self._power_mode()
        self._ntp = (0.0, None)          # (확인 시각, 동기화 여부) — 5분마다만 확인
        self._gpu = []                   # tick()이 모은 sysfs GPU 부하 표본 (0~1000, 폴백용)
        self._tegra = TegraStats() if tegrastats else None

    @staticmethod
    def _cpu_times():
        try:
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            return sum(v), idle
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _power_mode():
        try:
            out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in out.splitlines():
                if "Power Mode" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:                                    # noqa: BLE001
            pass
        return None

    @staticmethod
    def _default_iface():
        try:
            with open("/proc/net/route") as f:
                for line in f.readlines()[1:]:
                    p = line.split()
                    if p[1] == "00000000":
                        return p[0]
        except OSError:
            pass
        return None

    @staticmethod
    def _rssi(iface):
        """/proc/net/wireless의 신호(dBm). 유선이면 None."""
        try:
            with open("/proc/net/wireless") as f:
                for line in f.readlines()[2:]:
                    name, rest = line.split(":", 1)
                    if name.strip() == iface:
                        return int(float(rest.split()[2]))
        except (OSError, ValueError, IndexError):
            pass
        return None

    def _temps(self):
        out = {}
        base = "/sys/class/thermal"
        for z in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            t = _read(f"{base}/{z}/type", "")
            if t in ("cpu-thermal", "gpu-thermal"):
                v = _read(f"{base}/{z}/temp")
                if v and v.lstrip("-").isdigit():
                    out[t.split("-")[0]] = round(int(v) / 1000.0, 1)
        return out

    def ntp_synced(self):
        t, ok = self._ntp
        if time.monotonic() - t > 300:
            try:
                out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized"],
                                     capture_output=True, text=True, timeout=5).stdout
                ok = out.strip().endswith("yes")
            except Exception:                                # noqa: BLE001
                ok = None
            self._ntp = (time.monotonic(), ok)
        return ok

    def tick(self):
        """(tegrastats가 없을 때) sysfs GPU 부하 표본 1개 적재. 하트비트 대기 중 1초마다 부른다."""
        if self._tegra is not None and self._tegra.alive:
            return
        g = _read(GPU_LOAD)
        if g and g.isdigit():
            self._gpu.append(int(g))

    def gpu_pct(self):
        """직전 호출 이후 GPU 부하 평균 (%). tegrastats 표본 우선, 없으면 sysfs 표본."""
        tg = self._tegra.take() if self._tegra is not None else []
        self.tick()
        gpu, self._gpu = self._gpu, []
        if tg:
            return round(sum(tg) / len(tg))
        return round(sum(gpu) / len(gpu) / 10) if gpu else None

    def close(self):
        if self._tegra is not None:
            self._tegra.close()

    def sample(self):
        s = {}
        cur = self._cpu_times()
        if cur and self._cpu_prev:
            dt, di = cur[0] - self._cpu_prev[0], cur[1] - self._cpu_prev[1]
            s["cpu_pct"] = round(100.0 * (dt - di) / dt) if dt > 0 else 0
        self._cpu_prev = cur
        s["gpu_pct"] = self.gpu_pct()
        mem = {}
        for line in (_read("/proc/meminfo", "") or "").splitlines():
            k, _, v = line.partition(":")
            mem[k] = int(v.split()[0]) if v.split() else 0
        if mem.get("MemTotal"):
            s["mem_pct"] = round(100 - 100.0 * mem.get("MemAvailable", 0) / mem["MemTotal"])
        s["temp_c"] = self._temps()
        try:
            s["disk_free_mb"] = round(shutil.disk_usage(self.disk_path).free / (1024 * 1024))
        except OSError:
            s["disk_free_mb"] = None
        if self.output_path:
            try:
                s["output_free_mb"] = round(shutil.disk_usage(self.output_path).free
                                            / (1024 * 1024))
            except OSError:
                s["output_free_mb"] = None
        s["power_mode"] = self.power_mode
        iface = self._default_iface()
        tx = _read(f"/sys/class/net/{iface}/statistics/tx_bytes") if iface else None
        now = time.monotonic()
        tx_kbps = None
        if tx and tx.isdigit():
            if self._net_prev and self._net_prev[0] == iface and now > self._net_prev[2]:
                tx_kbps = round((int(tx) - self._net_prev[1]) * 8 / 1000
                                / (now - self._net_prev[2]))
            self._net_prev = (iface, int(tx), now)
        s["net"] = {"iface": iface, "rssi": self._rssi(iface) if iface else None,
                    "tx_kbps": tx_kbps}
        return s


class Heartbeat:
    """interval초마다 collect()+시스템 값을 보낸다. 시작 직후 1회 즉시 전송."""

    def __init__(self, base_url, token, device_id, collect, interval=30.0,
                 disk_path="/", verbose=True, roi=None, configs=(), output_path=None,
                 tegrastats=True):
        self.url = base_url.rstrip("/") + "/api/heartbeat"
        self.token = token
        self.device_id = device_id
        self.collect = collect
        self.configs = [c for c in (roi, *configs) if c is not None]
        self.interval = max(5.0, float(interval))
        self.verbose = verbose
        self.probe = SystemProbe(disk_path, output_path, tegrastats=tegrastats)
        self.t_start = time.monotonic()
        self.sent = 0
        self.failed = 0
        self.last_ok = None              # 상태가 바뀔 때만 로그 (30초마다 도배 방지)
        self._prev = None                # (monotonic, frames_total) — fps 계산용
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="heartbeat", daemon=True)
        self._th.start()

    def build(self):
        st = self.collect() or {}
        pipe = dict(st.get("pipeline") or {})
        now = time.monotonic()
        frames = pipe.get("frames_total")
        if frames is not None and self._prev and now > self._prev[0]:
            pipe["fps"] = round((frames - self._prev[1]) / (now - self._prev[0]), 1)
        else:
            pipe.setdefault("fps", None)
        if frames is not None:
            self._prev = (now, frames)
        system = self.probe.sample()

        issues = list(st.get("issues") or [])
        if pipe.get("stream") not in (None, "ok"):
            issues.append("stream_lost")
        elif pipe.get("fps") is not None and pipe["fps"] < LOW_FPS:
            issues.append("low_fps")
        if any(v >= HIGH_TEMP_C for v in system.get("temp_c", {}).values()):
            issues.append("high_temp")
        if system.get("disk_free_mb") is not None and system["disk_free_mb"] < DISK_LOW_MB:
            issues.append("disk_low")
        if (st.get("tts") or {}).get("worker") == "down":
            issues.append("tts_worker_down")
        if self.probe.ntp_synced() is False:
            issues.append("clock_unsynced")

        body = {"device_id": self.device_id, "ts": now_iso(),
                "uptime_s": round(now - self.t_start), "pipeline": pipe}
        for k in ("tts", "mosaic"):
            if st.get(k) is not None:
                body[k] = st[k]
        body["system"] = system
        body["issues"] = sorted(set(issues))
        if self.configs:
            body["config"] = {c.version_key: c.version for c in self.configs}
        return body

    def send_once(self):
        try:
            body = self.build()
            req = urllib.request.Request(
                self.url, data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                headers={"Authorization": f"Bearer {self.token}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                raw = r.read()
            self.sent += 1
            if self.last_ok is not True and self.verbose:
                print(f"[하트비트] 전송 {'복구' if self.last_ok is False else '시작'} "
                      f"({self.interval:.0f}초 주기) · fps={body['pipeline'].get('fps')} "
                      f"· issues={body['issues']}", flush=True)
            self.last_ok = True
        except Exception as e:                               # noqa: BLE001
            self.failed += 1
            if self.last_ok is not False and self.verbose:
                what = (f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError)
                        else f"{type(e).__name__}: {str(e)[:60]}")
                print(f"[하트비트] 전송 실패({what}) — 다음 주기에 재시도", flush=True)
            self.last_ok = False
            return False
        if self.configs:
            try:
                cfg = (json.loads(raw) or {}).get("config") or {}
            except (ValueError, AttributeError):
                cfg = {}
            for c in self.configs:
                ver = cfg.get(c.version_key) if isinstance(cfg, dict) else None
                if ver is not None:
                    c.sync(ver)              # 새 버전이면 받아서 적용 + ack (예외 안 냄)
        return True

    def _loop(self):
        while not self._stop.is_set():
            self.send_once()
            deadline = time.monotonic() + self.interval
            while not self._stop.is_set():
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._stop.wait(min(1.0, left))
                self.probe.tick()

    def close(self, timeout=5.0):
        self._stop.set()
        self._th.join(timeout)
        self.probe.close()

    def summary(self):
        return f"하트비트: {self.sent}회 전송" + (f" · 실패 {self.failed}회" if self.failed else "")


class ThumbnailUploader:
    """최신 프레임을 interval초마다 PUT /api/devices/{id}/thumbnail (JPEG 본문).

    감시 루프는 offer(frame)로 프레임 참조만 넘긴다 (리사이즈·인코딩·전송은 이 스레드).
    루프는 매번 새 프레임을 받으므로 넘긴 배열을 나중에 덮어쓰지 않는다 — 복사 불필요.
    실패는 조용히 넘기고 다음 주기에 새 프레임으로 다시 보낸다 (큐 없음).
    """

    def __init__(self, base_url, token, device_id, interval=5.0, size=(640, 360),
                 quality=75, verbose=True):
        self.url = f"{base_url.rstrip('/')}/api/devices/{device_id}/thumbnail"
        self.token = token
        self.interval = max(1.0, float(interval))
        self.size, self.quality = tuple(size), int(quality)
        self.verbose = verbose
        self.sent = self.failed = 0
        self.last_ok = None
        self._frame = None
        self._next_t = 0.0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="thumbnail", daemon=True)
        self._th.start()

    def offer(self, frame):
        now = time.monotonic()
        if now < self._next_t:
            return
        self._next_t = now + self.interval
        self._frame = frame
        self._wake.set()

    def _encode(self, frame):
        import cv2
        h, w = frame.shape[:2]
        tw, th = self.size
        r = min(tw / w, th / h)
        if r < 1:
            frame = cv2.resize(frame, (max(1, round(w * r)), max(1, round(h * r))),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def send_once(self, frame):
        try:
            data = self._encode(frame)
            if not data:
                return False
            req = urllib.request.Request(
                self.url, data=data, method="PUT",
                headers={"Authorization": f"Bearer {self.token}",
                         "Content-Type": "image/jpeg"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT):
                pass
            self.sent += 1
            if self.last_ok is not True and self.verbose:
                print(f"[썸네일] 전송 {'복구' if self.last_ok is False else '시작'} "
                      f"({self.interval:.0f}초 주기, {len(data) // 1024}KiB)", flush=True)
            self.last_ok = True
            return True
        except Exception as e:                               # noqa: BLE001
            self.failed += 1
            if self.last_ok is not False and self.verbose:
                what = (f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError)
                        else f"{type(e).__name__}: {str(e)[:60]}")
                print(f"[썸네일] 전송 실패({what}) — 다음 주기에 재시도", flush=True)
            self.last_ok = False
            return False

    def _loop(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            frame, self._frame = self._frame, None
            if frame is not None and not self._stop.is_set():
                self.send_once(frame)

    def close(self, timeout=3.0):
        self._stop.set()
        self._wake.set()
        self._th.join(timeout)

    def summary(self):
        return f"썸네일: {self.sent}회 전송" + (f" · 실패 {self.failed}회" if self.failed else "")
