# -*- coding: utf-8 -*-
"""사람 재식별(Re-ID) — 트래커 위에 얹는 외형 갤러리 층.

트래커(ByteTrack)는 IoU+칼만 필터라 사람이 화면 밖으로 나갔다 돌아오면 새 id를 준다.
이 모듈은 사람마다 외형 임베딩(yolo26n-reid, 512차원)을 트랙 전체에 걸쳐 평균해 두었다가,
새 트랙이 나타나면 "최근 사라진 사람들"(갤러리)과 비교해 같은 사람이면 **이전 id를 돌려준다**.

왜 필요한가: 투기자가 방송을 듣고 돌아와 봉투를 집어 가거나, 잠깐 나갔다 오는 경우
owner_pid가 끊기면 "owner 부재" 판정과 회수 귀속이 틀어진다.

판정 규칙 (실측 근거: 부감 CCTV 보행자 11명 트랙 평균 임베딩에서 본인 0.79~0.92,
타인 최대 0.84 — 절대 임계만으론 부족):
  1. 새 트랙은 PROBATION 프레임 동안 임베딩을 모아 평균한 뒤 판정 (단일 프레임은 노이즈가 큼)
  2. 갤러리 = 이미 사라진(LOST_AFTER 프레임 이상 미관측) 트랙 중 TTL 이내인 것만
  3. 최고 유사도 ≥ SIM_THRESH  그리고  (최고 − 2위) ≥ MARGIN  일 때만 병합
  4. 병합되면 호출자는 merges 목록으로 owner_pid / carried 키를 옛 id로 옮긴다

연산: 사람당 EMBED_EVERY 프레임마다 1회, 한 프레임의 크롭은 배치로 한 번에 추론.
모델은 models/yolo26n-reid.{engine,onnx} 순으로 찾는다 (engine은 convert_tensorrt.sh가 생성).
"""
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_STEM = "yolo26n-reid"

SIM_THRESH = 0.75      # 병합 최소 코사인 유사도
MARGIN = 0.05          # 1위-2위 차이 하한 (비슷한 옷차림 오병합 방지)
PROBATION = 5          # 새 트랙 판정 전 모을 임베딩 수
EMBED_EVERY = 3        # 기존 트랙 임베딩 갱신 주기(프레임)
LOST_AFTER = 30        # 이 프레임 이상 미관측이면 '사라짐' (갤러리 후보). 트래커 track_buffer와 동일
TTL_S = 600.0          # 갤러리 보관 시간(초) — 회수 판정 시간과 맞춘다
MIN_BOX_H = 40         # 이보다 작은 사람 크롭은 임베딩하지 않음 (품질 낮음)
EMA = 0.9              # 기존 임베딩 가중 (새 임베딩 0.1)
MAX_PENDING_FRAMES = 90  # 임베딩을 못 얻은 채 이만큼 지나면 판정 포기(그냥 새 사람)


def find_models(model_path=None):
    """후보 경로 목록. 명시 경로가 있으면 그것만, 아니면 models/에서 .engine > .onnx 순."""
    if model_path:
        return [Path(model_path)]
    return [p for p in (HERE / "models" / (MODEL_STEM + ext) for ext in (".engine", ".onnx"))
            if p.exists()]


def load_encoder(model_path=None, device=None):
    """ultralytics ReID 인코더 로드. .engine이 무효(JetPack 변경 등)면 .onnx로 폴백.
    전부 실패하면 None (호출자는 Re-ID 없이 진행)."""
    cands = find_models(model_path)
    if not cands:
        print(f"[ReID 경고] 모델 없음: models/{MODEL_STEM}.(engine|onnx) -> 재식별 비활성. "
              f"README의 'Re-ID 모델' 항목 참고")
        return None
    try:
        from ultralytics.trackers.utils.reid import ReID
    except Exception as e:                                        # noqa: BLE001
        print(f"[ReID 경고] ultralytics ReID 모듈 없음 ({type(e).__name__}) -> 재식별 비활성")
        return None
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    for path in cands:
        try:
            enc = ReID(str(path), imgsz=224, device=device)
            enc(dummy, np.array([[112, 112, 100, 200]], dtype=np.float32))   # 워밍업 + 동작 확인
            print(f"[ReID] 인코더 로드: {path.name}")
            return enc
        except Exception as e:                                    # noqa: BLE001
            print(f"[ReID 경고] {path.name} 사용 불가 ({type(e).__name__}: {str(e)[:100]}) -> 다음 포맷 시도")
    print("[ReID 경고] 사용 가능한 Re-ID 모델 포맷이 없음 -> 재식별 비활성")
    return None


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class _Track:
    __slots__ = ("emb", "n", "last_frame", "last_time", "box", "first_frame")

    def __init__(self, frame_idx, now, box):
        self.emb = None
        self.n = 0
        self.last_frame = frame_idx
        self.last_time = now
        self.box = box
        self.first_frame = frame_idx

    def add(self, e):
        e = _unit(e)
        self.emb = e if self.emb is None else _unit(EMA * self.emb + (1 - EMA) * e)
        self.n += 1


class PersonReID:
    """persons=[(raw_pid, box)] -> [(canonical_pid, box)], merges=[(old_pid, new_pid)]

    canonical_pid는 그 사람이 처음 받았던 트래커 id. 새 raw id가 갤러리의 옛 id와
    병합되면 이후 그 raw id는 계속 옛 id로 매핑된다.
    """

    def __init__(self, encoder, sim_thresh=SIM_THRESH, margin=MARGIN, probation=PROBATION,
                 embed_every=EMBED_EVERY, ttl_s=TTL_S, lost_after=LOST_AFTER, verbose=True):
        self.enc = encoder
        self.sim_thresh = sim_thresh
        self.margin = margin
        self.probation = probation
        self.embed_every = embed_every
        self.ttl_s = ttl_s
        self.lost_after = lost_after
        self.verbose = verbose
        self.alias = {}          # raw_pid -> canonical pid
        self.tracks = {}         # canonical pid -> _Track
        self.pending = {}        # raw_pid -> {"emb": sum, "n": k, "since": frame}
        self.n_merged = 0
        self.n_new = 0
        self.n_embeds = 0

    @property
    def enabled(self):
        return self.enc is not None

    # ------------------------------------------------------------------
    def update(self, frame, persons, frame_idx, now=None):
        if not self.enabled:
            return list(persons), []
        now = time.monotonic() if now is None else now
        merges = []

        # 1) 이번 프레임에 임베딩할 대상 고르기
        todo = []                                   # (raw_pid, box)
        for raw, box in persons:
            if box[3] - box[1] < MIN_BOX_H:
                continue
            if raw not in self.alias:
                todo.append((raw, box))             # 새 트랙: 매 프레임
            elif raw in self.pending or frame_idx % self.embed_every == 0:
                todo.append((raw, box))
        embs = self._embed(frame, todo) if todo else {}

        # 2) 트랙 갱신 / 새 트랙 등록
        active = set()
        for raw, box in persons:
            if raw not in self.alias:
                self.alias[raw] = raw
                self.tracks[raw] = _Track(frame_idx, now, box)
                self.pending[raw] = {"emb": None, "n": 0, "since": frame_idx}
                self.n_new += 1
            cid = self.alias[raw]
            tr = self.tracks[cid]
            tr.last_frame, tr.last_time, tr.box = frame_idx, now, box
            active.add(cid)
            e = embs.get(raw)
            if raw in self.pending:
                p = self.pending[raw]
                if e is not None:
                    e = _unit(e)
                    p["emb"] = e if p["emb"] is None else p["emb"] + e
                    p["n"] += 1
                if p["n"] >= self.probation:
                    m = self._try_merge(raw, _unit(p["emb"]), frame_idx, now, active)
                    if m is not None:
                        merges.append((raw, m))
                        active.discard(raw)
                        active.add(m)
                    else:
                        self.tracks[raw].add(_unit(p["emb"]))
                    del self.pending[raw]
                elif frame_idx - p["since"] > MAX_PENDING_FRAMES:
                    if p["emb"] is not None:
                        self.tracks[raw].add(_unit(p["emb"]))
                    del self.pending[raw]
            elif e is not None:
                tr.add(e)

        # 3) 갤러리 만료
        expired = [cid for cid, tr in self.tracks.items()
                   if cid not in active and now - tr.last_time > self.ttl_s]
        for cid in expired:
            del self.tracks[cid]
            for raw in [r for r, c in self.alias.items() if c == cid]:
                del self.alias[raw]
                self.pending.pop(raw, None)

        return [(self.alias[raw], box) for raw, box in persons], merges

    # ------------------------------------------------------------------
    def _embed(self, frame, todo):
        dets = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]]
                         for _, b in todo], dtype=np.float32)      # ReID는 xywh(중심) 입력
        try:
            feats = self.enc(frame, dets)
        except Exception as e:                                    # noqa: BLE001
            print(f"[ReID 경고] 임베딩 실패 ({type(e).__name__}: {str(e)[:80]})")
            return {}
        self.n_embeds += len(todo)
        return {raw: f for (raw, _), f in zip(todo, feats) if f is not None}

    def _try_merge(self, raw, q, frame_idx, now, active):
        """갤러리(사라진 트랙)에서 q와 가장 비슷한 사람을 찾아 병합. 성공 시 옛 cid 반환."""
        cands = []
        for cid, tr in self.tracks.items():
            if cid == raw or cid in active or tr.emb is None:
                continue
            if frame_idx - tr.last_frame < self.lost_after:       # 아직 트래커가 보고 있을 수 있음
                continue
            if now - tr.last_time > self.ttl_s:
                continue
            cands.append((float(q @ tr.emb), cid))
        if not cands:
            return None
        cands.sort(reverse=True)
        best, cid = cands[0]
        second = cands[1][0] if len(cands) > 1 else 0.0
        if best < self.sim_thresh or best - second < self.margin:
            return None
        # 병합: raw -> cid, 임시 트랙 제거, 갤러리 트랙 갱신
        old = self.tracks.pop(raw)
        tr = self.tracks[cid]
        gap_frames = old.first_frame - tr.last_frame            # 사라졌다 다시 나타나기까지
        tr.add(q)
        tr.last_frame, tr.last_time, tr.box = old.last_frame, old.last_time, old.box
        self.alias[raw] = cid
        self.n_merged += 1
        if self.verbose:
            print(f"[ReID] P{raw} -> P{cid} 재식별 (sim {best:.2f}, 2위 {second:.2f}, "
                  f"{gap_frames}프레임 만에 재등장)")
        return cid

    def summary(self):
        if not self.enabled:
            return "Re-ID 비활성"
        return (f"Re-ID: 새 트랙 {self.n_new}건 · 재식별 병합 {self.n_merged}건 · "
                f"임베딩 {self.n_embeds}회")


__all__ = ["PersonReID", "load_encoder", "find_models", "MODEL_STEM"]
