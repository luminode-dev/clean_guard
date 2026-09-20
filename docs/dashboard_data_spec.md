# Clean Guard 관제 대시보드 — 데이터 정의 (초안 v0.1)

**작성일:** 2026-09-19
**대상 독자:** 대시보드(웹 UI) 개발자, 관제 서버(API) 개발자
**범위:** 여러 대의 Jetson 감시 장치 → 관제 서버 → 지자체 운영자 화면으로 흐르는 **데이터의 정의**.
화면 설계와 서버 구현은 다루지 않는다. 전송 방식은 데이터 의미를 정하는 데 필요한 만큼만 언급한다.

---

## 0. 한눈에 보기

```
[Jetson N대]                     [관제 서버]                      [대시보드]
 dump_monitor_jetson.py  ──▶  수집·저장·집계  ──▶  ① 장치 현황 (지도/목록)
  · 하트비트 30s              (API + DB + 스토리지)   ② 실시간 영상 (썸네일 격자 / 단일 라이브)
  · 탐지 이벤트 + 스냅샷                             ③ 이벤트 목록·상세·처리
  · 라이브 스트림                                    ④ 통계 (지점별·시간대별·방송 효과)
                                                     ⑤ 알림 (장치 장애 / 새 이벤트)
```

데이터 종류는 여섯 가지다.

| # | 데이터 | 방향 | 성격 | 주기 |
|---|---|---|---|---|
| 1 | **Site** (설치 지점) | 서버 마스터 | 정적 | 등록 시 |
| 2 | **Device** (장치) | 장치 → 서버 | 준정적 + 상태 | 등록 시 / 상태 변경 시 |
| 3 | **Heartbeat** (상태·건강) | 장치 → 서버 | 시계열 | 30초 |
| 4 | **Event** (무단투기 이벤트) | 장치 → 서버 → 운영자 | 트랜잭션 + 처리 워크플로 | 발생 시 |
| 5 | **Stream** (실시간 영상) | 장치 → 대시보드 | 스트림 + 썸네일 | 상시 / 요청 시 |
| 6 | **Alert** (알림) | 서버 → 운영자 | 파생 | 조건 충족 시 |

여기에 화면용 **집계(Stats)** 가 서버에서 파생된다.

### 공통 규약

- 시각: ISO 8601 + 오프셋 (`2026-09-19T14:03:21+09:00`). 장치 시각(`ts`)과 서버 수신 시각(`received_at`)을 **둘 다** 둔다 — 보드 시계가 어긋날 수 있다(NTP 미동기, 재부팅 직후).
- ID: `device_id`는 사람이 읽을 수 있는 고정 문자열(`JT-GN-0007`: 지역코드-일련), `event_id`는 전역 유일 ULID(시간순 정렬 가능). 장치가 오프라인 중 만든 이벤트도 충돌 없이 나중에 올릴 수 있다.
- 좌표: 화면 픽셀 좌표(bbox)는 **원본 해상도 기준** `[x1, y1, x2, y2]` 정수. 스트림 해상도가 다르면 대시보드가 비율로 변환한다.
- 이미지/영상 파일은 페이로드에 넣지 않고 **URL**로 참조한다 (서버 스토리지에 업로드 후 서명 URL).
- 개인정보: 얼굴이 가려진 자료가 **기본**, 원본은 별도 권한(§4.4).

---

## 1. Site — 설치 지점

지자체가 관리하는 단위. 장치는 지점에 속한다 (지점당 장치 1대가 보통이지만 1:N 허용).

```json
{
  "site_id": "SITE-GN-0007",
  "name": "역삼1동 수거함 앞",
  "region": { "sido": "서울특별시", "sigungu": "강남구", "dong": "역삼1동", "code": "1168064000" },
  "address": "서울 강남구 역삼로 123",
  "location": { "lat": 37.4979, "lng": 127.0276 },
  "dept": "청소행정과",
  "contact": { "name": "홍길동", "phone": "02-000-0000" },
  "camera": { "install_height_m": 3.5, "view_note": "수거함 정면, 야간 IR" },
  "hours": { "notice": "무단투기 단속 구역 · CCTV 녹화 중" },
  "created_at": "2026-08-01T10:00:00+09:00"
}
```

- `region.code`: 행정동 코드 — 지자체 통계·필터의 기준 키
- `dept`/`contact`: 이벤트 처리 담당 부서. 알림 라우팅에 쓴다

## 2. Device — 장치

```json
{
  "device_id": "JT-GN-0007",
  "site_id": "SITE-GN-0007",
  "name": "역삼1동 #1",
  "hw": { "model": "Jetson Orin Nano 8GB", "serial": "1421xxxx", "jetpack": "6.2" },
  "sw": {
    "version": "clean_guard 2026.09.19",
    "git": "e24b34e",
    "models": {
      "waste": { "name": "waste10_yolo26n", "format": "engine", "fallback": false },
      "person": { "name": "person_yolo26n", "format": "engine", "fallback": false }
    }
  },
  "config": {
    "source": "rtsp://.../stream",
    "waste_conf": 0.20, "stable_age": 12, "clear_frames": 5,
    "tts": { "enabled": true, "mode": "live", "voice": "M4", "location": "역삼1동 수거함 앞",
             "cooldown_s": 12, "repeat_window_s": 30 },
    "mosaic": { "mode": "face", "raw_snapshot": true }
  },
  "status": "online",
  "status_since": "2026-09-19T06:12:00+09:00",
  "last_heartbeat_at": "2026-09-19T14:03:00+09:00",
  "registered_at": "2026-08-01T10:00:00+09:00"
}
```

**`status`** (서버가 하트비트로 판정):

| 값 | 조건 |
|---|---|
| `online` | 최근 하트비트 ≤ 90초 & 스트림 정상 & 추론 정상 |
| `degraded` | 온라인이지만 문제 있음 — 모델 폴백(.engine→.onnx), TTS 워커 다운, FPS 저하, 온도 경고 등. 사유는 하트비트 `issues` |
| `offline` | 하트비트 90초 초과 |
| `maintenance` | 운영자가 수동 설정 (점검 중 알림 억제) |

- `sw.models.*.format`: `engine` 이어야 정상. `onnx`/`pt`면 폴백 상태(느림) → `degraded`
- `config`는 **장치가 실제로 적용 중인 값**을 보고한다 (서버에서 내려보낸 값과 다를 수 있음). 원격 설정 변경은 §7

## 3. Heartbeat — 상태·건강 (30초)

```json
{
  "device_id": "JT-GN-0007",
  "ts": "2026-09-19T14:03:00+09:00",
  "uptime_s": 28440,
  "pipeline": {
    "fps": 14.2,
    "infer_ms": { "person": 21, "waste": 19 },
    "frames_total": 403000,
    "stream": "ok",
    "tracked_persons": 2,
    "tracked_objects": 1
  },
  "tts": { "worker": "ready", "played_total": 37, "suppressed_total": 12, "synth_fail_total": 0,
           "last_synth_ms": 2400 },
  "mosaic": { "mode": "face", "faces_total": 51200, "head_fallback_total": 9800 },
  "system": {
    "cpu_pct": 61, "gpu_pct": 74, "mem_pct": 58,
    "temp_c": { "cpu": 58.5, "gpu": 61.0 },
    "disk_free_mb": 12400,
    "power_mode": "15W",
    "net": { "iface": "eth0", "rssi": null, "tx_kbps": 820 }
  },
  "issues": []
}
```

- `pipeline.stream`: `ok` | `reconnecting` | `lost` — RTSP 끊김 감지의 1차 신호
- `issues`: 장치가 스스로 판단한 이상 목록. 코드 예: `model_fallback`, `tts_worker_down`, `stream_lost`, `low_fps`, `high_temp`, `disk_low`, `clock_unsynced`
- 서버는 최근 값을 Device에 반영하고, 시계열은 보존 기간(예: 30일)만 둔다
- `tts.last_synth_ms`: 실시간 합성 지연 실측 — 보드별 성능 확인용

## 4. Event — 무단투기 이벤트

### 4.1 장치가 보내는 것 (현재 `events.jsonl` + 첨부)

현재 파일의 필드를 그대로 쓰되 식별자·시각·첨부를 보강한다.

```json
{
  "event_id": "01J8ZK3V9Q6X2N4M8P0R5S7T9V",
  "device_id": "JT-GN-0007",
  "site_id": "SITE-GN-0007",
  "ts": "2026-09-19T14:02:41+09:00",
  "received_at": "2026-09-19T14:02:43+09:00",

  "detection": {
    "class": "쓰레기봉투",
    "class_id": 0,
    "color": "검은색",
    "night": null,
    "conf": 0.75,
    "bbox": [299, 300, 361, 359],
    "frame_size": [1920, 1080]
  },
  "suspect": {
    "obj_id": 4,
    "owner_pid": 4,
    "owner_matched": false,
    "person_bbox_at_drop": [280, 210, 330, 340]
  },
  "announce": {
    "played": true,
    "phrase": "역삼1동 수거함 앞에 검은색 쓰레기봉투를 무단으로 버리셨습니다. 되가져가 주시기 바랍니다. 이곳은 CCTV 녹화 중입니다.",
    "suppressed_reason": null
  },
  "media": {
    "snapshot": "https://.../ev/01J8ZK.../snapshot.jpg",
    "snapshot_raw": "https://.../ev/01J8ZK.../snapshot_raw.jpg",
    "clip": "https://.../ev/01J8ZK.../clip.mp4",
    "clip_range_s": [-10, 10]
  },
  "debug": { "frame": 403120, "local_seq": 38, "model_format": "engine" }
}
```

필드 출처:

| 필드 | 현재 파이프라인 | 비고 |
|---|---|---|
| `detection.class/color/night/conf/bbox` | `events.jsonl` 그대로 | `night`는 `"적외선"`/`"저조도"`/null |
| `suspect.obj_id/owner_pid/owner_matched` | `events.jsonl` 그대로 | `owner_matched=true`면 "들고 온 물건과 일치" (강한 귀속) |
| `suspect.person_bbox_at_drop` | **추가 필요** | 투기자 위치를 스냅샷에 같이 표시하기 위함 |
| `announce.*` | Announcer 반환값 | 방송 여부·억제 사유(`쿨다운`/`중복`/`합성실패`) |
| `media.snapshot` | `event_NNNN.jpg` (모자이크본) | 기본 공개 자료 |
| `media.snapshot_raw` | `event_NNNN_raw.jpg` | **권한 필요** (§4.4) |
| `media.clip` | **추가 필요** | 전후 10초 클립 — 프레임 링버퍼에서 잘라 인코딩 |

### 4.2 서버가 붙이는 것 — 처리 워크플로

지자체 운영자가 이벤트를 **확인하고 조치**하는 상태 기계. 판매 시 "단속 업무 시스템"으로서의 핵심.

```json
{
  "review": {
    "state": "confirmed",
    "assignee": "user:kim",
    "history": [
      { "at": "2026-09-19T14:10:00+09:00", "by": "user:kim", "from": "new", "to": "reviewing" },
      { "at": "2026-09-19T14:12:30+09:00", "by": "user:kim", "from": "reviewing", "to": "confirmed",
        "note": "검정 봉투 1개, 투기자 남성 추정" }
    ],
    "action": { "type": "field_visit", "at": "2026-09-19T16:00:00+09:00", "result": "수거 완료", "fine_issued": false }
  },
  "outcome": {
    "retrieved": false,
    "retrieved_at": null,
    "object_last_seen_at": "2026-09-19T15:58:10+09:00"
  }
}
```

**`review.state`**

| 상태 | 의미 | 전이 |
|---|---|---|
| `new` | 수신됨, 미확인 | → `reviewing`, `dismissed` |
| `reviewing` | 담당자가 보는 중 | → `confirmed`, `dismissed` |
| `confirmed` | 무단투기 맞음 | → `actioned` |
| `dismissed` | 오탐/해당 없음 (사유 필수: `false_positive`, `authorized`, `duplicate`) | 종결 |
| `actioned` | 현장 조치·과태료 등 완료 | 종결 |

**`outcome.retrieved`** — 방송 효과 지표. 장치가 이벤트 후 그 객체를 계속 추적하다가 **N분(예: 10분) 안에 사라지고 재식별되지 않으면** "되가져감"으로 보고한다. 파이프라인에 후속 메시지(`event_update`)가 **추가 필요**:

```json
{ "event_id": "01J8ZK...", "device_id": "JT-GN-0007", "ts": "...", "update": "retrieved", "after_s": 312 }
```

> 이 지표가 있어야 "방송을 켠 지점은 회수율 N%" 같은 근거를 지자체에 보여줄 수 있다.

### 4.3 dismissed 사유가 모델 개선 데이터가 된다

`dismissed(false_positive)`로 닫힌 이벤트의 스냅샷은 네거티브 학습 데이터 후보다. 서버는 `review.state`별로 자료를 내보낼 수 있어야 한다 (라벨링 파이프라인 연계는 별도).

### 4.4 개인정보 — 원본 접근

- 대시보드 기본 표시는 **모자이크 스냅샷/영상**. `snapshot_raw`·원본 클립은 별도 권한(`role: investigator`)이 있어야 URL이 발급되고, **발급 이력(누가·언제·어느 이벤트)** 을 남긴다.
- 보존 기간: 모자이크본 기본 90일, 원본 30일(지자체 규정에 맞춰 설정값). `dismissed`는 더 짧게.
- 장치 로컬 `output/` 도 같은 규칙으로 순환 삭제해야 한다 (현재 무제한 append → **추가 필요**).

## 5. Stream — 실시간 영상

대시보드의 두 가지 사용 방식에 맞춰 두 가지 데이터를 둔다.

### 5.1 썸네일 (격자·지도 팝업용)

장치가 **5초마다** 최신 프레임(모자이크·주석 포함)을 JPEG로 올린다. 수십 대를 한 화면에 띄워도 부담이 없다.

```json
{
  "device_id": "JT-GN-0007",
  "ts": "2026-09-19T14:03:05+09:00",
  "url": "https://.../thumb/JT-GN-0007/latest.jpg",
  "size": [640, 360],
  "overlay": { "persons": 2, "objects": 1, "events_today": 3 }
}
```

### 5.2 라이브 스트림 (단일 장치 상세 화면)

운영자가 장치를 열었을 때만 연결한다 (상시 N개 스트림은 회선·보드 부담).

```json
{
  "device_id": "JT-GN-0007",
  "protocol": "webrtc",
  "url": "https://media.../whep/JT-GN-0007",
  "variants": [
    { "id": "annotated", "desc": "추적 박스·이벤트 표시 + 모자이크", "default": true },
    { "id": "clean",     "desc": "모자이크만" },
    { "id": "raw",       "desc": "원본 (권한 필요)", "role": "investigator" }
  ],
  "resolution": [1280, 720], "fps": 10, "latency_target_ms": 800,
  "expires_at": "2026-09-19T14:33:00+09:00"
}
```

- 프로토콜 권고: **WebRTC(WHEP)** — 브라우저 지연 1초 미만. 폴백으로 HLS(지연 5~10초).
- `annotated` 변형은 파이프라인이 그리는 주석 프레임을 그대로 내보낸 것 — 대시보드가 박스를 다시 그릴 필요가 없다. 대신 이벤트 오버레이를 UI에서 직접 그리려면 §5.3의 실시간 트랙 채널을 쓴다.
- 장치당 인코딩 부담: Orin Nano 하드웨어 인코더(NVENC) 720p 1스트림은 여유 있음. 두 변형을 동시에 내보내는 건 피한다 (선택된 변형 하나만).

### 5.3 실시간 트랙 채널 (선택)

UI가 박스를 직접 그리거나 "지금 사람 몇 명" 같은 카운트를 실시간 표시하려면, 프레임당 경량 메시지(WebSocket):

```json
{ "device_id": "JT-GN-0007", "ts": "...", "frame": 403121,
  "persons": [ { "pid": 4, "bbox": [280,210,330,340], "carrying": true } ],
  "objects": [ { "obj_id": 4, "cls": "쓰레기봉투", "bbox": [299,300,361,359], "state": "fired", "owner_pid": 4 } ] }
```

`state`: `new` → `stable` → `fired` | `baseline` (기존 적치물, 이벤트 제외) — 화면 색과 대응(주황/빨강/회색).

## 6. Alert — 알림

서버가 규칙으로 만든다. 대시보드 상단 배지·알림 목록·(선택) 문자/메신저.

```json
{
  "alert_id": "01J8ZK...",
  "ts": "2026-09-19T14:05:00+09:00",
  "severity": "warning",
  "kind": "device_degraded",
  "device_id": "JT-GN-0007",
  "site_id": "SITE-GN-0007",
  "summary": "모델 폴백 (.engine → .onnx) — FPS 14 → 4",
  "detail": { "issue": "model_fallback", "since": "2026-09-19T13:58:00+09:00" },
  "acked": false, "acked_by": null
}
```

| `kind` | severity | 조건 |
|---|---|---|
| `event_new` | info | 새 이벤트 (지점별 구독) |
| `event_burst` | warning | 같은 지점 1시간 내 이벤트 ≥ N |
| `device_offline` | critical | 하트비트 90초 초과 |
| `device_degraded` | warning | `issues` 비어 있지 않음 |
| `stream_lost` | critical | `pipeline.stream = lost` 2분 지속 |
| `disk_low` | warning | `disk_free_mb` < 2000 |
| `high_temp` | warning | gpu/cpu ≥ 80°C |

## 7. Command — 서버 → 장치 (범위 밖, 인터페이스만)

대시보드에서 필요한 원격 조작. 지금 파이프라인엔 없고 **추가 필요**. 데이터 정의만 미리 둔다.

```json
{ "command_id": "...", "device_id": "JT-GN-0007", "type": "set_config",
  "payload": { "tts.enabled": false }, "issued_by": "user:kim", "issued_at": "...",
  "status": "acked", "result": null }
```

`type`: `set_config` · `restart_pipeline` · `snapshot_now` · `tts_test` (방송 점검) · `reboot` · `update_software`

## 8. Stats — 화면용 집계 (서버 파생)

쿼리로 뽑되, 화면이 자주 쓰는 형태를 정해 둔다.

```json
{
  "range": { "from": "2026-09-01", "to": "2026-09-19", "tz": "Asia/Seoul" },
  "group_by": "site",
  "rows": [
    { "site_id": "SITE-GN-0007", "events": 41, "confirmed": 33, "dismissed": 6, "actioned": 20,
      "retrieved": 18, "retrieved_rate": 0.44,
      "by_class": { "쓰레기봉투": 30, "종이박스": 7, "대형가구": 4 },
      "by_hour": [0,0,1,0,0,0,2,5,4,3,2,1,1,2,3,4,6,5,2,0,0,0,0,0],
      "night_share": 0.39,
      "device_uptime_pct": 99.2 }
  ]
}
```

- `group_by`: `site` | `region` | `day` | `hour` | `class`
- `retrieved_rate` = retrieved / confirmed — 방송 효과 KPI
- `night_share`: `night != null` 비율 — 야간 투기 비중

---

## 9. 파이프라인에 추가해야 하는 것 (정리)

| 항목 | 이유 | 난이도 |
|---|---|---|
| `device_id`/`site_id`/ULID를 이벤트에 부여, 서버 업로드(재시도 큐) | 다장치 식별, 오프라인 내성 | 중 |
| 하트비트 송신 (§3) — fps/온도/모델 포맷/TTS 워커 상태 | 장치 현황 | 중 |
| `person_bbox_at_drop` 기록 | 스냅샷에 투기자 표시 | 하 |
| 이벤트 전후 클립 (프레임 링버퍼 → mp4) | 증거 | 중 |
| `retrieved` 후속 판정 + `event_update` | 방송 효과 KPI | 중 |
| 썸네일 주기 업로드 / 라이브 스트림 송출 (WebRTC) | 실시간 영상 | 중~상 |
| `output/` 순환 삭제, 원본 스냅샷 보존 기간 | 개인정보 | 하 |
| 원격 명령 수신 (§7) | 운영 | 중 |

## 10. 정해야 할 것 (질문)

1. **장치 ↔ 서버 연결 환경**: 지자체 망(폐쇄망) vs 인터넷(LTE). 폐쇄망이면 서버는 지자체 내부에 두고 스트림은 사내 미디어 서버로.
2. **원본(비모자이크) 자료 정책**: 누가 볼 수 있는지, 보존 며칠인지 — 지자체마다 규정이 다르므로 설정값으로.
3. **처리 워크플로 깊이**: 과태료 부과 연계(행정 시스템 API)까지 갈지, 대시보드 안에서 `actioned` 기록까지만 할지.
4. **회수 판정 시간(N분)**: 10분 기본 제안. 짧으면 오판(잠깐 가려짐), 길면 KPI 지연.
5. **라이브 스트림 동시 시청 상한**: 장치당 1~2 세션 제안.
