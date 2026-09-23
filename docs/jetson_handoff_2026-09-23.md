# Jetson 보드 인수인계 — 2026-09-23

**대상:** Jetson Orin Nano 8GB 보드에서 검증·운용하는 담당자
**전제:** 보드에 `clean_guard/`가 이미 설치되어 있고, 마지막 보드 동기화는 커밋 `154773e`(방송 모듈 최초 추가) 시점
**현재 main:** `67ea53b`

이 문서 하나만 보고 (1) 보드를 최신 상태로 올리고 (2) 새 기능 4개를 순서대로 검증하고
(3) 결과를 보고할 수 있게 쓴다. **PC에서 검증한 것과 보드에서만 확인 가능한 것을 명확히 구분**했다.

---

## 0. 요약 — 무엇이 바뀌었나

`154773e` 이후 커밋 4개가 올라갔다.

| 커밋 | 기능 | 플래그 | PC 검증 | 보드 검증 |
|---|---|---|---|---|
| `12808b5` | 방송 v2: 이벤트 시 **즉석 합성** + 야간 색상 생략 | `--tts-mode live`(기본) | 완료 | **필요** (합성 지연) |
| `e24b34e` | 사람 **안면 모자이크** | `--mosaic head\|face` | 완료 | 권장 (FPS 영향) |
| `f76ca14` | 사람 **재식별** | `--reid` | 완료 | **필요** (engine 동적 배치) |
| `8ac5554` | **torch 없는 백엔드** | `--backend trt` | onnxruntime 경로만 | **필수** (TensorRT 경로 미검증) |

**기본값은 모두 이전과 같다.** `--mosaic`/`--reid`/`--backend trt`는 켜야 동작하므로,
`git pull` 만으로 기존 운용이 깨지지 않는다. 단 `--tts`는 기본 동작이 사전 렌더링 재생 →
**즉석 합성으로 바뀌었다** (이전 방식은 `--tts-mode cache`).

---

## 1. 보드 올리기

### 1-1. 디스크 먼저 확인

07-30 보고서 기준 **여유 1.3GB (96% 사용)** 였다. 이번에 모델 2개(Re-ID 9.4MB, YuNet 0.2MB)와
Re-ID 엔진(~20MB)이 늘어난다. 1GB 미만이면 먼저 정리한다.

```bash
df -h /
du -sh ~/Downloads/* 2>/dev/null | sort -rh | head -5      # 대개 여기가 범인
```

### 1-2. 코드·모델 받기

```bash
cd ~/clean_guard            # 실제 경로에 맞게
git fetch origin && git log --oneline HEAD..origin/main    # 받을 커밋 확인
git pull origin main
python3 -c "import scipy; print('scipy', scipy.__version__)" || pip3 install scipy
```

`scipy`는 `bytetrack.py`(헝가리안 매칭)가 쓴다. `--backend trt`를 쓸 때만 필요하지만 미리 깔아 둔다.

### 1-3. Re-ID 엔진 빌드 (선택이지만 권장)

```bash
./convert_tensorrt.sh       # 기존 두 엔진 + yolo26n-reid.engine 까지
```

- 기존 `.engine` 2개가 이미 있으면 다시 만들어도 되고, Re-ID만 필요하면 스크립트가 알아서 그 단계만 새로 수행한다
- Re-ID는 `trtexec`로 **동적 배치(1~16, 224x224)** 빌드한다. 실패해도 스크립트는 경고만 남기고 계속 진행하며, 런타임에 `.onnx`로 폴백한다
- **변환 중에는 감시 프로세스·TTS를 모두 끈다** (메모리 경합). 실패하면 `TRT_WORKSPACE_MB=1024 ./convert_tensorrt.sh`

### 1-4. `--backend trt`를 쓸 거면

```bash
pip3 install cuda-python
python3 -c "import tensorrt, cuda; print('trt', tensorrt.__version__)"
```

`tensorrt`는 JetPack에 포함(`python3-libnvinfer`). `cuda-python`은 GPU 버퍼 관리용으로 새로 필요하다.

---

## 2. 검증 순서

각 단계마다 **기대 로그**와 **기록할 값**을 적었다. 값은 §5 양식에 채워 보고.

### 2-0. 모델 없이 되는 테스트 먼저 (2분)

```bash
python3 tests/test_color_naming.py      # 색상·야간 판정
python3 tests/test_mosaic.py            # 모자이크 (YuNet 포함)
python3 tests/test_person_reid.py       # 재식별 갤러리 규칙
python3 tests/test_tts_offline.py       # 방송 문구/합성/폴백
python3 tests/test_trt_backend.py       # 백엔드 (아래 주의)
python3 tests/test_monitor_tts_hook.py  # 전체 경로 (YOLO 스텁)
```

> `test_trt_backend.py`의 3번 항목(ultralytics와 출력 일치)은 `.onnx` + onnxruntime을 쓴다.
> 보드에 onnxruntime이 없으면 그 부분은 실패하거나 건너뛴다 — **정상**이다. 1·2번만 통과하면 된다.
> (보드의 시스템 numpy는 1.x라 onnxruntime을 시스템에 깔면 cv2가 깨진다. 절대 깔지 말 것.)

### 2-1. 기준선 — 기존과 같은 조건 (회귀 확인)

```bash
python3 dump_monitor_jetson.py --source <테스트영상.mp4> --name base --no-save-video
```

기대: `backend=ultralytics waste=waste10_yolo26n.engine person=person_yolo26n.engine`
**기록: FPS, 이벤트 건수** — 이 값이 이후 비교 기준이다.

### 2-2. 방송 v2 — 즉석 합성 지연 (가장 궁금한 값)

```bash
python3 dump_monitor_jetson.py --source <테스트영상.mp4> --name tts --no-save-video --tts
```

기대 로그:
```
[TTS] 방송 활성 (실시간 합성, 백엔드=supertonic) · 재생=paplay
[TTS] 합성 워커 준비 완료 (N.Ns)        ← 모델 로드 (한 번만)
[TTS] 방송 (합성 N.Ns): 검은색 쓰레기봉투를 ...
```

**기록: `합성 워커 준비 완료` 초, `방송 (합성 N.Ns)` 초.**
PC(x86 CPU) 실측은 로드 후 첫 합성 7.3초, 이후 **2.0초**(11초 분량 문장)였다. 보드는 2~4배 느릴 것으로 본다.

- 합성이 5초를 넘으면 현장에서 "버리고 돌아서는 사이"를 놓친다 → `--tts-mode cache`로 전환 검토
  (사전 렌더링 필요: `.venv-tts/bin/python prerender_tts.py --all`, 약 20분/90MB)
- 스피커가 없으면 `[TTS 경고] 재생 명령... 방송 비활성` 이 뜬다. 배선 점검은 `--tts-backend null`(무음 wav)로

### 2-3. 모자이크 · 재식별

```bash
python3 dump_monitor_jetson.py --source <영상> --name mos --no-save-video --mosaic face --reid
```

기대 로그:
```
[ReID] 인코더 로드: yolo26n-reid.engine        ← .onnx 로 뜨면 엔진 빌드 실패한 것 (동작은 함, 느림)
[모자이크] face 모드
...
[모자이크(face): 얼굴 N건 · head 폴백 M건]
[Re-ID: 새 트랙 N건 · 재식별 병합 M건 · 임베딩 K회]
```

**기록: FPS(2-1 대비 하락폭), Re-ID 인코더가 engine인지 onnx인지, head 폴백 비율.**

- `head 폴백` 비율이 압도적으로 높으면(부감 CCTV에서 흔함) `--mosaic head`가 더 싸고 충분하다.
  PC 데모 영상에서는 얼굴 134 / 폴백 1217 이었다
- Re-ID `.engine` 로드가 실패하면 `사용 불가 -> 다음 포맷 시도` 후 `.onnx`로 내려간다.
  **이 경우 로그 전문을 보고**해 달라 — 동적 배치 처리 문제일 가능성이 크다
- 결과물 확인: `output/mos/event_*.jpg` 에서 **얼굴이 실제로 가려졌는지 눈으로** 볼 것

### 2-4. torch 없는 백엔드 (가장 검증이 필요한 부분)

```bash
python3 dump_monitor_jetson.py --source <영상> --name trt --no-save-video --backend trt --reid
```

기대 로그: `backend=trt waste=waste10_yolo26n.engine person=person_yolo26n.engine`

**이 경로는 PC에서 TensorRT로 실행해 본 적이 없다.** 문제가 날 만한 지점과 대처:

| 증상 | 원인 추정 | 대처 |
|---|---|---|
| `ModuleNotFoundError: cuda` | cuda-python 미설치 | `pip3 install cuda-python` |
| `ImportError: cannot import name 'runtime'` | cuda-python 버전별 경로 차이 | 로그 보고 — `trt_backend.py:67` 의 두 import 분기 수정 필요 |
| `엔진 역직렬화 실패` | JetPack/TensorRT 버전 불일치 | `./convert_tensorrt.sh` 재실행 |
| `CUDA 오류` / 출력 이상 | FP16 입력 dtype, 동적 shape | 로그 + `trtexec --loadEngine=models/person_yolo26n.engine --verbose` 앞부분 보고 |
| Re-ID만 실패 | 동적 배치 미지원 | `--reid` 빼고 나머지만 먼저 확인 |

**두 백엔드 결과가 같아야 한다:**

```bash
python3 dump_monitor_jetson.py --source <영상> --name cmp_ul --no-save-video --backend ultralytics
python3 dump_monitor_jetson.py --source <영상> --name cmp_trt --no-save-video --backend trt
diff <(cut -d, -f5-  output/cmp_ul/events.jsonl) <(cut -d, -f5- output/cmp_trt/events.jsonl)
```

`ts`(시각) 필드만 다르고 나머지(class/color/conf/box/owner_pid)는 같아야 한다.
PC 데모 영상 211프레임에서는 이벤트·트랙 id·재식별 병합까지 완전히 동일했다.

**기록: 실행 여부, FPS, 메모리(아래), events 일치 여부.**

### 2-5. 메모리 실측 (지금 가장 알고 싶은 값)

"VRAM 부족" 이슈 때문에 **추정이 아니라 실측**이 필요하다. Jetson은 CPU/GPU가 8GB를 공유하므로
GPU와 무관해 보이는 것(데스크톱, TTS 워커)도 전부 같은 통에서 뺀다.

각 조합을 돌리면서 **다른 터미널**에서:

```bash
tegrastats --interval 2000 | head -3                    # RAM x/7xxxMB
ps -eo rss,comm,args --sort=-rss | head -8              # 프로세스별 RSS (KB)
free -h
```

측정할 조합 (최소 3개):

| # | 명령 | 알고 싶은 것 |
|---|---|---|
| A | `--backend ultralytics` (기본) | 현재 기준선 |
| B | `--backend trt` | torch 제거 효과 |
| C | `--backend trt --tts --reid --mosaic face` | 전부 켠 실제 운용 |

참고로 PC(CPU, 같은 영상)에서는 피크 629MB → **481MB**, 1.5 → 3.2 FPS 였다.
보드에서는 CUDA 컨텍스트가 빠지므로 차이가 더 클 것으로 본다.

OOM이 났다면:
```bash
dmesg | grep -i -E 'oom|killed process' | tail
```

---

## 3. 메모리가 부족할 때 — 효과 큰 순서

| 조치 | 예상 회수 | 방법 |
|---|---|---|
| **헤드리스 전환** | ~1.5GB | `sudo systemctl set-default multi-user.target && sudo reboot` (되돌리기: `graphical.target`). 배포 장비는 모니터가 없으니 기본으로 이렇게 간다. `--show`만 못 쓴다 |
| **`--backend trt`** | ~1GB | torch 미적재. 단 §2-4 검증 후 |
| **`--tts-mode cache`** | ~0.45GB (실측) | 워커 상주 안 함. 사전 렌더링 필요 |
| Re-ID 최대 배치 축소 | 수백 MB | `convert_tensorrt.sh` 의 `--maxShapes=images:16x...` → `4x` 후 재빌드 |
| 스왑 추가 | 스파이크 흡수 | 디스크 여유 있을 때만 |

**INT8은 메모리 대책이 아니다.** 가중치가 엔진당 2~3MB 줄 뿐이고, 큰 덩어리(torch·데스크톱·TTS)와 무관하다.
INT8은 FPS가 모자랄 때 쓰는 카드이며 캘리브레이션 데이터와 정확도 재검증이 필요하다.

---

## 4. 알아두어야 할 것들

- **RTSP 지연 누적**: 현재 코드는 프레임을 스킵하지 않는다. 추론 FPS가 카메라 FPS보다 낮으면
  디코더 버퍼가 쌓여 화면이 점점 뒤처진다. 2-1의 FPS가 카메라 FPS보다 낮으면 알려 달라 —
  "최신 프레임만 읽기"로 고쳐야 한다
- **로직 상수가 프레임 단위**: `STABLE_AGE 12`, `CLEAR_FRAMES 5`, `LOST_KEEP 300` 등은 프레임 수다.
  실제 FPS가 15와 크게 다르면 의미가 달라진다 (8fps면 잔류 판정이 0.8초 → 1.5초).
  실측 FPS를 알려주면 초 단위로 환산하도록 바꾼다
- **`output/` 무한 증가**: 스냅샷·이벤트 로그가 계속 쌓인다. 순환 삭제가 아직 없으니
  장기 운용 전에 cron으로 정리하거나 요청해 달라
- **`--raw-snapshot`**: 얼굴을 가리지 않은 원본을 따로 저장하는 옵션. 증거 보관용이며
  **폴더 접근 통제가 필요**하다. 기본은 꺼져 있다
- **디스크**: 07-30 기준 여유 1.3GB였다. TTS 사전 렌더링(90MB)·엔진·`output/` 누적을 감안할 것

---

## 5. 보고 양식

아래를 채워 이슈나 메시지로 회신해 주면 된다.

```
[환경]
JetPack / L4T:
디스크 여유:
헤드리스 여부:

[2-0] 테스트 6종:  통과 / 실패( 어느 것 )
[2-1] 기준선      FPS ___  이벤트 ___건
[2-2] TTS         워커 준비 ___초,  합성 ___초 / 문장
[2-3] 모자이크    FPS ___,  Re-ID 인코더 = engine / onnx,  얼굴 ___ : head폴백 ___
[2-4] trt 백엔드  실행 성공 / 실패(로그 첨부),  FPS ___,  events 일치 Y/N
[2-5] 메모리      A(ultralytics) ___MB / B(trt) ___MB / C(전부) ___MB,  tegrastats RAM ___

[문제]
(에러 로그 원문 붙여넣기)
```

**실패해도 그대로 보고**해 주는 게 가장 도움이 된다. 특히 §2-4는 PC에서 검증할 수 없는 부분이라
에러 원문이 유일한 단서다.

---

## 6. 참고 문서

- [README.md](../README.md) — 전체 사용법, 플래그 표, 튜닝 상수
- [docs/dashboard_data_spec.md](dashboard_data_spec.md) — 다장치 관제 대시보드 데이터 정의 (보드가 앞으로 보내야 할 것)
- [docs/개발보고서_2026-07-30.md](개발보고서_2026-07-30.md) — 이전 보드 작업 기록 (환경 실측값)
