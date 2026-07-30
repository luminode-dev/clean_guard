# 무단투기 감지 — Jetson 배포 패키지

폐기물 10클래스 탐지 + 사람 추적 + 투기 이벤트 감지를 Jetson에서 TensorRT로 구동하기 위한 패키지.

## 폴더 구성

```
jetson_deploy/
├── models/
│   ├── waste10_yolo26n.pt      # 폐기물 10클래스 (최종 ft_neg 모델)
│   ├── waste10_yolo26n.onnx    #   └ 이식용 ONNX (opset17, 640, NMS 포함 e2e)
│   ├── person_yolo26n.pt       # 사람 탐지 (COCO 사전학습)
│   ├── person_yolo26n.onnx
│   ├── *.engine                # ← Jetson 위에서 convert_tensorrt.sh 로 생성
│   └── tts/supertonic-2/       # ← setup_tts.sh 가 내려받는 TTS 모델 (약 256MB)
├── dump_monitor_jetson.py      # 실행 모듈 (RTSP/웹캠/파일 입력)
├── convert_tensorrt.sh         # TensorRT 변환 (Jetson에서 실행)
├── tts/                        # 음성 경고 방송 모듈 (아래 "음성 경고 방송" 참고)
├── setup_tts.sh                # TTS 환경 셋업 (venv + 모델 다운로드)
├── prerender_tts.py            # 방송 문구 사전 렌더링 CLI
├── cache/tts/                  # ← 사전 렌더링된 방송 wav 120개 (약 90MB)
├── tests/
├── requirements.txt
└── README.md
```

폐기물 클래스(10): 쓰레기봉투, 대형가구, 가전제품, 페트, 캔, 병, 스티로폼, 종이박스, 의류, 플라스틱

## 설치 (Jetson, JetPack 6.x)

```bash
# 1) Jetson용 torch/torchvision 설치 (일반 pip wheel 아님!)
#    https://docs.ultralytics.com/guides/nvidia-jetson/ 의 JetPack 버전별 안내를 따를 것
# 2) 나머지 의존성
pip3 install -r requirements.txt
```

## TensorRT 변환 — Jetson Orin Nano에서 실행

`.engine` 파일은 빌드한 GPU 전용이므로 PC에서 만들 수 없다. 보드에서 1회 실행:

```bash
chmod +x convert_tensorrt.sh
./convert_tensorrt.sh          # FP16, 모델당 수 분 소요
```

스크립트는 먼저 Ultralytics의 `.pt` export를 시도하고, 실패하면 JetPack에 포함된
`/usr/src/tensorrt/bin/trtexec`로 `.onnx`를 빌드한다. 두 모델 모두 입력 크기 640,
배치 1, FP16으로 생성하며 기본 workspace는 2048 MiB다.

환경에 따라 다음 값을 조정할 수 있다:

```bash
IMG_SIZE=640 TRT_WORKSPACE_MB=2048 ./convert_tensorrt.sh
# trtexec 위치가 다른 경우
TRTEXEC=/usr/local/bin/trtexec ./convert_tensorrt.sh
```

변환이 끝나면 `models/*.engine` 파일 크기를 검사한다. 실제 엔진 생성은 대상 Orin
Nano에서 해야 하며, JetPack/TensorRT 버전이나 GPU가 바뀌면 다시 빌드해야 한다.
INT8은 캘리브레이션 데이터와 정확도 검증을 준비한 뒤 별도 작업으로 적용한다.

## 실행

`models/`에 `.engine`이 있으면 자동으로 TensorRT를 사용한다 (없으면 .onnx → .pt 순).

```bash
# RTSP CCTV 스트림
python3 dump_monitor_jetson.py --source "rtsp://user:pass@192.168.0.10:554/stream" --name cam01

# USB 웹캠 (화면 표시 포함)
python3 dump_monitor_jetson.py --source 0 --name webcam --show

# 영상 파일 테스트 (주석 영상 저장 생략으로 속도 확보)
python3 dump_monitor_jetson.py --source test.mp4 --name test --no-save-video
```

### 모델 자동 선택과 폴백

`models/`에서 **.engine → .onnx → .pt** 순으로 시도하며, 로드나 워밍업 추론이 실패하면 경고를 출력하고
다음 포맷으로 자동 폴백한다. JetPack/TensorRT 버전 변경으로 기존 `.engine`이 무효화돼도 서비스가
멈추지 않고 `.onnx`/`.pt`로 계속 동작한다 (속도는 느려지므로 로그에 경고가 보이면 재변환할 것).
특정 포맷을 강제하려면 `--waste-model models/waste10_yolo26n.pt` 처럼 확장자까지 지정한다.

### 출력 (`output/<name>/`)
- `events.jsonl` — 이벤트 로그 (시각, 클래스, **색상**, conf, 객체 id, 투기자 pid, bbox). append 방식이라 재시작해도 이어짐
- `event_NNNN.jpg` — 이벤트 증거 스냅샷 (빨간 박스 + 투기자 pid)
- `annotated.mp4` — 전체 주석 영상 (`--no-save-video`로 생략 가능)

## 음성 경고 방송 (`--tts`)

투기 발화 시 **"[색상] [쓰레기 종류]를 무단으로 버리셨습니다. 되가져가 주시기 바랍니다.
이곳은 CCTV 녹화 중입니다."** 를 현장 스피커로 방송한다. 클라우드 없이
[Supertonic](https://huggingface.co/Supertone/supertonic-2) ONNX 모델로 온디바이스 합성한다.

### 설계 원칙

- **사전 렌더링**: (색상 11 + 색상없음) × 쓰레기 10종 = **120개 문장을 미리 wav로 합성**해
  `cache/tts/`에 둔다. 이벤트 시점에는 파일 재생만 하므로 지연 0, 추론 자원 경합 없음
- **의존성 격리**: onnxruntime은 numpy 2.x를 요구하고 JetPack 기본 cv2는 numpy 1.x 빌드라
  섞이면 cv2가 깨진다. TTS 의존성은 전부 `.venv-tts/`에 격리하고, 합성은 항상 그 venv의
  서브프로세스([tts/synth_worker.py](tts/synth_worker.py))에서 실행한다.
  **감시 프로세스에는 추가 런타임 의존성이 없다**
- **장애 격리**: 모델·스피커·캐시에 무슨 문제가 생겨도 `[TTS 경고]`만 남기고 감시는 계속된다

### 셋업 (보드에서 1회)

```bash
bash setup_tts.sh          # venv + pip 부트스트랩 + supertonic-2 다운로드 + 스모크 테스트
# 방송 문구 120개 사전 렌더링 (약 20분, ~90MB). 모델 폴더는 models/tts/ 에서 자동 인식
.venv-tts/bin/python prerender_tts.py --all
```

디스크 여유가 600MB 이상 필요하다. 목소리를 바꾸려면 `ST_VOICE=F2 bash setup_tts.sh`
(M1~M5 / F1~F5). 더 고품질이 필요하면 `ST_MODEL=supertonic-3` (약 400MB, 31개 언어).

### 실행

```bash
python3 dump_monitor_jetson.py --source 0 --name cam01 --tts

# 장소명을 방송에 포함 (그 장소 전용 캐시를 먼저 만들어야 함)
.venv-tts/bin/python prerender_tts.py --all --location "정문 수거함 앞"
python3 dump_monitor_jetson.py --source 0 --name cam01 --tts --tts-location "정문 수거함 앞"
```

| 플래그 | 기본 | 의미 |
|---|---|---|
| `--tts` | off | 방송 활성화 |
| `--tts-cooldown` | 12.0 | 방송 간 최소 간격(초). 문장이 약 9초라 이보다 짧게 두면 대기열이 쌓인다 |
| `--tts-repeat-window` | 30.0 | 같은 (색상, 종류) 조합 재방송 억제 시간(초) |
| `--tts-location` | 없음 | 방송에 넣을 장소명 |
| `--tts-cache` | `cache/tts` | 방송 wav 캐시 폴더 |
| `--tts-no-runtime-synth` | off | 캐시 미스 시 실시간 합성 대신 방송 생략 (자원 보호) |
| `--tts-backend` | `supertonic` | `null`로 두면 무음 wav — 배선 점검용 |
| `--tts-model` / `--tts-voice` | `supertonic-2` / `M4` | 캐시 미스 합성용. 사전 렌더링과 같은 값이어야 목소리가 일관됨 |

### 색상 판정

`tts/color_naming.py`가 HSV 규칙으로 11색(빨간/주황/노란/초록/파란/보라/분홍/갈/흰/회/검은색)을
판정한다. 신규 의존성 없음. 배경 오염을 줄이기 위해 박스 안쪽 70%만 사용하고,
**이벤트 프레임 1장이 아니라 그 객체를 가장 확신했던 프레임의 크롭**(`WasteObject.best_crop`)에서
색을 뽑는다. 유채색 픽셀이 30% 미만이면 무채색(검정 봉투, 흰 스티로폼 등)으로 판정한다.

### 검증

```bash
python3 tests/test_color_naming.py      # 색상 판정 (모델 불필요)
python3 tests/test_tts_offline.py       # 문구/캐시/재생큐/쿨다운 (모델·스피커 불필요)
python3 tests/test_monitor_tts_hook.py  # YOLO 스텁으로 발화→색상→방송 전 경로 (ultralytics 불필요)
.venv-tts/bin/python prerender_tts.py --check   # 실제 합성 + 스피커 재생
```

## 이벤트 로직 요약

1. 사람: ByteTrack으로 track id 유지. 사람이 든 쓰레기는 그 사람의 "소지품"으로 등록(클래스+색 시그니처), 바닥 객체로 취급하지 않음
2. 새 바닥 폐기물 등장 시 근처 사람의 소지품과 대조해 투기자(owner pid) 귀속
3. 객체가 가려져도 300프레임까지 기억하고 위치+색상으로 재식별 → 기존 쓰레기가 "새 투기"로 둔갑하지 않음
4. 발화 조건: 새 객체 + 등장 시 사람 근접 + 12프레임 잔류 + **owner가 5프레임 연속 부재**
5. 영상 시작부터 있던 물체(baseline)는 이벤트 제외

## 튜닝 포인트 (dump_monitor_jetson.py 상단 상수)

| 상수 | 기본 | 의미 |
|---|---|---|
| `WASTE_CONF` | 0.20 | 폐기물 탐지 임계 (실환경 미탐이 많으면 0.15로) |
| `STABLE_AGE` | 12 | 잔류 판정 프레임 (fps에 맞춰 조정: 대략 1초 분량) |
| `NEAR_FACTOR` | 2.0 | 사람-객체 근접 판정 반경 |
| `LOST_KEEP` | 300 | 재식별 보관 프레임 (긴 가림이 잦으면 늘릴 것) |
| `CLEAR_FRAMES` | 5 | 사람 부재 확인 유예 |

## 성능 참고

- RTX 5090 (FP32 PyTorch) 기준 640px 2모델 파이프라인 ~16-22 FPS (영상 인코딩 포함)
- Jetson Orin 계열 + TensorRT FP16이면 nano 모델 2개는 실시간(15fps CCTV) 처리 가능 예상.
  부족하면: `--no-save-video`, 폐기물 추론을 2프레임당 1회로 스킵, INT8 변환 순으로 최적화

## 모델/로직 버전 (2026-07-27 기준)

- 폐기물 모델: **waste10_ft_bags** (AI Hub + 군집 합성 + 야간/저화질 열화 + 네거티브 배경 + Clean_Guard 실환경 봉투 6k장)
  - 메인 val mAP50 0.978 / 열화 val 0.974 / 실환경 봉투 벤치마크 mAP50 0.876 (recall 0.83)
- 이벤트 로직 v6: 소지품-사람 귀속, LOST 재식별, owner 이탈 5프레임 확인, **배경 차분으로 기존 적치물 오발화 억제**
- 합성 CCTV 투기 클립 검증: 주·야간 검정봉투 투기 정상 발화, 기존 더미 억제, 비투기 클립 오탐 0
- 남은 한계: 클립 종료 직전 투기(잔류 판정 프레임 부족)는 미발화 — 연속 스트림 운영에서는 해당 없음
