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
│   ├── face_detection_yunet_2023mar.onnx  # 안면 모자이크용 YuNet (230KB, --mosaic face)
│   └── tts/supertonic-2/       # ← setup_tts.sh 가 내려받는 TTS 모델 (약 256MB)
├── dump_monitor_jetson.py      # 실행 모듈 (RTSP/웹캠/파일 입력)
├── mosaic.py                   # 사람 안면 모자이크 (OpenCV만 사용)
├── convert_tensorrt.sh         # TensorRT 변환 (Jetson에서 실행)
├── tts/                        # 음성 경고 방송 모듈 (아래 "음성 경고 방송" 참고)
├── setup_tts.sh                # TTS 환경 셋업 (venv + 모델 다운로드)
├── prerender_tts.py            # 방송 문구 사전 렌더링 CLI (선택: cache 모드 / 폴백용)
├── cache/tts/                  # ← 사전 렌더링 wav (선택). live 모드에선 합성 실패 시 폴백으로만 사용
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
- `events.jsonl` — 이벤트 로그 (시각, 클래스, **색상**, **night**(`"적외선"`/`"저조도"`/null), conf, 객체 id, 투기자 pid, bbox). append 방식이라 재시작해도 이어짐
- `event_NNNN.jpg` — 이벤트 증거 스냅샷 (빨간 박스 + 투기자 pid). `--mosaic` 시 얼굴 가려짐
- `event_NNNN_raw.jpg` — `--mosaic --raw-snapshot` 일 때만: 얼굴을 가리지 않은 원본 스냅샷
- `annotated.mp4` — 전체 주석 영상 (`--no-save-video`로 생략 가능). `--mosaic` 시 얼굴 가려짐

## 사람 안면 모자이크 (`--mosaic`)

출력물(스냅샷 / `annotated.mp4` / `--show` 화면)에 찍히는 사람 얼굴을 픽셀화한다.
**추론은 원본 프레임으로 하고 출력 직전에만 가리므로** 탐지·재식별·색상 판정에는 영향이 없다.
OpenCV만 쓰며 추가 pip 의존성은 없다 ([mosaic.py](mosaic.py)).

```bash
python3 dump_monitor_jetson.py --source 0 --name cam01 --mosaic head          # 모델 불필요
python3 dump_monitor_jetson.py --source 0 --name cam01 --mosaic face          # YuNet 얼굴 탐지
python3 dump_monitor_jetson.py --source 0 --name cam01 --mosaic face --raw-snapshot
```

| 플래그 | 기본 | 의미 |
|---|---|---|
| `--mosaic` | `off` | `head`=ByteTrack 사람 박스의 상단 18%(머리)를 픽셀화. 비용 0, 사람이 탐지된 한 항상 가려짐 / `face`=YuNet으로 얼굴을 찾아 얼굴만 가리고, 못 찾으면(뒷모습·모자·저조도) 그 사람은 `head`로 폴백 |
| `--face-model` | `models/face_detection_yunet_2023mar.onnx` | YuNet onnx 경로 |
| `--raw-snapshot` | off | 얼굴을 가리지 않은 원본 스냅샷을 `event_NNNN_raw.jpg`로 **추가** 저장. 증거 보관용이므로 폴더 접근 통제가 필요하다 |

- 픽셀 블록은 얼굴 크기에 비례(짧은 변을 8칸으로)해서 가까운 큰 얼굴도 윤곽이 남지 않는다.
  강도는 `mosaic.py`의 `CELLS`(작을수록 강함), `HEAD_RATIO`, `FACE_PAD`로 조정
- `face` 모드는 전체 프레임이 아니라 **사람 박스 크롭에만** YuNet을 돌리므로 Orin Nano CPU에서
  프레임당 수 ms 수준이다. FPS가 아쉬우면 `head`로
- YuNet은 `cv2.FaceDetectorYN`(OpenCV 4.5.4+, JetPack 6의 cv2 4.8 포함)으로 실행한다.
  모델 파일이 없거나 cv2가 지원하지 않으면 경고 후 `head`로 자동 전환한다. 재다운로드:
  `curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- 종료 시 `[모자이크(face): 얼굴 N건 · head 폴백 M건]`으로 가린 건수를 출력한다.
  폴백 비율이 높으면 카메라 각도상 얼굴이 잘 안 잡히는 것이므로 `head`가 더 안전하다
- 한계: 사람 탐지(`PERSON_CONF=0.35`)에서 놓친 사람은 가리지 못한다

## 음성 경고 방송 (`--tts`)

투기 발화 시 **"[색상] [쓰레기 종류]를 무단으로 버리셨습니다. 되가져가 주시기 바랍니다.
이곳은 CCTV 녹화 중입니다."** 를 현장 스피커로 방송한다. 클라우드 없이
[Supertonic](https://huggingface.co/Supertone/supertonic-2) ONNX 모델로 온디바이스 합성한다.
**야간 적외선(흑백)·저조도 프레임에서는 색상을 판정하지 않고 종류만 방송한다**
("페트병을 무단으로 버리셨습니다…").

### 설계 원칙

- **실시간 합성 (기본, `--tts-mode live`)**: 이벤트가 나면 문구를 그 자리에서 합성해 방송한다.
  합성은 감시 루프와 분리된 스레드가 담당하고, `synth_worker.py --serve` **상주 프로세스**에
  작업을 보내므로 모델 로드(수 초~수십 초)는 시작 시 한 번뿐이고 이벤트 때는 합성 시간만 든다
  (Orin Nano CPU에서 한 문장 수 초 수준). 감시 루프는 방송 요청만 큐에 넣고 즉시 돌아온다.
  합성 워커는 시작 직후 미리 띄워 두며(예열), 죽으면 다음 방송 때 자동 재시작한다
- **의존성 격리**: onnxruntime은 numpy 2.x를 요구하고 JetPack 기본 cv2는 numpy 1.x 빌드라
  섞이면 cv2가 깨진다. TTS 의존성은 전부 `.venv-tts/`에 격리하고, 합성은 항상 그 venv의
  서브프로세스([tts/synth_worker.py](tts/synth_worker.py))에서 실행한다.
  **감시 프로세스에는 추가 런타임 의존성이 없다**
- **장애 격리**: 모델·스피커·워커에 무슨 문제가 생겨도 `[TTS 경고]`만 남기고 감시는 계속된다.
  실시간 합성이 실패했는데 `cache/tts/`에 같은 문구의 사전 렌더링본이 있으면 그것을 대신 튼다
- **사전 렌더링 (선택, `--tts-mode cache`)**: (색상 11 + 색상없음) × 쓰레기 10종 = 120문장을
  `prerender_tts.py --all`로 미리 만들어 두고 재생만 하는 이전 방식. 합성 지연이 0이지만
  장소명 등 문구를 바꿀 때마다 다시 만들어야 한다. live 모드의 폴백 캐시로도 쓰인다

### 셋업 (보드에서 1회)

```bash
bash setup_tts.sh          # venv + pip 부트스트랩 + supertonic-2 다운로드 + 스모크 테스트
# (선택) 합성 실패 시 폴백용 / cache 모드용 사전 렌더링 (약 20분, ~90MB)
.venv-tts/bin/python prerender_tts.py --all
```

디스크 여유가 600MB 이상 필요하다. 목소리를 바꾸려면 `ST_VOICE=F2 bash setup_tts.sh`
(M1~M5 / F1~F5). 더 고품질이 필요하면 `ST_MODEL=supertonic-3` (약 400MB, 31개 언어).

### 실행

```bash
python3 dump_monitor_jetson.py --source 0 --name cam01 --tts

# 장소명을 방송에 포함 (live 모드는 사전 작업 없이 바로 됨)
python3 dump_monitor_jetson.py --source 0 --name cam01 --tts --tts-location "정문 수거함 앞"

# 사전 렌더링본만 재생 (cache 모드; 장소명을 쓰려면 그 장소 전용 캐시를 먼저 만들어야 함)
.venv-tts/bin/python prerender_tts.py --all --location "정문 수거함 앞"
python3 dump_monitor_jetson.py --source 0 --name cam01 --tts --tts-mode cache --tts-location "정문 수거함 앞"
```

시작 로그에 `[TTS] 합성 워커 준비 완료 (N.Ns)`가 보이면 모델이 올라온 것이고, 이벤트 때는
`[TTS] 방송 (합성 N.Ns): …`로 실제 합성 시간이 찍힌다.

| 플래그 | 기본 | 의미 |
|---|---|---|
| `--tts` | off | 방송 활성화 |
| `--tts-mode` | `live` | `live`=이벤트마다 즉석 합성 / `cache`=사전 렌더링 wav만 재생 |
| `--tts-cooldown` | 12.0 | 방송 간 최소 간격(초). 문장이 약 9초라 이보다 짧게 두면 대기열이 쌓인다 |
| `--tts-repeat-window` | 30.0 | 같은 (색상, 종류) 조합 재방송 억제 시간(초) |
| `--tts-location` | 없음 | 방송에 넣을 장소명 |
| `--tts-cache` | `cache/tts` | 사전 렌더링 wav 폴더. live 모드에선 합성 실패 시 폴백용 |
| `--tts-no-runtime-synth` | off | (cache 모드) 캐시 미스 시 즉석 합성 대신 방송 생략 |
| `--tts-backend` | `supertonic` | `null`로 두면 무음 wav — 배선 점검용 |
| `--tts-model` / `--tts-voice` | `supertonic-2` / `M4` | 합성 모델 / 목소리 (M1~M5, F1~F5) |

### 색상 판정

`tts/color_naming.py`가 HSV 규칙으로 11색(빨간/주황/노란/초록/파란/보라/분홍/갈/흰/회/검은색)을
판정한다. 신규 의존성 없음. 배경 오염을 줄이기 위해 박스 안쪽 70%만 사용하고,
**이벤트 프레임 1장이 아니라 그 객체를 가장 확신했던 프레임의 크롭**(`WasteObject.best_crop`)에서
색을 뽑는다. 유채색 픽셀이 30% 미만이면 무채색(검정 봉투, 흰 스티로폼 등)으로 판정한다.

**야간에는 색상을 판정하지 않는다.** `night_mode(frame)`가 크롭을 딴 프레임 전체를 보고
- 평균 밝기(V) < 60 → `"저조도"`
- 유채색 픽셀(S≥45, V≥55) 비율 < 1% → `"적외선"` (IR 모드는 B=G=R이라 유채색이 사실상 없음.
  낮의 회색 콘크리트 바닥은 표지판·차량·초목 등으로 이 비율을 훌쩍 넘긴다)

둘 중 하나면 `color=null`로 기록·방송하고, 이벤트 로그 `night` 필드와 콘솔에 사유를 남긴다.
임계값은 `tts/color_naming.py` 상단 `NIGHT_V_MEAN`, `IR_CHROMA_FRAC`.

### 검증

```bash
python3 tests/test_color_naming.py      # 색상 판정 (모델 불필요)
python3 tests/test_mosaic.py            # 픽셀화/머리영역/YuNet 폴백 (YOLO 불필요)
python3 tests/test_tts_offline.py       # 문구/실시간 합성/캐시/재생큐/쿨다운/폴백 (모델·스피커 불필요)
python3 tests/test_monitor_tts_hook.py  # YOLO 스텁으로 낮/적외선/저조도/모자이크 발화→색상→방송 전 경로 (ultralytics 불필요)
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

### 방송 v2 (2026-09-19)

- 방송 문구를 이벤트 시점에 **즉석 합성**(`--tts-mode live`, 기본). 상주 합성 워커로 모델 로드는 1회,
  합성 실패 시 사전 렌더링본 폴백. 사전 렌더링 재생은 `--tts-mode cache`로 유지
- **야간 적외선/저조도 프레임에서는 색상 판정·방송 생략** (`night_mode`), `events.jsonl`에 `night` 필드 추가
- 보드에서 확인할 것: Orin Nano 실측 합성 지연(`[TTS] 방송 (합성 N.Ns)` 로그). 지연이 크면 cache 모드 권장

### 안면 모자이크 (2026-09-19)

- `--mosaic head|face`: 출력물의 사람 얼굴 픽셀화 (OpenCV YuNet, 추가 의존성 없음). 추론에는 영향 없음
- `--raw-snapshot`: 원본 스냅샷 별도 보관 옵션
