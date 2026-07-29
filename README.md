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
│   └── *.engine                # ← Jetson 위에서 convert_tensorrt.sh 로 생성
├── dump_monitor_jetson.py      # 실행 모듈 (RTSP/웹캠/파일 입력)
├── convert_tensorrt.sh         # TensorRT 변환 (Jetson에서 실행)
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

## TensorRT 변환 — 반드시 Jetson 위에서

`.engine` 파일은 빌드한 GPU 전용이므로 PC에서 만들 수 없다. 보드에서 1회 실행:

```bash
chmod +x convert_tensorrt.sh
./convert_tensorrt.sh          # FP16, 모델당 수 분 소요
```

실패 시 스크립트 안의 `trtexec` 경로(방법 2) 주석을 해제해 ONNX에서 직접 빌드.
INT8이 필요하면 `half=True`를 `int8=True`로 바꾸고 캘리브레이션 데이터를 지정할 것 (정확도 검증 필수).

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
- `events.jsonl` — 이벤트 로그 (시각, 클래스, conf, 객체 id, 투기자 pid, bbox). append 방식이라 재시작해도 이어짐
- `event_NNNN.jpg` — 이벤트 증거 스냅샷 (빨간 박스 + 투기자 pid)
- `annotated.mp4` — 전체 주석 영상 (`--no-save-video`로 생략 가능)

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
