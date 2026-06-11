# Brailer Monitor

녹화 EM 영상에서 **브레일러(참치 운반망)** 를 자동 탐지하고, 어획량을 두 가지 방식으로 추정하는 프로그램입니다.

- **A. 기하학적**: 세그멘테이션 마스크 + 충만도 → 부피 → kg
- **B. 이벤트 기반**: 이송 완료 1회 × 표준 용량(kg)

## 요구 사항

- Python 3.11+
- Jetson Thor: 시스템 OpenCV, TensorRT (JetPack)
- 그 외: `pip install -r requirements.txt`

```bash
cd ~/Documents/brailer-monitor
pip install -r requirements.txt
pip install -e .
```

## 빠른 시작

```bash
# 단위 테스트
python -m unittest discover -s tests

# 샘플 이벤트 집계
python -m brailer_monitor summarize examples/sample_events.json --out output/summary.json

# 영상 분석 (커스텀 모델 없으면 yolo11n-seg.pt 폴백)
python -m brailer_monitor analyze /path/to/video.mp4 \
  --calibration config/calibration.json \
  --capacity config/standard_capacity.json \
  --model models/brailer_seg.engine \
  --out output/events.json \
  --csv output/events.csv
```

## 모델 학습

1. CVAT 또는 Roboflow로 EM 영상 프레임에 `brailer_loaded` polygon 라벨링 (200장 이상 권장)
2. YOLO 형식으로 `data/dataset/` 배치 후 `config/dataset.yaml` 경로 수정
3. 학습:

```bash
python -m brailer_monitor train --dataset config/dataset.yaml --epochs 100
```

4. TensorRT 변환 (Jetson):

```bash
chmod +x scripts/export_tensorrt.sh
./scripts/export_tensorrt.sh models/brailer_seg.pt
```

## 설정

| 파일 | 내용 |
|------|------|
| `config/calibration.json` | `cm_per_pixel`, 이송 구역 polygon, 하역 라인 |
| `config/standard_capacity.json` | 브레일러 1회 표준 용량(kg), 신뢰도 가중치 |
| `config/dataset.yaml` | YOLO 학습 데이터셋 경로 |

## CLI

| 명령 | 설명 |
|------|------|
| `analyze` | 영상 → 브레일러 이벤트 JSON/CSV |
| `summarize` | 이벤트 → 항차/영상별 어획량 집계 |
| `train` | YOLO11-seg 커스텀 학습 |
| `export` | `.pt` → TensorRT `.engine` |

## 프로젝트 구조

```
brailer_monitor/
  detector.py          # YOLO + TensorRT
  tracker.py           # ByteTrack 상태 관리
  transfer_counter.py  # B 방식 (supervision zone)
  volume_estimator.py  # A 방식 (부피·충만도)
  pipeline.py          # 영상 분석 파이프라인
  events.py            # BrailerEvent 모델
  aggregation.py       # 집계
  cli.py               # CLI
```

## 라이브러리

- **OpenCV** — 영상 I/O, 보정, 충만도 추정
- **Ultralytics YOLO11** — 탐지·세그멘테이션·추적·TensorRT export
- **supervision** — 이송 구역/라인 crossing
- **NumPy / SciPy** — 기하 계산
