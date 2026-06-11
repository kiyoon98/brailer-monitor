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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 빠른 시작

```bash
# 단위 테스트
python -m unittest discover -s tests

# 1) 브레일러 구간 탐지 + 프레임 추출
python -m brailer_monitor extract-frames data/raw/JJR-102283_stream04_260310_202016.mp4 \
  --out data/dataset/staging/images \
  --manifest data/dataset/segments.json \
  --preview data/dataset/staging/preview

# 2) SAM 자동 라벨링 (YOLO-seg polygon) + train/val 분할
python -m brailer_monitor label --split

# 또는 한 번에 (extract + label)
python scripts/label_brailer.py --video data/raw/JJR-102283_stream04_260310_202016.mp4

# 샘플 이벤트 집계
python -m brailer_monitor summarize examples/sample_events.json --out output/summary.json

# 영상 분석 (학습된 모델 사용)
python -m brailer_monitor analyze data/raw/JJR-102283_stream04_260310_202016.mp4 \
  --calibration config/calibration.json \
  --capacity config/standard_capacity.json \
  --model models/brailer_seg.engine \
  --out output/events.json \
  --csv output/events.csv
```

## 라벨링 (자동)

제공된 5분 EM 영상에서 SAM 기반 `brailer_loaded` 라벨을 생성합니다:

```bash
# 영상 다운로드 (이미 data/raw/ 에 있으면 생략)
curl -L -o data/raw/JJR-102283_stream04_260310_202016.mp4 \
  "http://10.2.10.158:8041/media/lake_win/2026_decrypted/03/10/20/JJR-102283_stream04_260310_202016.mp4"

# 1초 간격 프레임 추출 + SAM polygon 라벨링
python scripts/label_brailer.py --frame-stride 15 --sam-model models/mobile_sam.pt
```

결과:
- `data/dataset/images/train` (206장), `images/val` (51장)
- `data/dataset/labels/train|val` — YOLO segmentation polygon 형식
- `data/dataset/label_manifest.json` — 프레임별 메타데이터
- `data/dataset/staging/preview/` — 라벨 검수용 오버레이 이미지

## 모델 학습

1. (완료) 위 자동 라벨링으로 `data/dataset/` 구성
2. 또는 CVAT/Roboflow로 수동 보정 후 동일 경로에 반영
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
| `extract-frames` | 브레일러 구간 탐지 → 해당 구간만 프레임 추출 |
| `label` | 추출된 프레임에 SAM polygon 라벨 생성 |
| `analyze` | 영상 → 브레일러 이벤트 JSON/CSV |
| `summarize` | 이벤트 → 항차/영상별 어획량 집계 |
| `train` | YOLO11-seg 커스텀 학습 |
| `export` | `.pt` → TensorRT `.engine` |
| `web` | 웹 뷰어 실행 (업로드 · 레이블 검수) |

## 웹 어노테이터 (수동 polygon 레이블)

자동 SAM 라벨은 부정확하므로, **사용자가 직접 다각형을 그려** `brailer_loaded` 레이블을 입력합니다.

```bash
pip install -e ".[web,opencv]"
python -m brailer_monitor web --port 8080
```

브라우저 http://localhost:8080:

1. **기본 영상 열기** 또는 영상 업로드
2. 시간(초) 입력 후 **프레임 가져오기** — 북마크 35s, 207s 제공
3. 캔버스 클릭으로 브레일러 외곽 polygon 점 추가
4. **레이블 저장** (Ctrl+S) — YOLO-seg 형식으로 저장
5. **학습용 dataset으로보내기** — `data/dataset/images/train`에 복사

잘못된 자동 라벨 삭제:

```bash
python scripts/clear_auto_labels.py
```

## 프로젝트 구조

```
brailer_monitor/
  frame_extractor.py   # 브레일러 구간 탐지 + 프레임 추출
  labeling.py          # SAM 자동 라벨링
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
