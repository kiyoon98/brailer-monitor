# Brailer Monitor

CVAT으로 라벨링한 브레일러/윈치 영상 데이터를 YOLO 학습 데이터셋으로 가져오고, 학습된 모델로 녹화 영상이나 실시간 HLS 스트림에서 객체를 탐지하는 웹 기반 파이프라인입니다.

현재 기본 화면은 `http://localhost:8080`의 **YOLO 학습 → 비디오 탐지 → 탐지 타임라인 → CVAT Import** 흐름입니다.

## 주요 기능

- CVAT 1.1 `.zip` 또는 `annotations.xml` import
- CVAT export에 이미지가 없는 경우 원본 영상에서 annotation 프레임 추출
- YOLO detect/segment 데이터셋 자동 구성 및 `config/dataset.yaml` 생성
- YOLO 학습, 중지, 진행률 표시
- 학습 완료 모델 자동 저장 및 모델 라이브러리 관리
- 모델별 학습 프레임 미리보기
- 파일 업로드, Lake 저장소 구간, 실시간 HLS 스트림 탐지
- 실시간 스트림 기본 주소: `http://127.0.0.1:8081/live_04.m3u8`
- 스트림 탐지 중 최신 탐지 프레임 오버레이 표시
- HLS 리셋/끊김 시 브라우저 재연결 및 worker 스트림 재연결
- 탐지 타임라인 실시간 갱신
- 탐지 구간 10초 이내 병합
- 탐지 결과 저장/불러오기
- 외부 문서용 탐지 리포트 생성
- segmentation 후처리 오류 발생 시 bbox detect fallback

## 요구 사항

- Python 3.11 이상
- OpenCV
- Ultralytics YOLO
- FastAPI / Uvicorn
- GPU 사용 시 CUDA 또는 Jetson 환경 권장

설치:

```bash
cd ~/Documents/brailer-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[web,opencv]"
```

Node.js는 개발 중 JS 문법 검사용으로만 필요합니다.

```bash
node --check brailer_monitor/web/static/detect.js
```

## 웹 실행

```bash
source .venv/bin/activate
python -m brailer_monitor web --port 8080
```

브라우저:

```text
http://localhost:8080
```

현재 서버가 백그라운드에서 떠 있는지 확인:

```bash
ps -eo pid,ppid,pgid,sid,stat,etime,args | rg 'brailer_monitor web --port 8080'
```

8080 포트 서버를 재시작해야 할 때:

```bash
kill -TERM -<PGID>
setsid -f .venv/bin/python -m brailer_monitor web --port 8080 \
  > /tmp/brailer-monitor-web-8080.log 2>&1
```

## 기본 사용 흐름

### 1. CVAT 데이터 Import

화면 하단의 **CVAT 데이터 Import** 섹션에서 CVAT export 파일을 올립니다.

- `annotations.xml`
- CVAT 1.1 `.zip`
- export 안에 이미지가 없으면 원본 영상도 함께 업로드

Import가 끝나면:

- `data/dataset/images/train`
- `data/dataset/images/val`
- `data/dataset/labels/train`
- `data/dataset/labels/val`
- `data/dataset/import_meta.json`
- `config/dataset.yaml`

이 갱신됩니다.

CLI로도 import할 수 있습니다.

```bash
python -m brailer_monitor import-cvat data/dataset/annotations.xml \
  --video data/raw/source.mp4 \
  --dataset-root data/dataset
```

### 2. YOLO 학습

웹의 **YOLO 학습** 섹션에서 `Epochs`, `Batch`, `학습 크기(imgsz)`를 지정하고 학습을 시작합니다.

학습 완료 모델은 `models/library/` 아래에 자동 저장되며, 목록에서 사용할 모델을 선택할 수 있습니다. 모델을 클릭하면 해당 모델 저장 시점의 학습 프레임을 확인할 수 있습니다.

CLI 학습:

```bash
python -m brailer_monitor train \
  --dataset config/dataset.yaml \
  --epochs 50 \
  --batch 4 \
  --imgsz 416
```

### 3. 비디오 탐지

웹의 **비디오 탐지** 섹션에서 소스를 선택합니다.

- 파일 업로드: 여러 영상 선택 가능
- Lake 저장소: 날짜/시간 구간으로 원격 영상 탐색 후 배치 탐지
- 실시간 스트림: HLS 주소를 입력해 현재 스트림 탐지

공통 옵션:

- `프레임 간격`: 몇 프레임마다 추론할지 지정
- `Confidence`: 탐지 confidence threshold
- `추론 크기`: YOLO `imgsz`
- `정밀 마스크(SAM)`: bbox 기반 추가 SAM mask 생성

실시간 스트림:

- 기본 주소는 `http://127.0.0.1:8081/live_04.m3u8`
- 스트림 preview 위에 최신 탐지 프레임이 오버레이됩니다.
- HLS playlist/segment가 리셋되면 preview는 자동 재연결을 시도합니다.
- worker는 OpenCV read 실패가 지속되면 `VideoCapture`를 다시 열어 탐지를 계속합니다.

CLI로 단일 영상 탐지:

```bash
python -m brailer_monitor detect-video data/raw/video.mp4 \
  --model models/library/<model-id>/weights.pt \
  --out output/detect \
  --frame-stride 5 \
  --confidence 0.6 \
  --segmentation auto
```

탐지 결과는 job별로 `data/pipeline/detect_jobs/<job_id>/`에 저장됩니다.

- `detections.json`
- `previews/frame_*.jpg`
- `events.jsonl`
- `progress.json`
- `worker.log`

### 4. 탐지 타임라인

탐지 중 객체가 발견될 때마다 타임라인이 실시간 갱신됩니다.

타임라인 기능:

- 탐지 구간 목록
- 시간축 표시/확대/축소
- 썸네일 클릭 확대
- 더블클릭으로 세그먼트 프레임 전체 보기
- 10초 이내 구간 병합
- 현재 탐지 결과 저장
- 저장된 결과 불러오기
- 외부 문서용 리포트 생성

타임라인 데이터는 `data/pipeline/detect_timeline.json`에 저장됩니다.

## 모델과 segmentation fallback

데이터셋이 polygon 기반이면 segment 모델을 학습하고, box 기반이면 detect 모델을 학습합니다. 탐지 시에는 import metadata와 모델명을 기준으로 segmentation 여부를 자동 판단합니다.

일부 모델/프레임 조합에서 Ultralytics segmentation 후처리가 다음 오류를 낼 수 있습니다.

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (300x0 and 32x6656)
```

이 경우 현재 detector는 mask 후처리를 우회하고 bbox detect fallback을 사용합니다. fallback에서는 segmentation mask/polygon 대신 bbox 결과만 사용됩니다.

## CLI 명령

| 명령 | 설명 |
|------|------|
| `web` | 웹 파이프라인 실행 |
| `import-cvat` | CVAT 1.1 export를 YOLO 데이터셋으로 변환 |
| `train` | `config/dataset.yaml` 기준 YOLO 학습 |
| `detect-video` | 단일 영상 탐지 및 manifest 생성 |
| `extract-frames` | 브레일러 구간 탐지 후 프레임 추출 |
| `label` | 추출 프레임에 SAM polygon 라벨 생성 |
| `oneshot` | 참조 라벨 기반 one-shot 탐지 |
| `analyze` | 이벤트 기반 영상 분석 |
| `summarize` | 이벤트 JSON 집계 |
| `export` | `.pt` 모델을 TensorRT `.engine`으로 변환 |

설치 후 console script도 사용할 수 있습니다.

```bash
brailer-monitor web --port 8080
```

## 주요 경로

| 경로 | 내용 |
|------|------|
| `brailer_monitor/web/static/detect.html` | 메인 웹 UI |
| `brailer_monitor/web/static/detect.js` | 웹 UI 로직 |
| `brailer_monitor/web/app.py` | FastAPI endpoint |
| `brailer_monitor/web/detect_pipeline.py` | import/train/detect job 관리 |
| `brailer_monitor/video_detect.py` | 영상/스트림 YOLO 추론 |
| `brailer_monitor/detector.py` | YOLO wrapper 및 fallback |
| `brailer_monitor/detect_timeline.py` | 누적 탐지 타임라인 |
| `brailer_monitor/model_library.py` | 학습 모델 저장/관리 |
| `data/dataset/` | YOLO 학습 데이터셋 |
| `data/pipeline/` | 웹 pipeline 상태, 탐지 job, 저장 결과, 리포트 |
| `models/library/` | 학습 완료 모델 라이브러리 |
| `config/dataset.yaml` | YOLO 학습 데이터셋 설정 |

## 테스트

```bash
source .venv/bin/activate
python -m unittest discover -s tests
node --check brailer_monitor/web/static/detect.js
```

## 개발 메모

- `detect_worker.py`는 탐지를 별도 subprocess에서 실행합니다. CUDA 오류가 발생해도 웹 서버 프로세스가 같이 망가지지 않도록 분리되어 있습니다.
- 실시간 스트림 중지 요청은 job 디렉터리의 `stop.txt`로 전달됩니다.
- 탐지 진행률은 `progress.json`, 실시간 타임라인 이벤트는 `events.jsonl`로 parent process에 전달됩니다.
- 완료 시 `detections.json`을 최종 병합하며, 같은 job의 임시 타임라인 이벤트는 중복되지 않도록 교체됩니다.
- 타임라인 구간 병합 기본값은 10초입니다.
