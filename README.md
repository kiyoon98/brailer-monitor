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
- 하나 이상의 모델을 선택해 앙상블 탐지
- 파일 업로드, Lake 저장소 구간, 실시간 HLS 스트림 탐지
- 파일/Lake 영상이 전체적으로 확실히 어두울 때 탐지 전 자동 skip
- SegFormer 의미 분할과 기존 OpenCV 보정을 결합한 선택형 바다 영역/조우 상태 분석
- Lake 저장소 media/vessel/stream 선택 및 5분 단위 시작 분 후보, 초 suffix 지정
- 실시간 스트림 기본 주소: `http://127.0.0.1:8081/live_04.m3u8`
- 스트림 탐지 중 최신 탐지 프레임 오버레이 표시
- HLS 리셋/끊김 시 브라우저 재연결 및 worker 스트림 재연결
- 탐지 타임라인 실시간 갱신
- 탐지 구간 8초 이내 병합 및 선택형 후처리 제거 조건
- 탐지 결과 저장/불러오기
- 외부 문서용 탐지 리포트 생성 및 HTML 전체 타임라인 표시
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

학습 완료 모델은 `models/library/` 아래에 자동 저장됩니다. 모델 라이브러리에서:

- **사용**: 기본/활성 모델을 지정합니다.
- **탐지 체크박스**: 실제 탐지에 사용할 모델을 하나 이상 선택합니다.
- **프레임**: 해당 모델 저장 시점의 학습 프레임을 확인합니다.
- **이름변경/삭제**: 모델 라이브러리 항목을 관리합니다.

여러 모델의 **탐지** 체크를 켜면 앙상블 탐지가 실행됩니다.

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
- `Confidence`: 탐지 confidence threshold. 기본값은 `0.6`입니다.
- `추론 크기`: YOLO `imgsz`
- `탐지 제외 여백(%)`: 상/우/하/좌 화면 가장자리에서 제외할 비율. 기본값은 모두 `15`이며, 이 경우 좌우 `15~85%`, 상하 `15~85%` 영역의 탐지만 결과로 인정합니다.
- `정밀 마스크(SAM)`: bbox 기반 추가 SAM mask 생성. 기본값은 켜짐입니다.
- `어두운 영상 건너뛰기(파일/Lake)`: 영상 여러 지점을 사전 샘플링해 전부 확실히 어두우면 YOLO 추론 없이 skip

모델 선택:

- 모델 라이브러리의 **탐지** 체크박스를 하나 이상 켜야 탐지를 시작할 수 있습니다.
- 기본 선택은 `brailer` 클래스가 있는 모델 전체이며, `two_03032220`/`two-03032220` 모델은 제외됩니다.
- 여러 모델을 선택하면 각 모델이 독립적으로 추론한 뒤 같은 클래스의 bbox를 IoU 또는 포함 겹침 기준으로 병합합니다.
- 병합된 결과의 bbox/mask는 병합 후보 중 bbox가 가장 큰 탐지를 사용하고, confidence와 대표 모델명은 가장 높은 confidence 탐지를 사용합니다.
- preview 이미지, 스트림 오버레이, manifest에는 탐지에 사용된 모델명/앙상블 모델명이 함께 기록됩니다.

Lake 저장소:

- media에는 `seibu`, `pharostar` 등을 선택할 수 있습니다.
- 선박에는 `JJR-102283`, `JJR-131066`, `JJR-151069` 등을 선택할 수 있습니다.
- 시작 분 후보는 5분 단위 묶음의 시작 offset입니다. 예를 들어 `02`를 지정하면 `02, 07, 12, ... 57`분 영상 후보를 확인합니다.
- 초 suffix 기본값은 `16`이며 필요하면 직접 지정할 수 있습니다.

실시간 스트림:

- 기본 주소는 `http://127.0.0.1:8081/live_04.m3u8`
- 스트림 preview 위에 최신 탐지 프레임이 오버레이됩니다.
- HLS playlist/segment가 리셋되면 preview는 자동 재연결을 시도합니다.
- worker는 OpenCV read 실패가 지속되면 `VideoCapture`를 다시 열어 탐지를 계속합니다.

어두운 영상 skip:

- 파일 업로드와 Lake 저장소 탐지에 적용됩니다.
- 영상 앞/중간/뒤 여러 프레임을 샘플링합니다.
- 모든 샘플이 낮은 밝기/낮은 대비 기준을 만족할 때만 전체 영상을 `dark_video`로 skip합니다.
- 일부 샘플이라도 충분히 밝으면 기존처럼 탐지를 실행합니다.
- skip된 job의 `detections.json`에는 `skipped`, `skip_reason`, `dark_video_assessment`가 기록됩니다.

CLI로 단일 영상 탐지:

```bash
python -m brailer_monitor detect-video data/raw/video.mp4 \
  --model models/library/<model-id>/weights.pt \
  --out output/detect \
  --frame-stride 5 \
  --confidence 0.6 \
  --sea-ratio --sea-analysis-interval-sec 5 \
  --roi-top 0.15 --roi-right 0.15 --roi-bottom 0.15 --roi-left 0.15 \
  --segmentation auto
```

파일 업로드, Lake 저장소, 실시간 스트림 탐지의 **바다 분석 간격(초)**은 공통으로 적용됩니다. 기본값은 `5`초이며 `0`은 파이프라인이 처리하는 모든 프레임, 최댓값 `300`은 5분마다 한 번을 뜻합니다. 녹화 파일과 Lake 영상은 영상 타임스탬프를, 실시간 스트림은 실제 경과시간을 기준으로 다음 분석 프레임을 선택합니다.

웹에서 **바다 영역만 분석(객체 탐지 안 함)**을 켜면 YOLO 앙상블과 SAM을 로드하거나 실행하지 않고 바다 영역 및 조우 상태만 계산합니다. 이 모드는 학습된 객체 탐지 모델을 선택하지 않아도 파일, Lake 저장소, 실시간 스트림에 사용할 수 있습니다. CLI에서는 `detect-video <video> --sea-only --sea-analysis-interval-sec 5`로 같은 모드를 실행합니다.

바다 영역만 독립적으로 검사할 때는 `sea-area` 명령을 사용합니다. `--frame-stride N`은 N프레임마다 한 번씩 바다 영역을 계산한다는 뜻입니다. 기본 엔진은 사전 학습된 `nvidia/segformer-b0-finetuned-ade-512-512`의 의미 분할 결과와 기존 HSV/Lab/GrabCut 결과를 결합하는 `hybrid`입니다. 사용자가 별도 학습할 필요는 없으며 최초 실행 시 고정된 model revision을 Hugging Face cache로 내려받습니다. 모델이나 네트워크를 사용할 수 없으면 작업을 중단하지 않고 기존 OpenCV 방식으로 자동 전환합니다.

Lake 저장소 시간 구간 검사:

```bash
python -m brailer_monitor sea-area storage \
  --start 2026-01-28T03 --end 2026-01-28T04 \
  --media lake_win --year-folder 2026_decrypted \
  --vessel JJR-102283 --camera-stream stream04 \
  --minute-offsets 0,1,2,3,4 --second-suffixes 16 \
  --frame-stride 30
```

저장소의 영상 URL 하나만 검사할 수도 있습니다.

```bash
python -m brailer_monitor sea-area storage \
  --url http://10.2.10.158:8041/media/em_data/lake_win/2026_decrypted/01/28/03/JJR-102283_stream04_260128_030016.mp4 \
  --frame-stride 30
```

실시간 스트림 검사:

```bash
python -m brailer_monitor sea-area stream \
  --url http://127.0.0.1:8081/live_04.m3u8 \
  --frame-stride 30 \
  --sea-engine hybrid --device cpu
```

스트림 검사는 기본적으로 `Ctrl+C`까지 계속됩니다. `--max-samples 20` 또는 `--duration-sec 120`을 지정하면 샘플 수나 실행 시간으로 종료할 수 있습니다. `--json-lines`는 표준 출력을 JSON Lines로 바꾸며, `--jsonl-out output/sea-area.jsonl`을 추가하면 화면 출력과 함께 결과를 파일에 즉시 기록합니다. 저장소 범위에서는 `--max-samples`가 영상별 최대 샘플 수로 적용됩니다. 의미 분할 없이 이전 계산만 사용하려면 `--sea-engine legacy`를 지정합니다. CUDA 컨텍스트를 YOLO/SAM과 분리하기 위해 웹 탐지의 바다 의미 분할과 독립 명령의 기본 장치는 CPU입니다. 독립 명령에서만 GPU를 시험하려면 `--device 0`을 명시합니다.

하이브리드 분석은 다음 상태를 출력합니다.

- `calibrating`: 첫 60초/15개 이상 샘플로 카메라별 평상시 기준선을 만드는 중
- `open_sea`: 평상시 바다 상태
- `encounter`: 바다 비율이 기준선보다 20% 이상 감소하거나 선박 영역이 기준선보다 0.5% 이상 증가한 상태가 10초 지속됨
- `unknown`: 영상이 너무 어둡거나 의미 분할 신뢰도가 낮아 판정을 보류함
- `departure`: 선박 증가분이 0.1% 미만이고 바다 감소폭이 10% 이하인 상태가 30초 지속될 때 발생하는 이탈 이벤트

카메라에 항상 보이는 자선 갑판/선체가 `ship`으로 분류될 수 있으므로 절대 선박 비율이 아닌 카메라별 기준선 대비 증가분을 조우 판정에 사용합니다. 기준과 가중치는 [`config/sea_area.json`](config/sea_area.json)에서 조정할 수 있습니다.

각 처리 프레임은 다음 형식으로 즉시 출력됩니다.

```text
[JJR-102283_stream04_260128_030016.mp4] frame=30 time=00:00:02.000 absolute=2026-01-28 03:00:02.000 sea=42.50% state=open_sea confidence=0.812 vessel=1.20% vessel_increase=+0.08% area_px=391680 horizon_y=128 roi=[0, 128, 1280, 720] processing_ms=18.4
```

탐지 결과는 job별로 `data/pipeline/detect_jobs/<job_id>/`에 저장됩니다.

- `detections.json`
- `previews/frame_*.jpg`
- `events.jsonl`
- `progress.json`
- `worker.log`

웹에서 **바다 영역 계산**을 켜면 진행 스케일과 함께 분석 신뢰도, 원시 선박 비율, 기준선 대비 선박 증가분, `calibrating/open_sea/encounter/unknown` 상태가 실시간으로 표시됩니다. 완료된 `detections.json`과 외부 HTML/JSON 리포트에는 바다 분석 요약, 조우 시작/이탈 시각, 구간 최소 바다 비율, 최대 선박 비율, 판정 불가 비율이 포함됩니다.

### 4. 탐지 타임라인

탐지 중 객체가 발견될 때마다 타임라인이 실시간 갱신됩니다.

타임라인 기능:

- 탐지 구간 목록
- 시간축 표시/확대/축소
- 썸네일 클릭 확대
- 더블클릭으로 세그먼트 프레임 전체 보기
- 8초 이내 구간 병합
- 위치/크기/세로형 빈 그물/3-4초 정지/시간 고립 및 1초 이하 3프레임 burst/색상 이상 탐지 제거 후처리
- 현재 탐지 결과 저장
- 저장된 결과 불러오기
- 외부 문서용 리포트 생성

타임라인 데이터는 `data/pipeline/detect_timeline.json`에 저장됩니다.

외부 문서 리포트:

- HTML, CSV, JSON 파일을 `data/pipeline/reports/` 아래에 생성합니다.
- HTML 리포트에는 분석 영상 수, 탐지 프레임 수, 연속 탐지 구간 수, 최장 연속 시간 요약이 포함됩니다.
- HTML 리포트의 **전체 타임라인**은 모든 탐지 구간을 시간 비율로 배치한 막대형 개요입니다.
- 타임라인 막대는 클래스별 색상으로 표시되며, 대표 프레임 preview가 있으면 클릭해서 열 수 있습니다.
- 이어서 긴 구간 순, 높은 confidence 순, 전체 구간 테이블을 제공합니다.

## 다중 모델 앙상블 탐지

웹 탐지는 하나 이상의 모델을 사용할 수 있습니다. 앙상블은 다음 방식으로 동작합니다.

1. 선택된 각 모델이 같은 프레임을 독립적으로 추론합니다.
2. 같은 클래스의 bbox끼리 IoU 또는 포함 겹침 기준으로 비교합니다.
3. 기준 이상으로 겹치는 bbox를 하나의 탐지로 병합합니다.
4. 대표 bbox/mask는 병합 후보 중 bbox가 가장 큰 탐지를 사용하고, confidence와 대표 모델명은 가장 높은 confidence 탐지를 사용합니다.
5. 결과에는 대표 모델명과 병합에 참여한 모델명 목록이 기록됩니다.

현재 기본 IoU 기준은 `0.5`이고, 작은 bbox가 큰 bbox에 대부분 포함되는 경우도 같은 객체로 병합합니다. `brailer`/`brailers`는 같은 클래스 alias로 취급합니다. 클래스가 다르면 bbox가 겹쳐도 병합하지 않습니다.

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
| `sea-area` | Lake 저장소 또는 실시간 스트림의 프레임별 바다 영역 계산 |
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
| `brailer_monitor/sea_area_analysis.py` | 하이브리드 바다 의미 분할과 조우/이탈 상태 추적 |
| `config/sea_area.json` | 바다 분석 모델, 융합 가중치, 상태 전이 기준 |
| `brailer_monitor/detector.py` | YOLO wrapper 및 fallback |
| `brailer_monitor/detect_timeline.py` | 누적 탐지 타임라인 |
| `brailer_monitor/detect_report.py` | HTML/CSV/JSON 외부 리포트 생성 |
| `brailer_monitor/lake_video_source.py` | Lake 원격 영상 후보 생성/확인/다운로드 |
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
- 타임라인 구간 병합 기본값은 8초입니다.
- `pipeline_state.json`은 고유 임시 파일을 거쳐 atomic replace로 저장해, UI 폴링과 탐지 스레드가 동시에 상태를 저장해도 임시 파일 rename 충돌이 나지 않도록 합니다.
