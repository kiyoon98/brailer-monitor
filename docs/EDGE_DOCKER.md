# Brailer Monitor Edge Docker

이 구성은 현재 개발 호스트와 같은 NVIDIA Jetson AGX Thor 계열을 대상으로 한다.

- Architecture: `aarch64`
- Jetson Linux: L4T R38
- CUDA: 13.x
- Default base: `nvcr.io/nvidia/pytorch:25.08-py3`
- Application port: `8080`
- Runtime user: uid/gid `1000:1000`

## 설계 원칙

`data/`는 현재 수백 GB까지 커질 수 있으므로 이미지에 넣지 않는다. 모델, 탐지 결과,
학습 결과와 설정도 bind mount로 분리한다. 애플리케이션 이미지는 코드와 Python
의존성만 포함한다.

컨테이너는 다음 제한을 기본으로 사용한다.

- Uvicorn worker 1개
- NVIDIA GPU 1개 사용
- CPU 라이브러리 thread 기본 4개
- read-only root filesystem
- `/tmp` 2GB tmpfs
- Tini가 PID 1로 동작하며 자식 탐지 프로세스까지 종료 신호 전달
- host network 사용

host network를 사용하는 이유는 기본 스트림 URL
`http://127.0.0.1:8081/live_04.m3u8`이 Jetson 호스트에서 실행되는 스트림
서버를 그대로 가리키게 하기 위해서다.

## 사전 조건

```bash
uname -m
cat /etc/nv_tegra_release
docker info --format '{{json .Runtimes}}'
```

출력에서 `aarch64`, L4T R38 및 `nvidia` runtime을 확인한다.

현재 배포 디렉터리에는 다음 항목이 있어야 한다.

```text
brailer_monitor/
config/
data/
models/
output/
runs/
sam2_t.pt
yolo11n.pt
yolo11n-seg.pt
```

학습을 하지 않으면 `yolo11n*.pt`는 생략할 수 있다. SAM을 사용하지 않으면
`sam2_t.pt`도 생략할 수 있다. 운영 모델은
`models/library/<model-id>/weights.pt`와 `meta.json`을 함께 옮긴다.

## 이미지 빌드

```bash
./scripts/docker-edge-build.sh
```

기본 태그는 `brailer-monitor:edge`이다. 다른 태그나 NVIDIA base를 사용할 때:

```bash
BRAILER_IMAGE=registry.example/brailer-monitor:edge-r38 \
BRAILER_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 \
./scripts/docker-edge-build.sh
```

빌드가 끝나면 스크립트가 Thor용 `sm_110` 포함 여부와 실제 CUDA 행렬 연산을
자동으로 검증한다. NVIDIA base image는 대상 JetPack/L4T뿐 아니라 대상 GPU의
CUDA architecture를 명시적으로 지원하는 태그만 사용한다. GPU가 없는 별도 빌드
서버에서 이미지만 만들 때는 `BRAILER_SKIP_GPU_CHECK=1`을 지정할 수 있지만,
대상 장비에서 아래 GPU 확인을 반드시 수행해야 한다.

## 컨테이너 시작

Docker Compose plugin이 있는 환경:

```bash
cp .env.edge.example .env.edge
docker compose --env-file .env.edge -f compose.edge.yaml up -d --build
```

Compose plugin이 없는 환경:

```bash
./scripts/docker-edge-run.sh
```

브라우저에서 `http://<jetson-ip>:8080`을 연다.

## 상태 확인

```bash
docker ps --filter name=brailer-monitor-edge
docker logs --tail 200 -f brailer-monitor-edge
docker inspect brailer-monitor-edge --format '{{json .State.Health}}'
curl -f http://127.0.0.1:8080/
```

GPU 확인:

```bash
docker exec brailer-monitor-edge python3 -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

OpenCV와 애플리케이션 확인:

```bash
docker exec brailer-monitor-edge python3 -c \
  "import cv2, brailer_monitor; print(cv2.__version__, brailer_monitor.__version__)"
```

## 중지와 재시작

탐지 또는 학습 중이라면 웹 UI에서 먼저 작업을 중지한다.

```bash
docker stop -t 45 brailer-monitor-edge
docker start brailer-monitor-edge
```

컨테이너를 다시 만들더라도 bind mount의 `data/`, `models/`, `output/`,
`runs/`, `config/`는 유지된다.

## 제한된 자원 설정

실행 스크립트는 선택적인 hard limit를 지원한다.

```bash
BRAILER_CPU_THREADS=4 \
BRAILER_CPU_LIMIT=6 \
BRAILER_MEMORY_LIMIT=16g \
./scripts/docker-edge-run.sh
```

메모리 제한을 너무 낮게 설정하면 멀티모델, SAM 또는 학습이 OOM으로 종료된다.
먼저 제한 없이 실제 peak memory를 측정한 뒤 limit를 설정하는 것이 안전하다.

엣지에서 처리량을 우선할 때 권장 순서:

1. 탐지 모델을 검증된 1~2개로 줄인다.
2. SAM이 필요하지 않은 작업은 정밀 마스크를 끈다.
3. 바다 영역 분석이 필요하지 않으면 끈다.
4. 짧은 객체를 놓치지 않는 범위에서 frame stride를 늘린다.
5. 추론 크기 416을 기준으로 시작한다.
6. 학습은 가능하면 별도 학습 장비에서 하고 `weights.pt`만 엣지에 배포한다.

## 이미지 이동

Registry 사용을 권장한다.

```bash
docker tag brailer-monitor:edge registry.example/brailer-monitor:edge-r38
docker push registry.example/brailer-monitor:edge-r38
```

폐쇄망에서는 전체 NVIDIA base layer를 포함하므로 파일이 매우 커질 수 있다.

```bash
docker save brailer-monitor:edge | zstd -T0 -10 -o brailer-monitor-edge.tar.zst
zstd -dc brailer-monitor-edge.tar.zst | docker load
```

## 문제 해결

### GPU를 찾지 못함

`docker info`에 `nvidia` runtime이 있는지 확인하고, 실행 시
`--runtime nvidia --gpus all`이 포함되었는지 확인한다.

### 쓰기 권한 오류

이미지의 runtime uid/gid는 `1000:1000`이다.

```bash
sudo chown -R 1000:1000 data models output runs config
```

### SAM 파일을 찾지 못함

`sam2_t.pt`를 `/app/sam2_t.pt`로 mount하거나 UI에서 SAM을 끈다.

### 모델이 보이지 않음

다음 구조를 확인한다.

```text
models/library/<model-id>/weights.pt
models/library/<model-id>/meta.json
```

### 이미지가 너무 큼

PyTorch, CUDA, cuDNN이 포함된 NVIDIA base가 대부분의 크기를 차지한다.
애플리케이션 데이터와 모델은 이미 image layer에서 제외되어 있다.
더 줄이려면 PyTorch 대신 TensorRT engine 전용 추론 runtime을 별도로 구현해야 한다.
