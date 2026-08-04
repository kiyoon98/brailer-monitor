#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

image="${BRAILER_IMAGE:-brailer-monitor:edge}"
container="${BRAILER_CONTAINER:-brailer-monitor-edge}"
threads="${BRAILER_CPU_THREADS:-4}"

if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "ERROR: container already exists: ${container}" >&2
  echo "Use 'docker start ${container}' or remove it with 'docker rm -f ${container}'." >&2
  exit 1
fi

for path in data models config output runs; do
  mkdir -p "${path}"
done

mounts=(
  --volume "${project_root}/data:/app/data"
  --volume "${project_root}/models:/app/models"
  --volume "${project_root}/config:/app/config"
  --volume "${project_root}/output:/app/output"
  --volume "${project_root}/runs:/app/runs"
)

for weight in sam2_t.pt yolo11n.pt yolo11n-seg.pt; do
  if [[ -f "${project_root}/${weight}" ]]; then
    mounts+=(--volume "${project_root}/${weight}:/app/${weight}:ro")
  else
    echo "WARNING: optional weight is missing: ${project_root}/${weight}" >&2
  fi
done

limits=()
if [[ -n "${BRAILER_MEMORY_LIMIT:-}" ]]; then
  limits+=(--memory "${BRAILER_MEMORY_LIMIT}")
fi
if [[ -n "${BRAILER_CPU_LIMIT:-}" ]]; then
  limits+=(--cpus "${BRAILER_CPU_LIMIT}")
fi

docker run --detach \
  --name "${container}" \
  --restart unless-stopped \
  --runtime nvidia \
  --gpus all \
  --network host \
  --ipc host \
  --read-only \
  --security-opt no-new-privileges:true \
  --tmpfs /tmp:size=2g,mode=1777 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --stop-timeout 45 \
  --env "TZ=${TZ:-Asia/Seoul}" \
  --env "BRAILER_DEVICE=${BRAILER_DEVICE:-0}" \
  --env "BRAILER_SAM_MODEL=/app/sam2_t.pt" \
  --env "OMP_NUM_THREADS=${threads}" \
  --env "OPENBLAS_NUM_THREADS=${threads}" \
  --env "MKL_NUM_THREADS=${threads}" \
  --env "NUMEXPR_NUM_THREADS=${threads}" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "MALLOC_ARENA_MAX=2" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "${mounts[@]}" \
  "${limits[@]}" \
  "${image}"

echo "Container started: ${container}"
echo "Web UI: http://127.0.0.1:8080"

