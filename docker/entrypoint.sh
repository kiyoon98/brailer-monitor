#!/usr/bin/env bash
set -euo pipefail

umask 0002

runtime_dirs=(
  /app/data/dataset
  /app/data/pipeline
  /app/data/raw
  /app/data/web_jobs
  /app/data/cache
  /app/data/config-cache
  /app/data/config-cache/ultralytics/Ultralytics
  /app/data/home
  /app/models/library
  /app/output
  /app/runs
)

for path in "${runtime_dirs[@]}"; do
  mkdir -p "${path}"
done

for path in /app/data /app/models /app/output /app/runs; do
  if [[ ! -w "${path}" ]]; then
    echo "ERROR: runtime path is not writable by uid $(id -u): ${path}" >&2
    echo "Fix the host directory ownership or run with a matching APP_UID/APP_GID." >&2
    exit 73
  fi
done

if [[ "${1:-}" == "web" && ! -f "${BRAILER_SAM_MODEL:-/app/sam2_t.pt}" ]]; then
  echo "WARNING: SAM weight not mounted at ${BRAILER_SAM_MODEL:-/app/sam2_t.pt}; disable SAM or mount the file." >&2
fi

case "${1:-}" in
  analyze|detect-video|export|extract-frames|import-cvat|label|oneshot|sea-area|summarize|train|web)
    exec python3 -m brailer_monitor "$@"
    ;;
  "")
    exec python3 -m brailer_monitor web --host 0.0.0.0 --port 8080
    ;;
  *)
    exec "$@"
    ;;
esac
