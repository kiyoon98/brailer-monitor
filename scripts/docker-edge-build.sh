#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

image="${BRAILER_IMAGE:-brailer-monitor:edge}"
base_image="${BRAILER_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
app_uid="${APP_UID:-1000}"
app_gid="${APP_GID:-1000}"
target_cuda_arch="${BRAILER_TARGET_CUDA_ARCH:-sm_110}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "WARNING: the default base image is intended for Jetson ARM64 (aarch64)." >&2
fi

docker build \
  --file docker/Dockerfile.edge \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "APP_UID=${app_uid}" \
  --build-arg "APP_GID=${app_gid}" \
  --tag "${image}" \
  .

docker image inspect "${image}" \
  --format 'built image={{.RepoTags}} size={{.Size}} architecture={{.Architecture}}'

if [[ "${BRAILER_SKIP_GPU_CHECK:-0}" != "1" ]]; then
  docker run --rm \
    --runtime nvidia \
    --gpus all \
    --env "BRAILER_TARGET_CUDA_ARCH=${target_cuda_arch}" \
    "${image}" \
    python3 -c '
import os
import torch

target = os.environ["BRAILER_TARGET_CUDA_ARCH"]
arches = torch.cuda.get_arch_list()
assert torch.cuda.is_available(), "CUDA is not available in the built image"
assert target in arches, f"PyTorch does not include {target}: {arches}"
x = torch.ones((256, 256), device="cuda")
value = (x @ x).sum().item()
assert value == 16777216.0, f"unexpected CUDA result: {value}"
print(
    f"gpu check=ok device={torch.cuda.get_device_name(0)} "
    f"capability={torch.cuda.get_device_capability(0)} target={target}"
)
'
else
  echo "WARNING: skipped GPU compatibility check (BRAILER_SKIP_GPU_CHECK=1)." >&2
fi
