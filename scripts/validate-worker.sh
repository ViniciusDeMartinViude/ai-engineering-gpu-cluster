#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f worker.env ]] || { echo 'Create worker.env first'; exit 1; }
set -a; source worker.env; set +a
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
docker info >/dev/null
docker run --rm --gpus all --entrypoint nvidia-smi "${TRAINER_IMAGE:-ai-gpu-trainer:0.2}" --query-gpu=name,memory.total --format=csv,noheader
curl --fail --silent --show-error "${MANAGER_URL%/}/health"; echo
