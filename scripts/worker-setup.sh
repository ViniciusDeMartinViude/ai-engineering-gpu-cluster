#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
docker build -t ai-gpu-trainer:0.2 -f trainer/Dockerfile .
echo 'Copy worker.env.example to worker.env and configure manager IP/key/node, then run scripts/start-worker.sh'
