#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f worker.env ];then echo 'Copy worker.env.example to worker.env and edit it';exit 1;fi
set -a
source ./worker.env
set +a
exec .venv/bin/python -m worker.agent
