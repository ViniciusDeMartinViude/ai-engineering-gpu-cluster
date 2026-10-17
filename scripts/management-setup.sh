#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null; then echo 'Install Docker Desktop + WSL integration first'; exit 1; fi
mkdir -p state/results state/datasets
if [ ! -f .env ]; then
  cp .env.example .env
  python3 - <<'PY2'
from pathlib import Path
import secrets
p=Path('.env');s=p.read_text().replace('REPLACE_WITH_LONG_RANDOM_KEY',secrets.token_urlsafe(36)).replace('REPLACE_WITH_DIFFERENT_LONG_RANDOM_KEY',secrets.token_urlsafe(36));p.write_text(s)
PY2
  chmod 600 .env
fi
docker compose -f docker-compose.management.yml up -d --build
echo 'API: http://localhost:8000/health'
echo 'Dashboard: http://localhost:8000/'
echo 'Keys: .env (never share this file)'
