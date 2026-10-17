#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .env ]]; then echo 'Missing .env. Run scripts/management-setup.sh'; exit 1; fi
set -a; source .env; set +a
BASE="${MANAGER_URL:-http://localhost:8000}"
echo 'Health:'; curl --fail --silent --show-error "$BASE/health"; echo
echo 'Nodes:'; curl --fail --silent --show-error -H "x-api-key: $API_KEY" "$BASE/nodes"; echo
echo 'Recent jobs:'; curl --fail --silent --show-error -H "x-api-key: $API_KEY" "$BASE/jobs?limit=5"; echo
