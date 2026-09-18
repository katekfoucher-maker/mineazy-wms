#!/usr/bin/env bash
# Mineazy WMS - dev launcher. Seeds (first run) then starts the API.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

if [ ! -f wms.db ]; then
  echo "==> Seeding demo database"
  "$PY" -m wms.scripts.seed
fi
echo "==> API on http://127.0.0.1:8000  (Swagger: /docs)"
echo "    Console:  $PY -m wms.console      (add --demo for a headless tour)"
exec "$PY" -m uvicorn wms.api.main:app --host 127.0.0.1 --port 8000
