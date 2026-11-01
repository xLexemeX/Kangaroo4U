#!/usr/bin/env bash
# Start Kangaroo4U: the GPU solver plus the live dashboard.
#   ./run.sh              search and dashboard
#   ./run.sh dashboard    dashboard only
#   ./run.sh bench        quick GPU self-test
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python) || { echo "Python 3 is required"; exit 1; }
exec "$PY" kangaroo4u.py "$@"
