#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${1:-"$REPO_ROOT/data"}
REPO_ID=${2:-"szq-nju/MapTab"}
PYTHON_BIN=${PYTHON_BIN:-python}

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m maptab_infer.cli download \
  --repo-id "$REPO_ID" \
  --data-root "$DATA_ROOT"
