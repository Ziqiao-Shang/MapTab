#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${1:-"$REPO_ROOT/data"}
SUBSET=${2:-test}
PYTHON_BIN=${PYTHON_BIN:-python}

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m maptab_infer.cli download \
  --repo-id "${MAPTAB_REPO_ID:-szq-nju/MapTab}" \
  --revision "${MAPTAB_REVISION:-main}" \
  --subset "$SUBSET" \
  --data-root "$DATA_ROOT"
