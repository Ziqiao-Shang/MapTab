#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INPUT_DIR=${1:-"$REPO_ROOT/results/response_generate"}
OUTPUT_DIR=${2:-"$REPO_ROOT/results_evaluate"}
PYTHON_BIN=${PYTHON_BIN:-python}

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m maptab_infer.cli evaluate-dir \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --family planning
