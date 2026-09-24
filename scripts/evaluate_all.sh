#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
INPUT_DIR=${1:-"$REPO_ROOT/results/response_generate"}
OUTPUT_DIR=${2:-"$REPO_ROOT/results_evaluate"}

"$SCRIPT_DIR/evaluate_qa.sh" "$INPUT_DIR" "$OUTPUT_DIR"
"$SCRIPT_DIR/evaluate_rp.sh" "$INPUT_DIR" "$OUTPUT_DIR"
