#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MAPTAB_DATA_ROOT=${1:?usage: run_all_qa.sh DATA_ROOT [MODEL]}
export MODEL_PATH=${2:-${MODEL_PATH:-gemini-3.5-flash}}
export PROVIDER=${PROVIDER:-openlux}
export DOMAIN=all
export QA_TASKS=all-qa
exec "$SCRIPT_DIR/generate_qa.sh"
