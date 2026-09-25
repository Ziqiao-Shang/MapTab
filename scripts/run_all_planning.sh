#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MAPTAB_DATA_ROOT=${1:?usage: run_all_planning.sh DATA_ROOT [MODEL] [SPLIT]}
export MODEL_PATH=${2:-${MODEL_PATH:-gemini-3.5-flash}}
export SPLIT=${3:-test}
export PROVIDER=${PROVIDER:-openlux}
export DOMAIN=all
export RP_TASKS=all-planning
exec "$SCRIPT_DIR/generate_rp.sh"
