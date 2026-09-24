#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MAPTAB_DATA_ROOT=${1:?usage: run_all_planning.sh DATA_ROOT MODEL BASE_URL [SPLIT]}
export MODEL_PATH=${2:?usage: run_all_planning.sh DATA_ROOT MODEL BASE_URL [SPLIT]}
export OPENAI_BASE_URL=${3:?usage: run_all_planning.sh DATA_ROOT MODEL BASE_URL [SPLIT]}
export SPLIT=${4:-test}
export PROVIDER=openai
export DOMAIN=all
export RP_TASKS=all-planning
exec "$SCRIPT_DIR/generate_rp.sh"
