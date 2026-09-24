#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MAPTAB_DATA_ROOT=${1:?usage: run_all_qa.sh DATA_ROOT MODEL BASE_URL}
export MODEL_PATH=${2:?usage: run_all_qa.sh DATA_ROOT MODEL BASE_URL}
export OPENAI_BASE_URL=${3:?usage: run_all_qa.sh DATA_ROOT MODEL BASE_URL}
export PROVIDER=openai
export DOMAIN=all
export QA_TASKS=all-qa
exec "$SCRIPT_DIR/generate_qa.sh"
