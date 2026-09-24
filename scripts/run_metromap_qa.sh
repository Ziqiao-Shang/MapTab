#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN=metromap
export QA_TASKS=${QA_TASKS:-all-qa}
exec "$SCRIPT_DIR/generate_qa.sh"
