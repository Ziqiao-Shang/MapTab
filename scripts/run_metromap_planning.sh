#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN=metromap
export RP_TASKS=${RP_TASKS:-all-planning}
export SPLIT=${SPLIT:-test}
exec "$SCRIPT_DIR/generate_rp.sh"
