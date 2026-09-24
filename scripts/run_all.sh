#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DOMAIN=${DOMAIN:-all} QA_TASKS=${QA_TASKS:-all-qa} "$SCRIPT_DIR/generate_qa.sh"
DOMAIN=${DOMAIN:-all} RP_TASKS=${RP_TASKS:-all-planning} SPLIT=${SPLIT:-test} "$SCRIPT_DIR/generate_rp.sh"
