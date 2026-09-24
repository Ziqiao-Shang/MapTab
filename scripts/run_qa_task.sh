#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DOMAIN=${1:?usage: run_qa_task.sh DOMAIN QA_TASK}
QA_TASKS=${2:?usage: run_qa_task.sh DOMAIN QA_TASK}
case "$DOMAIN" in
  metromap|travelmap) ;;
  *) echo "DOMAIN must be metromap or travelmap" >&2; exit 2 ;;
esac
export DOMAIN QA_TASKS
exec "$SCRIPT_DIR/generate_qa.sh"
