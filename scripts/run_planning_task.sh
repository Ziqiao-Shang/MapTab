#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DOMAIN=${1:?usage: run_planning_task.sh DOMAIN PLANNING_TASK [SPLIT]}
RP_TASKS=${2:?usage: run_planning_task.sh DOMAIN PLANNING_TASK [SPLIT]}
SPLIT=${3:-test}
case "$DOMAIN" in
  metromap|travelmap) ;;
  *) echo "DOMAIN must be metromap or travelmap" >&2; exit 2 ;;
esac
case "$SPLIT" in
  train|test|all) ;;
  *) echo "SPLIT must be train, test, or all" >&2; exit 2 ;;
esac
export DOMAIN RP_TASKS SPLIT
exec "$SCRIPT_DIR/generate_rp.sh"
