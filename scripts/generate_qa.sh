#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${MAPTAB_DATA_ROOT:?Set MAPTAB_DATA_ROOT to a MapTab data snapshot}"
PYTHON_BIN=${PYTHON_BIN:-python}
PROVIDER=${PROVIDER:-openlux}
MODEL_PATH=${MODEL_PATH:-gemini-3.5-flash}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/results/response_generate"}
ARGS=(
  -m maptab_infer.cli generate
  --data-root "$MAPTAB_DATA_ROOT"
  --domain "${DOMAIN:-all}"
  --task "${QA_TASKS:-all-qa}"
  --provider "$PROVIDER"
  --model "$MODEL_PATH"
  --temperature "${TEMPERATURE:-0}"
  --max-tokens "${MAX_TOKENS:-2048}"
  --max-pixels "${MAX_PIXELS:-10000000}"
  --max-retries "${MAX_RETRIES:-3}"
  --retry-backoff "${RETRY_BACKOFF:-2}"
  --timeout "${TIMEOUT:-120}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --max-model-len "${MAX_MODEL_LEN:-128000}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
  --output-dir "$OUTPUT_DIR"
  --seed "${SEED:-42}"
)
if [[ -n "${OFFSET:-}" ]]; then
  ARGS+=(--offset "$OFFSET")
fi
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then
  ARGS+=(--retry-errors)
fi
if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  ARGS+=(--continue-on-error)
fi

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "${ARGS[@]}"
