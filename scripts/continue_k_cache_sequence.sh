#!/usr/bin/env bash
# Continue a verified K=8 warmup through prefix export and the K=10 stage.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${TAG:?set TAG to the community-results model tag}"
: "${MODEL:?set MODEL to the served-model name and cache model tag}"
: "${API_BASE:?set API_BASE to an OpenAI-compatible /v1 endpoint}"
: "${MODEL_REVISION:?set MODEL_REVISION to the exact checkpoint revision}"
: "${GENERATION_COMMIT:?set GENERATION_COMMIT to the generation code commit}"
: "${WORKERS:?set WORKERS to the cache-generation concurrency}"
: "${CACHE_DIR:?set CACHE_DIR to the shared hypothetical cache directory}"
: "${INSTANCES_DIR:?set INSTANCES_DIR to the SR-Agents instances directory}"
: "${INITIAL_WARM_PID:?set INITIAL_WARM_PID to the running K=8 warmer PID}"
: "${INITIAL_WARM_LOG:?set INITIAL_WARM_LOG to the K=8 warmer log path}"

if ! [[ "$INITIAL_WARM_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "INITIAL_WARM_PID must be a positive integer: received=$INITIAL_WARM_PID" >&2
  exit 2
fi
if [[ ! -f "$INITIAL_WARM_LOG" ]]; then
  echo "initial warmup log does not exist: path=$INITIAL_WARM_LOG" >&2
  exit 2
fi

while kill -0 "$INITIAL_WARM_PID" 2>/dev/null; do
  sleep 30
done

if ! grep -Fq "WARMUP-DONE jobs=11910 empty=0" "$INITIAL_WARM_LOG"; then
  echo "initial K=8 warmup did not finish cleanly: log=$INITIAL_WARM_LOG" >&2
  tail -n 40 "$INITIAL_WARM_LOG" >&2
  exit 1
fi

TAG="$TAG" \
MODEL="$MODEL" \
API_BASE="$API_BASE" \
MODEL_REVISION="$MODEL_REVISION" \
GENERATION_COMMIT="$GENERATION_COMMIT" \
TARGET_K=8 \
PREFIX_K_VALUES="1 2 4 8" \
WORKERS="$WORKERS" \
CACHE_DIR="$CACHE_DIR" \
INSTANCES_DIR="$INSTANCES_DIR" \
  ./scripts/run_k_cache_stage.sh

TAG="$TAG" \
MODEL="$MODEL" \
API_BASE="$API_BASE" \
MODEL_REVISION="$MODEL_REVISION" \
GENERATION_COMMIT="$GENERATION_COMMIT" \
TARGET_K=10 \
PREFIX_K_VALUES="10" \
WORKERS="$WORKERS" \
CACHE_DIR="$CACHE_DIR" \
INSTANCES_DIR="$INSTANCES_DIR" \
  ./scripts/run_k_cache_stage.sh

echo "CACHE-SEQUENCE-DONE tag=$TAG model=$MODEL"
