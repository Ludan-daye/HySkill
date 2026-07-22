#!/usr/bin/env bash
# Own one vLLM service and run its K=8 then K=10 cache/export sequence.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${TAG:?set TAG to the community-results model tag}"
: "${MODEL:?set MODEL to the served-model name and cache model tag}"
: "${MODEL_PATH:?set MODEL_PATH to the local checkpoint path}"
: "${MODEL_REVISION:?set MODEL_REVISION to the exact checkpoint revision}"
: "${GENERATION_COMMIT:?set GENERATION_COMMIT to the generation code commit}"
: "${PORT:?set PORT to the local vLLM port}"
: "${GPU_DEVICE:?set GPU_DEVICE to one CUDA device index}"
: "${GPU_MEMORY_UTILIZATION:?set GPU_MEMORY_UTILIZATION explicitly}"
: "${MAX_MODEL_LEN:?set MAX_MODEL_LEN explicitly}"
: "${VLLM_BIN:?set VLLM_BIN to the vLLM executable}"
: "${VLLM_LIBRARY_PATH:?set VLLM_LIBRARY_PATH explicitly}"
: "${WORKERS:?set WORKERS to the cache-generation concurrency}"
: "${CACHE_DIR:?set CACHE_DIR to the shared hypothetical cache directory}"
: "${INSTANCES_DIR:?set INSTANCES_DIR to the SR-Agents instances directory}"
: "${LAUNCH_DIR:?set LAUNCH_DIR to the service and sequence log directory}"
if [[ ${VLLM_EXTRA_ARGS+x} != x ]]; then
  echo "set VLLM_EXTRA_ARGS explicitly; use an empty string when none are needed" >&2
  exit 2
fi

if ! [[ "$PORT" =~ ^[1-9][0-9]*$ ]]; then
  echo "PORT must be a positive integer: received=$PORT" >&2
  exit 2
fi
if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_MODEL_LEN must be a positive integer: received=$MAX_MODEL_LEN" >&2
  exit 2
fi
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM executable does not exist: path=$VLLM_BIN" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "model checkpoint directory does not exist: path=$MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$LAUNCH_DIR"
VLLM_LOG="$LAUNCH_DIR/vllm-$MODEL.log"
VLLM_PID_PATH="$LAUNCH_DIR/vllm-$MODEL.pid"
SEQUENCE_LOG="$LAUNCH_DIR/sequence-$MODEL.log"
read -r -a EXTRA_ARGS <<< "$VLLM_EXTRA_ARGS"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost"

setsid env \
  LD_LIBRARY_PATH="$VLLM_LIBRARY_PATH" \
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  "$VLLM_BIN" serve "$MODEL_PATH" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${EXTRA_ARGS[@]}" > "$VLLM_LOG" 2>&1 < /dev/null &
VLLM_PID=$!
printf '%s\n' "$VLLM_PID" > "$VLLM_PID_PATH"

stop_vllm() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM -- "-$VLLM_PID"
    for _ in $(seq 1 30); do
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        return 0
      fi
      sleep 2
    done
    echo "vLLM did not stop after SIGTERM: pid=$VLLM_PID log=$VLLM_LOG" >&2
    return 1
  fi
}
trap stop_vllm EXIT

READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready: pid=$VLLM_PID log=$VLLM_LOG" >&2
    tail -n 60 "$VLLM_LOG" >&2
    exit 1
  fi
  if curl --fail --silent --max-time 3 \
      "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    READY=1
    break
  fi
  sleep 5
done
if [[ "$READY" != 1 ]]; then
  echo "vLLM readiness timed out: pid=$VLLM_PID log=$VLLM_LOG" >&2
  tail -n 60 "$VLLM_LOG" >&2
  exit 1
fi

TAG="$TAG" \
MODEL="$MODEL" \
API_BASE="http://127.0.0.1:$PORT/v1" \
MODEL_REVISION="$MODEL_REVISION" \
GENERATION_COMMIT="$GENERATION_COMMIT" \
TARGET_K=8 \
PREFIX_K_VALUES="1 2 4 8" \
WORKERS="$WORKERS" \
CACHE_DIR="$CACHE_DIR" \
INSTANCES_DIR="$INSTANCES_DIR" \
  ./scripts/run_k_cache_stage.sh 2>&1 | tee "$SEQUENCE_LOG"

TAG="$TAG" \
MODEL="$MODEL" \
API_BASE="http://127.0.0.1:$PORT/v1" \
MODEL_REVISION="$MODEL_REVISION" \
GENERATION_COMMIT="$GENERATION_COMMIT" \
TARGET_K=10 \
PREFIX_K_VALUES="10" \
WORKERS="$WORKERS" \
CACHE_DIR="$CACHE_DIR" \
INSTANCES_DIR="$INSTANCES_DIR" \
  ./scripts/run_k_cache_stage.sh 2>&1 | tee -a "$SEQUENCE_LOG"

echo "MODEL-SEQUENCE-DONE tag=$TAG model=$MODEL" | tee -a "$SEQUENCE_LOG"
