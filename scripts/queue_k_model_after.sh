#!/usr/bin/env bash
# Wait for a previous cache sequence, stop its vLLM, then run the next model.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${PREVIOUS_SEQUENCE_PID:?set PREVIOUS_SEQUENCE_PID explicitly}"
: "${PREVIOUS_SEQUENCE_LOG:?set PREVIOUS_SEQUENCE_LOG explicitly}"
: "${PREVIOUS_VLLM_PID:?set PREVIOUS_VLLM_PID explicitly}"

if ! [[ "$PREVIOUS_SEQUENCE_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREVIOUS_SEQUENCE_PID must be a positive integer" >&2
  exit 2
fi
if ! [[ "$PREVIOUS_VLLM_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREVIOUS_VLLM_PID must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$PREVIOUS_SEQUENCE_LOG" ]]; then
  echo "previous sequence log does not exist: path=$PREVIOUS_SEQUENCE_LOG" >&2
  exit 2
fi

while kill -0 "$PREVIOUS_SEQUENCE_PID" 2>/dev/null; do
  sleep 30
done
if ! grep -Fq "CACHE-SEQUENCE-DONE" "$PREVIOUS_SEQUENCE_LOG"; then
  echo "previous cache sequence did not finish cleanly: log=$PREVIOUS_SEQUENCE_LOG" >&2
  tail -n 60 "$PREVIOUS_SEQUENCE_LOG" >&2
  exit 1
fi

if kill -0 "$PREVIOUS_VLLM_PID" 2>/dev/null; then
  kill -TERM "$PREVIOUS_VLLM_PID"
  for _ in $(seq 1 30); do
    if ! kill -0 "$PREVIOUS_VLLM_PID" 2>/dev/null; then
      break
    fi
    sleep 2
  done
fi
if kill -0 "$PREVIOUS_VLLM_PID" 2>/dev/null; then
  echo "previous vLLM did not stop: pid=$PREVIOUS_VLLM_PID" >&2
  exit 1
fi

GPU_PIDS=""
for _ in $(seq 1 30); do
  GPU_PIDS=$(nvidia-smi --id="$GPU_DEVICE" \
    --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -z "$GPU_PIDS" ]]; then
    break
  fi
  sleep 2
done
if [[ -n "$GPU_PIDS" ]]; then
  echo "GPU still has compute processes after stopping previous vLLM: gpu=$GPU_DEVICE pids=$GPU_PIDS" >&2
  exit 1
fi

exec ./scripts/run_k_model_sequence.sh
