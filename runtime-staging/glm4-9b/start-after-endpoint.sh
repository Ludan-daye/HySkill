#!/usr/bin/env bash
set -euo pipefail

result_root=/root/HySkill-baseline-runtime-matched-20260724-results/glm4-9b
endpoint_pid=$(cat "$result_root/runtime/vllm.retry1.pid")
ready=0
for _attempt in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8000/v1/models | grep -q '"id":"glm4-9b"'; then
    ready=1
    break
  fi
  if ! kill -0 "$endpoint_pid" 2>/dev/null; then
    echo "GLM endpoint exited during startup: pid=$endpoint_pid" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "GLM endpoint did not become ready within 20 minutes" >&2
  exit 1
fi

exec /root/HySkill-baseline-runtime-matched-20260724/repo/runtime-staging/glm4-9b/launch-wave-a.sh
