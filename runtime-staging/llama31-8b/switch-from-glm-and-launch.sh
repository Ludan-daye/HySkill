#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
result_base=/root/HySkill-baseline-runtime-matched-20260724-results
glm_root="$result_base/glm4-9b"
llama_root="$result_base/llama31-8b"
checkpoint=/root/.cache/modelscope/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master

if [[ ! -s "$glm_root/audits/bare-completeness.json" ]]; then
  "$repo_root/runtime-staging/glm4-9b/finalize-wave-a-bare.sh"
fi

/root/HySkill-k-run-20260723/.venv/bin/python -c \
  'import json,sys; p=json.load(open(sys.argv[1])); ok=p.get("valid") is True and p.get("observed_rows")==2830 and p.get("unresolved")==0; sys.exit(0 if ok else "GLM Bare completeness audit is invalid")' \
  "$glm_root/audits/bare-completeness.json"

glm_endpoint_pid=$(<"$glm_root/runtime/vllm.retry1.pid")
glm_command=$(tr '\0' ' ' <"/proc/$glm_endpoint_pid/cmdline")
if [[ "$glm_command" != *"vllm serve"* || "$glm_command" != *"glm-4-9b-chat"* ]]; then
  echo "Refusing to stop an endpoint that is not the completed GLM experiment: pid=$glm_endpoint_pid command=$glm_command" >&2
  exit 1
fi
kill -TERM "$glm_endpoint_pid"
for _attempt in $(seq 1 120); do
  if ! kill -0 "$glm_endpoint_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$glm_endpoint_pid" 2>/dev/null; then
  echo "GLM endpoint did not stop after SIGTERM: pid=$glm_endpoint_pid" >&2
  exit 1
fi

mkdir -p "$llama_root/runtime" "$llama_root/logs"
nohup /root/vllmenv/bin/vllm serve "$checkpoint" \
  --port 8000 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --enforce-eager \
  --served-model-name llama31-8b \
  >"$llama_root/runtime/vllm.log" 2>&1 &
llama_endpoint_pid=$!
echo "$llama_endpoint_pid" >"$llama_root/runtime/vllm.pid"

ready=0
for _attempt in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8000/v1/models | grep -q '"id":"llama31-8b"'; then
    ready=1
    break
  fi
  if ! kill -0 "$llama_endpoint_pid" 2>/dev/null; then
    echo "Llama endpoint exited during startup: pid=$llama_endpoint_pid" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "Llama endpoint did not become ready within 20 minutes" >&2
  exit 1
fi

exec "$repo_root/runtime-staging/llama31-8b/launch-wave-a.sh"
