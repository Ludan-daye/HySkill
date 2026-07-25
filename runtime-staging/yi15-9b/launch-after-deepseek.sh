#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
deepseek_root=/root/HySkill-baseline-runtime-matched-20260724-results/deepseek7b
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/yi15-9b
instance_root=/root/HySkill-k-run-20260723/external/SR-Agents/data/bench/instances
corpus_path=/root/HySkill-k-run-20260723/external/SR-Agents/data/bench/corpus/corpus.json
python_path=/root/HySkill-k-run-20260723/.venv/bin/python
vllm_python=/root/vllmenv/bin/python3
vllm_entry=/root/vllmenv/bin/vllm
checkpoint=/root/.cache/modelscope/models/01ai--Yi-1.5-9B-Chat/snapshots/master
api_base=http://127.0.0.1:8000/v1
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
mkdir -p "$result_root/runtime" "$result_root/answers" "$result_root/logs"

domains=(theoremqa logicbench medcalcbench champ)
expected_counts=(747 760 1100 223)
for domain in "${domains[@]}"; do
  deepseek_pid=$(cat "$deepseek_root/logs/$domain-bare.full.pid")
  while kill -0 "$deepseek_pid" 2>/dev/null; do
    sleep 10
  done
done

for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  observed=$(wc -l <"$deepseek_root/answers/$domain-bare.jsonl")
  expected=${expected_counts[$index]}
  if [[ "$observed" -ne "$expected" ]]; then
    echo "DeepSeek output incomplete: domain=$domain expected=$expected observed=$observed" >&2
    exit 1
  fi
done

deepseek_endpoint_pid=$(cat "$deepseek_root/runtime/vllm.pid")
if kill -0 "$deepseek_endpoint_pid" 2>/dev/null; then
  kill -TERM "$deepseek_endpoint_pid"
  for _attempt in $(seq 1 60); do
    if ! kill -0 "$deepseek_endpoint_pid" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if kill -0 "$deepseek_endpoint_pid" 2>/dev/null; then
    echo "DeepSeek endpoint did not exit after SIGTERM: pid=$deepseek_endpoint_pid" >&2
    exit 1
  fi
fi

nohup "$vllm_python" "$vllm_entry" serve "$checkpoint" \
  --port 8000 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --enforce-eager \
  --served-model-name yi15-9b \
  >"$result_root/runtime/vllm.log" 2>&1 </dev/null &
yi_endpoint_pid=$!
echo "$yi_endpoint_pid" >"$result_root/runtime/vllm.pid"

ready=0
for _attempt in $(seq 1 240); do
  if curl -fsS "$api_base/models" | grep -q '"id":"yi15-9b"'; then
    ready=1
    break
  fi
  if ! kill -0 "$yi_endpoint_pid" 2>/dev/null; then
    echo "Yi endpoint exited during startup: pid=$yi_endpoint_pid" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "Yi endpoint did not become ready within 20 minutes" >&2
  exit 1
fi

for domain in "${domains[@]}"; do
  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$repo_root/runtime-staging/yi15-9b/$domain.runtime-facts.json" \
    --generation "$repo_root/runtime-staging/yi15-9b/generation.json" \
    --artifact "instances=$instance_root/$domain.json" \
    --artifact "corpus=$corpus_path" \
    --code-file hyskill/runtime_matched_execution.py \
    --code-file scripts/run_runtime_matched_bare.py \
    --code-file external/SR-Agents/src/sragents/llm.py \
    --code-file external/SR-Agents/src/sragents/prompts.py \
    --code-file external/SR-Agents/src/sragents/infer/base.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/direct.py \
    --code-file external/SR-Agents/src/sragents/infer/engines/tool_loop.py \
    --repository-root "$repo_root" \
    --output "$result_root/runtime/$domain-bare.manifest.json"
done

for domain in "${domains[@]}"; do
  "$python_path" "$repo_root/scripts/run_runtime_matched_bare.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --runtime-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --repository-root "$repo_root" \
    --output "$result_root/answers/$domain-bare.jsonl" \
    --usage-log "$result_root/logs/$domain-bare.usage.jsonl" \
    --attempt-log "$result_root/logs/$domain-bare.attempts.jsonl" \
    --result-tag yi15-9b \
    --model yi15-9b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 4 \
    --max-new-records 5
done

for domain in "${domains[@]}"; do
  nohup "$python_path" "$repo_root/scripts/run_runtime_matched_bare.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --runtime-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --repository-root "$repo_root" \
    --output "$result_root/answers/$domain-bare.jsonl" \
    --usage-log "$result_root/logs/$domain-bare.usage.jsonl" \
    --attempt-log "$result_root/logs/$domain-bare.attempts.jsonl" \
    --result-tag yi15-9b \
    --model yi15-9b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 4 \
    --max-new-records 0 \
    >"$result_root/logs/$domain-bare.full.log" 2>&1 &
  echo "$!" >"$result_root/logs/$domain-bare.full.pid"
done

wait
