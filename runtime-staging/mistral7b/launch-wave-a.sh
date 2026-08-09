#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/mistral7b
instance_root="$repo_root/external/SR-Agents/data/bench/instances"
corpus_path="$repo_root/external/SR-Agents/data/bench/corpus/corpus.json"
validation_root="$repo_root/results/k2-main/mistral7b"
python_path="$repo_root/runtime/runner/bin/python"
vllm_entry="$repo_root/runtime/vllm0191/bin/vllm"
checkpoint="$repo_root/checkpoints/mistral7b-c8cfccbcfd71d4e3479498c30b2823bab19c4687"
frozen_manifest="$repo_root/staging/mistral7b/k2-mistral.files.sha256"
api_base=http://127.0.0.1:8000/v1
expected_manifest_sha=559840283ece7b8cbbb937d74d5ce47aff520cda4a453a3331ac3e8f26bfa6df
frozen_checkpoint_prefix=/root/.cache/modelscope/models/LLM-Research--Mistral-7B-Instruct-v0.3/snapshots/c8cfccbcfd71d4e3479498c30b2823bab19c4687
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export CPATH="$repo_root/runtime/python310-dev-3.10.12/usr/include/python3.10:$repo_root/runtime/python310-dev-3.10.12/usr/include"

mkdir -p "$result_root/runtime" "$result_root/answers" "$result_root/logs" "$result_root/eval" "$result_root/audits"

actual_manifest_sha=$(sha256sum "$frozen_manifest" | awk '{print $1}')
if [[ "$actual_manifest_sha" != "$expected_manifest_sha" ]]; then
  echo "Frozen Mistral checkpoint manifest mismatch: expected=$expected_manifest_sha actual=$actual_manifest_sha" >&2
  exit 1
fi
sed "s#$frozen_checkpoint_prefix#$checkpoint#" "$frozen_manifest" | sha256sum --status -c -

active_gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
if [[ "$active_gpu_processes" -ne 0 ]]; then
  echo "Refusing to start Mistral on a busy GPU: active_processes=$active_gpu_processes" >&2
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8000$'; then
  echo "Refusing to start Mistral because TCP port 8000 is already listening" >&2
  exit 1
fi

nohup "$vllm_entry" serve "$checkpoint" \
  --port 8000 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --enforce-eager \
  --served-model-name mistral7b \
  >"$result_root/runtime/vllm.log" 2>&1 </dev/null &
endpoint_pid=$!
echo "$endpoint_pid" >"$result_root/runtime/vllm.pid"

ready=0
for _attempt in $(seq 1 240); do
  if wget -qO "$result_root/runtime/models-readback.json" "$api_base/models"; then
    if grep -q '"id":"mistral7b"' "$result_root/runtime/models-readback.json"; then
      ready=1
      break
    fi
  fi
  if ! kill -0 "$endpoint_pid" 2>/dev/null; then
    echo "Mistral endpoint exited during startup: pid=$endpoint_pid" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "Mistral endpoint did not become ready within 20 minutes" >&2
  exit 1
fi

"$python_path" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); m=[x for x in p["data"] if x.get("id")=="mistral7b"]; ok=len(m)==1 and m[0].get("max_model_len")==8192 and m[0].get("root")==sys.argv[2]; sys.exit(0 if ok else f"Invalid Mistral endpoint readback: {p}")' \
  "$result_root/runtime/models-readback.json" \
  "$checkpoint"

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
for domain in "${domains[@]}"; do
  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$repo_root/staging/mistral7b/$domain.runtime-facts.json" \
    --generation "$repo_root/staging/mistral7b/generation.json" \
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
    --result-tag mistral7b \
    --model mistral7b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 4 \
    --max-new-records 5
done

answer_pids=()
for domain in "${domains[@]}"; do
  nohup "$python_path" "$repo_root/scripts/run_runtime_matched_bare.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --runtime-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --repository-root "$repo_root" \
    --output "$result_root/answers/$domain-bare.jsonl" \
    --usage-log "$result_root/logs/$domain-bare.usage.jsonl" \
    --attempt-log "$result_root/logs/$domain-bare.attempts.jsonl" \
    --result-tag mistral7b \
    --model mistral7b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 4 \
    --max-new-records 0 \
    >"$result_root/logs/$domain-bare.full.log" 2>&1 &
  answer_pid=$!
  answer_pids+=("$answer_pid")
  echo "$answer_pid" >"$result_root/logs/$domain-bare.full.pid"
done

for answer_pid in "${answer_pids[@]}"; do
  wait "$answer_pid"
done

for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  expected_count=${counts[$index]}
  observed_count=$(wc -l <"$result_root/answers/$domain-bare.jsonl")
  if [[ "$observed_count" -ne "$expected_count" ]]; then
    echo "Mistral Bare output incomplete: domain=$domain expected=$expected_count observed=$observed_count" >&2
    exit 1
  fi
  "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
    --answers "$result_root/answers/$domain-bare.jsonl" \
    --instances "$instance_root/$domain.json" \
    --validation-source "$validation_root/$domain-routed-taus.json" \
    --result-tag mistral7b \
    --served-model mistral7b \
    --domain "$domain" \
    --arm bare \
    --expected-count "$expected_count" \
    --output "$result_root/eval/$domain-bare.eval.json"
done

"$python_path" "$repo_root/scripts/audit_runtime_matched_bare.py" \
  --result-root "$result_root" \
  --instances-dir "$instance_root" \
  --result-tag mistral7b \
  --served-model mistral7b \
  --output "$result_root/audits/bare-completeness.json"
