#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-run-20260724
result_root="$repo_root/results/baselines-runtime-matched-v1/qwen35-9b"
instance_root="$repo_root/external/SR-Agents/data/bench/instances"
corpus_path="$repo_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path=/root/HySkill-k2-main-20260723/runtime/vllm-env/bin/python
api_base=http://127.0.0.1:8000/v1
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
mkdir -p "$result_root/logs"

domains=(theoremqa logicbench medcalcbench champ)
for domain in "${domains[@]}"; do
  nohup "$python_path" "$repo_root/scripts/run_runtime_matched_bare.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --runtime-manifest "$result_root/runtime/$domain-bare.manifest.json" \
    --repository-root "$repo_root" \
    --output "$result_root/answers/$domain-bare.jsonl" \
    --usage-log "$result_root/logs/$domain-bare.usage.jsonl" \
    --attempt-log "$result_root/logs/$domain-bare.attempts.jsonl" \
    --result-tag qwen35-9b \
    --model qwen35-9b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 3 \
    --max-new-records 0 \
    >"$result_root/logs/$domain-bare.full.log" 2>&1 &
  printf '%s\n' "$!" >"$result_root/logs/$domain-bare.full.pid"
done
