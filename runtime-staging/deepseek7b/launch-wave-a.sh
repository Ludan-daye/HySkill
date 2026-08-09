#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/deepseek7b
instance_root=/root/HySkill-k-run-20260723/external/SR-Agents/data/bench/instances
corpus_path=/root/HySkill-k-run-20260723/external/SR-Agents/data/bench/corpus/corpus.json
python_path=/root/HySkill-k-run-20260723/.venv/bin/python
api_base=http://127.0.0.1:8000/v1
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
mkdir -p "$result_root/runtime" "$result_root/answers" "$result_root/logs"

domains=(theoremqa logicbench medcalcbench champ)
for domain in "${domains[@]}"; do
  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$repo_root/runtime-staging/deepseek7b/$domain.runtime-facts.json" \
    --generation "$repo_root/runtime-staging/deepseek7b/generation.json" \
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
    --result-tag deepseek7b \
    --model deepseek7b \
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
    --result-tag deepseek7b \
    --model deepseek7b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 4 \
    --max-new-records 0 \
    >"$result_root/logs/$domain-bare.full.log" 2>&1 &
  echo "$!" >"$result_root/logs/$domain-bare.full.pid"
done

wait
