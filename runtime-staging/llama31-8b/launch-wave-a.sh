#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
validation_root="$source_root/results/k2-main/llama31-8b"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
mkdir -p \
  "$result_root/runtime" \
  "$result_root/answers" \
  "$result_root/logs" \
  "$result_root/eval" \
  "$result_root/audits"

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)

for domain in "${domains[@]}"; do
  "$python_path" "$repo_root/scripts/build_runtime_matched_runtime_manifest.py" \
    --runtime-facts "$repo_root/runtime-staging/llama31-8b/$domain.runtime-facts.json" \
    --generation "$repo_root/runtime-staging/llama31-8b/generation.json" \
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
    --result-tag llama31-8b \
    --model llama31-8b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 4 \
    --max-new-records 5 \
    >"$result_root/logs/$domain-bare.canary.log" 2>&1
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
    --result-tag llama31-8b \
    --model llama31-8b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 4 \
    --max-new-records 0 \
    >"$result_root/logs/$domain-bare.full.log" 2>&1 &
  echo "$!" >"$result_root/logs/$domain-bare.full.pid"
done

wait

for index in "${!domains[@]}"; do
  domain="${domains[$index]}"
  expected_count="${counts[$index]}"
  "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
    --answers "$result_root/answers/$domain-bare.jsonl" \
    --instances "$instance_root/$domain.json" \
    --validation-source "$validation_root/$domain-routed-taus.json" \
    --result-tag llama31-8b \
    --served-model llama31-8b \
    --domain "$domain" \
    --arm bare \
    --expected-count "$expected_count" \
    --output "$result_root/eval/$domain-bare.eval.json"
done

"$python_path" "$repo_root/scripts/audit_runtime_matched_bare.py" \
  --result-root "$result_root" \
  --instances-dir "$instance_root" \
  --result-tag llama31-8b \
  --served-model llama31-8b \
  --output "$result_root/audits/bare-completeness.json"
