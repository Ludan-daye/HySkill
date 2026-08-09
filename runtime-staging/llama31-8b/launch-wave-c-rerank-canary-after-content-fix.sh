#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
sragents_revision=277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

domains=(theoremqa logicbench medcalcbench champ)
pids=()

for domain in "${domains[@]}"; do
  output_path="$result_root/decisions/$domain-always_rerank.jsonl"
  if test -s "$output_path"; then
    echo "Refusing to replace an existing Rerank decision output: $output_path" >&2
    exit 1
  fi
  old_log="$result_root/logs/$domain-always_rerank.decision.canary.log"
  preserved_log="$old_log.pre-empty-content-fix"
  if test -f "$old_log" && test ! -e "$preserved_log"; then
    cp "$old_log" "$preserved_log"
  fi
  "$python_path" "$repo_root/scripts/run_runtime_matched_rerank_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$result_root/bm25/$domain-bm25.json" \
    --output "$output_path" \
    --attempt-log "$result_root/logs/$domain-always_rerank.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-always_rerank-decision.manifest.json" \
    --result-tag llama31-8b \
    --model llama31-8b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 1 \
    --max-new-records 5 \
    --sragents-checkout "$repo_root/external/SR-Agents" \
    --repository-root "$repo_root" \
    --sragents-revision "$sragents_revision" \
    >"$old_log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" >"$result_root/logs/$domain-always_rerank.decision.canary.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
