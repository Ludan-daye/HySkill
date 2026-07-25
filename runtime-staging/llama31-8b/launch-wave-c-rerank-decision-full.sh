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
labels=()
for domain in "${domains[@]}"; do
  label="$domain-always_rerank"
  "$python_path" "$repo_root/scripts/run_runtime_matched_rerank_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$result_root/bm25/$domain-bm25.json" \
    --output "$result_root/decisions/$domain-always_rerank.jsonl" \
    --attempt-log "$result_root/logs/$domain-always_rerank.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-always_rerank-decision.manifest.json" \
    --result-tag llama31-8b \
    --model llama31-8b \
    --api-base "$api_base" \
    --domain "$domain" \
    --workers 1 \
    --max-new-records 0 \
    --sragents-checkout "$repo_root/external/SR-Agents" \
    --repository-root "$repo_root" \
    --sragents-revision "$sragents_revision" \
    >"$result_root/logs/$label.decision.full.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$result_root/logs/$label.decision.full.pid"
  pids+=("$pid")
  labels+=("$label")
done

status=0
for index in "${!pids[@]}"; do
  pid=${pids[$index]}
  label=${labels[$index]}
  if ! wait "$pid"; then
    printf 'Rerank decision full run failed: label=%s pid=%s\n' "$label" "$pid" >&2
    status=1
  fi
done
exit "$status"
