#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

launch_select_full() {
  local domain=$1
  local expected=$2
  "$python_path" "$repo_root/scripts/run_runtime_matched_select_decisions.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --bm25-source "$result_root/bm25/$domain-bm25.json" \
    --output "$result_root/decisions/$domain-select_bm25.jsonl" \
    --attempt-log "$result_root/logs/$domain-select_bm25.decision.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select_bm25-decision.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag llama31-8b \
    --model llama31-8b \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 1 \
    --max-new-records 0 \
    >"$result_root/logs/$domain-select_bm25.decision.full.log" 2>&1 &
}

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
pids=()
labels=()
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  expected=${counts[$index]}
  launch_select_full "$domain" "$expected"
  pids+=("$!")
  labels+=("$domain-select_bm25")
done

status=0
for index in "${!pids[@]}"; do
  pid=${pids[$index]}
  label=${labels[$index]}
  printf '%s\n' "$pid" >"$result_root/logs/$label.decision.full.pid"
  if ! wait "$pid"; then
    printf 'Select decision full run failed: label=%s pid=%s\n' "$label" "$pid" >&2
    status=1
  fi
done
exit "$status"
