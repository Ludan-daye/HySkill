#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/glm4-9b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
mkdir -p "$result_root/answers" "$result_root/logs"
pids=()
labels=()
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  expected=${counts[$index]}
  label="$domain-select_bm25"
  "$python_path" "$repo_root/scripts/run_runtime_matched_select_answers.py" \
    --instances "$instance_root/$domain.json" \
    --corpus "$corpus_path" \
    --decisions "$result_root/decisions/$domain-select_bm25.jsonl" \
    --output "$result_root/answers/$domain-select_bm25.jsonl" \
    --attempt-log "$result_root/logs/$domain-select_bm25.answer.attempts.jsonl" \
    --runtime-manifest "$result_root/runtime/$domain-select_bm25-answer.manifest.json" \
    --repository-root "$repo_root" \
    --result-tag glm4-9b \
    --model glm4-9b \
    --api-base "$api_base" \
    --domain "$domain" \
    --expected-count "$expected" \
    --workers 1 \
    --max-new-records 5 \
    >"$result_root/logs/$label.answer.canary.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$result_root/logs/$label.answer.canary.pid"
  pids+=("$pid")
  labels+=("$label")
done

status=0
for index in "${!pids[@]}"; do
  pid=${pids[$index]}
  label=${labels[$index]}
  if ! wait "$pid"; then
    printf 'Select answer canary failed: label=%s pid=%s\n' "$label" "$pid" >&2
    status=1
  fi
done
exit "$status"
