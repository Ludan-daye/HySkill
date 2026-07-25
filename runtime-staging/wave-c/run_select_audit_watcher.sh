#!/usr/bin/env bash
set -euo pipefail

if (( $# != 6 )); then
  printf '%s\n' \
    "usage: run_select_audit_watcher.sh REPO_ROOT RESULT_TAG SERVED_MODEL DOMAIN EXPECTED_COUNT PYTHON_PATH" >&2
  exit 2
fi

repo_root=$1
result_tag=$2
served_model=$3
domain=$4
expected_count=$5
python_path=$6

result_root="$repo_root/results/baselines-runtime-matched-v1/$result_tag"
evaluation_path="$result_root/eval/$domain-select-bm25.eval.json"
audit_path="$result_root/audits/$domain-select-bm25.audit.json"

export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"

while [[ ! -f "$evaluation_path" ]]; do
  sleep 10
done

mkdir -p "$result_root/audits"
"$python_path" "$repo_root/scripts/audit_runtime_matched_native_domain.py" \
  --instances "$repo_root/external/SR-Agents/data/bench/instances/$domain.json" \
  --decisions "$result_root/decisions/$domain-select-bm25.decisions.jsonl" \
  --answers "$result_root/answers/$domain-select-bm25.jsonl" \
  --decision-attempt-log "$result_root/logs/$domain-select-bm25.decision.attempts.jsonl" \
  --answer-attempt-log "$result_root/logs/$domain-select-bm25.answer.attempts.jsonl" \
  --decision-full-log "$result_root/logs/$domain-select-bm25.decision.full.log" \
  --answer-full-log "$result_root/logs/$domain-select-bm25.answer.full.log" \
  --decision-manifest "$result_root/runtime/$domain-select-bm25-decision.manifest.json" \
  --answer-manifest "$result_root/runtime/$domain-select-bm25-answer.manifest.json" \
  --evaluation "$evaluation_path" \
  --repository-root "$repo_root" \
  --result-tag "$result_tag" \
  --served-model "$served_model" \
  --domain "$domain" \
  --arm select_bm25 \
  --expected-count "$expected_count" \
  --output "$audit_path"
