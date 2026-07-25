#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/mistral7b
instance_root="$repo_root/external/SR-Agents/data/bench/instances"
python_path="$repo_root/runtime/runner/bin/python"

export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"

mkdir -p "$result_root/audits" "$result_root/logs"

run_audit() {
  local domain=$1
  local expected=$2
  local arm=$3
  local file_label
  if [[ "$arm" == "select_bm25" ]]; then
    file_label=select-bm25
  elif [[ "$arm" == "always_rerank" ]]; then
    file_label=always_rerank
  else
    printf 'Unsupported native arm: %s\n' "$arm" >&2
    exit 2
  fi

  local evaluation_path="$result_root/eval/$domain-$file_label.eval.json"
  while [[ ! -f "$evaluation_path" ]]; do
    sleep 10
  done

  "$python_path" "$repo_root/scripts/audit_runtime_matched_native_domain.py" \
    --instances "$instance_root/$domain.json" \
    --decisions "$result_root/decisions/$domain-$file_label.decisions.jsonl" \
    --answers "$result_root/answers/$domain-$file_label.jsonl" \
    --decision-attempt-log "$result_root/logs/$domain-$file_label.decision.attempts.jsonl" \
    --answer-attempt-log "$result_root/logs/$domain-$file_label.answer.attempts.jsonl" \
    --decision-full-log "$result_root/logs/$domain-$file_label.decision.full.log" \
    --answer-full-log "$result_root/logs/$domain-$file_label.answer.full.log" \
    --decision-manifest "$result_root/runtime/$domain-$file_label-decision.manifest.json" \
    --answer-manifest "$result_root/runtime/$domain-$file_label-answer.manifest.json" \
    --evaluation "$evaluation_path" \
    --repository-root "$repo_root" \
    --result-tag mistral7b \
    --served-model mistral7b \
    --domain "$domain" \
    --arm "$arm" \
    --expected-count "$expected" \
    --output "$result_root/audits/$domain-$file_label.audit.json"
}

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
arms=(select_bm25 always_rerank)
pids=()
labels=()
for arm in "${arms[@]}"; do
  for index in "${!domains[@]}"; do
    domain=${domains[$index]}
    expected=${counts[$index]}
    run_audit "$domain" "$expected" "$arm" \
      >"$result_root/logs/$domain-$arm.audit.log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$result_root/logs/$domain-$arm.audit.pid"
    pids+=("$pid")
    labels+=("$domain-$arm")
  done
done

status=0
for index in "${!pids[@]}"; do
  pid=${pids[$index]}
  label=${labels[$index]}
  if ! wait "$pid"; then
    printf 'Native audit failed: label=%s pid=%s log=%s\n' \
      "$label" "$pid" "$result_root/logs/$label.audit.log" >&2
    status=1
  fi
done
exit "$status"
