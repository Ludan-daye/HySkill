#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: s1-finalize-wave-a.sh <yi-launcher-pid>" >&2
  exit 2
fi

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_base=/root/HySkill-baseline-runtime-matched-20260724-results
instance_root="$source_root/external/SR-Agents/data/bench/instances"
python_path="$source_root/.venv/bin/python"
yi_launcher_pid="$1"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)

wait_for_job_pids() {
  local result_root="$1"
  local owner_pid="$2"
  local ready=0
  local attempt
  local domain

  for attempt in $(seq 1 120); do
    ready=1
    for domain in "${domains[@]}"; do
      if [[ ! -s "$result_root/logs/$domain-bare.full.pid" ]]; then
        ready=0
      fi
    done
    if [[ "$ready" -eq 1 ]]; then
      return
    fi
    if ! kill -0 "$owner_pid" 2>/dev/null; then
      echo "Owner process exited before all job PID files appeared: pid=$owner_pid result_root=$result_root" >&2
      exit 1
    fi
    sleep 30
  done
  echo "Timed out waiting for job PID files: result_root=$result_root" >&2
  exit 1
}

wait_for_jobs() {
  local result_root="$1"
  local domain
  local pid

  for domain in "${domains[@]}"; do
    pid=$(<"$result_root/logs/$domain-bare.full.pid")
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
    done
  done
}

verify_counts() {
  local result_root="$1"
  local index
  local domain
  local expected
  local observed

  for index in "${!domains[@]}"; do
    domain="${domains[$index]}"
    expected="${counts[$index]}"
    observed=$(wc -l <"$result_root/answers/$domain-bare.jsonl")
    if [[ "$observed" -ne "$expected" ]]; then
      echo "Bare output is incomplete: domain=$domain expected=$expected observed=$observed result_root=$result_root" >&2
      exit 1
    fi
  done
}

evaluate_and_audit() {
  local result_tag="$1"
  local served_model="$2"
  local result_root="$result_base/$result_tag"
  local validation_root="$source_root/results/k2-main/$result_tag"
  local index
  local domain
  local expected

  mkdir -p "$result_root/eval" "$result_root/audits"
  for index in "${!domains[@]}"; do
    domain="${domains[$index]}"
    expected="${counts[$index]}"
    "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
      --answers "$result_root/answers/$domain-bare.jsonl" \
      --instances "$instance_root/$domain.json" \
      --validation-source "$validation_root/$domain-routed-taus.json" \
      --result-tag "$result_tag" \
      --served-model "$served_model" \
      --domain "$domain" \
      --arm bare \
      --expected-count "$expected" \
      --output "$result_root/eval/$domain-bare.eval.json"
  done
  "$python_path" "$repo_root/scripts/audit_runtime_matched_bare.py" \
    --result-root "$result_root" \
    --instances-dir "$instance_root" \
    --result-tag "$result_tag" \
    --served-model "$served_model" \
    --output "$result_root/audits/bare-completeness.json"
}

deepseek_root="$result_base/deepseek7b"
wait_for_jobs "$deepseek_root"
verify_counts "$deepseek_root"
evaluate_and_audit deepseek7b deepseek7b

yi_root="$result_base/yi15-9b"
wait_for_job_pids "$yi_root" "$yi_launcher_pid"
wait_for_jobs "$yi_root"
verify_counts "$yi_root"
evaluate_and_audit yi15-9b yi15-9b
