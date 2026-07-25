#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/yi15-9b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
validation_root="$source_root/results/k2-main/yi15-9b"
python_path="$source_root/.venv/bin/python"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)

wait_for_jobs() {
  local running=1
  local domain
  local pid

  while (( running != 0 )); do
    running=0
    for domain in "${domains[@]}"; do
      if [[ ! -s "$result_root/logs/$domain-bare.full.pid" ]]; then
        echo "Yi Bare job PID file is missing: domain=$domain" >&2
        exit 1
      fi
      pid=$(<"$result_root/logs/$domain-bare.full.pid")
      if kill -0 "$pid" 2>/dev/null; then
        running=1
      fi
    done
    if (( running != 0 )); then
      sleep 30
    fi
  done
}

verify_counts() {
  local index
  local domain
  local expected
  local observed

  for index in "${!domains[@]}"; do
    domain=${domains[$index]}
    expected=${counts[$index]}
    observed=$(wc -l <"$result_root/answers/$domain-bare.jsonl")
    if [[ "$observed" -ne "$expected" ]]; then
      echo "Yi Bare output is incomplete: domain=$domain expected=$expected observed=$observed" >&2
      exit 1
    fi
  done
}

evaluate_missing_domains() {
  local index
  local domain
  local expected
  local output

  mkdir -p "$result_root/eval"
  for index in "${!domains[@]}"; do
    domain=${domains[$index]}
    expected=${counts[$index]}
    output="$result_root/eval/$domain-bare.eval.json"
    if [[ -e "$output" ]]; then
      echo "Preserving existing Yi evaluation: path=$output"
      continue
    fi
    "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
      --answers "$result_root/answers/$domain-bare.jsonl" \
      --instances "$instance_root/$domain.json" \
      --validation-source "$validation_root/$domain-routed-taus.json" \
      --result-tag yi15-9b \
      --served-model yi15-9b \
      --domain "$domain" \
      --arm bare \
      --expected-count "$expected" \
      --output "$output"
  done
}

audit_once() {
  local output="$result_root/audits/bare-completeness.json"

  mkdir -p "$result_root/audits"
  if [[ -e "$output" ]]; then
    echo "Preserving existing Yi completeness audit: path=$output"
  else
    "$python_path" "$repo_root/scripts/audit_runtime_matched_bare.py" \
      --result-root "$result_root" \
      --instances-dir "$instance_root" \
      --result-tag yi15-9b \
      --served-model yi15-9b \
      --output "$output"
  fi
  "$python_path" - "$output" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("valid") is not True:
    raise RuntimeError(f"Yi Bare completeness audit is not valid: path={path}")
print(json.dumps({
    "event": "yi_bare_finalization_complete",
    "audit": str(path),
    "valid": True,
}, sort_keys=True))
PY
}

wait_for_jobs
verify_counts
evaluate_missing_domains
audit_once
