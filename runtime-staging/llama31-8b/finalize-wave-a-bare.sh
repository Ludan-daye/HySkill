#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/llama31-8b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
validation_root="$source_root/results/k2-main/llama31-8b"
python_path="$source_root/.venv/bin/python"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
running=1
while (( running != 0 )); do
  running=0
  for domain in "${domains[@]}"; do
    pid=$(<"$result_root/logs/$domain-bare.full.pid")
    if kill -0 "$pid" 2>/dev/null; then
      running=1
    fi
  done
  if (( running != 0 )); then
    sleep 30
  fi
done

mkdir -p "$result_root/eval" "$result_root/audits"
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  expected_count=${counts[$index]}
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
