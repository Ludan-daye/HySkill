#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/glm4-9b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
validation_root="$source_root/results/k2-main/glm4-9b"
python_path="$source_root/.venv/bin/python"
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"

domains=(theoremqa logicbench medcalcbench champ)
counts=(747 760 1100 223)
arms=(always_rerank select_bm25)
mkdir -p "$result_root/eval"
for index in "${!domains[@]}"; do
  domain=${domains[$index]}
  expected=${counts[$index]}
  for arm in "${arms[@]}"; do
    answers_path="$result_root/answers/$domain-$arm.jsonl"
    if [[ ! -f "$answers_path" ]]; then
      printf 'Missing answer file: %s\n' "$answers_path" >&2
      exit 1
    fi
    observed=$(wc -l < "$answers_path")
    if [[ "$observed" -ne "$expected" ]]; then
      printf \
        'Answer count mismatch: domain=%s arm=%s expected=%s observed=%s\n' \
        "$domain" \
        "$arm" \
        "$expected" \
        "$observed" >&2
      exit 1
    fi
    "$python_path" "$repo_root/scripts/evaluate_runtime_matched_baselines.py" \
      --answers "$answers_path" \
      --instances "$instance_root/$domain.json" \
      --validation-source "$validation_root/$domain-routed-taus.json" \
      --result-tag glm4-9b \
      --served-model glm4-9b \
      --domain "$domain" \
      --arm "$arm" \
      --expected-count "$expected" \
      --output "$result_root/eval/$domain-$arm.eval.json"
  done
done
