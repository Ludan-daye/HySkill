#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/HySkill-baseline-runtime-matched-20260724/repo
source_root=/root/HySkill-k-run-20260723
result_root=/root/HySkill-baseline-runtime-matched-20260724-results/yi15-9b
instance_root="$source_root/external/SR-Agents/data/bench/instances"
corpus_path="$source_root/external/SR-Agents/data/bench/corpus/corpus.json"
python_path="$source_root/.venv/bin/python"
api_base=http://127.0.0.1:8000/v1
audit_root="$result_root/gate-audit/champ-routed"
domain=champ
arm=routed_gated
export PYTHONPATH="$repo_root:$repo_root/external/SR-Agents/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

"$python_path" "$repo_root/scripts/run_runtime_matched_gate_answers.py" \
  --instances "$instance_root/$domain.json" \
  --corpus "$corpus_path" \
  --gate-audit "$audit_root/audit.json" \
  --gate-decisions "$audit_root/new-decisions.jsonl" \
  --gate-rerun "$audit_root/rerun-ids.json" \
  --runtime-manifest "$result_root/runtime/$domain-$arm.manifest.json" \
  --repository-root "$repo_root" \
  --output "$result_root/gate-rerun/$domain-$arm.answers.jsonl" \
  --usage-log "$result_root/logs/$domain-$arm.gate-rerun.usage.jsonl" \
  --attempt-log "$result_root/logs/$domain-$arm.gate-rerun.attempts.jsonl" \
  --api-base "$api_base" \
  --workers 2 \
  --max-new-records 0
